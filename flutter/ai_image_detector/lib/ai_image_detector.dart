/// Flutter 온디바이스 AI 이미지 탐지 패키지.
///
/// ONNX Runtime을 사용해 오프라인 추론. Python onnx_detector.py와 동일한
/// 전처리 파이프라인(rescale + normalize)을 Dart로 포팅.
library ai_image_detector;

import 'dart:typed_data';

import 'package:flutter/services.dart';
import 'package:flutter_onnxruntime/flutter_onnxruntime.dart';
import 'package:image/image.dart' as img;

import 'src/meta.dart';
import 'src/postprocess.dart';
import 'src/preprocess.dart';

export 'src/meta.dart' show ModelMeta;
export 'src/postprocess.dart' show LabelScore, aiLabelKeywords, realLabelKeywords;

/// 단일 이미지 탐지 결과.
class DetectionResult {
  /// AI 생성 확률 (0.0 ~ 1.0).
  final double aiProbability;

  /// "AI-generated" 또는 "Real".
  final String verdict;

  /// 모델 asset 경로.
  final String model;

  /// 모든 라벨의 점수 목록.
  final List<LabelScore> labels;

  const DetectionResult({
    required this.aiProbability,
    required this.verdict,
    required this.model,
    required this.labels,
  });

  @override
  String toString() =>
      'DetectionResult(verdict: $verdict, aiProbability: ${(aiProbability * 100).toStringAsFixed(1)}%, model: $model)';
}

/// ONNX 추론 인터페이스 — 단위 테스트에서 가짜 구현 주입용.
abstract class OnnxInferenceRunner {
  /// [inputData]를 추론해 logits 리스트를 반환.
  Future<List<double>> run(Float32List inputData, List<int> shape);

  /// 리소스 해제.
  Future<void> dispose();
}

/// 실제 flutter_onnxruntime 기반 추론 구현.
class _RealOnnxRunner implements OnnxInferenceRunner {
  final OrtSession _session;
  final String _inputName;
  final String _outputName;

  _RealOnnxRunner(this._session, this._inputName, this._outputName);

  @override
  Future<List<double>> run(Float32List inputData, List<int> shape) async {
    final inputTensor = await OrtValue.fromList(inputData, shape);
    try {
      final outputs = await _session.run({_inputName: inputTensor});
      try {
        final outputTensor = outputs[_outputName];
        if (outputTensor == null) {
          throw StateError('Output "$_outputName" not found in model outputs');
        }
        final raw = await outputTensor.asList();
        // raw는 nested list일 수 있음 (배치 차원): [[logit0, logit1, ...]]
        final List<double> logits;
        if (raw is List && raw.isNotEmpty && raw.first is List) {
          logits = (raw.first as List).map((e) => (e as num).toDouble()).toList();
        } else {
          logits = (raw as List).map((e) => (e as num).toDouble()).toList();
        }
        return logits;
      } finally {
        for (final t in outputs.values) {
          await t.dispose();
        }
      }
    } finally {
      await inputTensor.dispose();
    }
  }

  @override
  Future<void> dispose() => _session.close();
}

/// Flutter 온디바이스 AI 이미지 탐지기.
///
/// ```dart
/// final detector = await AiImageDetector.load(
///   onnxAssetPath: 'assets/model/model.onnx',
///   metaAssetPath: 'assets/model/meta.json',
/// );
/// final result = await detector.detect(imageBytes);
/// print(result.verdict); // "AI-generated" or "Real"
/// await detector.dispose();
/// ```
class AiImageDetector {
  final ModelMeta _meta;
  final OnnxInferenceRunner _runner;
  final String _modelPath;

  AiImageDetector._(this._meta, this._runner, this._modelPath);

  /// Assets에서 ONNX 모델과 meta.json을 로드해 탐지기 인스턴스를 생성.
  ///
  /// [onnxAssetPath]: pubspec.yaml에 등록된 .onnx 파일 asset 경로.
  /// [metaAssetPath]: pubspec.yaml에 등록된 meta.json asset 경로.
  static Future<AiImageDetector> load({
    required String onnxAssetPath,
    required String metaAssetPath,
  }) async {
    final metaJson = await rootBundle.loadString(metaAssetPath);
    final meta = ModelMeta.fromJson(metaJson);

    final ort = OnnxRuntime();
    final session = await ort.createSessionFromAsset(onnxAssetPath);

    if (session.inputNames.isEmpty) {
      throw StateError(
          'ONNX model "$onnxAssetPath" has no input tensors');
    }
    if (session.outputNames.isEmpty) {
      throw StateError(
          'ONNX model "$onnxAssetPath" has no output tensors');
    }

    final inputName = session.inputNames.first;
    final outputName = session.outputNames.first;

    final runner = _RealOnnxRunner(session, inputName, outputName);
    return AiImageDetector._(meta, runner, onnxAssetPath);
  }

  /// 테스트용: 가짜 추론 구현 주입.
  static AiImageDetector fromRunner({
    required ModelMeta meta,
    required OnnxInferenceRunner runner,
    String modelPath = 'test/mock.onnx',
  }) {
    return AiImageDetector._(meta, runner, modelPath);
  }

  /// 이미지 바이트([imageBytes])를 받아 AI 탐지 결과를 반환.
  ///
  /// [threshold]: AI 판정 임계값 (기본 0.5).
  ///
  /// Throws [ArgumentError] if:
  ///   - imageBytes exceeds 50 MB
  ///   - imageBytes cannot be decoded as an image
  ///   - decoded image width or height exceeds 8192 px
  Future<DetectionResult> detect(Uint8List imageBytes,
      {double threshold = 0.5}) async {
    const int maxBytes = 50 * 1024 * 1024; // 50 MB
    if (imageBytes.lengthInBytes > maxBytes) {
      throw ArgumentError(
          'imageBytes too large: ${imageBytes.lengthInBytes} bytes '
          '(max ${maxBytes} bytes = 50 MB)');
    }

    final img.Image? image;
    try {
      image = img.decodeImage(imageBytes);
    } catch (e) {
      throw ArgumentError('Cannot decode image bytes: $e');
    }

    if (image == null) {
      throw ArgumentError('Cannot decode image bytes: unsupported format or corrupted data');
    }

    const int maxDimension = 8192;
    if (image.width > maxDimension || image.height > maxDimension) {
      throw ArgumentError(
          'Image dimensions too large: ${image.width}x${image.height} '
          '(max ${maxDimension}x${maxDimension})');
    }

    final (data, shape) = preprocessImageWithShape(image, _meta);
    final logits = await _runner.run(data, shape);
    final probs = softmax(logits);
    final labelScores = buildLabelScores(_meta.id2label, probs);
    final aiProb = extractAiProbability(labelScores);
    final verdict = determineVerdict(aiProb, threshold);

    return DetectionResult(
      aiProbability: aiProb,
      verdict: verdict,
      model: _modelPath,
      labels: labelScores,
    );
  }

  /// 리소스 해제 (ONNX 세션 종료).
  Future<void> dispose() => _runner.dispose();
}
