import 'dart:typed_data';
import 'package:flutter_test/flutter_test.dart';
import 'package:image/image.dart' as img;
import 'package:ai_image_detector/ai_image_detector.dart';
import 'package:ai_image_detector/src/meta.dart';

/// 가짜 ONNX 추론 구현 — 고정 logits 반환.
class _FakeRunner implements OnnxInferenceRunner {
  final List<double> logits;
  _FakeRunner(this.logits);

  @override
  Future<List<double>> run(Float32List inputData, List<int> shape) async =>
      logits;

  @override
  Future<void> dispose() async {}
}

ModelMeta _defaultMeta() => ModelMeta(
      imageSize: 32,
      cropSize: 32,
      resizeMode: 'exact',
      imageMean: [0.485, 0.456, 0.406],
      imageStd: [0.229, 0.224, 0.225],
      doRescale: true,
      rescaleFactor: 1.0 / 255.0,
      doNormalize: true,
      resample: 3,
      id2label: {'0': 'artificial', '1': 'human'},
    );

Uint8List _makeTestImageBytes() {
  final image = img.Image(width: 32, height: 32, numChannels: 3);
  for (int y = 0; y < 32; y++) {
    for (int x = 0; x < 32; x++) {
      image.setPixelRgb(x, y, 128, 128, 128);
    }
  }
  return Uint8List.fromList(img.encodePng(image));
}

void main() {
  group('AiImageDetector.fromRunner — 전처리+후처리 통합', () {
    test('AI 라벨 logit이 높으면 AI-generated 판정', () async {
      // artificial(0)=2.0, human(1)=-2.0 → softmax artificial ≈0.982 → AI-generated
      final detector = AiImageDetector.fromRunner(
        meta: _defaultMeta(),
        runner: _FakeRunner([2.0, -2.0]),
      );
      final result = await detector.detect(_makeTestImageBytes());
      expect(result.verdict, 'AI-generated');
      expect(result.aiProbability, greaterThan(0.9));
    });

    test('Real 라벨 logit이 높으면 Real 판정', () async {
      // artificial(0)=-2.0, human(1)=2.0 → softmax human ≈0.982 → aiProb≈0.018 → Real
      final detector = AiImageDetector.fromRunner(
        meta: _defaultMeta(),
        runner: _FakeRunner([-2.0, 2.0]),
      );
      final result = await detector.detect(_makeTestImageBytes());
      expect(result.verdict, 'Real');
      expect(result.aiProbability, lessThan(0.1));
    });

    test('threshold 조정 — threshold=0.9이면 낮은 AI 확률도 Real', () async {
      // artificial≈0.7 → 기본(0.5)이면 AI-generated, threshold=0.9이면 Real
      final detector = AiImageDetector.fromRunner(
        meta: _defaultMeta(),
        runner: _FakeRunner([0.85, 0.15]),
      );
      final result = await detector.detect(_makeTestImageBytes(), threshold: 0.9);
      expect(result.verdict, 'Real');
    });

    test('labels 목록이 올바르게 채워짐', () async {
      final detector = AiImageDetector.fromRunner(
        meta: _defaultMeta(),
        runner: _FakeRunner([1.0, -1.0]),
      );
      final result = await detector.detect(_makeTestImageBytes());
      expect(result.labels.length, 2);
      expect(result.labels[0].label, 'artificial');
      expect(result.labels[1].label, 'human');
      expect(result.labels[0].score + result.labels[1].score, closeTo(1.0, 1e-5));
    });

    test('손상된 이미지 바이트 → ArgumentError', () async {
      final detector = AiImageDetector.fromRunner(
        meta: _defaultMeta(),
        runner: _FakeRunner([0.0, 0.0]),
      );
      expect(
        () => detector.detect(Uint8List.fromList([0, 1, 2, 3])),
        throwsA(isA<ArgumentError>()),
      );
    });

    test('빈 이미지 바이트 → ArgumentError', () async {
      final detector = AiImageDetector.fromRunner(
        meta: _defaultMeta(),
        runner: _FakeRunner([0.0]),
      );
      expect(
        () => detector.detect(Uint8List(0)),
        throwsA(isA<ArgumentError>()),
      );
    });

    test('50MB 초과 바이트 → ArgumentError (OOM 방어)', () async {
      final detector = AiImageDetector.fromRunner(
        meta: _defaultMeta(),
        runner: _FakeRunner([0.0]),
      );
      // 50MB + 1 byte 크기의 더미 데이터
      final oversized = Uint8List(50 * 1024 * 1024 + 1);
      expect(
        () => detector.detect(oversized),
        throwsA(isA<ArgumentError>()),
      );
    });

    test('정확히 50MB 바이트 → ArgumentError 없음 (JPEG 디코딩 실패지만 크기 검증은 통과)', () async {
      // 50MB 이내면 크기 검증은 통과. 디코딩은 실패해 ArgumentError가 뜨지만
      // 그 이유는 "크기 초과"가 아닌 "디코딩 불가" 여야 함.
      final detector = AiImageDetector.fromRunner(
        meta: _defaultMeta(),
        runner: _FakeRunner([0.0]),
      );
      final atLimit = Uint8List(50 * 1024 * 1024);
      // ArgumentError는 발생하지만 메시지에 'too large'가 없어야 함
      expect(
        () => detector.detect(atLimit),
        throwsA(predicate<ArgumentError>(
            (e) => !e.message.toString().contains('too large'))),
      );
    });
  });

  group('DetectionResult', () {
    test('toString에 verdict와 aiProbability 포함', () async {
      final detector = AiImageDetector.fromRunner(
        meta: _defaultMeta(),
        runner: _FakeRunner([2.0, -2.0]),
      );
      final result = await detector.detect(_makeTestImageBytes());
      expect(result.toString(), contains('AI-generated'));
      expect(result.toString(), contains('aiProbability'));
    });
  });
}
