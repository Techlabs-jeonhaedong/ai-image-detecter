import 'dart:typed_data';
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:ai_image_detector/ai_image_detector.dart';

void main() => runApp(const AiDetectorApp());

class AiDetectorApp extends StatelessWidget {
  const AiDetectorApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'AI Image Detector',
      theme: ThemeData(colorSchemeSeed: Colors.indigo, useMaterial3: true),
      home: const DetectorScreen(),
    );
  }
}

class DetectorScreen extends StatefulWidget {
  const DetectorScreen({super.key});

  @override
  State<DetectorScreen> createState() => _DetectorScreenState();
}

class _DetectorScreenState extends State<DetectorScreen> {
  AiImageDetector? _detector;
  DetectionResult? _result;
  bool _loading = false;
  String? _error;

  // 모델 파일을 example/assets/model/ 에 복사한 뒤 사용
  static const String _onnxAssetPath = 'assets/model/model.onnx';
  static const String _metaAssetPath = 'assets/model/meta.json';

  @override
  void initState() {
    super.initState();
    _initDetector();
  }

  Future<void> _initDetector() async {
    try {
      final detector = await AiImageDetector.load(
        onnxAssetPath: _onnxAssetPath,
        metaAssetPath: _metaAssetPath,
      );
      setState(() => _detector = detector);
    } catch (e) {
      debugPrint('모델 로드 실패 (상세): $e');
      setState(() =>
          _error = '모델을 불러오지 못했습니다.\n\n'
              'assets/model/model.onnx 와 meta.json 파일을 복사했는지 확인하세요.\n'
              '변환 방법: python convert_to_onnx.py --model Organika/sdxl-detector');
    }
  }

  Future<void> _pickAndDetect(ImageSource source) async {
    final picker = ImagePicker();
    final XFile? file = await picker.pickImage(source: source);
    if (file == null) return;

    setState(() {
      _loading = true;
      _error = null;
      _result = null;
    });

    try {
      final Uint8List bytes = await file.readAsBytes();
      final result = await _detector!.detect(bytes);
      setState(() => _result = result);
    } catch (e) {
      debugPrint('이미지 탐지 실패 (상세): $e');
      setState(() => _error = '이미지 분석에 실패했습니다.\n지원되는 이미지 파일인지 확인하고 다시 시도해 주세요.');
    } finally {
      setState(() => _loading = false);
    }
  }

  @override
  void dispose() {
    _detector?.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('AI 이미지 탐지기')),
      body: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          children: [
            if (_detector == null && _error == null)
              const Center(child: Column(
                mainAxisSize: MainAxisSize.min,
                children: [
                  CircularProgressIndicator(),
                  SizedBox(height: 8),
                  Text('모델 로딩 중...'),
                ],
              ))
            else if (_error != null)
              _ErrorCard(_error!)
            else ...[
              Row(
                mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                children: [
                  ElevatedButton.icon(
                    onPressed: _loading
                        ? null
                        : () => _pickAndDetect(ImageSource.gallery),
                    icon: const Icon(Icons.photo_library),
                    label: const Text('갤러리'),
                  ),
                  ElevatedButton.icon(
                    onPressed: _loading
                        ? null
                        : () => _pickAndDetect(ImageSource.camera),
                    icon: const Icon(Icons.camera_alt),
                    label: const Text('카메라'),
                  ),
                ],
              ),
              const SizedBox(height: 24),
              if (_loading) const CircularProgressIndicator(),
              if (_result != null) _ResultCard(_result!),
            ],
          ],
        ),
      ),
    );
  }
}

class _ResultCard extends StatelessWidget {
  final DetectionResult result;
  const _ResultCard(this.result);

  @override
  Widget build(BuildContext context) {
    final isAi = result.verdict == 'AI-generated';
    return Card(
      color: isAi ? Colors.red.shade50 : Colors.green.shade50,
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(children: [
              Icon(isAi ? Icons.warning_rounded : Icons.check_circle_rounded,
                  color: isAi ? Colors.red : Colors.green, size: 32),
              const SizedBox(width: 8),
              Text(
                result.verdict,
                style: Theme.of(context).textTheme.headlineSmall?.copyWith(
                    color: isAi ? Colors.red : Colors.green,
                    fontWeight: FontWeight.bold),
              ),
            ]),
            const SizedBox(height: 8),
            Text('AI 확률: ${(result.aiProbability * 100).toStringAsFixed(1)}%'),
            const Divider(),
            ...result.labels.map((ls) => Text(
                '  ${ls.label}: ${(ls.score * 100).toStringAsFixed(1)}%')),
          ],
        ),
      ),
    );
  }
}

class _ErrorCard extends StatelessWidget {
  final String message;
  const _ErrorCard(this.message);

  @override
  Widget build(BuildContext context) {
    return Card(
      color: Colors.orange.shade50,
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Text(message,
            style: const TextStyle(color: Colors.deepOrange, height: 1.5)),
      ),
    );
  }
}
