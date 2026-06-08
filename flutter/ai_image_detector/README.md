# ai_image_detector

Flutter 온디바이스 AI 이미지 탐지 패키지. ONNX Runtime으로 오프라인 추론.

## 요구 사항

- Flutter 3.0+
- Dart 3.10+
- 플랫폼: iOS, Android, macOS, Windows, Linux

## 설치

`pubspec.yaml`에 추가:

```yaml
dependencies:
  ai_image_detector:
    path: flutter/ai_image_detector  # 로컬 경로 또는 pub.dev 버전
```

## 모델 준비

### 1. ONNX 변환

```bash
pip install -r requirements-onnx.txt
python convert_to_onnx.py --model Organika/sdxl-detector
# → onnx_models/Organika__sdxl-detector/model.onnx + meta.json 생성
```

### 2. 모델 파일 복사

```bash
cp onnx_models/Organika__sdxl-detector/model.onnx  myapp/assets/model/
cp onnx_models/Organika__sdxl-detector/meta.json   myapp/assets/model/
```

### 3. pubspec.yaml 에 assets 등록

```yaml
flutter:
  assets:
    - assets/model/
```

## 사용법

```dart
import 'package:ai_image_detector/ai_image_detector.dart';

// 1. 앱 시작 시 1회 로드 (세션 재사용)
final detector = await AiImageDetector.load(
  onnxAssetPath: 'assets/model/model.onnx',
  metaAssetPath: 'assets/model/meta.json',
);

// 2. 이미지 탐지
final Uint8List imageBytes = await File('photo.jpg').readAsBytes();
final result = await detector.detect(imageBytes);

print(result.verdict);         // "AI-generated" or "Real"
print(result.aiProbability);   // 0.0 ~ 1.0
print(result.labels);          // [LabelScore(label: artificial, score: 0.95), ...]

// 3. threshold 조정 (기본 0.5)
final strictResult = await detector.detect(imageBytes, threshold: 0.7);

// 4. 리소스 해제
await detector.dispose();
```

## 전처리 파이프라인

Python `onnx_detector.preprocess_image()`와 **1:1 수치 동등**하도록 구현:

| 단계 | Python | Dart |
|------|--------|------|
| RGB 변환 | `image.convert("RGB")` | `image.convert(numChannels: 3)` |
| resize (exact) | `img.resize((size, size), BICUBIC)` | `copyResize(width: size, height: size, interpolation: cubic)` |
| resize (shortest_edge) | 짧은 변 → image_size, aspect 유지 | 동일 |
| center-crop | `(w-cw)//2, (h-ch)//2` | 동일 |
| rescale | `× rescale_factor` (기본 1/255) | 동일 |
| normalize | `(x - mean) / std` | 동일 |
| HWC → CHW | `transpose(2,0,1)` | 직접 인덱스 계산 |
| 배치 | `[np.newaxis, :]` | shape `[1, 3, H, W]` |

### 수치 검증

`tools/export_preprocess_golden.py`로 Python 골든 벡터를 생성하고,
`test/preprocess_golden_test.dart`에서 동일 입력에 대한 Dart 출력과 비교합니다.
허용 오차: **절대오차 1e-2** (PIL BICUBIC vs Dart cubic 보간 구현 차이 허용).

### 보간 주의사항

PIL `resample=3`(BICUBIC)과 Dart `image` 패키지 `Interpolation.cubic`은
구현 방식이 다를 수 있어 경계 픽셀에서 최대 1e-2 오차가 발생할 수 있습니다.
단색 이미지 및 내부 픽셀에서는 오차가 0에 가깝습니다.

그라데이션 이미지 실측: `gradient_exact_32` 케이스 mean 차이 ≤ 0.05, std 차이 ≤ 0.05,
샘플 인덱스별 절대오차 ≤ 1e-2.

## 한계 사항

- **캐시 stale 주의**: `createSessionFromAsset`은 임시 디렉토리에 파일명 기준으로 모델을 캐시합니다.
  모델 파일을 교체할 때는 파일명을 바꾸거나 앱을 재설치해야 stale 캐시가 제거됩니다.

- **비정사각 모델 미지원**: 현재 구현은 height = width인 정사각 출력 크기만 지원합니다.
  height ≠ width인 모델은 동작을 보장하지 않습니다.

- **예외 타입**: Dart에서는 Python `ValueError` 대신 `ArgumentError`를 사용합니다.
  잘못된 입력(이미지 크기 초과, meta.json 위반 등)은 모두 `ArgumentError`로 발생합니다.

- **image_size 상한 (Dart 전용)**: 모바일 OOM 방어를 위해 `image_size` / `crop_size`의 상한을 **1024**로 제한합니다.
  서버·데스크탑용 Python 코드는 4096까지 허용합니다.

- **입력 이미지 제한 (detect)**: 50 MB 초과 바이트 또는 8192 × 8192 초과 이미지는 `ArgumentError`로 거부됩니다.

- **전처리 허용오차**: PIL BICUBIC과 Dart cubic 보간 차이로 인해 절대오차 최대 **1e-2** 이내 차이가 발생할 수 있습니다.
  그라데이션 이미지 실측 기준: mean/std 오차 ≤ 0.05, 개별 픽셀 오차 ≤ 1e-2.

## 라벨 키워드

Python `detector.py`와 동일:

- AI 계열: `ai`, `artificial`, `fake`, `generated`, `synthetic`
- Real 계열: `real`, `human`, `natural`, `photo`, `authentic`

## 성능 참고

| 항목 | 비고 |
|------|------|
| 모델 크기 | ~75MB (model.onnx, 양자화 전) |
| 양자화 모델 | ~20MB (model_quantized.onnx) |
| 추론 시간 | 기기·모델에 따라 다름 (ARM64 ~100-500ms) |
| 메모리 | InferenceSession 1회 생성 후 재사용 |

## 테스트

```bash
cd flutter/ai_image_detector
flutter test
```

골든 파일 재생성:

```bash
python tools/export_preprocess_golden.py
flutter test test/preprocess_golden_test.dart
```
