# ai_image_detector example

`ai_image_detector` 패키지를 사용하는 온디바이스 AI 이미지 탐지 예제 앱.

## 바로 실행

경량 ONNX 모델이 `assets/model/` 에 **이미 번들**되어 있어 별도 모델 준비 없이 바로 실행됩니다.

```bash
cd flutter/ai_image_detector/example
flutter pub get
flutter run        # iOS / Android / macOS / Windows / Linux
```

앱 실행 후 갤러리 또는 카메라로 이미지를 선택하면 AI 생성 여부를 온디바이스에서 판별합니다.

## 번들된 모델

- `assets/model/model.onnx` — Organika/sdxl-detector 의 MatMul-only INT8 양자화 모델
  (약 91MB, 모든 플랫폼의 onnxruntime CPU/모바일 EP 호환)
- `assets/model/meta.json` — 전처리·라벨 설정

모델을 교체/재생성하려면 `assets/model/PLACE_MODEL_HERE.txt` 를 참고하세요.
