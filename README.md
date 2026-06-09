# AI Image Detector

로컬 ML 모델로 **이미지가 AI로 생성된 것인지 실제 사진인지** 판별하는 CLI 도구다.
API 키·과금 없이 완전 오프라인 추론한다. **이 저장소를 클론하면 별도 모델 준비 없이 바로 추론할 수 있다.** (경량 ONNX 모델이 저장소에 포함되어 있음)

기본 모델: [`Organika/sdxl-detector`](https://huggingface.co/Organika/sdxl-detector) (ViT 기반 image-classification)

---

## 주요 기능

- 단일 / 여러 이미지를 한 번에 판별
- **다중 모델** (`--model` 반복 지정) 및 **앙상블** (`--ensemble`) 지원 — 여러 모델의 평균 확률로 판정
- **메타데이터/출처 검사** — PNG text chunk, EXIF, XMP, C2PA 흔적에서 AI 생성 신호 탐지
- 사람이 읽기 좋은 출력 + JSON 출력(`--json`) 지원
- 판정 임계값 조정(`--threshold`) 및 모델 교체(`--model`)
- 한 이미지가 실패해도 나머지는 계속 처리 (개별 실패 격리)
- 존재하지 않는 파일 / 손상된 이미지 / 권한 없음 / 비이미지 파일 등 예외를 명확한 메시지로 처리

---

## 동작 원리

### ML 추론 (앙상블)

1. 이미지를 로드하고 RGB로 변환한다.
2. 지정된 모든 모델에 대해 HuggingFace `image-classification` 파이프라인으로 추론한다.
3. 각 모델의 AI 생성 확률을 구하고 **평균**낸다.
4. `평균 AI 확률 >= threshold` 이면 **AI-generated**, 아니면 **Real** 로 판정한다.

### 메타데이터 검사 (기본 활성화)

PNG text chunk, EXIF `Software` 태그, XMP 패킷, C2PA/JUMBF 시그니처에서 AI 생성 도구의 흔적을 탐지한다.

신호는 **결정적 신호(decisive)**와 **약한 흔적(weak)**으로 구분한다:

| 신호 유형 | 예시 | has_ai_signal |
|-----------|------|---------------|
| 결정적 (decisive) | SD 파라미터 구조(Steps:, Sampler:, CFG scale:), DALL-E/Midjourney/Stable Diffusion 등 명시적 도구명, IPTC trainedAlgorithmicMedia, C2PA AI assertion | `True` |
| 약한 흔적 (weak) | C2PA/JUMBF 박스 단순 존재, Content Credentials 태그 | `False` (signals에는 기록) |

> **오탐 방지**: 일반 카메라 앱 이름("diffusion" 단어 단독), `comment="family trip"`, `Description="Photo taken with iPhone"` 같은 일반 메타데이터는 신호로 처리하지 않는다.
>
> 메타데이터가 없어도 AI 이미지일 수 있다 (SNS 업로드 시 메타데이터 제거).

파일 크기가 50MB를 초과하면 raw 바이트 스캔(XMP/C2PA/도구명 검색)을 생략하고 Pillow 기반 검사만 수행한다.

### 종합 판정

- **결정적 메타데이터 신호(decisive=True)**가 있으면 ML 결과와 무관하게 **AI-generated**로 override한다.
  - 결과 JSON의 `verdict_source` 필드가 `"metadata"`로 표시된다.
  - ML 원본 확률(`ai_probability`)은 보존된다.
  - 사람용 출력: `Verdict : AI-generated (metadata)`
- 약한 흔적만 있으면 verdict는 ML 결과를 그대로 따른다 (`verdict_source="model"`).
- ML 로드 실패 + 결정적 메타 신호 → AI-generated 판정, exit code 0 (성공).
- 메타데이터 신호가 없으면 ML 앙상블 확률로 판정한다.

---

## 요구사항

- Python 3.9 이상

**경량 추론 (기본, ONNX 백엔드):**
- `onnxruntime`, `numpy`, `Pillow` (합계 ~100MB)
- `torch`, `transformers` 불필요

**torch 백엔드 (`--backend torch`, 선택):**
- `torch`, `transformers`, `Pillow`

> **C2PA 정식 검증 (선택)**: Python 3.10+ 환경에서 `pip install c2pa-python` 시
> C2PA 매니페스트 정식 파싱이 자동으로 활성화된다. 미설치가 정상 상태이며 에러가 발생하지 않는다.

---

## 설치 및 빠른 시작

```bash
# 1. 저장소 클론
git clone <repo-url>
cd ai-image-detecter

# 2. 런타임 의존성 설치 (onnxruntime, numpy, Pillow — torch 불필요)
pip install -r requirements-onnx.txt

# 3. 바로 추론! (번들된 경량 ONNX 모델 사용 — 모델 다운로드 없음)
python detect.py photo.jpg
```

> **기본 백엔드는 onnx다.** 저장소에 경량 ONNX 모델(~91MB)이 번들되어 있으므로
> 클론 후 별도 모델 준비 없이 즉시 추론할 수 있다.

**(선택) setup.py로 모델 재빌드:**
이미 번들된 모델이 있으면 자동으로 빌드를 건너뛰고 검증만 수행한다.

```bash
python setup.py               # 번들 감지 → 건너뜀 (빠름)
python setup.py --force       # 강제 재빌드
python setup.py --skip-install  # 의존성 설치 건너뜀
```

**(선택) torch 백엔드 사용:**
첫 실행 시 모델(~300MB)이 `~/.cache/huggingface/` 에 자동 다운로드된다.

```bash
pip install -r requirements.txt
python detect.py photo.jpg --backend torch
```

**(선택) 테스트/개발 의존성:**

```bash
pip install -r requirements-dev.txt
```

---

## 사용법

### 단일 이미지

```bash
python detect.py photo.jpg
```

AI로 판정된 경우 출력 예시:

```
[AI] photo.jpg
  Verdict  : AI-generated
  AI Prob  : 87.0%
  Model    : Organika/sdxl-detector
```

실제 사진으로 판정된 경우 출력 예시:

```
[OK] camera.jpg
  Verdict  : Real
  AI Prob  : 12.0%
  Model    : Organika/sdxl-detector
```

### 여러 이미지 한 번에

```bash
python detect.py img1.jpg img2.png img3.webp
```

각 이미지 결과가 순서대로 출력된다. 중간에 한 파일이 실패해도 나머지는 계속 처리된다.

### 다중 모델 (`--model` 반복)

```bash
# 두 모델의 평균 확률로 판정
python detect.py photo.jpg --model Organika/sdxl-detector --model umm-maybe/AI-image-detector
```

여러 모델의 AI 확률을 **평균**내서 최종 판정에 사용한다. 일부 모델이 실패해도 성공한 모델들의 평균으로 계속 진행한다.

### 앙상블 (`--ensemble`)

```bash
# 사전 정의된 3개 모델 세트로 앙상블
python detect.py photo.jpg --ensemble
```

앙상블 모델 세트 (`ENSEMBLE_MODELS`):
- `Organika/sdxl-detector`
- `yaya36095/ai-image-detector`
- `umm-maybe/AI-image-detector`

`--ensemble`과 `--model`을 동시 지정하면 둘을 **합쳐서** 사용한다 (중복 제거):

```bash
# ENSEMBLE_MODELS + extra/model (중복 제거)
python detect.py photo.jpg --ensemble --model extra/model
```

앙상블 출력 예시 (사람용):

```
[AI] photo.jpg
  Verdict  : AI-generated
  AI Prob  : 82.3%
  Model    : ensemble(3 models)
  Per-model:
    [Organika/sdxl-detector] 91.2%
    [yaya36095/ai-image-detector] 78.5%
    [umm-maybe/AI-image-detector] 77.1%
```

### 메타데이터 검사

기본적으로 활성화된다. `--no-metadata`로 비활성화할 수 있다.

```bash
# 메타데이터 검사 비활성화
python detect.py photo.jpg --no-metadata
```

SD parameters가 있는 이미지 출력 예시 (결정적 메타 신호로 override):

```
[AI] sd_generated.png
  Verdict  : AI-generated (metadata)
  AI Prob  : 73.0%
  Model    : Organika/sdxl-detector
  Metadata : AI signal detected
    - PNG text chunk 'parameters': Stable Diffusion parameters structure detected
```

### JSON 출력

```bash
python detect.py photo.jpg --json
```

단일 모델 JSON 출력 예시:

```json
[{
  "image": "photo.jpg",
  "ai_probability": 0.87,
  "verdict": "AI-generated",
  "model": "Organika/sdxl-detector",
  "error": null,
  "models": [{"model": "Organika/sdxl-detector", "ai_probability": 0.87, "error": null}],
  "metadata": {"has_ai_signal": false, "decisive": false, "signals": [], "source": null, "checked": true}
}]
```

앙상블 JSON 출력 예시:

```json
[{
  "image": "photo.jpg",
  "ai_probability": 0.82,
  "verdict": "AI-generated",
  "model": "ensemble(3 models)",
  "error": null,
  "models": [
    {"model": "Organika/sdxl-detector", "ai_probability": 0.91, "error": null},
    {"model": "yaya36095/ai-image-detector", "ai_probability": 0.79, "error": null},
    {"model": "umm-maybe/AI-image-detector", "ai_probability": 0.77, "error": null}
  ],
  "metadata": {"has_ai_signal": false, "decisive": false, "signals": [], "source": null, "checked": true}
}]
```

### 임계값 조정

```bash
# AI 확률이 70% 이상일 때만 AI로 판정 (더 보수적)
python detect.py photo.jpg --threshold 0.7
```

---

## 옵션

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `IMAGE...` | (필수) | 분석할 이미지 경로. 하나 이상, 여러 개 가능 |
| `--json` | off | 결과를 JSON 배열로 출력 |
| `--model MODEL_ID` | `Organika/sdxl-detector` | HuggingFace 모델 ID (반복 가능) |
| `--ensemble` | off | 사전 정의된 앙상블 모델 세트 사용 |
| `--threshold FLOAT` | `0.5` | AI 판정 임계값 (0.0 ~ 1.0) |
| `--no-metadata` | off | 메타데이터/출처 검사 비활성화 |
| `--backend` | `onnx` | 추론 백엔드: `onnx`(기본, 번들 경량 모델) 또는 `torch`(transformers 필요) |
| `--onnx-models-dir DIR` | (번들 경로) | ONNX 모델 디렉토리. 기본은 저장소 내 번들 경로 |
| `-h`, `--help` | - | 도움말 출력 |

> **`--model`과 `--ensemble` 동시 지정**: 둘을 합쳐서 중복 없이 모든 모델을 사용한다.

---

## 출력 필드(JSON)

| 필드 | 타입 | 의미 |
|------|------|------|
| `image` | string | 입력한 이미지 경로 |
| `ai_probability` | float \| null | AI 생성 확률 앙상블 평균 (0.0~1.0). 실패 시 `null` |
| `verdict` | string \| null | `"AI-generated"` 또는 `"Real"`. 실패 시 `null` |
| `model` | string | 단일 모델이면 모델 ID, 앙상블이면 `"ensemble(N models)"` |
| `error` | string \| null | 실패 시 에러 메시지, 성공 시 `null` |
| `models` | array | 모델별 개별 결과 (`model`, `ai_probability`, `error`) |
| `metadata` | object | 메타데이터 검사 결과 (아래 참고) |

### metadata 필드

| 필드 | 타입 | 의미 |
|------|------|------|
| `has_ai_signal` | bool | **결정적** AI 신호 발견 여부 (약한 흔적만 있으면 `false`) |
| `decisive` | bool | 결정적 신호 여부 (신규, `has_ai_signal`과 동일값) |
| `signals` | array | 발견된 근거 문자열 목록 (결정적+약한 흔적 모두 포함) |
| `source` | `"c2pa"` \| `"metadata"` \| null | 신호 출처 |
| `checked` | bool | 정상 검사 완료 여부 (손상 파일 등은 `false`) |

최상위 결과 JSON에 추가된 필드:

| 필드 | 타입 | 의미 |
|------|------|------|
| `verdict_source` | `"model"` \| `"metadata"` | verdict 결정 출처. 메타데이터 override 시 `"metadata"` |

사람이 읽는 출력의 줄머리 마커:

| 마커 | 의미 |
|------|------|
| `[AI]` | AI 생성으로 판정 |
| `[OK]` | 실제 사진으로 판정 |
| `[ERROR]` | 처리 실패 (메시지는 `Error:` 줄에 표시) |

---

## 종료 코드(Exit Code)

| 코드 | 상황 |
|------|------|
| `0` | 모든 이미지 정상 처리 |
| `1` | 하나 이상의 이미지 처리 실패 |
| `2` | 잘못된 인자 (이미지 미입력, `--threshold` 범위 초과 등) |

---

## 에러 처리

다음 상황은 프로그램을 중단시키지 않고 해당 이미지에만 에러로 기록되며, 나머지 이미지는 계속 처리된다.

- 존재하지 않는 파일 → `File not found`
- 권한 없는 파일 → `Permission denied`
- 이미지가 아닌 파일 → `Not a valid image file`
- 손상된 이미지 → `Cannot open/load image`
- 비정상적으로 큰 이미지(압축 폭탄) → `Image too large (potential decompression bomb)`
- 모델 로드 실패(오프라인·잘못된 모델 ID 등) → 해당 이미지에 에러 기록 (앙상블 시 나머지 모델로 계속)

하나라도 실패하면 종료 코드는 `1`이 된다.

---

## 테스트

```bash
python -m pytest
```

모든 테스트는 실제 모델 다운로드 없이 **mock**으로 동작하므로 네트워크가 필요 없고 빠르게 끝난다.

```bash
# 현재 수집 테스트 수 확인
python -m pytest --collect-only -q 2>&1 | tail -1
```

---

## 프로젝트 구조

```
ai-image-detecter/
├── detect.py            # CLI 엔트리포인트 (argparse)
├── detector.py          # 추론 핵심 로직 (앙상블, 모델 주입 가능 / mock 가능)
├── metadata.py          # 메타데이터/출처 검사 (PNG, EXIF, XMP, C2PA)
├── backends.py          # 백엔드 선택 공용 헬퍼 (detect.py / server.py 공유)
├── onnx_detector.py     # 경량 ONNX Runtime 추론 모듈 (torch 불필요)
├── convert_to_onnx.py   # HuggingFace → ONNX + INT8 양자화 변환 스크립트
├── requirements.txt     # 실행 의존성 (torch, transformers, Pillow)
├── requirements-onnx.txt    # ONNX 경량 런타임 의존성 (onnxruntime, numpy, Pillow)
├── requirements-convert.txt # 변환 전용 의존성 (optimum, torch, transformers)
├── requirements-dev.txt # 테스트 의존성 (pytest)
├── pytest.ini
├── .gitignore
├── README.md
└── tests/
    ├── conftest.py              # 공유 fixture (PIL로 테스트 이미지 생성)
    ├── test_detector.py         # 단위 테스트
    ├── test_cli.py              # CLI E2E 테스트
    ├── test_metadata.py         # 메타데이터 + 앙상블 단위 테스트
    ├── test_ensemble_cli.py     # 앙상블/다중모델/메타데이터 E2E 테스트
    ├── test_server.py           # HTTP 서버 테스트
    └── test_onnx_detector.py    # ONNX 백엔드 단위/통합/E2E 테스트
```

---

## 모델 정보

### 기본 모델

**Organika/sdxl-detector**
- 아키텍처: ViT (Vision Transformer) 기반 image-classification
- 출력 라벨: `artificial` (AI 생성) / `human` (실제 사진)
- 링크: https://huggingface.co/Organika/sdxl-detector

### 앙상블 모델 세트 (`--ensemble`)

| 모델 | 설명 |
|------|------|
| `Organika/sdxl-detector` | ViT sdxl 탐지기 (기본 모델) |
| `yaya36095/ai-image-detector` | ViT 범용 AI 이미지 탐지기 |
| `umm-maybe/AI-image-detector` | ViT AI 이미지 탐지기 |

모든 모델은 캐시 위치: `~/.cache/huggingface/`

다른 모델을 쓰려면 `--model`로 지정한다. 모델마다 출력 라벨이 다를 수 있는데, 라벨이 위의 AI/Real 키워드 집합과 매칭되지 않으면 `Cannot determine AI probability from label(s)...` 에러가 발생한다.

---

## 정확도

정확도 향상을 위한 권장 설정:

- **앙상블** (`--ensemble`): 여러 모델의 합의로 단일 모델 오탐을 줄임
- **메타데이터 검사** (기본 활성화): PNG/JPEG 생성 메타데이터로 확정적 판정 가능
- **threshold 조정**: 보수적 판정이 필요하면 `--threshold 0.7` 이상 권장

> **앙상블 정확도 특성**: 개별 모델이 다른 아키텍처/학습 데이터를 사용하므로 앙상블이 단일 모델보다 일반적으로 낮은 오탐율을 보인다. 단, 앙상블 모델들이 모두 같은 방식으로 틀리는 edge case는 여전히 존재한다.

---

## HTTP 서버 (PHP 연동)

CLI와 동일한 추론 로직을 HTTP API로 제공한다. PHP 등 외부 언어에서 curl로 이미지를 업로드하면 JSON으로 결과를 받을 수 있다.

### 설치

```bash
pip install -r requirements-server.txt
```

### 실행

```bash
uvicorn server:app --host 127.0.0.1 --port 8000
```

첫 실행 시 기본 모델(~300MB)이 자동 다운로드된다. 이후 요청부터는 **메모리에 로드된 모델을 재사용**하므로 응답이 빠르다.

### 엔드포인트

| 메서드 | 경로 | 설명 |
|--------|------|------|
| `GET` | `/health` | 헬스체크. `{"status": "ok"}` 반환 |
| `POST` | `/detect` | 이미지 업로드 후 AI 생성 여부 판정 |

#### POST /detect 파라미터 (multipart/form-data)

| 파라미터 | 타입 | 필수 | 기본값 | 설명 |
|----------|------|------|--------|------|
| `file` | 파일 | 필수 | - | 분석할 이미지 파일 |
| `threshold` | float | 선택 | `0.5` | AI 판정 임계값 (0~1). 범위 초과 시 400 |
| `ensemble` | bool | 선택 | `false` | `true`면 ENSEMBLE_MODELS 전체 사용 |
| `model` | string (반복 가능) | 선택 | - | 지정 시 해당 모델 사용. ensemble과 함께면 합쳐서 중복 제거 |
| `no_metadata` | bool | 선택 | `false` | `true`면 메타데이터 검사 생략 |

#### 응답 JSON 스키마

CLI의 `--json` 출력과 동일한 단일 결과 dict:

```json
{
  "image": "photo.jpg",
  "ai_probability": 0.87,
  "verdict": "AI-generated",
  "verdict_source": "model",
  "model": "Organika/sdxl-detector",
  "error": null,
  "models": [{"model": "Organika/sdxl-detector", "ai_probability": 0.87, "error": null}],
  "metadata": {"has_ai_signal": false, "decisive": false, "signals": [], "source": null, "checked": true}
}
```

#### HTTP 에러 코드

| 코드 | 상황 |
|------|------|
| 400 | `file` 누락, 빈 파일(0바이트), threshold 범위/형식 오류, `ALLOWED_MODELS` 목록 외 모델 지정 |
| 413 | 파일 크기 초과 (기본 20MB). Content-Length 헤더 단계에서 사전 거부하므로 본문을 전송하지 않아도 됨 |
| 422 | FastAPI 파라미터 파싱 실패 |
| 200 + `error` 필드 | 이미지 자체 처리 실패 (비이미지, 손상 파일 등) |

### PHP 호출 예시

#### exec + curl 방식

```php
<?php
$imagePath = '/path/to/photo.jpg';
$url = 'http://127.0.0.1:8000/detect';

// exec으로 curl 호출
$escapedPath = escapeshellarg($imagePath);
$output = shell_exec("curl -s -F 'file=@{$escapedPath}' {$url}");
$result = json_decode($output, true);

echo $result['verdict'];         // "AI-generated" 또는 "Real"
echo $result['ai_probability'];  // 0.87 등
```

#### CurlFile 멀티파트 업로드 방식 (권장)

```php
<?php
$imagePath = '/path/to/photo.jpg';

$ch = curl_init('http://127.0.0.1:8000/detect');
curl_setopt($ch, CURLOPT_POST, true);
curl_setopt($ch, CURLOPT_POSTFIELDS, [
    'file'      => new CURLFile($imagePath),
    'ensemble'  => 'false',
    'threshold' => '0.5',
]);
curl_setopt($ch, CURLOPT_RETURNTRANSFER, true);
curl_setopt($ch, CURLOPT_TIMEOUT, 60);

$response = curl_exec($ch);
$httpCode = curl_getinfo($ch, CURLINFO_HTTP_CODE);
curl_close($ch);

if ($httpCode !== 200) {
    // 4xx/5xx 에러 처리
    throw new RuntimeException("AI detector error: HTTP {$httpCode}");
}

$result = json_decode($response, true);

if ($result['error'] !== null) {
    // 이미지 처리 실패 (비이미지, 손상 파일 등)
    error_log("Image error: " . $result['error']);
} else {
    $verdict      = $result['verdict'];          // "AI-generated" | "Real"
    $probability  = $result['ai_probability'];   // 0.0 ~ 1.0
    $source       = $result['verdict_source'];   // "model" | "metadata"
    $isAi         = ($verdict === 'AI-generated');
}
```

### 운영 팁

**백그라운드 상시 실행**

```bash
# nohup으로 백그라운드 실행 (간단)
nohup uvicorn server:app --host 127.0.0.1 --port 8000 > /var/log/ai-detector.log 2>&1 &

# systemd 서비스로 등록 (권장, 자동 재시작 지원)
# /etc/systemd/system/ai-detector.service 작성 후 systemctl enable --now ai-detector
```

**동시성**

기본 단일 워커. 동시 요청이 많으면 `--workers` 옵션으로 늘릴 수 있지만, **워커마다 모델이 메모리에 별도 로드**된다는 점에 주의 (워커 N개 = 모델 메모리 N배).

```bash
uvicorn server:app --host 127.0.0.1 --port 8000 --workers 2
```

> **ONNX 백엔드 멀티워커 시**: `DETECTOR_BACKEND=onnx`로 실행할 때 ONNX `InferenceSession`은 **워커(프로세스)별로 독립적으로 로드**된다. 세션 캐시(`_SESSION_CACHE`)는 프로세스 메모리에 있으므로 워커 간 공유되지 않는다. 워커 N개 = ONNX 세션 N개(모델당). 메모리 사용량이 torch 대비 훨씬 낮으므로 멀티워커 운영이 실용적이다.

**외부 노출 시 보안**

기본 바인딩은 `127.0.0.1`(로컬 전용). 외부에서 접근하려면 Nginx/Apache 리버스 프록시 뒤에 두고 방화벽으로 직접 포트를 차단하는 것을 권장한다.

### 서버 환경변수

| 환경변수 | 기본값 | 설명 |
|----------|--------|------|
| `HOST` | `127.0.0.1` | 바인딩 주소. 외부 노출 시 `0.0.0.0` |
| `PORT` | `8000` | 바인딩 포트 |
| `MAX_UPLOAD_BYTES` | `20971520` (20MB) | 업로드 허용 최대 바이트 수. Content-Length 헤더에서 사전 차단 + 실제 본문 재검사(이중 방어) |
| `MAX_CACHED_PIPELINES` | `4` | 메모리에 유지할 최대 pipeline 수. LRU 방식으로 초과 시 가장 오래된 항목 evict |
| `ALLOWED_MODELS` | (미설정) | 허용 모델 ID 목록(콤마 구분). **미설정 시 모든 모델 허용**(기존 동작 유지). 설정 시 목록 외 모델 요청은 400 반환. 예: `ALLOWED_MODELS=Organika/sdxl-detector,umm-maybe/AI-image-detector` |
| `_AI_DETECTOR_MOCK` | (미설정) | `1`로 설정 시 테스트용 mock pipeline 사용(고정 결과). **평소 사용 금지** |

> **ALLOWED_MODELS 사용 팁**: `ENSEMBLE_MODELS`(3개 모델)와 기본 모델을 함께 등록하는 것을 권장한다.
> ```bash
> export ALLOWED_MODELS="Organika/sdxl-detector,yaya36095/ai-image-detector,umm-maybe/AI-image-detector"
> ```

---

## Python API 직접 사용 (bytes/파일 스트림)

서버 없이 Python 코드에서 `detect()` 함수를 import해 파일 경로뿐 아니라 **파일 바이트나 스트림을 그대로** 전달할 수 있다. 업로드된 파일을 디스크에 임시저장하지 않고 메모리에서 바로 처리할 때 유용하다.

### 기본 사용 (bytes)

```python
from detector import detect

# 파일 바이트를 직접 전달 (디스크 임시저장 불필요)
with open("photo.jpg", "rb") as f:
    image_bytes = f.read()

result = detect(image_bytes, name="photo.jpg")

print(result["verdict"])          # "AI-generated" 또는 "Real"
print(result["ai_probability"])   # 0.0 ~ 1.0
print(result["verdict_source"])   # "model" 또는 "metadata"
```

### 웹 프레임워크 연동 (FastAPI / Django / Flask)

```python
# FastAPI — UploadFile을 디스크 임시저장 없이 처리
from fastapi import FastAPI, UploadFile, File
from detector import detect

app = FastAPI()

@app.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):
    content = await file.read()                      # 메모리에서 읽기
    result = detect(content, name=file.filename)     # bytes 직접 전달
    return result
```

### 앙상블 + 메타데이터

```python
from detector import detect

with open("sd_image.png", "rb") as f:
    data = f.read()

result = detect(
    data,
    name="sd_image.png",
    ensemble=True,       # 앙상블 모델 세트 사용
    threshold=0.6,       # 더 엄격한 임계값
    with_metadata=True,  # 메타데이터 AI 신호 검사 (기본 True)
)

print(result["verdict"])        # SD params 있으면 "AI-generated"
print(result["verdict_source"]) # "metadata" (결정적 신호 기반 override)
print(result["metadata"]["signals"])  # 탐지된 신호 목록
```

### 파일 경로 입력도 동일하게 동작 (하위호환)

```python
from detector import detect

result = detect("photo.jpg")       # 기존 경로 입력도 그대로 사용 가능
result = detect("photo.jpg", ensemble=True, threshold=0.7)
```

### io.BytesIO / 파일 핸들

```python
import io
from detector import detect

# BytesIO 스트림 전달
data = b"..."  # 이미지 바이트
result = detect(io.BytesIO(data), name="stream.jpg")

# 파일 핸들 전달 (seek이 가능한 바이너리 스트림)
with open("photo.jpg", "rb") as f:
    result = detect(f, name="photo.jpg")
```

> **중요: file-like 스트림은 1회만 사용 가능** — `detect()`와 `analyze_images_batch()`의
> 진입부에서 `_normalize_source()`를 통해 file-like를 1회 `read()`로 bytes로 변환한다.
> 따라서 같은 스트림 객체를 여러 번 전달하면 두 번째 호출부터는 빈 bytes가 된다.
> 동일 데이터를 여러 번 처리하려면 bytes를 직접 전달하거나 매번 새 `BytesIO`를 생성할 것.

### bytes/file-like 입력 시 C2PA 정식 검증의 한계

`inspect_metadata()`는 bytes나 file-like 입력을 받을 때 **`c2pa-python` 라이브러리의 정식 매니페스트 검증을 수행하지 않는다**. `c2pa.Reader.from_file()` API가 파일 경로만 지원하기 때문이다.

- 파일 경로(str) 입력: Pillow 기반 탐지 + raw bytes 스캔 + **c2pa-python 정식 검증** (c2pa-python 설치 시)
- bytes / file-like 입력: Pillow 기반 탐지 + raw bytes 스캔만 수행 (**c2pa 정식 검증 skip**)

> raw bytes 스캔(XMP 패킷, JUMBF 시그니처, AI 도구명 직접 탐지)은 bytes 입력에서도 동작한다.
> c2pa 정식 검증이 필요한 경우에는 파일 경로(str)로 전달할 것.

### 서버가 임시파일 없이 처리한다는 보장

`/detect` HTTP 엔드포인트는 업로드된 파일을 디스크 임시파일로 저장하지 않고 메모리 bytes로 직접 처리한다. `tempfile.mkstemp`가 호출되지 않으며, 요청 처리 후 디스크에 파일이 남지 않는다.

```bash
# 서버 실행
uvicorn server:app --host 127.0.0.1 --port 8000

# 업로드 (임시파일 없이 메모리에서 바로 처리됨)
curl -F "file=@photo.jpg" http://127.0.0.1:8000/detect
```

응답의 `"image"` 필드는 업로드 파일의 원본 파일명(`file.filename`)으로 채워진다.

---

## 경량 모드 (ONNX Runtime)

### 왜 경량인가?

| 항목 | torch 백엔드 | onnx 백엔드 |
|------|-------------|-------------|
| 런타임 의존성 | torch (~2GB), transformers | onnxruntime (~75MB), numpy, Pillow |
| 설치 크기 | ~2.5GB | ~100MB |
| 추론 속도 | 보통 | 동등~빠름 (INT8 양자화 시) |
| 크로스플랫폼 | 제한적 | CPU 추론으로 모든 플랫폼 지원 |
| 정확도 | 원본 | INT8 양자화로 소폭 하락 가능 (~1% 이내) |

> **주의**: INT8 dynamic quantization은 모델에 따라 정확도가 소폭 하락할 수 있다. 정밀도가 중요하면 `--no-quantize` 옵션으로 변환 후 사용하거나 torch 백엔드를 유지할 것.

### 1단계: 모델 변환 (빌드/개발 머신에서 1회)

#### 권장: `setup.py` 빌드 스크립트 사용

> **`setup.py`는 setuptools 패키징 파일이 아닌 독립 빌드 스크립트다.** `pip install -e`와 무관하며 `python setup.py ...` 형태로 직접 실행한다.

```bash
# 변환 전용 의존성 설치 + 기본 모델 변환 + self-test (한 번에)
python setup.py

# 의존성 이미 설치된 경우 (재변환 시 빠름)
python setup.py --skip-install

# 저장 디렉토리 지정
python setup.py --output-dir /opt/onnx_models

# 여러 모델 한 번에 변환
python setup.py Organika/sdxl-detector umm-maybe/AI-image-detector

# 도움말
python setup.py --help
```

`setup.py`는 다음을 자동으로 수행한다:
1. 의존성 설치 (`requirements-convert.txt`)
2. ONNX export + `meta.json` 생성 (`convert_to_onnx.py --no-quantize` 재사용)
3. stale `model_quantized.onnx` 제거 (멱등성 보장)
4. **MatMul-only INT8 양자화** (`op_types_to_quantize=['MatMul']` — Conv 제외로 ConvInteger 생성 차단)
5. self-test: ConvInteger 0개 확인 + CPUExecutionProvider 로드 + 더미 추론

> **왜 `setup.py`를 쓰는가**: `convert_to_onnx.py`의 기본 양자화(`--quantize-arch portable`)는 Conv까지 양자화해 ConvInteger 노드를 생성한다. onnxruntime CPU EP는 ConvInteger를 미구현(NOT_IMPLEMENTED)으로 처리해 **모든 플랫폼에서 세션 로드가 실패**한다. `setup.py`는 MatMul만 양자화해 이 문제를 회피하며, 크기도 337MB → 91MB로 줄어든다.

#### 개별 스크립트 사용 (세밀한 제어가 필요한 경우)

```bash
# 변환 전용 의존성 설치 (torch, transformers, optimum 포함)
pip install -r requirements-convert.txt

# 모델 ONNX + INT8 양자화 변환 (기본: portable, 모든 CPU 호환)
python convert_to_onnx.py --model Organika/sdxl-detector

# 양자화 없이 변환만 (정확도 우선)
python convert_to_onnx.py --model Organika/sdxl-detector --no-quantize

# 아키텍처별 최적화 양자화 (기본: portable)
python convert_to_onnx.py --model Organika/sdxl-detector --quantize-arch portable   # 기본: 모든 CPU 호환
python convert_to_onnx.py --model Organika/sdxl-detector --quantize-arch avx2       # Intel/AMD x86-64
python convert_to_onnx.py --model Organika/sdxl-detector --quantize-arch arm64      # Apple Silicon / ARM
python convert_to_onnx.py --model Organika/sdxl-detector --quantize-arch avx512_vnni  # 최신 Intel 전용

# 저장 디렉토리 지정
python convert_to_onnx.py --model Organika/sdxl-detector --output-dir /opt/onnx_models
```

> **`--quantize-arch` 선택 가이드**:
> - `portable` (기본): onnxruntime의 `quantize_dynamic` 직접 사용. 아키텍처 프리셋 없이 모든 CPU에서 동작. **크로스플랫폼 배포 시 권장**.
> - `avx2`: Intel/AMD x86-64 서버 전용 최적화.
> - `avx512_vnni`: 최신 Intel Xeon/Core 전용. ARM·구형 x86에서는 오류나 정확도 저하 가능.
> - `arm64`: Apple Silicon(M1/M2) 또는 ARM 서버 전용.

변환 후 `onnx_models/Organika__sdxl-detector/` 디렉토리에 `.onnx` 파일과 `meta.json`이 생성된다. `meta.json`에는 사용된 양자화 모드(`quantize_arch`)도 기록된다.

### 2단계: 경량 런타임 설치 및 추론

```bash
# 런타임 경량 의존성 설치 (torch 불필요!)
pip install -r requirements-onnx.txt

# CLI에서 onnx 백엔드 사용
python detect.py photo.jpg --backend onnx

# 사용자 지정 모델 디렉토리
python detect.py photo.jpg --backend onnx --onnx-models-dir /opt/onnx_models
```

### 서버에서 ONNX 백엔드 사용

```bash
# 환경변수로 백엔드 선택
DETECTOR_BACKEND=onnx uvicorn server:app --host 127.0.0.1 --port 8000

# 사용자 지정 모델 디렉토리
DETECTOR_BACKEND=onnx ONNX_MODELS_DIR=/opt/onnx_models uvicorn server:app --host 127.0.0.1 --port 8000
```

### 서버 환경변수 (ONNX 관련 추가)

| 환경변수 | 기본값 | 설명 |
|----------|--------|------|
| `DETECTOR_BACKEND` | `onnx` | 추론 백엔드: `onnx`(기본, 번들 경량 모델) 또는 `torch` |
| `ONNX_MODELS_DIR` | (번들 경로) | ONNX 모델 디렉토리. 기본은 저장소 내 번들 경로 |

---

## 한계 및 주의사항

- AI 이미지 탐지는 **100% 정확하지 않다.** 결과는 참고용 신호이며, 중요한 판단의 단독 근거로 삼지 말 것.
- 모델 학습 시점 이후에 등장한 새로운 생성 모델의 이미지는 탐지율이 떨어질 수 있다.
- 리사이즈·압축·편집·스크린샷을 거친 이미지는 정확도가 낮아질 수 있다.
- 메타데이터는 위조 가능하다. 신호가 있으면 AI 생성 가능성이 높지만, 없다고 Real을 보장하지 않는다.
- `_AI_DETECTOR_MOCK=1` 환경변수는 **테스트 전용**이다. 이 값이 설정된 채 실행하면 가짜 고정 결과(AI 확률 0.73)가 나오며, pytest 외부에서는 stderr에 경고가 출력된다. 평소 사용 시에는 절대 설정하지 말 것.
