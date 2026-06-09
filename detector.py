"""
AI 이미지 탐지 핵심 로직 모듈.

기본 모델: Organika/sdxl-detector
  - 실제 출력 라벨: "artificial" (AI 생성) / "human" (실제 사진)
  - ViT 기반 image-classification pipeline 사용

앙상블 모델 세트 (ENSEMBLE_MODELS):
  - Organika/sdxl-detector          : ViT 기반 sdxl 탐지기
  - yaya36095/ai-image-detector     : ViT 기반 범용 AI 이미지 탐지기
  - umm-maybe/AI-image-detector     : ViT 기반 AI 이미지 탐지기

라벨 매칭은 대소문자 무시 + 여러 변형 허용:
  AI 계열: ai, artificial, fake, generated, synthetic
  Real 계열: real, human, natural, photo, authentic
"""

import io
import os
from typing import Any, BinaryIO, Callable, Dict, List, Optional, Union

# 이미지 입력 소스 타입 별칭
# str: 파일 경로, bytes/bytearray: 파일 내용, BinaryIO: read()를 가진 바이너리 스트림
ImageSource = Union[str, bytes, bytearray, BinaryIO]

try:
    from transformers import pipeline as transformers_pipeline
except ImportError:  # transformers가 설치되지 않은 환경(CI 등) 대비
    transformers_pipeline = None  # type: ignore

# ── 이미지 해상도 상한 ──────────────────────────────────────────────────────
# PIL DecompressionBomb 방어: 1억 픽셀(약 10000x10000)로 명시 설정.
# 기본값(178M)보다 낮춰 메모리 DoS 위험 감소.
MAX_IMAGE_PIXELS: int = 100_000_000
try:
    from PIL import Image as _PILImage
    _PILImage.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS
except Exception:
    pass

# 기본 모델
DEFAULT_MODEL = "Organika/sdxl-detector"

# 앙상블 모델 세트 — 모두 실재하는 ViT image-classification AI 탐지기
ENSEMBLE_MODELS = [
    "Organika/sdxl-detector",        # ViT sdxl 탐지기 (기본 모델)
    "yaya36095/ai-image-detector",   # ViT 범용 AI 이미지 탐지기
    "umm-maybe/AI-image-detector",   # ViT AI 이미지 탐지기
]

# AI 이미지를 나타내는 라벨 키워드 (소문자)
_AI_LABEL_KEYWORDS = {"ai", "artificial", "fake", "generated", "synthetic"}
# Real 이미지를 나타내는 라벨 키워드 (소문자)
_REAL_LABEL_KEYWORDS = {"real", "human", "natural", "photo", "authentic"}

def _is_ai_label(label: str) -> bool:
    return label.lower() in _AI_LABEL_KEYWORDS


def _is_real_label(label: str) -> bool:
    return label.lower() in _REAL_LABEL_KEYWORDS


def extract_ai_probability(results: Any) -> float:
    """
    모델 출력(라벨-점수 리스트)에서 AI 생성 확률을 추출한다.

    Args:
        results: [{"label": str, "score": float}, ...] 형식의 리스트

    Returns:
        AI 생성 확률 (0.0 ~ 1.0)

    Raises:
        ValueError: 스키마 불일치, 빈 리스트, 알 수 없는 라벨인 경우
    """
    # 스키마 방어: list 타입 확인
    if not isinstance(results, list):
        raise ValueError(f"Malformed model output: expected list, got {type(results).__name__}")

    # 빈 리스트 확인
    if not results:
        raise ValueError("Model returned empty results list")

    # 각 항목 스키마 검증
    for i, item in enumerate(results):
        if not isinstance(item, dict):
            raise ValueError(
                f"Malformed model output: item[{i}] expected dict, got {type(item).__name__}"
            )
        if "label" not in item:
            raise ValueError(f"Malformed model output: item[{i}] missing 'label' key")
        if "score" not in item:
            raise ValueError(f"Malformed model output: item[{i}] missing 'score' key")
        if item["label"] is None:
            raise ValueError(f"Malformed model output: item[{i}]['label'] is None")
        try:
            float(item["score"])
        except (TypeError, ValueError):
            raise ValueError(
                f"Malformed model output: item[{i}]['score'] cannot be converted to float"
            )

    # label 정규화: str 변환 + strip
    def _normalize_label(raw_label: Any) -> str:
        return str(raw_label).strip()

    # AI 라벨 점수 직접 탐색
    for item in results:
        label = _normalize_label(item["label"])
        if _is_ai_label(label):
            return float(item["score"])

    # AI 라벨 없으면 Real 라벨의 보완 값 사용
    for item in results:
        label = _normalize_label(item["label"])
        if _is_real_label(label):
            return 1.0 - float(item["score"])

    known_labels = [_normalize_label(item["label"]) for item in results]
    raise ValueError(
        f"Cannot determine AI probability from label(s): {known_labels}. "
        f"Expected AI keywords {_AI_LABEL_KEYWORDS} or Real keywords {_REAL_LABEL_KEYWORDS}."
    )


def determine_verdict(ai_probability: float, threshold: float) -> str:
    """
    AI 확률과 threshold로 최종 판정을 반환한다.

    Args:
        ai_probability: 0.0 ~ 1.0 사이의 AI 생성 확률
        threshold: 0.0 ~ 1.0 사이의 판정 임계값

    Returns:
        "AI-generated" 또는 "Real"

    Raises:
        ValueError: 범위를 벗어난 값 입력 시
    """
    if not (0.0 <= threshold <= 1.0):
        raise ValueError(f"threshold must be in [0, 1], got {threshold}")
    if not (0.0 <= ai_probability <= 1.0):
        raise ValueError(f"ai_probability must be in [0, 1], got {ai_probability}")

    return "AI-generated" if ai_probability >= threshold else "Real"


def _sanitize_for_terminal(text: str, max_len: int = 120) -> str:
    """
    사람용 출력에 찍기 전 제어문자/ANSI 이스케이프 제거 및 길이 상한 적용.
    JSON 출력에는 적용하지 않는다 (json.dumps가 이스케이프하므로 안전).

    Args:
        text: 새니타이즈할 문자열
        max_len: 최대 길이 (기본 120)

    Returns:
        제어문자가 제거되고 max_len 이하로 잘린 문자열
    """
    import re as _re
    # ANSI/C0/C1 제어문자 제거: 0x00-0x1F (탭/뉴라인 포함), 0x7F-0x9F
    cleaned = _re.sub(r"[\x00-\x1f\x7f-\x9f]", "", text)
    return cleaned[:max_len]


def _apply_metadata_override(result: Dict[str, Any], meta: Dict[str, Any]) -> Dict[str, Any]:
    """
    메타데이터 신호에 따라 verdict를 override한다.
    결정적 신호(decisive=True)일 때만 override, 약한 흔적은 verdict를 바꾸지 않음.

    Args:
        result: analyze_image_ensemble 등의 결과 dict (in-place 변경하지 않고 복사본 반환)
        meta: inspect_metadata() 반환값

    Returns:
        verdict_source 필드가 추가된 결과 dict
    """
    out = dict(result)
    decisive = meta.get("decisive", False)

    if decisive:
        out["verdict"] = "AI-generated"
        out["verdict_source"] = "metadata"
        # ML 에러가 있어도 메타데이터로 정상 판정 → error를 None으로
        if out.get("error") is not None:
            out["error"] = None
    else:
        out["verdict_source"] = "model"

    return out


def format_result(result: Dict[str, Any], json_mode: bool) -> str:
    """
    단일 이미지 결과를 출력 문자열로 변환한다.

    Args:
        result: analyze_image() / analyze_image_ensemble() 의 반환값
        json_mode: True이면 JSON 문자열, False이면 사람이 읽기 좋은 형식

    Returns:
        출력할 문자열
    """
    import json as _json

    if json_mode:
        return _json.dumps(result, ensure_ascii=False)

    image = result["image"]
    model = result["model"]
    error = result["error"]

    if error:
        return f"[ERROR] {image}\n  Error: {error}\n  Model: {model}"

    prob = result["ai_probability"]
    verdict = result["verdict"]
    prob_pct = prob * 100

    verdict_source = result.get("verdict_source", "model")
    verdict_display = f"{verdict} (metadata)" if verdict_source == "metadata" else verdict

    verdict_marker = "[AI]" if verdict == "AI-generated" else "[OK]"
    lines = [
        f"{verdict_marker} {image}",
        f"  Verdict  : {verdict_display}",
        f"  AI Prob  : {prob_pct:.1f}%",
        f"  Model    : {model}",
    ]

    # 앙상블/다중 모델일 때 개별 모델 확률 추가
    models_detail = result.get("models", [])
    if len(models_detail) > 1:
        lines.append("  Per-model:")
        for m in models_detail:
            if m.get("error"):
                lines.append(f"    [{m['model']}] ERROR: {m['error']}")
            elif m.get("ai_probability") is not None:
                lines.append(f"    [{m['model']}] {m['ai_probability'] * 100:.1f}%")

    # 메타데이터 신호 있으면 근거 표시 (신호 문자열 새니타이즈 적용)
    metadata = result.get("metadata")
    if metadata and (metadata.get("has_ai_signal") or metadata.get("signals")):
        label = "AI signal detected" if metadata.get("has_ai_signal") else "weak signal"
        lines.append(f"  Metadata : {label}")
        for sig in metadata.get("signals", [])[:3]:  # 최대 3개만 표시
            lines.append(f"    - {_sanitize_for_terminal(sig)}")

    return "\n".join(lines)


def _normalize_source(source: ImageSource) -> Union[str, bytes]:
    """
    ImageSource를 str(경로) 또는 bytes 중 하나로 정규화한다.

    진입점(detect, analyze_images_batch 등)에서 1회 호출해
    file-like 스트림 소진 버그를 방지한다.

    - str → 그대로 반환
    - bytes/bytearray → bytes 반환 (1회 복사)
    - file-like (.read() 보유) → 1회 read()로 bytes 변환
    - 그 외 타입 → TypeError

    seek 불가 스트림(소켓 등)도 1회 read이므로 안전하다.
    """
    if isinstance(source, str):
        return source
    if isinstance(source, (bytes, bytearray)):
        return bytes(source)
    if hasattr(source, "read") and callable(source.read):
        return source.read()
    raise TypeError(f"Unsupported image source type: {type(source).__name__!r}")


def _load_image(source: ImageSource):
    """
    이미지를 열고 PIL Image를 반환한다.
    source가 str이면 파일 경로로, bytes/bytearray/file-like이면 메모리에서 직접 로드.
    가능한 모든 예외를 체크해서 명확한 에러 메시지와 함께 raise.
    """
    from PIL import Image, UnidentifiedImageError

    # ── 소스 타입 판별 ─────────────────────────────────────────────────────
    is_path = isinstance(source, str)

    if is_path:
        # 경로일 때만 존재/접근 체크
        if not os.path.exists(source):
            raise FileNotFoundError(f"File not found: {source}")
        if not os.access(source, os.R_OK):
            raise PermissionError(f"Permission denied: {source}")
        display_name = source
    elif isinstance(source, (bytes, bytearray)):
        # bytes / bytearray → BytesIO로 감싸기
        display_name = "<in-memory image>"
        raw_bytes: bytes = bytes(source)
    elif hasattr(source, "read") and callable(source.read):
        # file-like: read한 뒤 bytes로 보관 (진입부에서 _normalize_source 미호출 시 폴백)
        display_name = "<in-memory image>"
        raw_bytes = source.read()
    else:
        raise TypeError(f"Unsupported image source type: {type(source).__name__!r}")

    # ── verify() 단계 ──────────────────────────────────────────────────────
    try:
        if is_path:
            img = Image.open(source)
        else:
            img = Image.open(io.BytesIO(raw_bytes))
        img.verify()  # 손상 여부 체크 (verify 후 이미지가 닫힘)
    except UnidentifiedImageError:
        raise ValueError(f"Not a valid image file: {display_name}")
    except Image.DecompressionBombError:
        raise ValueError(f"Image too large (potential decompression bomb): {display_name}")
    except Image.DecompressionBombWarning:
        raise ValueError(f"Image too large (potential decompression bomb): {display_name}")
    except (FileNotFoundError, PermissionError):
        raise
    except Exception as e:
        raise ValueError(f"Cannot open image '{display_name}': {e}")

    # ── verify() 후 재오픈 및 완전 디코딩 ─────────────────────────────────
    try:
        if is_path:
            img = Image.open(source)
        else:
            img = Image.open(io.BytesIO(raw_bytes))  # raw_bytes로 재생성
        img.load()  # 완전히 디코딩
    except Image.DecompressionBombError:
        raise ValueError(f"Image too large (potential decompression bomb): {display_name}")
    except Image.DecompressionBombWarning:
        raise ValueError(f"Image too large (potential decompression bomb): {display_name}")
    except Exception as e:
        raise ValueError(f"Cannot load image data '{display_name}': {e}")

    return img


def _get_display_name(source: ImageSource, name: Optional[str] = None) -> str:
    """
    이미지 소스에서 표시명을 결정한다.
    - 경로(str)면 경로 그대로
    - bytes/file-like + name 지정 → name
    - bytes/file-like + name 미지정 → "<in-memory>"
    """
    if isinstance(source, str):
        return source
    return name if name is not None else "<in-memory>"


def analyze_image(
    image_path: ImageSource,
    pipeline_fn: Callable,
    model_id: str,
    threshold: float,
    *,
    name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    단일 이미지를 분석하고 결과 dict를 반환한다.

    Args:
        image_path: 분석할 이미지 — 파일 경로(str), bytes, bytearray, file-like 모두 허용
        pipeline_fn: transformers.pipeline 또는 호환 callable (DI / mock용)
        model_id: 사용할 HuggingFace 모델 ID
        threshold: AI 판정 임계값
        name: bytes/file-like 입력 시 결과 image 필드에 쓸 표시명 (경로 입력이면 무시)

    Returns:
        {
            "image": str,
            "ai_probability": float | None,
            "verdict": str | None,
            "model": str,
            "error": str | None,
            "models": [{"model": str, "ai_probability": float|None, "error": str|None}],
            "metadata": {"has_ai_signal": bool, "signals": [...], "source": ..., "checked": bool},
        }
    """
    display_name = _get_display_name(image_path, name)
    base_result = {
        "image": display_name,
        "ai_probability": None,
        "verdict": None,
        "model": model_id,
        "error": None,
        "models": [],
        "metadata": {"has_ai_signal": False, "signals": [], "source": None, "checked": False},
    }

    try:
        pipe = pipeline_fn("image-classification", model=model_id)
    except Exception as e:
        err = f"Unexpected error: {e}"
        return {
            **base_result,
            "error": err,
            "models": [{"model": model_id, "ai_probability": None, "error": err}],
        }

    result = _run_inference(image_path, pipe, model_id, threshold, display_name=display_name)
    # models 상세 추가
    result["models"] = [
        {"model": model_id, "ai_probability": result["ai_probability"], "error": result["error"]}
    ]
    # metadata 기본값 추가 (단일 모델 경로에서도 schema 일관성 유지)
    if "metadata" not in result:
        result["metadata"] = {"has_ai_signal": False, "signals": [], "source": None, "checked": False}
    return result


def _run_inference(
    source: ImageSource,
    pipe: Callable,
    model_id: str,
    threshold: float,
    pil_image=None,
    display_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    단일 이미지에 대해 이미 생성된 pipe로 추론을 실행한다.
    analyze_image / analyze_images / analyze_image_ensemble 에서 공통으로 사용.

    Args:
        source: 이미지 소스 (경로/bytes/file-like)
        pil_image: 이미 로드된 PIL Image (있으면 재사용, 없으면 새로 로드)
        display_name: 결과 image 필드에 쓸 표시명 (None이면 source에서 자동 결정)
    """
    if display_name is None:
        display_name = _get_display_name(source)

    base_result = {
        "image": display_name,
        "ai_probability": None,
        "verdict": None,
        "model": model_id,
        "error": None,
    }
    try:
        if pil_image is not None:
            img = pil_image.convert("RGB")
        else:
            img = _load_image(source)
            img = img.convert("RGB")
        raw_results = pipe(img)
        ai_prob = extract_ai_probability(raw_results)
        verdict = determine_verdict(ai_prob, threshold)
        return {**base_result, "ai_probability": ai_prob, "verdict": verdict}
    except (FileNotFoundError, PermissionError, ValueError) as e:
        return {**base_result, "error": str(e)}
    except Exception as e:
        return {**base_result, "error": f"Unexpected error: {e}"}


def analyze_image_ensemble(
    image_path: ImageSource,
    model_pipelines: Dict[str, Callable],
    threshold: float,
    *,
    name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    여러 모델(pipeline)로 단일 이미지를 앙상블 추론한다.
    이미지는 1회만 로드해 재사용한다.

    Args:
        image_path: 분석할 이미지 (경로/bytes/file-like)
        model_pipelines: {model_id: pipeline_callable} 딕셔너리
        threshold: AI 판정 임계값
        name: bytes/file-like 입력 시 결과 image 필드에 쓸 표시명

    Returns:
        확장 스키마 결과 dict:
        {
            "image": str,
            "ai_probability": float | None,   # 성공 모델들의 평균
            "verdict": str | None,
            "model": str,                      # "ensemble(N models)"
            "error": str | None,
            "models": [...],                   # 모델별 개별 결과
            "metadata": {...},
        }
    """
    n = len(model_pipelines)
    model_label = f"ensemble({n} models)"
    display_name = _get_display_name(image_path, name)

    base_result: Dict[str, Any] = {
        "image": display_name,
        "ai_probability": None,
        "verdict": None,
        "model": model_label,
        "error": None,
        "models": [],
        "metadata": {"has_ai_signal": False, "signals": [], "source": None, "checked": False},
    }

    # 이미지 1회 로드 (재사용)
    pil_image = None
    load_error: Optional[str] = None
    try:
        pil_image = _load_image(image_path)
    except Exception as e:
        load_error = str(e)

    per_model_results = []

    for model_id, pipeline_fn in model_pipelines.items():
        model_entry: Dict[str, Any] = {"model": model_id, "ai_probability": None, "error": None}

        if load_error:
            model_entry["error"] = load_error
            per_model_results.append(model_entry)
            continue

        try:
            pipe = pipeline_fn("image-classification", model=model_id)
        except Exception as e:
            model_entry["error"] = f"Unexpected error: {e}"
            per_model_results.append(model_entry)
            continue

        infer_result = _run_inference(
            image_path, pipe, model_id, threshold,
            pil_image=pil_image, display_name=display_name,
        )
        model_entry["ai_probability"] = infer_result["ai_probability"]
        model_entry["error"] = infer_result["error"]
        per_model_results.append(model_entry)

    # 성공한 모델만 평균
    success_probs = [
        m["ai_probability"]
        for m in per_model_results
        if m["error"] is None and m["ai_probability"] is not None
    ]

    if not success_probs:
        # 전부 실패
        errors = [m["error"] for m in per_model_results if m["error"]]
        error_msg = errors[0] if errors else "All models failed"
        return {**base_result, "error": error_msg, "models": per_model_results}

    avg_prob = sum(success_probs) / len(success_probs)
    verdict = determine_verdict(avg_prob, threshold)

    return {
        **base_result,
        "ai_probability": avg_prob,
        "verdict": verdict,
        "models": per_model_results,
    }


def analyze_images(
    image_paths: List[ImageSource],
    pipeline_fn: Callable,
    model_id: str,
    threshold: float,
    *,
    names: Optional[List[Optional[str]]] = None,
) -> List[Dict[str, Any]]:
    """
    여러 이미지를 순서대로 분석한다. 개별 실패는 에러 결과로 기록되고 나머지는 계속 처리.
    pipeline 생성 실패 시에도 프로세스가 종료되지 않고 모든 이미지에 에러 결과를 반환한다.

    Args:
        image_paths: 이미지 소스 목록 (경로/bytes/file-like 혼합 가능)
        names: 각 소스에 대한 표시명 목록 (bytes/file-like 소스에만 의미 있음)
    """
    error_template = {
        "ai_probability": None,
        "verdict": None,
        "model": model_id,
    }

    try:
        pipe = pipeline_fn("image-classification", model=model_id)
    except Exception as e:
        err = f"Unexpected error: {e}"
        return [
            {
                **error_template,
                "image": _get_display_name(p, names[i] if names else None),
                "error": err,
                "models": [{"model": model_id, "ai_probability": None, "error": err}],
                "metadata": {"has_ai_signal": False, "signals": [], "source": None, "checked": False},
            }
            for i, p in enumerate(image_paths)
        ]

    results = []
    for i, p in enumerate(image_paths):
        display = _get_display_name(p, names[i] if names else None)
        r = _run_inference(p, pipe, model_id, threshold, display_name=display)
        r["models"] = [
            {"model": model_id, "ai_probability": r["ai_probability"], "error": r["error"]}
        ]
        if "metadata" not in r:
            r["metadata"] = {"has_ai_signal": False, "signals": [], "source": None, "checked": False}
        results.append(r)
    return results


def analyze_images_batch(
    image_paths: List[ImageSource],
    pipeline_fn: Callable,
    model_ids: List[str],
    threshold: float,
    *,
    names: Optional[List[Optional[str]]] = None,
) -> List[Dict[str, Any]]:
    """
    여러 이미지를 여러 모델로 앙상블 분석한다.
    모델별 pipeline을 1회만 생성해 모든 이미지에 재사용한다.

    Args:
        image_paths: 분석할 이미지 소스 목록 (경로/bytes/file-like 혼합 가능)
        pipeline_fn: pipeline 팩토리 함수
        model_ids: 모델 ID 목록
        threshold: AI 판정 임계값
        names: 각 소스에 대한 표시명 목록 (bytes/file-like 소스에만 의미 있음)

    Returns:
        이미지별 analyze_image_ensemble 결과 리스트
    """
    # 모델별 pipeline 1회 생성 후 캐시
    cached_pipelines: Dict[str, Optional[Callable]] = {}
    pipeline_errors: Dict[str, str] = {}  # 생성 실패 시 에러 메시지 보존
    for model_id in model_ids:
        try:
            cached_pipelines[model_id] = pipeline_fn("image-classification", model=model_id)
        except Exception as e:
            cached_pipelines[model_id] = None  # 실패 시 None으로 표시
            pipeline_errors[model_id] = str(e)

    results = []
    for idx, image_path in enumerate(image_paths):
        # 진입부 1회 정규화: file-like 스트림 소진 버그 방지
        try:
            image_path = _normalize_source(image_path)
        except TypeError as e:
            display_name = _get_display_name(image_path, names[idx] if names else None)
            n = len(model_ids)
            model_label = f"ensemble({n} models)" if n > 1 else model_ids[0]
            results.append({
                "image": display_name,
                "ai_probability": None,
                "verdict": None,
                "model": model_label,
                "error": str(e),
                "models": [{"model": m, "ai_probability": None, "error": str(e)} for m in model_ids],
                "metadata": {"has_ai_signal": False, "decisive": False, "signals": [], "source": None, "checked": False},
            })
            continue

        display_name = _get_display_name(image_path, names[idx] if names else None)

        # 이미지 1회 로드
        pil_image = None
        load_error: Optional[str] = None
        try:
            pil_image = _load_image(image_path)
        except Exception as e:
            load_error = str(e)

        per_model_results = []
        for model_id in model_ids:
            model_entry: Dict[str, Any] = {"model": model_id, "ai_probability": None, "error": None}
            if load_error:
                model_entry["error"] = load_error
                per_model_results.append(model_entry)
                continue

            pipe = cached_pipelines.get(model_id)
            if pipe is None:
                err_msg = pipeline_errors.get(model_id) or f"Pipeline creation failed for {model_id}"
                model_entry["error"] = err_msg
                per_model_results.append(model_entry)
                continue

            infer_result = _run_inference(
                image_path, pipe, model_id, threshold,
                pil_image=pil_image, display_name=display_name,
            )
            model_entry["ai_probability"] = infer_result["ai_probability"]
            model_entry["error"] = infer_result["error"]
            per_model_results.append(model_entry)

        success_probs = [
            m["ai_probability"]
            for m in per_model_results
            if m["error"] is None and m["ai_probability"] is not None
        ]

        n = len(model_ids)
        model_label = f"ensemble({n} models)" if n > 1 else model_ids[0]

        if not success_probs:
            errors = [m["error"] for m in per_model_results if m["error"]]
            error_msg = errors[0] if errors else "All models failed"
            results.append({
                "image": display_name,
                "ai_probability": None,
                "verdict": None,
                "model": model_label,
                "error": error_msg,
                "models": per_model_results,
                "metadata": {"has_ai_signal": False, "decisive": False, "signals": [], "source": None, "checked": False},
            })
        else:
            avg_prob = sum(success_probs) / len(success_probs)
            verdict = determine_verdict(avg_prob, threshold)
            results.append({
                "image": display_name,
                "ai_probability": avg_prob,
                "verdict": verdict,
                "model": model_label,
                "error": None,
                "models": per_model_results,
                "metadata": {"has_ai_signal": False, "decisive": False, "signals": [], "source": None, "checked": False},
            })

    return results


def _make_pipeline_with_trust_guard(task: str, **kwargs) -> Any:
    """
    transformers.pipeline을 trust_remote_code=False로 고정해서 호출한다.
    직접 호출하지 말고 get_real_pipeline()이 반환하는 래퍼를 사용할 것.
    """
    kwargs.setdefault("trust_remote_code", False)
    return transformers_pipeline(task, **kwargs)


def get_real_pipeline():
    """
    실제 transformers pipeline을 반환한다. 모델은 첫 호출 시 자동 다운로드됨.
    trust_remote_code=False를 명시적으로 전달해 공급망 공격을 방어한다.
    테스트에서는 이 함수 대신 mock_pipeline을 주입해서 사용.
    """
    return _make_pipeline_with_trust_guard


def detect(
    source: ImageSource,
    *,
    name: Optional[str] = None,
    backend: str = "onnx",
    models: Optional[List[str]] = None,
    ensemble: bool = False,
    threshold: float = 0.5,
    with_metadata: bool = True,
    onnx_models_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    고수준 편의 API. 이미지 소스를 받아 AI 탐지 결과 dict를 반환한다.

    Args:
        source: 이미지 소스 — 파일 경로(str), bytes, bytearray, file-like 모두 허용
        name: bytes/file-like 입력 시 결과 image 필드에 쓸 표시명 (경로 입력이면 무시)
        backend: 추론 백엔드 ("onnx"(기본) 또는 "torch")
        models: 사용할 모델 ID 목록 (None이면 기본 모델)
        ensemble: True면 ENSEMBLE_MODELS 전체 사용
        threshold: AI 판정 임계값 (기본 0.5)
        with_metadata: True면 메타데이터 검사 결과를 포함하고 verdict_source 설정
        onnx_models_dir: ONNX 모델 디렉토리 (None이면 번들 기본 경로 사용)

    Returns:
        단일 이미지 분석 결과 dict (verdict, ai_probability, metadata, verdict_source 포함)

    Example:
        >>> result = detect(open("photo.jpg", "rb").read(), name="photo.jpg", ensemble=True)
        >>> print(result["verdict"])  # "AI-generated" or "Real"
    """
    from backends import get_pipeline_fn_with_mock, DEFAULT_ONNX_MODELS_DIR
    from metadata import inspect_metadata

    if onnx_models_dir is None:
        onnx_models_dir = DEFAULT_ONNX_MODELS_DIR

    # ── 진입부 1회 정규화: file-like 스트림 소진 버그 방지 ─────────────────
    # str/bytes는 그대로, file-like는 여기서 1회 read() → bytes로 변환.
    # 이후 _load_image와 inspect_metadata 양쪽이 같은 bytes 객체를 사용한다.
    source = _normalize_source(source)

    pipeline_fn = get_pipeline_fn_with_mock(backend, onnx_models_dir)

    # 모델 목록 결정
    import sys as _sys
    from backends import resolve_ensemble_models_for_onnx
    is_mock = os.environ.get("_AI_DETECTOR_MOCK") == "1"

    ensemble_model_ids: List[str] = list(ENSEMBLE_MODELS) if ensemble else []
    explicit_model_ids: List[str] = []
    if models:
        for m in models:
            if m not in explicit_model_ids:
                explicit_model_ids.append(m)

    model_ids = resolve_ensemble_models_for_onnx(
        ensemble_models=ensemble_model_ids,
        explicit_models=explicit_model_ids,
        backend=backend,
        onnx_models_dir=onnx_models_dir,
        is_mock=is_mock,
        warn=lambda msg: _sys.stderr.write(msg + "\n"),
    )

    if not model_ids:
        model_ids = [DEFAULT_MODEL]

    # 배치 함수로 처리 (단일 이미지도 리스트로 감싸서)
    raw_results = analyze_images_batch(
        image_paths=[source],
        pipeline_fn=pipeline_fn,
        model_ids=model_ids,
        threshold=threshold,
        names=[name],
    )
    result = raw_results[0]

    # 단일 모델이면 model 필드를 ID 자체로 표시
    if len(model_ids) == 1:
        result["model"] = model_ids[0]

    # 메타데이터 검사
    if with_metadata:
        meta = inspect_metadata(source)
        result["metadata"] = meta
        result = _apply_metadata_override(result, meta)
    else:
        result["metadata"] = {
            "has_ai_signal": False,
            "decisive": False,
            "signals": [],
            "source": None,
            "checked": False,
        }
        result["verdict_source"] = "model"

    return result
