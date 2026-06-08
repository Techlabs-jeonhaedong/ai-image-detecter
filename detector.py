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

import os
from typing import Any, Callable, Dict, List, Optional

try:
    from transformers import pipeline as transformers_pipeline
except ImportError:  # transformers가 설치되지 않은 환경(CI 등) 대비
    transformers_pipeline = None  # type: ignore

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


def _load_image(image_path: str):
    """
    이미지를 열고 PIL Image를 반환한다.
    가능한 모든 예외를 체크해서 명확한 에러 메시지와 함께 raise.
    """
    from PIL import Image, UnidentifiedImageError

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"File not found: {image_path}")

    if not os.access(image_path, os.R_OK):
        raise PermissionError(f"Permission denied: {image_path}")

    try:
        img = Image.open(image_path)
        img.verify()  # 손상 여부 체크
    except UnidentifiedImageError:
        raise ValueError(f"Not a valid image file: {image_path}")
    except Image.DecompressionBombError:
        raise ValueError(f"Image too large (potential decompression bomb): {image_path}")
    except Image.DecompressionBombWarning:
        raise ValueError(f"Image too large (potential decompression bomb): {image_path}")
    except Exception as e:
        raise ValueError(f"Cannot open image '{image_path}': {e}")

    # verify() 후 이미지가 닫히므로 다시 열어야 함
    try:
        img = Image.open(image_path)
        img.load()  # 완전히 디코딩
    except Image.DecompressionBombError:
        raise ValueError(f"Image too large (potential decompression bomb): {image_path}")
    except Image.DecompressionBombWarning:
        raise ValueError(f"Image too large (potential decompression bomb): {image_path}")
    except Exception as e:
        raise ValueError(f"Cannot load image data '{image_path}': {e}")

    return img


def analyze_image(
    image_path: str,
    pipeline_fn: Callable,
    model_id: str,
    threshold: float,
) -> Dict[str, Any]:
    """
    단일 이미지를 분석하고 결과 dict를 반환한다.

    Args:
        image_path: 분석할 이미지 파일 경로
        pipeline_fn: transformers.pipeline 또는 호환 callable (DI / mock용)
        model_id: 사용할 HuggingFace 모델 ID
        threshold: AI 판정 임계값

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
    base_result = {
        "image": image_path,
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

    result = _run_inference(image_path, pipe, model_id, threshold)
    # models 상세 추가
    result["models"] = [
        {"model": model_id, "ai_probability": result["ai_probability"], "error": result["error"]}
    ]
    # metadata 기본값 추가 (단일 모델 경로에서도 schema 일관성 유지)
    if "metadata" not in result:
        result["metadata"] = {"has_ai_signal": False, "signals": [], "source": None, "checked": False}
    return result


def _run_inference(
    image_path: str,
    pipe: Callable,
    model_id: str,
    threshold: float,
    pil_image=None,
) -> Dict[str, Any]:
    """
    단일 이미지에 대해 이미 생성된 pipe로 추론을 실행한다.
    analyze_image / analyze_images / analyze_image_ensemble 에서 공통으로 사용.

    Args:
        pil_image: 이미 로드된 PIL Image (있으면 재사용, 없으면 새로 로드)
    """
    base_result = {
        "image": image_path,
        "ai_probability": None,
        "verdict": None,
        "model": model_id,
        "error": None,
    }
    try:
        if pil_image is not None:
            img = pil_image.convert("RGB")
        else:
            img = _load_image(image_path)
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
    image_path: str,
    model_pipelines: Dict[str, Callable],
    threshold: float,
) -> Dict[str, Any]:
    """
    여러 모델(pipeline)로 단일 이미지를 앙상블 추론한다.
    이미지는 1회만 로드해 재사용한다.

    Args:
        image_path: 분석할 이미지 경로
        model_pipelines: {model_id: pipeline_callable} 딕셔너리
        threshold: AI 판정 임계값

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

    base_result: Dict[str, Any] = {
        "image": image_path,
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

        infer_result = _run_inference(image_path, pipe, model_id, threshold, pil_image=pil_image)
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
    image_paths: List[str],
    pipeline_fn: Callable,
    model_id: str,
    threshold: float,
) -> List[Dict[str, Any]]:
    """
    여러 이미지를 순서대로 분석한다. 개별 실패는 에러 결과로 기록되고 나머지는 계속 처리.
    pipeline 생성 실패 시에도 프로세스가 종료되지 않고 모든 이미지에 에러 결과를 반환한다.
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
                "image": p,
                "error": err,
                "models": [{"model": model_id, "ai_probability": None, "error": err}],
                "metadata": {"has_ai_signal": False, "signals": [], "source": None, "checked": False},
            }
            for p in image_paths
        ]

    results = []
    for p in image_paths:
        r = _run_inference(p, pipe, model_id, threshold)
        r["models"] = [
            {"model": model_id, "ai_probability": r["ai_probability"], "error": r["error"]}
        ]
        if "metadata" not in r:
            r["metadata"] = {"has_ai_signal": False, "signals": [], "source": None, "checked": False}
        results.append(r)
    return results


def analyze_images_batch(
    image_paths: List[str],
    pipeline_fn: Callable,
    model_ids: List[str],
    threshold: float,
) -> List[Dict[str, Any]]:
    """
    여러 이미지를 여러 모델로 앙상블 분석한다.
    모델별 pipeline을 1회만 생성해 모든 이미지에 재사용한다.

    Args:
        image_paths: 분석할 이미지 경로 목록
        pipeline_fn: pipeline 팩토리 함수
        model_ids: 모델 ID 목록
        threshold: AI 판정 임계값

    Returns:
        이미지별 analyze_image_ensemble 결과 리스트
    """
    # 모델별 pipeline 1회 생성 후 캐시
    cached_pipelines: Dict[str, Optional[Callable]] = {}
    for model_id in model_ids:
        try:
            cached_pipelines[model_id] = pipeline_fn("image-classification", model=model_id)
        except Exception as e:
            cached_pipelines[model_id] = None  # 실패 시 None으로 표시

    results = []
    for image_path in image_paths:
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
                model_entry["error"] = f"Pipeline creation failed for {model_id}"
                per_model_results.append(model_entry)
                continue

            infer_result = _run_inference(image_path, pipe, model_id, threshold, pil_image=pil_image)
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
                "image": image_path,
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
                "image": image_path,
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
