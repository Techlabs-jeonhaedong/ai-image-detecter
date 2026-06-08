"""
경량 ONNX Runtime 추론 모듈.

의존성: onnxruntime, numpy, Pillow (torch/transformers/optimum 불필요)

사용법:
    from onnx_detector import get_onnx_pipeline_fn
    pipeline_fn = get_onnx_pipeline_fn(onnx_models_dir="onnx_models")
    # detector.py의 analyze_images_batch 등에 pipeline_fn으로 주입
    from detector import analyze_images_batch
    results = analyze_images_batch(
        image_paths=["img.jpg"],
        pipeline_fn=pipeline_fn,
        model_ids=["Organika/sdxl-detector"],
        threshold=0.5,
    )
"""
import json
import os
from typing import Any, Callable, Dict, List, Optional

import numpy as np
from PIL import Image

# 세션 캐시: {model_id: InferenceSession}
_SESSION_CACHE: Dict[str, Any] = {}


def _clear_session_cache() -> None:
    """테스트 격리용 세션 캐시 초기화."""
    _SESSION_CACHE.clear()


def _sanitize_model_id(model_id: str) -> str:
    """
    모델 ID의 '/'를 '__'로 치환해 파일시스템 안전한 경로 이름으로 변환.

    Examples:
        "Organika/sdxl-detector" → "Organika__sdxl-detector"
        "a/b/c" → "a__b__c"
    """
    return model_id.replace("/", "__")


def _validate_model_path(model_id: str, onnx_models_dir: str) -> str:
    """
    model_id와 onnx_models_dir을 조합한 최종 모델 경로를 검증해 반환.

    path traversal 공격 방어:
      1. model_id에 '..'·절대경로 시작('/')·백슬래시가 있으면 즉시 거부
      2. sanitize 후 realpath로 경로 해소해 onnx_models_dir 밖인지 재검증

    Args:
        model_id: HuggingFace 모델 ID (예: "Organika/sdxl-detector")
        onnx_models_dir: ONNX 모델 베이스 디렉토리

    Returns:
        검증된 모델 디렉토리 절대 경로

    Raises:
        ValueError: model_id가 models_dir 밖을 가리킬 때
    """
    # 1차 방어: '..' 컴포넌트, 절대경로, 백슬래시 포함 여부를 raw 문자열에서 검사
    # '/' 구분자로 각 컴포넌트를 분리해 '..' 컴포넌트 탐지
    parts = model_id.replace("\\", "/").split("/")
    for part in parts:
        if part == "..":
            raise ValueError(f"Unsafe model id/path: {model_id!r}")
    # 절대경로 시작 (Unix '/', Windows 드라이브 문자)
    if model_id.startswith("/") or (len(model_id) >= 2 and model_id[1] == ":"):
        raise ValueError(f"Unsafe model id/path: {model_id!r}")

    # 2차 방어: sanitize 후 realpath로 경계 벗어남 재확인
    sanitized = _sanitize_model_id(model_id)
    base_real = os.path.realpath(os.path.abspath(onnx_models_dir))
    model_path = os.path.realpath(os.path.join(base_real, sanitized))
    try:
        common = os.path.commonpath([base_real, model_path])
    except ValueError:
        raise ValueError(f"Unsafe model id/path: {model_id!r}")
    if common != base_real:
        raise ValueError(f"Unsafe model id/path: {model_id!r}")
    return model_path


def _validate_meta(meta: Dict[str, Any]) -> None:
    """
    meta.json 스키마를 검증한다. 위반 시 ValueError 발생.

    필수 키: (image_size 또는 height+width), image_mean, image_std, id2label
    - image_size / crop_size: 정수, 1~4096 범위
    - image_mean / image_std: 길이 3 숫자 리스트
    - id2label: dict

    Args:
        meta: _load_meta로 로드한 dict

    Raises:
        ValueError: 스키마 위반 시 명확한 메시지
    """
    # image_size 또는 height/width 필수
    if "image_size" not in meta and not ("height" in meta and "width" in meta):
        raise ValueError("Invalid meta.json: 'image_size' or 'height'+'width' required")

    for key in ("image_size", "crop_size"):
        if key in meta:
            val = meta[key]
            if not isinstance(val, int):
                raise ValueError(f"Invalid meta.json: '{key}' must be int, got {type(val).__name__}")
            if not (1 <= val <= 4096):
                raise ValueError(f"Invalid meta.json: '{key}' out of range 1-4096, got {val}")

    for key in ("height", "width"):
        if key in meta:
            val = meta[key]
            if not isinstance(val, int):
                raise ValueError(f"Invalid meta.json: '{key}' must be int, got {type(val).__name__}")
            if not (1 <= val <= 4096):
                raise ValueError(f"Invalid meta.json: '{key}' out of range 1-4096, got {val}")

    for key in ("image_mean", "image_std"):
        if key not in meta:
            raise ValueError(f"Invalid meta.json: '{key}' required")
        val = meta[key]
        if not isinstance(val, (list, tuple)) or len(val) != 3:
            raise ValueError(f"Invalid meta.json: '{key}' must be a list of 3 numbers")
        for v in val:
            if not isinstance(v, (int, float)):
                raise ValueError(f"Invalid meta.json: '{key}' elements must be numeric")

    if "id2label" not in meta:
        raise ValueError("Invalid meta.json: 'id2label' required")
    if not isinstance(meta["id2label"], dict):
        raise ValueError("Invalid meta.json: 'id2label' must be a dict")


def _softmax(logits: np.ndarray) -> np.ndarray:
    """
    수치적으로 안정적인 softmax.

    Args:
        logits: 1D numpy array

    Returns:
        확률값 1D array (합이 1)
    """
    shifted = logits - np.max(logits)
    exp_x = np.exp(shifted)
    return exp_x / exp_x.sum()


def _build_label_scores(id2label: Dict[str, str], probs: np.ndarray) -> List[Dict[str, Any]]:
    """
    id2label 매핑과 확률 배열로 [{label, score}, ...] 리스트를 구성한다.
    원래 인덱스 순서를 유지하며 score는 Python float으로 반환.

    Args:
        id2label: {"0": "artificial", "1": "human", ...}
        probs: softmax 적용된 확률 배열

    Returns:
        [{"label": str, "score": float}, ...] (기존 extract_ai_probability 소비 가능)
    """
    result = []
    for i, prob in enumerate(probs):
        label = id2label.get(str(i), f"label_{i}")
        result.append({"label": label, "score": float(prob)})
    return result


def _center_crop(img: Image.Image, crop_h: int, crop_w: int) -> Image.Image:
    """
    PIL 이미지를 중앙에서 crop_h x crop_w 크기로 center-crop한다.

    Args:
        img: PIL Image
        crop_h: crop 높이 (pixels)
        crop_w: crop 너비 (pixels)

    Returns:
        center-crop된 PIL Image
    """
    w, h = img.size
    left = (w - crop_w) // 2
    top = (h - crop_h) // 2
    right = left + crop_w
    bottom = top + crop_h
    return img.crop((left, top, right, bottom))


def preprocess_image(image: Image.Image, meta: Dict[str, Any]) -> np.ndarray:
    """
    PIL 이미지를 ONNX 추론용 배치 텐서로 전처리.
    torch/transformers 없이 numpy로 직접 구현.

    resize_mode에 따라 두 가지 경로를 지원:
      - "shortest_edge": 짧은 변을 image_size로 aspect 유지 resize 후 crop_size로 center-crop
      - "exact" (기본): image_size × image_size 정사각 resize

    Args:
        image: PIL Image (모드 무관, 내부에서 RGB 변환)
        meta: convert_to_onnx.py가 저장한 전처리 설정
            {
                "image_size": int,          # exact 모드의 정사각 크기 / shortest_edge 모드의 shortest_edge 값
                "crop_size": int,           # (선택) shortest_edge 모드의 center-crop 크기. 없으면 image_size
                "resize_mode": str,         # "shortest_edge" | "exact" (기본: "exact")
                "image_mean": [R, G, B],
                "image_std": [R, G, B],
                "do_normalize": bool,
                "do_rescale": bool,
                "rescale_factor": float,    # 보통 1/255
                "resample": int,            # PIL resample 필터 (3=BICUBIC 기본)
            }

    Returns:
        shape (1, 3, H, W), dtype float32
    """
    _validate_meta(meta)

    resize_mode = meta.get("resize_mode", "exact")
    resample = meta.get("resample", 3)  # 기본 BICUBIC (transformers 공통 기본값)

    # RGB 변환 (grayscale, RGBA 등 모두 처리)
    img = image.convert("RGB")

    if resize_mode == "shortest_edge":
        # 짧은 변을 image_size로 aspect 유지 resize
        size = meta["image_size"]
        w, h = img.size
        if w <= h:
            new_w = size
            new_h = int(h * size / w)
        else:
            new_h = size
            new_w = int(w * size / h)
        img = img.resize((new_w, new_h), resample)
        # crop_size로 center-crop (없으면 image_size)
        crop_size = meta.get("crop_size", size)
        img = _center_crop(img, crop_size, crop_size)
    else:
        # exact 모드: 정사각 resize
        size = meta["image_size"]
        img = img.resize((size, size), resample)

    # numpy 변환: (H, W, 3), float32
    arr = np.array(img, dtype=np.float32)

    # Rescale (픽셀값 → [0, 1])
    if meta.get("do_rescale", True):
        arr = arr * meta.get("rescale_factor", 1.0 / 255.0)

    # Normalize: (x - mean) / std
    if meta.get("do_normalize", True):
        mean = np.array(meta["image_mean"], dtype=np.float32)
        std = np.array(meta["image_std"], dtype=np.float32)
        arr = (arr - mean) / std

    # HWC → CHW
    arr = arr.transpose(2, 0, 1)

    # 배치 차원 추가: (3, H, W) → (1, 3, H, W)
    arr = arr[np.newaxis, :]

    return arr.astype(np.float32)


def _load_meta(model_dir: str) -> Dict[str, Any]:
    """model_dir/meta.json 로드 후 스키마 검증."""
    meta_path = os.path.join(model_dir, "meta.json")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    _validate_meta(meta)
    return meta


def _find_onnx_file(model_dir: str) -> str:
    """
    model_dir에서 .onnx 파일을 찾아 경로 반환.
    양자화된 파일(model_quantized.onnx)을 우선, 없으면 첫 번째 .onnx 사용.
    """
    quantized = os.path.join(model_dir, "model_quantized.onnx")
    if os.path.exists(quantized):
        return quantized
    for fname in os.listdir(model_dir):
        if fname.endswith(".onnx"):
            return os.path.join(model_dir, fname)
    raise FileNotFoundError(f"ONNX 파일을 찾을 수 없습니다: {model_dir}")


def _get_or_create_session(model_dir: str, model_id: str) -> Any:
    """
    모델 ID별 InferenceSession 캐시. 없으면 생성, 있으면 재사용.
    """
    if model_id not in _SESSION_CACHE:
        import onnxruntime
        onnx_path = _find_onnx_file(model_dir)
        session = onnxruntime.InferenceSession(onnx_path)
        _SESSION_CACHE[model_id] = session
    return _SESSION_CACHE[model_id]


def get_onnx_pipeline_fn(onnx_models_dir: str = "onnx_models") -> Callable:
    """
    detector.py의 pipeline_fn과 동일한 인터페이스를 제공하는 ONNX 기반 팩토리 반환.

    반환된 함수는 `(task, model=model_id)` → `infer(pil_image)` 콜러블.

    Args:
        onnx_models_dir: ONNX 모델 디렉토리 (convert_to_onnx.py의 --output-dir)

    Returns:
        pipeline_fn(task, model=model_id) → infer 콜러블

    Raises:
        FileNotFoundError: 모델 디렉토리가 없을 때 (infer 호출 시)
        ValueError: task가 "image-classification"이 아니거나 model_id가 안전하지 않을 때
    """
    def pipeline_fn(task: str, model: str = "", **kwargs) -> Callable:
        if task != "image-classification":
            raise ValueError(
                f"ONNX 백엔드는 'image-classification' task만 지원합니다. 입력값: {task!r}"
            )

        # path traversal 방어: model_id 검증
        model_dir = _validate_model_path(model, onnx_models_dir)

        if not os.path.isdir(model_dir):
            raise FileNotFoundError(
                f"ONNX 모델 미변환: '{model}'. "
                f"먼저 convert_to_onnx.py 실행 필요 "
                f"(python convert_to_onnx.py --model {model})"
            )

        meta = _load_meta(model_dir)
        session = _get_or_create_session(model_dir, model)
        id2label: Dict[str, str] = meta.get("id2label", {})
        input_name: str = session.get_inputs()[0].name

        def infer(pil_image: Image.Image) -> List[Dict[str, Any]]:
            input_tensor = preprocess_image(pil_image, meta)
            outputs = session.run(None, {input_name: input_tensor})
            logits = outputs[0][0]  # shape (num_classes,)
            probs = _softmax(logits)
            return _build_label_scores(id2label, probs)

        return infer

    return pipeline_fn
