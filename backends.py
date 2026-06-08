"""
백엔드 선택 공용 헬퍼.

detect.py와 server.py에서 공유하는 _get_backend_pipeline_fn 로직을 한 곳에서 관리.
백엔드 문자열 정규화(strip().lower())도 여기서 처리.
"""
import os
import sys
from typing import Callable


def get_backend_pipeline_fn(
    backend: str = "torch",
    onnx_models_dir: str = "onnx_models",
) -> Callable:
    """
    백엔드 종류에 따라 pipeline_fn을 반환하는 공통 헬퍼.
    detect.py와 server.py가 공유한다.

    백엔드 문자열은 strip().lower()로 정규화하므로
    "ONNX", " Torch " 등 대소문자·공백 혼합 입력도 허용한다.

    Args:
        backend: "torch" 또는 "onnx" (대소문자·공백 무관)
        onnx_models_dir: ONNX 모델 디렉토리 (backend="onnx"일 때만 사용)

    Returns:
        pipeline_fn(task, model=model_id) → infer 콜러블

    Raises:
        ValueError: 알 수 없는 backend 값
    """
    normalized = backend.strip().lower()
    if normalized == "onnx":
        from onnx_detector import get_onnx_pipeline_fn
        return get_onnx_pipeline_fn(onnx_models_dir)
    if normalized == "torch":
        from detector import get_real_pipeline
        return get_real_pipeline()
    raise ValueError(f"알 수 없는 backend: {backend!r}. 'torch' 또는 'onnx'를 사용하세요.")


def get_pipeline_fn_with_mock(
    backend: str = "torch",
    onnx_models_dir: str = "onnx_models",
    pytest_context: bool = False,
) -> Callable:
    """
    _AI_DETECTOR_MOCK=1 이면 mock pipeline 반환, 그 외에는 backend에 따라 실제 pipeline 반환.
    pytest 컨텍스트 밖에서 mock이 활성화되면 stderr에 경고를 출력한다.

    Args:
        backend: "torch" 또는 "onnx"
        onnx_models_dir: ONNX 모델 디렉토리
        pytest_context: True면 mock 경고 억제 (테스트 환경)

    Returns:
        pipeline_fn callable
    """
    if os.environ.get("_AI_DETECTOR_MOCK") == "1":
        if not pytest_context and not os.environ.get("PYTEST_CURRENT_TEST"):
            sys.stderr.write(
                "WARNING: mock detector pipeline is active (_AI_DETECTOR_MOCK=1); "
                "results are not real.\n"
            )

        def mock_pipeline(*args, **kwargs):
            def infer(image):
                return [
                    {"label": "artificial", "score": 0.73},
                    {"label": "human", "score": 0.27},
                ]
            return infer
        return mock_pipeline

    return get_backend_pipeline_fn(backend, onnx_models_dir)
