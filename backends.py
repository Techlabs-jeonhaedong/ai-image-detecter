"""
백엔드 선택 공용 헬퍼.

detect.py와 server.py에서 공유하는 _get_backend_pipeline_fn 로직을 한 곳에서 관리.
백엔드 문자열 정규화(strip().lower())도 여기서 처리.
"""
import os
import sys
from typing import Callable, List, Optional

# 기본 ONNX 모델 디렉토리 — 이 파일 기준 절대경로 (어느 cwd에서 실행해도 동일)
DEFAULT_ONNX_MODELS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "onnx_models"
)


def resolve_ensemble_models_for_onnx(
    ensemble_models: List[str],
    explicit_models: List[str],
    *,
    backend: str,
    onnx_models_dir: str,
    is_mock: bool,
    warn,  # callable(str) | None — 제외 경고 메시지 수신 콜백
) -> List[str]:
    """
    세 진입점(detect.py, detector.detect(), server.py)이 공유하는 ensemble 필터 헬퍼.

    정책:
    - backend != "onnx" 또는 is_mock=True 이면 필터 없이 ensemble + explicit 병합만 한다.
    - backend == "onnx" + is_mock=False 이면:
        * ensemble_models 중 미번들(is_model_available=False)을 제외하고 warn 콜백 호출.
        * explicit_models 는 번들 여부와 무관하게 절대 자동 제외하지 않는다.
    - 반환값은 중복 제거·순서 유지된 최종 model_ids 리스트.

    Args:
        ensemble_models: --ensemble 에서 비롯된 모델 ID 목록 (ENSEMBLE_MODELS 서브셋)
        explicit_models: --model / models= 로 사용자가 명시한 모델 ID 목록
        backend: "onnx" 또는 "torch" (대소문자 정규화 없이 사용)
        onnx_models_dir: ONNX 모델 베이스 디렉토리
        is_mock: _AI_DETECTOR_MOCK=1 환경 여부
        warn: 제외 경고 메시지를 받는 콜백. None이면 무시.

    Returns:
        중복 제거·순서 유지된 최종 model_ids 리스트.
    """
    normalized_backend = backend.strip().lower()
    available_ensemble = list(ensemble_models)

    if normalized_backend == "onnx" and not is_mock and ensemble_models:
        from onnx_detector import is_model_available
        unavailable = [m for m in ensemble_models if not is_model_available(m, onnx_models_dir)]
        if unavailable and warn is not None:
            names = ", ".join(unavailable)
            warn(
                f"WARNING: 다음 ensemble 모델이 onnx 번들에 없음; "
                f"--backend torch 또는 setup.py로 변환 필요: {names}"
            )
        available_ensemble = [m for m in ensemble_models if m not in unavailable]

    # 중복 제거·순서 유지: ensemble 가용 목록 먼저, explicit 추가
    result: List[str] = list(available_ensemble)
    for m in explicit_models:
        if m not in result:
            result.append(m)
    return result


def get_backend_pipeline_fn(
    backend: str = "onnx",
    onnx_models_dir: str = DEFAULT_ONNX_MODELS_DIR,
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
    backend: str = "onnx",
    onnx_models_dir: str = DEFAULT_ONNX_MODELS_DIR,
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
