"""
AI 이미지 탐지 HTTP 마이크로서비스.

FastAPI 기반. PHP 등 외부 서버에서 curl로 이미지를 업로드하면
AI 생성 여부를 JSON으로 반환한다.

핵심 설계:
- 모델 pipeline은 _PIPELINE_CACHE에 LRU로 최대 MAX_CACHED_PIPELINES개 캐시
- 캐시는 threading.Lock으로 동시성 보호 (double-checked locking)
- 추론 로직은 detector.py / metadata.py를 그대로 재사용 (중복 구현 없음)
- _AI_DETECTOR_MOCK=1 환경변수로 테스트용 mock pipeline 주입 가능
- ALLOWED_MODELS 환경변수(콤마 구분) 설정 시 해당 모델만 허용

실행:
    uvicorn server:app --host 127.0.0.1 --port 8000
    or
    python server.py
"""

import logging
import os
import sys
import threading
from collections import OrderedDict
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# 설정 상수
# ──────────────────────────────────────────────────────────────────────────────

MAX_UPLOAD_BYTES: int = int(os.environ.get("MAX_UPLOAD_BYTES", str(20 * 1024 * 1024)))  # 20MB
MAX_CACHED_PIPELINES: int = int(os.environ.get("MAX_CACHED_PIPELINES", "4"))
HOST: str = os.environ.get("HOST", "127.0.0.1")
PORT: int = int(os.environ.get("PORT", "8000"))

# ──────────────────────────────────────────────────────────────────────────────
# 전역 pipeline 캐시 (LRU OrderedDict + Lock)
# ──────────────────────────────────────────────────────────────────────────────

_PIPELINE_CACHE: OrderedDict = OrderedDict()
_PIPELINE_CACHE_LOCK = threading.Lock()


def _make_pipeline():
    """
    환경변수 _AI_DETECTOR_MOCK=1이면 mock pipeline 팩토리 반환.
    그 외에는 DETECTOR_BACKEND 환경변수(기본 onnx)에 따라 backend 선택.
    백엔드 선택 로직은 backends.get_pipeline_fn_with_mock으로 위임.
    """
    from backends import get_pipeline_fn_with_mock, DEFAULT_ONNX_MODELS_DIR
    backend = os.environ.get("DETECTOR_BACKEND", "onnx")
    onnx_models_dir = os.environ.get("ONNX_MODELS_DIR", DEFAULT_ONNX_MODELS_DIR)
    return get_pipeline_fn_with_mock(backend, onnx_models_dir)


def cached_pipeline_fn(task: str, model: str = "", **kwargs) -> Any:
    """
    캐시에 있으면 기존 pipe 반환, 없으면 생성 후 캐시에 저장.
    LRU 방식으로 MAX_CACHED_PIPELINES 초과 시 가장 오래된 항목 evict.
    double-checked locking으로 동시성 보호.
    """
    cache_key = model
    # 1차 체크 (락 없이 빠른 경로)
    if cache_key in _PIPELINE_CACHE:
        _PIPELINE_CACHE.move_to_end(cache_key)
        return _PIPELINE_CACHE[cache_key]

    with _PIPELINE_CACHE_LOCK:
        # 2차 체크 (락 안에서 재확인)
        if cache_key in _PIPELINE_CACHE:
            _PIPELINE_CACHE.move_to_end(cache_key)
            return _PIPELINE_CACHE[cache_key]

        factory = _make_pipeline()
        pipe = factory(task, model=model, **kwargs)
        _PIPELINE_CACHE[cache_key] = pipe
        _PIPELINE_CACHE.move_to_end(cache_key)

        # LRU eviction: 한도 초과 시 가장 오래된 항목 제거
        while len(_PIPELINE_CACHE) > MAX_CACHED_PIPELINES:
            _PIPELINE_CACHE.popitem(last=False)

        return pipe


# ──────────────────────────────────────────────────────────────────────────────
# FastAPI 앱
# ──────────────────────────────────────────────────────────────────────────────

app = FastAPI(title="AI Image Detector", version="1.0.0")


# ── Content-Length 사전 검사 미들웨어 ─────────────────────────────────────────

class UploadSizeLimitMiddleware(BaseHTTPMiddleware):
    """
    요청의 Content-Length 헤더가 MAX_UPLOAD_BYTES 초과이면 본문을 읽기 전에 413으로 거부.
    Content-Length가 없는 경우에는 패스스루하고 핸들러 안의 실제 크기 재검사가 처리.
    """

    async def dispatch(self, request: Request, call_next):
        content_length = request.headers.get("content-length")
        if content_length is not None:
            try:
                cl = int(content_length)
            except (ValueError, TypeError):
                cl = 0
            if cl > MAX_UPLOAD_BYTES:
                return JSONResponse(
                    status_code=413,
                    content={
                        "detail": (
                            f"Upload too large: {cl} bytes "
                            f"(limit: {MAX_UPLOAD_BYTES} bytes)"
                        )
                    },
                )
        return await call_next(request)


app.add_middleware(UploadSizeLimitMiddleware)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """예외가 서버를 죽이지 않도록 전역 처리. HTTPException은 그대로 재발생."""
    if isinstance(exc, HTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.detail},
        )
    # 내부 예외는 서버 로그에만 기록하고, 클라이언트에는 일반화된 메시지만 반환
    logger.error("Unhandled exception: %s: %s", type(exc).__name__, exc, exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )


@app.get("/health")
def health():
    """헬스체크 엔드포인트."""
    return {"status": "ok"}


@app.post("/detect")
def detect_endpoint(
    file: UploadFile = File(...),
    threshold: Optional[str] = Form(default=None),
    ensemble: bool = Form(default=False),
    model: Optional[List[str]] = Form(default=None),
    no_metadata: bool = Form(default=False),
):
    """
    이미지를 업로드받아 AI 생성 여부를 판정한다.

    Args:
        file: 업로드할 이미지 파일 (multipart/form-data)
        threshold: AI 판정 임계값 (0~1, 기본 0.5)
        ensemble: True면 ENSEMBLE_MODELS 전체 사용
        model: 모델 ID 목록 (반복 가능)
        no_metadata: True면 메타데이터 검사 건너뜀

    Returns:
        단일 이미지 분석 결과 dict (JSON)
    """
    from detector import (
        DEFAULT_MODEL,
        ENSEMBLE_MODELS,
        _apply_metadata_override,
        analyze_images_batch,
    )
    from metadata import inspect_metadata

    # ── threshold 파싱 및 검증 ─────────────────────────────────────────────
    parsed_threshold: float = 0.5
    if threshold is not None:
        try:
            parsed_threshold = float(threshold)
        except (ValueError, TypeError):
            raise HTTPException(status_code=400, detail="threshold must be a number between 0 and 1")
        if not (0.0 <= parsed_threshold <= 1.0):
            raise HTTPException(status_code=400, detail=f"threshold must be in [0, 1], got {parsed_threshold}")

    # ── 파일 크기 제한 검사 (상한 읽기로 메모리 DoS 방어) ─────────────────
    # Content-Length 없는 chunked 요청이 미들웨어를 우회해도 여기서 방어.
    # MAX_UPLOAD_BYTES + 1 바이트만 읽어 상한 초과 여부를 확인한다.
    # 상한 이상은 메모리에 적재되지 않는다 (이중 방어: 미들웨어 + 여기).
    content = file.file.read(MAX_UPLOAD_BYTES + 1)
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Upload too large (limit: {MAX_UPLOAD_BYTES} bytes)",
        )

    # ── 빈 파일 검사 ─────────────────────────────────────────────────────
    if len(content) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty (0 bytes)")

    # ── 모델 목록 결정 ───────────────────────────────────────────────────
    # 빈 문자열/공백 필터링 및 중복 제거
    model_ids: List[str] = []
    if ensemble:
        model_ids.extend(ENSEMBLE_MODELS)
    if model:
        for m in model:
            stripped = m.strip()
            if stripped and stripped not in model_ids:
                model_ids.append(stripped)
    if not model_ids:
        model_ids = [DEFAULT_MODEL]

    # ── ALLOWED_MODELS 화이트리스트 검사 ─────────────────────────────────
    allowed_env = os.environ.get("ALLOWED_MODELS", "").strip()
    if allowed_env:
        allowed_set = {m.strip() for m in allowed_env.split(",") if m.strip()}
        blocked = [m for m in model_ids if m not in allowed_set]
        if blocked:
            raise HTTPException(
                status_code=400,
                detail=f"Model(s) not in ALLOWED_MODELS: {blocked}",
            )

    # ── 원본 파일명 추출 및 정리 ─────────────────────────────────────────
    # os.path.basename으로 경로 구분자 및 '..' 제거
    raw_filename = file.filename or "upload.bin"
    filename = os.path.basename(raw_filename) or "upload.bin"

    # ── bytes를 직접 분석 (임시파일 없이) ────────────────────────────────
    results = analyze_images_batch(
        image_paths=[content],
        pipeline_fn=cached_pipeline_fn,
        model_ids=model_ids,
        threshold=parsed_threshold,
        names=[filename],
    )
    result = results[0]

    # 단일 모델이면 model 필드를 ID 자체로 표시 (detect.py와 동일 정책)
    if len(model_ids) == 1:
        result["model"] = model_ids[0]

    # ── 메타데이터 검사 ───────────────────────────────────────────────────
    if not no_metadata:
        meta = inspect_metadata(content)
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

    # image 필드는 analyze_images_batch에서 names로 지정되므로 그대로 사용
    # (이미 filename으로 설정됨)

    return result


# ──────────────────────────────────────────────────────────────────────────────
# 직접 실행 진입점
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "server:app",
        host=HOST,
        port=PORT,
        reload=False,
    )
