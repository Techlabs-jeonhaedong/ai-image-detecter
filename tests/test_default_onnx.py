"""
기본 백엔드가 onnx인지 검증하는 테스트 모음.
Goal #1 (onnx 기본 백엔드) + Goal #3 (bytes 입력) + Goal #4 (번들 모델) 통합 증명.
"""
import inspect
import io
import os
import struct
import sys
import zlib
from unittest.mock import patch

import pytest


# ──────────────────────────────────────────────────────────────────────────────
# 헬퍼
# ──────────────────────────────────────────────────────────────────────────────

def _has_onnxruntime() -> bool:
    try:
        import onnxruntime
        return True
    except ImportError:
        return False


def _make_minimal_png_bytes() -> bytes:
    """유효한 1x1 RGB PNG bytes를 생성한다."""
    def _chunk(name: bytes, data: bytes) -> bytes:
        length = struct.pack(">I", len(data))
        crc = struct.pack(">I", zlib.crc32(name + data) & 0xFFFFFFFF)
        return length + name + data + crc

    signature = b"\x89PNG\r\n\x1a\n"
    ihdr_data = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
    ihdr = _chunk(b"IHDR", ihdr_data)
    raw_row = b"\x00\xFF\x00\x00"  # filter=0, R=255, G=0, B=0
    idat_data = zlib.compress(raw_row)
    idat = _chunk(b"IDAT", idat_data)
    iend = _chunk(b"IEND", b"")
    return signature + ihdr + idat + iend


BUNDLED_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "onnx_models",
    "Organika__sdxl-detector",
    "model_quantized.onnx",
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ──────────────────────────────────────────────────────────────────────────────
# Goal #1: CLI/백엔드 기본값
# ──────────────────────────────────────────────────────────────────────────────

class TestDetectCliDefaultBackend:
    def test_detect_cli_default_backend_onnx(self):
        """--backend 미지정 시 기본값이 'onnx'여야 한다."""
        sys.path.insert(0, PROJECT_ROOT)
        from detect import _build_parser
        args = _build_parser().parse_args(["a.png"])
        assert args.backend == "onnx", f"expected 'onnx', got {args.backend!r}"

    def test_detect_cli_default_onnx_dir_absolute_and_exists(self):
        """기본 --onnx-models-dir이 절대경로이고 onnx_models로 끝나며 실제 존재해야 한다."""
        sys.path.insert(0, PROJECT_ROOT)
        from detect import _build_parser
        args = _build_parser().parse_args(["a.png"])
        d = args.onnx_models_dir
        assert os.path.isabs(d), f"onnx_models_dir should be absolute, got {d!r}"
        assert d.endswith("onnx_models"), f"should end with 'onnx_models', got {d!r}"
        assert os.path.isdir(d), f"directory should exist: {d!r}"


class TestDetectorDetectSignature:
    def test_detector_detect_signature_default_onnx(self):
        """detector.detect()의 backend 기본값이 'onnx'여야 한다."""
        sys.path.insert(0, PROJECT_ROOT)
        import detector
        sig = inspect.signature(detector.detect)
        default = sig.parameters["backend"].default
        assert default == "onnx", f"expected 'onnx', got {default!r}"


class TestServerDefaultOnnx:
    def test_server_default_onnx(self, monkeypatch):
        """DETECTOR_BACKEND 미설정 시 server._make_pipeline()이 'onnx'로 get_backend_pipeline_fn을 호출해야 한다."""
        monkeypatch.delenv("DETECTOR_BACKEND", raising=False)
        monkeypatch.delenv("_AI_DETECTOR_MOCK", raising=False)

        sys.path.insert(0, PROJECT_ROOT)
        with patch("backends.get_backend_pipeline_fn") as mock_fn:
            mock_fn.return_value = lambda *a, **kw: lambda img: [
                {"label": "artificial", "score": 0.8},
                {"label": "human", "score": 0.2},
            ]
            import server
            server._make_pipeline()
            mock_fn.assert_called_once()
            called_backend = mock_fn.call_args[0][0]
            assert called_backend == "onnx", f"expected 'onnx', got {called_backend!r}"


# ──────────────────────────────────────────────────────────────────────────────
# Goal #4: 번들 모델 존재
# ──────────────────────────────────────────────────────────────────────────────

class TestBundledModelExists:
    def test_bundled_model_exists(self):
        """번들된 model_quantized.onnx, meta.json, config.json이 모두 존재해야 한다."""
        model_dir = os.path.dirname(BUNDLED_MODEL_PATH)
        assert os.path.isfile(BUNDLED_MODEL_PATH), (
            f"번들 모델 없음: {BUNDLED_MODEL_PATH}"
        )
        meta_path = os.path.join(model_dir, "meta.json")
        assert os.path.isfile(meta_path), f"meta.json 없음: {meta_path}"
        config_path = os.path.join(model_dir, "config.json")
        assert os.path.isfile(config_path), f"config.json 없음: {config_path}"


# ──────────────────────────────────────────────────────────────────────────────
# Goal #1 + #3 + #4 통합 E2E
# ──────────────────────────────────────────────────────────────────────────────

@pytest.mark.skipif(not _has_onnxruntime(), reason="onnxruntime 미설치")
class TestCloneAndRunRealOnnxInference:
    def test_clone_and_run_real_onnx_inference(self):
        """
        mock 없이 번들 onnx 모델로 실제 추론 수행.
        Goal #1(onnx 기본), #3(bytes 입력), #4(번들 모델) 통합 증명.
        """
        sys.path.insert(0, PROJECT_ROOT)
        from onnx_detector import _clear_session_cache
        _clear_session_cache()

        import detector
        png_bytes = _make_minimal_png_bytes()
        r = detector.detect(
            png_bytes,
            name="x.png",
            backend="onnx",
            with_metadata=False,
        )
        assert r["error"] is None, f"추론 에러 발생: {r['error']}"
        assert r["verdict"] in ("AI-generated", "Real"), (
            f"unexpected verdict: {r['verdict']!r}"
        )
        assert isinstance(r["ai_probability"], float), (
            f"ai_probability가 float이 아님: {type(r['ai_probability'])}"
        )

    @pytest.mark.skipif(not _has_onnxruntime(), reason="onnxruntime 미설치")
    def test_detect_cli_real_run_bundled(self, tmp_path):
        """번들 모델로 CLI main 실행 시 exit 0 반환."""
        sys.path.insert(0, PROJECT_ROOT)
        from onnx_detector import _clear_session_cache
        _clear_session_cache()

        # tmp에 유효 PNG 저장
        png_path = tmp_path / "test.png"
        png_path.write_bytes(_make_minimal_png_bytes())

        from detect import main
        with pytest.raises(SystemExit) as exc_info:
            main([str(png_path)])
        assert exc_info.value.code == 0, (
            f"CLI가 exit 0을 반환해야 하는데 {exc_info.value.code} 반환"
        )
