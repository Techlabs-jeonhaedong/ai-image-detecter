"""
앙상블 + ONNX 백엔드 회귀 테스트.

검증 항목:
  1. is_model_available — 번들 존재/미존재/안전하지 않은 id
  2. CLI --ensemble + onnx → exit 0, 미번들 모델 stderr 경고
  3. detect() API ensemble=True + onnx → error None, 정상 verdict
  4. 명시적 미번들 모델 → exit 1, actionable 에러 메시지
  5. mock 환경에서 ensemble → 모든 3개 모델 포함 (필터링 금지)
"""
import json
import os
import struct
import sys
import zlib
import pytest
from unittest.mock import patch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


# ──────────────────────────────────────────────────────────────────────────────
# 헬퍼
# ──────────────────────────────────────────────────────────────────────────────

def _has_onnxruntime() -> bool:
    try:
        import onnxruntime  # noqa: F401
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
    raw_row = b"\x00\xFF\x00\x00"
    idat_data = zlib.compress(raw_row)
    idat = _chunk(b"IDAT", idat_data)
    iend = _chunk(b"IEND", b"")
    return signature + ihdr + idat + iend


BUNDLED_ONNX_DIR = os.path.join(PROJECT_ROOT, "onnx_models")


# ──────────────────────────────────────────────────────────────────────────────
# 1. is_model_available
# ──────────────────────────────────────────────────────────────────────────────

class TestIsModelAvailable:
    """onnx_detector.is_model_available 단위 테스트."""

    def test_bundled_model_returns_true(self):
        """번들된 Organika/sdxl-detector는 True를 반환해야 한다."""
        from onnx_detector import is_model_available
        assert is_model_available("Organika/sdxl-detector", BUNDLED_ONNX_DIR) is True

    def test_unbundled_model_returns_false(self):
        """번들 안 된 모델은 False를 반환해야 한다."""
        from onnx_detector import is_model_available
        assert is_model_available("yaya36095/ai-image-detector", BUNDLED_ONNX_DIR) is False

    def test_another_unbundled_model_returns_false(self):
        """다른 미번들 모델도 False를 반환해야 한다."""
        from onnx_detector import is_model_available
        assert is_model_available("umm-maybe/AI-image-detector", BUNDLED_ONNX_DIR) is False

    def test_path_traversal_returns_false_not_exception(self):
        """../x 같은 안전하지 않은 id는 예외 없이 False를 반환해야 한다."""
        from onnx_detector import is_model_available
        assert is_model_available("../x", BUNDLED_ONNX_DIR) is False

    def test_nonexistent_model_dir_returns_false(self, tmp_path):
        """onnx_models_dir 자체가 없으면 False를 반환해야 한다."""
        from onnx_detector import is_model_available
        empty_dir = str(tmp_path / "no_models")
        assert is_model_available("Organika/sdxl-detector", empty_dir) is False

    def test_model_dir_exists_but_no_onnx_file_returns_false(self, tmp_path):
        """모델 디렉토리는 있지만 .onnx 파일 없으면 False를 반환해야 한다."""
        from onnx_detector import is_model_available
        model_dir = tmp_path / "Organika__sdxl-detector"
        model_dir.mkdir()
        assert is_model_available("Organika/sdxl-detector", str(tmp_path)) is False

    def test_absolute_path_id_returns_false(self):
        """절대경로 형태의 id는 False를 반환해야 한다."""
        from onnx_detector import is_model_available
        assert is_model_available("/etc/passwd", BUNDLED_ONNX_DIR) is False


# ──────────────────────────────────────────────────────────────────────────────
# 2. CLI --ensemble + onnx → exit 0
# ──────────────────────────────────────────────────────────────────────────────

@pytest.mark.skipif(not _has_onnxruntime(), reason="onnxruntime 미설치")
class TestCliEnsembleOnnxExitZero:
    """--ensemble + onnx 백엔드 시 exit 0 및 stderr 경고 검증."""

    def test_cli_ensemble_onnx_exit_zero(self, tmp_path, capsys):
        """
        onnx 기본 백엔드 + --ensemble: 번들 안 된 모델은 자동 제외하고 exit 0.
        """
        from onnx_detector import _clear_session_cache
        _clear_session_cache()

        png_path = tmp_path / "test.png"
        png_path.write_bytes(_make_minimal_png_bytes())

        from detect import main
        with pytest.raises(SystemExit) as exc_info:
            main([str(png_path), "--ensemble"])
        assert exc_info.value.code == 0, (
            f"--ensemble + onnx는 exit 0이어야 하는데 {exc_info.value.code} 반환"
        )

    def test_cli_ensemble_onnx_stderr_warning_unbundled(self, tmp_path, capsys):
        """
        onnx 기본 백엔드 + --ensemble: 제외된 미번들 모델 경고가 stderr에 출력된다.
        """
        from onnx_detector import _clear_session_cache
        _clear_session_cache()

        png_path = tmp_path / "test.png"
        png_path.write_bytes(_make_minimal_png_bytes())

        from detect import main
        with pytest.raises(SystemExit):
            main([str(png_path), "--ensemble"])

        captured = capsys.readouterr()
        # 미번들 모델 중 하나 이상이 stderr에 언급되어야 한다
        assert (
            "yaya36095" in captured.err
            or "umm-maybe" in captured.err
        ), f"미번들 모델 경고가 stderr에 없음. stderr={captured.err!r}"

    def test_cli_ensemble_onnx_result_has_verdict(self, tmp_path, capsys):
        """
        onnx 기본 백엔드 + --ensemble + --json: verdict가 유효한 값이어야 한다.
        """
        from onnx_detector import _clear_session_cache
        _clear_session_cache()

        png_path = tmp_path / "test.png"
        png_path.write_bytes(_make_minimal_png_bytes())

        from detect import main
        with pytest.raises(SystemExit) as exc_info:
            main([str(png_path), "--ensemble", "--json"])
        assert exc_info.value.code == 0

        captured = capsys.readouterr()
        parsed = json.loads(captured.out)
        assert parsed[0]["verdict"] in ("AI-generated", "Real"), (
            f"unexpected verdict: {parsed[0]['verdict']!r}"
        )


# ──────────────────────────────────────────────────────────────────────────────
# 3. detect() API ensemble=True + onnx
# ──────────────────────────────────────────────────────────────────────────────

@pytest.mark.skipif(not _has_onnxruntime(), reason="onnxruntime 미설치")
class TestDetectApiEnsembleOnnx:
    """detector.detect(ensemble=True, backend='onnx') 단위 테스트."""

    def test_detect_api_ensemble_onnx_uses_bundled(self):
        """
        detect(bytes, ensemble=True, backend='onnx') →
          error is None, verdict 유효, Organika 모델 성공.
        """
        from onnx_detector import _clear_session_cache
        _clear_session_cache()

        import detector
        png_bytes = _make_minimal_png_bytes()
        result = detector.detect(
            png_bytes,
            name="x.png",
            backend="onnx",
            ensemble=True,
            with_metadata=False,
        )
        assert result["error"] is None, (
            f"ensemble onnx 추론에서 에러 발생: {result['error']}"
        )
        assert result["verdict"] in ("AI-generated", "Real"), (
            f"unexpected verdict: {result['verdict']!r}"
        )
        # Organika 모델은 반드시 성공해야 함
        organika_results = [
            m for m in result["models"]
            if "Organika" in m["model"] or "organika" in m["model"].lower()
        ]
        assert organika_results, "Organika 모델 결과가 없음"
        assert organika_results[0]["error"] is None, (
            f"Organika 모델 에러: {organika_results[0]['error']}"
        )

    def test_detect_api_ensemble_onnx_no_full_failure(self):
        """
        미번들 모델이 있어도 ensemble onnx는 전체 실패 없이 결과를 반환한다.
        """
        from onnx_detector import _clear_session_cache
        _clear_session_cache()

        import detector
        png_bytes = _make_minimal_png_bytes()
        result = detector.detect(
            png_bytes,
            name="x.png",
            backend="onnx",
            ensemble=True,
            with_metadata=False,
        )
        # 전체 에러(root level)는 None이어야 함
        assert result["error"] is None


# ──────────────────────────────────────────────────────────────────────────────
# 4. 명시적 미번들 모델 → exit 1 + actionable 에러
# ──────────────────────────────────────────────────────────────────────────────

@pytest.mark.skipif(not _has_onnxruntime(), reason="onnxruntime 미설치")
class TestExplicitUnbundledModelErrors:
    """명시적 --model 지정 시 미번들이면 기존처럼 에러 처리."""

    def test_explicit_unbundled_model_onnx_errors(self, tmp_path, capsys):
        """
        --model nope/x + 빈 onnx-models-dir → exit 1.
        에러 메시지에 'torch' 또는 'setup.py' 안내 포함.
        """
        from onnx_detector import _clear_session_cache
        _clear_session_cache()

        # 빈 onnx-models-dir
        empty_dir = tmp_path / "empty_onnx"
        empty_dir.mkdir()

        png_path = tmp_path / "test.png"
        png_path.write_bytes(_make_minimal_png_bytes())

        from detect import main
        with pytest.raises(SystemExit) as exc_info:
            main([
                str(png_path),
                "--model", "nope/x",
                "--onnx-models-dir", str(empty_dir),
            ])
        assert exc_info.value.code == 1, (
            f"명시적 미번들 모델은 exit 1이어야 하는데 {exc_info.value.code} 반환"
        )

        captured = capsys.readouterr()
        combined = captured.out + captured.err
        assert "torch" in combined or "setup.py" in combined, (
            f"actionable 에러 메시지에 'torch' 또는 'setup.py' 안내가 없음. output={combined!r}"
        )

    def test_explicit_unbundled_model_not_silently_skipped(self, tmp_path, capsys):
        """
        명시적 --model이 미번들이면 자동 제외가 아니라 에러 처리 (사용자 의도 보존).
        """
        from onnx_detector import _clear_session_cache
        _clear_session_cache()

        empty_dir = tmp_path / "empty_onnx2"
        empty_dir.mkdir()

        png_path = tmp_path / "test.png"
        png_path.write_bytes(_make_minimal_png_bytes())

        from detect import main
        with pytest.raises(SystemExit) as exc_info:
            main([
                str(png_path),
                "--model", "unbundled/model",
                "--onnx-models-dir", str(empty_dir),
            ])
        # 자동 제외가 아니라 에러 (exit 1)
        assert exc_info.value.code == 1


# ──────────────────────────────────────────────────────────────────────────────
# 5. mock 환경에서는 필터링 금지
# ──────────────────────────────────────────────────────────────────────────────

class TestMockEnsembleStillAllModels:
    """_AI_DETECTOR_MOCK=1 환경에서는 ensemble 필터링을 하면 안 된다."""

    def test_mock_ensemble_still_all_models(self, tmp_path, capsys, monkeypatch):
        """
        mock 환경 + --ensemble → exit 0, JSON models 길이 == 3 (필터링 금지).
        """
        monkeypatch.setenv("_AI_DETECTOR_MOCK", "1")

        png_path = tmp_path / "test.png"
        png_path.write_bytes(_make_minimal_png_bytes())

        from detect import main
        with pytest.raises(SystemExit) as exc_info:
            main([str(png_path), "--ensemble", "--json"])
        assert exc_info.value.code == 0

        captured = capsys.readouterr()
        parsed = json.loads(captured.out)
        models_count = len(parsed[0]["models"])
        assert models_count == 3, (
            f"mock 환경에서는 ensemble 모델이 3개여야 하는데 {models_count}개"
        )

    def test_mock_ensemble_no_stderr_warning(self, tmp_path, capsys, monkeypatch):
        """
        mock 환경에서는 미번들 모델 경고를 출력하면 안 된다.
        """
        monkeypatch.setenv("_AI_DETECTOR_MOCK", "1")

        png_path = tmp_path / "test.png"
        png_path.write_bytes(_make_minimal_png_bytes())

        from detect import main
        with pytest.raises(SystemExit):
            main([str(png_path), "--ensemble"])

        captured = capsys.readouterr()
        # mock 환경에서는 "onnx 번들에 없음" 경고 없어야 함
        assert "onnx 번들에 없음" not in captured.err


# ──────────────────────────────────────────────────────────────────────────────
# 6. 에러 메시지 actionable 검증
# ──────────────────────────────────────────────────────────────────────────────

class TestActionableErrorMessage:
    """onnx_detector.get_onnx_pipeline_fn의 에러 메시지 actionable 검증."""

    def test_file_not_found_error_has_torch_hint(self, tmp_path):
        """
        미변환 모델 호출 시 FileNotFoundError 메시지에 'torch' 또는 'setup.py' 안내 포함.
        """
        from onnx_detector import get_onnx_pipeline_fn

        empty_dir = tmp_path / "empty_onnx"
        empty_dir.mkdir()

        pipeline_fn = get_onnx_pipeline_fn(str(empty_dir))
        with pytest.raises(FileNotFoundError) as exc_info:
            pipeline_fn("image-classification", model="missing/model")

        msg = str(exc_info.value)
        assert "torch" in msg or "setup.py" in msg, (
            f"에러 메시지에 actionable 안내가 없음: {msg!r}"
        )
