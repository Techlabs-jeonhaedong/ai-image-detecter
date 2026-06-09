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

    @pytest.fixture(autouse=True)
    def clear_mock_env(self, monkeypatch):
        """test_server.py가 모듈 수준에서 설정한 _AI_DETECTOR_MOCK을 이 테스트에서 해제."""
        monkeypatch.delenv("_AI_DETECTOR_MOCK", raising=False)

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

    @pytest.fixture(autouse=True)
    def clear_mock_env(self, monkeypatch):
        """test_server.py가 모듈 수준에서 설정한 _AI_DETECTOR_MOCK을 이 테스트에서 해제."""
        monkeypatch.delenv("_AI_DETECTOR_MOCK", raising=False)

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

    @pytest.fixture(autouse=True)
    def clear_mock_env(self, monkeypatch):
        """test_server.py가 모듈 수준에서 설정한 _AI_DETECTOR_MOCK을 이 테스트에서 해제."""
        monkeypatch.delenv("_AI_DETECTOR_MOCK", raising=False)

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


# ──────────────────────────────────────────────────────────────────────────────
# 7. 수정 1 — 명시 모델은 ensemble 필터에서 절대 제외 금지
# ──────────────────────────────────────────────────────────────────────────────

@pytest.mark.skipif(not _has_onnxruntime(), reason="onnxruntime 미설치")
class TestExplicitEnsembleMemberNotSilentlyDropped:
    """--ensemble --model <ENSEMBLE_MODELS 멤버>일 때 명시 모델이 조용히 사라지면 안 된다."""

    @pytest.fixture(autouse=True)
    def clear_mock_env(self, monkeypatch):
        monkeypatch.delenv("_AI_DETECTOR_MOCK", raising=False)

    def test_explicit_ensemble_member_not_silently_dropped(self, tmp_path, capsys):
        """
        빈 onnx-models-dir + --ensemble --model yaya36095/ai-image-detector →
        yaya가 models 결과에 등장해야 하고 (error로든), exit code는 1 (명시 미번들).
        """
        from onnx_detector import _clear_session_cache
        _clear_session_cache()

        empty_dir = tmp_path / "empty_onnx"
        empty_dir.mkdir()

        png_path = tmp_path / "test.png"
        png_path.write_bytes(_make_minimal_png_bytes())

        from detect import main
        with pytest.raises(SystemExit) as exc_info:
            main([
                str(png_path),
                "--ensemble",
                "--model", "yaya36095/ai-image-detector",
                "--onnx-models-dir", str(empty_dir),
            ])
        # 명시 모델이 미번들이므로 exit 1
        assert exc_info.value.code == 1, (
            f"명시 모델이 미번들이면 exit 1이어야 하는데 {exc_info.value.code}"
        )

        captured = capsys.readouterr()
        combined = captured.out + captured.err
        # yaya가 결과(JSON) 또는 에러 메시지에 언급되어야 함 — 조용히 사라지면 안 됨
        assert "yaya36095" in combined, (
            f"명시한 yaya36095 모델이 출력에서 사라짐 (조용히 제외됨). output={combined!r}"
        )


# ──────────────────────────────────────────────────────────────────────────────
# 8. 수정 1 — 번들된 ENSEMBLE_MODELS 멤버를 명시하면 중복 없이 정상 동작
# ──────────────────────────────────────────────────────────────────────────────

@pytest.mark.skipif(not _has_onnxruntime(), reason="onnxruntime 미설치")
class TestExplicitBundledMemberWithEnsemble:
    """번들된 Organika를 명시할 때 중복 없이 정상 verdict 반환."""

    @pytest.fixture(autouse=True)
    def clear_mock_env(self, monkeypatch):
        monkeypatch.delenv("_AI_DETECTOR_MOCK", raising=False)

    def test_explicit_bundled_member_with_ensemble(self):
        """
        detect(png, ensemble=True, models=["Organika/sdxl-detector"], backend="onnx") →
        Organika 중복 없이 정상 verdict, error None.
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
            models=["Organika/sdxl-detector"],
            with_metadata=False,
        )
        assert result["error"] is None, f"에러 발생: {result['error']}"
        assert result["verdict"] in ("AI-generated", "Real"), (
            f"unexpected verdict: {result['verdict']!r}"
        )
        # Organika가 models에 정확히 1번만 등장해야 함 (중복 없음)
        organika_entries = [
            m for m in result["models"]
            if "Organika" in m["model"] or "organika" in m["model"].lower()
        ]
        assert len(organika_entries) == 1, (
            f"Organika가 중복 포함됨: {[m['model'] for m in result['models']]}"
        )
        assert organika_entries[0]["error"] is None, (
            f"Organika 에러: {organika_entries[0]['error']}"
        )


# ──────────────────────────────────────────────────────────────────────────────
# 9. 수정 2 — 공유 헬퍼: torch/mock일 때 필터 없음
# ──────────────────────────────────────────────────────────────────────────────

class TestSharedHelperTorchAndMockNoFilter:
    """resolve_ensemble_models_for_onnx: backend torch 또는 is_mock이면 필터 없이 전부 보존."""

    def test_mock_no_filter(self, tmp_path):
        """is_mock=True이면 ensemble+explicit 모두 보존."""
        from backends import resolve_ensemble_models_for_onnx
        ensemble = ["Organika/sdxl-detector", "yaya36095/ai-image-detector"]
        explicit = ["umm-maybe/AI-image-detector"]
        result = resolve_ensemble_models_for_onnx(
            ensemble_models=ensemble,
            explicit_models=explicit,
            backend="onnx",
            onnx_models_dir=str(tmp_path),
            is_mock=True,
            warn=None,
        )
        # 필터 없이 중복 제거된 전체 목록 반환
        assert set(result) == set(ensemble + explicit), (
            f"mock 환경에서 필터가 적용됨: {result}"
        )

    def test_torch_no_filter(self, tmp_path):
        """backend='torch'이면 ensemble+explicit 모두 보존."""
        from backends import resolve_ensemble_models_for_onnx
        ensemble = ["Organika/sdxl-detector", "yaya36095/ai-image-detector"]
        explicit = ["custom/model"]
        result = resolve_ensemble_models_for_onnx(
            ensemble_models=ensemble,
            explicit_models=explicit,
            backend="torch",
            onnx_models_dir=str(tmp_path),
            is_mock=False,
            warn=None,
        )
        assert set(result) == set(ensemble + explicit), (
            f"torch 환경에서 필터가 적용됨: {result}"
        )

    def test_onnx_non_mock_filters_unavailable_ensemble(self, tmp_path):
        """backend=onnx + is_mock=False이면 미번들 ensemble 모델은 필터됨."""
        from backends import resolve_ensemble_models_for_onnx
        ensemble = ["yaya36095/ai-image-detector", "umm-maybe/AI-image-detector"]
        explicit: list = []
        warnings_received = []
        result = resolve_ensemble_models_for_onnx(
            ensemble_models=ensemble,
            explicit_models=explicit,
            backend="onnx",
            onnx_models_dir=str(tmp_path),  # 빈 dir → 전부 미번들
            is_mock=False,
            warn=warnings_received.append,
        )
        # 미번들 ensemble 모두 필터됨
        assert result == [], f"미번들 ensemble이 필터 안 됨: {result}"
        # 경고가 발생해야 함
        assert warnings_received, "필터 시 warn 콜백이 호출되지 않음"

    def test_onnx_non_mock_explicit_not_filtered(self, tmp_path):
        """backend=onnx + is_mock=False이더라도 explicit 모델은 필터하지 않음."""
        from backends import resolve_ensemble_models_for_onnx
        explicit = ["yaya36095/ai-image-detector"]
        result = resolve_ensemble_models_for_onnx(
            ensemble_models=[],
            explicit_models=explicit,
            backend="onnx",
            onnx_models_dir=str(tmp_path),  # 빈 dir → 미번들
            is_mock=False,
            warn=None,
        )
        # explicit은 필터 안 됨
        assert result == explicit, f"explicit 모델이 필터됨: {result}"


# ──────────────────────────────────────────────────────────────────────────────
# 10. 수정 3 — is_model_available: meta.json 없으면 False
# ──────────────────────────────────────────────────────────────────────────────

class TestIsModelAvailableRequiresMeta:
    """is_model_available은 .onnx + meta.json 둘 다 있어야 True."""

    def test_onnx_only_no_meta_returns_false(self, tmp_path):
        """.onnx 파일만 있고 meta.json 없으면 False."""
        from onnx_detector import is_model_available
        model_dir = tmp_path / "Organika__sdxl-detector"
        model_dir.mkdir()
        (model_dir / "model_quantized.onnx").write_bytes(b"fake onnx")
        # meta.json 없음
        assert is_model_available("Organika/sdxl-detector", str(tmp_path)) is False

    def test_meta_only_no_onnx_returns_false(self, tmp_path):
        """meta.json만 있고 .onnx 파일 없으면 False."""
        from onnx_detector import is_model_available
        model_dir = tmp_path / "Organika__sdxl-detector"
        model_dir.mkdir()
        (model_dir / "meta.json").write_text("{}")
        # .onnx 없음
        assert is_model_available("Organika/sdxl-detector", str(tmp_path)) is False

    def test_both_onnx_and_meta_returns_true(self, tmp_path):
        """.onnx와 meta.json 둘 다 있으면 True."""
        from onnx_detector import is_model_available
        model_dir = tmp_path / "Organika__sdxl-detector"
        model_dir.mkdir()
        (model_dir / "model_quantized.onnx").write_bytes(b"fake onnx")
        (model_dir / "meta.json").write_text("{}")
        assert is_model_available("Organika/sdxl-detector", str(tmp_path)) is True


# ──────────────────────────────────────────────────────────────────────────────
# 11. 수정 2 — server.py ensemble onnx 필터 검증
# ──────────────────────────────────────────────────────────────────────────────

@pytest.mark.skipif(not _has_onnxruntime(), reason="onnxruntime 미설치")
class TestServerEnsembleOnnxFilters:
    """server.py /detect?ensemble=true + onnx → 미번들 yaya/umm error 없어야 함."""

    def test_server_ensemble_onnx_filters_unbundled(self, tmp_path, monkeypatch):
        """
        DETECTOR_BACKEND=onnx + 번들 ONNX_MODELS_DIR + ensemble=true →
        응답 models에 yaya/umm의 미번들 에러가 없고 verdict 정상.
        """
        from onnx_detector import _clear_session_cache
        _clear_session_cache()

        # mock 해제 후 실제 onnx 환경
        monkeypatch.delenv("_AI_DETECTOR_MOCK", raising=False)
        monkeypatch.setenv("DETECTOR_BACKEND", "onnx")
        monkeypatch.setenv("ONNX_MODELS_DIR", BUNDLED_ONNX_DIR)

        # server 모듈 리로드 (환경변수 굳음 이슈 방지)
        import importlib
        import server as srv
        srv._PIPELINE_CACHE.clear()

        from fastapi.testclient import TestClient
        client = TestClient(srv.app)

        png_bytes = _make_minimal_png_bytes()
        resp = client.post(
            "/detect",
            data={"ensemble": "true", "no_metadata": "true"},
            files={"file": ("test.png", png_bytes, "image/png")},
        )
        assert resp.status_code == 200, f"status: {resp.status_code}, body: {resp.text}"
        data = resp.json()

        # verdict 정상
        assert data.get("verdict") in ("AI-generated", "Real"), (
            f"unexpected verdict: {data.get('verdict')!r}"
        )

        # 미번들 yaya/umm가 models에 error로 남아있으면 안 됨
        for m in data.get("models", []):
            model_id = m.get("model", "")
            if "yaya36095" in model_id or "umm-maybe" in model_id:
                assert False, (
                    f"미번들 모델 {model_id!r}이 응답 models에 포함됨 (필터됐어야 함)"
                )
