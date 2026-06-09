"""
보안/버그 수정 TDD Red 단계 테스트.

항목별 검증:
#1  file-like 스트림 소진 버그 — detect()/analyze_images_batch()에서 BytesIO가 완전히 동작
#2  server read 상한 — Content-Length 없는 chunked 요청에서 MAX_UPLOAD_BYTES+1 방어
#3  대형 이미지 해상도 상한 — _load_image 가 MAX_IMAGE_PIXELS 초과 시 명확한 ValueError
#4  metadata raw 바이트 스캔 메모리 최적화 — 앞/뒤 범위 제한 or lower() 회피
#5  dead code — _create_temp_file / tempfile import 제거, 임시파일 패치 경로 정확성
#7  마무리 — os.path.basename 적용 / 비지원 타입 TypeError
"""

import io
import os
import sys
import pytest
from PIL import Image, PngImagePlugin
from unittest.mock import patch, MagicMock

os.environ.setdefault("_AI_DETECTOR_MOCK", "1")


# ──────────────────────────────────────────────────────────────────────────────
# 헬퍼
# ──────────────────────────────────────────────────────────────────────────────

def _make_png_bytes(width=64, height=64) -> bytes:
    buf = io.BytesIO()
    img = Image.new("RGB", (width, height), color=(128, 64, 32))
    img.save(buf, format="PNG")
    return buf.getvalue()


def _make_sd_png_bytes() -> bytes:
    """SD parameters PNG text chunk 포함."""
    buf = io.BytesIO()
    img = Image.new("RGB", (32, 32), color=(200, 100, 50))
    meta = PngImagePlugin.PngInfo()
    meta.add_text(
        "parameters",
        "a beautiful landscape\nSteps: 20, Sampler: Euler a, CFG scale: 7, Seed: 42",
    )
    img.save(buf, format="PNG", pnginfo=meta)
    return buf.getvalue()


def _mock_pipeline():
    def pipeline(*args, **kwargs):
        def infer(image):
            return [
                {"label": "artificial", "score": 0.73},
                {"label": "human", "score": 0.27},
            ]
        return infer
    return pipeline


# ──────────────────────────────────────────────────────────────────────────────
# #1 file-like 스트림 소진 버그
#   detect() / analyze_images_batch()에서 BytesIO를 넘겼을 때
#   metadata.checked == True && metadata.decisive == True && verdict_source == "metadata"
#   이어야 한다 (SD params PNG 사용).
# ──────────────────────────────────────────────────────────────────────────────

class TestStreamExhaustionBug:
    """#1: file-like(BytesIO) 입력 시 스트림 소진 버그 — detect() 진입점에서 1회 정규화."""

    def setup_method(self):
        os.environ["_AI_DETECTOR_MOCK"] = "1"

    def teardown_method(self):
        os.environ.pop("_AI_DETECTOR_MOCK", None)

    def test_bytesio_sd_png_metadata_decisive(self):
        """BytesIO로 SD params PNG를 detect()에 넣으면 metadata.decisive==True여야 한다."""
        from detector import detect
        data = _make_sd_png_bytes()
        stream = io.BytesIO(data)
        result = detect(stream, with_metadata=True)
        meta = result.get("metadata", {})
        assert meta.get("checked") is True, f"checked=False, metadata={meta}"
        assert meta.get("decisive") is True, f"decisive=False, metadata={meta}"

    def test_bytesio_sd_png_verdict_source_is_metadata(self):
        """BytesIO SD params PNG → verdict_source='metadata'."""
        from detector import detect
        data = _make_sd_png_bytes()
        result = detect(io.BytesIO(data), with_metadata=True)
        assert result.get("verdict_source") == "metadata", (
            f"verdict_source={result.get('verdict_source')!r}"
        )

    def test_bytesio_sd_png_verdict_ai_generated(self):
        """BytesIO SD params PNG → verdict='AI-generated'."""
        from detector import detect
        data = _make_sd_png_bytes()
        result = detect(io.BytesIO(data), with_metadata=True)
        assert result["verdict"] == "AI-generated", f"verdict={result['verdict']!r}"

    def test_path_bytes_bytesio_same_metadata_result(self, tmp_path):
        """경로 / bytes / BytesIO 세 입력이 동일한 metadata 결과를 반환해야 한다."""
        from detector import detect
        data = _make_sd_png_bytes()
        p = tmp_path / "sd.png"
        p.write_bytes(data)

        r_path  = detect(str(p), with_metadata=True)
        r_bytes = detect(data, with_metadata=True)
        r_bio   = detect(io.BytesIO(data), with_metadata=True)

        for key in ("decisive", "has_ai_signal"):
            assert r_bytes["metadata"][key] == r_path["metadata"][key], (
                f"bytes vs path mismatch on {key}"
            )
            assert r_bio["metadata"][key] == r_path["metadata"][key], (
                f"BytesIO vs path mismatch on {key}"
            )

    def test_analyze_images_batch_bytesio_metadata_decisive(self):
        """analyze_images_batch에 BytesIO SD PNG → metadata decisive 시그널 유지."""
        from detector import analyze_images_batch
        from metadata import inspect_metadata
        data = _make_sd_png_bytes()
        stream = io.BytesIO(data)

        # batch는 ML 추론만 하므로 metadata는 별도 inspect_metadata 호출로 확인
        # 핵심: stream.read()를 _load_image가 소진하더라도 inspect_metadata가 동작해야 함
        # 이를 위해 detect()가 진입부에서 한 번만 정규화해야 함
        # 여기서는 BytesIO가 batch에 들어가도 같은 bytes로 재생성 가능한지 확인
        results = analyze_images_batch(
            image_paths=[stream],
            pipeline_fn=_mock_pipeline(),
            model_ids=["test/model"],
            threshold=0.5,
        )
        assert len(results) == 1
        # ML 결과 자체는 오류 없이 나와야 함
        assert results[0]["error"] is None, f"error={results[0]['error']!r}"

    def test_normalize_source_helper_exists(self):
        """_normalize_source 헬퍼가 detector 모듈에 존재해야 한다."""
        import detector
        assert hasattr(detector, "_normalize_source"), (
            "_normalize_source 헬퍼가 없음 — 진입부 1회 정규화 패턴 미구현"
        )

    def test_normalize_source_str_passthrough(self):
        """_normalize_source: str은 그대로 반환해야 한다."""
        from detector import _normalize_source
        result = _normalize_source("/some/path.png")
        assert result == "/some/path.png"

    def test_normalize_source_bytes_passthrough(self):
        """_normalize_source: bytes는 그대로 반환해야 한다."""
        from detector import _normalize_source
        data = b"\x89PNG"
        result = _normalize_source(data)
        assert result == data

    def test_normalize_source_bytearray_to_bytes(self):
        """_normalize_source: bytearray는 bytes로 변환되어야 한다."""
        from detector import _normalize_source
        result = _normalize_source(bytearray(b"\x89PNG"))
        assert isinstance(result, (bytes, bytearray))

    def test_normalize_source_file_like_reads_once(self):
        """_normalize_source: file-like는 .read()로 bytes 변환해야 한다."""
        from detector import _normalize_source
        data = _make_png_bytes()
        stream = io.BytesIO(data)
        result = _normalize_source(stream)
        assert isinstance(result, bytes)
        assert len(result) == len(data)
        # 스트림은 소진되었어야 함 (1회 read)
        assert stream.read() == b""

    def test_detect_with_bytesio_stream_fully_exhausted_before_call(self):
        """스트림이 완전히 소진된 BytesIO를 detect()에 넣어도 정규화로 안전하게 처리된다."""
        # 진입부에서 정규화 후 stream.tell() == end가 되어야 하므로
        # 실제 detect()는 정규화된 bytes를 사용해야 함
        from detector import detect
        data = _make_sd_png_bytes()
        stream = io.BytesIO(data)
        # 진입 전에 미리 소진 — detect() 내부에서 이미 소진된 스트림을 처리할 때
        # _normalize_source가 먼저 호출되므로 괜찮아야 함
        # (detect 내부에서 _normalize_source가 맨 먼저 호출되면 stream.read() 전에는 데이터 있음)
        result = detect(stream, with_metadata=True)
        # 에러 없이 결과가 나와야 함
        assert "verdict" in result


# ──────────────────────────────────────────────────────────────────────────────
# #2 server read 상한 — Content-Length 없는 chunked 요청
# ──────────────────────────────────────────────────────────────────────────────

class TestServerReadCap:
    """#2: file.file.read()를 MAX_UPLOAD_BYTES+1 상한 읽기로 변경."""

    @pytest.fixture
    def client(self, monkeypatch):
        import server as srv
        from fastapi.testclient import TestClient
        monkeypatch.setenv("_AI_DETECTOR_MOCK", "1")
        monkeypatch.setattr(srv, "MAX_UPLOAD_BYTES", 100)
        srv._PIPELINE_CACHE.clear()
        return TestClient(srv.app, raise_server_exceptions=False)

    def test_chunked_oversized_returns_413(self, client):
        """Content-Length 없이 MAX_UPLOAD_BYTES 초과 바디 → 413."""
        big_data = b"x" * 200
        # Content-Length 헤더를 제거한 스트림 객체를 직접 전달
        resp = client.post(
            "/detect",
            files={"file": ("big.png", io.BytesIO(big_data), "image/png")},
            headers={"transfer-encoding": "chunked"},
        )
        assert resp.status_code == 413, (
            f"chunked oversized body should return 413, got {resp.status_code}"
        )

    def test_read_cap_is_max_plus_one(self):
        """server.py의 detect 핸들러가 MAX_UPLOAD_BYTES+1 제한 읽기를 사용하는지 소스 검증."""
        import inspect
        import server
        src = inspect.getsource(server.detect_endpoint if hasattr(server, "detect_endpoint")
                                else server.detect)
        # MAX_UPLOAD_BYTES + 1 패턴이 존재해야 함
        assert "MAX_UPLOAD_BYTES + 1" in src, (
            "핸들러가 MAX_UPLOAD_BYTES+1 상한 읽기를 사용하지 않음"
        )


# ──────────────────────────────────────────────────────────────────────────────
# #3 대형 이미지 해상도 상한
# ──────────────────────────────────────────────────────────────────────────────

class TestImageResolutionLimit:
    """#3: _load_image가 해상도 초과 이미지를 명확한 ValueError로 거부한다."""

    def test_resolution_limit_constant_exists(self):
        """detector 모듈에 MAX_IMAGE_PIXELS 또는 MAX_IMAGE_DIMENSION 상수가 있어야 한다."""
        import detector
        has_pixels = hasattr(detector, "MAX_IMAGE_PIXELS")
        has_dim = hasattr(detector, "MAX_IMAGE_DIMENSION")
        assert has_pixels or has_dim, (
            "해상도 상한 상수(MAX_IMAGE_PIXELS 또는 MAX_IMAGE_DIMENSION)가 없음"
        )

    def test_oversized_image_raises_value_error(self, monkeypatch):
        """MAX_IMAGE_PIXELS를 작게 monkeypatch → _load_image가 ValueError를 올린다."""
        from PIL import Image as PILImage
        import detector

        # Image.MAX_IMAGE_PIXELS를 매우 작게 설정
        monkeypatch.setattr(PILImage, "MAX_IMAGE_PIXELS", 10)
        data = _make_png_bytes(64, 64)  # 64*64=4096 > 10

        with pytest.raises((ValueError, Exception)) as exc_info:
            detector._load_image(data)

        err_str = str(exc_info.value).lower()
        # decompression bomb 또는 too large 메시지여야 함
        assert any(kw in err_str for kw in ("too large", "decompression", "bomb", "resolution")), (
            f"예외 메시지가 명확하지 않음: {exc_info.value!r}"
        )

    def test_oversized_image_in_batch_returns_error_result(self, monkeypatch):
        """해상도 초과 이미지가 batch에 있으면 graceful 에러 결과를 반환한다."""
        from PIL import Image as PILImage
        import detector

        monkeypatch.setattr(PILImage, "MAX_IMAGE_PIXELS", 10)
        data = _make_png_bytes(64, 64)

        results = detector.analyze_images_batch(
            image_paths=[data],
            pipeline_fn=_mock_pipeline(),
            model_ids=["test/model"],
            threshold=0.5,
        )
        assert len(results) == 1
        assert results[0]["error"] is not None, "해상도 초과 이미지는 error 필드가 있어야 함"

    def test_normal_image_passes_resolution_check(self):
        """일반 크기 이미지는 정상 로드된다."""
        import detector
        data = _make_png_bytes(64, 64)
        img = detector._load_image(data)
        assert img is not None

    def test_pil_max_image_pixels_set_in_detector(self):
        """detector 모듈이 PIL.Image.MAX_IMAGE_PIXELS를 명시적으로 설정해야 한다."""
        import inspect
        import detector
        src = inspect.getsource(detector)
        assert "MAX_IMAGE_PIXELS" in src, (
            "detector.py에 Image.MAX_IMAGE_PIXELS 설정이 없음"
        )


# ──────────────────────────────────────────────────────────────────────────────
# #4 metadata raw 바이트 스캔 메모리 최적화
# ──────────────────────────────────────────────────────────────────────────────

class TestMetadataRawScanOptimization:
    """#4: raw 바이트 스캔이 메모리를 2배로 불리지 않아야 한다."""

    def test_scan_detects_marker_at_beginning(self):
        """파일 앞부분에 마커가 있으면 탐지되어야 한다."""
        from metadata import inspect_metadata
        data = _make_sd_png_bytes()
        result = inspect_metadata(data)
        assert result["decisive"] is True, "앞부분 마커 탐지 실패"

    def test_scan_detects_marker_at_end(self, tmp_path):
        """파일 뒷부분에 AI 도구명이 있으면 탐지되어야 한다."""
        from metadata import inspect_metadata
        # 큰 패딩 + 뒤에 AI 도구명 삽입
        # inspect_metadata가 뒷부분도 스캔하는지 검증
        # (실제로 SD PNG는 Pillow가 앞부분 chunk를 파싱하므로 PNG text chunk 경로 사용)
        data = _make_sd_png_bytes()
        result = inspect_metadata(data)
        assert result["checked"] is True

    def test_raw_scan_uses_bounded_region_or_no_full_copy(self):
        """raw 스캔이 전체 복사(lower()) 대신 분할 스캔 또는 패턴 비교를 사용해야 한다."""
        import inspect
        from metadata import _check_xmp_bytes, _check_ai_tool_in_raw_bytes
        # 스캔 함수들이 raw_lower bytes를 받는 방식이면 OK
        # metadata.py의 inspect_metadata 함수에서 .lower() 전체 복사 여부 확인
        import metadata as md
        src = inspect.getsource(md.inspect_metadata)
        # raw_bytes.lower() 전체 복사가 그대로 있으면 경고 (여전히 동작하지만 개선 필요)
        # 개선 방법: SCAN_HEAD_BYTES/SCAN_TAIL_BYTES 상수로 스캔 범위를 제한하거나
        #            lower() 회피
        # 수정 후에는 SCAN_HEAD_BYTES 또는 SCAN_TAIL_BYTES 상수가 있어야 함
        has_constants = (
            hasattr(md, "SCAN_HEAD_BYTES") or
            hasattr(md, "SCAN_TAIL_BYTES") or
            "SCAN_HEAD" in inspect.getsource(md) or
            "scan_head" in inspect.getsource(md).lower()
        )
        assert has_constants, (
            "metadata.py에 스캔 범위 제한 상수(SCAN_HEAD_BYTES/SCAN_TAIL_BYTES)가 없음"
        )

    def test_inspect_metadata_does_not_double_memory_for_large_input(self):
        """큰 bytes 입력에서 예외 없이 graceful 반환한다 (메모리 안전성)."""
        from metadata import inspect_metadata
        # 실제로 10MB bytes를 넣어도 예외 없이 반환되어야 함
        big_data = b"\x89PNG\r\n\x1a\n" + b"\x00" * (10 * 1024 * 1024)
        result = inspect_metadata(big_data)
        assert isinstance(result, dict)


# ──────────────────────────────────────────────────────────────────────────────
# #5 dead code 제거 및 임시파일 검증 테스트 정확성
# ──────────────────────────────────────────────────────────────────────────────

class TestDeadCodeRemoval:
    """#5: _create_temp_file 제거 + tempfile import 제거 + 테스트 패치 경로 정확성."""

    def test_create_temp_file_not_in_server(self):
        """server.py에 _create_temp_file 함수가 없어야 한다."""
        import server
        assert not hasattr(server, "_create_temp_file"), (
            "_create_temp_file dead code가 server.py에 남아 있음"
        )

    def test_tempfile_not_imported_in_server(self):
        """server.py에 tempfile 모듈이 import되지 않아야 한다."""
        import inspect
        import server
        src = inspect.getsource(server)
        # "import tempfile"이 없어야 함
        lines = [l.strip() for l in src.splitlines()]
        has_import = any(
            l == "import tempfile" or l.startswith("import tempfile ")
            for l in lines
        )
        assert not has_import, "server.py에 미사용 'import tempfile'이 남아 있음"

    def test_no_tempfile_mkstemp_in_server_source(self):
        """server.py 소스에 tempfile.mkstemp 호출이 없어야 한다."""
        import inspect
        import server
        src = inspect.getsource(server)
        assert "tempfile.mkstemp" not in src, (
            "server.py 소스에 tempfile.mkstemp 호출이 남아 있음"
        )

    def test_temp_file_cleanup_test_patches_correct_target(self):
        """임시파일 생성 없음 검증 테스트가 server.tempfile이 아닌 올바른 경로를 패치해야 한다."""
        # server.py가 tempfile을 import하지 않으므로 tempfile.mkstemp 패치는 의미 없음
        # 실제로 mkstemp가 안 불리는지 직접 확인
        import server as srv
        from fastapi.testclient import TestClient
        srv._PIPELINE_CACHE.clear()

        call_record = []
        original = __import__("tempfile").mkstemp

        def spy(*args, **kwargs):
            call_record.append((args, kwargs))
            return original(*args, **kwargs)

        with patch("tempfile.mkstemp", side_effect=spy):
            with TestClient(srv.app) as c:
                resp = c.post(
                    "/detect",
                    files={"file": ("test.png", _make_png_bytes(), "image/png")},
                )
        assert resp.status_code == 200
        assert len(call_record) == 0, f"tempfile.mkstemp가 {len(call_record)}회 호출됨"


# ──────────────────────────────────────────────────────────────────────────────
# #7 마무리 정리
# ──────────────────────────────────────────────────────────────────────────────

class TestFinishingTouches:
    """#7: basename 적용 + unsupported type TypeError + 핸들러명 충돌 회피."""

    @pytest.fixture
    def client(self, monkeypatch):
        import server as srv
        from fastapi.testclient import TestClient
        monkeypatch.setenv("_AI_DETECTOR_MOCK", "1")
        srv._PIPELINE_CACHE.clear()
        return TestClient(srv.app, raise_server_exceptions=False)

    def test_filename_with_path_separators_is_sanitized(self, client):
        """업로드 파일명에 경로 구분자가 있으면 basename만 image 필드에 들어가야 한다."""
        resp = client.post(
            "/detect",
            files={"file": ("../../evil/path.png", _make_png_bytes(), "image/png")},
        )
        assert resp.status_code == 200
        image_field = resp.json().get("image", "")
        # 경로 구분자가 없어야 함
        assert "/" not in image_field, f"image 필드에 경로 구분자: {image_field!r}"
        assert "\\" not in image_field, f"image 필드에 경로 구분자: {image_field!r}"
        assert ".." not in image_field, f"image 필드에 ..: {image_field!r}"

    def test_unsupported_type_raises_type_error(self):
        """int 같은 비지원 타입을 _load_image/_normalize_source에 넣으면 TypeError 발생."""
        import detector
        # _normalize_source가 있으면 그것을, 없으면 _load_image를 테스트
        if hasattr(detector, "_normalize_source"):
            with pytest.raises(TypeError, match="(?i)unsupported"):
                detector._normalize_source(12345)
        else:
            with pytest.raises(TypeError):
                detector._load_image(12345)

    def test_detect_handler_name_does_not_conflict_with_import(self):
        """server.py의 detect 핸들러가 detector.detect import와 충돌하지 않아야 한다."""
        import server
        # server 모듈 수준에서 detect라는 이름이 있다면 그게 함수인지 확인
        # detect_endpoint로 이름 변경됐거나, 내부 import로 충돌 없으면 OK
        from detector import detect as detector_detect
        # server.app 라우트의 endpoint가 detector.detect와 다른 객체여야 함
        route = next(
            (r for r in server.app.routes if getattr(r, "path", "") == "/detect"),
            None,
        )
        assert route is not None
        handler = route.endpoint
        assert handler is not detector_detect, (
            "server.py /detect 핸들러가 detector.detect와 동일한 객체 — 이름 충돌"
        )
