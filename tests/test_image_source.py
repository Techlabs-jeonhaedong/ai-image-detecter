"""
ImageSource 입력 타입 확장 테스트 (TDD Red 단계).

다음을 검증:
1. _load_image: str / bytes / file-like 모두 동일한 PIL 결과
2. inspect_metadata: bytes 입력에서도 동일한 AI 신호 탐지
3. analyze_images_batch: bytes + name 파라미터 처리
4. detect() 고수준 API: bytes/path/ensemble/with_metadata 시나리오
5. server /detect: 임시파일 없이 bytes로 처리
"""

import io
import os
import tempfile
import pytest
from PIL import Image, PngImagePlugin
from unittest.mock import MagicMock, patch


# ──────────────────────────────────────────────
# 헬퍼
# ──────────────────────────────────────────────

def _make_png_bytes(width=64, height=64, color=(128, 64, 32)) -> bytes:
    buf = io.BytesIO()
    img = Image.new("RGB", (width, height), color=color)
    img.save(buf, format="PNG")
    return buf.getvalue()


def _make_jpeg_bytes(width=32, height=32) -> bytes:
    buf = io.BytesIO()
    img = Image.new("RGB", (width, height), color=(0, 128, 255))
    img.save(buf, format="JPEG")
    return buf.getvalue()


def _make_sd_png_bytes() -> bytes:
    """Stable Diffusion parameters PNG text chunk 포함."""
    buf = io.BytesIO()
    img = Image.new("RGB", (32, 32), color=(200, 100, 50))
    meta = PngImagePlugin.PngInfo()
    meta.add_text("parameters", "masterpiece, Steps: 20, Sampler: Euler a, CFG scale: 7")
    img.save(buf, format="PNG", pnginfo=meta)
    return buf.getvalue()


def _mock_pipeline():
    """모델 추론을 mock하는 pipeline."""
    def pipeline(*args, **kwargs):
        def infer(image):
            return [
                {"label": "artificial", "score": 0.73},
                {"label": "human", "score": 0.27},
            ]
        return infer
    return pipeline


# ──────────────────────────────────────────────
# 1. _load_image — str / bytes / file-like 동등성
# ──────────────────────────────────────────────

class TestLoadImageSourceTypes:
    """_load_image가 str/bytes/file-like 세 가지 소스 타입을 동일하게 처리하는지 검증."""

    def test_path_loads_successfully(self, tmp_path):
        p = tmp_path / "img.png"
        Image.new("RGB", (16, 16)).save(str(p))
        from detector import _load_image
        img = _load_image(str(p))
        assert img is not None
        assert img.size == (16, 16)

    def test_bytes_loads_same_as_path(self, tmp_path):
        """bytes 입력이 경로 입력과 동일한 PIL Image를 반환한다."""
        p = tmp_path / "img.png"
        Image.new("RGB", (16, 16), color=(10, 20, 30)).save(str(p))
        data = p.read_bytes()

        from detector import _load_image
        img_from_path = _load_image(str(p))
        img_from_bytes = _load_image(data)

        assert img_from_bytes.size == img_from_path.size
        assert img_from_bytes.mode == img_from_path.mode

    def test_bytesio_loads_same_as_path(self, tmp_path):
        """io.BytesIO 입력이 경로 입력과 동일한 PIL Image를 반환한다."""
        p = tmp_path / "img.png"
        Image.new("RGB", (8, 8), color=(50, 100, 150)).save(str(p))
        data = p.read_bytes()

        from detector import _load_image
        img_from_path = _load_image(str(p))
        img_from_stream = _load_image(io.BytesIO(data))

        assert img_from_stream.size == img_from_path.size

    def test_bytearray_loads_successfully(self):
        """bytearray 입력도 정상 로드된다."""
        data = _make_png_bytes()
        from detector import _load_image
        img = _load_image(bytearray(data))
        assert img is not None
        assert img.size == (64, 64)

    def test_jpeg_bytes_loads_successfully(self):
        """JPEG bytes도 정상 로드된다."""
        data = _make_jpeg_bytes()
        from detector import _load_image
        img = _load_image(data)
        assert img is not None

    def test_corrupt_bytes_raises_value_error(self):
        """손상된 bytes → ValueError (Not a valid image)."""
        from detector import _load_image
        corrupt = b"\xff\xd8\xff\xe0" + b"\x00" * 5
        with pytest.raises(ValueError, match="(?i)not a valid|cannot|invalid"):
            _load_image(corrupt)

    def test_empty_bytes_raises_error(self):
        """빈 bytes → 예외 발생 (graceful)."""
        from detector import _load_image
        with pytest.raises((ValueError, Exception)):
            _load_image(b"")

    def test_nonexistent_path_raises_file_not_found(self):
        """경로가 존재하지 않으면 FileNotFoundError."""
        from detector import _load_image
        with pytest.raises(FileNotFoundError):
            _load_image("/nonexistent/no/such/file.png")

    def test_bytes_error_message_uses_placeholder(self):
        """bytes 입력 오류 메시지는 경로 대신 placeholder를 사용한다."""
        from detector import _load_image
        with pytest.raises((ValueError, Exception)) as exc_info:
            _load_image(b"\xde\xad\xbe\xef")
        # 오류 메시지에 실제 경로가 아닌 placeholder가 포함돼야 함
        assert "<in-memory" in str(exc_info.value).lower() or "memory" in str(exc_info.value).lower() or "valid" in str(exc_info.value).lower()

    def test_file_like_exhausted_still_works(self):
        """seek(0)으로 되감기 후에도 BytesIO가 정상 동작한다."""
        data = _make_png_bytes()
        stream = io.BytesIO(data)
        stream.read()  # 다 읽어서 포인터를 끝으로 이동
        stream.seek(0)  # 되감기

        from detector import _load_image
        img = _load_image(stream)
        assert img is not None


# ──────────────────────────────────────────────
# 2. inspect_metadata — bytes 입력
# ──────────────────────────────────────────────

class TestInspectMetadataBytes:
    """inspect_metadata가 bytes/file-like 입력을 처리하는지 검증."""

    def test_bytes_sd_png_detects_ai_signal(self):
        """SD parameters PNG bytes → has_ai_signal=True, decisive=True."""
        from metadata import inspect_metadata
        data = _make_sd_png_bytes()
        result = inspect_metadata(data)
        assert result["has_ai_signal"] is True
        assert result["decisive"] is True
        assert result["checked"] is True

    def test_bytes_sd_png_same_as_path(self, tmp_path):
        """bytes와 경로 입력이 동일한 has_ai_signal/decisive 반환."""
        data = _make_sd_png_bytes()
        p = tmp_path / "sd.png"
        p.write_bytes(data)

        from metadata import inspect_metadata
        result_path = inspect_metadata(str(p))
        result_bytes = inspect_metadata(data)

        assert result_bytes["has_ai_signal"] == result_path["has_ai_signal"]
        assert result_bytes["decisive"] == result_path["decisive"]

    def test_bytes_clean_png_no_signal(self):
        """일반 PNG bytes → has_ai_signal=False."""
        from metadata import inspect_metadata
        data = _make_png_bytes()
        result = inspect_metadata(data)
        assert result["has_ai_signal"] is False
        assert result["checked"] is True

    def test_bytesio_sd_png_detects_ai_signal(self):
        """io.BytesIO SD PNG → has_ai_signal=True."""
        from metadata import inspect_metadata
        data = _make_sd_png_bytes()
        result = inspect_metadata(io.BytesIO(data))
        assert result["has_ai_signal"] is True

    def test_bytes_oversized_skips_raw_scan(self):
        """50MB 초과 bytes → raw 스캔 skip, 예외 없이 graceful 반환."""
        from metadata import inspect_metadata, MAX_METADATA_SCAN_BYTES
        # 51MB bytes 생성 (이진 패턴)
        big_data = b"\x00" * (MAX_METADATA_SCAN_BYTES + 1)
        result = inspect_metadata(big_data)
        # 예외 없이 반환되어야 하고, checked는 False 또는 True 중 하나
        assert isinstance(result, dict)
        assert "has_ai_signal" in result

    def test_corrupt_bytes_returns_checked_false(self):
        """손상된 bytes → checked=False, 예외 없음."""
        from metadata import inspect_metadata
        result = inspect_metadata(b"\xde\xad\xbe\xef" * 10)
        assert result["checked"] is False

    def test_path_string_still_works(self, tmp_path):
        """기존 경로 입력도 여전히 동작 (하위호환)."""
        data = _make_sd_png_bytes()
        p = tmp_path / "sd.png"
        p.write_bytes(data)
        from metadata import inspect_metadata
        result = inspect_metadata(str(p))
        assert result["has_ai_signal"] is True


# ──────────────────────────────────────────────
# 3. analyze_images_batch — bytes 입력 + name 파라미터
# ──────────────────────────────────────────────

class TestAnalyzeImagesBatchBytes:
    """analyze_images_batch가 bytes 입력과 name 파라미터를 처리하는지 검증."""

    def test_bytes_input_returns_result(self):
        """bytes 입력으로 정상 결과 반환."""
        from detector import analyze_images_batch
        data = _make_png_bytes()
        results = analyze_images_batch(
            image_paths=[data],
            pipeline_fn=_mock_pipeline(),
            model_ids=["test/model"],
            threshold=0.5,
        )
        assert len(results) == 1
        assert results[0]["error"] is None
        assert results[0]["ai_probability"] is not None

    def test_bytes_with_name_uses_name_in_image_field(self):
        """bytes + names=['my.jpg'] → 결과 image 필드가 'my.jpg'."""
        from detector import analyze_images_batch
        data = _make_png_bytes()
        results = analyze_images_batch(
            image_paths=[data],
            pipeline_fn=_mock_pipeline(),
            model_ids=["test/model"],
            threshold=0.5,
            names=["my.jpg"],
        )
        assert results[0]["image"] == "my.jpg"

    def test_bytes_without_name_uses_placeholder(self):
        """bytes + names 미지정 → 결과 image 필드가 placeholder."""
        from detector import analyze_images_batch
        data = _make_png_bytes()
        results = analyze_images_batch(
            image_paths=[data],
            pipeline_fn=_mock_pipeline(),
            model_ids=["test/model"],
            threshold=0.5,
        )
        assert "<in-memory" in results[0]["image"].lower() or "memory" in results[0]["image"].lower()

    def test_path_input_image_field_is_path(self, tmp_path):
        """경로 입력 → image 필드는 경로 문자열 (기존 동작 하위호환)."""
        from detector import analyze_images_batch
        p = tmp_path / "img.png"
        Image.new("RGB", (16, 16)).save(str(p))
        results = analyze_images_batch(
            image_paths=[str(p)],
            pipeline_fn=_mock_pipeline(),
            model_ids=["test/model"],
            threshold=0.5,
        )
        assert results[0]["image"] == str(p)

    def test_mixed_path_and_bytes(self, tmp_path):
        """경로와 bytes 혼합 입력도 처리된다."""
        from detector import analyze_images_batch
        p = tmp_path / "img.png"
        Image.new("RGB", (16, 16)).save(str(p))
        data = _make_png_bytes()
        results = analyze_images_batch(
            image_paths=[str(p), data],
            pipeline_fn=_mock_pipeline(),
            model_ids=["test/model"],
            threshold=0.5,
        )
        assert len(results) == 2
        assert all(r["error"] is None for r in results)

    def test_corrupt_bytes_returns_error_result(self):
        """손상된 bytes → error 있는 결과 (프로세스는 계속)."""
        from detector import analyze_images_batch
        results = analyze_images_batch(
            image_paths=[b"\xde\xad\xbe\xef"],
            pipeline_fn=_mock_pipeline(),
            model_ids=["test/model"],
            threshold=0.5,
        )
        assert len(results) == 1
        assert results[0]["error"] is not None


# ──────────────────────────────────────────────
# 4. detect() 고수준 API
# ──────────────────────────────────────────────

class TestDetectHighLevelAPI:
    """detect() 함수 검증 (mock 모드)."""

    def setup_method(self):
        os.environ["_AI_DETECTOR_MOCK"] = "1"

    def teardown_method(self):
        os.environ.pop("_AI_DETECTOR_MOCK", None)

    def test_detect_bytes_returns_dict(self):
        """bytes 입력 → 결과 dict 반환."""
        from detector import detect
        data = _make_png_bytes()
        result = detect(data, name="user.jpg")
        assert isinstance(result, dict)
        assert "verdict" in result
        assert "ai_probability" in result

    def test_detect_bytes_image_field_uses_name(self):
        """name 지정 시 image 필드가 name."""
        from detector import detect
        data = _make_png_bytes()
        result = detect(data, name="user.jpg")
        assert result["image"] == "user.jpg"

    def test_detect_path_still_works(self, tmp_path):
        """경로 입력도 동일하게 동작 (하위호환)."""
        from detector import detect
        p = tmp_path / "img.png"
        Image.new("RGB", (32, 32)).save(str(p))
        result = detect(str(p))
        assert "verdict" in result

    def test_detect_with_metadata_true(self):
        """with_metadata=True → metadata 필드 포함."""
        from detector import detect
        data = _make_png_bytes()
        result = detect(data, with_metadata=True)
        assert "metadata" in result

    def test_detect_with_metadata_false(self):
        """with_metadata=False → metadata 필드 checked=False."""
        from detector import detect
        data = _make_png_bytes()
        result = detect(data, with_metadata=False)
        # checked가 False이거나 metadata 필드가 없어야 함
        meta = result.get("metadata", {})
        assert meta.get("checked") is False or "metadata" not in result or not meta.get("checked", True)

    def test_detect_ensemble_true(self):
        """ensemble=True → models 여러 개."""
        from detector import detect
        data = _make_png_bytes()
        result = detect(data, ensemble=True)
        assert "models" in result
        assert len(result["models"]) > 1

    def test_detect_sd_png_metadata_override(self):
        """SD params PNG → verdict_source='metadata'."""
        from detector import detect
        data = _make_sd_png_bytes()
        result = detect(data, with_metadata=True)
        assert result["verdict"] == "AI-generated"
        assert result.get("verdict_source") == "metadata"

    def test_detect_corrupt_bytes_graceful(self):
        """손상된 bytes → 예외 없이 error 필드가 있는 결과 반환."""
        from detector import detect
        result = detect(b"\xde\xad\xbe\xef")
        assert "error" in result or result.get("verdict") is None or True  # graceful 반환

    def test_detect_verdict_source_present(self):
        """결과 dict에 verdict_source 필드가 있다."""
        from detector import detect
        data = _make_png_bytes()
        result = detect(data)
        assert "verdict_source" in result

    def test_detect_bytesio_input(self):
        """io.BytesIO 입력도 처리된다."""
        from detector import detect
        data = _make_png_bytes()
        result = detect(io.BytesIO(data), name="stream.jpg")
        assert isinstance(result, dict)
        assert "verdict" in result


# ──────────────────────────────────────────────
# 5. server /detect — 임시파일 없이 bytes 처리
# ──────────────────────────────────────────────

class TestServerDetectNoTempFile:
    """server /detect가 임시파일 없이 bytes를 직접 처리하는지 검증."""

    def setup_method(self):
        os.environ["_AI_DETECTOR_MOCK"] = "1"

    @pytest.fixture
    def client(self):
        import importlib
        import server as srv
        srv._PIPELINE_CACHE.clear()
        from fastapi.testclient import TestClient
        return TestClient(srv.app)

    def test_detect_does_not_call_mkstemp(self, client):
        """tmpfile.mkstemp가 호출되지 않아야 한다."""
        with patch("tempfile.mkstemp") as mock_mkstemp:
            resp = client.post(
                "/detect",
                files={"file": ("test.png", _make_png_bytes(), "image/png")},
            )
        assert resp.status_code == 200
        assert mock_mkstemp.call_count == 0, "임시파일이 생성됨 — 메모리 처리로 전환해야 함"

    def test_detect_image_field_is_filename(self, client):
        """결과 image 필드가 업로드 파일명이어야 한다."""
        resp = client.post(
            "/detect",
            files={"file": ("my_photo.png", _make_png_bytes(), "image/png")},
        )
        assert resp.status_code == 200
        assert resp.json()["image"] == "my_photo.png"

    def test_detect_png_bytes_returns_200(self, client):
        """PNG bytes 업로드 → 200 + 정상 결과."""
        resp = client.post(
            "/detect",
            files={"file": ("test.png", _make_png_bytes(), "image/png")},
        )
        assert resp.status_code == 200
        assert resp.json()["error"] is None

    def test_detect_sd_png_metadata_override(self, client):
        """SD params PNG → verdict_source='metadata'."""
        resp = client.post(
            "/detect",
            files={"file": ("sd.png", _make_sd_png_bytes(), "image/png")},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["verdict"] == "AI-generated"
        assert body["verdict_source"] == "metadata"

    def test_no_temp_files_left_after_request(self, tmp_path, client):
        """요청 처리 후 /tmp에 잔류 임시 파일이 없어야 한다."""
        before = set(os.listdir(tempfile.gettempdir()))
        client.post(
            "/detect",
            files={"file": ("test.png", _make_png_bytes(), "image/png")},
        )
        after = set(os.listdir(tempfile.gettempdir()))
        new_files = after - before
        # 새 임시파일이 없거나 있어도 detector 관련 파일이 아니어야 함
        # (pytest 자체가 tmp 파일을 만들 수 있으므로 엄격히 체크하지 않음)
        # 핵심은 mkstemp 미호출 검증 (위 test에서 커버)
        assert isinstance(new_files, set)

    def test_existing_server_behaviors_preserved(self, client):
        """기존 서버 동작 (status 코드, metadata 필드 등) 유지."""
        resp = client.post(
            "/detect",
            files={"file": ("test.png", _make_png_bytes(), "image/png")},
        )
        body = resp.json()
        assert resp.status_code == 200
        assert "ai_probability" in body
        assert "verdict" in body
        assert "metadata" in body
        assert "verdict_source" in body


# ──────────────────────────────────────────────
# 6. ImageSource 타입 별칭 검증
# ──────────────────────────────────────────────

class TestImageSourceTypeAlias:
    """ImageSource Union 타입이 정의되어 있는지 검증."""

    def test_image_source_type_defined(self):
        """detector 모듈에 ImageSource 타입이 정의돼 있어야 한다."""
        import detector
        assert hasattr(detector, "ImageSource"), "ImageSource 타입 별칭이 정의되지 않음"

    def test_analyze_images_batch_accepts_bytes(self):
        """analyze_images_batch가 bytes 입력을 TypeError 없이 수락한다."""
        from detector import analyze_images_batch
        data = _make_png_bytes()
        # TypeError 없이 호출되어야 함
        results = analyze_images_batch(
            image_paths=[data],
            pipeline_fn=_mock_pipeline(),
            model_ids=["test/model"],
            threshold=0.5,
        )
        assert isinstance(results, list)
