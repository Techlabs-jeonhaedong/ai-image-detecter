"""
server.py HTTP 마이크로서비스 테스트.

_AI_DETECTOR_MOCK=1 환경변수로 mock pipeline 사용.
모든 테스트는 실제 네트워크/모델 없이 실행된다.
"""
import io
import os
import struct
import threading
import zlib
import pytest
from unittest.mock import patch
from PIL import Image

# 테스트 전체에 mock 환경변수 적용
os.environ["_AI_DETECTOR_MOCK"] = "1"


# ──────────────────────────────────────────────────────────────────────────────
# 헬퍼
# ──────────────────────────────────────────────────────────────────────────────

def _make_png_bytes(width: int = 64, height: int = 64, color=(128, 64, 32)) -> bytes:
    """PIL로 RGB PNG 바이트 생성."""
    buf = io.BytesIO()
    img = Image.new("RGB", (width, height), color=color)
    img.save(buf, format="PNG")
    return buf.getvalue()


def _make_jpeg_bytes(width: int = 32, height: int = 32) -> bytes:
    buf = io.BytesIO()
    img = Image.new("RGB", (width, height), color=(0, 128, 255))
    img.save(buf, format="JPEG")
    return buf.getvalue()


def _make_sd_png_bytes() -> bytes:
    """Stable Diffusion parameters PNG text chunk이 심긴 PNG 바이트."""
    buf = io.BytesIO()
    img = Image.new("RGB", (64, 64), color=(200, 100, 50))
    # SD A1111 파라미터 형식 (2개 이상 키워드 필요)
    sd_params = "a beautiful landscape\nSteps: 20, Sampler: Euler a, CFG scale: 7, Seed: 12345"
    img.save(buf, format="PNG", pnginfo=_make_pnginfo({"parameters": sd_params}))
    return buf.getvalue()


def _make_pnginfo(text_dict: dict):
    """PIL PngImagePlugin.PngInfo 생성 헬퍼."""
    from PIL import PngImagePlugin
    info = PngImagePlugin.PngInfo()
    for key, val in text_dict.items():
        info.add_text(key, val)
    return info


@pytest.fixture(autouse=True)
def _set_mock_env(monkeypatch):
    """모든 테스트에 mock 환경변수 보장."""
    monkeypatch.setenv("_AI_DETECTOR_MOCK", "1")


@pytest.fixture
def client():
    """FastAPI TestClient. 캐시를 테스트마다 리셋."""
    # server 모듈을 매 테스트마다 fresh하게 임포트하기 위해 캐시 초기화
    import importlib
    import server as srv
    srv._PIPELINE_CACHE.clear()

    from fastapi.testclient import TestClient
    return TestClient(srv.app)


# ──────────────────────────────────────────────────────────────────────────────
# /health
# ──────────────────────────────────────────────────────────────────────────────

class TestHealthEndpoint:
    def test_health_returns_200_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}

    def test_health_content_type_json(self, client):
        resp = client.get("/health")
        assert "application/json" in resp.headers["content-type"]


# ──────────────────────────────────────────────────────────────────────────────
# /detect — 정상 케이스
# ──────────────────────────────────────────────────────────────────────────────

class TestDetectNormalCases:
    def test_png_upload_returns_200(self, client):
        resp = client.post(
            "/detect",
            files={"file": ("test.png", _make_png_bytes(), "image/png")},
        )
        assert resp.status_code == 200

    def test_result_has_required_fields(self, client):
        resp = client.post(
            "/detect",
            files={"file": ("test.png", _make_png_bytes(), "image/png")},
        )
        body = resp.json()
        assert "ai_probability" in body
        assert "verdict" in body
        assert "model" in body

    def test_result_ai_probability_is_float(self, client):
        resp = client.post(
            "/detect",
            files={"file": ("test.png", _make_png_bytes(), "image/png")},
        )
        prob = resp.json()["ai_probability"]
        assert isinstance(prob, float)
        assert 0.0 <= prob <= 1.0

    def test_result_verdict_is_string(self, client):
        resp = client.post(
            "/detect",
            files={"file": ("test.png", _make_png_bytes(), "image/png")},
        )
        verdict = resp.json()["verdict"]
        assert verdict in ("AI-generated", "Real")

    def test_jpeg_upload_returns_200(self, client):
        resp = client.post(
            "/detect",
            files={"file": ("test.jpg", _make_jpeg_bytes(), "image/jpeg")},
        )
        assert resp.status_code == 200
        assert resp.json()["error"] is None

    def test_result_has_metadata_field(self, client):
        resp = client.post(
            "/detect",
            files={"file": ("test.png", _make_png_bytes(), "image/png")},
        )
        body = resp.json()
        assert "metadata" in body

    def test_result_has_verdict_source(self, client):
        resp = client.post(
            "/detect",
            files={"file": ("test.png", _make_png_bytes(), "image/png")},
        )
        body = resp.json()
        assert "verdict_source" in body

    def test_mock_pipeline_gives_ai_generated(self, client):
        """mock pipeline은 0.73 반환 → 기본 threshold 0.5 기준 AI-generated."""
        resp = client.post(
            "/detect",
            files={"file": ("test.png", _make_png_bytes(), "image/png")},
        )
        assert resp.json()["verdict"] == "AI-generated"
        assert resp.json()["ai_probability"] == pytest.approx(0.73, abs=1e-9)


# ──────────────────────────────────────────────────────────────────────────────
# /detect — 에러 입력
# ──────────────────────────────────────────────────────────────────────────────

class TestDetectErrorCases:
    def test_missing_file_returns_4xx(self, client):
        resp = client.post("/detect")
        assert resp.status_code in (400, 422)

    def test_empty_file_returns_400(self, client):
        resp = client.post(
            "/detect",
            files={"file": ("empty.png", b"", "image/png")},
        )
        assert resp.status_code == 400

    def test_non_image_file_returns_200_with_error(self, client):
        """비이미지 파일 업로드 → 200 + result.error 채워짐."""
        resp = client.post(
            "/detect",
            files={"file": ("text.txt", b"this is not an image", "text/plain")},
        )
        assert resp.status_code == 200
        assert resp.json()["error"] is not None

    def test_corrupt_image_returns_200_with_error(self, client):
        """손상된 이미지 → 200 + error."""
        corrupt = b"\xff\xd8\xff\xe0" + b"\x00" * 10  # 잘못된 JPEG
        resp = client.post(
            "/detect",
            files={"file": ("corrupt.jpg", corrupt, "image/jpeg")},
        )
        assert resp.status_code == 200
        assert resp.json()["error"] is not None


# ──────────────────────────────────────────────────────────────────────────────
# /detect — threshold 파라미터
# ──────────────────────────────────────────────────────────────────────────────

class TestDetectThreshold:
    def test_threshold_1_5_returns_400(self, client):
        resp = client.post(
            "/detect",
            data={"threshold": "1.5"},
            files={"file": ("test.png", _make_png_bytes(), "image/png")},
        )
        assert resp.status_code == 400

    def test_threshold_negative_returns_400(self, client):
        resp = client.post(
            "/detect",
            data={"threshold": "-0.1"},
            files={"file": ("test.png", _make_png_bytes(), "image/png")},
        )
        assert resp.status_code == 400

    def test_threshold_0_returns_ai_generated(self, client):
        """threshold=0이면 항상 AI-generated."""
        resp = client.post(
            "/detect",
            data={"threshold": "0.0"},
            files={"file": ("test.png", _make_png_bytes(), "image/png")},
        )
        assert resp.status_code == 200
        assert resp.json()["verdict"] == "AI-generated"

    def test_threshold_1_returns_real(self, client):
        """threshold=1이면 항상 Real (mock은 0.73)."""
        resp = client.post(
            "/detect",
            data={"threshold": "1.0"},
            files={"file": ("test.png", _make_png_bytes(), "image/png")},
        )
        assert resp.status_code == 200
        assert resp.json()["verdict"] == "Real"

    def test_threshold_non_numeric_returns_400(self, client):
        resp = client.post(
            "/detect",
            data={"threshold": "abc"},
            files={"file": ("test.png", _make_png_bytes(), "image/png")},
        )
        assert resp.status_code == 400


# ──────────────────────────────────────────────────────────────────────────────
# /detect — ensemble 파라미터
# ──────────────────────────────────────────────────────────────────────────────

class TestDetectEnsemble:
    def test_ensemble_true_models_list_has_multiple(self, client):
        """ensemble=true → models 배열에 여러 모델 결과."""
        resp = client.post(
            "/detect",
            data={"ensemble": "true"},
            files={"file": ("test.png", _make_png_bytes(), "image/png")},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert "models" in body
        assert len(body["models"]) > 1

    def test_ensemble_false_single_model(self, client):
        """ensemble=false → models 배열은 1개."""
        resp = client.post(
            "/detect",
            data={"ensemble": "false"},
            files={"file": ("test.png", _make_png_bytes(), "image/png")},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert len(body.get("models", [])) == 1

    def test_ensemble_with_extra_model_combined(self, client):
        """ensemble + model 파라미터 동시 지정 → 합쳐서 중복 제거."""
        from detector import ENSEMBLE_MODELS
        resp = client.post(
            "/detect",
            data={"ensemble": "true", "model": ENSEMBLE_MODELS[0]},
            files={"file": ("test.png", _make_png_bytes(), "image/png")},
        )
        assert resp.status_code == 200
        body = resp.json()
        model_ids = [m["model"] for m in body["models"]]
        # 중복 없어야 함
        assert len(model_ids) == len(set(model_ids))
        # ENSEMBLE_MODELS 전체 포함
        for m in ENSEMBLE_MODELS:
            assert m in model_ids


# ──────────────────────────────────────────────────────────────────────────────
# /detect — no_metadata 파라미터
# ──────────────────────────────────────────────────────────────────────────────

class TestDetectNoMetadata:
    def test_no_metadata_true_checked_false(self, client):
        """no_metadata=true → metadata.checked == False."""
        resp = client.post(
            "/detect",
            data={"no_metadata": "true"},
            files={"file": ("test.png", _make_png_bytes(), "image/png")},
        )
        assert resp.status_code == 200
        meta = resp.json().get("metadata", {})
        assert meta.get("checked") == False

    def test_no_metadata_false_checked_true(self, client):
        """no_metadata=false(기본) → metadata.checked == True."""
        resp = client.post(
            "/detect",
            files={"file": ("test.png", _make_png_bytes(), "image/png")},
        )
        assert resp.status_code == 200
        meta = resp.json().get("metadata", {})
        assert meta.get("checked") == True


# ──────────────────────────────────────────────────────────────────────────────
# /detect — SD parameters PNG (메타데이터 결정적 신호)
# ──────────────────────────────────────────────────────────────────────────────

class TestDetectMetadataOverride:
    def test_sd_params_png_verdict_ai_and_source_metadata(self, client):
        """SD parameters PNG → verdict 'AI-generated', verdict_source 'metadata'."""
        resp = client.post(
            "/detect",
            files={"file": ("sd_image.png", _make_sd_png_bytes(), "image/png")},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["verdict"] == "AI-generated"
        assert body["verdict_source"] == "metadata"


# ──────────────────────────────────────────────────────────────────────────────
# 파이프라인 캐시 재사용 검증
# ──────────────────────────────────────────────────────────────────────────────

class TestPipelineCache:
    def test_same_model_pipeline_created_once(self, monkeypatch):
        """동일 모델로 두 번 요청 시 실제 pipeline 생성이 1회만 일어남."""
        import server as srv
        from fastapi.testclient import TestClient

        srv._PIPELINE_CACHE.clear()
        call_count = {"n": 0}
        original_get_pipeline = srv._make_pipeline

        def counting_make_pipeline():
            original = original_get_pipeline()
            def wrapper(task, **kwargs):
                call_count["n"] += 1
                return original(task, **kwargs)
            return wrapper

        monkeypatch.setenv("_AI_DETECTOR_MOCK", "1")
        monkeypatch.setattr(srv, "_make_pipeline", counting_make_pipeline)
        srv._PIPELINE_CACHE.clear()

        with TestClient(srv.app) as c:
            c.post("/detect", files={"file": ("a.png", _make_png_bytes(), "image/png")})
            c.post("/detect", files={"file": ("b.png", _make_png_bytes(), "image/png")})

        # mock pipeline도 1회만 생성되어야 함 (캐시 재사용)
        assert call_count["n"] == 1

    def test_cache_dict_populated_after_request(self, monkeypatch):
        """요청 후 _PIPELINE_CACHE에 항목이 생겨야 함."""
        import server as srv
        from fastapi.testclient import TestClient

        monkeypatch.setenv("_AI_DETECTOR_MOCK", "1")
        srv._PIPELINE_CACHE.clear()

        with TestClient(srv.app) as c:
            c.post("/detect", files={"file": ("a.png", _make_png_bytes(), "image/png")})

        assert len(srv._PIPELINE_CACHE) >= 1


# ──────────────────────────────────────────────────────────────────────────────
# 업로드 크기 제한
# ──────────────────────────────────────────────────────────────────────────────

class TestUploadSizeLimit:
    def test_oversized_upload_returns_413(self, monkeypatch):
        """MAX_UPLOAD_BYTES보다 큰 파일 → 413."""
        import server as srv
        from fastapi.testclient import TestClient

        monkeypatch.setenv("_AI_DETECTOR_MOCK", "1")
        monkeypatch.setattr(srv, "MAX_UPLOAD_BYTES", 100)  # 100바이트 제한
        srv._PIPELINE_CACHE.clear()

        big_data = b"x" * 200  # 200바이트 (제한 초과)

        with TestClient(srv.app) as c:
            resp = c.post(
                "/detect",
                files={"file": ("big.png", big_data, "image/png")},
            )
        assert resp.status_code == 413

    def test_within_size_limit_returns_200(self, monkeypatch):
        """크기 제한 이하 파일 → 정상 처리."""
        import server as srv
        from fastapi.testclient import TestClient

        monkeypatch.setenv("_AI_DETECTOR_MOCK", "1")
        monkeypatch.setattr(srv, "MAX_UPLOAD_BYTES", 10 * 1024 * 1024)
        srv._PIPELINE_CACHE.clear()

        with TestClient(srv.app) as c:
            resp = c.post(
                "/detect",
                files={"file": ("test.png", _make_png_bytes(), "image/png")},
            )
        assert resp.status_code == 200


# ──────────────────────────────────────────────────────────────────────────────
# 임시 파일 정리 검증
# ──────────────────────────────────────────────────────────────────────────────

class TestTempFileCleanup:
    def test_temp_file_deleted_after_request(self, client, tmp_path, monkeypatch):
        """요청 처리 후 임시 파일이 삭제되어야 함."""
        import server as srv
        created_paths = []

        original_create = srv._create_temp_file

        def tracking_create(suffix):
            path = original_create(suffix)
            created_paths.append(path)
            return path

        monkeypatch.setattr(srv, "_create_temp_file", tracking_create)

        client.post(
            "/detect",
            files={"file": ("test.png", _make_png_bytes(), "image/png")},
        )

        for path in created_paths:
            assert not os.path.exists(path), f"임시 파일이 삭제되지 않음: {path}"

    def test_temp_file_deleted_even_when_analysis_raises(self, monkeypatch):
        """분석 도중 예외가 발생해도 임시 파일이 정리되어야 함."""
        import server as srv
        from fastapi.testclient import TestClient

        monkeypatch.setenv("_AI_DETECTOR_MOCK", "1")
        srv._PIPELINE_CACHE.clear()

        created_paths = []
        original_create = srv._create_temp_file

        def tracking_create(suffix):
            path = original_create(suffix)
            created_paths.append(path)
            return path

        monkeypatch.setattr(srv, "_create_temp_file", tracking_create)

        # analyze_images_batch가 예외를 던지도록 강제
        def exploding_batch(*args, **kwargs):
            raise RuntimeError("forced analysis failure")

        monkeypatch.setattr("server.analyze_images_batch", exploding_batch, raising=False)

        with TestClient(srv.app) as c:
            # 예외가 500으로 처리됨
            c.post("/detect", files={"file": ("test.png", _make_png_bytes(), "image/png")})

        # 임시 파일은 반드시 삭제돼야 함
        for path in created_paths:
            assert not os.path.exists(path), f"예외 후 임시 파일 누수: {path}"


# ──────────────────────────────────────────────────────────────────────────────
# [항목 1] detect 핸들러가 동기(def)인지 검증
# ──────────────────────────────────────────────────────────────────────────────

class TestDetectHandlerIsSync:
    def test_detect_handler_is_not_coroutine_function(self):
        """detect 핸들러는 async def가 아닌 일반 def여야 한다 (이벤트루프 블로킹 방지)."""
        import asyncio
        import server as srv
        # FastAPI app의 라우트에서 endpoint 함수를 꺼낸다
        route = next(r for r in srv.app.routes if getattr(r, "path", "") == "/detect")
        assert not asyncio.iscoroutinefunction(route.endpoint), (
            "detect 핸들러가 async def입니다. 동기 def로 변경해야 이벤트루프를 블로킹하지 않습니다."
        )


# ──────────────────────────────────────────────────────────────────────────────
# [항목 2] _PIPELINE_CACHE 동시성 보호
# ──────────────────────────────────────────────────────────────────────────────

class TestPipelineCacheConcurrency:
    def test_concurrent_same_model_calls_factory_once(self, monkeypatch):
        """다중 스레드에서 동시에 같은 모델 요청 시 factory가 1회만 호출되어야 함."""
        import server as srv
        from fastapi.testclient import TestClient

        monkeypatch.setenv("_AI_DETECTOR_MOCK", "1")
        srv._PIPELINE_CACHE.clear()

        call_count = {"n": 0}
        lock = threading.Lock()
        original_make = srv._make_pipeline

        def counting_make():
            original = original_make()
            def wrapper(task, **kwargs):
                with lock:
                    call_count["n"] += 1
                return original(task, **kwargs)
            return wrapper

        monkeypatch.setattr(srv, "_make_pipeline", counting_make)
        srv._PIPELINE_CACHE.clear()

        errors = []

        def send_request(client):
            try:
                client.post("/detect", files={"file": ("a.png", _make_png_bytes(), "image/png")})
            except Exception as e:
                errors.append(e)

        with TestClient(srv.app) as c:
            threads = [threading.Thread(target=send_request, args=(c,)) for _ in range(5)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

        assert not errors, f"스레드 오류: {errors}"
        # factory는 정확히 1회만 호출돼야 함
        assert call_count["n"] == 1, f"factory 호출 횟수: {call_count['n']} (1회 기대)"


# ──────────────────────────────────────────────────────────────────────────────
# [항목 3] Content-Length 헤더 사전 거부 (미들웨어)
# ──────────────────────────────────────────────────────────────────────────────

class TestContentLengthMiddleware:
    def test_content_length_exceeds_limit_rejected_before_body_read(self, monkeypatch):
        """Content-Length가 MAX_UPLOAD_BYTES 초과이면 413으로 거부해야 함."""
        import server as srv
        from fastapi.testclient import TestClient

        monkeypatch.setenv("_AI_DETECTOR_MOCK", "1")
        monkeypatch.setattr(srv, "MAX_UPLOAD_BYTES", 100)
        srv._PIPELINE_CACHE.clear()

        # Content-Length를 직접 지정하지 않아도 httpx/TestClient가 자동 계산하는데
        # 여기서는 실제로 큰 바디를 보내서 미들웨어가 거부하는지 검증
        big_data = b"x" * 200

        with TestClient(srv.app, raise_server_exceptions=False) as c:
            resp = c.post(
                "/detect",
                files={"file": ("big.png", big_data, "image/png")},
            )
        assert resp.status_code == 413

    def test_content_length_within_limit_passes_middleware(self, monkeypatch):
        """Content-Length가 한도 이내이면 미들웨어를 통과해야 함."""
        import server as srv
        from fastapi.testclient import TestClient

        monkeypatch.setenv("_AI_DETECTOR_MOCK", "1")
        monkeypatch.setattr(srv, "MAX_UPLOAD_BYTES", 10 * 1024 * 1024)
        srv._PIPELINE_CACHE.clear()

        with TestClient(srv.app) as c:
            resp = c.post(
                "/detect",
                files={"file": ("test.png", _make_png_bytes(), "image/png")},
            )
        assert resp.status_code == 200


# ──────────────────────────────────────────────────────────────────────────────
# [항목 4] pipeline 캐시 LRU 크기 제한
# ──────────────────────────────────────────────────────────────────────────────

class TestPipelineCacheLRU:
    def test_cache_evicts_oldest_when_over_limit(self, monkeypatch):
        """MAX_CACHED_PIPELINES=2 시 3번째 모델 추가 시 가장 오래된 항목이 evict돼야 함."""
        import server as srv
        from fastapi.testclient import TestClient

        monkeypatch.setenv("_AI_DETECTOR_MOCK", "1")
        monkeypatch.setattr(srv, "MAX_CACHED_PIPELINES", 2)
        srv._PIPELINE_CACHE.clear()

        with TestClient(srv.app) as c:
            # 3개의 서로 다른 모델 요청
            for model_id in ["model-A", "model-B", "model-C"]:
                c.post(
                    "/detect",
                    data={"model": model_id},
                    files={"file": ("test.png", _make_png_bytes(), "image/png")},
                )

        assert len(srv._PIPELINE_CACHE) <= 2, (
            f"캐시 크기 {len(srv._PIPELINE_CACHE)}개 — MAX_CACHED_PIPELINES=2 초과"
        )
        # 가장 오래된 model-A는 evict됐어야 함
        assert "model-A" not in srv._PIPELINE_CACHE

    def test_cache_size_never_exceeds_max(self, monkeypatch):
        """연속으로 다른 모델을 10개 요청해도 캐시 크기가 MAX_CACHED_PIPELINES를 넘지 않음."""
        import server as srv
        from fastapi.testclient import TestClient

        monkeypatch.setenv("_AI_DETECTOR_MOCK", "1")
        monkeypatch.setattr(srv, "MAX_CACHED_PIPELINES", 3)
        srv._PIPELINE_CACHE.clear()

        with TestClient(srv.app) as c:
            for i in range(10):
                c.post(
                    "/detect",
                    data={"model": f"model-{i}"},
                    files={"file": ("test.png", _make_png_bytes(), "image/png")},
                )

        assert len(srv._PIPELINE_CACHE) <= 3


# ──────────────────────────────────────────────────────────────────────────────
# [항목 5] ALLOWED_MODELS 화이트리스트
# ──────────────────────────────────────────────────────────────────────────────

class TestAllowedModels:
    def test_allowed_models_unset_permits_any_model(self, monkeypatch):
        """ALLOWED_MODELS 미설정 시 임의 모델 ID가 허용된다."""
        import server as srv
        from fastapi.testclient import TestClient

        monkeypatch.delenv("ALLOWED_MODELS", raising=False)
        monkeypatch.setenv("_AI_DETECTOR_MOCK", "1")
        srv._PIPELINE_CACHE.clear()

        with TestClient(srv.app) as c:
            resp = c.post(
                "/detect",
                data={"model": "any/arbitrary-model"},
                files={"file": ("test.png", _make_png_bytes(), "image/png")},
            )
        assert resp.status_code == 200

    def test_allowed_models_set_blocks_unlisted_model(self, monkeypatch):
        """ALLOWED_MODELS 설정 시 목록 밖 모델 요청은 400이어야 함."""
        import server as srv
        from fastapi.testclient import TestClient

        monkeypatch.setenv("ALLOWED_MODELS", "allowed/model-a,allowed/model-b")
        monkeypatch.setenv("_AI_DETECTOR_MOCK", "1")
        srv._PIPELINE_CACHE.clear()

        with TestClient(srv.app) as c:
            resp = c.post(
                "/detect",
                data={"model": "not-in/allowed-list"},
                files={"file": ("test.png", _make_png_bytes(), "image/png")},
            )
        assert resp.status_code == 400

    def test_allowed_models_set_permits_listed_model(self, monkeypatch):
        """ALLOWED_MODELS 설정 시 목록 안 모델 요청은 정상 처리돼야 함."""
        import server as srv
        from fastapi.testclient import TestClient

        monkeypatch.setenv("ALLOWED_MODELS", "allowed/model-a,allowed/model-b")
        monkeypatch.setenv("_AI_DETECTOR_MOCK", "1")
        srv._PIPELINE_CACHE.clear()

        with TestClient(srv.app) as c:
            resp = c.post(
                "/detect",
                data={"model": "allowed/model-a"},
                files={"file": ("test.png", _make_png_bytes(), "image/png")},
            )
        assert resp.status_code == 200

    def test_allowed_models_set_blocks_unlisted_ensemble(self, monkeypatch):
        """ALLOWED_MODELS 설정 시 ensemble에 목록 밖 모델이 있으면 400이어야 함."""
        import server as srv
        from fastapi.testclient import TestClient

        # ENSEMBLE_MODELS 전부 허용하지 않음
        monkeypatch.setenv("ALLOWED_MODELS", "only/one-model")
        monkeypatch.setenv("_AI_DETECTOR_MOCK", "1")
        srv._PIPELINE_CACHE.clear()

        with TestClient(srv.app) as c:
            resp = c.post(
                "/detect",
                data={"ensemble": "true"},
                files={"file": ("test.png", _make_png_bytes(), "image/png")},
            )
        assert resp.status_code == 400


# ──────────────────────────────────────────────────────────────────────────────
# [항목 6] 전역 예외 핸들러 — HTTPException 패스스루 / 예외 정보 미노출
# ──────────────────────────────────────────────────────────────────────────────

class TestGlobalExceptionHandler:
    def test_http_exception_400_passthrough(self, monkeypatch):
        """HTTPException(400)이 500으로 둔갑하지 않아야 함."""
        import server as srv
        from fastapi.testclient import TestClient

        monkeypatch.setenv("_AI_DETECTOR_MOCK", "1")
        srv._PIPELINE_CACHE.clear()

        with TestClient(srv.app) as c:
            resp = c.post(
                "/detect",
                data={"threshold": "abc"},
                files={"file": ("test.png", _make_png_bytes(), "image/png")},
            )
        assert resp.status_code == 400

    def test_http_exception_413_passthrough(self, monkeypatch):
        """HTTPException(413)이 500으로 둔갑하지 않아야 함."""
        import server as srv
        from fastapi.testclient import TestClient

        monkeypatch.setenv("_AI_DETECTOR_MOCK", "1")
        monkeypatch.setattr(srv, "MAX_UPLOAD_BYTES", 10)
        srv._PIPELINE_CACHE.clear()

        with TestClient(srv.app, raise_server_exceptions=False) as c:
            resp = c.post(
                "/detect",
                files={"file": ("big.png", b"x" * 50, "image/png")},
            )
        assert resp.status_code == 413

    def test_internal_exception_does_not_expose_class_name(self, monkeypatch):
        """내부 예외 발생 시 응답 본문에 예외 클래스명이 노출되지 않아야 함."""
        import server as srv
        import detector as det
        from fastapi.testclient import TestClient

        monkeypatch.setenv("_AI_DETECTOR_MOCK", "1")
        srv._PIPELINE_CACHE.clear()

        def boom(*args, **kwargs):
            raise RuntimeError("super_secret_internal_error")

        # server.py는 함수 실행 시점에 detector에서 import하므로 detector 모듈을 패치
        monkeypatch.setattr(det, "analyze_images_batch", boom)

        with TestClient(srv.app, raise_server_exceptions=False) as c:
            resp = c.post(
                "/detect",
                files={"file": ("test.png", _make_png_bytes(), "image/png")},
            )

        assert resp.status_code == 500
        body = resp.json()
        detail = body.get("detail", "")
        assert "RuntimeError" not in detail, "응답에 예외 클래스명이 노출됨"
        assert "super_secret_internal_error" not in detail, "응답에 내부 에러 메시지가 노출됨"
        assert "Internal server error" in detail


# ──────────────────────────────────────────────────────────────────────────────
# [항목 8] model 빈 문자열 필터링
# ──────────────────────────────────────────────────────────────────────────────

class TestModelEmptyStringFilter:
    def test_empty_model_string_falls_back_to_default(self, monkeypatch):
        """model="" 전송 시 기본 모델로 처리돼야 함 (빈 ID로 추론 시도 없음)."""
        import server as srv
        from fastapi.testclient import TestClient

        monkeypatch.setenv("_AI_DETECTOR_MOCK", "1")
        srv._PIPELINE_CACHE.clear()

        with TestClient(srv.app) as c:
            resp = c.post(
                "/detect",
                data={"model": ""},
                files={"file": ("test.png", _make_png_bytes(), "image/png")},
            )
        assert resp.status_code == 200
        from detector import DEFAULT_MODEL
        body = resp.json()
        # 단일 모델일 때 model 필드는 모델 ID
        assert body.get("model") == DEFAULT_MODEL

    def test_whitespace_only_model_string_falls_back_to_default(self, monkeypatch):
        """model="  " 전송 시 기본 모델로 처리돼야 함."""
        import server as srv
        from fastapi.testclient import TestClient

        monkeypatch.setenv("_AI_DETECTOR_MOCK", "1")
        srv._PIPELINE_CACHE.clear()

        with TestClient(srv.app) as c:
            resp = c.post(
                "/detect",
                data={"model": "   "},
                files={"file": ("test.png", _make_png_bytes(), "image/png")},
            )
        assert resp.status_code == 200
        from detector import DEFAULT_MODEL
        assert resp.json().get("model") == DEFAULT_MODEL

    def test_duplicate_model_strings_deduplicated(self, monkeypatch):
        """같은 모델 ID를 여러 번 지정해도 중복 제거돼야 함."""
        import server as srv
        from fastapi.testclient import TestClient

        monkeypatch.setenv("_AI_DETECTOR_MOCK", "1")
        srv._PIPELINE_CACHE.clear()

        png = _make_png_bytes()
        with TestClient(srv.app) as c:
            # httpx multipart: data dict에 list value로 반복 파라미터 전달
            resp = c.post(
                "/detect",
                data={"model": ["some/model", "some/model"]},
                files={"file": ("test.png", png, "image/png")},
            )
        assert resp.status_code == 200
        body = resp.json()
        model_ids = [m["model"] for m in body.get("models", [])]
        assert len(model_ids) == len(set(model_ids)), "중복 모델이 제거되지 않음"
