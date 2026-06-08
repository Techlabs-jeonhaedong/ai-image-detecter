"""detector.py 단위 테스트."""
import os
import pytest
from unittest.mock import MagicMock, patch


class TestExtractAiProbability:
    """extract_ai_probability: 모델 출력 → AI 확률 변환 로직."""

    def test_artificial_label_returns_its_score(self):
        from detector import extract_ai_probability
        results = [
            {"label": "artificial", "score": 0.87},
            {"label": "human", "score": 0.13},
        ]
        assert extract_ai_probability(results) == pytest.approx(0.87)

    def test_human_label_only_returns_complement(self):
        """human 라벨만 있을 때 1 - human_score 반환."""
        from detector import extract_ai_probability
        results = [{"label": "human", "score": 0.9}]
        assert extract_ai_probability(results) == pytest.approx(0.1)

    def test_case_insensitive_label_matching(self):
        from detector import extract_ai_probability
        results = [
            {"label": "Artificial", "score": 0.75},
            {"label": "Human", "score": 0.25},
        ]
        assert extract_ai_probability(results) == pytest.approx(0.75)

    def test_ai_variant_label(self):
        from detector import extract_ai_probability
        results = [{"label": "AI", "score": 0.6}, {"label": "Real", "score": 0.4}]
        assert extract_ai_probability(results) == pytest.approx(0.6)

    def test_fake_variant_label(self):
        from detector import extract_ai_probability
        results = [{"label": "fake", "score": 0.55}, {"label": "real", "score": 0.45}]
        assert extract_ai_probability(results) == pytest.approx(0.55)

    def test_generated_variant_label(self):
        from detector import extract_ai_probability
        results = [{"label": "generated", "score": 0.70}]
        assert extract_ai_probability(results) == pytest.approx(0.70)

    def test_empty_results_raises(self):
        from detector import extract_ai_probability
        with pytest.raises(ValueError, match="empty"):
            extract_ai_probability([])

    def test_unknown_labels_raises(self):
        from detector import extract_ai_probability
        results = [{"label": "unknown_xyz", "score": 0.5}]
        with pytest.raises(ValueError, match="label"):
            extract_ai_probability(results)

    def test_score_zero(self):
        from detector import extract_ai_probability
        results = [{"label": "artificial", "score": 0.0}, {"label": "human", "score": 1.0}]
        assert extract_ai_probability(results) == pytest.approx(0.0)

    def test_score_one(self):
        from detector import extract_ai_probability
        results = [{"label": "artificial", "score": 1.0}, {"label": "human", "score": 0.0}]
        assert extract_ai_probability(results) == pytest.approx(1.0)

    def test_sdxl_detector_labels(self):
        """Organika/sdxl-detector 실제 라벨 형식."""
        from detector import extract_ai_probability
        # 모델 실제 출력 예시: artificial / human
        results = [
            {"label": "artificial", "score": 0.9912},
            {"label": "human", "score": 0.0088},
        ]
        assert extract_ai_probability(results) == pytest.approx(0.9912)


class TestDetermineVerdict:
    """determine_verdict: threshold 기반 판정 로직."""

    def test_above_threshold_is_ai(self):
        from detector import determine_verdict
        assert determine_verdict(0.8, threshold=0.5) == "AI-generated"

    def test_below_threshold_is_real(self):
        from detector import determine_verdict
        assert determine_verdict(0.3, threshold=0.5) == "Real"

    def test_exactly_at_threshold_is_ai(self):
        """threshold와 같은 경우 AI-generated 판정."""
        from detector import determine_verdict
        assert determine_verdict(0.5, threshold=0.5) == "AI-generated"

    def test_custom_threshold_high(self):
        from detector import determine_verdict
        assert determine_verdict(0.7, threshold=0.9) == "Real"

    def test_custom_threshold_low(self):
        from detector import determine_verdict
        assert determine_verdict(0.1, threshold=0.05) == "AI-generated"

    def test_zero_probability_is_real(self):
        from detector import determine_verdict
        assert determine_verdict(0.0, threshold=0.5) == "Real"

    def test_one_probability_is_ai(self):
        from detector import determine_verdict
        assert determine_verdict(1.0, threshold=0.5) == "AI-generated"

    def test_invalid_threshold_below_zero_raises(self):
        from detector import determine_verdict
        with pytest.raises(ValueError):
            determine_verdict(0.5, threshold=-0.1)

    def test_invalid_threshold_above_one_raises(self):
        from detector import determine_verdict
        with pytest.raises(ValueError):
            determine_verdict(0.5, threshold=1.1)

    def test_invalid_probability_raises(self):
        from detector import determine_verdict
        with pytest.raises(ValueError):
            determine_verdict(1.5, threshold=0.5)


class TestFormatResult:
    """format_result: 출력 포맷팅 로직."""

    def test_human_readable_contains_probability(self):
        from detector import format_result
        result = {"image": "test.jpg", "ai_probability": 0.87, "verdict": "AI-generated", "model": "test/model", "error": None}
        text = format_result(result, json_mode=False)
        assert "87.0%" in text or "87%" in text

    def test_human_readable_contains_verdict(self):
        from detector import format_result
        result = {"image": "test.jpg", "ai_probability": 0.87, "verdict": "AI-generated", "model": "test/model", "error": None}
        text = format_result(result, json_mode=False)
        assert "AI-generated" in text

    def test_human_readable_contains_model_name(self):
        from detector import format_result
        result = {"image": "test.jpg", "ai_probability": 0.3, "verdict": "Real", "model": "test/model", "error": None}
        text = format_result(result, json_mode=False)
        assert "test/model" in text

    def test_json_mode_is_valid_json(self):
        import json
        from detector import format_result
        result = {"image": "test.jpg", "ai_probability": 0.87, "verdict": "AI-generated", "model": "test/model", "error": None}
        text = format_result(result, json_mode=True)
        parsed = json.loads(text)
        assert parsed["ai_probability"] == pytest.approx(0.87)
        assert parsed["verdict"] == "AI-generated"

    def test_error_result_human_readable(self):
        from detector import format_result
        result = {"image": "bad.jpg", "ai_probability": None, "verdict": None, "model": "test/model", "error": "File not found"}
        text = format_result(result, json_mode=False)
        assert "ERROR" in text.upper() or "error" in text.lower() or "File not found" in text

    def test_error_result_json(self):
        import json
        from detector import format_result
        result = {"image": "bad.jpg", "ai_probability": None, "verdict": None, "model": "test/model", "error": "File not found"}
        text = format_result(result, json_mode=True)
        parsed = json.loads(text)
        assert parsed["error"] == "File not found"
        assert parsed["verdict"] is None

    def test_image_path_in_output(self):
        from detector import format_result
        result = {"image": "/path/to/my_photo.jpg", "ai_probability": 0.1, "verdict": "Real", "model": "m", "error": None}
        text = format_result(result, json_mode=False)
        assert "my_photo.jpg" in text


class TestAnalyzeImage:
    """analyze_image: 이미지 로딩 + 추론 통합 로직."""

    def test_returns_ai_generated_result(self, valid_image_path, mock_pipeline_ai):
        from detector import analyze_image
        result = analyze_image(valid_image_path, pipeline_fn=mock_pipeline_ai, model_id="test/model", threshold=0.5)
        assert result["verdict"] == "AI-generated"
        assert result["ai_probability"] == pytest.approx(0.87)
        assert result["error"] is None

    def test_returns_real_result(self, valid_image_path, mock_pipeline_real):
        from detector import analyze_image
        result = analyze_image(valid_image_path, pipeline_fn=mock_pipeline_real, model_id="test/model", threshold=0.5)
        assert result["verdict"] == "Real"
        assert result["ai_probability"] == pytest.approx(0.08)
        assert result["error"] is None

    def test_model_name_in_result(self, valid_image_path, mock_pipeline_ai):
        from detector import analyze_image
        result = analyze_image(valid_image_path, pipeline_fn=mock_pipeline_ai, model_id="Organika/sdxl-detector", threshold=0.5)
        assert result["model"] == "Organika/sdxl-detector"

    def test_image_path_in_result(self, valid_image_path, mock_pipeline_ai):
        from detector import analyze_image
        result = analyze_image(valid_image_path, pipeline_fn=mock_pipeline_ai, model_id="test/model", threshold=0.5)
        assert result["image"] == valid_image_path

    def test_nonexistent_file_returns_error(self, mock_pipeline_ai):
        from detector import analyze_image
        result = analyze_image("/nonexistent/path/image.jpg", pipeline_fn=mock_pipeline_ai, model_id="test/model", threshold=0.5)
        assert result["error"] is not None
        assert result["verdict"] is None
        assert "not found" in result["error"].lower() or "no such" in result["error"].lower() or "exist" in result["error"].lower()

    def test_non_image_file_returns_error(self, non_image_path, mock_pipeline_ai):
        from detector import analyze_image
        result = analyze_image(non_image_path, pipeline_fn=mock_pipeline_ai, model_id="test/model", threshold=0.5)
        assert result["error"] is not None
        assert result["verdict"] is None

    def test_corrupt_image_returns_error(self, corrupt_image_path, mock_pipeline_ai):
        from detector import analyze_image
        result = analyze_image(corrupt_image_path, pipeline_fn=mock_pipeline_ai, model_id="test/model", threshold=0.5)
        assert result["error"] is not None
        assert result["verdict"] is None

    def test_custom_threshold_respected(self, valid_image_path, mock_pipeline_low_confidence):
        from detector import analyze_image
        result_ai = analyze_image(valid_image_path, pipeline_fn=mock_pipeline_low_confidence, model_id="m", threshold=0.5)
        assert result_ai["verdict"] == "AI-generated"  # 0.5 >= 0.5

        result_real = analyze_image(valid_image_path, pipeline_fn=mock_pipeline_low_confidence, model_id="m", threshold=0.51)
        assert result_real["verdict"] == "Real"  # 0.5 < 0.51

    def test_jpeg_image_works(self, valid_jpeg_path, mock_pipeline_ai):
        from detector import analyze_image
        result = analyze_image(valid_jpeg_path, pipeline_fn=mock_pipeline_ai, model_id="test/model", threshold=0.5)
        assert result["error"] is None

    def test_no_permission_file_returns_error(self, no_permission_image_path, mock_pipeline_ai):
        from detector import analyze_image
        result = analyze_image(no_permission_image_path, pipeline_fn=mock_pipeline_ai, model_id="test/model", threshold=0.5)
        assert result["error"] is not None


class TestAnalyzeImages:
    """analyze_images: 여러 이미지 배치 처리."""

    def test_all_succeed(self, multiple_valid_images, mock_pipeline_ai):
        from detector import analyze_images
        results = analyze_images(multiple_valid_images, pipeline_fn=mock_pipeline_ai, model_id="m", threshold=0.5)
        assert len(results) == 3
        assert all(r["error"] is None for r in results)

    def test_partial_failure_continues(self, valid_image_path, mock_pipeline_ai):
        """한 이미지 실패해도 나머지 계속 처리."""
        from detector import analyze_images
        paths = ["/nonexistent.jpg", valid_image_path]
        results = analyze_images(paths, pipeline_fn=mock_pipeline_ai, model_id="m", threshold=0.5)
        assert len(results) == 2
        assert results[0]["error"] is not None
        assert results[1]["error"] is None

    def test_all_fail_gracefully(self, mock_pipeline_ai):
        from detector import analyze_images
        results = analyze_images(["/no1.jpg", "/no2.jpg"], pipeline_fn=mock_pipeline_ai, model_id="m", threshold=0.5)
        assert len(results) == 2
        assert all(r["error"] is not None for r in results)


# ──────────────────────────────────────────────
# 항목 1: Decompression bomb 방어
# ──────────────────────────────────────────────
class TestDecompressionBomb:
    """_load_image: decompression bomb 방어."""

    def test_load_raises_value_error_on_decompression_bomb(self, tmp_path):
        """MAX_IMAGE_PIXELS를 매우 작은 값으로 설정하면 ValueError가 발생해야 한다."""
        from PIL import Image
        import detector

        # 10x10 이미지 생성 (100 픽셀 > 10 제한)
        p = tmp_path / "big.png"
        Image.new("RGB", (10, 10)).save(str(p))

        import unittest.mock as mock
        with mock.patch.object(Image, "MAX_IMAGE_PIXELS", 10):
            with pytest.raises(ValueError, match="too large|decompression bomb"):
                detector._load_image(str(p))

    def test_decompression_bomb_error_message_contains_path(self, tmp_path):
        """ValueError 메시지에 경로가 포함돼야 한다."""
        from PIL import Image
        import detector

        p = tmp_path / "big2.png"
        Image.new("RGB", (10, 10)).save(str(p))

        import unittest.mock as mock
        with mock.patch.object(Image, "MAX_IMAGE_PIXELS", 10):
            with pytest.raises(ValueError) as exc_info:
                detector._load_image(str(p))
            assert str(p) in str(exc_info.value)

    def test_decompression_bomb_batch_skips_bad_image(self, tmp_path, valid_image_path, mock_pipeline_ai):
        """배치 처리 시 bomb 이미지만 error로 기록되고 나머지는 계속 처리."""
        from PIL import Image
        from detector import analyze_images
        import unittest.mock as mock

        bomb_path = tmp_path / "bomb.png"
        Image.new("RGB", (10, 10)).save(str(bomb_path))

        original_load_image = __import__("detector")._load_image

        def patched_load_image(path):
            if str(bomb_path) == path:
                from PIL import Image as PILImage
                with mock.patch.object(PILImage, "MAX_IMAGE_PIXELS", 10):
                    return original_load_image(path)
            return original_load_image(path)

        with mock.patch("detector._load_image", side_effect=patched_load_image):
            results = analyze_images(
                [str(bomb_path), valid_image_path],
                pipeline_fn=mock_pipeline_ai,
                model_id="m",
                threshold=0.5,
            )

        assert len(results) == 2
        assert results[0]["error"] is not None
        assert "too large" in results[0]["error"].lower() or "decompression" in results[0]["error"].lower()
        assert results[1]["error"] is None


# ──────────────────────────────────────────────
# 항목 2: trust_remote_code=False 명시
# ──────────────────────────────────────────────
class TestTrustRemoteCode:
    """get_real_pipeline: trust_remote_code=False 전달 검증."""

    def test_real_pipeline_passes_trust_remote_code_false(self):
        """get_real_pipeline()이 반환하는 함수가 trust_remote_code=False를 전달해야 한다."""
        from unittest.mock import patch, MagicMock
        import detector

        mock_transformers_pipeline = MagicMock()
        mock_pipe_instance = MagicMock(return_value=[{"label": "artificial", "score": 0.9}])
        mock_transformers_pipeline.return_value = mock_pipe_instance

        with patch("detector.transformers_pipeline", mock_transformers_pipeline):
            pipeline_fn = detector.get_real_pipeline()
            pipeline_fn("image-classification", model="some/model")

        call_kwargs = mock_transformers_pipeline.call_args
        assert call_kwargs is not None
        assert call_kwargs.kwargs.get("trust_remote_code") is False or (
            len(call_kwargs.args) > 0 and False  # positional이면 kwargs 확인 불가이므로 kwargs만 체크
        )
        assert "trust_remote_code" in call_kwargs.kwargs
        assert call_kwargs.kwargs["trust_remote_code"] is False


# ──────────────────────────────────────────────
# 항목 3: 모델 라벨 파싱 스키마 방어
# ──────────────────────────────────────────────
class TestExtractAiProbabilitySchemaGuard:
    """extract_ai_probability: 입력 스키마 방어."""

    def test_result_not_a_list_raises(self):
        from detector import extract_ai_probability
        with pytest.raises(ValueError, match="[Mm]alformed"):
            extract_ai_probability({"label": "artificial", "score": 0.9})

    def test_item_not_a_dict_raises(self):
        from detector import extract_ai_probability
        with pytest.raises(ValueError, match="[Mm]alformed"):
            extract_ai_probability([["artificial", 0.9]])

    def test_label_key_missing_raises(self):
        from detector import extract_ai_probability
        with pytest.raises(ValueError, match="[Mm]alformed"):
            extract_ai_probability([{"score": 0.9}])

    def test_score_key_missing_raises(self):
        from detector import extract_ai_probability
        with pytest.raises(ValueError, match="[Mm]alformed"):
            extract_ai_probability([{"label": "artificial"}])

    def test_label_none_raises(self):
        from detector import extract_ai_probability
        with pytest.raises(ValueError, match="[Mm]alformed"):
            extract_ai_probability([{"label": None, "score": 0.9}])

    def test_label_integer_is_normalized(self):
        """label=정수는 str 변환 후 처리 — 알 수 없는 라벨로 ValueError."""
        from detector import extract_ai_probability
        # 정수 label은 str 변환 → 알 수 없는 라벨이므로 label 관련 ValueError
        with pytest.raises(ValueError):
            extract_ai_probability([{"label": 42, "score": 0.9}])

    def test_score_none_raises(self):
        from detector import extract_ai_probability
        with pytest.raises(ValueError, match="[Mm]alformed"):
            extract_ai_probability([{"label": "artificial", "score": None}])

    def test_label_with_surrounding_whitespace_is_normalized(self):
        """' artificial ' 같이 공백 포함된 라벨은 정상 파싱돼야 한다."""
        from detector import extract_ai_probability
        results = [{"label": " artificial ", "score": 0.85}]
        assert extract_ai_probability(results) == pytest.approx(0.85)

    def test_label_with_surrounding_whitespace_human(self):
        """' human ' 공백 포함 라벨도 정상 파싱."""
        from detector import extract_ai_probability
        results = [{"label": " human ", "score": 0.9}]
        assert extract_ai_probability(results) == pytest.approx(0.1)


# ──────────────────────────────────────────────
# 항목 4: analyze_images/analyze_image 통합 + pipeline 로드 실패 graceful
# ──────────────────────────────────────────────
class TestAnalyzeImagesIntegration:
    """analyze_images: pipeline 생성 실패 graceful 처리 및 단일/다중 경로 일관성."""

    def test_pipeline_creation_failure_returns_error_for_all(self, valid_image_path):
        """pipeline_fn이 예외를 던질 때 모든 이미지에 error가 담긴 결과 리스트 반환."""
        from detector import analyze_images

        def failing_pipeline(*args, **kwargs):
            raise RuntimeError("Model load failed")

        results = analyze_images(
            [valid_image_path, "/other.jpg"],
            pipeline_fn=failing_pipeline,
            model_id="m",
            threshold=0.5,
        )

        assert len(results) == 2
        assert all(r["error"] is not None for r in results)
        assert all("Model load failed" in r["error"] for r in results)

    def test_pipeline_creation_failure_does_not_raise(self, valid_image_path):
        """pipeline_fn 예외 시 프로세스가 죽지 않아야 한다."""
        from detector import analyze_images

        def failing_pipeline(*args, **kwargs):
            raise RuntimeError("boom")

        # 예외 없이 완료돼야 함
        results = analyze_images(
            [valid_image_path],
            pipeline_fn=failing_pipeline,
            model_id="m",
            threshold=0.5,
        )
        assert isinstance(results, list)

    def test_single_and_batch_use_same_logic(self, valid_image_path, mock_pipeline_ai):
        """단일/다중 경로가 동일한 결과 스키마를 반환한다."""
        from detector import analyze_image, analyze_images

        single = analyze_image(valid_image_path, mock_pipeline_ai, "m", 0.5)
        batch = analyze_images([valid_image_path], mock_pipeline_ai, "m", 0.5)

        assert single["verdict"] == batch[0]["verdict"]
        assert single["ai_probability"] == pytest.approx(batch[0]["ai_probability"])
        assert single["error"] == batch[0]["error"]


# ──────────────────────────────────────────────
# 항목 5: _AI_DETECTOR_MOCK 가드 (detect.py 경유)
# ──────────────────────────────────────────────
class TestMockGuardWarning:
    """_get_pipeline_fn: mock 활성화 시 pytest 외부 컨텍스트에서 stderr 경고 출력."""

    def test_mock_warning_emitted_outside_pytest(self):
        """PYTEST_CURRENT_TEST 없는 환경에서 mock=1 이면 stderr에 경고가 나와야 한다."""
        import importlib
        import io
        from unittest.mock import patch
        import sys

        # detect 모듈을 새로 로드해서 상태 격리
        import detect as detect_module

        fake_stderr = io.StringIO()
        env_without_pytest = {k: v for k, v in os.environ.items() if k != "PYTEST_CURRENT_TEST"}
        env_without_pytest["_AI_DETECTOR_MOCK"] = "1"

        with patch.dict(os.environ, env_without_pytest, clear=True):
            with patch("sys.stderr", fake_stderr):
                detect_module._get_pipeline_fn()

        output = fake_stderr.getvalue()
        assert "WARNING" in output or "warning" in output.lower()
        assert "mock" in output.lower()

    def test_mock_warning_not_emitted_inside_pytest(self):
        """PYTEST_CURRENT_TEST 있는 환경에서는 경고가 나오지 않아야 한다."""
        import io
        from unittest.mock import patch
        import detect as detect_module

        fake_stderr = io.StringIO()
        env_with_pytest = {**os.environ, "_AI_DETECTOR_MOCK": "1", "PYTEST_CURRENT_TEST": "some::test"}

        with patch.dict(os.environ, env_with_pytest, clear=True):
            with patch("sys.stderr", fake_stderr):
                detect_module._get_pipeline_fn()

        output = fake_stderr.getvalue()
        assert "WARNING" not in output


# ──────────────────────────────────────────────
# 항목 B: verdict override 정교화 테스트
# ──────────────────────────────────────────────
class TestVerdictOverride:
    """verdict override 정책 — 결정적 메타 신호일 때만 적용."""

    def _make_real_pipeline(self, ai_prob: float):
        """주어진 AI 확률을 반환하는 mock pipeline."""
        def pipeline(*args, **kwargs):
            def infer(image):
                return [
                    {"label": "artificial", "score": ai_prob},
                    {"label": "human", "score": 1.0 - ai_prob},
                ]
            return infer
        return pipeline

    def test_decisive_meta_overrides_real_ml(self, valid_image_path):
        """ML=Real(낮은 확률) + 결정적 메타 신호 → verdict=AI-generated, verdict_source=metadata."""
        from detector import _apply_metadata_override
        result = {
            "image": valid_image_path,
            "ai_probability": 0.05,
            "verdict": "Real",
            "model": "test/m",
            "error": None,
        }
        meta = {
            "has_ai_signal": True,
            "decisive": True,
            "signals": ["SD params detected"],
            "source": "metadata",
            "checked": True,
        }
        updated = _apply_metadata_override(result, meta)
        assert updated["verdict"] == "AI-generated"
        assert updated["verdict_source"] == "metadata"
        assert updated["ai_probability"] == pytest.approx(0.05)  # ML 원본 보존

    def test_weak_meta_does_not_override_verdict(self, valid_image_path):
        """약한 흔적만 있을 때 verdict는 ML 결과 유지."""
        from detector import _apply_metadata_override
        result = {
            "image": valid_image_path,
            "ai_probability": 0.3,
            "verdict": "Real",
            "model": "test/m",
            "error": None,
        }
        meta = {
            "has_ai_signal": False,
            "decisive": False,
            "signals": ["C2PA/JUMBF signature found (weak)"],
            "source": "metadata",
            "checked": True,
        }
        updated = _apply_metadata_override(result, meta)
        assert updated["verdict"] == "Real"
        assert updated.get("verdict_source") == "model"

    def test_ml_error_plus_decisive_meta_gives_ai_verdict(self, valid_image_path):
        """ML 로드 실패 + 결정적 메타 신호 → verdict=AI-generated, verdict_source=metadata."""
        from detector import _apply_metadata_override
        result = {
            "image": valid_image_path,
            "ai_probability": None,
            "verdict": None,
            "model": "test/m",
            "error": "Model load failed",
        }
        meta = {
            "has_ai_signal": True,
            "decisive": True,
            "signals": ["SD params detected"],
            "source": "metadata",
            "checked": True,
        }
        updated = _apply_metadata_override(result, meta)
        assert updated["verdict"] == "AI-generated"
        assert updated["verdict_source"] == "metadata"
        assert updated["error"] is None  # 성공으로 처리

    def test_no_meta_signal_verdict_source_is_model(self, valid_image_path):
        """메타 신호 없으면 verdict_source는 'model'."""
        from detector import _apply_metadata_override
        result = {
            "image": valid_image_path,
            "ai_probability": 0.8,
            "verdict": "AI-generated",
            "model": "test/m",
            "error": None,
        }
        meta = {
            "has_ai_signal": False,
            "decisive": False,
            "signals": [],
            "source": None,
            "checked": True,
        }
        updated = _apply_metadata_override(result, meta)
        assert updated["verdict_source"] == "model"
        assert updated["verdict"] == "AI-generated"


# ──────────────────────────────────────────────
# 항목 D: 출력 새니타이즈 테스트
# ──────────────────────────────────────────────
class TestSanitizeForTerminal:
    """_sanitize_for_terminal: 사람용 출력에서 제어문자/ANSI 이스케이프 제거."""

    def test_ansi_escape_removed(self):
        """ANSI 이스케이프 시퀀스가 제거되어야 한다."""
        from detector import _sanitize_for_terminal
        dirty = "\x1b[2J\x1b]0;pwn\x07 clean text"
        result = _sanitize_for_terminal(dirty)
        assert "\x1b" not in result
        assert "\x07" not in result
        assert "clean text" in result

    def test_c0_control_chars_removed(self):
        """C0 제어문자(0x00-0x1F)가 제거되어야 한다 (탭/뉴라인 제외)."""
        from detector import _sanitize_for_terminal
        dirty = "hello\x00\x01\x02world\x1aend"
        result = _sanitize_for_terminal(dirty)
        assert "\x00" not in result
        assert "\x01" not in result
        assert "\x1a" not in result

    def test_length_truncated(self):
        """120자 상한이 적용되어야 한다."""
        from detector import _sanitize_for_terminal
        long_text = "a" * 200
        result = _sanitize_for_terminal(long_text)
        assert len(result) <= 120

    def test_normal_text_preserved(self):
        """정상 텍스트는 그대로 보존되어야 한다."""
        from detector import _sanitize_for_terminal
        normal = "Stable Diffusion parameters: Steps: 20"
        result = _sanitize_for_terminal(normal)
        assert result == normal

    def test_format_result_sanitizes_signals(self, tmp_path):
        """format_result의 사람용 출력에서 ANSI 이스케이프가 제거되어야 한다."""
        from detector import format_result
        result = {
            "image": "test.jpg",
            "ai_probability": 0.9,
            "verdict": "AI-generated",
            "model": "test/model",
            "error": None,
            "models": [],
            "metadata": {
                "has_ai_signal": True,
                "decisive": True,
                "signals": ["\x1b[2J\x1b]0;pwn\x07 injected signal"],
                "source": "metadata",
                "checked": True,
            },
        }
        output = format_result(result, json_mode=False)
        assert "\x1b" not in output
        assert "\x07" not in output


# ──────────────────────────────────────────────
# 항목 E: pipeline 재사용 테스트
# ──────────────────────────────────────────────
class TestPipelineReuse:
    """다중 이미지 처리 시 pipeline이 모델당 1회만 생성되는지 검증."""

    def test_pipeline_created_once_per_model(self, tmp_path):
        """이미지 3장 × 모델 2개일 때 pipeline 생성 횟수는 2회(모델당 1회)."""
        from PIL import Image as PILImage
        from detector import analyze_images_batch

        # 이미지 3장 생성
        image_paths = []
        for i in range(3):
            p = tmp_path / f"img{i}.png"
            PILImage.new("RGB", (16, 16), color=(i * 80, 0, 0)).save(str(p))
            image_paths.append(str(p))

        creation_count = {"count": 0}

        def counting_pipeline(*args, **kwargs):
            creation_count["count"] += 1
            def infer(image):
                return [{"label": "artificial", "score": 0.7}, {"label": "human", "score": 0.3}]
            return infer

        models = ["model/A", "model/B"]
        analyze_images_batch(
            image_paths=image_paths,
            pipeline_fn=counting_pipeline,
            model_ids=models,
            threshold=0.5,
        )
        # 모델 2개, 이미지가 3장이어도 pipeline 생성은 2회여야 함
        assert creation_count["count"] == 2, (
            f"pipeline 생성 횟수 {creation_count['count']}회 (기대: 2회)"
        )
