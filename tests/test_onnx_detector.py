"""
onnx_detector.py 단위 테스트 및 통합 테스트.

- onnxruntime.InferenceSession을 mock으로 교체 (실제 변환/추론 없음)
- 임시 onnx_models 디렉토리 + meta.json + 더미 .onnx 파일 사용
"""
import json
import os
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image


# ─── 헬퍼: 임시 모델 디렉토리 생성 ────────────────────────────────────────────

def _make_model_dir(base_dir: str, model_id: str, meta: dict = None) -> str:
    """
    onnx_models/<sanitized>/ 디렉토리를 만들고 meta.json + 더미 .onnx 저장.
    """
    from onnx_detector import _sanitize_model_id
    sanitized = _sanitize_model_id(model_id)
    model_dir = os.path.join(base_dir, sanitized)
    os.makedirs(model_dir, exist_ok=True)

    if meta is None:
        meta = {
            "image_size": 224,
            "image_mean": [0.5, 0.5, 0.5],
            "image_std": [0.5, 0.5, 0.5],
            "do_normalize": True,
            "do_rescale": True,
            "rescale_factor": 0.00392156862745098,  # 1/255
            "resample": 2,  # PIL BILINEAR
            "id2label": {"0": "artificial", "1": "human"},
        }
    with open(os.path.join(model_dir, "meta.json"), "w") as f:
        json.dump(meta, f)
    # 더미 .onnx 파일 (실제 세션은 mock)
    with open(os.path.join(model_dir, "model_quantized.onnx"), "wb") as f:
        f.write(b"dummy_onnx")
    return model_dir


def _make_rgb_image(width=64, height=64, color=(128, 64, 32)) -> Image.Image:
    return Image.new("RGB", (width, height), color=color)


def _make_mock_session(logits: list):
    """고정 logits를 반환하는 mock InferenceSession."""
    session = MagicMock()
    session.get_inputs.return_value = [MagicMock(name="pixel_values")]
    session.run.return_value = [np.array([logits], dtype=np.float32)]
    return session


# ─── _sanitize_model_id ────────────────────────────────────────────────────────

class TestSanitizeModelId:
    def test_slash_replaced_with_underscore(self):
        from onnx_detector import _sanitize_model_id
        assert _sanitize_model_id("Organika/sdxl-detector") == "Organika__sdxl-detector"

    def test_no_slash_unchanged(self):
        from onnx_detector import _sanitize_model_id
        assert _sanitize_model_id("plain-model") == "plain-model"

    def test_multiple_slashes(self):
        from onnx_detector import _sanitize_model_id
        assert _sanitize_model_id("a/b/c") == "a__b__c"

    def test_empty_string(self):
        from onnx_detector import _sanitize_model_id
        assert _sanitize_model_id("") == ""


# ─── _softmax ─────────────────────────────────────────────────────────────────

class TestSoftmax:
    def test_output_sums_to_one(self):
        from onnx_detector import _softmax
        logits = np.array([1.0, 2.0, 0.5])
        result = _softmax(logits)
        assert abs(sum(result) - 1.0) < 1e-6

    def test_monotonicity(self):
        """더 큰 logit → 더 큰 확률."""
        from onnx_detector import _softmax
        logits = np.array([1.0, 3.0, 2.0])
        result = _softmax(logits)
        assert result[1] > result[2] > result[0]

    def test_numerical_stability_large_logit(self):
        """매우 큰 값에서 inf/nan 없이 계산되어야 한다."""
        from onnx_detector import _softmax
        logits = np.array([1000.0, 999.0, 998.0])
        result = _softmax(logits)
        assert not any(np.isnan(result))
        assert not any(np.isinf(result))
        assert abs(sum(result) - 1.0) < 1e-6

    def test_single_element(self):
        from onnx_detector import _softmax
        result = _softmax(np.array([5.0]))
        assert abs(result[0] - 1.0) < 1e-6

    def test_uniform_logits(self):
        """같은 값이면 모두 동일한 확률."""
        from onnx_detector import _softmax
        logits = np.array([2.0, 2.0, 2.0])
        result = _softmax(logits)
        assert all(abs(r - 1 / 3) < 1e-6 for r in result)


# ─── _build_label_scores ──────────────────────────────────────────────────────

class TestBuildLabelScores:
    def test_basic_two_class(self):
        from onnx_detector import _build_label_scores
        id2label = {"0": "artificial", "1": "human"}
        probs = np.array([0.7, 0.3])
        result = _build_label_scores(id2label, probs)
        assert len(result) == 2
        assert {"label": "artificial", "score": pytest.approx(0.7)} in result
        assert {"label": "human", "score": pytest.approx(0.3)} in result

    def test_preserves_original_index_order(self):
        """score 내림차순 정렬 강제 없음 — 원래 인덱스 순서 유지."""
        from onnx_detector import _build_label_scores
        id2label = {"0": "a", "1": "b", "2": "c"}
        probs = np.array([0.1, 0.8, 0.1])
        result = _build_label_scores(id2label, probs)
        assert result[0]["label"] == "a"
        assert result[1]["label"] == "b"
        assert result[2]["label"] == "c"

    def test_scores_are_python_floats(self):
        from onnx_detector import _build_label_scores
        id2label = {"0": "x"}
        probs = np.array([1.0])
        result = _build_label_scores(id2label, probs)
        assert isinstance(result[0]["score"], float)

    def test_compatible_with_extract_ai_probability(self):
        """기존 extract_ai_probability가 올바른 값을 추출해야 한다."""
        from onnx_detector import _build_label_scores
        from detector import extract_ai_probability
        id2label = {"0": "artificial", "1": "human"}
        probs = np.array([0.85, 0.15])
        label_scores = _build_label_scores(id2label, probs)
        ai_prob = extract_ai_probability(label_scores)
        assert abs(ai_prob - 0.85) < 1e-6


# ─── preprocess_image ─────────────────────────────────────────────────────────

class TestPreprocessImage:
    def _default_meta(self):
        return {
            "image_size": 224,
            "image_mean": [0.5, 0.5, 0.5],
            "image_std": [0.5, 0.5, 0.5],
            "do_normalize": True,
            "do_rescale": True,
            "rescale_factor": 1 / 255.0,
            "resample": 2,
            "id2label": {"0": "artificial", "1": "human"},
        }

    def test_output_shape(self):
        from onnx_detector import preprocess_image
        meta = self._default_meta()
        img = _make_rgb_image(64, 64)
        result = preprocess_image(img, meta)
        assert result.shape == (1, 3, 224, 224)

    def test_output_dtype_float32(self):
        from onnx_detector import preprocess_image
        meta = self._default_meta()
        img = _make_rgb_image(64, 64)
        result = preprocess_image(img, meta)
        assert result.dtype == np.float32

    def test_grayscale_converted_to_rgb(self):
        """L 모드 이미지도 정상 처리 (shape (1,3,H,W))."""
        from onnx_detector import preprocess_image
        meta = self._default_meta()
        img = Image.new("L", (64, 64), 128)
        result = preprocess_image(img, meta)
        assert result.shape == (1, 3, 224, 224)

    def test_rgba_converted_to_rgb(self):
        """RGBA 이미지도 정상 처리."""
        from onnx_detector import preprocess_image
        meta = self._default_meta()
        img = Image.new("RGBA", (64, 64), (128, 64, 32, 255))
        result = preprocess_image(img, meta)
        assert result.shape == (1, 3, 224, 224)

    def test_different_input_size(self):
        """입력 크기와 무관하게 meta의 image_size로 resize."""
        from onnx_detector import preprocess_image
        meta = self._default_meta()
        meta["image_size"] = 128
        img = _make_rgb_image(300, 200)
        result = preprocess_image(img, meta)
        assert result.shape == (1, 3, 128, 128)

    def test_normalization_range(self):
        """정규화 후 값이 mean=0.5, std=0.5 기준으로 [-1, 1] 근방."""
        from onnx_detector import preprocess_image
        meta = self._default_meta()
        img = _make_rgb_image(224, 224, color=(0, 0, 0))
        result = preprocess_image(img, meta)
        # pixel=0 → rescale=0.0 → normalize=(0.0 - 0.5) / 0.5 = -1.0
        assert abs(result[0, 0, 0, 0] - (-1.0)) < 1e-5

    def test_normalization_max_value(self):
        """pixel=255 → rescale=1.0 → normalize=(1.0-0.5)/0.5 = 1.0"""
        from onnx_detector import preprocess_image
        meta = self._default_meta()
        img = _make_rgb_image(224, 224, color=(255, 255, 255))
        result = preprocess_image(img, meta)
        assert abs(result[0, 0, 0, 0] - 1.0) < 1e-5

    def test_no_normalize(self):
        """do_normalize=False면 rescale만 적용."""
        from onnx_detector import preprocess_image
        meta = self._default_meta()
        meta["do_normalize"] = False
        img = _make_rgb_image(224, 224, color=(255, 255, 255))
        result = preprocess_image(img, meta)
        # 255 * (1/255) = 1.0 (normalize 없음)
        assert abs(result[0, 0, 0, 0] - 1.0) < 1e-5

    def test_no_rescale(self):
        """do_rescale=False면 픽셀값 그대로 (0~255)."""
        from onnx_detector import preprocess_image
        meta = self._default_meta()
        meta["do_rescale"] = False
        meta["do_normalize"] = False
        img = _make_rgb_image(224, 224, color=(128, 128, 128))
        result = preprocess_image(img, meta)
        assert abs(result[0, 0, 0, 0] - 128.0) < 1e-5


# ─── get_onnx_pipeline_fn ─────────────────────────────────────────────────────

class TestGetOnnxPipelineFn:
    def test_missing_model_dir_raises_file_not_found(self, tmp_path):
        from onnx_detector import get_onnx_pipeline_fn
        pipeline_fn = get_onnx_pipeline_fn(str(tmp_path))
        with pytest.raises(FileNotFoundError, match="ONNX 모델 미변환"):
            pipeline_fn("image-classification", model="nonexistent/model")

    def test_infer_returns_label_score_list(self, tmp_path):
        """모델 디렉토리 있을 때 infer가 [{label,score},...] 반환."""
        _make_model_dir(str(tmp_path), "Organika/sdxl-detector")

        mock_session = _make_mock_session([1.5, -0.5])  # artificial 높음
        with patch("onnxruntime.InferenceSession", return_value=mock_session):
            from onnx_detector import get_onnx_pipeline_fn
            pipeline_fn = get_onnx_pipeline_fn(str(tmp_path))
            infer = pipeline_fn("image-classification", model="Organika/sdxl-detector")
            result = infer(_make_rgb_image())

        assert isinstance(result, list)
        assert len(result) == 2
        for item in result:
            assert "label" in item
            assert "score" in item
            assert isinstance(item["score"], float)

    def test_infer_ai_label_detected(self, tmp_path):
        """logits[0] 높으면 artificial 확률 > 0.5."""
        _make_model_dir(str(tmp_path), "Organika/sdxl-detector")

        mock_session = _make_mock_session([5.0, -5.0])
        with patch("onnxruntime.InferenceSession", return_value=mock_session):
            from onnx_detector import get_onnx_pipeline_fn, _clear_session_cache
            from detector import extract_ai_probability
            _clear_session_cache()
            pipeline_fn = get_onnx_pipeline_fn(str(tmp_path))
            infer = pipeline_fn("image-classification", model="Organika/sdxl-detector")
            result = infer(_make_rgb_image())
            ai_prob = extract_ai_probability(result)
            assert ai_prob > 0.99

    def test_session_cached_same_model(self, tmp_path):
        """같은 model_id로 2회 호출 → InferenceSession은 1회만 생성."""
        _make_model_dir(str(tmp_path), "Organika/sdxl-detector")
        mock_session = _make_mock_session([1.0, 0.0])

        with patch("onnxruntime.InferenceSession", return_value=mock_session) as mock_cls:
            from onnx_detector import get_onnx_pipeline_fn, _clear_session_cache
            _clear_session_cache()  # 테스트 격리
            pipeline_fn = get_onnx_pipeline_fn(str(tmp_path))
            pipeline_fn("image-classification", model="Organika/sdxl-detector")
            pipeline_fn("image-classification", model="Organika/sdxl-detector")
            assert mock_cls.call_count == 1

    def test_session_not_cached_different_model(self, tmp_path):
        """다른 model_id → 세션 각각 생성."""
        _make_model_dir(str(tmp_path), "model/a")
        _make_model_dir(str(tmp_path), "model/b")
        mock_session = _make_mock_session([1.0, 0.0])

        with patch("onnxruntime.InferenceSession", return_value=mock_session) as mock_cls:
            from onnx_detector import get_onnx_pipeline_fn, _clear_session_cache
            _clear_session_cache()
            pipeline_fn = get_onnx_pipeline_fn(str(tmp_path))
            pipeline_fn("image-classification", model="model/a")
            pipeline_fn("image-classification", model="model/b")
            assert mock_cls.call_count == 2


# ─── 통합: analyze_images_batch와 연동 ───────────────────────────────────────

class TestOnnxIntegrationWithDetector:
    def _make_pipeline_fn(self, tmp_path, logits, model_id="Organika/sdxl-detector"):
        _make_model_dir(str(tmp_path), model_id)
        mock_session = _make_mock_session(logits)
        return mock_session

    def test_analyze_images_batch_schema(self, tmp_path, valid_image_path):
        """onnx pipeline_fn → analyze_images_batch → 결과 스키마 검증."""
        _make_model_dir(str(tmp_path), "Organika/sdxl-detector")
        mock_session = _make_mock_session([5.0, -5.0])

        with patch("onnxruntime.InferenceSession", return_value=mock_session):
            from onnx_detector import get_onnx_pipeline_fn, _clear_session_cache
            from detector import analyze_images_batch
            _clear_session_cache()
            pipeline_fn = get_onnx_pipeline_fn(str(tmp_path))
            results = analyze_images_batch(
                image_paths=[valid_image_path],
                pipeline_fn=pipeline_fn,
                model_ids=["Organika/sdxl-detector"],
                threshold=0.5,
            )

        assert len(results) == 1
        r = results[0]
        assert "ai_probability" in r
        assert "verdict" in r
        assert "model" in r
        assert "error" in r
        assert "models" in r
        assert "metadata" in r
        assert r["error"] is None
        assert r["ai_probability"] is not None

    def test_analyze_images_batch_ai_verdict(self, tmp_path, valid_image_path):
        """logits[0]이 높으면 AI-generated 판정."""
        _make_model_dir(str(tmp_path), "Organika/sdxl-detector")
        mock_session = _make_mock_session([5.0, -5.0])

        with patch("onnxruntime.InferenceSession", return_value=mock_session):
            from onnx_detector import get_onnx_pipeline_fn, _clear_session_cache
            from detector import analyze_images_batch
            _clear_session_cache()
            pipeline_fn = get_onnx_pipeline_fn(str(tmp_path))
            results = analyze_images_batch(
                image_paths=[valid_image_path],
                pipeline_fn=pipeline_fn,
                model_ids=["Organika/sdxl-detector"],
                threshold=0.5,
            )

        assert results[0]["verdict"] == "AI-generated"

    def test_analyze_images_batch_real_verdict(self, tmp_path, valid_image_path):
        """logits[1]이 높으면 Real 판정."""
        _make_model_dir(str(tmp_path), "Organika/sdxl-detector")
        mock_session = _make_mock_session([-5.0, 5.0])

        with patch("onnxruntime.InferenceSession", return_value=mock_session):
            from onnx_detector import get_onnx_pipeline_fn, _clear_session_cache
            from detector import analyze_images_batch
            _clear_session_cache()
            pipeline_fn = get_onnx_pipeline_fn(str(tmp_path))
            results = analyze_images_batch(
                image_paths=[valid_image_path],
                pipeline_fn=pipeline_fn,
                model_ids=["Organika/sdxl-detector"],
                threshold=0.5,
            )

        assert results[0]["verdict"] == "Real"

    def test_ensemble_onnx_schema(self, tmp_path, valid_image_path):
        """앙상블(2모델) onnx pipeline_fn 결과 스키마."""
        _make_model_dir(str(tmp_path), "model/a")
        _make_model_dir(str(tmp_path), "model/b")
        mock_session = _make_mock_session([3.0, -3.0])

        with patch("onnxruntime.InferenceSession", return_value=mock_session):
            from onnx_detector import get_onnx_pipeline_fn, _clear_session_cache
            from detector import analyze_images_batch
            _clear_session_cache()
            pipeline_fn = get_onnx_pipeline_fn(str(tmp_path))
            results = analyze_images_batch(
                image_paths=[valid_image_path],
                pipeline_fn=pipeline_fn,
                model_ids=["model/a", "model/b"],
                threshold=0.5,
            )

        r = results[0]
        assert r["model"] == "ensemble(2 models)"
        assert len(r["models"]) == 2


# ─── CLI --backend onnx ───────────────────────────────────────────────────────

class TestCLIBackendOption:
    def test_help_includes_backend(self):
        """--help에 --backend 옵션이 포함되어야 한다."""
        import subprocess
        result = subprocess.run(
            ["/Users/jeonhaedong/opt/anaconda3/bin/python3", "detect.py", "--help"],
            cwd="/Users/jeonhaedong/Desktop/ai-image-detecter",
            capture_output=True,
            text=True,
        )
        assert "--backend" in result.stdout

    def test_backend_onnx_missing_model_graceful_error(self, tmp_path, valid_image_path):
        """--backend onnx + 모델 미변환 → graceful error (exit 1, error 필드)."""
        from detect import main
        import sys
        from io import StringIO

        with patch("detect._get_pipeline_fn") as mock_get:
            from onnx_detector import get_onnx_pipeline_fn, _clear_session_cache
            _clear_session_cache()
            mock_get.return_value = get_onnx_pipeline_fn(str(tmp_path))

            with pytest.raises(SystemExit) as exc_info:
                main([valid_image_path, "--backend", "onnx",
                      "--onnx-models-dir", str(tmp_path), "--json"])
            assert exc_info.value.code == 1

    def test_backend_onnx_with_mock_model(self, tmp_path, valid_image_path):
        """--backend onnx + mock 모델 → 정상 판정."""
        _make_model_dir(str(tmp_path), "Organika/sdxl-detector")
        mock_session = _make_mock_session([5.0, -5.0])

        import json as json_module

        with patch("onnxruntime.InferenceSession", return_value=mock_session):
            from onnx_detector import _clear_session_cache
            _clear_session_cache()
            with patch("detect._get_pipeline_fn") as mock_get:
                from onnx_detector import get_onnx_pipeline_fn
                mock_get.return_value = get_onnx_pipeline_fn(str(tmp_path))
                import io
                from contextlib import redirect_stdout
                buf = io.StringIO()
                with redirect_stdout(buf):
                    with pytest.raises(SystemExit) as exc_info:
                        from detect import main
                        main([valid_image_path, "--backend", "onnx",
                              "--onnx-models-dir", str(tmp_path), "--json"])
                assert exc_info.value.code == 0
                output = buf.getvalue()
                results = json_module.loads(output)
                assert results[0]["verdict"] in ("AI-generated", "Real")

    def test_backend_default_is_onnx(self, tmp_path, valid_image_path):
        """`--backend` 미지정 시 onnx가 기본 백엔드여야 한다."""
        from detect import _build_parser
        args = _build_parser().parse_args([valid_image_path])
        assert args.backend == "onnx", f"expected 'onnx', got {args.backend!r}"

        with patch("detect._get_pipeline_fn") as mock_get:
            mock_pipeline = MagicMock()
            mock_pipeline.return_value = lambda img: [
                {"label": "artificial", "score": 0.8},
                {"label": "human", "score": 0.2},
            ]

            def pipeline_fn(task, model=""):
                return mock_pipeline(task, model=model)

            mock_get.return_value = pipeline_fn
            import io
            from contextlib import redirect_stdout
            buf = io.StringIO()
            with redirect_stdout(buf):
                with pytest.raises(SystemExit) as exc_info:
                    from detect import main
                    main([valid_image_path, "--json"])
            assert exc_info.value.code == 0


# ─── 변환 스크립트 단위 테스트 ───────────────────────────────────────────────

class TestConvertToOnnxScript:
    def test_help_works(self):
        """--help 실행 시 에러 없이 종료 (torch/optimum 미설치 허용)."""
        import subprocess
        result = subprocess.run(
            ["/Users/jeonhaedong/opt/anaconda3/bin/python3",
             "convert_to_onnx.py", "--help"],
            cwd="/Users/jeonhaedong/Desktop/ai-image-detecter",
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == 0
        assert "--model" in result.stdout

    def test_sanitize_model_id_for_path(self):
        """경로 sanitize 로직 단위 테스트."""
        from onnx_detector import _sanitize_model_id
        assert _sanitize_model_id("Organika/sdxl-detector") == "Organika__sdxl-detector"

    def test_meta_saved_and_loadable(self, tmp_path):
        """meta.json 저장 헬퍼가 올바른 JSON을 만드는지."""
        from convert_to_onnx import _save_meta
        meta = {
            "image_size": 224,
            "image_mean": [0.5, 0.5, 0.5],
            "image_std": [0.5, 0.5, 0.5],
            "do_normalize": True,
            "do_rescale": True,
            "rescale_factor": 1 / 255.0,
            "resample": 2,
            "id2label": {"0": "artificial", "1": "human"},
        }
        _save_meta(str(tmp_path), meta)
        with open(os.path.join(str(tmp_path), "meta.json")) as f:
            loaded = json.load(f)
        assert loaded["id2label"]["0"] == "artificial"
        assert loaded["image_size"] == 224

    def test_parse_args_defaults(self):
        """인자 파싱 기본값 확인."""
        from convert_to_onnx import _build_arg_parser
        parser = _build_arg_parser()
        args = parser.parse_args(["--model", "Organika/sdxl-detector"])
        assert args.model == "Organika/sdxl-detector"
        assert args.output_dir == "onnx_models"
        assert args.no_quantize is False

    def test_parse_args_no_quantize_flag(self):
        from convert_to_onnx import _build_arg_parser
        parser = _build_arg_parser()
        args = parser.parse_args(["--model", "test/model", "--no-quantize"])
        assert args.no_quantize is True

    def test_parse_args_custom_output_dir(self):
        from convert_to_onnx import _build_arg_parser
        parser = _build_arg_parser()
        args = parser.parse_args(["--model", "test/model", "--output-dir", "/tmp/mymodels"])
        assert args.output_dir == "/tmp/mymodels"


# ─── server.py DETECTOR_BACKEND 환경변수 테스트 ───────────────────────────────

class TestServerBackendEnvVar:
    def test_server_uses_onnx_backend_when_env_set(self, tmp_path, monkeypatch):
        """DETECTOR_BACKEND=onnx 환경변수 시 onnx pipeline_fn 반환."""
        monkeypatch.setenv("DETECTOR_BACKEND", "onnx")
        monkeypatch.setenv("ONNX_MODELS_DIR", str(tmp_path))

        # backends.get_pipeline_fn_with_mock이 onnx 경로를 타는지 확인
        with patch("backends.get_backend_pipeline_fn") as mock_backend:
            from onnx_detector import get_onnx_pipeline_fn, _clear_session_cache
            _clear_session_cache()
            mock_backend.return_value = get_onnx_pipeline_fn(str(tmp_path))
            import server
            fn = server._make_pipeline()
            # onnx 경로는 callable이어야 함
            assert callable(fn)

    def test_server_default_backend_is_onnx(self, monkeypatch):
        """DETECTOR_BACKEND 미설정 시 onnx가 기본 백엔드여야 한다."""
        monkeypatch.delenv("DETECTOR_BACKEND", raising=False)
        monkeypatch.delenv("_AI_DETECTOR_MOCK", raising=False)

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

    def test_server_make_pipeline_onnx_callable(self, tmp_path, monkeypatch):
        """DETECTOR_BACKEND=onnx + mock session → pipeline_fn이 callable."""
        monkeypatch.setenv("DETECTOR_BACKEND", "onnx")
        monkeypatch.setenv("ONNX_MODELS_DIR", str(tmp_path))
        monkeypatch.delenv("_AI_DETECTOR_MOCK", raising=False)

        _make_model_dir(str(tmp_path), "Organika/sdxl-detector")
        mock_session = _make_mock_session([1.0, 0.0])

        with patch("onnxruntime.InferenceSession", return_value=mock_session):
            from onnx_detector import _clear_session_cache
            _clear_session_cache()
            import importlib
            import server as srv
            importlib.reload(srv)
            fn = srv._make_pipeline()
            assert callable(fn)


# ─── Path traversal 방어 테스트 ───────────────────────────────────────────────

class TestPathTraversalDefense:
    def test_dotdot_path_rejected(self, tmp_path):
        """'../../etc' 형태의 model_id는 ValueError를 발생시킨다."""
        from onnx_detector import _validate_model_path
        with pytest.raises(ValueError, match="Unsafe model id/path"):
            _validate_model_path("../../etc/passwd", str(tmp_path))

    def test_dotdot_with_slash_rejected(self, tmp_path):
        """'../outside' 형태도 거부된다."""
        from onnx_detector import _validate_model_path
        with pytest.raises(ValueError, match="Unsafe model id/path"):
            _validate_model_path("../outside", str(tmp_path))

    def test_normal_model_id_accepted(self, tmp_path):
        """'Organika/sdxl-detector' 같은 정상 ID는 통과한다."""
        from onnx_detector import _validate_model_path
        # 예외 없이 경로 반환
        result = _validate_model_path("Organika/sdxl-detector", str(tmp_path))
        assert "Organika__sdxl-detector" in result

    def test_nested_normal_id_accepted(self, tmp_path):
        """'org/repo' 형태의 정상 중첩 ID는 통과한다."""
        from onnx_detector import _validate_model_path
        result = _validate_model_path("org/repo", str(tmp_path))
        assert "org__repo" in result

    def test_path_traversal_via_pipeline_fn(self, tmp_path):
        """pipeline_fn에서도 path traversal이 차단된다."""
        from onnx_detector import get_onnx_pipeline_fn
        pipeline_fn = get_onnx_pipeline_fn(str(tmp_path))
        with pytest.raises(ValueError, match="Unsafe model id/path"):
            pipeline_fn("image-classification", model="../../etc/passwd")

    def test_absolute_path_as_model_id_rejected(self, tmp_path):
        """/etc 형태의 절대경로 model_id도 base 밖이면 거부된다."""
        from onnx_detector import _validate_model_path
        # /etc는 tmp_path 밖이므로 거부
        with pytest.raises(ValueError, match="Unsafe model id/path"):
            _validate_model_path("/etc", str(tmp_path))


# ─── meta.json 스키마 검증 테스트 ────────────────────────────────────────────

class TestMetaValidation:
    def _valid_meta(self):
        return {
            "image_size": 224,
            "image_mean": [0.5, 0.5, 0.5],
            "image_std": [0.5, 0.5, 0.5],
            "id2label": {"0": "artificial", "1": "human"},
        }

    def test_valid_meta_passes(self):
        from onnx_detector import _validate_meta
        _validate_meta(self._valid_meta())  # 예외 없어야 함

    def test_missing_image_size_raises(self):
        from onnx_detector import _validate_meta
        meta = self._valid_meta()
        del meta["image_size"]
        with pytest.raises(ValueError, match="image_size"):
            _validate_meta(meta)

    def test_image_size_too_large_raises(self):
        from onnx_detector import _validate_meta
        meta = self._valid_meta()
        meta["image_size"] = 1000000
        with pytest.raises(ValueError, match="out of range"):
            _validate_meta(meta)

    def test_image_size_zero_raises(self):
        from onnx_detector import _validate_meta
        meta = self._valid_meta()
        meta["image_size"] = 0
        with pytest.raises(ValueError, match="out of range"):
            _validate_meta(meta)

    def test_image_size_not_int_raises(self):
        from onnx_detector import _validate_meta
        meta = self._valid_meta()
        meta["image_size"] = "224"
        with pytest.raises(ValueError, match="must be int"):
            _validate_meta(meta)

    def test_missing_image_mean_raises(self):
        from onnx_detector import _validate_meta
        meta = self._valid_meta()
        del meta["image_mean"]
        with pytest.raises(ValueError, match="image_mean"):
            _validate_meta(meta)

    def test_image_std_wrong_length_raises(self):
        from onnx_detector import _validate_meta
        meta = self._valid_meta()
        meta["image_std"] = [0.5, 0.5]  # 길이 2 (잘못됨)
        with pytest.raises(ValueError, match="list of 3"):
            _validate_meta(meta)

    def test_image_mean_non_numeric_raises(self):
        from onnx_detector import _validate_meta
        meta = self._valid_meta()
        meta["image_mean"] = [0.5, "bad", 0.5]
        with pytest.raises(ValueError, match="numeric"):
            _validate_meta(meta)

    def test_missing_id2label_raises(self):
        from onnx_detector import _validate_meta
        meta = self._valid_meta()
        del meta["id2label"]
        with pytest.raises(ValueError, match="id2label"):
            _validate_meta(meta)

    def test_id2label_not_dict_raises(self):
        from onnx_detector import _validate_meta
        meta = self._valid_meta()
        meta["id2label"] = ["artificial", "human"]
        with pytest.raises(ValueError, match="dict"):
            _validate_meta(meta)

    def test_crop_size_out_of_range_raises(self):
        from onnx_detector import _validate_meta
        meta = self._valid_meta()
        meta["crop_size"] = 5000
        with pytest.raises(ValueError, match="out of range"):
            _validate_meta(meta)

    def test_invalid_meta_in_pipeline_fn_raises_gracefully(self, tmp_path):
        """잘못된 meta.json이 있는 모델 디렉토리 → pipeline_fn 호출 시 ValueError."""
        from onnx_detector import _sanitize_model_id
        model_id = "bad/model"
        sanitized = _sanitize_model_id(model_id)
        model_dir = os.path.join(str(tmp_path), sanitized)
        os.makedirs(model_dir, exist_ok=True)
        # image_size 누락 meta.json
        bad_meta = {"image_mean": [0.5, 0.5, 0.5], "image_std": [0.5, 0.5, 0.5], "id2label": {}}
        with open(os.path.join(model_dir, "meta.json"), "w") as f:
            json.dump(bad_meta, f)
        with open(os.path.join(model_dir, "model_quantized.onnx"), "wb") as f:
            f.write(b"dummy")

        from onnx_detector import get_onnx_pipeline_fn, _clear_session_cache
        _clear_session_cache()
        pipeline_fn = get_onnx_pipeline_fn(str(tmp_path))
        mock_session = _make_mock_session([1.0, 0.0])
        with patch("onnxruntime.InferenceSession", return_value=mock_session):
            with pytest.raises(ValueError, match="Invalid meta.json"):
                pipeline_fn("image-classification", model=model_id)


# ─── preprocess_image center-crop (shortest_edge) 테스트 ─────────────────────

class TestPreprocessImageCenterCrop:
    def _shortest_edge_meta(self, image_size=224, crop_size=224):
        return {
            "image_size": image_size,
            "crop_size": crop_size,
            "resize_mode": "shortest_edge",
            "image_mean": [0.5, 0.5, 0.5],
            "image_std": [0.5, 0.5, 0.5],
            "do_normalize": True,
            "do_rescale": True,
            "rescale_factor": 1 / 255.0,
            "resample": 3,
            "id2label": {"0": "a", "1": "b"},
        }

    def test_shortest_edge_output_shape_from_landscape(self):
        """400x300 입력 + shortest_edge=224, crop_size=224 → (1,3,224,224)."""
        from onnx_detector import preprocess_image
        meta = self._shortest_edge_meta(224, 224)
        img = Image.new("RGB", (400, 300), color=(128, 64, 32))
        result = preprocess_image(img, meta)
        assert result.shape == (1, 3, 224, 224)

    def test_shortest_edge_output_shape_from_portrait(self):
        """300x400 입력 + shortest_edge=224, crop_size=224 → (1,3,224,224)."""
        from onnx_detector import preprocess_image
        meta = self._shortest_edge_meta(224, 224)
        img = Image.new("RGB", (300, 400), color=(128, 64, 32))
        result = preprocess_image(img, meta)
        assert result.shape == (1, 3, 224, 224)

    def test_shortest_edge_output_dtype(self):
        from onnx_detector import preprocess_image
        meta = self._shortest_edge_meta(224, 224)
        img = Image.new("RGB", (400, 300))
        result = preprocess_image(img, meta)
        assert result.dtype == np.float32

    def test_center_crop_crops_center(self):
        """center-crop이 실제로 중앙 영역을 자르는지 검증.

        100x100 이미지에서 좌측 절반을 빨간색, 우측 절반을 파란색으로 만들고
        shortest_edge=100, crop_size=50 으로 center-crop하면
        결과 이미지의 첫 픽셀(crop 좌측)은 빨간색과 파란색 경계 근처여야 한다.
        정확히는 (100/2 - 50/2) = 25픽셀부터 시작 → 25번 컬럼이 첫 픽셀.
        25컬럼은 빨간색(0~49), 즉 R 채널이 높아야 한다.
        """
        from onnx_detector import preprocess_image
        meta = self._shortest_edge_meta(image_size=100, crop_size=50)
        meta["do_normalize"] = False
        meta["do_rescale"] = False

        # 100x100 이미지: 왼쪽 절반(x<50)은 (255,0,0) 빨강, 오른쪽 절반은 (0,0,255) 파랑
        img = Image.new("RGB", (100, 100), color=(255, 0, 0))
        import PIL.ImageDraw as ImageDraw
        draw = ImageDraw.Draw(img)
        draw.rectangle([50, 0, 100, 100], fill=(0, 0, 255))

        result = preprocess_image(img, meta)  # shape (1,3,50,50)
        # center-crop: x=25부터 x=75까지 → 첫 픽셀(x=25)은 빨강 영역 안
        # R 채널(index 0): 값이 파랑보다 높아야 함
        r_channel_first_pixel = result[0, 0, 0, 0]  # (batch, R, y, x)
        b_channel_first_pixel = result[0, 2, 0, 0]  # (batch, B, y, x)
        assert r_channel_first_pixel > b_channel_first_pixel, (
            f"crop 시작점이 빨강 영역이어야 하는데 R={r_channel_first_pixel}, B={b_channel_first_pixel}"
        )

    def test_exact_mode_unchanged(self):
        """resize_mode='exact'는 기존과 동일하게 동작해야 한다."""
        from onnx_detector import preprocess_image
        meta = {
            "image_size": 128,
            "resize_mode": "exact",
            "image_mean": [0.5, 0.5, 0.5],
            "image_std": [0.5, 0.5, 0.5],
            "do_normalize": True,
            "do_rescale": True,
            "rescale_factor": 1 / 255.0,
            "resample": 3,
            "id2label": {"0": "a"},
        }
        img = Image.new("RGB", (300, 200), color=(100, 100, 100))
        result = preprocess_image(img, meta)
        assert result.shape == (1, 3, 128, 128)

    def test_default_resize_mode_is_exact(self):
        """resize_mode 키 없을 때 기본 동작(exact)이 유지된다."""
        from onnx_detector import preprocess_image
        meta = {
            "image_size": 64,
            "image_mean": [0.5, 0.5, 0.5],
            "image_std": [0.5, 0.5, 0.5],
            "do_normalize": False,
            "do_rescale": False,
            "id2label": {"0": "a"},
        }
        img = Image.new("RGB", (200, 100), color=(0, 0, 0))
        result = preprocess_image(img, meta)
        assert result.shape == (1, 3, 64, 64)


# ─── convert_to_onnx.py — quantize_arch 인자 파싱 테스트 ────────────────────

class TestQuantizeArchArg:
    def test_default_quantize_arch_is_portable(self):
        """--quantize-arch 미지정 시 기본값이 portable이어야 한다."""
        from convert_to_onnx import _build_arg_parser
        parser = _build_arg_parser()
        args = parser.parse_args(["--model", "test/model"])
        assert args.quantize_arch == "portable"

    def test_quantize_arch_avx2(self):
        from convert_to_onnx import _build_arg_parser
        parser = _build_arg_parser()
        args = parser.parse_args(["--model", "test/model", "--quantize-arch", "avx2"])
        assert args.quantize_arch == "avx2"

    def test_quantize_arch_avx512_vnni(self):
        from convert_to_onnx import _build_arg_parser
        parser = _build_arg_parser()
        args = parser.parse_args(["--model", "test/model", "--quantize-arch", "avx512_vnni"])
        assert args.quantize_arch == "avx512_vnni"

    def test_quantize_arch_arm64(self):
        from convert_to_onnx import _build_arg_parser
        parser = _build_arg_parser()
        args = parser.parse_args(["--model", "test/model", "--quantize-arch", "arm64"])
        assert args.quantize_arch == "arm64"

    def test_invalid_quantize_arch_rejected(self):
        """choices에 없는 값은 argparse가 거부해야 한다."""
        import subprocess
        result = subprocess.run(
            ["/Users/jeonhaedong/opt/anaconda3/bin/python3",
             "convert_to_onnx.py", "--model", "test/model",
             "--quantize-arch", "invalid_cpu"],
            cwd="/Users/jeonhaedong/Desktop/ai-image-detecter",
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode != 0

    def test_build_quantization_config_portable_returns_mode_string(self):
        """portable 선택 시 'portable' 문자열과 설명 반환."""
        from convert_to_onnx import _build_quantization_config
        mode, desc = _build_quantization_config("portable")
        assert mode == "portable"
        assert "portable" in desc.lower() or "arch" in desc.lower() or "INT8" in desc

    def test_extract_meta_includes_resize_mode_exact(self, tmp_path):
        """size=int 형태의 preprocessor_config.json → resize_mode='exact'."""
        from convert_to_onnx import _extract_meta_from_model
        prep = {"size": 224, "image_mean": [0.5, 0.5, 0.5], "image_std": [0.5, 0.5, 0.5]}
        prep_path = os.path.join(str(tmp_path), "preprocessor_config.json")
        with open(prep_path, "w") as f:
            json.dump(prep, f)
        meta = _extract_meta_from_model("test/model", str(tmp_path))
        assert meta["resize_mode"] == "exact"
        assert meta["image_size"] == 224

    def test_extract_meta_shortest_edge_sets_resize_mode(self, tmp_path):
        """size={"shortest_edge": N} → resize_mode='shortest_edge', crop_size 추출."""
        from convert_to_onnx import _extract_meta_from_model
        prep = {
            "size": {"shortest_edge": 256},
            "crop_size": 224,
            "image_mean": [0.485, 0.456, 0.406],
            "image_std": [0.229, 0.224, 0.225],
        }
        prep_path = os.path.join(str(tmp_path), "preprocessor_config.json")
        with open(prep_path, "w") as f:
            json.dump(prep, f)
        meta = _extract_meta_from_model("test/model", str(tmp_path))
        assert meta["resize_mode"] == "shortest_edge"
        assert meta["image_size"] == 256
        assert meta["crop_size"] == 224

    def test_extract_meta_height_width_dict(self, tmp_path):
        """size={"height": H, "width": W} → resize_mode='exact'."""
        from convert_to_onnx import _extract_meta_from_model
        prep = {
            "size": {"height": 224, "width": 224},
            "image_mean": [0.5, 0.5, 0.5],
            "image_std": [0.5, 0.5, 0.5],
        }
        prep_path = os.path.join(str(tmp_path), "preprocessor_config.json")
        with open(prep_path, "w") as f:
            json.dump(prep, f)
        meta = _extract_meta_from_model("test/model", str(tmp_path))
        assert meta["resize_mode"] == "exact"
        assert meta["image_size"] == 224


# ─── backends.py 통합 테스트 ─────────────────────────────────────────────────

class TestBackendsModule:
    def test_get_backend_pipeline_fn_onnx(self, tmp_path):
        """backend='onnx' → get_onnx_pipeline_fn 경로."""
        from backends import get_backend_pipeline_fn
        from onnx_detector import _clear_session_cache
        _clear_session_cache()
        fn = get_backend_pipeline_fn("onnx", str(tmp_path))
        assert callable(fn)

    def test_get_backend_pipeline_fn_case_insensitive(self, tmp_path):
        """'ONNX', ' onnx ' 등 대소문자·공백 혼합도 처리된다."""
        from backends import get_backend_pipeline_fn
        from onnx_detector import _clear_session_cache
        _clear_session_cache()
        fn = get_backend_pipeline_fn("  ONNX  ", str(tmp_path))
        assert callable(fn)

    def test_get_backend_pipeline_fn_unknown_raises(self, tmp_path):
        """알 수 없는 backend → ValueError."""
        from backends import get_backend_pipeline_fn
        with pytest.raises(ValueError, match="알 수 없는 backend"):
            get_backend_pipeline_fn("tensorrt", str(tmp_path))

    def test_get_pipeline_fn_with_mock_returns_mock(self, monkeypatch, tmp_path):
        """_AI_DETECTOR_MOCK=1 → mock pipeline 반환."""
        monkeypatch.setenv("_AI_DETECTOR_MOCK", "1")
        monkeypatch.setenv("PYTEST_CURRENT_TEST", "1")
        from backends import get_pipeline_fn_with_mock
        fn = get_pipeline_fn_with_mock("torch", str(tmp_path))
        assert callable(fn)
        infer = fn("image-classification", model="any/model")
        result = infer(Image.new("RGB", (64, 64)))
        assert result[0]["label"] == "artificial"
        assert abs(result[0]["score"] - 0.73) < 1e-6


# ─── task 파라미터 방어 테스트 ───────────────────────────────────────────────

class TestTaskParameterDefense:
    def test_non_image_classification_task_raises(self, tmp_path):
        """task='text-classification' → ValueError."""
        _make_model_dir(str(tmp_path), "Organika/sdxl-detector")
        from onnx_detector import get_onnx_pipeline_fn, _clear_session_cache
        _clear_session_cache()
        pipeline_fn = get_onnx_pipeline_fn(str(tmp_path))
        with pytest.raises(ValueError, match="image-classification"):
            pipeline_fn("text-classification", model="Organika/sdxl-detector")

    def test_empty_task_raises(self, tmp_path):
        """task='' → ValueError."""
        _make_model_dir(str(tmp_path), "Organika/sdxl-detector")
        from onnx_detector import get_onnx_pipeline_fn, _clear_session_cache
        _clear_session_cache()
        pipeline_fn = get_onnx_pipeline_fn(str(tmp_path))
        with pytest.raises(ValueError, match="image-classification"):
            pipeline_fn("", model="Organika/sdxl-detector")

    def test_image_classification_task_accepted(self, tmp_path):
        """task='image-classification' → 정상 동작."""
        _make_model_dir(str(tmp_path), "Organika/sdxl-detector")
        mock_session = _make_mock_session([1.0, 0.0])
        with patch("onnxruntime.InferenceSession", return_value=mock_session):
            from onnx_detector import get_onnx_pipeline_fn, _clear_session_cache
            _clear_session_cache()
            pipeline_fn = get_onnx_pipeline_fn(str(tmp_path))
            infer = pipeline_fn("image-classification", model="Organika/sdxl-detector")
            assert callable(infer)
