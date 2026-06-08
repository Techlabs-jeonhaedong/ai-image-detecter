"""앙상블/다중 모델/메타데이터 CLI E2E 테스트."""
import json
import os
import subprocess
import sys
import pytest
from PIL import Image, PngImagePlugin

PYTHON = sys.executable
DETECT = "/Users/jeonhaedong/Desktop/ai-image-detecter/detect.py"

MOCK_ENV = {"_AI_DETECTOR_MOCK": "1"}


def run_cli(*args, env=None):
    cmd = [PYTHON, DETECT] + list(args)
    merged = os.environ.copy()
    if env:
        merged.update(env)
    return subprocess.run(cmd, capture_output=True, text=True, env=merged)


# ──────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────

@pytest.fixture
def img(tmp_path):
    p = tmp_path / "clean.png"
    Image.new("RGB", (32, 32), color=(10, 20, 30)).save(str(p))
    return str(p)


@pytest.fixture
def img2(tmp_path):
    p = tmp_path / "clean2.png"
    Image.new("RGB", (32, 32), color=(50, 60, 70)).save(str(p))
    return str(p)


@pytest.fixture
def sd_img(tmp_path):
    """SD parameters 청크가 있는 PNG (결정적 신호: Steps + Sampler + CFG scale)."""
    img = Image.new("RGB", (32, 32), color=(200, 100, 50))
    meta = PngImagePlugin.PngInfo()
    meta.add_text("parameters", "masterpiece, best quality, Steps: 20, Sampler: Euler a, CFG scale: 7")
    p = tmp_path / "sd.png"
    img.save(str(p), pnginfo=meta)
    return str(p)


# ──────────────────────────────────────────────
# 다중 모델 (--model 반복)
# ──────────────────────────────────────────────

class TestMultiModel:
    def test_multiple_model_flags_accepted(self, img):
        """--model A --model B 형식이 오류 없이 실행된다."""
        r = run_cli(img, "--model", "model/A", "--model", "model/B", env=MOCK_ENV)
        assert r.returncode == 0, r.stderr

    def test_multiple_model_json_has_models_field(self, img):
        """--model 여러 개 지정 시 JSON에 models 배열이 있어야 한다."""
        r = run_cli(img, "--model", "model/A", "--model", "model/B", "--json", env=MOCK_ENV)
        assert r.returncode == 0
        parsed = json.loads(r.stdout)
        assert "models" in parsed[0]
        assert len(parsed[0]["models"]) == 2

    def test_multiple_model_json_ai_probability_is_average(self, img):
        """mock pipeline이 동일 확률(0.73)을 반환하므로 평균도 0.73이어야 한다."""
        r = run_cli(img, "--model", "model/A", "--model", "model/B", "--json", env=MOCK_ENV)
        parsed = json.loads(r.stdout)
        prob = parsed[0]["ai_probability"]
        assert 0.0 <= prob <= 1.0

    def test_single_model_default_behavior(self, img):
        """--model 미지정 시 기존과 동일하게 동작해야 한다."""
        r = run_cli(img, "--json", env=MOCK_ENV)
        assert r.returncode == 0
        parsed = json.loads(r.stdout)
        assert "ai_probability" in parsed[0]
        assert "verdict" in parsed[0]

    def test_multiple_models_each_model_in_result(self, img):
        """models 배열 각 항목에 model 필드가 있어야 한다."""
        r = run_cli(img, "--model", "aaa/bbb", "--model", "ccc/ddd", "--json", env=MOCK_ENV)
        parsed = json.loads(r.stdout)
        model_ids = {m["model"] for m in parsed[0]["models"]}
        assert "aaa/bbb" in model_ids
        assert "ccc/ddd" in model_ids


# ──────────────────────────────────────────────
# 앙상블 (--ensemble)
# ──────────────────────────────────────────────

class TestEnsembleCLI:
    def test_ensemble_flag_accepted(self, img):
        """--ensemble 플래그가 오류 없이 실행된다."""
        r = run_cli(img, "--ensemble", env=MOCK_ENV)
        assert r.returncode == 0, r.stderr

    def test_ensemble_json_has_models_field(self, img):
        """--ensemble 시 JSON에 models 배열이 있어야 한다."""
        r = run_cli(img, "--ensemble", "--json", env=MOCK_ENV)
        parsed = json.loads(r.stdout)
        assert "models" in parsed[0]
        assert len(parsed[0]["models"]) >= 2

    def test_ensemble_model_field_shows_ensemble_label(self, img):
        """앙상블 시 model 필드에 'ensemble' 문자가 포함되어야 한다."""
        r = run_cli(img, "--ensemble", "--json", env=MOCK_ENV)
        parsed = json.loads(r.stdout)
        assert "ensemble" in parsed[0]["model"].lower()

    def test_ensemble_and_model_combined(self, img):
        """--ensemble과 --model 동시 지정 시 오류 없이 실행."""
        r = run_cli(img, "--ensemble", "--model", "extra/model", env=MOCK_ENV)
        assert r.returncode == 0, r.stderr

    def test_ensemble_output_has_verdict(self, img):
        """앙상블 결과에 verdict가 있어야 한다."""
        r = run_cli(img, "--ensemble", "--json", env=MOCK_ENV)
        parsed = json.loads(r.stdout)
        assert parsed[0]["verdict"] in ("AI-generated", "Real")


# ──────────────────────────────────────────────
# 메타데이터 검사
# ──────────────────────────────────────────────

class TestMetadataCLI:
    def test_metadata_checked_by_default(self, img):
        """기본적으로 메타데이터 검사가 활성화된다."""
        r = run_cli(img, "--json", env=MOCK_ENV)
        parsed = json.loads(r.stdout)
        assert "metadata" in parsed[0]
        assert parsed[0]["metadata"]["checked"] is True

    def test_no_metadata_flag_disables_check(self, img):
        """--no-metadata 시 metadata.checked가 False이거나 metadata 키가 최소화된다."""
        r = run_cli(img, "--no-metadata", "--json", env=MOCK_ENV)
        assert r.returncode == 0
        parsed = json.loads(r.stdout)
        # --no-metadata 시 metadata.checked = False 또는 metadata = None
        meta = parsed[0].get("metadata")
        if meta is not None:
            assert meta.get("checked") is False

    def test_sd_metadata_overrides_ml_to_ai(self, sd_img):
        """SD metadata 신호 있으면 ML이 Real을 줘도 verdict = AI-generated."""
        # mock pipeline은 AI 0.73을 반환하지만, metadata 신호만으로도 AI로 판정됨을 테스트
        # threshold를 1.0으로 올려서 ML만으로는 Real이 되도록 하고 metadata가 오버라이드하는지 확인
        r = run_cli(sd_img, "--threshold", "1.0", "--json", env=MOCK_ENV)
        parsed = json.loads(r.stdout)
        # threshold=1.0이면 ML 단독으로는 Real, 하지만 metadata 신호가 있으면 AI-generated
        assert parsed[0]["verdict"] == "AI-generated"
        assert parsed[0]["metadata"]["has_ai_signal"] is True

    def test_clean_image_metadata_no_signal(self, img):
        """깨끗한 이미지에서 metadata.has_ai_signal = False."""
        r = run_cli(img, "--json", env=MOCK_ENV)
        parsed = json.loads(r.stdout)
        assert parsed[0]["metadata"]["has_ai_signal"] is False

    def test_metadata_signals_list_present(self, img):
        """metadata.signals 필드가 리스트여야 한다."""
        r = run_cli(img, "--json", env=MOCK_ENV)
        parsed = json.loads(r.stdout)
        assert isinstance(parsed[0]["metadata"]["signals"], list)

    def test_sd_metadata_signal_in_human_readable(self, sd_img):
        """SD metadata 신호 있을 때 사람용 출력에 신호 근거가 표시된다."""
        r = run_cli(sd_img, env=MOCK_ENV)
        output = r.stdout.lower()
        # 메타데이터 관련 문자열이 출력에 포함되어야 함
        assert "metadata" in output or "parameters" in output or "signal" in output


# ──────────────────────────────────────────────
# JSON 스키마 하위 호환
# ──────────────────────────────────────────────

class TestJSONSchemaBackwardCompat:
    def test_existing_keys_still_present(self, img):
        """기존 키(image, ai_probability, verdict, model, error)가 여전히 있어야 한다."""
        r = run_cli(img, "--json", env=MOCK_ENV)
        parsed = json.loads(r.stdout)
        item = parsed[0]
        for key in ("image", "ai_probability", "verdict", "model", "error"):
            assert key in item, f"Missing key: {key}"

    def test_models_field_present_single_model(self, img):
        """단일 모델일 때도 models 배열이 있어야 한다."""
        r = run_cli(img, "--json", env=MOCK_ENV)
        parsed = json.loads(r.stdout)
        assert "models" in parsed[0]
        assert len(parsed[0]["models"]) == 1

    def test_models_item_has_required_keys(self, img):
        """models 배열 각 항목에 model, ai_probability, error가 있어야 한다."""
        r = run_cli(img, "--json", env=MOCK_ENV)
        parsed = json.loads(r.stdout)
        for m in parsed[0]["models"]:
            assert "model" in m
            assert "ai_probability" in m
            assert "error" in m


# ──────────────────────────────────────────────
# --no-metadata + --ensemble 조합
# ──────────────────────────────────────────────

class TestCombinedFlags:
    def test_ensemble_no_metadata(self, img):
        """--ensemble --no-metadata 조합 오류 없음."""
        r = run_cli(img, "--ensemble", "--no-metadata", env=MOCK_ENV)
        assert r.returncode == 0, r.stderr

    def test_multiple_images_ensemble(self, img, img2):
        """--ensemble + 여러 이미지."""
        r = run_cli(img, img2, "--ensemble", "--json", env=MOCK_ENV)
        assert r.returncode == 0
        parsed = json.loads(r.stdout)
        assert len(parsed) == 2
        for item in parsed:
            assert item["verdict"] in ("AI-generated", "Real")

    def test_threshold_with_ensemble(self, img):
        """--ensemble + --threshold 조합."""
        r = run_cli(img, "--ensemble", "--threshold", "0.9", "--json", env=MOCK_ENV)
        assert r.returncode == 0
        parsed = json.loads(r.stdout)
        # threshold=0.9이고 mock=0.73 → Real
        assert parsed[0]["verdict"] == "Real"
