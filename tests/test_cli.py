"""detect.py CLI E2E 테스트."""
import json
import os
import subprocess
import sys
import pytest
from PIL import Image
from unittest.mock import patch

PYTHON = sys.executable
DETECT = "/Users/jeonhaedong/Desktop/ai-image-detecter/detect.py"


def run_cli(*args, env=None):
    """detect.py를 subprocess로 실행하고 CompletedProcess를 반환."""
    cmd = [PYTHON, DETECT] + list(args)
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    return subprocess.run(cmd, capture_output=True, text=True, env=merged_env)


# ──────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────

@pytest.fixture
def img(tmp_path):
    p = tmp_path / "real.png"
    Image.new("RGB", (64, 64), color=(10, 20, 30)).save(str(p))
    return str(p)


@pytest.fixture
def img2(tmp_path):
    p = tmp_path / "real2.jpg"
    Image.new("RGB", (32, 32), color=(200, 100, 50)).save(str(p), format="JPEG")
    return str(p)


@pytest.fixture
def corrupt(tmp_path):
    p = tmp_path / "corrupt.jpg"
    p.write_bytes(b"\xff\xd8\xff" + b"\xAB" * 20)
    return str(p)


@pytest.fixture
def txt_file(tmp_path):
    p = tmp_path / "doc.txt"
    p.write_text("hello world")
    return str(p)


# detect.py를 import 단계에서 실제 모델 로드 없이 동작하게 하려면
# 환경변수로 mock pipeline을 주입하는 대신,
# detect.py가 detector.get_pipeline()을 통해 pipeline을 얻도록 설계하고
# 테스트에서는 patch를 사용한다.
# subprocess 기반 E2E에서는 직접 patch가 어려우므로
# 실제 모델 로딩을 트리거하지 않도록 _MOCK_PIPELINE 환경변수를 사용한다.

MOCK_ENV = {"_AI_DETECTOR_MOCK": "1"}


class TestCLIHappyPath:
    def test_single_image_exits_zero(self, img):
        r = run_cli(img, env=MOCK_ENV)
        assert r.returncode == 0, r.stderr

    def test_single_image_output_has_verdict(self, img):
        r = run_cli(img, env=MOCK_ENV)
        output = r.stdout
        assert "AI-generated" in output or "Real" in output

    def test_single_image_output_has_probability(self, img):
        r = run_cli(img, env=MOCK_ENV)
        assert "%" in r.stdout

    def test_single_image_output_has_model_name(self, img):
        r = run_cli(img, env=MOCK_ENV)
        assert "sdxl-detector" in r.stdout or "Organika" in r.stdout or "model" in r.stdout.lower()

    def test_multiple_images(self, img, img2):
        r = run_cli(img, img2, env=MOCK_ENV)
        assert r.returncode == 0
        # 두 이미지 모두 결과 포함
        assert r.stdout.count("AI-generated") + r.stdout.count("Real") >= 2

    def test_json_output_single(self, img):
        r = run_cli(img, "--json", env=MOCK_ENV)
        assert r.returncode == 0
        parsed = json.loads(r.stdout)
        assert isinstance(parsed, list)
        assert len(parsed) == 1
        assert "ai_probability" in parsed[0]
        assert "verdict" in parsed[0]

    def test_json_output_multiple(self, img, img2):
        r = run_cli(img, img2, "--json", env=MOCK_ENV)
        assert r.returncode == 0
        parsed = json.loads(r.stdout)
        assert len(parsed) == 2

    def test_custom_model_option(self, img):
        r = run_cli(img, "--model", "custom/model-test", env=MOCK_ENV)
        assert r.returncode == 0
        assert "custom/model-test" in r.stdout

    def test_custom_threshold_option(self, img):
        r = run_cli(img, "--threshold", "0.8", env=MOCK_ENV)
        assert r.returncode == 0

    def test_json_output_has_model_field(self, img):
        r = run_cli(img, "--json", env=MOCK_ENV)
        parsed = json.loads(r.stdout)
        assert "model" in parsed[0]

    def test_json_output_image_path(self, img):
        r = run_cli(img, "--json", env=MOCK_ENV)
        parsed = json.loads(r.stdout)
        assert parsed[0]["image"] == img


class TestCLIErrorHandling:
    def test_no_args_exits_nonzero(self):
        r = run_cli(env=MOCK_ENV)
        assert r.returncode != 0

    def test_no_args_shows_usage(self):
        r = run_cli(env=MOCK_ENV)
        stderr_or_stdout = r.stderr + r.stdout
        assert "usage" in stderr_or_stdout.lower() or "error" in stderr_or_stdout.lower()

    def test_nonexistent_file_graceful(self):
        r = run_cli("/no/such/file.jpg", env=MOCK_ENV)
        # 프로세스는 0 또는 비0, 중요한 건 에러 메시지가 있어야 함
        assert "error" in r.stdout.lower() or "not found" in r.stdout.lower() or "ERROR" in r.stdout

    def test_nonexistent_file_json_mode(self):
        r = run_cli("/no/such/file.jpg", "--json", env=MOCK_ENV)
        parsed = json.loads(r.stdout)
        assert parsed[0]["error"] is not None
        assert parsed[0]["verdict"] is None

    def test_non_image_file_graceful(self, txt_file):
        r = run_cli(txt_file, env=MOCK_ENV)
        output = r.stdout + r.stderr
        assert "error" in output.lower() or "ERROR" in output

    def test_non_image_file_json_mode(self, txt_file):
        r = run_cli(txt_file, "--json", env=MOCK_ENV)
        parsed = json.loads(r.stdout)
        assert parsed[0]["error"] is not None

    def test_corrupt_image_graceful(self, corrupt):
        r = run_cli(corrupt, env=MOCK_ENV)
        output = r.stdout + r.stderr
        assert "error" in output.lower() or "ERROR" in output

    def test_corrupt_image_json_mode(self, corrupt):
        r = run_cli(corrupt, "--json", env=MOCK_ENV)
        parsed = json.loads(r.stdout)
        assert parsed[0]["error"] is not None

    def test_partial_failure_continues(self, img):
        """한 이미지 실패해도 나머지 처리됨."""
        r = run_cli("/nonexistent.jpg", img, env=MOCK_ENV)
        # 두 번째 이미지는 성공해야 함
        assert "AI-generated" in r.stdout or "Real" in r.stdout

    def test_partial_failure_json_has_both(self, img):
        r = run_cli("/nonexistent.jpg", img, "--json", env=MOCK_ENV)
        parsed = json.loads(r.stdout)
        assert len(parsed) == 2
        assert parsed[0]["error"] is not None
        assert parsed[1]["error"] is None

    def test_invalid_threshold_exits_nonzero(self, img):
        r = run_cli(img, "--threshold", "1.5", env=MOCK_ENV)
        assert r.returncode != 0

    def test_invalid_threshold_negative(self, img):
        r = run_cli(img, "--threshold", "-0.1", env=MOCK_ENV)
        assert r.returncode != 0

    def test_invalid_threshold_string(self, img):
        r = run_cli(img, "--threshold", "notanumber", env=MOCK_ENV)
        assert r.returncode != 0

    def test_no_permission_file(self, tmp_path, img):
        import shutil
        no_perm = tmp_path / "noperm.png"
        shutil.copy(img, str(no_perm))
        os.chmod(str(no_perm), 0o000)
        try:
            r = run_cli(str(no_perm), env=MOCK_ENV)
            output = r.stdout + r.stderr
            assert "error" in output.lower() or "ERROR" in output
        finally:
            os.chmod(str(no_perm), 0o644)


class TestCLIEdgeCases:
    def test_very_large_image(self, tmp_path):
        p = tmp_path / "large.png"
        Image.new("RGB", (4096, 4096), color=(0, 0, 0)).save(str(p))
        r = run_cli(str(p), env=MOCK_ENV)
        assert r.returncode == 0

    def test_grayscale_image(self, tmp_path):
        p = tmp_path / "gray.png"
        Image.new("L", (64, 64), color=128).save(str(p))
        r = run_cli(str(p), env=MOCK_ENV)
        assert r.returncode == 0

    def test_rgba_image(self, tmp_path):
        p = tmp_path / "rgba.png"
        Image.new("RGBA", (64, 64), color=(255, 0, 0, 128)).save(str(p))
        r = run_cli(str(p), env=MOCK_ENV)
        assert r.returncode == 0

    def test_1x1_image(self, tmp_path):
        p = tmp_path / "tiny.png"
        Image.new("RGB", (1, 1), color=(255, 255, 255)).save(str(p))
        r = run_cli(str(p), env=MOCK_ENV)
        assert r.returncode == 0

    def test_path_with_spaces(self, tmp_path):
        d = tmp_path / "dir with spaces"
        d.mkdir()
        p = d / "my image.png"
        Image.new("RGB", (32, 32), color=(50, 50, 50)).save(str(p))
        r = run_cli(str(p), env=MOCK_ENV)
        assert r.returncode == 0

    def test_threshold_boundary_zero(self, tmp_path):
        p = tmp_path / "t.png"
        Image.new("RGB", (32, 32)).save(str(p))
        r = run_cli(str(p), "--threshold", "0.0", env=MOCK_ENV)
        assert r.returncode == 0

    def test_threshold_boundary_one(self, tmp_path):
        p = tmp_path / "t.png"
        Image.new("RGB", (32, 32)).save(str(p))
        r = run_cli(str(p), "--threshold", "1.0", env=MOCK_ENV)
        assert r.returncode == 0

    def test_many_images(self, tmp_path):
        paths = []
        for i in range(10):
            p = tmp_path / f"img{i}.png"
            Image.new("RGB", (16, 16), color=(i * 25, 0, 0)).save(str(p))
            paths.append(str(p))
        r = run_cli(*paths, env=MOCK_ENV)
        assert r.returncode == 0
        parsed_count = r.stdout.count("AI-generated") + r.stdout.count("Real")
        assert parsed_count == 10

    def test_duplicate_paths(self, img):
        """같은 경로를 두 번 넣어도 두 번 처리."""
        r = run_cli(img, img, env=MOCK_ENV)
        assert r.returncode == 0
        parsed_count = r.stdout.count("AI-generated") + r.stdout.count("Real")
        assert parsed_count == 2

    def test_json_ai_probability_is_float(self, img):
        r = run_cli(img, "--json", env=MOCK_ENV)
        parsed = json.loads(r.stdout)
        assert isinstance(parsed[0]["ai_probability"], float)

    def test_json_ai_probability_range(self, img):
        r = run_cli(img, "--json", env=MOCK_ENV)
        parsed = json.loads(r.stdout)
        prob = parsed[0]["ai_probability"]
        assert 0.0 <= prob <= 1.0
