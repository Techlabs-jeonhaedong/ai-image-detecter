"""Shared fixtures for tests."""
import io
import os
import pytest
from PIL import Image


@pytest.fixture
def valid_image_path(tmp_path):
    """작은 RGB PNG 이미지를 임시 파일로 생성."""
    img = Image.new("RGB", (64, 64), color=(128, 64, 32))
    path = tmp_path / "test_image.png"
    img.save(str(path))
    return str(path)


@pytest.fixture
def valid_jpeg_path(tmp_path):
    img = Image.new("RGB", (32, 32), color=(0, 128, 255))
    path = tmp_path / "test_image.jpg"
    img.save(str(path), format="JPEG")
    return str(path)


@pytest.fixture
def multiple_valid_images(tmp_path):
    paths = []
    for i in range(3):
        img = Image.new("RGB", (32, 32), color=(i * 80, i * 40, 100))
        path = tmp_path / f"img_{i}.png"
        img.save(str(path))
        paths.append(str(path))
    return paths


@pytest.fixture
def corrupt_image_path(tmp_path):
    """손상된 이미지 파일."""
    path = tmp_path / "corrupt.jpg"
    path.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 10)  # 잘못된 JPEG
    return str(path)


@pytest.fixture
def non_image_path(tmp_path):
    """이미지가 아닌 텍스트 파일."""
    path = tmp_path / "not_image.txt"
    path.write_text("this is not an image file")
    return str(path)


@pytest.fixture
def no_permission_image_path(tmp_path, valid_image_path):
    """읽기 권한이 없는 이미지 파일."""
    import shutil
    path = tmp_path / "no_perm.png"
    shutil.copy(valid_image_path, str(path))
    os.chmod(str(path), 0o000)
    yield str(path)
    os.chmod(str(path), 0o644)


@pytest.fixture
def mock_pipeline_ai():
    """AI 생성 이미지로 판정하는 mock pipeline."""
    def pipeline(*args, **kwargs):
        def infer(image):
            # Organika/sdxl-detector 실제 라벨: 'artificial', 'human'
            return [
                {"label": "artificial", "score": 0.87},
                {"label": "human", "score": 0.13},
            ]
        return infer
    return pipeline


@pytest.fixture
def mock_pipeline_real():
    """Real 이미지로 판정하는 mock pipeline."""
    def pipeline(*args, **kwargs):
        def infer(image):
            return [
                {"label": "human", "score": 0.92},
                {"label": "artificial", "score": 0.08},
            ]
        return infer
    return pipeline


@pytest.fixture
def mock_pipeline_low_confidence():
    """임계값 경계 근처 결과를 반환하는 mock pipeline."""
    def pipeline(*args, **kwargs):
        def infer(image):
            return [
                {"label": "artificial", "score": 0.50},
                {"label": "human", "score": 0.50},
            ]
        return infer
    return pipeline
