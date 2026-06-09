"""
setup.py 번들 모델 스킵 + --force 플래그 + install_runtime_deps 테스트.
Goal #2 (setup.py 단일 실행 완비).
"""
import importlib
import importlib.util
import json
import os
import struct
import sys
import zlib
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ──────────────────────────────────────────────────────────────────────────────
# 헬퍼
# ──────────────────────────────────────────────────────────────────────────────

def _has_onnxruntime() -> bool:
    try:
        import onnxruntime
        return True
    except ImportError:
        return False


def _has_onnx() -> bool:
    try:
        import onnx
        return True
    except ImportError:
        return False


def _load_setup_module():
    """setup.py를 모듈로 동적 로드한다."""
    setup_path = os.path.join(PROJECT_ROOT, "setup.py")
    spec = importlib.util.spec_from_file_location("setup_script", setup_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _build_matmul_only_onnx(path: str) -> None:
    """입력 1개인 MatMul 모델을 주어진 경로에 저장한다."""
    try:
        import onnx
        from onnx import TensorProto, helper, numpy_helper

        A = helper.make_tensor_value_info("A", TensorProto.FLOAT, [1, 4])
        C = helper.make_tensor_value_info("C", TensorProto.FLOAT, [1, 2])
        W_data = np.random.randn(4, 2).astype(np.float32)
        W_init = numpy_helper.from_array(W_data, name="W")
        node = helper.make_node("MatMul", inputs=["A", "W"], outputs=["C"])
        graph = helper.make_graph([node], "test", [A], [C], initializer=[W_init])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
        model.ir_version = 8
        onnx.save(model, path)
    except ImportError:
        pytest.skip("onnx 패키지 미설치")


# ──────────────────────────────────────────────────────────────────────────────
# --force 플래그 파싱
# ──────────────────────────────────────────────────────────────────────────────

class TestForceFlagParses:
    def test_force_flag_parses_true(self):
        """--force 플래그가 True로 파싱되어야 한다."""
        mod = _load_setup_module()
        args = mod._build_arg_parser().parse_args(["--force"])
        assert args.force is True

    def test_force_flag_default_false(self):
        """--force 미지정 시 기본값 False."""
        mod = _load_setup_module()
        args = mod._build_arg_parser().parse_args([])
        assert args.force is False


# ──────────────────────────────────────────────────────────────────────────────
# install_runtime_deps 존재 확인
# ──────────────────────────────────────────────────────────────────────────────

class TestInstallRuntimeDepsExists:
    def test_install_runtime_deps_function_exists(self):
        """setup 모듈에 install_runtime_deps 함수가 있어야 한다."""
        mod = _load_setup_module()
        assert hasattr(mod, "install_runtime_deps"), (
            "install_runtime_deps 함수가 setup.py에 없음"
        )
        assert callable(mod.install_runtime_deps)

    def test_install_runtime_deps_references_requirements_onnx(self):
        """install_runtime_deps 소스가 requirements-onnx.txt를 참조해야 한다."""
        setup_path = os.path.join(PROJECT_ROOT, "setup.py")
        with open(setup_path, encoding="utf-8") as f:
            source = f.read()
        assert "requirements-onnx.txt" in source, (
            "setup.py 소스에 'requirements-onnx.txt' 참조가 없음"
        )


# ──────────────────────────────────────────────────────────────────────────────
# process_model 번들 스킵
# ──────────────────────────────────────────────────────────────────────────────

@pytest.mark.skipif(
    not (_has_onnxruntime() and _has_onnx()),
    reason="onnxruntime 또는 onnx 미설치",
)
class TestProcessModelSkipsWhenBundled:
    def test_process_model_skips_when_bundled(self, tmp_path):
        """
        tmp에 유효한 model_quantized.onnx + meta.json + config.json이 있으면
        export_onnx와 quantize_matmul_only를 호출하지 않고 return해야 한다.
        """
        mod = _load_setup_module()

        # 모델 디렉토리 준비
        model_id = "Organika/sdxl-detector"
        sanitized = model_id.replace("/", "__")
        model_dir = tmp_path / sanitized
        model_dir.mkdir(parents=True)

        # 유효한 MatMul-only ONNX 모델 생성
        quantized_path = model_dir / "model_quantized.onnx"
        _build_matmul_only_onnx(str(quantized_path))

        # meta.json, config.json 배치
        meta = {
            "image_size": 224,
            "image_mean": [0.5, 0.5, 0.5],
            "image_std": [0.5, 0.5, 0.5],
            "id2label": {"0": "artificial", "1": "human"},
        }
        (model_dir / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
        (model_dir / "config.json").write_text("{}", encoding="utf-8")

        # export_onnx와 quantize_matmul_only가 호출되면 예외를 던지도록
        def _raise_export(*args, **kwargs):
            raise AssertionError("export_onnx should NOT be called when bundled model exists")

        def _raise_quantize(*args, **kwargs):
            raise AssertionError("quantize_matmul_only should NOT be called when bundled model exists")

        with patch.object(mod, "export_onnx", side_effect=_raise_export), \
             patch.object(mod, "quantize_matmul_only", side_effect=_raise_quantize):
            # force=False → 번들 감지 후 빌드 스킵
            mod.process_model(model_id, str(tmp_path), force=False)
            # 예외 없이 통과하면 성공

    def test_process_model_force_calls_build(self, tmp_path):
        """
        force=True이면 번들이 있어도 export_onnx가 호출되어야 한다.
        """
        mod = _load_setup_module()

        model_id = "Organika/sdxl-detector"
        sanitized = model_id.replace("/", "__")
        model_dir = tmp_path / sanitized
        model_dir.mkdir(parents=True)

        # 기존 양자화 모델 배치
        quantized_path = model_dir / "model_quantized.onnx"
        _build_matmul_only_onnx(str(quantized_path))
        (model_dir / "meta.json").write_text(
            json.dumps({"image_size": 224, "image_mean": [0.5, 0.5, 0.5],
                        "image_std": [0.5, 0.5, 0.5],
                        "id2label": {"0": "artificial", "1": "human"}}),
            encoding="utf-8",
        )

        called = []

        def _mock_export(mid, outbase):
            called.append("export_onnx")
            # onnx_path 반환 흉내 (실제 파일 없어도 됨)
            return quantized_path.with_name("model.onnx")

        def _mock_quantize(src, dst):
            called.append("quantize_matmul_only")

        def _mock_verify(path, image_size=224):
            called.append("verify")
            return True

        with patch.object(mod, "export_onnx", side_effect=_mock_export), \
             patch.object(mod, "quantize_matmul_only", side_effect=_mock_quantize), \
             patch.object(mod, "verify_onnx_cpu_loadable", side_effect=_mock_verify), \
             patch.object(mod, "remove_stale_quantized"):
            try:
                mod.process_model(model_id, str(tmp_path), force=True)
            except SystemExit:
                pass  # export 결과 파일 없어서 sys.exit 할 수 있음

        assert "export_onnx" in called, "force=True인데 export_onnx가 호출되지 않음"
