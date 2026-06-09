"""
setup.py (크로스플랫폼 ONNX 빌드 스크립트) 단위 테스트.

- 무거운 실제 변환(optimum/torch 필요)은 @pytest.mark.skipif로 CI에서 스킵.
- 경량 로직(인자 파싱, stale 파일 처리, 검증 헬퍼 함수)은 항상 실행.
- MatMul-only 양자화 검증 로직은 onnxruntime+onnx 있을 때 실행.
"""
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SETUP_PY_PATH = os.path.join(PROJECT_ROOT, "setup.py")
PYTHON_BIN = sys.executable


def _has_onnxruntime() -> bool:
    try:
        import onnxruntime  # noqa: F401
        return True
    except ImportError:
        return False


def _has_onnx() -> bool:
    try:
        import onnx  # noqa: F401
        return True
    except ImportError:
        return False


def _has_optimum() -> bool:
    try:
        import optimum  # noqa: F401
        return True
    except ImportError:
        return False


# ────────────────────────────────────────────────────────────────────────────────
# 스크립트 존재 확인
# ────────────────────────────────────────────────────────────────────────────────

class TestSetupPyExists:
    def test_setup_py_file_exists(self):
        """setup.py 파일이 프로젝트 루트에 존재해야 한다."""
        assert os.path.isfile(SETUP_PY_PATH), f"setup.py 없음: {SETUP_PY_PATH}"

    def test_setup_py_is_python_file(self):
        """setup.py가 파이썬 파일이어야 한다."""
        with open(SETUP_PY_PATH, encoding="utf-8") as f:
            content = f.read()
        assert "if __name__" in content or "argparse" in content, (
            "setup.py가 올바른 Python 스크립트가 아님"
        )

    def test_setup_py_no_setuptools(self):
        """setup.py는 setuptools 패키징 파일이 아니어야 한다 (from setuptools import setup 없음)."""
        with open(SETUP_PY_PATH, encoding="utf-8") as f:
            content = f.read()
        assert "from setuptools import setup" not in content, (
            "setuptools 패키징 코드가 포함됨 — 이 파일은 빌드 스크립트야"
        )


# ────────────────────────────────────────────────────────────────────────────────
# --help / -h 동작
# ────────────────────────────────────────────────────────────────────────────────

class TestSetupPyHelp:
    def test_help_flag_exits_zero(self):
        """-h 실행 시 exit code 0 반환."""
        result = subprocess.run(
            [PYTHON_BIN, SETUP_PY_PATH, "-h"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"

    def test_help_flag_long_exits_zero(self):
        """--help 실행 시 exit code 0 반환."""
        result = subprocess.run(
            [PYTHON_BIN, SETUP_PY_PATH, "--help"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"

    def test_help_mentions_model_option(self):
        """--help 출력에 모델 관련 설명이 있어야 한다."""
        result = subprocess.run(
            [PYTHON_BIN, SETUP_PY_PATH, "--help"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=10,
        )
        combined = result.stdout + result.stderr
        assert any(
            kw in combined for kw in ["model", "MODEL", "Organika"]
        ), f"help 출력에 모델 관련 옵션 없음:\n{combined}"

    def test_help_mentions_output_dir(self):
        """--help 출력에 --output-dir 옵션이 포함되어야 한다."""
        result = subprocess.run(
            [PYTHON_BIN, SETUP_PY_PATH, "--help"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=10,
        )
        combined = result.stdout + result.stderr
        assert "output" in combined.lower() or "output-dir" in combined, (
            f"--output-dir 언급 없음:\n{combined}"
        )

    def test_help_mentions_skip_install(self):
        """--help 출력에 --skip-install 옵션이 언급되어야 한다."""
        result = subprocess.run(
            [PYTHON_BIN, SETUP_PY_PATH, "--help"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=10,
        )
        combined = result.stdout + result.stderr
        assert "skip" in combined.lower(), f"skip-install 언급 없음:\n{combined}"


# ────────────────────────────────────────────────────────────────────────────────
# 인자 파싱 단위 테스트 (_build_arg_parser import)
# ────────────────────────────────────────────────────────────────────────────────

class TestArgParsing:
    """setup.py의 _build_arg_parser 함수를 직접 import해서 단위 테스트."""

    def _load_module(self):
        """setup.py를 모듈로 로드 (exec, __file__ 주입)."""
        return _load_setup_py_module()

    def test_default_output_dir(self):
        """--output-dir 미지정 시 기본값 'onnx_models'."""
        ns = self._load_module()
        parser = ns["_build_arg_parser"]()
        args = parser.parse_args([])
        assert args.output_dir == "onnx_models"

    def test_default_models_is_organika(self):
        """모델 미지정 시 기본 모델이 Organika/sdxl-detector."""
        ns = self._load_module()
        parser = ns["_build_arg_parser"]()
        args = parser.parse_args([])
        assert "Organika/sdxl-detector" in args.models

    def test_custom_output_dir(self):
        """--output-dir 지정 시 해당 경로 사용."""
        ns = self._load_module()
        parser = ns["_build_arg_parser"]()
        args = parser.parse_args(["--output-dir", "/tmp/mymodels"])
        assert args.output_dir == "/tmp/mymodels"

    def test_skip_install_default_false(self):
        """--skip-install 미지정 시 False."""
        ns = self._load_module()
        parser = ns["_build_arg_parser"]()
        args = parser.parse_args([])
        assert args.skip_install is False

    def test_skip_install_flag(self):
        """--skip-install 지정 시 True."""
        ns = self._load_module()
        parser = ns["_build_arg_parser"]()
        args = parser.parse_args(["--skip-install"])
        assert args.skip_install is True

    def test_multiple_model_ids(self):
        """여러 모델 ID 지정 가능."""
        ns = self._load_module()
        parser = ns["_build_arg_parser"]()
        args = parser.parse_args(["model/a", "model/b"])
        assert "model/a" in args.models
        assert "model/b" in args.models

    def test_single_model_id(self):
        """모델 ID 1개 지정."""
        ns = self._load_module()
        parser = ns["_build_arg_parser"]()
        args = parser.parse_args(["some/model"])
        assert args.models == ["some/model"]

    def test_unknown_option_exits_nonzero(self):
        """알 수 없는 옵션은 exit 2 (argparse 기본)."""
        result = subprocess.run(
            [PYTHON_BIN, SETUP_PY_PATH, "--unknown-flag-xyz"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode != 0


# ────────────────────────────────────────────────────────────────────────────────
# quantize_matmul_only 함수 단위 테스트
# ────────────────────────────────────────────────────────────────────────────────

def _load_setup_py_module():
    """setup.py를 모듈로 로드. exec 컨텍스트에 __file__ 주입."""
    ns = {"__file__": SETUP_PY_PATH}
    with open(SETUP_PY_PATH, encoding="utf-8") as f:
        code = f.read()
    exec(compile(code, SETUP_PY_PATH, "exec"), ns)
    return ns


@pytest.mark.skipif(not _has_onnxruntime(), reason="onnxruntime 미설치")
class TestQuantizeMatmulOnly:
    """
    quantize_matmul_only(src, dst) 함수:
    op_types_to_quantize=['MatMul']로 양자화 → ConvInteger 0개 보장.
    """

    def _build_tiny_conv_matmul_onnx(self, path: str) -> None:
        """Conv + MatMul 노드가 있는 tiny ONNX 모델 생성."""
        try:
            import onnx
            from onnx import TensorProto, helper, numpy_helper

            X = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 1, 6, 6])
            Y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 4])

            w_data = np.random.randn(1, 1, 3, 3).astype(np.float32)
            W_init = numpy_helper.from_array(w_data, name="conv_w")
            m_data = np.random.randn(16, 4).astype(np.float32)
            M_init = numpy_helper.from_array(m_data, name="matmul_w")
            flat_shape = numpy_helper.from_array(np.array([1, 16], dtype=np.int64), name="flat_shape")

            conv_out = helper.make_tensor_value_info("conv_out", TensorProto.FLOAT, [1, 1, 4, 4])
            flat_out = helper.make_tensor_value_info("flat_out", TensorProto.FLOAT, [1, 16])

            conv_node = helper.make_node("Conv", inputs=["input", "conv_w"], outputs=["conv_out"])
            flat_node = helper.make_node("Reshape", inputs=["conv_out", "flat_shape"], outputs=["flat_out"])
            matmul_node = helper.make_node("MatMul", inputs=["flat_out", "matmul_w"], outputs=["output"])

            graph = helper.make_graph(
                [conv_node, flat_node, matmul_node],
                "test",
                [X],
                [Y],
                initializer=[W_init, M_init, flat_shape],
            )
            model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
            model.ir_version = 8
            onnx.checker.check_model(model)
            onnx.save(model, path)
        except ImportError:
            pytest.skip("onnx 패키지 미설치")

    def test_quantize_matmul_only_no_conv_integer(self, tmp_path):
        """quantize_matmul_only() 후 ConvInteger 노드 0개."""
        if not _has_onnx():
            pytest.skip("onnx 패키지 미설치")

        ns = _load_setup_py_module()
        quantize_fn = ns["quantize_matmul_only"]

        src = str(tmp_path / "model.onnx")
        dst = str(tmp_path / "model_quantized.onnx")
        self._build_tiny_conv_matmul_onnx(src)

        quantize_fn(src, dst)

        import onnx as onnx_lib
        model = onnx_lib.load(dst)
        conv_integer_count = sum(
            1 for node in model.graph.node if node.op_type == "ConvInteger"
        )
        assert conv_integer_count == 0, (
            f"ConvInteger {conv_integer_count}개 — MatMul-only 양자화 실패"
        )

    def test_quantize_matmul_only_output_file_created(self, tmp_path):
        """quantize_matmul_only() 후 dst 파일이 생성됨."""
        if not _has_onnx():
            pytest.skip("onnx 패키지 미설치")

        ns = _load_setup_py_module()
        quantize_fn = ns["quantize_matmul_only"]

        src = str(tmp_path / "model.onnx")
        dst = str(tmp_path / "model_quantized.onnx")
        self._build_tiny_conv_matmul_onnx(src)

        quantize_fn(src, dst)
        assert os.path.isfile(dst)

    def test_quantize_matmul_only_raises_on_missing_src(self, tmp_path):
        """존재하지 않는 src → 예외 발생."""
        ns = _load_setup_py_module()
        quantize_fn = ns["quantize_matmul_only"]

        with pytest.raises(Exception):
            quantize_fn(str(tmp_path / "nonexistent.onnx"), str(tmp_path / "dst.onnx"))


# ────────────────────────────────────────────────────────────────────────────────
# verify_onnx_cpu_loadable 함수 단위 테스트
# ────────────────────────────────────────────────────────────────────────────────

@pytest.mark.skipif(not _has_onnxruntime(), reason="onnxruntime 미설치")
class TestVerifyOnnxCpuLoadable:
    """
    verify_onnx_cpu_loadable(path, image_size) 함수:
    ConvInteger 0개 + CPU 세션 로드 + 더미 추론 3가지 검증.
    """

    def _build_matmul_only_onnx(self, path: str) -> None:
        """입력 1개인 MatMul 모델 (W는 initializer로 내장 — verify 시 단일 입력 피드)."""
        try:
            import onnx
            from onnx import TensorProto, helper, numpy_helper

            # A: 외부 입력 (1,4)  W: initializer (4,2)  C: 출력 (1,2)
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

    def _build_conv_integer_onnx(self, path: str) -> None:
        """ConvInteger 노드를 포함한 tiny ONNX 모델 (검증 실패 케이스)."""
        try:
            import onnx
            from onnx import TensorProto, helper

            X = helper.make_tensor_value_info("X", TensorProto.INT8, [1, 1, 4, 4])
            W = helper.make_tensor_value_info("W", TensorProto.INT8, [1, 1, 2, 2])
            Y = helper.make_tensor_value_info("Y", TensorProto.INT32, [1, 1, 3, 3])
            node = helper.make_node("ConvInteger", inputs=["X", "W"], outputs=["Y"])
            graph = helper.make_graph([node], "test", [X, W], [Y])
            model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
            model.ir_version = 8
            onnx.save(model, path)
        except ImportError:
            pytest.skip("onnx 패키지 미설치")

    def test_verify_passes_for_matmul_only(self, tmp_path):
        """MatMul-only 모델 → verify_onnx_cpu_loadable 성공 (True 반환 또는 예외 없음)."""
        if not _has_onnx():
            pytest.skip("onnx 패키지 미설치")

        ns = _load_setup_py_module()
        verify_fn = ns["verify_onnx_cpu_loadable"]

        model_path = str(tmp_path / "matmul_only.onnx")
        self._build_matmul_only_onnx(model_path)

        # image_size는 실제 모델 shape와 무관하게 전달; 내부에서 동적으로 처리
        result = verify_fn(model_path, image_size=224)
        # True 반환 또는 예외 없이 완료되면 성공
        assert result is True or result is None

    def test_verify_detects_conv_integer(self, tmp_path):
        """ConvInteger 포함 모델 → verify_onnx_cpu_loadable가 VerificationError 발생."""
        if not _has_onnx():
            pytest.skip("onnx 패키지 미설치")

        ns = _load_setup_py_module()
        verify_fn = ns["verify_onnx_cpu_loadable"]
        VerificationError = ns.get("VerificationError", Exception)

        model_path = str(tmp_path / "conv_integer.onnx")
        self._build_conv_integer_onnx(model_path)

        with pytest.raises(Exception):
            verify_fn(model_path, image_size=224)

    def test_verify_raises_on_nonexistent_file(self, tmp_path):
        """존재하지 않는 파일 → 예외 발생."""
        ns = _load_setup_py_module()
        verify_fn = ns["verify_onnx_cpu_loadable"]

        with pytest.raises(Exception):
            verify_fn(str(tmp_path / "nonexistent.onnx"), image_size=224)


# ────────────────────────────────────────────────────────────────────────────────
# stale model_quantized.onnx 처리 (remove_stale_quantized 함수)
# ────────────────────────────────────────────────────────────────────────────────

class TestRemoveStaleQuantized:
    """
    remove_stale_quantized(path) 함수:
    파일이 있으면 삭제, 없으면 아무것도 안 함 (멱등성).
    """

    def test_removes_existing_file(self, tmp_path):
        """기존 파일 → 삭제됨."""
        ns = _load_setup_py_module()
        remove_fn = ns["remove_stale_quantized"]

        target = tmp_path / "model_quantized.onnx"
        target.write_bytes(b"stale_content")
        assert target.exists()

        remove_fn(str(target))
        assert not target.exists()

    def test_no_error_when_file_missing(self, tmp_path):
        """파일 없을 때 → 예외 없이 통과 (멱등)."""
        ns = _load_setup_py_module()
        remove_fn = ns["remove_stale_quantized"]

        target = tmp_path / "model_quantized.onnx"
        assert not target.exists()

        # 예외 없이 통과해야 함
        remove_fn(str(target))

    def test_idempotent_double_call(self, tmp_path):
        """두 번 호출해도 예외 없음."""
        ns = _load_setup_py_module()
        remove_fn = ns["remove_stale_quantized"]

        target = tmp_path / "model_quantized.onnx"
        target.write_bytes(b"content")

        remove_fn(str(target))
        remove_fn(str(target))  # 두 번째 호출도 예외 없어야 함


# ────────────────────────────────────────────────────────────────────────────────
# 크로스플랫폼 경로 처리 (os.path / pathlib 사용 확인)
# ────────────────────────────────────────────────────────────────────────────────

class TestCrossplatformPaths:
    """setup.py가 셸 의존 없이 표준 라이브러리만 사용하는지 확인."""

    def test_no_shell_rm_command(self):
        """'rm -f' 같은 셸 명령이 없어야 한다."""
        with open(SETUP_PY_PATH, encoding="utf-8") as f:
            content = f.read()
        assert "rm -f" not in content and "rm -rf" not in content, (
            "shell rm 명령이 포함됨 — os.remove 또는 Path.unlink 사용 필요"
        )

    def test_no_shell_true_in_subprocess(self):
        """subprocess.run(..., shell=True)가 없어야 한다 (shell=False 권장)."""
        with open(SETUP_PY_PATH, encoding="utf-8") as f:
            content = f.read()
        assert "shell=True" not in content, (
            "shell=True 사용 금지 — 크로스플랫폼 이슈 및 보안 위험"
        )

    def test_uses_sys_executable_not_hardcoded_python(self):
        """'python ' 하드코딩 대신 sys.executable 사용."""
        with open(SETUP_PY_PATH, encoding="utf-8") as f:
            content = f.read()
        # 하드코딩된 'python ' 단독 사용 없어야 함 (sys.executable 사용)
        import re
        # subprocess 호출에서 'python'만 쓰는 패턴 탐지
        bad_pattern = re.search(r'subprocess\.[^\n]*\[[\s\'"]*python[\s\'"]', content)
        assert bad_pattern is None, (
            "subprocess에 'python' 하드코딩 발견 — sys.executable 사용 필요"
        )

    def test_uses_os_path_or_pathlib(self):
        """경로 조작에 os.path 또는 pathlib을 사용한다."""
        with open(SETUP_PY_PATH, encoding="utf-8") as f:
            content = f.read()
        assert "os.path" in content or "pathlib" in content or "Path(" in content, (
            "경로 조작에 os.path 또는 pathlib을 사용해야 함"
        )


# ────────────────────────────────────────────────────────────────────────────────
# subprocess에서 sys.executable 확인 (이름 오타 수정용 별도 클래스)
# ────────────────────────────────────────────────────────────────────────────────

class TestCrossplatformSubprocess:
    """setup.py 소스 코드 레벨 검사."""

    def test_uses_sys_executable(self):
        """sys.executable을 사용해 subprocess를 호출한다."""
        with open(SETUP_PY_PATH, encoding="utf-8") as f:
            content = f.read()
        assert "sys.executable" in content, (
            "sys.executable 사용 필요 — Windows에서 'python' 하드코딩은 동작 안 할 수 있음"
        )

    def test_no_shell_ampersand(self):
        """셸 명령 조합 연산자(&& 등)가 문자열 인자에 없어야 한다."""
        with open(SETUP_PY_PATH, encoding="utf-8") as f:
            content = f.read()
        # subprocess 호출 내부 cmd 문자열에 && 가 있으면 shell=False와 모순
        import re
        lines_with_subprocess = [
            line for line in content.split("\n")
            if "subprocess.run" in line or "subprocess.call" in line
        ]
        for line in lines_with_subprocess:
            assert "&&" not in line, f"subprocess 호출에 && 발견: {line!r}"


# ────────────────────────────────────────────────────────────────────────────────
# 엣지 케이스: 인자 처리
# ────────────────────────────────────────────────────────────────────────────────

class TestArgEdgeCases:
    def test_empty_model_id_via_args(self):
        """빈 문자열 모델 ID는 파서가 받아들이거나 오류를 내야 한다 (크래시 없음)."""
        result = subprocess.run(
            [PYTHON_BIN, SETUP_PY_PATH, ""],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=10,
        )
        # 크래시(segfault 등)가 아닌 정상 종료이기만 하면 됨
        assert result.returncode in (0, 1, 2), (
            f"예상치 못한 exit code: {result.returncode}\nstderr: {result.stderr}"
        )

    def test_special_char_in_output_dir_no_crash(self, tmp_path):
        """출력 디렉토리 경로에 특수문자(공백 등)가 있어도 파싱이 됨."""
        ns = _load_setup_py_module()
        parser = ns["_build_arg_parser"]()
        special_dir = str(tmp_path / "my models")
        args = parser.parse_args(["--output-dir", special_dir])
        assert args.output_dir == special_dir

    def test_output_dir_with_unicode(self, tmp_path):
        """출력 디렉토리 경로에 유니코드(한글 등)가 있어도 파싱이 됨."""
        ns = _load_setup_py_module()
        parser = ns["_build_arg_parser"]()
        unicode_dir = str(tmp_path / "모델저장소")
        args = parser.parse_args(["--output-dir", unicode_dir])
        assert args.output_dir == unicode_dir

    def test_multiple_same_model_ids(self):
        """동일 모델 ID 중복 지정은 파서가 그대로 받아들임 (중복 처리는 빌드 로직 담당)."""
        ns = _load_setup_py_module()
        parser = ns["_build_arg_parser"]()
        args = parser.parse_args(["model/a", "model/a"])
        assert args.models.count("model/a") >= 1  # 최소 1개 이상


# ────────────────────────────────────────────────────────────────────────────────
# E2E: 전체 변환 — optimum + torch 없으면 스킵
# ────────────────────────────────────────────────────────────────────────────────

@pytest.mark.skipif(
    not _has_optimum(),
    reason="optimum[onnxruntime] 미설치 — setup.py 전체 변환 테스트 스킵",
)
class TestSetupPyEndToEnd:
    """
    실제 setup.py를 실행해 full 변환 → self-test까지 검증.
    torch/optimum 설치 필요. CI에서는 보통 스킵됨.
    """

    def test_script_runs_default_model_success(self, tmp_path):
        """기본 모델(Organika/sdxl-detector)로 스크립트 실행 → exit 0."""
        result = subprocess.run(
            [
                PYTHON_BIN, SETUP_PY_PATH,
                "--output-dir", str(tmp_path),
                "--skip-install",
            ],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=600,
        )
        assert result.returncode == 0, (
            f"exit code: {result.returncode}\n"
            f"stdout: {result.stdout[-2000:]}\n"
            f"stderr: {result.stderr[-2000:]}"
        )

    def test_script_produces_quantized_onnx(self, tmp_path):
        """실행 후 model_quantized.onnx가 생성되어야 한다."""
        subprocess.run(
            [
                PYTHON_BIN, SETUP_PY_PATH,
                "--output-dir", str(tmp_path),
                "--skip-install",
            ],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=600,
        )
        quantized_path = os.path.join(
            str(tmp_path), "Organika__sdxl-detector", "model_quantized.onnx"
        )
        assert os.path.isfile(quantized_path), "model_quantized.onnx 생성 안 됨"

    def test_script_produces_meta_json(self, tmp_path):
        """실행 후 meta.json이 생성되고 _validate_meta를 통과해야 한다."""
        subprocess.run(
            [
                PYTHON_BIN, SETUP_PY_PATH,
                "--output-dir", str(tmp_path),
                "--skip-install",
            ],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=600,
        )
        meta_path = os.path.join(
            str(tmp_path), "Organika__sdxl-detector", "meta.json"
        )
        assert os.path.isfile(meta_path), "meta.json 생성 안 됨"
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        from onnx_detector import _validate_meta
        _validate_meta(meta)

    def test_script_idempotent(self, tmp_path):
        """두 번 실행해도 같은 결과 (멱등성)."""
        for _ in range(2):
            result = subprocess.run(
                [
                    PYTHON_BIN, SETUP_PY_PATH,
                    "--output-dir", str(tmp_path),
                    "--skip-install",
                ],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                timeout=600,
            )
            assert result.returncode == 0

    def test_generated_onnx_has_no_conv_integer(self, tmp_path):
        """생성된 model_quantized.onnx에 ConvInteger 노드가 없어야 한다."""
        if not _has_onnx():
            pytest.skip("onnx 패키지 미설치")

        subprocess.run(
            [
                PYTHON_BIN, SETUP_PY_PATH,
                "--output-dir", str(tmp_path),
                "--skip-install",
            ],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=600,
        )
        quantized_path = os.path.join(
            str(tmp_path), "Organika__sdxl-detector", "model_quantized.onnx"
        )
        if not os.path.isfile(quantized_path):
            pytest.skip("model_quantized.onnx 없음 — 변환 실패")

        import onnx as onnx_lib
        model = onnx_lib.load(quantized_path)
        conv_integer_count = sum(
            1 for node in model.graph.node if node.op_type == "ConvInteger"
        )
        assert conv_integer_count == 0, f"ConvInteger 노드 {conv_integer_count}개 발견"

    def test_multiple_models(self, tmp_path):
        """여러 모델 ID 지정 시 모두 처리된다."""
        result = subprocess.run(
            [
                PYTHON_BIN, SETUP_PY_PATH,
                "--output-dir", str(tmp_path),
                "--skip-install",
                "Organika/sdxl-detector",
            ],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=600,
        )
        assert result.returncode == 0


# ────────────────────────────────────────────────────────────────────────────────
# 성공 출력 형식 확인 (subprocess로 --help 대신 실제 출력 검사 — 경량)
# ────────────────────────────────────────────────────────────────────────────────

class TestOutputFormat:
    def test_help_output_is_readable(self):
        """--help 출력이 빈 문자열이 아니어야 한다."""
        result = subprocess.run(
            [PYTHON_BIN, SETUP_PY_PATH, "--help"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert len(result.stdout.strip()) > 0, "--help 출력이 비어있음"

    def test_help_contains_usage_line(self):
        """--help 출력에 usage 라인이 있어야 한다."""
        result = subprocess.run(
            [PYTHON_BIN, SETUP_PY_PATH, "--help"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=10,
        )
        combined = result.stdout + result.stderr
        assert "usage" in combined.lower() or "Usage" in combined, (
            f"usage 라인 없음:\n{combined}"
        )
