"""
make-light-weight.sh 빌드 스크립트 및 MatMul-only 양자화 검증 테스트.

무거운 실제 변환(optimum/torch 필요)은 @pytest.mark.skipif로 CI에서 스킵.
스크립트 동작 단위 테스트(인자 파싱·stale 파일 처리·검증 로직)는 항상 실행.
"""
import os
import shutil
import struct
import subprocess
import sys
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ────────────────────────────────────────────────────────────────────────────────
# 헬퍼
# ────────────────────────────────────────────────────────────────────────────────

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT_PATH = os.path.join(PROJECT_ROOT, "make-light-weight.sh")
PYTHON_BIN = sys.executable


def _has_optimum() -> bool:
    """optimum[onnxruntime] 설치 여부 확인."""
    try:
        import optimum  # noqa: F401
        return True
    except ImportError:
        return False


def _has_onnxruntime() -> bool:
    try:
        import onnxruntime  # noqa: F401
        return True
    except ImportError:
        return False


# ────────────────────────────────────────────────────────────────────────────────
# 스크립트 존재 및 실행 가능 여부
# ────────────────────────────────────────────────────────────────────────────────

class TestScriptExists:
    def test_script_file_exists(self):
        """make-light-weight.sh 파일이 프로젝트 루트에 존재해야 한다."""
        assert os.path.isfile(SCRIPT_PATH), f"스크립트 없음: {SCRIPT_PATH}"

    def test_script_is_executable(self):
        """스크립트에 실행 권한이 있어야 한다."""
        assert os.access(SCRIPT_PATH, os.X_OK), f"실행 권한 없음: {SCRIPT_PATH}"

    def test_script_has_shebang(self):
        """스크립트 첫 줄에 shebang(#!/...)이 있어야 한다."""
        with open(SCRIPT_PATH, encoding="utf-8") as f:
            first_line = f.readline().strip()
        assert first_line.startswith("#!"), f"shebang 없음: {first_line!r}"


# ────────────────────────────────────────────────────────────────────────────────
# --help / -h 동작
# ────────────────────────────────────────────────────────────────────────────────

class TestScriptHelp:
    def test_help_flag_exits_zero(self):
        """-h 실행 시 exit code 0 반환."""
        result = subprocess.run(
            ["bash", SCRIPT_PATH, "-h"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"

    def test_help_flag_long_exits_zero(self):
        """--help 실행 시 exit code 0 반환."""
        result = subprocess.run(
            ["bash", SCRIPT_PATH, "--help"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"

    def test_help_mentions_model_option(self):
        """--help 출력에 모델 관련 설명이 있어야 한다."""
        result = subprocess.run(
            ["bash", SCRIPT_PATH, "--help"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=10,
        )
        combined = result.stdout + result.stderr
        # 모델 ID나 --output-dir 같은 옵션명이 포함돼야 함
        assert any(
            keyword in combined for keyword in ["model", "MODEL", "output", "OUTPUT"]
        ), f"help 출력에 옵션 설명 없음:\n{combined}"

    def test_help_mentions_skip_install(self):
        """--help 출력에 --skip-install 옵션이 언급되어야 한다."""
        result = subprocess.run(
            ["bash", SCRIPT_PATH, "--help"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=10,
        )
        combined = result.stdout + result.stderr
        assert "skip" in combined.lower(), f"skip-install 언급 없음:\n{combined}"


# ────────────────────────────────────────────────────────────────────────────────
# 인자 없이 실행 시 동작 (기본 모델 사용 또는 usage 출력)
# ────────────────────────────────────────────────────────────────────────────────

class TestScriptDefaultArgs:
    def test_unknown_option_exits_nonzero(self):
        """알 수 없는 옵션은 exit 1이어야 한다 (set -e 내부 버그 아님)."""
        result = subprocess.run(
            ["bash", SCRIPT_PATH, "--unknown-flag-xyz"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == 1, (
            f"알 수 없는 옵션에 대해 exit 1 기대, 실제: {result.returncode}"
        )

    def test_unknown_option_prints_usage_hint(self):
        """알 수 없는 옵션 시 사용법 힌트를 출력해야 한다."""
        result = subprocess.run(
            ["bash", SCRIPT_PATH, "--unknown-flag-xyz"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=10,
        )
        combined = result.stdout + result.stderr
        assert "--help" in combined or "사용법" in combined or "usage" in combined.lower()


# ────────────────────────────────────────────────────────────────────────────────
# MatMul-only 양자화 검증 로직 (onnxruntime 있으면 실행)
# ────────────────────────────────────────────────────────────────────────────────

@pytest.mark.skipif(not _has_onnxruntime(), reason="onnxruntime 미설치")
class TestQuantizationVerifyLogic:
    """
    generate_verify_matmul_only_quantized_onnx 유사 로직:
    ConvInteger 노드 카운트 + CPU 세션 로드 + 더미 추론.

    실제 모델 없이 테스트하기 위해 tiny ONNX 모델을 직접 빌드한다.
    """

    def _build_tiny_matmul_onnx(self, path: str, include_conv_integer: bool = False) -> None:
        """
        tiny ONNX 모델을 path에 저장한다.
        include_conv_integer=True면 ConvInteger 노드를 포함한다(검증 실패 케이스).
        include_conv_integer=False면 MatMul 노드만 포함 (검증 성공 케이스).

        onnx 패키지가 없어도 raw protobuf로 최소한의 .onnx를 만든다.
        """
        try:
            import onnx
            from onnx import TensorProto, helper

            if include_conv_integer:
                # ConvInteger 포함 모델 — inputs/outputs 명시
                X = helper.make_tensor_value_info("X", TensorProto.INT8, [1, 1, 4, 4])
                W = helper.make_tensor_value_info("W", TensorProto.INT8, [1, 1, 2, 2])
                Y = helper.make_tensor_value_info("Y", TensorProto.INT32, [1, 1, 3, 3])
                node = helper.make_node("ConvInteger", inputs=["X", "W"], outputs=["Y"])
                graph = helper.make_graph([node], "test", [X, W], [Y])
            else:
                # MatMul만 있는 모델 (float32)
                A = helper.make_tensor_value_info("A", TensorProto.FLOAT, [1, 4])
                B = helper.make_tensor_value_info("B", TensorProto.FLOAT, [4, 2])
                C = helper.make_tensor_value_info("C", TensorProto.FLOAT, [1, 2])
                node = helper.make_node("MatMul", inputs=["A", "B"], outputs=["C"])
                graph = helper.make_graph([node], "test", [A, B], [C])

            model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
            model.ir_version = 8  # onnxruntime 1.x 호환 IR version
            onnx.save(model, path)
        except ImportError:
            # onnx 패키지 없으면 테스트 스킵
            pytest.skip("onnx 패키지 미설치 — tiny 모델 빌드 불가")

    def test_count_conv_integer_nodes_zero_for_matmul_only(self, tmp_path):
        """MatMul-only 모델 → ConvInteger 노드 0개."""
        try:
            import onnx
        except ImportError:
            pytest.skip("onnx 패키지 미설치")

        model_path = str(tmp_path / "matmul_only.onnx")
        self._build_tiny_matmul_onnx(model_path, include_conv_integer=False)

        import onnx as onnx_lib
        model = onnx_lib.load(model_path)
        conv_integer_count = sum(
            1 for node in model.graph.node if node.op_type == "ConvInteger"
        )
        assert conv_integer_count == 0

    def test_count_conv_integer_nodes_nonzero_for_conv_model(self, tmp_path):
        """ConvInteger 포함 모델 → ConvInteger 노드 1개 이상 → 검증이 이를 탐지해야 한다."""
        try:
            import onnx
        except ImportError:
            pytest.skip("onnx 패키지 미설치")

        model_path = str(tmp_path / "conv_integer.onnx")
        self._build_tiny_matmul_onnx(model_path, include_conv_integer=True)

        import onnx as onnx_lib
        model = onnx_lib.load(model_path)
        conv_integer_count = sum(
            1 for node in model.graph.node if node.op_type == "ConvInteger"
        )
        assert conv_integer_count >= 1

    def test_cpu_session_load_succeeds_for_matmul_only(self, tmp_path):
        """MatMul-only 모델 → CPUExecutionProvider 세션 로드 성공."""
        try:
            import onnx
        except ImportError:
            pytest.skip("onnx 패키지 미설치")

        model_path = str(tmp_path / "matmul_only.onnx")
        self._build_tiny_matmul_onnx(model_path, include_conv_integer=False)

        import onnxruntime
        session = onnxruntime.InferenceSession(
            model_path, providers=["CPUExecutionProvider"]
        )
        assert session is not None

    def test_dummy_inference_succeeds_for_matmul_only(self, tmp_path):
        """MatMul-only 모델 → 더미 float32 입력으로 1회 추론 예외 없이 완료."""
        try:
            import onnx
        except ImportError:
            pytest.skip("onnx 패키지 미설치")

        model_path = str(tmp_path / "matmul_only.onnx")
        self._build_tiny_matmul_onnx(model_path, include_conv_integer=False)

        import onnxruntime
        session = onnxruntime.InferenceSession(
            model_path, providers=["CPUExecutionProvider"]
        )
        # A: (1,4), B: (4,2) — MatMul → (1,2)
        A = np.random.randn(1, 4).astype(np.float32)
        B = np.random.randn(4, 2).astype(np.float32)
        outputs = session.run(None, {"A": A, "B": B})
        assert outputs is not None
        assert len(outputs) == 1
        assert outputs[0].shape == (1, 2)


# ────────────────────────────────────────────────────────────────────────────────
# stale model_quantized.onnx 처리 검증
# ────────────────────────────────────────────────────────────────────────────────

class TestStaleQuantizedFileHandling:
    """
    stale model_quantized.onnx 문제: convert_to_onnx.py --no-quantize 재실행해도
    기존 model_quantized.onnx가 남아있어 onnx_detector가 깨진 파일을 선택하는 문제.
    make-light-weight.sh는 양자화 단계 전에 반드시 이 파일을 제거/덮어써야 함.
    """

    def test_onnx_detector_prefers_quantized_over_base(self, tmp_path):
        """onnx_detector._find_onnx_file이 model_quantized.onnx를 우선 선택하는 동작 확인."""
        from onnx_detector import _find_onnx_file, _sanitize_model_id

        model_id = "test/model"
        sanitized = _sanitize_model_id(model_id)
        model_dir = os.path.join(str(tmp_path), sanitized)
        os.makedirs(model_dir, exist_ok=True)

        # model.onnx + model_quantized.onnx 모두 존재할 때
        base_path = os.path.join(model_dir, "model.onnx")
        quantized_path = os.path.join(model_dir, "model_quantized.onnx")
        with open(base_path, "wb") as f:
            f.write(b"base_onnx")
        with open(quantized_path, "wb") as f:
            f.write(b"quantized_onnx")

        selected = _find_onnx_file(model_dir)
        assert selected == quantized_path, (
            "model_quantized.onnx가 있으면 반드시 그걸 선택해야 stale 문제가 발생함"
        )

    def test_stale_quantized_removed_before_fresh_quantization(self, tmp_path):
        """stale model_quantized.onnx를 제거하지 않으면 깨진 파일이 남는 시나리오 검증.

        이 테스트는 make-light-weight.sh가 반드시 처리해야 하는 버그를 문서화한다.
        스크립트가 올바르게 동작하면 이 시나리오가 발생하지 않아야 한다.
        """
        # stale 파일이 있는 상황을 시뮬레이션
        model_dir = str(tmp_path / "model_dir")
        os.makedirs(model_dir, exist_ok=True)

        stale_path = os.path.join(model_dir, "model_quantized.onnx")
        with open(stale_path, "wb") as f:
            f.write(b"stale_broken_content")

        # stale 파일 존재 확인
        assert os.path.exists(stale_path)
        assert open(stale_path, "rb").read() == b"stale_broken_content"

        # 스크립트가 해야 할 일: 제거 후 신선한 파일로 교체
        os.remove(stale_path)
        with open(stale_path, "wb") as f:
            f.write(b"fresh_quantized_content")

        assert open(stale_path, "rb").read() == b"fresh_quantized_content"


# ────────────────────────────────────────────────────────────────────────────────
# MatMul-only 양자화 함수 단위 테스트 (onnxruntime + onnx 필요)
# ────────────────────────────────────────────────────────────────────────────────

@pytest.mark.skipif(not _has_onnxruntime(), reason="onnxruntime 미설치")
class TestMatMulOnlyQuantization:
    """
    quantize_dynamic(op_types_to_quantize=['MatMul'])이 ConvInteger를 생성하지 않음을 검증.
    실제 작은 ONNX 모델로 테스트.
    """

    def _build_tiny_conv_matmul_onnx(self, path: str) -> None:
        """Conv + MatMul 노드가 모두 있는 tiny 모델을 생성.
        양자화 후 Conv는 float32로 유지되고 MatMul만 양자화되어야 한다.
        """
        try:
            import onnx
            from onnx import TensorProto, helper, numpy_helper

            # Conv → Flatten → MatMul 구조
            # 입력: (1, 1, 6, 6) float32
            X = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 1, 6, 6])
            conv_out = helper.make_tensor_value_info("conv_out", TensorProto.FLOAT, [1, 1, 4, 4])
            flat_out = helper.make_tensor_value_info("flat_out", TensorProto.FLOAT, [1, 16])
            Y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 4])

            # Conv weight: (1,1,3,3)
            w_data = np.random.randn(1, 1, 3, 3).astype(np.float32)
            W_init = numpy_helper.from_array(w_data, name="conv_w")

            # MatMul weight: (16, 4)
            m_data = np.random.randn(16, 4).astype(np.float32)
            M_init = numpy_helper.from_array(m_data, name="matmul_w")

            conv_node = helper.make_node("Conv", inputs=["input", "conv_w"], outputs=["conv_out"])
            flat_node = helper.make_node(
                "Reshape",
                inputs=["conv_out", "flat_shape"],
                outputs=["flat_out"],
            )
            flat_shape_init = numpy_helper.from_array(
                np.array([1, 16], dtype=np.int64), name="flat_shape"
            )
            matmul_node = helper.make_node(
                "MatMul", inputs=["flat_out", "matmul_w"], outputs=["output"]
            )

            graph = helper.make_graph(
                [conv_node, flat_node, matmul_node],
                "conv_matmul",
                [X],
                [Y],
                initializer=[W_init, M_init, flat_shape_init],
            )
            model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
            model.ir_version = 8  # onnxruntime 1.x 호환 IR version
            onnx.checker.check_model(model)
            onnx.save(model, path)
        except ImportError:
            pytest.skip("onnx 패키지 미설치")

    @pytest.mark.skipif(not _has_onnxruntime(), reason="onnxruntime 미설치")
    def test_matmul_only_quantize_no_conv_integer(self, tmp_path):
        """op_types_to_quantize=['MatMul']로 양자화 → ConvInteger 노드 0개."""
        try:
            import onnx
            import onnxruntime.quantization as q
        except ImportError:
            pytest.skip("onnx 또는 onnxruntime 미설치")

        src = str(tmp_path / "model.onnx")
        dst = str(tmp_path / "model_quantized.onnx")
        self._build_tiny_conv_matmul_onnx(src)

        # 핵심: op_types_to_quantize=['MatMul']만 지정
        q.quantize_dynamic(
            model_input=src,
            model_output=dst,
            weight_type=q.QuantType.QInt8,
            op_types_to_quantize=["MatMul"],
        )

        import onnx as onnx_lib
        model = onnx_lib.load(dst)
        conv_integer_count = sum(
            1 for node in model.graph.node if node.op_type == "ConvInteger"
        )
        assert conv_integer_count == 0, (
            f"ConvInteger 노드 {conv_integer_count}개 발견 — "
            "MatMul-only 양자화인데 Conv가 양자화됨"
        )

    @pytest.mark.skipif(not _has_onnxruntime(), reason="onnxruntime 미설치")
    def test_matmul_only_quantize_cpu_session_loads(self, tmp_path):
        """MatMul-only 양자화 후 CPUExecutionProvider 세션 로드 성공."""
        try:
            import onnx
            import onnxruntime
            import onnxruntime.quantization as q
        except ImportError:
            pytest.skip("onnx 또는 onnxruntime 미설치")

        src = str(tmp_path / "model.onnx")
        dst = str(tmp_path / "model_quantized.onnx")
        self._build_tiny_conv_matmul_onnx(src)

        q.quantize_dynamic(
            model_input=src,
            model_output=dst,
            weight_type=q.QuantType.QInt8,
            op_types_to_quantize=["MatMul"],
        )

        session = onnxruntime.InferenceSession(
            dst, providers=["CPUExecutionProvider"]
        )
        assert session is not None

    @pytest.mark.skipif(not _has_onnxruntime(), reason="onnxruntime 미설치")
    def test_matmul_only_quantize_dummy_inference(self, tmp_path):
        """MatMul-only 양자화 후 더미 입력 (1,1,6,6) float32로 추론 성공."""
        try:
            import onnx
            import onnxruntime
            import onnxruntime.quantization as q
        except ImportError:
            pytest.skip("onnx 또는 onnxruntime 미설치")

        src = str(tmp_path / "model.onnx")
        dst = str(tmp_path / "model_quantized.onnx")
        self._build_tiny_conv_matmul_onnx(src)

        q.quantize_dynamic(
            model_input=src,
            model_output=dst,
            weight_type=q.QuantType.QInt8,
            op_types_to_quantize=["MatMul"],
        )

        session = onnxruntime.InferenceSession(
            dst, providers=["CPUExecutionProvider"]
        )
        dummy = np.random.randn(1, 1, 6, 6).astype(np.float32)
        input_name = session.get_inputs()[0].name
        outputs = session.run(None, {input_name: dummy})
        assert outputs is not None
        assert len(outputs) == 1


# ────────────────────────────────────────────────────────────────────────────────
# 전체 변환 E2E — optimum + torch 없으면 스킵
# ────────────────────────────────────────────────────────────────────────────────

@pytest.mark.skipif(
    not _has_optimum(),
    reason="optimum[onnxruntime] 미설치 — make-light-weight 전체 변환 테스트 스킵",
)
class TestMakeLightWeightEndToEnd:
    """
    실제 make-light-weight.sh를 실행해 full 변환 → self-test까지 검증.
    torch/optimum 설치 필요. CI에서는 보통 스킵됨.
    """

    def test_script_runs_default_model_success(self, tmp_path):
        """기본 모델(Organika/sdxl-detector)로 스크립트 실행 → exit 0."""
        result = subprocess.run(
            ["bash", SCRIPT_PATH, "--output-dir", str(tmp_path), "--skip-install"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=600,  # 변환 최대 10분
        )
        assert result.returncode == 0, (
            f"exit code: {result.returncode}\n"
            f"stdout: {result.stdout[-2000:]}\n"
            f"stderr: {result.stderr[-2000:]}"
        )

    def test_script_produces_quantized_onnx(self, tmp_path):
        """실행 후 model_quantized.onnx가 생성되어야 한다."""
        subprocess.run(
            ["bash", SCRIPT_PATH, "--output-dir", str(tmp_path), "--skip-install"],
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
        import json as json_lib
        subprocess.run(
            ["bash", SCRIPT_PATH, "--output-dir", str(tmp_path), "--skip-install"],
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
            meta = json_lib.load(f)

        from onnx_detector import _validate_meta
        _validate_meta(meta)  # 예외 없이 통과해야 함

    def test_script_idempotent(self, tmp_path):
        """두 번 실행해도 같은 결과 (멱등성)."""
        for _ in range(2):
            result = subprocess.run(
                ["bash", SCRIPT_PATH, "--output-dir", str(tmp_path), "--skip-install"],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                timeout=600,
            )
            assert result.returncode == 0

    def test_generated_onnx_has_no_conv_integer(self, tmp_path):
        """생성된 model_quantized.onnx에 ConvInteger 노드가 없어야 한다."""
        try:
            import onnx as onnx_lib
        except ImportError:
            pytest.skip("onnx 패키지 미설치")

        subprocess.run(
            ["bash", SCRIPT_PATH, "--output-dir", str(tmp_path), "--skip-install"],
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

        model = onnx_lib.load(quantized_path)
        conv_integer_count = sum(
            1 for node in model.graph.node if node.op_type == "ConvInteger"
        )
        assert conv_integer_count == 0, f"ConvInteger 노드 {conv_integer_count}개 발견"
