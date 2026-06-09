"""
크로스플랫폼 ONNX 경량 빌드 스크립트.

목적: 모든 CPU / 모든 플랫폼(Linux·macOS·Windows)에서 동작하는
      경량화된 ONNX 모델을 생성한다.

setuptools 패키징 파일이 아님 — pip install -e 와 무관한 독립 빌드 스크립트다.

핵심 전략:
  quantize_dynamic(op_types_to_quantize=['MatMul']) — Conv를 양자화 제외.
  Swin Transformer 계열 모델은 patch embedding에 Conv가 있어 전체 양자화 시
  ConvInteger 연산자가 생성되는데, onnxruntime CPU ExecutionProvider가 이를
  지원하지 않아(NOT_IMPLEMENTED) 세션 로드가 실패한다.
  MatMul만 양자화하면 Conv는 float32로 유지되어 모든 CPU/플랫폼에서 로드 가능하고,
  모델 크기도 337MB → 91MB로 충분히 줄어든다(트랜스포머 가중치 대부분이 MatMul).

사용법:
    python setup.py [옵션] [모델ID ...]

    python setup.py
    python setup.py Organika/sdxl-detector
    python setup.py --output-dir /opt/onnx_models Organika/sdxl-detector
    python setup.py --skip-install Organika/sdxl-detector
    python setup.py model/a model/b

의존성:
    변환: pip install -r requirements-convert.txt
    런타임(추론 전용): pip install -r requirements-onnx.txt
"""
import argparse
import os
import sys
from pathlib import Path


# ── 상수 ──────────────────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).parent.resolve()
DEFAULT_OUTPUT_DIR = "onnx_models"
DEFAULT_MODEL = "Organika/sdxl-detector"


# ── 커스텀 예외 ───────────────────────────────────────────────────────────────

class VerificationError(Exception):
    """self-test 검증 실패 시 발생."""


# ── 인자 파서 ─────────────────────────────────────────────────────────────────

def _build_arg_parser() -> argparse.ArgumentParser:
    """인자 파서 반환 (--help 동작 보장)."""
    parser = argparse.ArgumentParser(
        prog="setup.py",
        description=(
            "크로스플랫폼 ONNX 경량 빌드 스크립트.\n"
            "모든 CPU/플랫폼에서 동작하는 MatMul-only 양자화 ONNX 모델을 생성한다.\n"
            "\n"
            "※ 이 파일은 setuptools 패키징 파일이 아니라 독립 빌드 스크립트다.\n"
            "  pip install -e 와 무관하게 동작한다.\n"
            "\n"
            "변환 의존성 설치: pip install -r requirements-convert.txt\n"
            "런타임 의존성 설치: pip install -r requirements-onnx.txt"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "models",
        nargs="*",
        default=[DEFAULT_MODEL],
        metavar="MODEL_ID",
        help=(
            f"HuggingFace 모델 ID. 여러 개 지정 가능. "
            f"(기본: {DEFAULT_MODEL})"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        metavar="DIR",
        dest="output_dir",
        help=f"ONNX 모델 저장 디렉토리 (기본: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--skip-install",
        action="store_true",
        dest="skip_install",
        help="pip install -r requirements-convert.txt 단계를 건너뜀",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        dest="force",
        help="번들 모델이 있어도 강제 재빌드 (기본: 번들 감지 시 빌드 건너뜀)",
    )
    return parser


# ── 헬퍼: 경로 유틸 ───────────────────────────────────────────────────────────

def _sanitize_model_id(model_id: str) -> str:
    """모델 ID의 '/'를 '__'로 치환해 파일시스템 안전한 이름으로 변환."""
    return model_id.replace("/", "__")


def _model_dir(output_base: str, model_id: str) -> Path:
    """모델 저장 디렉토리 경로를 반환."""
    return Path(output_base) / _sanitize_model_id(model_id)


def _human_size(path: Path) -> str:
    """파일 크기를 사람이 읽기 좋은 형식(MB/KB)으로 반환."""
    if not path.is_file():
        return "N/A"
    size = path.stat().st_size
    if size >= 1_048_576:
        return f"{size / 1_048_576:.1f} MB"
    return f"{size / 1024:.0f} KB"


# ── 단계 0: 런타임 의존성 설치 ───────────────────────────────────────────────

def install_runtime_deps() -> None:
    """requirements-onnx.txt를 pip install로 설치한다 (추론에 필요)."""
    import subprocess
    req_file = SCRIPT_DIR / "requirements-onnx.txt"
    if not req_file.is_file():
        print(f"[경고] requirements-onnx.txt 없음: {req_file} — 런타임 의존성 설치 건너뜀")
        return
    print("[0단계] 런타임 의존성 설치 중 (requirements-onnx.txt)...")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "-r", str(req_file)],
        cwd=str(SCRIPT_DIR),
    )
    if result.returncode != 0:
        print("[오류] 런타임 의존성 설치 실패", file=sys.stderr)
        sys.exit(result.returncode)
    print("[완료] 런타임 의존성 설치 완료")


# ── 단계 1: 의존성 설치 ───────────────────────────────────────────────────────

def install_deps() -> None:
    """requirements-convert.txt를 pip install로 설치한다."""
    import subprocess
    req_file = SCRIPT_DIR / "requirements-convert.txt"
    if not req_file.is_file():
        print(f"[오류] requirements-convert.txt 없음: {req_file}", file=sys.stderr)
        sys.exit(1)
    print("[1단계] 변환 의존성 설치 중 (requirements-convert.txt)...")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "-r", str(req_file)],
        cwd=str(SCRIPT_DIR),
    )
    if result.returncode != 0:
        print("[오류] 의존성 설치 실패", file=sys.stderr)
        sys.exit(result.returncode)
    print("[완료] 의존성 설치 완료")


# ── 단계 2: ONNX export + meta.json ──────────────────────────────────────────

def export_onnx(model_id: str, output_base: str) -> Path:
    """
    기존 convert_to_onnx.py --no-quantize를 재사용해 ONNX export와
    meta.json 생성을 위임한다.

    Returns:
        model.onnx 경로

    Raises:
        SystemExit: convert_to_onnx.py 실패 시
    """
    import subprocess
    convert_script = SCRIPT_DIR / "convert_to_onnx.py"
    print(f"[2단계] ONNX export + meta.json 생성 중 ({model_id})...")
    result = subprocess.run(
        [
            sys.executable,
            str(convert_script),
            "--model", model_id,
            "--output-dir", output_base,
            "--no-quantize",
        ],
        cwd=str(SCRIPT_DIR),
    )
    if result.returncode != 0:
        print(f"[오류] ONNX export 실패 (model={model_id})", file=sys.stderr)
        sys.exit(result.returncode)

    onnx_path = _model_dir(output_base, model_id) / "model.onnx"
    print(f"[완료] ONNX export: {onnx_path} ({_human_size(onnx_path)})")
    return onnx_path


# ── 단계 3: stale 파일 제거 ──────────────────────────────────────────────────

def remove_stale_quantized(quantized_path: str) -> None:
    """
    기존 model_quantized.onnx를 제거한다.

    onnx_detector._find_onnx_file은 model_quantized.onnx를 우선 선택하므로
    새 양자화 전에 반드시 stale 파일을 제거해야 한다.
    파일이 없으면 아무것도 하지 않는다 (멱등).

    Args:
        quantized_path: 제거할 model_quantized.onnx 경로 (문자열)
    """
    path = Path(quantized_path)
    path.unlink(missing_ok=True)


# ── 단계 4: MatMul-only 양자화 ────────────────────────────────────────────────

def quantize_matmul_only(src: str, dst: str) -> None:
    """
    op_types_to_quantize=['MatMul']로 동적 INT8 양자화를 적용한다.

    Conv를 양자화 대상에서 제외함으로써 ConvInteger 노드 생성을 차단한다.
    onnxruntime CPU ExecutionProvider는 ConvInteger를 미구현(NOT_IMPLEMENTED)이므로
    전체 양자화 시 세션 로드가 실패한다.

    Args:
        src: 원본 model.onnx 경로
        dst: 출력 model_quantized.onnx 경로

    Raises:
        ImportError: onnxruntime.quantization 미설치 시
        FileNotFoundError: src 파일 없을 때
    """
    if not os.path.isfile(src):
        raise FileNotFoundError(f"원본 ONNX 파일 없음: {src}")

    try:
        import onnxruntime.quantization as q
    except ImportError:
        raise ImportError(
            "onnxruntime.quantization 임포트 실패.\n"
            "    pip install -r requirements-convert.txt"
        )

    q.quantize_dynamic(
        model_input=src,
        model_output=dst,
        weight_type=q.QuantType.QInt8,
        op_types_to_quantize=["MatMul"],  # Conv 제외 → ConvInteger 생성 차단
    )


# ── 단계 5: self-test 검증 ────────────────────────────────────────────────────

def verify_onnx_cpu_loadable(path: str, image_size: int = 224) -> bool:
    """
    생성된 ONNX 모델에 대해 3가지 self-test를 수행한다.

    1. ConvInteger 노드 0개인지 확인 (onnx 패키지 있을 때)
    2. CPUExecutionProvider로 InferenceSession 로드 성공
    3. 더미 입력으로 1회 session.run 추론이 예외 없이 완료

    Args:
        path: 검증할 model_quantized.onnx 경로
        image_size: 더미 입력 H/W (기본 224)

    Returns:
        True (모든 검증 통과)

    Raises:
        VerificationError: 검증 항목 하나라도 실패 시
        ImportError: onnxruntime 미설치 시
    """
    import numpy as np

    # 검증 1: ConvInteger 노드 0개 확인
    try:
        import onnx as onnx_lib
        model = onnx_lib.load(path)
        conv_integer_count = sum(
            1 for node in model.graph.node if node.op_type == "ConvInteger"
        )
        if conv_integer_count > 0:
            raise VerificationError(
                f"ConvInteger 노드 {conv_integer_count}개 발견.\n"
                "  MatMul-only 양자화가 적용되지 않았습니다."
            )
        print(f"  [검증 1/3] ConvInteger 노드: 0개 (정상)")
    except ImportError:
        print("  [검증 1/3] onnx 패키지 미설치 — ConvInteger 노드 검사 스킵 (경고)")

    # 검증 2: CPUExecutionProvider 세션 로드
    try:
        import onnxruntime
    except ImportError:
        raise ImportError(
            "onnxruntime 미설치.\n    pip install -r requirements-onnx.txt"
        )

    try:
        session = onnxruntime.InferenceSession(
            path,
            providers=["CPUExecutionProvider"],
        )
    except Exception as e:
        raise VerificationError(
            f"CPUExecutionProvider 세션 로드 실패: {e}\n"
            "  ConvInteger 노드가 남아있거나 모델이 손상되었을 수 있습니다."
        )
    print("  [검증 2/3] CPUExecutionProvider 세션 로드: 성공")

    # 검증 3: 더미 입력으로 1회 추론
    try:
        inp = session.get_inputs()[0]
        # 동적 차원(-1, None, symbolic)을 구체 값으로 고정
        shape = []
        for i, d in enumerate(inp.shape):
            if isinstance(d, int) and d > 0:
                shape.append(d)
            else:
                # 배치=1, 채널=3, H/W=image_size
                if i == 0:
                    shape.append(1)
                elif i == 1:
                    shape.append(3)
                else:
                    shape.append(image_size)
        # BCHW 보정: 채널 위치가 1이 아닌 경우 강제 보정
        if len(shape) == 4 and shape[1] not in (1, 3):
            shape = [1, 3, image_size, image_size]
        dummy = np.random.randn(*shape).astype(np.float32)
        outputs = session.run(None, {inp.name: dummy})
        print(
            f"  [검증 3/3] 더미 추론 (shape={tuple(shape)}): "
            f"성공 (출력 shape={outputs[0].shape})"
        )
    except Exception as e:
        raise VerificationError(f"더미 추론 실패: {e}")

    return True


# ── 모델 1개 처리 ─────────────────────────────────────────────────────────────

def process_model(model_id: str, output_base: str, force: bool = False) -> None:
    """
    단일 모델에 대해 4단계 빌드를 수행한다:
      2단계: ONNX export + meta.json
      3단계: stale model_quantized.onnx 제거
      4단계: MatMul-only 양자화
      5단계: self-test 검증

    번들 모델(model_quantized.onnx)이 이미 존재하고 force=False이면
    self-test 검증만 수행하고 빌드를 건너뛴다.

    Args:
        model_id: HuggingFace 모델 ID
        output_base: 기본 출력 디렉토리
        force: True이면 번들 모델이 있어도 강제 재빌드

    Raises:
        SystemExit: 각 단계 실패 시 비0 exit
        VerificationError: self-test 실패 시
    """
    model_dir = _model_dir(output_base, model_id)
    print()
    print(f"{'=' * 50}")
    print(f"모델: {model_id}")
    print(f"출력: {model_dir}")
    print(f"{'=' * 50}")

    # 번들 감지: model_quantized.onnx가 이미 있고 force가 아니면 빌드 건너뜀
    quantized_path = model_dir / "model_quantized.onnx"
    if quantized_path.exists() and not force:
        print(f"[번들 감지] 이미 번들된 경량 모델 존재: {quantized_path}")
        print("  → 빌드 건너뜀. self-test 검증만 수행합니다.")
        # meta.json에서 image_size 읽기
        meta_path = model_dir / "meta.json"
        image_size = 224
        if meta_path.is_file():
            import json
            with open(meta_path, encoding="utf-8") as f:
                meta = json.load(f)
            image_size = meta.get("crop_size") or meta.get("image_size") or 224
        try:
            verify_onnx_cpu_loadable(str(quantized_path), image_size=image_size)
        except VerificationError as e:
            print(f"[경고] 번들 모델 self-test 실패: {e}", file=sys.stderr)
            print("  → 자동 재빌드를 진행합니다.")
            # 검증 실패 시 재빌드로 fall-through
        else:
            print(f"[완료] 번들 모델 검증 성공 — 빌드 건너뜀")
            return

    # 2단계: ONNX export + meta.json (convert_to_onnx.py 재사용)
    onnx_path = export_onnx(model_id, output_base)

    # 3단계: stale model_quantized.onnx 제거
    quantized_path = model_dir / "model_quantized.onnx"
    print("[3단계] stale 양자화 파일 제거 중...")
    if quantized_path.exists():
        remove_stale_quantized(str(quantized_path))
        print("  기존 model_quantized.onnx 삭제 완료 (stale 방지)")
    else:
        print("  기존 model_quantized.onnx 없음 — 건너뜀")

    # 4단계: MatMul-only 동적 양자화
    print("[4단계] MatMul-only INT8 dynamic quantization 적용 중...")
    print("  전략: op_types_to_quantize=['MatMul'] — Conv는 float32로 유지")
    try:
        quantize_matmul_only(str(onnx_path), str(quantized_path))
    except ImportError as e:
        print(f"[오류] {e}", file=sys.stderr)
        sys.exit(1)

    if not quantized_path.is_file():
        print("[오류] model_quantized.onnx 생성 실패", file=sys.stderr)
        sys.exit(1)

    base_size = onnx_path.stat().st_size if onnx_path.is_file() else 0
    q_size = quantized_path.stat().st_size
    ratio = f"{(1 - q_size / base_size) * 100:.0f}%" if base_size else "N/A"
    print(
        f"[완료] 양자화: {quantized_path}\n"
        f"  원본: {_human_size(onnx_path)} → 양자화: {_human_size(quantized_path)}"
        f" ({ratio} 절감)"
    )

    # 5단계: self-test
    print("[5단계] self-test 실행 중...")
    # meta.json에서 image_size 읽기
    meta_path = model_dir / "meta.json"
    image_size = 224
    if meta_path.is_file():
        import json
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        image_size = meta.get("crop_size") or meta.get("image_size") or 224

    try:
        verify_onnx_cpu_loadable(str(quantized_path), image_size=image_size)
    except VerificationError as e:
        print(f"[오류] self-test 실패: {e}", file=sys.stderr)
        sys.exit(1)

    print()
    print(f"{'=' * 50}")
    print(f"[성공] 빌드 완료: {quantized_path}")
    print(f"  - model.onnx:           {_human_size(onnx_path)}")
    print(f"  - model_quantized.onnx: {_human_size(quantized_path)} ({ratio} 절감)")
    print(f"  - ConvInteger 노드: 0개 (모든 CPU/플랫폼 호환)")
    print(f"  - CPUExecutionProvider 로드: 성공")
    print(f"  - 더미 추론: 성공")
    print(f"{'=' * 50}")


# ── 메인 ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    # 빈/공백 모델 ID 제거 — 유효 모델이 없으면 (무거운) 의존성 설치 전에 즉시 종료
    models = [m for m in args.models if m and m.strip()]
    if not models:
        print("error: 유효한 모델 ID가 없습니다.", file=sys.stderr)
        sys.exit(1)

    # 출력 디렉토리 절대경로 보정
    output_base = str(Path(args.output_dir).resolve())

    print("setup.py — ONNX 경량 빌드 스크립트 시작")
    print(f"출력 디렉토리: {output_base}")
    print(f"대상 모델: {models}")

    # 번들 모델 사전 조사: 하나라도 없으면 변환 의존성 설치 필요
    build_needed = args.force or any(
        not (_model_dir(output_base, m) / "model_quantized.onnx").exists()
        for m in models
    )

    # 의존성 설치
    if not args.skip_install:
        install_runtime_deps()  # 추론에 필요 — 항상 설치
        if build_needed:
            install_deps()       # 변환에 필요 — 빌드가 필요할 때만
    else:
        print("[0단계] 의존성 설치 건너뜀 (--skip-install)")

    # 출력 디렉토리 생성
    Path(output_base).mkdir(parents=True, exist_ok=True)

    # 모델별 처리
    failed = []
    for model_id in models:
        try:
            process_model(model_id, output_base, force=args.force)
        except SystemExit:
            failed.append(model_id)
        except Exception as e:
            print(f"[오류] 모델 처리 실패 ({model_id}): {e}", file=sys.stderr)
            failed.append(model_id)

    print()
    if failed:
        print(f"[오류] 다음 모델 처리 실패: {failed}", file=sys.stderr)
        print(
            "추론 사용법: "
            f"python detect.py image.jpg --backend onnx --onnx-models-dir {output_base}"
        )
        sys.exit(1)

    print(f"[완료] 전체 완료. 처리 모델 수: {len(models)}")
    bundled_all = not build_needed
    if bundled_all:
        print(
            "\n[번들 모델 사용 안내]\n"
            "  이 저장소에는 경량 ONNX 모델이 이미 포함되어 있습니다.\n"
            "  별도 모델 준비 없이 바로 추론할 수 있습니다:\n"
            "\n"
            "    python detect.py photo.jpg\n"
            "\n"
            "  (기본 백엔드: onnx — onnxruntime만 필요, torch 불필요)\n"
            "  torch 백엔드를 사용하려면: python detect.py photo.jpg --backend torch"
        )
    else:
        print(f"런타임 의존성 설치: pip install -r requirements-onnx.txt")
        print(
            f"추론 사용법: python detect.py image.jpg\n"
            f"  (기본 백엔드 onnx — onnxruntime만 필요)\n"
            f"  옵션: --backend onnx --onnx-models-dir {output_base}"
        )


if __name__ == "__main__":
    main()
