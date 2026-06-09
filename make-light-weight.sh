#!/usr/bin/env bash
# make-light-weight.sh
#
# 모든 CPU / 모든 플랫폼에서 동작하는 경량 ONNX 모델을 생성하는 빌드 스크립트.
#
# 핵심 전략:
#   quantize_dynamic(op_types_to_quantize=['MatMul']) — Conv를 양자화 제외.
#   Swin Transformer 계열 모델은 patch embedding에 Conv가 있어 전체 양자화 시
#   ConvInteger 연산자가 생성되는데, onnxruntime CPU ExecutionProvider가 이를
#   지원하지 않아 세션 로드가 실패한다. MatMul만 양자화하면 Conv는 float32로
#   유지되므로 모든 플랫폼 CPU EP에서 로드·추론이 정상 동작한다.
#
# 사용법:
#   ./make-light-weight.sh [옵션] [모델ID ...]
#
#   옵션:
#     -o, --output-dir DIR    ONNX 모델 저장 디렉토리 (기본: onnx_models)
#     -s, --skip-install      pip install 단계 건너뜀
#     -h, --help              이 도움말 출력
#
#   모델ID:
#     HuggingFace 모델 ID (기본: Organika/sdxl-detector)
#     여러 개 지정 가능: ./make-light-weight.sh model/a model/b
#
# 예시:
#   ./make-light-weight.sh
#   ./make-light-weight.sh Organika/sdxl-detector
#   ./make-light-weight.sh -o /opt/onnx_models Organika/sdxl-detector
#   ./make-light-weight.sh --skip-install Organika/sdxl-detector
#   ./make-light-weight.sh Organika/sdxl-detector umm-maybe/AI-image-detector

set -euo pipefail

# ── 상수 ──────────────────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_OUTPUT_DIR="onnx_models"
DEFAULT_MODELS=("Organika/sdxl-detector")

# ── 인자 파싱 ──────────────────────────────────────────────────────────────────

usage() {
    grep '^#' "$0" | grep -v '^#!' | sed 's/^# \{0,2\}//'
    exit 0
}

OUTPUT_DIR="${DEFAULT_OUTPUT_DIR}"
SKIP_INSTALL=false
MODELS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            usage
            ;;
        -o|--output-dir)
            shift
            OUTPUT_DIR="${1:?--output-dir 뒤에 디렉토리 경로를 지정하세요}"
            shift
            ;;
        -s|--skip-install)
            SKIP_INSTALL=true
            shift
            ;;
        -*)
            echo "[오류] 알 수 없는 옵션: $1" >&2
            echo "  사용법: $0 --help" >&2
            exit 1
            ;;
        *)
            MODELS+=("$1")
            shift
            ;;
    esac
done

# 모델 미지정 시 기본 모델 사용
if [[ ${#MODELS[@]} -eq 0 ]]; then
    MODELS=("${DEFAULT_MODELS[@]}")
fi

# ── 헬퍼 함수 ──────────────────────────────────────────────────────────────────

info()  { echo "[INFO] $*"; }
ok()    { echo "[OK]   $*"; }
warn()  { echo "[경고] $*" >&2; }
error() { echo "[오류] $*" >&2; exit 1; }

# 모델 ID → 파일시스템 안전한 경로명 (/ → __)
sanitize_model_id() {
    echo "${1//\//__}"
}

# 사람이 읽기 좋은 파일 크기 (KB/MB)
human_size() {
    local path="$1"
    if [[ -f "$path" ]]; then
        local bytes
        bytes=$(wc -c < "$path" | tr -d ' ')
        if (( bytes >= 1048576 )); then
            awk "BEGIN { printf \"%.1f MB\", ${bytes}/1048576 }"
        else
            awk "BEGIN { printf \"%.0f KB\", ${bytes}/1024 }"
        fi
    else
        echo "N/A"
    fi
}

# ── 의존성 설치 ───────────────────────────────────────────────────────────────

install_deps() {
    local req_file="${SCRIPT_DIR}/requirements-convert.txt"
    if [[ ! -f "$req_file" ]]; then
        error "requirements-convert.txt 없음: ${req_file}"
    fi
    info "변환 의존성 설치 중 (requirements-convert.txt)..."
    python -m pip install -q -r "$req_file"
    ok "의존성 설치 완료"
}

# ── 모델별 처리 ───────────────────────────────────────────────────────────────

process_model() {
    local model_id="$1"
    local sanitized
    sanitized="$(sanitize_model_id "${model_id}")"
    local model_dir="${OUTPUT_DIR}/${sanitized}"

    echo ""
    info "=========================================="
    info "모델: ${model_id}"
    info "출력 디렉토리: ${model_dir}"
    info "=========================================="

    # ── 1단계: ONNX export + meta.json (기존 convert_to_onnx.py 재사용) ──────

    info "[1/4] ONNX export + meta.json 생성 중..."
    python "${SCRIPT_DIR}/convert_to_onnx.py" \
        --model "${model_id}" \
        --output-dir "${OUTPUT_DIR}" \
        --no-quantize
    ok "ONNX export 완료: ${model_dir}/model.onnx ($(human_size "${model_dir}/model.onnx"))"

    # ── 2단계: stale model_quantized.onnx 제거 ───────────────────────────────

    info "[2/4] stale 양자화 파일 제거 중..."
    local quantized_path="${model_dir}/model_quantized.onnx"
    if [[ -f "${quantized_path}" ]]; then
        rm -f "${quantized_path}"
        info "  기존 model_quantized.onnx 삭제 완료 (stale 방지)"
    else
        info "  기존 model_quantized.onnx 없음 — 건너뜀"
    fi

    # ── 3단계: MatMul-only 동적 양자화 ──────────────────────────────────────

    info "[3/4] MatMul-only INT8 dynamic quantization 적용 중..."
    info "  전략: op_types_to_quantize=['MatMul'] — Conv는 float32로 유지"
    info "  근거: ConvInteger 미구현(NOT_IMPLEMENTED) → onnxruntime CPU EP 로드 실패 방지"

    python - <<'PYEOF'
import sys, os

model_dir  = os.environ["MODEL_DIR"]
src        = os.path.join(model_dir, "model.onnx")
dst        = os.path.join(model_dir, "model_quantized.onnx")

try:
    import onnxruntime.quantization as q
except ImportError:
    print("[오류] onnxruntime.quantization 임포트 실패. requirements-convert.txt 설치 필요.", file=sys.stderr)
    sys.exit(1)

q.quantize_dynamic(
    model_input=src,
    model_output=dst,
    weight_type=q.QuantType.QInt8,
    op_types_to_quantize=["MatMul"],   # Conv 제외 — ConvInteger 생성 차단
)
print(f"양자화 완료: {dst}")
PYEOF

    if [[ ! -f "${quantized_path}" ]]; then
        error "model_quantized.onnx 생성 실패"
    fi
    ok "양자화 완료: ${quantized_path} ($(human_size "${quantized_path}"))"

    # 크기 절감 비율 출력
    local base_path="${model_dir}/model.onnx"
    if [[ -f "${base_path}" ]]; then
        local base_bytes quantized_bytes
        base_bytes=$(wc -c < "${base_path}" | tr -d ' ')
        quantized_bytes=$(wc -c < "${quantized_path}" | tr -d ' ')
        if (( base_bytes > 0 )); then
            awk "BEGIN { printf \"  크기 절감: %.0f%% (%.1f MB → %.1f MB)\n\",
                (1 - ${quantized_bytes}/${base_bytes})*100,
                ${base_bytes}/1048576, ${quantized_bytes}/1048576 }"
        fi
    fi

    # ── 4단계: self-test ─────────────────────────────────────────────────────

    info "[4/4] self-test 실행 중..."

    python - <<PYEOF
import sys, os

model_dir  = os.environ["MODEL_DIR"]
model_id   = os.environ["MODEL_ID"]
quantized_path = os.path.join(model_dir, "model_quantized.onnx")

fail = False

# 검증 1: ConvInteger 노드 0개인지
try:
    import onnx as onnx_lib
    model = onnx_lib.load(quantized_path)
    conv_integer_count = sum(1 for node in model.graph.node if node.op_type == "ConvInteger")
    if conv_integer_count > 0:
        print(f"[self-test 실패] ConvInteger 노드 {conv_integer_count}개 발견", file=sys.stderr)
        print("  원인: MatMul-only 양자화가 적용되지 않았거나 모델에 Conv 레이어가 남아있음", file=sys.stderr)
        fail = True
    else:
        print(f"[self-test] ConvInteger 노드: 0개 (정상)")
except ImportError:
    print("[self-test] onnx 패키지 미설치 — ConvInteger 노드 검사 스킵 (경고)")

# 검증 2: CPUExecutionProvider 세션 로드
try:
    import onnxruntime
    session = onnxruntime.InferenceSession(
        quantized_path,
        providers=["CPUExecutionProvider"],
    )
    print("[self-test] CPUExecutionProvider 세션 로드: 성공")
except Exception as e:
    print(f"[self-test 실패] CPUExecutionProvider 세션 로드 실패: {e}", file=sys.stderr)
    fail = True

# 검증 3: 더미 입력으로 1회 추론
if not fail:
    try:
        import numpy as np
        inp = session.get_inputs()[0]
        # 동적 차원(-1)을 1로 고정, 기본 이미지 크기 224
        shape = []
        for d in inp.shape:
            if isinstance(d, int) and d > 0:
                shape.append(d)
            else:
                shape.append(1 if len(shape) == 0 else 3 if len(shape) == 1 else 224)
        # BCHW 보정
        if len(shape) == 4 and shape[1] not in (1, 3):
            shape = [1, 3, 224, 224]
        dummy = np.random.randn(*shape).astype(np.float32)
        outputs = session.run(None, {inp.name: dummy})
        print(f"[self-test] 더미 추론 (shape={tuple(shape)}): 성공 (출력 shape={outputs[0].shape})")
    except Exception as e:
        print(f"[self-test 실패] 더미 추론 실패: {e}", file=sys.stderr)
        fail = True

if fail:
    sys.exit(1)
PYEOF

    ok "self-test 모두 통과"
    ok "=========================================="
    ok "빌드 완료: ${model_dir}/model_quantized.onnx"
    ok "  - ConvInteger 노드 0개 (모든 CPU/플랫폼 호환)"
    ok "  - onnxruntime CPUExecutionProvider 로드 성공"
    ok "  - 더미 추론 성공"
    ok "=========================================="
}

# ── 메인 ──────────────────────────────────────────────────────────────────────

main() {
    info "make-light-weight.sh 시작"
    info "출력 디렉토리: ${OUTPUT_DIR}"
    info "대상 모델: ${MODELS[*]}"
    echo ""

    # 의존성 설치
    if [[ "${SKIP_INSTALL}" == "false" ]]; then
        install_deps
    else
        info "의존성 설치 건너뜀 (--skip-install)"
    fi

    # 출력 디렉토리 생성
    mkdir -p "${OUTPUT_DIR}"

    # 모델별 처리 (MODEL_DIR / MODEL_ID 환경변수로 Python 내부 스크립트에 전달)
    local failed_models=()
    for model_id in "${MODELS[@]}"; do
        local sanitized
        sanitized="$(sanitize_model_id "${model_id}")"
        export MODEL_DIR="${OUTPUT_DIR}/${sanitized}"
        export MODEL_ID="${model_id}"

        if process_model "${model_id}"; then
            :
        else
            warn "모델 처리 실패: ${model_id}"
            failed_models+=("${model_id}")
        fi
    done

    echo ""
    if [[ ${#failed_models[@]} -gt 0 ]]; then
        error "다음 모델 처리 실패: ${failed_models[*]}"
    fi

    info "전체 완료. 생성 모델 수: ${#MODELS[@]}"
    info "ONNX 런타임 의존성 설치: pip install -r requirements-onnx.txt"
    info "추론 사용법: python detect.py image.jpg --backend onnx --onnx-models-dir ${OUTPUT_DIR}"
}

main
