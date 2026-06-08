"""
HuggingFace image-classification 모델 → ONNX + INT8 dynamic quantization 변환 스크립트.

빌드/개발 머신에서 1회 실행하는 도구. 런타임 의존성 아님.

의존성: pip install -r requirements-convert.txt
    (optimum[onnxruntime], torch, transformers)

사용법:
    python convert_to_onnx.py --model Organika/sdxl-detector
    python convert_to_onnx.py --model Organika/sdxl-detector --output-dir onnx_models
    python convert_to_onnx.py --model Organika/sdxl-detector --no-quantize
"""
import json
import os


def _build_arg_parser():
    """인자 파서 반환 (--help 동작 보장)."""
    import argparse
    parser = argparse.ArgumentParser(
        prog="convert_to_onnx.py",
        description=(
            "HuggingFace image-classification 모델을 ONNX + INT8 양자화로 변환한다.\n"
            "torch, transformers, optimum[onnxruntime] 필요.\n"
            "pip install -r requirements-convert.txt"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model",
        required=True,
        metavar="MODEL_ID",
        help="HuggingFace 모델 ID (예: Organika/sdxl-detector)",
    )
    parser.add_argument(
        "--output-dir",
        default="onnx_models",
        metavar="DIR",
        help="ONNX 모델 저장 디렉토리 (기본: onnx_models)",
    )
    parser.add_argument(
        "--no-quantize",
        action="store_true",
        help="INT8 양자화 생략 (ONNX export만 수행)",
    )
    parser.add_argument(
        "--quantize-arch",
        choices=["portable", "avx2", "avx512_vnni", "arm64"],
        default="portable",
        dest="quantize_arch",
        help=(
            "양자화 아키텍처 프리셋 (기본: portable — 모든 CPU 호환).\n"
            "  portable: onnxruntime quantize_dynamic, 아키텍처 무관 INT8 (기본/권장)\n"
            "  avx2    : optimum AVX2 프리셋 (Intel/AMD x86-64 최적화)\n"
            "  avx512_vnni: optimum AVX-512 VNNI 프리셋 (최신 Intel 전용)\n"
            "  arm64   : optimum ARM64 프리셋 (Apple Silicon / ARM 서버)"
        ),
    )
    return parser


def _sanitize_model_id(model_id: str) -> str:
    """모델 ID를 파일시스템 안전한 경로 이름으로 변환."""
    return model_id.replace("/", "__")


def _get_output_dir(output_base: str, model_id: str) -> str:
    """모델 저장 디렉토리 경로 반환."""
    sanitized = _sanitize_model_id(model_id)
    return os.path.join(output_base, sanitized)


def _save_meta(model_dir: str, meta: dict) -> None:
    """
    전처리·라벨 메타데이터를 model_dir/meta.json에 저장.
    런타임(onnx_detector.py)이 torch 없이 전처리/라벨매핑에 사용.

    Args:
        model_dir: 저장 디렉토리
        meta: {image_size, image_mean, image_std, do_normalize, do_rescale,
               rescale_factor, resample, id2label}
    """
    os.makedirs(model_dir, exist_ok=True)
    meta_path = os.path.join(model_dir, "meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def _extract_meta_from_model(model_id: str, model_dir: str) -> dict:
    """
    모델 디렉토리에서 전처리/라벨 메타를 추출해 dict로 반환.
    preprocessor_config.json + config.json에서 읽는다.

    size 구조에 따라 resize_mode를 결정한다:
      - {"shortest_edge": N} → resize_mode="shortest_edge", image_size=N, crop_size=crop_size(있으면)
      - {"height": H, "width": W} 또는 int → resize_mode="exact", image_size=H(또는 N)

    Args:
        model_id: HuggingFace 모델 ID
        model_dir: export된 모델 디렉토리 (preprocessor_config.json, config.json 포함)

    Returns:
        onnx_detector.preprocess_image가 소비하는 meta dict
    """
    import json as _json

    meta = {
        "image_size": 224,
        "resize_mode": "exact",
        "image_mean": [0.5, 0.5, 0.5],
        "image_std": [0.5, 0.5, 0.5],
        "do_normalize": True,
        "do_rescale": True,
        "rescale_factor": 1.0 / 255.0,
        "resample": 3,  # BICUBIC — transformers 공통 기본값
        "id2label": {},
    }

    # preprocessor_config.json 읽기
    prep_path = os.path.join(model_dir, "preprocessor_config.json")
    if os.path.exists(prep_path):
        with open(prep_path, encoding="utf-8") as f:
            prep = _json.load(f)
        size = prep.get("size", {})
        if isinstance(size, dict):
            if "shortest_edge" in size:
                meta["image_size"] = size["shortest_edge"]
                meta["resize_mode"] = "shortest_edge"
                # crop_size 추출 (dict 또는 int)
                crop_size_raw = prep.get("crop_size", None)
                if crop_size_raw is not None:
                    if isinstance(crop_size_raw, dict):
                        meta["crop_size"] = crop_size_raw.get(
                            "height", crop_size_raw.get("width", size["shortest_edge"])
                        )
                    elif isinstance(crop_size_raw, int):
                        meta["crop_size"] = crop_size_raw
                else:
                    meta["crop_size"] = size["shortest_edge"]
            else:
                meta["image_size"] = size.get("height", size.get("width", 224))
                meta["resize_mode"] = "exact"
        elif isinstance(size, int):
            meta["image_size"] = size
            meta["resize_mode"] = "exact"

        if "image_mean" in prep:
            meta["image_mean"] = prep["image_mean"]
        if "image_std" in prep:
            meta["image_std"] = prep["image_std"]
        if "do_normalize" in prep:
            meta["do_normalize"] = prep["do_normalize"]
        if "do_rescale" in prep:
            meta["do_rescale"] = prep["do_rescale"]
        if "rescale_factor" in prep:
            meta["rescale_factor"] = prep["rescale_factor"]
        if "resample" in prep:
            meta["resample"] = prep["resample"]

    # config.json에서 id2label 읽기
    config_path = os.path.join(model_dir, "config.json")
    if os.path.exists(config_path):
        with open(config_path, encoding="utf-8") as f:
            cfg = _json.load(f)
        if "id2label" in cfg:
            meta["id2label"] = {str(k): v for k, v in cfg["id2label"].items()}

    return meta


def _build_quantization_config(quantize_arch: str):
    """
    quantize_arch 선택에 따라 양자화 설정을 반환한다.

    Args:
        quantize_arch: "portable" | "avx2" | "avx512_vnni" | "arm64"

    Returns:
        (quantize_fn, config_description)
        quantize_fn: None이면 onnxruntime quantize_dynamic 사용 (portable)
                     그 외에는 (ORTQuantizer, qconfig) 튜플
    """
    if quantize_arch == "portable":
        return "portable", "onnxruntime quantize_dynamic (아키텍처 중립 INT8)"

    try:
        from optimum.onnxruntime.configuration import AutoQuantizationConfig
    except ImportError:
        raise ImportError(
            "optimum[onnxruntime]이 설치되지 않았습니다.\n"
            "    pip install -r requirements-convert.txt"
        )

    arch_map = {
        "avx2": (AutoQuantizationConfig.avx2, "AVX2 INT8"),
        "avx512_vnni": (AutoQuantizationConfig.avx512_vnni, "AVX-512 VNNI INT8"),
        "arm64": (AutoQuantizationConfig.arm64, "ARM64 INT8"),
    }
    config_fn, desc = arch_map[quantize_arch]
    qconfig = config_fn(is_static=False, per_channel=False)
    return qconfig, desc


def _convert_model(
    model_id: str,
    output_base: str,
    no_quantize: bool = False,
    quantize_arch: str = "portable",
) -> None:
    """
    모델 변환 핵심 로직 (optimum, torch 필요).

    Args:
        model_id: HuggingFace 모델 ID
        output_base: 기본 출력 디렉토리
        no_quantize: True면 양자화 생략
        quantize_arch: 양자화 아키텍처 프리셋 (기본: "portable" — 모든 CPU 호환)

    Raises:
        ImportError: optimum/torch 미설치 시 친절한 안내 메시지
    """
    try:
        from optimum.onnxruntime import ORTModelForImageClassification
    except ImportError:
        raise ImportError(
            "optimum[onnxruntime]이 설치되지 않았습니다.\n"
            "변환 전용 의존성을 설치하세요:\n"
            "    pip install -r requirements-convert.txt\n"
            "또는:\n"
            "    pip install 'optimum[onnxruntime]' torch transformers"
        )

    model_dir = _get_output_dir(output_base, model_id)
    os.makedirs(model_dir, exist_ok=True)

    print(f"[변환] 모델: {model_id}")
    print(f"[변환] 출력: {model_dir}")

    # ONNX export
    print("[변환] ONNX export 중...")
    model = ORTModelForImageClassification.from_pretrained(model_id, export=True)
    model.save_pretrained(model_dir)

    onnx_path = os.path.join(model_dir, "model.onnx")
    onnx_size = os.path.getsize(onnx_path) if os.path.exists(onnx_path) else 0
    print(f"[변환] export 완료: {onnx_size / 1024 / 1024:.1f} MB")

    if not no_quantize:
        arch_desc = f"quantize_arch={quantize_arch}"
        print(f"[변환] INT8 dynamic quantization 적용 중 ({arch_desc})...")
        try:
            qconfig_or_mode, desc = _build_quantization_config(quantize_arch)
            print(f"[변환] 양자화 모드: {desc}")

            quantized_path = os.path.join(model_dir, "model_quantized.onnx")

            if qconfig_or_mode == "portable":
                # onnxruntime 내장 quantize_dynamic — 아키텍처 프리셋 없음
                import onnxruntime.quantization as ort_quant
                ort_quant.quantize_dynamic(
                    model_input=onnx_path,
                    model_output=quantized_path,
                    weight_type=ort_quant.QuantType.QInt8,
                )
            else:
                from optimum.onnxruntime import ORTQuantizer
                quantizer = ORTQuantizer.from_pretrained(model_dir)
                quantizer.quantize(
                    save_dir=model_dir,
                    quantization_config=qconfig_or_mode,
                )

            if os.path.exists(quantized_path):
                q_size = os.path.getsize(quantized_path)
                ratio = f"{q_size / onnx_size * 100:.0f}%" if onnx_size else "N/A"
                print(
                    f"[변환] 양자화 완료: {q_size / 1024 / 1024:.1f} MB "
                    f"(원본 대비 {ratio})"
                )
            else:
                print("[변환] 양자화 파일을 찾지 못했습니다. model.onnx를 사용합니다.")
        except Exception as e:
            print(f"[경고] 양자화 실패 ({e}). ONNX 원본 파일을 사용합니다.")

    # 메타데이터 저장 (양자화 모드 기록 포함)
    print("[변환] 메타데이터 저장 중...")
    meta = _extract_meta_from_model(model_id, model_dir)
    if not no_quantize:
        meta["quantize_arch"] = quantize_arch
    _save_meta(model_dir, meta)
    print(f"[변환] meta.json 저장 완료: {os.path.join(model_dir, 'meta.json')}")
    print("[변환] 완료!")


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    _convert_model(args.model, args.output_dir, args.no_quantize, args.quantize_arch)


if __name__ == "__main__":
    main()
