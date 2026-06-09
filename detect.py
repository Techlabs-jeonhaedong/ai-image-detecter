#!/usr/bin/env python3
"""
AI 이미지 탐지 CLI.

사용법:
  python detect.py <이미지경로> [이미지경로...] [옵션]

옵션:
  --json                결과를 JSON 배열로 출력
  --model <model_id>    사용할 HuggingFace 모델 ID (반복 가능). 기본: Organika/sdxl-detector
  --ensemble            사전 정의된 모델 세트(ENSEMBLE_MODELS) 전체 사용
  --threshold <0~1>     AI 판정 임계값 (기본: 0.5)
  --no-metadata         메타데이터/출처 검사 비활성화

--model과 --ensemble 동시 지정:
  둘을 합쳐서 중복 없이 모든 모델을 사용한다.
  예: --ensemble --model extra/model → ENSEMBLE_MODELS + extra/model (중복 제거)

기본 백엔드: onnx (번들된 경량 ONNX 모델 사용, onnxruntime만 필요)
  torch 백엔드 사용 시: --backend torch (transformers 설치 필요)

예시:
  python detect.py photo.jpg
  python detect.py img1.jpg img2.png --json
  python detect.py photo.jpg --model A --model B
  python detect.py photo.jpg --ensemble
  python detect.py photo.jpg --ensemble --threshold 0.7
  python detect.py photo.jpg --no-metadata
  python detect.py photo.jpg --backend torch
"""

import argparse
import json
import os
import sys

from backends import DEFAULT_ONNX_MODELS_DIR
from detector import DEFAULT_MODEL

DEFAULT_THRESHOLD = 0.5


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="detect.py",
        description="AI 생성 이미지 탐지 CLI (앙상블·메타데이터 검사 지원)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "images",
        nargs="+",
        metavar="IMAGE",
        help="분석할 이미지 파일 경로 (여러 개 가능)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="json_mode",
        help="결과를 JSON 배열로 출력",
    )
    parser.add_argument(
        "--model",
        action="append",
        dest="models",
        metavar="MODEL_ID",
        help=f"HuggingFace 모델 ID (반복 가능). 기본: {DEFAULT_MODEL}",
    )
    parser.add_argument(
        "--ensemble",
        action="store_true",
        help="사전 정의된 앙상블 모델 세트(ENSEMBLE_MODELS) 전체 사용",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        metavar="FLOAT",
        help=f"AI 판정 임계값 0~1 (기본: {DEFAULT_THRESHOLD})",
    )
    parser.add_argument(
        "--no-metadata",
        action="store_true",
        dest="no_metadata",
        help="메타데이터/출처 검사 비활성화",
    )
    parser.add_argument(
        "--backend",
        choices=["torch", "onnx"],
        default="onnx",
        help="추론 백엔드 선택: onnx(기본, 번들 경량 모델·onnxruntime만 필요) 또는 torch(transformers 필요)",
    )
    parser.add_argument(
        "--onnx-models-dir",
        default=DEFAULT_ONNX_MODELS_DIR,
        metavar="DIR",
        dest="onnx_models_dir",
        help=f"ONNX 모델 디렉토리 (기본: {DEFAULT_ONNX_MODELS_DIR})",
    )
    return parser


def _validate_threshold(value: float) -> None:
    if not (0.0 <= value <= 1.0):
        raise argparse.ArgumentTypeError(
            f"--threshold 값은 0.0 ~ 1.0 사이여야 합니다. 입력값: {value}"
        )


def _get_pipeline_fn(backend: str = "onnx", onnx_models_dir: str = DEFAULT_ONNX_MODELS_DIR):
    """
    환경변수 _AI_DETECTOR_MOCK=1 이면 mock pipeline 반환 (테스트용).
    그 외에는 backend에 따라 실제 pipeline 반환.
    백엔드 선택 로직은 backends.get_pipeline_fn_with_mock으로 위임.
    """
    from backends import get_pipeline_fn_with_mock
    return get_pipeline_fn_with_mock(backend, onnx_models_dir)


def _resolve_models(args) -> list:
    """
    --model / --ensemble 인자를 조합해 최종 모델 목록(중복 제거)을 반환한다.

    --ensemble과 --model 동시 지정 시 둘을 합친다.
    둘 다 없으면 기본 모델 하나를 반환한다.

    onnx 백엔드 + 비mock 환경일 때:
      ensemble 유래 모델 중 번들 안 된 것은 자동 제외하고 stderr에 경고 출력.
      --model로 명시된 모델은 제외하지 않는다 (사용자 의도 보존).
    """
    from detector import ENSEMBLE_MODELS

    ensemble_models = []
    explicit_models = []

    if args.ensemble:
        ensemble_models.extend(ENSEMBLE_MODELS)

    if args.models:
        for m in args.models:
            if m not in ensemble_models and m not in explicit_models:
                explicit_models.append(m)

    # onnx 백엔드 + 비mock 환경: ensemble 유래 모델만 필터링
    backend = getattr(args, "backend", "onnx")
    is_mock = os.environ.get("_AI_DETECTOR_MOCK") == "1"
    if ensemble_models and backend == "onnx" and not is_mock:
        from onnx_detector import is_model_available
        onnx_models_dir = getattr(args, "onnx_models_dir", DEFAULT_ONNX_MODELS_DIR)
        unavailable = [m for m in ensemble_models if not is_model_available(m, onnx_models_dir)]
        if unavailable:
            names = ", ".join(unavailable)
            sys.stderr.write(
                f"WARNING: 다음 ensemble 모델이 onnx 번들에 없음; "
                f"--backend torch 또는 setup.py로 변환 필요: {names}\n"
            )
            ensemble_models = [m for m in ensemble_models if m not in unavailable]

    # 중복 제거 후 합치기
    model_set = list(ensemble_models)
    for m in explicit_models:
        if m not in model_set:
            model_set.append(m)

    if not model_set:
        model_set = [DEFAULT_MODEL]

    return model_set


def main(argv=None):
    parser = _build_parser()

    # 인자 없이 실행 시 usage 출력 후 종료
    if argv is not None:
        args = parser.parse_args(argv)
    else:
        if len(sys.argv) == 1:
            parser.print_usage(sys.stderr)
            sys.stderr.write("error: 이미지 경로를 하나 이상 입력하세요.\n")
            sys.exit(2)
        args = parser.parse_args()

    # threshold 유효성 검사 (argparse type=float 이후 범위 체크)
    try:
        _validate_threshold(args.threshold)
    except argparse.ArgumentTypeError as e:
        sys.stderr.write(f"error: {e}\n")
        sys.exit(2)

    from detector import analyze_images_batch, _apply_metadata_override, format_result
    from metadata import inspect_metadata

    pipeline_fn = _get_pipeline_fn(
        backend=args.backend,
        onnx_models_dir=args.onnx_models_dir,
    )
    models = _resolve_models(args)

    # 모델별 pipeline 1회 생성 — 모든 이미지에 재사용 (E 항목)
    raw_results = analyze_images_batch(
        image_paths=args.images,
        pipeline_fn=pipeline_fn,
        model_ids=models,
        threshold=args.threshold,
    )

    # 단일 모델이면 model 필드를 모델 ID 자체로 표시 (하위 호환)
    if len(models) == 1:
        for r in raw_results:
            r["model"] = models[0]

    results = []
    for image_path, result in zip(args.images, raw_results):
        # 메타데이터 검사 및 결정적 신호 기반 override (B 항목)
        if not args.no_metadata:
            meta = inspect_metadata(image_path)
            result["metadata"] = meta
            result = _apply_metadata_override(result, meta)
        else:
            result["metadata"] = {
                "has_ai_signal": False,
                "decisive": False,
                "signals": [],
                "source": None,
                "checked": False,
            }
            result["verdict_source"] = "model"

        results.append(result)

    if args.json_mode:
        print(json.dumps(results, ensure_ascii=False))
    else:
        for result in results:
            print(format_result(result, json_mode=False))
            print()  # 빈 줄로 구분

    # 하나라도 에러가 있으면 exit code 1
    has_error = any(r["error"] is not None for r in results)
    sys.exit(1 if has_error else 0)


if __name__ == "__main__":
    main()
