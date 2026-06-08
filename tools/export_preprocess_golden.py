#!/usr/bin/env python3
"""
전처리 골든 벡터 생성 스크립트.

Python preprocess_image() 출력을 JSON으로 내보내 Dart 전처리 검증에 사용.
transformers/실모델 불필요 — onnx_detector.preprocess_image만 사용.

사용법:
    python tools/export_preprocess_golden.py
    # → flutter/ai_image_detector/test/golden/preprocess_golden.json 생성
"""
import base64
import io
import json
import os
import sys

import numpy as np
from PIL import Image

# 프로젝트 루트를 sys.path에 추가
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _PROJECT_ROOT)

from onnx_detector import preprocess_image  # noqa: E402

OUTPUT_PATH = os.path.join(
    _PROJECT_ROOT,
    "flutter", "ai_image_detector", "test", "golden",
    "preprocess_golden.json",
)

# 골든 테스트 케이스 정의
CASES = [
    {
        "name": "exact_224_square",
        "image_mode": "RGB",
        "image_size_wh": [224, 224],
        "pixel_rgb": [128, 64, 32],  # 단색
        "meta": {
            "image_size": 224,
            "resize_mode": "exact",
            "image_mean": [0.485, 0.456, 0.406],
            "image_std": [0.229, 0.224, 0.225],
            "do_rescale": True,
            "rescale_factor": 1.0 / 255.0,
            "do_normalize": True,
            "resample": 3,
            "id2label": {"0": "artificial", "1": "human"},
        },
    },
    {
        "name": "exact_224_landscape_400x300",
        "image_mode": "RGB",
        "image_size_wh": [400, 300],
        "pixel_rgb": [200, 150, 100],
        "meta": {
            "image_size": 224,
            "resize_mode": "exact",
            "image_mean": [0.485, 0.456, 0.406],
            "image_std": [0.229, 0.224, 0.225],
            "do_rescale": True,
            "rescale_factor": 1.0 / 255.0,
            "do_normalize": True,
            "resample": 3,
            "id2label": {"0": "artificial", "1": "human"},
        },
    },
    {
        "name": "exact_224_portrait_300x400",
        "image_mode": "RGB",
        "image_size_wh": [300, 400],
        "pixel_rgb": [50, 120, 200],
        "meta": {
            "image_size": 224,
            "resize_mode": "exact",
            "image_mean": [0.485, 0.456, 0.406],
            "image_std": [0.229, 0.224, 0.225],
            "do_rescale": True,
            "rescale_factor": 1.0 / 255.0,
            "do_normalize": True,
            "resample": 3,
            "id2label": {"0": "artificial", "1": "human"},
        },
    },
    {
        "name": "shortest_edge_256_crop_224_landscape",
        "image_mode": "RGB",
        "image_size_wh": [400, 300],
        "pixel_rgb": [180, 90, 45],
        "meta": {
            "image_size": 256,
            "crop_size": 224,
            "resize_mode": "shortest_edge",
            "image_mean": [0.485, 0.456, 0.406],
            "image_std": [0.229, 0.224, 0.225],
            "do_rescale": True,
            "rescale_factor": 1.0 / 255.0,
            "do_normalize": True,
            "resample": 3,
            "id2label": {"0": "artificial", "1": "human"},
        },
    },
    {
        "name": "shortest_edge_256_crop_224_portrait",
        "image_mode": "RGB",
        "image_size_wh": [300, 400],
        "pixel_rgb": [60, 120, 240],
        "meta": {
            "image_size": 256,
            "crop_size": 224,
            "resize_mode": "shortest_edge",
            "image_mean": [0.485, 0.456, 0.406],
            "image_std": [0.229, 0.224, 0.225],
            "do_rescale": True,
            "rescale_factor": 1.0 / 255.0,
            "do_normalize": True,
            "resample": 3,
            "id2label": {"0": "artificial", "1": "human"},
        },
    },
    {
        "name": "grayscale_exact_32",
        "image_mode": "L",
        "image_size_wh": [64, 64],
        "pixel_gray": 128,
        "meta": {
            "image_size": 32,
            "resize_mode": "exact",
            "image_mean": [0.5, 0.5, 0.5],
            "image_std": [0.5, 0.5, 0.5],
            "do_rescale": True,
            "rescale_factor": 1.0 / 255.0,
            "do_normalize": True,
            "resample": 3,
            "id2label": {"0": "ai"},
        },
    },
    {
        "name": "rgba_exact_32",
        "image_mode": "RGBA",
        "image_size_wh": [64, 64],
        "pixel_rgba": [200, 100, 50, 128],
        "meta": {
            "image_size": 32,
            "resize_mode": "exact",
            "image_mean": [0.5, 0.5, 0.5],
            "image_std": [0.5, 0.5, 0.5],
            "do_rescale": True,
            "rescale_factor": 1.0 / 255.0,
            "do_normalize": True,
            "resample": 3,
            "id2label": {"0": "ai"},
        },
    },
    # -----------------------------------------------------------------
    # #7: 1px-차이 유발 해상도 (portrait, w<=h)
    #   w=25, h=32, size=256 → new_h = int(32*256/25) = int(327.68) = 327
    #   Dart round() 버그: round(327.68) = 328  ← 수정 전에 shape 불일치
    # -----------------------------------------------------------------
    {
        "name": "shortest_edge_1px_diff_portrait_25x32",
        "image_mode": "RGB",
        "image_size_wh": [25, 32],
        "pixel_rgb": [100, 150, 200],
        "meta": {
            "image_size": 256,
            "crop_size": 224,
            "resize_mode": "shortest_edge",
            "image_mean": [0.485, 0.456, 0.406],
            "image_std": [0.229, 0.224, 0.225],
            "do_rescale": True,
            "rescale_factor": 1.0 / 255.0,
            "do_normalize": True,
            "resample": 3,
            "id2label": {"0": "artificial", "1": "human"},
        },
    },
    # -----------------------------------------------------------------
    # #8: 1px-차이 유발 해상도 (landscape, w>h)
    #   w=64, h=50, size=256 → new_w = int(64*256/50) = int(327.68) = 327
    #   Dart round() 버그: round(327.68) = 328  ← 수정 전에 shape 불일치
    # -----------------------------------------------------------------
    {
        "name": "shortest_edge_1px_diff_landscape_64x50",
        "image_mode": "RGB",
        "image_size_wh": [64, 50],
        "pixel_rgb": [80, 120, 180],
        "meta": {
            "image_size": 256,
            "crop_size": 224,
            "resize_mode": "shortest_edge",
            "image_mean": [0.485, 0.456, 0.406],
            "image_std": [0.229, 0.224, 0.225],
            "do_rescale": True,
            "rescale_factor": 1.0 / 255.0,
            "do_normalize": True,
            "resample": 3,
            "id2label": {"0": "artificial", "1": "human"},
        },
    },
    # -----------------------------------------------------------------
    # #9: 비단색 그라데이션 이미지 (exact 모드)
    #   가로 그라데이션: x축으로 0→255 변화, 단색 케이스에서 못 잡는 보간 차이 실측
    # -----------------------------------------------------------------
    {
        "name": "gradient_exact_32",
        "image_mode": "RGB",
        "image_size_wh": [64, 64],
        "pixel_rgb": None,  # gradient 케이스: _make_image 에서 분기 처리
        "meta": {
            "image_size": 32,
            "resize_mode": "exact",
            "image_mean": [0.485, 0.456, 0.406],
            "image_std": [0.229, 0.224, 0.225],
            "do_rescale": True,
            "rescale_factor": 1.0 / 255.0,
            "do_normalize": True,
            "resample": 3,
            "id2label": {"0": "artificial", "1": "human"},
        },
        "_gradient": True,
    },
    # -----------------------------------------------------------------
    # #10: 비단색 그라데이션 이미지 (shortest_edge 모드)
    #   1px-차이 케이스와 그라데이션 조합 → 보간 허용오차 실측
    # -----------------------------------------------------------------
    {
        "name": "gradient_shortest_edge_256_crop_224",
        "image_mode": "RGB",
        "image_size_wh": [300, 400],
        "pixel_rgb": None,  # gradient 케이스
        "meta": {
            "image_size": 256,
            "crop_size": 224,
            "resize_mode": "shortest_edge",
            "image_mean": [0.485, 0.456, 0.406],
            "image_std": [0.229, 0.224, 0.225],
            "do_rescale": True,
            "rescale_factor": 1.0 / 255.0,
            "do_normalize": True,
            "resample": 3,
            "id2label": {"0": "artificial", "1": "human"},
        },
        "_gradient": True,
    },
]


def _make_image(case: dict) -> Image.Image:
    """케이스 정의에서 PIL Image 생성 (단색 또는 그라데이션)."""
    w, h = case["image_size_wh"]
    mode = case["image_mode"]

    if case.get("_gradient"):
        # 가로·세로 그라데이션 패턴 (R=x방향, G=y방향, B=고정 100)
        arr = np.zeros((h, w, 3), dtype=np.uint8)
        for y in range(h):
            for x in range(w):
                arr[y, x, 0] = int(x * 255 / max(w - 1, 1))   # R: 0→255 (x방향)
                arr[y, x, 1] = int(y * 255 / max(h - 1, 1))   # G: 0→255 (y방향)
                arr[y, x, 2] = 100                              # B: 고정
        return Image.fromarray(arr, mode="RGB")

    if mode == "L":
        return Image.new("L", (w, h), case["pixel_gray"])
    elif mode == "RGBA":
        r, g, b, a = case["pixel_rgba"]
        return Image.new("RGBA", (w, h), (r, g, b, a))
    else:  # RGB
        r, g, b = case["pixel_rgb"]
        return Image.new("RGB", (w, h), (r, g, b))


def _image_to_png_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _tensor_summary(arr: np.ndarray) -> dict:
    """텐서 요약 통계 + 일부 인덱스 값 (전체 저장 대신 크기 절약)."""
    flat = arr.flatten()
    # 비교를 위한 샘플 인덱스: 첫 10개, 중간 10개, 마지막 10개
    n = len(flat)
    sample_indices = (
        list(range(min(10, n)))
        + list(range(n // 2, min(n // 2 + 10, n)))
        + list(range(max(0, n - 10), n))
    )
    sample_indices = sorted(set(sample_indices))
    return {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "mean": float(flat.mean()),
        "std": float(flat.std()),
        "min": float(flat.min()),
        "max": float(flat.max()),
        "sample_indices": sample_indices,
        "sample_values": [float(flat[i]) for i in sample_indices],
    }


def main():
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    golden = []
    for case in CASES:
        pil_img = _make_image(case)
        png_b64 = _image_to_png_b64(pil_img)
        tensor = preprocess_image(pil_img, case["meta"])
        summary = _tensor_summary(tensor)

        golden.append({
            "name": case["name"],
            "input_image_b64_png": png_b64,
            "input_image_size_wh": case["image_size_wh"],
            "input_image_mode": case["image_mode"],
            "meta": case["meta"],
            "output": summary,
        })
        print(f"  [{case['name']}] shape={summary['shape']} "
              f"mean={summary['mean']:.4f} std={summary['std']:.4f}")

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(golden, f, indent=2, ensure_ascii=False)

    print(f"\n골든 파일 저장: {OUTPUT_PATH}")
    print(f"총 {len(golden)}개 케이스")


if __name__ == "__main__":
    main()
