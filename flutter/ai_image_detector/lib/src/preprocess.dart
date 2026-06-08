import 'dart:typed_data';
import 'package:image/image.dart' as img;
import 'meta.dart';

/// Python onnx_detector.preprocess_image()의 Dart 포팅.
///
/// 파이프라인:
///   1. RGB 변환 (grayscale, RGBA 등)
///   2. resize_mode에 따른 리사이즈:
///      - "shortest_edge": 짧은 변을 image_size로 aspect 유지 resize → crop_size center-crop
///      - "exact" (기본): image_size × image_size 정사각 resize
///   3. rescale: × rescale_factor (기본 1/255)
///   4. normalize: (x - mean) / std
///   5. HWC → CHW, 배치 차원 추가 → shape (1, 3, H, W), float32
///
/// 보간 주의:
///   Python PIL은 resample=3이 BICUBIC.
///   Dart image 패키지는 [Interpolation.cubic]이 유사하지만 완전 동일하지 않을 수 있음.
///   절대오차 허용범위: 1e-2 이내.
Float32List preprocessImage(img.Image image, ModelMeta meta) {
  // 1. RGB 변환 (grayscale, RGBA 등 → 3채널 RGB)
  img.Image rgb = image.convert(numChannels: 3);

  // 2. Resize
  final img.Interpolation interp = _pilResampleToInterpolation(meta.resample);

  if (meta.resizeMode == 'shortest_edge') {
    final int w = rgb.width;
    final int h = rgb.height;
    final int size = meta.imageSize;
    final int newW;
    final int newH;
    if (w <= h) {
      newW = size;
      newH = (h * size / w).floor();
    } else {
      newH = size;
      newW = (w * size / h).floor();
    }
    rgb = img.copyResize(rgb, width: newW, height: newH, interpolation: interp);
    // center-crop
    rgb = _centerCrop(rgb, meta.cropSize, meta.cropSize);
  } else {
    // exact
    final int size = meta.imageSize;
    rgb = img.copyResize(rgb, width: size, height: size, interpolation: interp);
  }

  final int h = rgb.height;
  final int w = rgb.width;
  final int numPixels = h * w;

  // 3+4. rescale + normalize → CHW float32 배열
  // 채널 순서: 0=R, 1=G, 2=B
  final Float32List chw = Float32List(3 * numPixels);

  for (int y = 0; y < h; y++) {
    for (int x = 0; x < w; x++) {
      final pixel = rgb.getPixel(x, y);
      final double r = pixel.r.toDouble();
      final double g = pixel.g.toDouble();
      final double b = pixel.b.toDouble();

      double vr = r;
      double vg = g;
      double vb = b;

      if (meta.doRescale) {
        vr *= meta.rescaleFactor;
        vg *= meta.rescaleFactor;
        vb *= meta.rescaleFactor;
      }

      if (meta.doNormalize) {
        vr = (vr - meta.imageMean[0]) / meta.imageStd[0];
        vg = (vg - meta.imageMean[1]) / meta.imageStd[1];
        vb = (vb - meta.imageMean[2]) / meta.imageStd[2];
      }

      final int idx = y * w + x;
      chw[idx] = vr;
      chw[numPixels + idx] = vg;
      chw[2 * numPixels + idx] = vb;
    }
  }

  return chw;
}

/// PIL resample 코드 → Dart img.Interpolation 매핑.
/// 3 = BICUBIC → img.Interpolation.cubic
/// 2 = BILINEAR → img.Interpolation.linear
/// 0 = NEAREST → img.Interpolation.nearest
img.Interpolation _pilResampleToInterpolation(int pilCode) {
  switch (pilCode) {
    case 3:
      return img.Interpolation.cubic;
    case 2:
      return img.Interpolation.linear;
    default:
      return img.Interpolation.nearest;
  }
}

/// Python _center_crop()과 동일: 중앙에서 cropH × cropW 크기로 crop.
img.Image _centerCrop(img.Image src, int cropH, int cropW) {
  final int w = src.width;
  final int h = src.height;
  final int left = (w - cropW) ~/ 2;
  final int top = (h - cropH) ~/ 2;
  return img.copyCrop(src, x: left, y: top, width: cropW, height: cropH);
}

/// 전처리 결과를 ONNX 입력용 shape 리스트와 함께 반환.
/// shape: [1, 3, H, W]
(Float32List data, List<int> shape) preprocessImageWithShape(
    img.Image image, ModelMeta meta) {
  final Float32List data = preprocessImage(image, meta);
  final int size = meta.resizeMode == 'shortest_edge' ? meta.cropSize : meta.imageSize;
  return (data, [1, 3, size, size]);
}
