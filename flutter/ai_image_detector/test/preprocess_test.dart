import 'dart:typed_data';
import 'package:flutter_test/flutter_test.dart';
import 'package:image/image.dart' as img;
import 'package:ai_image_detector/src/meta.dart';
import 'package:ai_image_detector/src/preprocess.dart';

ModelMeta _makeMeta({
  int imageSize = 224,
  int? cropSize,
  String resizeMode = 'exact',
  List<double> mean = const [0.485, 0.456, 0.406],
  List<double> std = const [0.229, 0.224, 0.225],
  bool doRescale = true,
  double rescaleFactor = 1.0 / 255.0,
  bool doNormalize = true,
}) {
  return ModelMeta(
    imageSize: imageSize,
    cropSize: cropSize ?? imageSize,
    resizeMode: resizeMode,
    imageMean: mean,
    imageStd: std,
    doRescale: doRescale,
    rescaleFactor: rescaleFactor,
    doNormalize: doNormalize,
    resample: 3,
    id2label: {'0': 'artificial', '1': 'human'},
  );
}

/// 단색 이미지 생성 (RGB)
img.Image _solidImage(int w, int h, int r, int g, int b) {
  final image = img.Image(width: w, height: h, numChannels: 3);
  for (int y = 0; y < h; y++) {
    for (int x = 0; x < w; x++) {
      image.setPixelRgb(x, y, r, g, b);
    }
  }
  return image;
}

void main() {
  group('출력 shape 및 dtype', () {
    test('exact 모드 → (1, 3, 224, 224) 크기 float32', () {
      final image = _solidImage(300, 200, 128, 64, 32);
      final meta = _makeMeta(imageSize: 224);
      final (data, shape) = preprocessImageWithShape(image, meta);
      expect(shape, [1, 3, 224, 224]);
      expect(data.length, 1 * 3 * 224 * 224);
      expect(data, isA<Float32List>());
    });

    test('shortest_edge + crop → (1, 3, cropSize, cropSize)', () {
      final image = _solidImage(400, 300, 100, 150, 200);
      final meta = _makeMeta(
          imageSize: 256, cropSize: 224, resizeMode: 'shortest_edge');
      final (data, shape) = preprocessImageWithShape(image, meta);
      expect(shape, [1, 3, 224, 224]);
      expect(data.length, 1 * 3 * 224 * 224);
    });
  });

  group('rescale + normalize 수식', () {
    test('알려진 픽셀값 → 기대 정규화값 (R채널)', () {
      // 픽셀 R=128: 128/255=0.50196..., (0.50196-0.485)/0.229 ≈ 0.0738
      final image = _solidImage(4, 4, 128, 128, 128);
      final meta = _makeMeta(imageSize: 4);
      final data = preprocessImage(image, meta);

      const double pixelR = 128.0;
      const double rescaled = pixelR / 255.0;
      const double normalized = (rescaled - 0.485) / 0.229;

      // R채널: data[0..15]
      for (int i = 0; i < 16; i++) {
        expect(data[i], closeTo(normalized, 1e-4));
      }
    });

    test('do_rescale=false → 픽셀값 그대로 normalize', () {
      final image = _solidImage(4, 4, 128, 128, 128);
      final meta = _makeMeta(imageSize: 4, doRescale: false);
      final data = preprocessImage(image, meta);

      // rescale 없이: (128.0 - 0.485) / 0.229
      const double normalized = (128.0 - 0.485) / 0.229;
      expect(data[0], closeTo(normalized, 1e-3));
    });

    test('do_normalize=false → rescale만 적용', () {
      final image = _solidImage(4, 4, 255, 0, 0);
      final meta = _makeMeta(imageSize: 4, doNormalize: false);
      final data = preprocessImage(image, meta);

      // R채널: 255/255 = 1.0
      expect(data[0], closeTo(1.0, 1e-4));
      // G채널: 0/255 = 0.0
      expect(data[4 * 4], closeTo(0.0, 1e-4));
    });

    test('do_rescale=false, do_normalize=false → 픽셀값 그대로', () {
      final image = _solidImage(4, 4, 200, 100, 50);
      final meta = _makeMeta(imageSize: 4, doRescale: false, doNormalize: false);
      final data = preprocessImage(image, meta);
      expect(data[0], closeTo(200.0, 1e-3));
      expect(data[4 * 4], closeTo(100.0, 1e-3));
      expect(data[2 * 4 * 4], closeTo(50.0, 1e-3));
    });
  });

  group('CHW 채널 순서', () {
    test('R, G, B 채널이 순서대로 배치됨', () {
      // R=255, G=128, B=64 단색 이미지
      final image = _solidImage(4, 4, 255, 128, 64);
      final meta = _makeMeta(imageSize: 4, doRescale: false, doNormalize: false);
      final data = preprocessImage(image, meta);

      // R채널: [0, 16), G채널: [16, 32), B채널: [32, 48)
      expect(data[0], closeTo(255.0, 1e-3));   // R
      expect(data[16], closeTo(128.0, 1e-3));  // G
      expect(data[32], closeTo(64.0, 1e-3));   // B
    });
  });

  group('exact vs shortest_edge resize', () {
    test('exact 모드: 정사각 resize', () {
      final image = _solidImage(400, 300, 0, 0, 0);
      final meta = _makeMeta(imageSize: 32);
      final (_, shape) = preprocessImageWithShape(image, meta);
      expect(shape, [1, 3, 32, 32]);
    });

    test('shortest_edge: landscape(400x300) → 짧은 변 256으로 resize 후 crop 224', () {
      final image = _solidImage(400, 300, 0, 0, 0);
      final meta = _makeMeta(
          imageSize: 256, cropSize: 224, resizeMode: 'shortest_edge');
      final (_, shape) = preprocessImageWithShape(image, meta);
      expect(shape, [1, 3, 224, 224]);
    });

    test('shortest_edge: portrait(300x400) → 짧은 변 256으로 resize 후 crop', () {
      final image = _solidImage(300, 400, 0, 0, 0);
      final meta = _makeMeta(
          imageSize: 256, cropSize: 224, resizeMode: 'shortest_edge');
      final (_, shape) = preprocessImageWithShape(image, meta);
      expect(shape, [1, 3, 224, 224]);
    });

    test('shortest_edge: 정사각(300x300) → 그대로 256 resize 후 crop', () {
      final image = _solidImage(300, 300, 0, 0, 0);
      final meta = _makeMeta(
          imageSize: 256, cropSize: 224, resizeMode: 'shortest_edge');
      final (_, shape) = preprocessImageWithShape(image, meta);
      expect(shape, [1, 3, 224, 224]);
    });
  });

  group('center-crop 위치', () {
    // center-crop이 정확히 중앙에서 잘리는지 검증.
    // 왼쪽 절반=빨강, 오른쪽 절반=파랑인 128x128 이미지를 shortest_edge로
    // 128→128 (no resize), crop 64×64하면 중앙이므로 각 절반이 섞임.
    test('landscape: 중앙 crop 검증 — 단색 블록으로 crop 위치 확인', () {
      // 64x32 이미지: 왼쪽 32px=255, 오른쪽 32px=0 (R채널)
      final image = img.Image(width: 64, height: 32, numChannels: 3);
      for (int y = 0; y < 32; y++) {
        for (int x = 0; x < 32; x++) {
          image.setPixelRgb(x, y, 255, 0, 0); // 왼쪽 빨강
        }
        for (int x = 32; x < 64; x++) {
          image.setPixelRgb(x, y, 0, 0, 255); // 오른쪽 파랑
        }
      }
      // exact 모드, 32x32로 resize 후 crop 16
      // shortest_edge 32 → 짧은 변(높이32) → newH=32, newW=64 → crop 32
      final meta = _makeMeta(
          imageSize: 32, cropSize: 32, resizeMode: 'shortest_edge',
          doRescale: false, doNormalize: false);
      final (data, shape) = preprocessImageWithShape(image, meta);
      expect(shape, [1, 3, 32, 32]);
      // center-crop(32x32) from (64x32): left=(64-32)//2=16
      // 중앙 32픽셀이라면 왼쪽 16px=빨강(255), 오른쪽 16px=파랑(0)이어야 함
      // R채널 첫 픽셀(0,0): (64-32)//2=16 → x=16 위치는 빨강이므로 ≈255
      expect(data[0], greaterThan(100.0)); // R채널 첫 픽셀 밝음
    });
  });

  group('grayscale / RGBA 처리', () {
    test('grayscale → RGB 변환 후 정상 전처리', () {
      final gray = img.Image(width: 4, height: 4, numChannels: 1);
      for (int y = 0; y < 4; y++) {
        for (int x = 0; x < 4; x++) {
          gray.setPixelR(x, y, 128);
        }
      }
      // mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]이면 grayscale 128 → 세 채널 동일값
      final meta = _makeMeta(imageSize: 4, mean: [0.5, 0.5, 0.5], std: [0.5, 0.5, 0.5]);
      final (data, shape) = preprocessImageWithShape(gray, meta);
      expect(shape, [1, 3, 4, 4]);
      // 세 채널 모두 동일한 normalize 결과: (128/255 - 0.5) / 0.5 ≈ 0.0039
      const double expected = (128.0 / 255.0 - 0.5) / 0.5;
      expect(data[0], closeTo(expected, 1e-3));
      expect(data[0], closeTo(data[16], 1e-3));
      expect(data[0], closeTo(data[32], 1e-3));
    });

    test('RGBA → RGB 변환 후 정상 전처리 (알파 채널 버림)', () {
      final rgba = img.Image(width: 4, height: 4, numChannels: 4);
      for (int y = 0; y < 4; y++) {
        for (int x = 0; x < 4; x++) {
          rgba.setPixelRgba(x, y, 200, 100, 50, 128);
        }
      }
      final meta = _makeMeta(imageSize: 4, doRescale: false, doNormalize: false);
      final (data, shape) = preprocessImageWithShape(rgba, meta);
      expect(shape, [1, 3, 4, 4]);
      expect(data[0], closeTo(200.0, 1e-2));   // R
      expect(data[16], closeTo(100.0, 1e-2));  // G
      expect(data[32], closeTo(50.0, 1e-2));   // B
    });
  });
}
