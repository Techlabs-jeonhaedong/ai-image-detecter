import 'package:flutter_test/flutter_test.dart';
import 'package:ai_image_detector/src/meta.dart';

void main() {
  group('ModelMeta.fromJson', () {
    // 기본 유효한 meta
    const validMeta = {
      'image_size': 224,
      'image_mean': [0.485, 0.456, 0.406],
      'image_std': [0.229, 0.224, 0.225],
      'id2label': {'0': 'artificial', '1': 'human'},
    };

    test('정상 meta 파싱 성공', () {
      final meta = ModelMeta.fromJson(validMeta);
      expect(meta.imageSize, 224);
      expect(meta.cropSize, 224); // crop_size 없으면 image_size
      expect(meta.resizeMode, 'exact');
      expect(meta.imageMean, [0.485, 0.456, 0.406]);
      expect(meta.imageStd, [0.229, 0.224, 0.225]);
      expect(meta.doRescale, true);
      expect(meta.rescaleFactor, closeTo(1.0 / 255.0, 1e-9));
      expect(meta.doNormalize, true);
      expect(meta.resample, 3);
      expect(meta.id2label['0'], 'artificial');
      expect(meta.id2label['1'], 'human');
    });

    test('shortest_edge + crop_size 파싱', () {
      final m = {
        ...validMeta,
        'resize_mode': 'shortest_edge',
        'crop_size': 200,
      };
      final meta = ModelMeta.fromJson(m);
      expect(meta.resizeMode, 'shortest_edge');
      expect(meta.cropSize, 200);
    });

    test('height+width 키로도 파싱 가능', () {
      final m = {
        'height': 256,
        'width': 256,
        'image_mean': [0.5, 0.5, 0.5],
        'image_std': [0.5, 0.5, 0.5],
        'id2label': {'0': 'ai'},
      };
      final meta = ModelMeta.fromJson(m);
      expect(meta.imageSize, 256);
    });

    test('do_rescale=false, do_normalize=false 명시', () {
      final m = {
        ...validMeta,
        'do_rescale': false,
        'do_normalize': false,
        'rescale_factor': 0.00392,
      };
      final meta = ModelMeta.fromJson(m);
      expect(meta.doRescale, false);
      expect(meta.doNormalize, false);
      expect(meta.rescaleFactor, closeTo(0.00392, 1e-9));
    });

    // --- 필수 키 누락 오류 ---
    test('image_size 없고 height+width도 없으면 ArgumentError', () {
      final m = {
        'image_mean': [0.5, 0.5, 0.5],
        'image_std': [0.5, 0.5, 0.5],
        'id2label': <String, dynamic>{},
      };
      expect(() => ModelMeta.fromJson(m), throwsA(isA<ArgumentError>()));
    });

    test('image_mean 누락 시 ArgumentError', () {
      final m = {
        'image_size': 224,
        'image_std': [0.229, 0.224, 0.225],
        'id2label': <String, dynamic>{},
      };
      expect(() => ModelMeta.fromJson(m), throwsA(isA<ArgumentError>()));
    });

    test('image_std 누락 시 ArgumentError', () {
      final m = {
        'image_size': 224,
        'image_mean': [0.485, 0.456, 0.406],
        'id2label': <String, dynamic>{},
      };
      expect(() => ModelMeta.fromJson(m), throwsA(isA<ArgumentError>()));
    });

    test('id2label 누락 시 ArgumentError', () {
      final m = {
        'image_size': 224,
        'image_mean': [0.485, 0.456, 0.406],
        'image_std': [0.229, 0.224, 0.225],
      };
      expect(() => ModelMeta.fromJson(m), throwsA(isA<ArgumentError>()));
    });

    // --- 타입 오류 ---
    test('image_size가 float이면 ArgumentError', () {
      final m = {
        'image_size': 224.5,
        'image_mean': [0.5, 0.5, 0.5],
        'image_std': [0.5, 0.5, 0.5],
        'id2label': <String, dynamic>{},
      };
      expect(() => ModelMeta.fromJson(m), throwsA(isA<ArgumentError>()));
    });

    test('image_mean 원소가 문자열이면 ArgumentError', () {
      final m = {
        'image_size': 224,
        'image_mean': ['a', 'b', 'c'],
        'image_std': [0.5, 0.5, 0.5],
        'id2label': <String, dynamic>{},
      };
      expect(() => ModelMeta.fromJson(m), throwsA(isA<ArgumentError>()));
    });

    test('image_mean 길이 2이면 ArgumentError', () {
      final m = {
        'image_size': 224,
        'image_mean': [0.5, 0.5],
        'image_std': [0.5, 0.5, 0.5],
        'id2label': <String, dynamic>{},
      };
      expect(() => ModelMeta.fromJson(m), throwsA(isA<ArgumentError>()));
    });

    test('id2label이 list이면 ArgumentError', () {
      final m = {
        'image_size': 224,
        'image_mean': [0.5, 0.5, 0.5],
        'image_std': [0.5, 0.5, 0.5],
        'id2label': ['a', 'b'],
      };
      expect(() => ModelMeta.fromJson(m), throwsA(isA<ArgumentError>()));
    });

    // --- 범위 초과 오류 ---
    test('image_size=0 이면 ArgumentError', () {
      final m = {
        'image_size': 0,
        'image_mean': [0.5, 0.5, 0.5],
        'image_std': [0.5, 0.5, 0.5],
        'id2label': <String, dynamic>{},
      };
      expect(() => ModelMeta.fromJson(m), throwsA(isA<ArgumentError>()));
    });

    test('image_size=1025 이면 ArgumentError (모바일 OOM 방어 상한 1024)', () {
      final m = {
        'image_size': 1025,
        'image_mean': [0.5, 0.5, 0.5],
        'image_std': [0.5, 0.5, 0.5],
        'id2label': <String, dynamic>{},
      };
      expect(() => ModelMeta.fromJson(m), throwsA(isA<ArgumentError>()));
    });

    test('image_size=2048 이면 ArgumentError', () {
      final m = {
        'image_size': 2048,
        'image_mean': [0.5, 0.5, 0.5],
        'image_std': [0.5, 0.5, 0.5],
        'id2label': <String, dynamic>{},
      };
      expect(() => ModelMeta.fromJson(m), throwsA(isA<ArgumentError>()));
    });

    test('image_size=1024 이면 통과 (상한 이내)', () {
      final m = {
        'image_size': 1024,
        'image_mean': [0.5, 0.5, 0.5],
        'image_std': [0.5, 0.5, 0.5],
        'id2label': {'0': 'ai'},
      };
      expect(() => ModelMeta.fromJson(m), returnsNormally);
    });

    test('crop_size 범위 초과 시 ArgumentError', () {
      final m = {
        'image_size': 224,
        'crop_size': 2000,
        'image_mean': [0.5, 0.5, 0.5],
        'image_std': [0.5, 0.5, 0.5],
        'id2label': <String, dynamic>{},
      };
      expect(() => ModelMeta.fromJson(m), throwsA(isA<ArgumentError>()));
    });

    // --- JSON 문자열 입력 ---
    test('JSON 문자열로 파싱 가능', () {
      const jsonStr = '{"image_size":224,"image_mean":[0.485,0.456,0.406],'
          '"image_std":[0.229,0.224,0.225],"id2label":{"0":"ai","1":"real"}}';
      final meta = ModelMeta.fromJson(jsonStr);
      expect(meta.imageSize, 224);
      expect(meta.id2label['0'], 'ai');
    });

    test('JSON 배열 문자열이면 ArgumentError', () {
      expect(
          () => ModelMeta.fromJson('[1, 2, 3]'), throwsA(isA<ArgumentError>()));
    });
  });
}
