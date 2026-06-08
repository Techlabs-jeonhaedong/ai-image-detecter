import 'dart:convert';

/// meta.json에서 파싱된 전처리 설정.
/// Python onnx_detector._validate_meta()와 동일한 범위 검증.
class ModelMeta {
  final int imageSize;
  final int cropSize;
  final String resizeMode;
  final List<double> imageMean;
  final List<double> imageStd;
  final bool doRescale;
  final double rescaleFactor;
  final bool doNormalize;
  final int resample; // PIL resample 코드: 3=BICUBIC
  final Map<String, String> id2label;

  const ModelMeta({
    required this.imageSize,
    required this.cropSize,
    required this.resizeMode,
    required this.imageMean,
    required this.imageStd,
    required this.doRescale,
    required this.rescaleFactor,
    required this.doNormalize,
    required this.resample,
    required this.id2label,
  });

  /// JSON 문자열 또는 Map에서 파싱 + 검증.
  /// Python _validate_meta()와 동일한 제약 적용.
  factory ModelMeta.fromJson(dynamic source) {
    final Map<String, dynamic> m;
    if (source is String) {
      final decoded = jsonDecode(source);
      if (decoded is! Map<String, dynamic>) {
        throw ArgumentError('meta.json must be a JSON object');
      }
      m = decoded;
    } else if (source is Map<String, dynamic>) {
      m = source;
    } else {
      throw ArgumentError('source must be a JSON string or Map');
    }

    _validateMeta(m);

    // image_size: height/width 우선, 없으면 image_size
    final int imageSize;
    if (m.containsKey('image_size')) {
      imageSize = (m['image_size'] as num).toInt();
    } else {
      // height/width 모두 있는 경우 — 정사각 가정으로 height 사용
      imageSize = (m['height'] as num).toInt();
    }

    final int cropSize = m.containsKey('crop_size')
        ? (m['crop_size'] as num).toInt()
        : imageSize;

    final String resizeMode = (m['resize_mode'] as String?) ?? 'exact';

    final List<double> imageMean =
        (m['image_mean'] as List).map((e) => (e as num).toDouble()).toList();
    final List<double> imageStd =
        (m['image_std'] as List).map((e) => (e as num).toDouble()).toList();

    final bool doRescale = (m['do_rescale'] as bool?) ?? true;
    final double rescaleFactor =
        m.containsKey('rescale_factor') ? (m['rescale_factor'] as num).toDouble() : 1.0 / 255.0;
    final bool doNormalize = (m['do_normalize'] as bool?) ?? true;
    final int resample = (m['resample'] as num?)?.toInt() ?? 3;

    final rawId2label = m['id2label'] as Map<String, dynamic>;
    final Map<String, String> id2label =
        rawId2label.map((k, v) => MapEntry(k, v.toString()));

    return ModelMeta(
      imageSize: imageSize,
      cropSize: cropSize,
      resizeMode: resizeMode,
      imageMean: imageMean,
      imageStd: imageStd,
      doRescale: doRescale,
      rescaleFactor: rescaleFactor,
      doNormalize: doNormalize,
      resample: resample,
      id2label: id2label,
    );
  }
}

/// Python _validate_meta()와 동일한 검증 로직.
/// 위반 시 [ArgumentError] 발생.
void _validateMeta(Map<String, dynamic> m) {
  // image_size 또는 height+width 필수
  if (!m.containsKey('image_size') &&
      !(m.containsKey('height') && m.containsKey('width'))) {
    throw ArgumentError(
        "Invalid meta.json: 'image_size' or 'height'+'width' required");
  }

  for (final key in ['image_size', 'crop_size']) {
    if (m.containsKey(key)) {
      final val = m[key];
      if (val is! int) {
        throw ArgumentError(
            "Invalid meta.json: '$key' must be int, got ${val.runtimeType}");
      }
      if (val < 1 || val > 1024) {
        throw ArgumentError(
            "Invalid meta.json: '$key' out of range 1-1024, got $val");
      }
    }
  }

  for (final key in ['height', 'width']) {
    if (m.containsKey(key)) {
      final val = m[key];
      if (val is! int) {
        throw ArgumentError(
            "Invalid meta.json: '$key' must be int, got ${val.runtimeType}");
      }
      if (val < 1 || val > 1024) {
        throw ArgumentError(
            "Invalid meta.json: '$key' out of range 1-1024, got $val");
      }
    }
  }

  for (final key in ['image_mean', 'image_std']) {
    if (!m.containsKey(key)) {
      throw ArgumentError("Invalid meta.json: '$key' required");
    }
    final val = m[key];
    if (val is! List || val.length != 3) {
      throw ArgumentError(
          "Invalid meta.json: '$key' must be a list of 3 numbers");
    }
    for (final v in val) {
      if (v is! num) {
        throw ArgumentError(
            "Invalid meta.json: '$key' elements must be numeric");
      }
    }
  }

  if (!m.containsKey('id2label')) {
    throw ArgumentError("Invalid meta.json: 'id2label' required");
  }
  if (m['id2label'] is! Map) {
    throw ArgumentError("Invalid meta.json: 'id2label' must be a dict");
  }
}
