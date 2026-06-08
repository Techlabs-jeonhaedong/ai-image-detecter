import 'dart:math' as math;
import 'meta.dart';

/// 라벨과 점수 쌍.
class LabelScore {
  final String label;
  final double score;
  const LabelScore({required this.label, required this.score});

  @override
  String toString() => 'LabelScore(label: $label, score: $score)';
}

/// Python detector._AI_LABEL_KEYWORDS (소문자)
const Set<String> aiLabelKeywords = {
  'ai', 'artificial', 'fake', 'generated', 'synthetic'
};

/// Python detector._REAL_LABEL_KEYWORDS (소문자)
const Set<String> realLabelKeywords = {
  'real', 'human', 'natural', 'photo', 'authentic'
};

/// Python onnx_detector._softmax()와 동일: 수치 안정적 softmax.
List<double> softmax(List<double> logits) {
  if (logits.isEmpty) return [];
  final double maxVal = logits.reduce(math.max);
  final List<double> exps = logits.map((x) => math.exp(x - maxVal)).toList();
  final double sum = exps.reduce((a, b) => a + b);
  return exps.map((e) => e / sum).toList();
}

/// Python onnx_detector._build_label_scores()와 동일.
/// id2label 매핑 + 확률 배열 → [LabelScore] 리스트 (원래 인덱스 순서 유지).
List<LabelScore> buildLabelScores(
    Map<String, String> id2label, List<double> probs) {
  final List<LabelScore> result = [];
  for (int i = 0; i < probs.length; i++) {
    final String label = id2label[i.toString()] ?? 'label_$i';
    result.add(LabelScore(label: label, score: probs[i]));
  }
  return result;
}

/// Python detector.extract_ai_probability()와 동일 로직.
/// AI 라벨 직접 탐색 → 없으면 Real 보완값 (1 - realScore).
/// 모두 없으면 [ArgumentError] 발생.
double extractAiProbability(List<LabelScore> labelScores) {
  if (labelScores.isEmpty) {
    throw ArgumentError('Model returned empty results list');
  }

  // AI 라벨 직접 탐색
  for (final ls in labelScores) {
    if (aiLabelKeywords.contains(ls.label.toLowerCase().trim())) {
      return ls.score;
    }
  }

  // Real 라벨의 보완값
  for (final ls in labelScores) {
    if (realLabelKeywords.contains(ls.label.toLowerCase().trim())) {
      return 1.0 - ls.score;
    }
  }

  final known = labelScores.map((ls) => ls.label).toList();
  throw ArgumentError(
      'Cannot determine AI probability from label(s): $known. '
      'Expected AI keywords $aiLabelKeywords or Real keywords $realLabelKeywords.');
}

/// Python detector.determine_verdict()와 동일.
/// aiProbability >= threshold → "AI-generated", 아니면 "Real".
String determineVerdict(double aiProbability, double threshold) {
  if (threshold < 0.0 || threshold > 1.0) {
    throw ArgumentError('threshold must be in [0, 1], got $threshold');
  }
  if (aiProbability < 0.0 || aiProbability > 1.0) {
    throw ArgumentError('aiProbability must be in [0, 1], got $aiProbability');
  }
  return aiProbability >= threshold ? 'AI-generated' : 'Real';
}
