import 'dart:math' as math;
import 'package:flutter_test/flutter_test.dart';
import 'package:ai_image_detector/src/postprocess.dart';

void main() {
  group('softmax', () {
    test('합이 1에 수렴', () {
      final probs = softmax([1.0, 2.0, 3.0]);
      expect(probs.reduce((a, b) => a + b), closeTo(1.0, 1e-6));
    });

    test('큰 logit이 가장 높은 확률', () {
      final probs = softmax([0.0, 10.0, 0.0]);
      expect(probs[1], greaterThan(probs[0]));
      expect(probs[1], greaterThan(probs[2]));
    });

    test('모든 logit이 같으면 균등 분포', () {
      final probs = softmax([1.0, 1.0, 1.0]);
      for (final p in probs) {
        expect(p, closeTo(1.0 / 3.0, 1e-6));
      }
    });

    test('수치 안정성 — 매우 큰 logit', () {
      final probs = softmax([1000.0, 999.0]);
      expect(probs[0], greaterThan(0.5));
      expect(probs[0] + probs[1], closeTo(1.0, 1e-6));
    });

    test('수치 안정성 — 매우 작은 logit', () {
      final probs = softmax([-1000.0, -999.0]);
      expect(probs[1], greaterThan(0.5));
      expect(probs[0] + probs[1], closeTo(1.0, 1e-6));
    });

    test('단일 원소', () {
      final probs = softmax([5.0]);
      expect(probs, [closeTo(1.0, 1e-9)]);
    });

    test('빈 입력 → 빈 출력', () {
      final probs = softmax([]);
      expect(probs, isEmpty);
    });

    test('알려진 값 수치 검증: [0, 1] → e^0/(e^0+e^1), e^1/(e^0+e^1)', () {
      final probs = softmax([0.0, 1.0]);
      final e0 = math.exp(0.0);
      final e1 = math.exp(1.0);
      expect(probs[0], closeTo(e0 / (e0 + e1), 1e-6));
      expect(probs[1], closeTo(e1 / (e0 + e1), 1e-6));
    });
  });

  group('buildLabelScores', () {
    test('id2label 매핑 정상', () {
      final scores = buildLabelScores(
          {'0': 'artificial', '1': 'human'}, [0.8, 0.2]);
      expect(scores.length, 2);
      expect(scores[0].label, 'artificial');
      expect(scores[0].score, closeTo(0.8, 1e-9));
      expect(scores[1].label, 'human');
      expect(scores[1].score, closeTo(0.2, 1e-9));
    });

    test('id2label에 없는 인덱스는 label_N 형식 사용', () {
      final scores = buildLabelScores({'0': 'ai'}, [0.6, 0.4]);
      expect(scores[1].label, 'label_1');
    });

    test('빈 probs → 빈 결과', () {
      final scores = buildLabelScores({}, []);
      expect(scores, isEmpty);
    });
  });

  group('extractAiProbability', () {
    LabelScore ls(String label, double score) =>
        LabelScore(label: label, score: score);

    test('AI 라벨 키워드 직접 추출 — artificial', () {
      final result = extractAiProbability([ls('artificial', 0.9), ls('human', 0.1)]);
      expect(result, closeTo(0.9, 1e-9));
    });

    test('AI 라벨 — fake', () {
      final result = extractAiProbability([ls('fake', 0.7), ls('real', 0.3)]);
      expect(result, closeTo(0.7, 1e-9));
    });

    test('AI 라벨 — generated', () {
      final result = extractAiProbability([ls('generated', 0.55)]);
      expect(result, closeTo(0.55, 1e-9));
    });

    test('AI 라벨 — synthetic', () {
      final result = extractAiProbability([ls('synthetic', 0.4), ls('photo', 0.6)]);
      expect(result, closeTo(0.4, 1e-9));
    });

    test('AI 라벨 — ai (소문자)', () {
      final result = extractAiProbability([ls('ai', 0.8)]);
      expect(result, closeTo(0.8, 1e-9));
    });

    test('AI 라벨 없을 때 Real 보완값 — human', () {
      final result = extractAiProbability([ls('human', 0.7)]);
      expect(result, closeTo(0.3, 1e-9));
    });

    test('AI 라벨 없을 때 Real 보완값 — real', () {
      final result = extractAiProbability([ls('real', 0.6)]);
      expect(result, closeTo(0.4, 1e-9));
    });

    test('AI 라벨 없을 때 Real 보완값 — photo', () {
      final result = extractAiProbability([ls('photo', 0.9)]);
      expect(result, closeTo(0.1, 1e-9));
    });

    test('AI 라벨 없을 때 Real 보완값 — authentic', () {
      final result = extractAiProbability([ls('authentic', 0.8)]);
      expect(result, closeTo(0.2, 1e-9));
    });

    test('AI 라벨 없을 때 Real 보완값 — natural', () {
      final result = extractAiProbability([ls('natural', 0.75)]);
      expect(result, closeTo(0.25, 1e-9));
    });

    test('대소문자 무시 — ARTIFICIAL', () {
      final result = extractAiProbability([ls('ARTIFICIAL', 0.85)]);
      expect(result, closeTo(0.85, 1e-9));
    });

    test('대소문자 무시 — Human', () {
      final result = extractAiProbability([ls('Human', 0.65)]);
      expect(result, closeTo(0.35, 1e-9));
    });

    test('공백 포함 라벨 trim 처리', () {
      final result = extractAiProbability([ls('  artificial  ', 0.9)]);
      expect(result, closeTo(0.9, 1e-9));
    });

    test('AI 라벨이 먼저 나오면 AI 라벨 우선', () {
      // AI + Real 둘 다 있을 때 AI 라벨 직접 추출
      final result = extractAiProbability([ls('artificial', 0.7), ls('human', 0.3)]);
      expect(result, closeTo(0.7, 1e-9));
    });

    test('알 수 없는 라벨만 있으면 ArgumentError', () {
      expect(
        () => extractAiProbability([ls('unknown', 0.5)]),
        throwsA(isA<ArgumentError>()),
      );
    });

    test('빈 리스트 → ArgumentError', () {
      expect(() => extractAiProbability([]), throwsA(isA<ArgumentError>()));
    });
  });

  group('determineVerdict', () {
    test('0.8 >= 0.5 → AI-generated', () {
      expect(determineVerdict(0.8, 0.5), 'AI-generated');
    });

    test('0.3 < 0.5 → Real', () {
      expect(determineVerdict(0.3, 0.5), 'Real');
    });

    test('정확히 threshold 경계 — 같으면 AI-generated', () {
      expect(determineVerdict(0.5, 0.5), 'AI-generated');
    });

    test('threshold=0.0 → 항상 AI-generated', () {
      expect(determineVerdict(0.0, 0.0), 'AI-generated');
    });

    test('threshold=1.0, aiProb=1.0 → AI-generated', () {
      expect(determineVerdict(1.0, 1.0), 'AI-generated');
    });

    test('threshold=1.0, aiProb=0.99 → Real', () {
      expect(determineVerdict(0.99, 1.0), 'Real');
    });

    test('threshold 범위 초과 → ArgumentError', () {
      expect(() => determineVerdict(0.5, 1.1), throwsA(isA<ArgumentError>()));
      expect(() => determineVerdict(0.5, -0.1), throwsA(isA<ArgumentError>()));
    });

    test('aiProbability 범위 초과 → ArgumentError', () {
      expect(() => determineVerdict(1.1, 0.5), throwsA(isA<ArgumentError>()));
      expect(() => determineVerdict(-0.1, 0.5), throwsA(isA<ArgumentError>()));
    });
  });
}
