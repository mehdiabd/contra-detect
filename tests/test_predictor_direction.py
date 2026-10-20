from unittest import TestCase
import config
import numpy as np
import torch
from src.data.schema import Pair
from src.models.predictor import ContradictionPredictor
from scripts.analyze_errors import metrics


class DirectionTests(TestCase):
    def test_reverse_contradiction_does_not_replace_forward_entailment(self):
        predictor = ContradictionPredictor.__new__(ContradictionPredictor)
        predictor.id2label = config.NLI_ID2LABEL
        predictor.contradiction_index = 1
        scores = iter([torch.tensor([[.8, .1, .1]]), torch.tensor([[.1, .7, .2]])])
        predictor._scores = lambda a, b: next(scores)
        result = predictor.predict_pairs([Pair('مقدمه', 'فرضیه')])[0]
        self.assertEqual(result['label'], 'entailment')
        self.assertAlmostEqual(result['contradiction_score'], .1)
        self.assertAlmostEqual(result['probs']['contradiction'], result['contradiction_score'])
        self.assertAlmostEqual(result['bidirectional_contradiction_score'], .7)

    def test_metrics_count_missed_and_false_contradictions(self):
        gold = np.array([0, 1, 1, 2])
        pred = np.eye(3)[[0, 1, 2, 1]]
        result = metrics(gold, pred.argmax(axis=1))
        self.assertEqual(result['accuracy'], .5)
        self.assertEqual(result['per_class']['contradiction']['f1'], .5)

    def test_absent_class_has_zero_metrics(self):
        result = metrics(np.array([0]), np.array([0]))
        self.assertEqual(result['per_class']['contradiction']['f1'], 0.)
        self.assertEqual(result['per_class']['neutral']['f1'], 0.)
