from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import Mock
import torch
from src.models.predictor import canonical_label_order, ContradictionPredictor


class LabelAlignmentTests(TestCase):
    def test_entailment_neutral_contradiction_order(self):
        cfg = SimpleNamespace(id2label={0: 'entailment', 1: 'neutral', 2: 'contradiction'})
        self.assertEqual(canonical_label_order(cfg), [0, 2, 1])

    def test_legacy_generic_labels_keep_project_order(self):
        cfg = SimpleNamespace(id2label={'0': 'LABEL_0', '1': 'LABEL_1', '2': 'LABEL_2'})
        self.assertEqual(canonical_label_order(cfg), [0, 1, 2])

    def test_unknown_labels_fail_explicitly(self):
        with self.assertRaises(ValueError):
            canonical_label_order(SimpleNamespace(id2label={0: 'yes', 1: 'no', 2: 'maybe'}))

    def test_ensemble_aligns_columns_before_averaging(self):
        predictor = ContradictionPredictor.__new__(ContradictionPredictor)
        predictor.device = torch.device('cpu')
        predictor.max_length = 256
        tokenize = Mock(return_value={'input_ids': torch.tensor([[1]])})
        a = Mock(return_value=SimpleNamespace(logits=torch.tensor([[0., 2., 1.]])))
        b = Mock(return_value=SimpleNamespace(logits=torch.tensor([[0., 1., 2.]])))
        predictor.members = [(tokenize, a), (tokenize, b)]
        predictor.member_label_orders = [[0, 1, 2], [0, 2, 1]]
        predictor.member_weights = [.5, .5]
        actual = predictor._scores(['a'], ['b'])
        expected = torch.softmax(torch.tensor([[0., 2., 1.]]), dim=1)
        self.assertTrue(torch.allclose(actual, expected))
