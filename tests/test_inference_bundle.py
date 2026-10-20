import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import Mock
import torch
from src.models.predictor import load_inference_bundle, ContradictionPredictor


class BundleTests(TestCase):
    def make_bundle(self, root, weights):
        members = []
        for i, w in enumerate(weights):
            path = root / f'member{i}'
            path.mkdir(exist_ok=True)
            (path / 'config.json').write_text('{}')
            members.append({'path': f'member{i}', 'weight': w})
        (root / 'inference_config.json').write_text(json.dumps({'version': 1, 'members': members, 'max_length': 512}))

    def test_resolves_members_and_normalizes_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.make_bundle(root, [1, 3])
            sources, weights, length = load_inference_bundle(root)
            self.assertEqual(weights, [.25, .75])
            self.assertEqual(length, 512)
            self.assertEqual(sources[0], str((root / 'member0').resolve()))

    def test_invalid_weights_are_rejected(self):
        for weights in ([0, 0], [-1, 2], [float('nan'), 1]):
            with self.subTest(weights=weights), tempfile.TemporaryDirectory() as tmp:
                self.make_bundle(Path(tmp), weights)
                with self.assertRaises(ValueError):
                    load_inference_bundle(tmp)

    def test_weighted_probabilities_preserve_canonical_labels(self):
        predictor = ContradictionPredictor.__new__(ContradictionPredictor)
        predictor.device = torch.device('cpu')
        predictor.max_length = 512
        tokenize = Mock(return_value={'input_ids': torch.tensor([[1]])})
        a = Mock(return_value=SimpleNamespace(logits=torch.log(torch.tensor([[.8, .1, .1]]))))
        b = Mock(return_value=SimpleNamespace(logits=torch.log(torch.tensor([[.1, .1, .8]]))))
        predictor.members = [(tokenize, a), (tokenize, b)]
        predictor.member_label_orders = [[0, 1, 2], [0, 2, 1]]
        predictor.member_weights = [.25, .75]
        self.assertTrue(torch.allclose(predictor._scores(['a'], ['b']), torch.tensor([[.275, .625, .1]])))
