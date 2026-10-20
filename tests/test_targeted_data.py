from unittest import TestCase
from scripts.build_targeted_data import build
from src.data.loader import load_farstail_split


class TargetedDataTests(TestCase):
    def test_training_examples_do_not_reuse_heldout_texts(self):
        heldout = load_farstail_split('val') + load_farstail_split('test')
        texts = {t for p in heldout for t in (p.text_a, p.text_b)}
        rows = build()
        self.assertTrue(all(p.text_a not in texts and p.text_b not in texts for p, _ in rows))
        keys = [(p.text_a, p.text_b) for p, _ in rows]
        self.assertEqual(len(keys), len(set(keys)))
        self.assertEqual({p.label for p, _ in rows}, {0, 1, 2})
        # Every premise has all three contrasts, so topic alone cannot determine the class.
        grouped = {}
        for p, _ in rows:
            grouped.setdefault(p.text_a, set()).add(p.label)
        self.assertTrue(all(labels == {0, 1, 2} for labels in grouped.values()))
