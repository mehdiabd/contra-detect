from unittest import TestCase
from config import BEST_MODEL_DIR
from src.models.trainer import ContradictionTrainer, compute_classification_metrics
import numpy as np


class TrainingSafetyTests(TestCase):
    def test_production_output_rejected_before_training(self):
        trainer = ContradictionTrainer.__new__(ContradictionTrainer)
        with self.assertRaisesRegex(ValueError, 'production'):
            trainer.train([], output_dir=BEST_MODEL_DIR)
        with self.assertRaisesRegex(ValueError, "production"):
            trainer.train([], output_dir=BEST_MODEL_DIR.parent / "targeted_v1")

    def test_metrics_include_missing_class_and_contradiction_errors(self):
        # Gold: one of each class. A missed contradiction and a false alarm.
        logits = np.eye(3)[[0, 2, 1]]
        metrics = compute_classification_metrics((logits, np.array([0, 1, 2])))
        self.assertAlmostEqual(metrics['accuracy'], 1 / 3)
        self.assertAlmostEqual(metrics['f1'], 1 / 3)
        self.assertEqual(metrics['f1_contradiction'], 0)
