"""Validation split and early-stopping regimes of the fine-tuning loop."""
import types
import unittest

import main


def _dataset(n, n_classes=2):
    targets = [i % n_classes for i in range(n)]
    return types.SimpleNamespace(train_text=[f"t{i}" for i in range(n)], train_targets=targets,
                                 n_classes=n_classes, train=True)


class TestTrainValSplit(unittest.TestCase):

    def test_small_n_trains_on_everything_without_early_stopping(self):
        train, val, early = main.make_train_val_loaders(_dataset(100), 4, shuffle_seed=1)
        self.assertFalse(early)
        self.assertEqual(len(train.dataset.train_text), 100)
        self.assertEqual(len(val.dataset.train_text), 0)

    def test_holdout_scores_small_n_without_early_stopping(self):
        train, val, early = main.make_train_val_loaders(_dataset(100), 4, shuffle_seed=1, holdout_val=True)
        self.assertFalse(early)
        self.assertEqual(len(val.dataset.train_text), 10)
        self.assertEqual(len(train.dataset.train_text), 90)

    def test_holdout_needs_one_sample_per_class(self):
        _, val, early = main.make_train_val_loaders(_dataset(56, n_classes=7), 4, shuffle_seed=1,
                                                    holdout_val=True)
        self.assertFalse(early)
        self.assertEqual(len(val.dataset.train_text), 0)

    def test_large_n_keeps_early_stopping(self):
        train, val, early = main.make_train_val_loaders(_dataset(250), 8, shuffle_seed=1)
        self.assertTrue(early)
        self.assertEqual(len(val.dataset.train_text), main.FT_MIN_VAL_SAMPLES)
        self.assertEqual(len(train.dataset.train_text), 225)


if __name__ == "__main__":
    unittest.main()
