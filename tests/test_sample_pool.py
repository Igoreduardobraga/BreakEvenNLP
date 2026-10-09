# tests/test_sample_pool.py
"""
Unit tests for the deep SamplePool module.
Tests dataset partitioning, stratified budget selection, and reproducibility
without requiring external network downloads or GPU dependencies.
"""

import unittest
from sample_pool import SamplePool


class TestSamplePoolPartitioning(unittest.TestCase):
    """Tests train/test partitioning and stratified label selection."""

    def setUp(self):
        # 10 samples: 5 negative (0), 5 positive (1)
        self.texts = tuple([f"Sentence {i}" for i in range(10)])
        self.targets = tuple([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        self.classes = ("negative", "positive")
        # 8 train indices (4 of class 0, 4 of class 1), 2 test indices
        self.train_test_indices = ([0, 1, 2, 3, 4, 5, 6, 7], [8, 9])

    def test_partitioning_from_indices(self):
        pool = SamplePool(
            dataset_name="dummy",
            train_test_indices=self.train_test_indices,
            num_labelled=0,
            full_test=True,
            texts=self.texts,
            targets=self.targets,
            classes=self.classes,
        )
        self.assertEqual(len(pool.train_text), 8)
        self.assertEqual(len(pool.test_text), 2)
        self.assertEqual(pool.classes, ["negative", "positive"])
        self.assertEqual(pool.num_classes, 2)

    def test_stratified_budget_selection(self):
        # Request 4 labelled train samples (2 per class)
        pool = SamplePool(
            dataset_name="dummy",
            train_test_indices=self.train_test_indices,
            num_labelled=4,
            label_seed=42,
            full_test=True,
            texts=self.texts,
            targets=self.targets,
            classes=self.classes,
        )
        self.assertEqual(len(pool.train_text), 4)
        self.assertEqual(pool.train_targets.count(0), 2)
        self.assertEqual(pool.train_targets.count(1), 2)

    def test_label_seed_determinism(self):
        pool1 = SamplePool(
            dataset_name="dummy",
            train_test_indices=self.train_test_indices,
            num_labelled=4,
            label_seed=123,
            full_test=True,
            texts=self.texts,
            targets=self.targets,
            classes=self.classes,
        )
        pool2 = SamplePool(
            dataset_name="dummy",
            train_test_indices=self.train_test_indices,
            num_labelled=4,
            label_seed=123,
            full_test=True,
            texts=self.texts,
            targets=self.targets,
            classes=self.classes,
        )
        self.assertEqual(pool1.train_text, pool2.train_text)
        self.assertEqual(pool1.train_targets, pool2.train_targets)

    def test_full_test_subsampling(self):
        # When full_test is False, test set should be subsampled according to num_labelled_test
        pool = SamplePool(
            dataset_name="dummy",
            train_test_indices=([0, 1, 2, 3], [4, 5, 6, 7, 8, 9]),
            num_labelled=0,
            num_labelled_test=2,
            full_test=False,
            label_seed=0,
            texts=self.texts,
            targets=self.targets,
            classes=self.classes,
        )
        self.assertEqual(len(pool.test_text), 2)


class TestSamplePoolDemonstrationsAndBatching(unittest.TestCase):
    """Tests few-shot demonstration sampling and batch evaluation."""

    def setUp(self):
        # 20 samples: 10 of class 0, 10 of class 1
        self.texts = tuple([f"Text {i}" for i in range(20)])
        self.targets = tuple([0, 1] * 10)
        self.classes = ("negative", "positive")
        self.train_test_indices = (list(range(16)), [16, 17, 18, 19])
        self.pool = SamplePool(
            dataset_name="sst2",
            train_test_indices=self.train_test_indices,
            num_labelled=16,
            full_test=True,
            texts=self.texts,
            targets=self.targets,
            classes=self.classes,
        )

    def test_get_shots_balanced_representation(self):
        shots = self.pool.get_shots(num_shots=2, choice_seed=0, order_seed=0)
        self.assertEqual(len(shots), 4)
        labels = [s[1] for s in shots]
        self.assertEqual(labels.count("negative"), 2)
        self.assertEqual(labels.count("positive"), 2)

    def test_choice_seed_affects_selection(self):
        shots1 = self.pool.get_shots(num_shots=2, choice_seed=0, order_seed=0)
        shots2 = self.pool.get_shots(num_shots=2, choice_seed=999, order_seed=0)
        # Verify deterministic selection is stable with same seed
        shots1_repeat = self.pool.get_shots(num_shots=2, choice_seed=0, order_seed=0)
        self.assertEqual(shots1, shots1_repeat)
        # Different seeds select different items
        self.assertNotEqual(set(s[0] for s in shots1), set(s[0] for s in shots2))

    def test_order_seed_affects_permutation(self):
        shots1 = self.pool.get_shots(num_shots=3, choice_seed=0, order_seed=1)
        shots2 = self.pool.get_shots(num_shots=3, choice_seed=0, order_seed=2)
        # Same elements chosen
        self.assertEqual(set(s[0] for s in shots1), set(s[0] for s in shots2))
        # But different sequence order
        self.assertNotEqual(shots1, shots2)

    def test_batch_data_for_evaluation(self):
        batches = list(self.pool.batch_data_for_evaluation(batch_size=2, split="test"))
        self.assertEqual(len(batches), 2)
        # Batch 1
        data1, labels1 = batches[0]
        self.assertEqual(len(data1), 2)
        self.assertEqual(len(labels1), 2)
        # Batch 2
        data2, labels2 = batches[1]
        self.assertEqual(len(data2), 2)
        self.assertEqual(len(labels2), 2)

    def test_model_evaluator_integration(self):
        from evaluator import ModelEvaluator

        class DummyModel:
            def generate(self, **kwargs):
                return ["positive", "negative", "positive", "negative"]

        class DummyTokenizer:
            def __call__(self, texts, **kwargs):
                return texts

            def batch_decode(self, outputs, **kwargs):
                return outputs

        self.pool.context_samples = self.pool.get_shots(num_shots=1)

        evaluator = ModelEvaluator(
            model=DummyModel(),
            tokenizer=DummyTokenizer(),
            model_name="flan-t5",
            batch_size=4,
            prompt_format=0,
            device="cpu",
        )

        golden, predicted, decodeds = evaluator.evaluate(self.pool, mode="icl")
        self.assertEqual(len(golden), 4)
        self.assertEqual(len(predicted), 4)
        self.assertEqual(len(decodeds), 4)


class TestSamplePoolTorchAdapter(unittest.TestCase):
    """Tests PyTorch Dataset adapter creation without state mutation."""

    def setUp(self):
        self.texts = tuple([f"Text {i}" for i in range(10)])
        self.targets = tuple([0, 1] * 5)
        self.classes = ("negative", "positive")
        self.pool = SamplePool(
            dataset_name="sst2",
            train_test_indices=([0, 1, 2, 3, 4, 5, 6, 7], [8, 9]),
            num_labelled=8,
            full_test=True,
            texts=self.texts,
            targets=self.targets,
            classes=self.classes,
        )

    def test_to_torch_dataset_train_and_test(self):
        class DummyTokenizer:
            def encode_plus(self, text, **kwargs):
                return {
                    "input_ids": [10, 20, 30],
                    "attention_mask": [1, 1, 1],
                    "token_type_ids": [0, 0, 0],
                }

        tokenizer = DummyTokenizer()
        train_ds = self.pool.to_torch_dataset(split="train", tokenizer=tokenizer, max_len=10)
        test_ds = self.pool.to_torch_dataset(split="test", tokenizer=tokenizer, max_len=10)

        self.assertEqual(len(train_ds), 8)
        self.assertEqual(len(test_ds), 2)

        item = train_ds[0]
        self.assertIn("ids", item)
        self.assertIn("mask", item)
        self.assertIn("token_type_ids", item)
        self.assertIn("targets", item)

        # Immutability check: reading test_ds does not change train_ds or pool
        self.assertEqual(len(train_ds), 8)
        self.assertEqual(len(self.pool.train_text), 8)


class TestDatasetWrappersRetrofit(unittest.TestCase):
    """Tests backward compatibility wrappers in data.py delegating to SamplePool."""

    def setUp(self):
        import sys
        from unittest.mock import MagicMock

        class BaseDataset:
            pass

        mocked_modules = [
            'torch', 'torch.utils', 'torch.utils.data', 'datasets',
            'transformers', 'pandas', 'numpy', 'sklearn',
            'sklearn.metrics', 'sklearn.metrics.pairwise', 'sklearn.model_selection'
        ]
        self.originals = {}
        for mod in mocked_modules:
            if mod not in sys.modules:
                self.originals[mod] = None
                mock = MagicMock()
                if mod == 'torch.utils.data':
                    mock.Dataset = BaseDataset
                sys.modules[mod] = mock

        if hasattr(sys.modules.get('torch'), 'utils'):
            sys.modules['torch'].utils.data.Dataset = BaseDataset

    def tearDown(self):
        import sys
        for mod, orig in self.originals.items():
            if orig is None:
                sys.modules.pop(mod, None)

    def test_icl_and_prompt_dataset_delegation(self):
        from data import ICLDataset, PromptDataset

        train_test = ([0, 1], [2, 3])
        canned_t = ("Text 0", "Text 1", "Text 2", "Text 3")
        canned_y = (0, 1, 0, 1)
        canned_c = ("neg", "pos")

        from unittest.mock import patch
        with patch('data.load_text_and_targets', return_value=(canned_t, canned_y, canned_c)):
            icl_ds = ICLDataset("sst2", train_test_indices=train_test, num_shots=1)
            self.assertEqual(len(icl_ds.context_samples), 2)
            self.assertTrue(hasattr(icl_ds, 'pool'))

            batches = list(icl_ds.batch_data_for_evaluation(batch=2))
            self.assertEqual(len(batches), 1)

            prompt_ds = PromptDataset("sst2", train_test_indices=train_test)
            self.assertEqual(len(prompt_ds.context_samples), 0)
            batches_p = list(prompt_ds.batch_data_for_evaluation(batch=2))
            self.assertEqual(len(batches_p), 1)

    def test_text_dataset_backward_compat(self):
        from data import TextDataset
        train_test = ([0, 1], [2, 3])
        canned_t = ("Text 0", "Text 1", "Text 2", "Text 3")
        canned_y = (0, 1, 0, 1)
        canned_c = ("neg", "pos")

        from unittest.mock import patch
        with patch('data.load_text_and_targets', return_value=(canned_t, canned_y, canned_c)):
            ds = TextDataset("sst2", train_test_indices=train_test)
            self.assertEqual(len(ds.train_text), 2)
            self.assertEqual(len(ds.test_text), 2)
            self.assertEqual(ds.num_classes, 2)
            self.assertEqual(len(ds), 2)

    def test_dataset_loader_delegation(self):
        from data import TextDataset, DatasetLoader
        train_test = ([0, 1], [2, 3])
        canned_t = ("Text 0", "Text 1", "Text 2", "Text 3")
        canned_y = (0, 1, 0, 1)
        canned_c = ("neg", "pos")

        from unittest.mock import patch
        with patch('data.load_text_and_targets', return_value=(canned_t, canned_y, canned_c)):
            ds = TextDataset("sst2", train_test_indices=train_test)
            loader = DatasetLoader("sst2", batch_size=2, dataset=ds)
            self.assertTrue(hasattr(loader, "train_dataset"))
            self.assertTrue(hasattr(loader, "test_dataset"))
            self.assertEqual(len(loader.train_dataset), 2)
            self.assertEqual(len(loader.test_dataset), 2)

    def test_similarity_icl_dataset_warning(self):
        import warnings
        from data import SimilarityICLDataset
        train_test = ([0, 1], [2, 3])
        canned_t = ("Text 0", "Text 1", "Text 2", "Text 3")
        canned_y = (0, 1, 0, 1)
        canned_c = ("neg", "pos")

        from unittest.mock import patch
        with patch('data.load_text_and_targets', return_value=(canned_t, canned_y, canned_c)):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                sim_ds = SimilarityICLDataset("sst2", train_test_indices=train_test, num_shots=1)
                self.assertTrue(any(issubclass(item.category, DeprecationWarning) for item in w))
                self.assertIsNone(sim_ds.embeddings)

    def test_instruction_tuning_dataset(self):
        from data import InstructionTuningDataset
        train_test = ([0, 1], [2, 3])
        canned_t = ("Text 0", "Text 1", "Text 2", "Text 3")
        canned_y = (0, 1, 0, 1)
        canned_c = ("neg", "pos")

        from unittest.mock import patch
        with patch('data.load_text_and_targets', return_value=(canned_t, canned_y, canned_c)):
            it_ds = InstructionTuningDataset("sst2", train_test_indices=train_test)
            self.assertEqual(len(it_ds.context_samples), 2)
            batches = list(it_ds.batch_data_for_evaluation(batch=2))
            self.assertEqual(len(batches), 1)

    def test_dataset_loader_testloader_really_encodes(self):
        """Regression: DatasetLoader must forward the tokenizer to the pool,
        otherwise test batches are all-zero ids and eval collapses."""
        from data import FineTuningDataset, DatasetLoader

        class DummyTokenizer:
            def encode_plus(self, text, **kwargs):
                n = [len(text)] * 5
                return {"input_ids": n, "attention_mask": [1] * 5,
                        "token_type_ids": [0] * 5}

        train_test = ([0, 1], [2, 3])
        canned_t = ("Text 0", "Text 1", "Text 2", "Text 3")
        canned_y = (0, 1, 0, 1)
        canned_c = ("neg", "pos")

        from unittest.mock import patch
        with patch('data.load_text_and_targets', return_value=(canned_t, canned_y, canned_c)):
            ds = FineTuningDataset("sst2", train_test_indices=train_test,
                                   tokenizer=DummyTokenizer(), max_len=5)
            loader = DatasetLoader("sst2", batch_size=2, dataset=ds)
            batch = next(iter(loader.testloader()))
            ids = batch["ids"].tolist() if hasattr(batch["ids"], "tolist") else batch["ids"]
            flat = [t for seq in ids for t in seq]
            self.assertTrue(any(t != 0 for t in flat),
                            "testloader ids are all zeros: tokenizer was not forwarded")
            self.assertEqual(len(batch["targets"]), 2)


if __name__ == "__main__":
    unittest.main()



