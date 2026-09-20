# sample_pool.py
"""
Deep SamplePool module for BreakEvenNLP.
Consolidates dataset loading, stratified partitioning, deterministic k-shot
demonstration extraction, and batch evaluation into an immutable data container.
"""

import math
import random
from typing import Any, Generator, List, Optional, Tuple, Union

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


def _deterministic_permute(elements: list, seed: int) -> list:
    """Permutes a list deterministically using torch if available, otherwise random.Random."""
    if _HAS_TORCH:
        old_state = torch.get_rng_state()
        torch.manual_seed(seed)
        perm = torch.randperm(len(elements)).tolist()
        torch.set_rng_state(old_state)
        return [elements[i] for i in perm]
    else:
        rng = random.Random(seed)
        copy_el = list(elements)
        rng.shuffle(copy_el)
        return copy_el


class SamplePool:
    """
    Deep module managing dataset samples, stratified train/test partitions,
    few-shot demonstrations, and evaluation generators without mutable flags.
    """

    def __init__(
        self,
        dataset_name: str,
        train_test_indices: Tuple[List[int], List[int]],
        train_size: float = 0.8,
        num_labelled: int = 1000,
        num_labelled_test: int = 1000,
        label_seed: int = 0,
        device: Any = None,
        full_test: bool = True,
        prompt_format: int = 0,
        texts: Optional[Tuple[str, ...]] = None,
        targets: Optional[Tuple[int, ...]] = None,
        classes: Optional[Tuple[str, ...]] = None,
    ):
        self.dataset_name = dataset_name
        self.train_size = train_size
        self.num_labelled = num_labelled
        self.num_labelled_test = num_labelled_test
        if not full_test and self.num_labelled_test == 0:
            self.num_labelled_test = self.num_labelled

        self.label_seed = label_seed
        self.device = device
        self.full_test = full_test
        self.prompt_format = prompt_format

        if texts is None or targets is None or classes is None:
            from data import load_text_and_targets
            t, y, c = load_text_and_targets(self.dataset_name, self.prompt_format)
            self.text = list(t)
            self.targets = list(y)
            self.classes = list(c)
        else:
            self.text = list(texts)
            self.targets = list(targets)
            self.classes = list(classes)

        self.num_classes = len(self.classes)

        if train_test_indices is None:
            raise ValueError("train_test_indices must be provided when using external KFold.")
        self.train_indices, self.test_indices = train_test_indices

        self.train_text = [self.text[idx] for idx in self.train_indices]
        self.train_targets = [self.targets[idx] for idx in self.train_indices]

        self.test_text = [self.text[idx] for idx in self.test_indices]
        self.test_targets = [self.targets[idx] for idx in self.test_indices]

        self._select_labelled_data()

    @classmethod
    def from_dataset_name(
        cls,
        dataset_name: str,
        train_test_indices: Tuple[List[int], List[int]],
        train_size: float = 0.8,
        num_labelled: int = 1000,
        num_labelled_test: int = 1000,
        label_seed: int = 0,
        device: Any = None,
        full_test: bool = True,
        prompt_format: int = 0,
    ) -> "SamplePool":
        return cls(
            dataset_name=dataset_name,
            train_test_indices=train_test_indices,
            train_size=train_size,
            num_labelled=num_labelled,
            num_labelled_test=num_labelled_test,
            label_seed=label_seed,
            device=device,
            full_test=full_test,
            prompt_format=prompt_format,
        )

    def _select_labelled_data(self):
        """Applies stratified budget selection to train and test splits."""
        if self.num_labelled > 0 and len(self.train_targets) > 0:
            to_select = math.ceil(self.num_labelled / self.num_classes)
            texts = []
            labels = []
            new_train_indices = []

            for cls in range(self.num_classes):
                cls_indices = [i for i, target in enumerate(self.train_targets) if target == cls]
                shuffled_cls = _deterministic_permute(cls_indices, self.label_seed)
                selected = shuffled_cls[:to_select]
                new_train_indices.extend([self.train_indices[i] for i in selected])
                for idx in selected:
                    texts.append(self.train_text[idx])
                    labels.append(self.train_targets[idx])

            # Reorder the combined sampled items
            paired = list(zip(texts, labels, new_train_indices))
            shuffled_paired = _deterministic_permute(paired, self.label_seed)

            self.train_text = [p[0] for p in shuffled_paired]
            self.train_targets = [p[1] for p in shuffled_paired]
            self.train_indices = [p[2] for p in shuffled_paired]

        if not self.full_test and self.num_labelled_test > 0 and len(self.test_targets) > 0:
            to_select = math.ceil(self.num_labelled_test / self.num_classes)
            texts = []
            labels = []
            new_test_indices = []

            for cls in range(self.num_classes):
                cls_indices = [i for i, target in enumerate(self.test_targets) if target == cls]
                shuffled_cls = _deterministic_permute(cls_indices, self.label_seed)
                selected = shuffled_cls[:to_select]
                new_test_indices.extend([self.test_indices[i] for i in selected])
                for idx in selected:
                    texts.append(self.test_text[idx])
                    labels.append(self.test_targets[idx])

            paired = list(zip(texts, labels, new_test_indices))
            shuffled_paired = _deterministic_permute(paired, self.label_seed)

            self.test_text = [p[0] for p in shuffled_paired]
            self.test_targets = [p[1] for p in shuffled_paired]
            self.test_indices = [p[2] for p in shuffled_paired]

    def __len__(self) -> int:
        return len(self.test_text)

    @property
    def instructions(self) -> dict:
        from prompter import PromptFormatter
        prompter = PromptFormatter(model_name="seq2seq", prompt_format=self.prompt_format)
        return prompter.get_task_info(self.dataset_name, classes=self.classes)

    @property
    def context_samples(self) -> List[Tuple[str, str]]:
        if not hasattr(self, '_context_samples'):
            self._context_samples = []
        return self._context_samples

    @context_samples.setter
    def context_samples(self, samples: List[Tuple[str, str]]):
        self._context_samples = list(samples)

    def get_shots(self, num_shots: int, choice_seed: int = 0, order_seed: int = 0) -> List[Tuple[str, str]]:
        """
        Samples num_shots demonstrations per class from train split deterministically
        and reorders them using order_seed.
        """
        to_choose = int(num_shots)
        texts = []
        labels = []

        for cls in range(self.num_classes):
            cls_indices = [i for i, target in enumerate(self.train_targets) if target == cls]
            shuffled_cls = _deterministic_permute(cls_indices, choice_seed)
            selected = shuffled_cls[:to_choose]
            for idx in selected:
                texts.append(self.train_text[idx])
                labels.append(self.train_targets[idx])

        paired = list(zip(texts, labels))
        shuffled_paired = _deterministic_permute(paired, order_seed)
        return [(p[0], self.classes[p[1]]) for p in shuffled_paired]

    def batch_data_for_evaluation(
        self, batch_size: int = 64, split: str = "test"
    ) -> Generator[Tuple[List[str], List[int]], None, None]:
        """
        Yields batches of (texts, labels) without mutating internal state.
        """
        src_texts = self.test_text if split == "test" else self.train_text
        src_targets = self.test_targets if split == "test" else self.train_targets

        start_idx = 0
        while start_idx < len(src_texts):
            end_idx = start_idx + batch_size
            yield src_texts[start_idx:end_idx], src_targets[start_idx:end_idx]
            start_idx = end_idx

    def to_torch_dataset(self, split: str = "train", tokenizer: Any = None, max_len: int = 50):
        """
        Exports an immutable PyTorch-compatible Dataset without mutating pool flags.
        """
        texts = self.train_text if split == "train" else self.test_text
        targets = self.train_targets if split == "train" else self.test_targets
        return _TorchDatasetAdapter(texts=texts, targets=targets, tokenizer=tokenizer, max_len=max_len)


class _TorchDatasetAdapter:
    """Lightweight Dataset adapter that encodes samples on the fly without state mutation."""

    def __init__(self, texts: List[str], targets: List[int], tokenizer: Any = None, max_len: int = 50):
        self.texts = list(texts)
        self.targets = list(targets)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int) -> dict:
        text = str(self.texts[index])
        target = self.targets[index]

        if self.tokenizer is not None and hasattr(self.tokenizer, "encode_plus"):
            inputs = self.tokenizer.encode_plus(
                text,
                add_special_tokens=True,
                max_length=self.max_len,
                padding="max_length",
                truncation=True,
                return_token_type_ids=True,
            )
            ids = inputs.get("input_ids", [])
            mask = inputs.get("attention_mask", [])
            token_type_ids = inputs.get("token_type_ids", [])
        else:
            ids = [0] * self.max_len
            mask = [1] * self.max_len
            token_type_ids = [0] * self.max_len

        if _HAS_TORCH:
            return {
                "ids": torch.tensor(ids, dtype=torch.long),
                "mask": torch.tensor(mask, dtype=torch.long),
                "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
                "targets": torch.tensor(target, dtype=torch.long),
            }

        return {
            "ids": ids,
            "mask": mask,
            "token_type_ids": token_type_ids,
            "targets": target,
        }


