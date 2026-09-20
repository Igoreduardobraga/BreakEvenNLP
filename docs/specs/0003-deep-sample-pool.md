# Spec: Deep SamplePool Module

**Triage Label:** `ready-for-agent`  
**Status:** Ready for Implementation  
**Area:** Dataset Management & Sampling Pipeline  

---

## Problem Statement

Dataset handling in `BreakEvenNLP/data.py` suffers from several architectural defects:
1. **Shallow, Redundant Inheritance Hierarchy:** Five dataset subclasses (`TextDataset`, `ICLDataset`, `PromptDataset`, `InstructionTuningDataset`, `SimilarityICLDataset`) duplicate batch generation loops (`batch_data_for_evaluation`) across files.
2. **Hidden State Mutation & Costly Deepcopies:** `DatasetLoader` performs `copy.deepcopy(dataset)` and mutates internal state flags (`train_dataset.train = True`, `test_dataset.train = False`) at runtime, making datasets non-thread-safe and fragile.
3. **Broken Classes & Stale Attributes:** `SimilarityICLDataset` attempts to load missing `.pkl` embeddings from disk and accesses uninitialized attributes (`self.true_indices`, `self.false_indices`), causing runtime crashes. `InstructionTuningDataset` inherits from `ICLDataset` solely to duplicate prompt formatting that is now handled by `PromptFormatter`.
4. **Tight Coupling to PyTorch:** `TextDataset` forces inheritance from `torch.utils.data.Dataset`, even though LLM evaluation loops (Prompting and ICL) only need pure Python string lists and batch generators.

## Solution

Consolidate dataset partitioning, stratified label selection, few-shot demonstration sampling, and evaluation batching into a single deep `SamplePool` module located at `BreakEvenNLP/sample_pool.py`. 
`SamplePool` operates on pure Python data structures with deterministic RNG isolation, provides a clean generator for `ModelEvaluator`, exports lightweight PyTorch datasets on demand (`.to_torch_dataset()`) without state mutation, and provides thin backward-compatibility adapters in `data.py`.

## User Stories

1. As a researcher, I want `SamplePool` to partition datasets using external K-Fold train/test indices without mutating internal state flags.
2. As an evaluator, I want `SamplePool.batch_data_for_evaluation(batch_size, split)` to yield `(batch_texts, batch_labels)` directly for `ModelEvaluator`.
3. As a researcher running ICL experiments, I want `SamplePool.get_shots(num_shots, choice_seed, order_seed)` to return deterministically sampled and reordered `(text, label_str)` demonstration pairs.
4. As a researcher training models with PyTorch, I want `.to_torch_dataset(split, tokenizer, max_len)` to produce a PyTorch `Dataset` on demand, eliminating `copy.deepcopy` and mutable `.train` flags.
5. As a developer writing tests, I want `SamplePool` to accept in-memory `texts`, `targets`, and `classes` tuples, so that all partitioning and sampling logic can be verified in milliseconds without downloading Hugging Face datasets.
6. As a researcher running few-shot experiments with label budgets (`num_labelled`), I want each class to receive balanced representation `ceil(num_labelled / num_classes)`.
7. As a researcher running test sampling with `num_labelled_test`, I want the test split sampled with `label_seed` when `full_test=False`.
8. As a maintainer, I want existing dataset classes (`TextDataset`, `ICLDataset`, `PromptDataset`) in `data.py` to act as thin backward-compatibility wrappers delegating to `SamplePool`.
9. As a maintainer, I want broken classes like `SimilarityICLDataset` deprecated and isolated so they cannot cause uninitialized attribute errors.
10. As an experiment runner, I want `SamplePool` to be immutable after initialization so concurrent evaluation calls cannot corrupt sample distributions.

## Implementation Decisions

- The module is created at `BreakEvenNLP/sample_pool.py`.
- The public interface is:
  ```python
  class SamplePool:
      def __init__(
          self,
          dataset_name: str,
          train_test_indices: tuple[list[int], list[int]],
          num_labelled: int = 1000,
          num_labelled_test: int = 1000,
          label_seed: int = 0,
          full_test: bool = True,
          prompt_format: int = 0,
          texts: tuple = None,
          targets: tuple = None,
          classes: tuple = None
      ):
          ...

      def get_shots(self, num_shots: int, choice_seed: int = 0, order_seed: int = 0) -> list[tuple[str, str]]:
          ...

      def batch_data_for_evaluation(self, batch_size: int = 64, split: str = 'test'):
          ...

      def to_torch_dataset(self, split: str = 'train', tokenizer = None, max_len: int = 50):
          ...
  ```
- Stratified sampling logic (`select_labelled_data`) is encapsulated as an internal method producing immutable `train_text`, `train_targets`, `test_text`, `test_targets`.
- Lightweight `_TorchDatasetAdapter` implements `__len__` and `__getitem__` for PyTorch DataLoaders without mutable flags.
- In `BreakEvenNLP/data.py`:
  - `TextDataset`, `ICLDataset`, `PromptDataset`, and `DatasetLoader` are refactored to delegate to `SamplePool`.
  - `SimilarityICLDataset` is deprecated and provides an explicit warning.

## Testing Decisions

- Tests verify:
  - Stratified partition size and deterministic selection across multiple seeds.
  - Balanced k-shot demonstration selection and order permutation.
  - Evaluation batching without state alteration or sample leakage.
  - PyTorch dataset adapter tokenization and shape consistency.
  - Backward compatibility of `TextDataset`, `ICLDataset`, and `PromptDataset` wrappers.
- Test suite located at `BreakEvenNLP/tests/test_sample_pool.py`.
- 100% offline, deterministic tests (0 Hugging Face downloads, 0 GPU memory).

## Out of Scope

- Encapsulating random number generators (`RNGController`) &mdash; addressed by Candidate 4.
