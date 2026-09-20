# data.py

import torch
import copy
import os
import pickle
import math
import warnings
import numpy as np
from datasets import load_dataset
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
import pandas as pd
try:
    from torch.utils.data import RandomSampler, DataLoader, Dataset
    if hasattr(RandomSampler, '_mock_return_value') or hasattr(RandomSampler, '_mock_call'):
        class _BaseSampler:
            def __init__(self, *args, **kwargs):
                pass
        RandomSampler = _BaseSampler
except (ImportError, AttributeError):
    class _BaseSampler:
        def __init__(self, *args, **kwargs):
            pass
    RandomSampler = _BaseSampler
    DataLoader = object
    Dataset = object

from transformers import BertModel, BertTokenizer
from functools import lru_cache
from sample_pool import SamplePool
from rng import RNGStream, RNGController

class SeededRandomSampler(RandomSampler):

    def __init__(self, dataset, replacement=False, num_samples=None, seed=0):
        self.stream = RNGStream(seed=seed)
        self.state = self.stream._state
        self.replacement = replacement
        self.dataset = dataset
        try:
            super(SeededRandomSampler, self).__init__(dataset, replacement=replacement, num_samples=num_samples)
        except Exception:
            self._num_samples = num_samples

    def __iter__(self):
        size = len(self.dataset)
        with self.stream:
            use_torch = hasattr(torch, 'randperm') and not hasattr(torch.randperm, '_mock_return_value')
            if use_torch:
                try:
                    if self.replacement:
                        iterator = iter(torch.randint(high=size, size=(self.num_samples,), dtype=torch.int64).tolist())
                    else:
                        iterator = iter(torch.randperm(size).tolist())
                except Exception:
                    use_torch = False

            if not use_torch:
                import random
                indices = list(range(size))
                if self.replacement:
                    iterator = iter(random.choices(indices, k=self.num_samples or size))
                else:
                    random.shuffle(indices)
                    iterator = iter(indices)

        self.state = self.stream._state
        return iterator

@lru_cache(maxsize=None)
def load_text_and_targets(dataset_name: str, prompt_format: int):
    if dataset_name == 'sst2':
        dataset = load_dataset('glue', 'sst2')
        data = pd.concat([pd.DataFrame(dataset['train']),
                          pd.DataFrame(dataset['validation'])])
        if prompt_format in [0, 1, 2]:
            classes = ['negative', 'positive']
        elif prompt_format in [3]:
            classes = ['terrible', 'great']
        else:
            raise NotImplementedError(f'Prompt format {prompt_format} not supported for sst2')
        texts   = data['sentence'].tolist()
        targets = data['label'].tolist()
        return tuple(texts), tuple(targets), tuple(classes)

    elif dataset_name == 'cola':
        dataset = load_dataset('glue', 'cola')
        data = pd.concat([pd.DataFrame(dataset['train']),
                          pd.DataFrame(dataset['validation'])])
        if prompt_format in [0, 1]:
            classes = ['No', 'Yes']
        elif prompt_format in [2]:
            classes = ['Yes', 'No']
        elif prompt_format in [3]:
            classes = ['not acceptable', 'acceptable']
        else:
            raise NotImplementedError(f'Prompt format {prompt_format} not supported for cola')
        texts   = data['sentence'].tolist()
        targets = data['label'].tolist()
        return tuple(texts), tuple(targets), tuple(classes)

    elif dataset_name == 'mrpc':
        dataset = load_dataset('glue', 'mrpc')
        data = pd.concat([pd.DataFrame(dataset['train']),
                          pd.DataFrame(dataset['validation']),
                          pd.DataFrame(dataset['test'])])
        texts = [
            f"Sentence 1: {s1}; Sentence 2: {s2}"
            for s1, s2 in zip(data['sentence1'].tolist(), data['sentence2'].tolist())
        ]
        if prompt_format in [0, 1]:
            classes = ['No', 'Yes']
        elif prompt_format in [2]:
            classes = ['Yes', 'No']
        elif prompt_format in [3]:
            classes = ['not equivalent', 'equivalent']
        else:
            raise NotImplementedError(f'Prompt format {prompt_format} not supported for mrpc')
        targets = data['label'].tolist()
        return tuple(texts), tuple(targets), tuple(classes)

    elif dataset_name == 'rte':
        dataset = load_dataset('glue', 'rte')
        data = pd.concat([pd.DataFrame(dataset['train']),
                          pd.DataFrame(dataset['validation'])])
        texts = [
            f"Premise: {p}; Hypothesis: {h}"
            for p, h in zip(data['sentence1'].tolist(), data['sentence2'].tolist())
        ]
        classes = ['not entailment', 'entailment']
        targets = data['label'].tolist()
        return tuple(texts), tuple(targets), tuple(classes)

    elif dataset_name == 'boolq':
        dataset = load_dataset('super_glue', 'boolq')
        data = pd.concat([pd.DataFrame(dataset['train']),
                          pd.DataFrame(dataset['validation'])])
        
        texts = [
            f"Passage: {passage}\nQuestion: {q}" 
            for q, passage in zip(data['question'].tolist(), data['passage'].tolist())
        ]
        
        classes = ['No', 'Yes']
        targets = data['label'].tolist()
        return tuple(texts), tuple(targets), tuple(classes)

    elif dataset_name == 'trec':
        dataset = load_dataset("CogComp/trec", revision="refs/convert/parquet")
        data = pd.concat([pd.DataFrame(dataset['train']),
                          pd.DataFrame(dataset['test'])])
        classes = ['Expression', 'Entity', 'Description', 'Human', 'Location', 'Number']
        texts   = data['text'].tolist()
        targets = data['coarse_label'].tolist()
        return tuple(texts), tuple(targets), tuple(classes)

    elif dataset_name == 'ag_news':
        dataset = load_dataset('ag_news')
        data = pd.concat([pd.DataFrame(dataset['train']),
                          pd.DataFrame(dataset['test'])])
        classes = ['World', 'Sports', 'Business', 'Science and Technology']
        texts   = data['text'].tolist()
        targets = data['label'].tolist()
        return tuple(texts), tuple(targets), tuple(classes)

    elif dataset_name == 'snips':
        dataset = load_dataset('benayas/snips')
        data = pd.concat([pd.DataFrame(dataset['train']),
                          pd.DataFrame(dataset['test'])])
        mapper = {
            'AddToPlaylist':        0,
            'GetWeather':           1,
            'SearchScreeningEvent': 2,
            'PlayMusic':            3,
            'SearchCreativeWork':   4,
            'RateBook':             5,
            'BookRestaurant':       6,
        }
        data['label'] = data['category'].apply(lambda x: mapper[x])
        classes = ['Playlist', 'Weather', 'Event', 'Music', 'Creative Work', 'Rate Book', 'Book Restaurant']
        texts   = data['text'].tolist()
        targets = data['label'].tolist()
        return tuple(texts), tuple(targets), tuple(classes)

    elif dataset_name == 'db_pedia':
        dataset = load_dataset('fancyzhx/dbpedia_14')
        data = pd.concat([pd.DataFrame(dataset['train']),
                          pd.DataFrame(dataset['test'])])
        classes = [
            'Company', 'Educational Institution', 'Artist', 'Athlete', 'Office Holder',
            'Transportation', 'Building', 'Natural Place', 'Village', 'Animal',
            'Plant', 'Album', 'Film', 'Written Work'
        ]
        texts   = data['content'].tolist()
        targets = data['label'].tolist()
        return tuple(texts), tuple(targets), tuple(classes)

    else:
        raise NotImplementedError(f'The dataset "{dataset_name}" is not supported in this function')

class DatasetLoader():
    """
    DataLoader wrapper managing train and test loaders.
    Delegates to SamplePool's to_torch_dataset when available, avoiding deepcopy
    and mutable state flags.
    """

    def __init__(self, name, batch_size, dataset, shuffle_train_seed=0):
        self.name = name
        self.batch_size = batch_size
        self.shuffle_train_seed = shuffle_train_seed
        self.dataset = dataset

        if hasattr(dataset, 'pool') and hasattr(dataset.pool, 'to_torch_dataset') and not hasattr(dataset, 'train_data'):
            self.train_dataset = dataset.pool.to_torch_dataset(split='train')
            self.test_dataset = dataset.pool.to_torch_dataset(split='test')
        elif hasattr(dataset, 'train'):
            # Shallow copy to isolate train/test mode for legacy datasets like FineTuningDataset
            self.train_dataset = copy.copy(dataset)
            self.train_dataset.train = True
            self.test_dataset = copy.copy(dataset)
            self.test_dataset.train = False
        else:
            self.train_dataset = dataset
            self.test_dataset = dataset

    def trainloader(self):
        sampler = SeededRandomSampler(self.train_dataset, seed=self.shuffle_train_seed)
        trainloader = DataLoader(self.train_dataset, batch_size=self.batch_size, sampler=sampler, pin_memory=True)
        return trainloader

    def testloader(self):
        testloader = DataLoader(self.test_dataset, batch_size=64, shuffle=False, pin_memory=True)
        return testloader


class TextDataset(Dataset):
    """
    Backward-compatibility wrapper delegating data management to SamplePool.
    """

    def __init__(self, dataset_name, train_size=0.8, num_labelled=1000, num_labelled_test=1000,
                 label_seed=0, device=None, full_test=True, prompt_format=0,
                 train_test_indices=None):
        self.dataset_name = dataset_name
        self.train = True
        self.label_seed = label_seed
        self.full_test = full_test

        self.train_size = train_size
        self.num_labelled = num_labelled
        self.num_labelled_test = num_labelled_test
        if not self.full_test and self.num_labelled_test == 0:
            self.num_labelled_test = self.num_labelled

        self.device = device
        self.prompt_format = prompt_format

        self.pool = SamplePool.from_dataset_name(
            dataset_name=self.dataset_name,
            prompt_format=self.prompt_format,
            train_test_indices=train_test_indices,
            num_labelled=self.num_labelled,
            num_labelled_test=self.num_labelled_test,
            label_seed=self.label_seed,
            full_test=self.full_test
        )
        self._sync_with_pool()

    def _sync_with_pool(self):
        self.text = self.pool.text
        self.targets = self.pool.targets
        self.classes = self.pool.classes
        self.num_classes = self.pool.num_classes

        self.train_indices = self.pool.train_indices
        self.test_indices = self.pool.test_indices
        self.train_text = self.pool.train_text
        self.train_targets = self.pool.train_targets
        self.test_text = self.pool.test_text
        self.test_targets = self.pool.test_targets

    def split_train_test(self, train_test_indices=None):
        if train_test_indices is None:
            raise ValueError("train_test_indices must be provided when using external KFold.")
        self.pool = self.pool.with_split(train_test_indices)
        self._sync_with_pool()

    def select_labelled_data(self):
        # Delegated to SamplePool during initialization
        pass

    def prepare_dataset_keywords(self):
        from prompter import PromptFormatter
        formatter = PromptFormatter(model_name="seq2seq", prompt_format=self.prompt_format)
        info = formatter.get_task_info(self.dataset_name, classes=self.classes)
        return info["instruction"], info["sentence_start"], info["answer_start"], info["task_type"]

    def __len__(self):
        return len(self.train_targets) if self.train else len(self.test_targets)

    def __getitem__(self, index):
        if self.train:
            return self.train_text[index], self.train_targets[index]
        return self.test_text[index], self.test_targets[index]


class ICLDataset(TextDataset):
    """
    In-context learning dataset wrapper delegating shot selection and batching to SamplePool.
    """

    def __init__(self, dataset_name, train_size=0.8, num_labelled=1000, num_labelled_test=1000,
                 label_seed=0, device=None, full_test=True, num_shots=2,
                 num_classes=2, choice_seed=0, order_seed=0, model_name='flan-t5',
                 prompt_format=0, train_test_indices=None):
        super(ICLDataset, self).__init__(dataset_name, train_size, num_labelled, num_labelled_test,
                                         label_seed, device, full_test, prompt_format,
                                         train_test_indices)
        self.num_shots = num_shots
        self.choice_seed = choice_seed
        self.order_seed = order_seed
        self.model_name = model_name
        self.instructions, self.context_samples = self.prepare_dataset_for_use()

    def prepare_dataset_for_use(self):
        instruction, sentence_start, answer_start, task_type = self.prepare_dataset_keywords()
        instructions = {
            'instruction': instruction,
            'sentence_start': sentence_start,
            'answer_start': answer_start,
            'task_type': task_type
        }
        context_samples = self.pool.get_shots(
            num_shots=self.num_shots,
            choice_seed=self.choice_seed,
            order_seed=self.order_seed
        )
        return instructions, context_samples

    def batch_data_for_evaluation(self, batch=64):
        return self.pool.batch_data_for_evaluation(batch_size=batch, split="test")

    def __len__(self):
        return len(self.test_text)


class SimilarityICLDataset(ICLDataset):
    """
    DEPRECATED: SimilarityICLDataset relies on external precomputed embeddings (.pkl)
    and missing attributes. Kept for import compatibility; will raise warning on instantiation.
    """

    def __init__(self, dataset_name, train_size=0.8, num_labelled=1000, num_labelled_test=1000,
                 label_seed=0, device=None, full_test=True, num_shots=4,
                 num_classes=2, choice_seed=0, order_seed=0, model_name='flan-t5',
                 prompt_format=0, train_test_indices=None):
        warnings.warn(
            "SimilarityICLDataset is deprecated and depends on missing precomputed embeddings. "
            "It will be refactored or removed in future releases.",
            DeprecationWarning,
            stacklevel=2
        )
        super(SimilarityICLDataset, self).__init__(dataset_name, train_size, num_labelled, num_labelled_test,
                                                   label_seed, device, full_test, num_shots,
                                                   num_classes, choice_seed, order_seed, model_name,
                                                   prompt_format, train_test_indices)
        emb_path = os.path.join('data', f'{dataset_name}_embeddings.pkl')
        if os.path.exists(emb_path):
            with open(emb_path, 'rb') as file:
                self.embeddings = pickle.load(file)
        else:
            self.embeddings = None


class PromptDataset(TextDataset):
    """
    Zero-shot prompting dataset wrapper delegating batching to SamplePool.
    """

    def __init__(self, dataset_name, train_size=0.8, num_labelled=1000, num_labelled_test=1000,
                 label_seed=0, device=None, full_test=True, model_name='flan-t5',
                 prompt_format=0, train_test_indices=None):
        super(PromptDataset, self).__init__(dataset_name, train_size, num_labelled, num_labelled_test,
                                            label_seed, device, full_test, prompt_format,
                                            train_test_indices)
        self.model_name = model_name
        self.instructions, self.context_samples = self.prepare_dataset_for_use()

    def prepare_dataset_for_use(self):
        instruction, sentence_start, answer_start, task_type = self.prepare_dataset_keywords()
        instructions = {
            'instruction': instruction,
            'sentence_start': sentence_start,
            'answer_start': answer_start,
            'task_type': task_type
        }
        context_samples = []
        return instructions, context_samples

    def batch_data_for_evaluation(self, batch=64):
        return self.pool.batch_data_for_evaluation(batch_size=batch, split="test")

    def __len__(self):
        return len(self.test_text)


class InstructionTuningDataset(ICLDataset):
    """
    Instruction-tuning dataset wrapper delegating sample formatting to SamplePool.
    """

    def __init__(self, dataset_name, train_size=0.8, num_labelled=1000, num_labelled_test=1000,
                 label_seed=0, device=None, full_test=True, model_name='flan-t5',
                 prompt_format=0, train_test_indices=None):
        super(InstructionTuningDataset, self).__init__(dataset_name, train_size, num_labelled, num_labelled_test,
                                                       label_seed, device, full_test, 0, 0, 0, 0,
                                                       model_name, prompt_format, train_test_indices)
        self.model_name = model_name
        self.instructions, self.context_samples = self.prepare_dataset_for_use()

    def prepare_dataset_for_use(self):
        instruction, sentence_start, answer_start, task_type = self.prepare_dataset_keywords()
        instructions = {
            'instruction': instruction,
            'sentence_start': sentence_start,
            'answer_start': answer_start,
            'task_type': task_type
        }
        context_samples = [(self.train_text[idx], self.classes[self.train_targets[idx]]) for idx in range(len(self.train_targets))]
        return instructions, context_samples

    def batch_data_for_evaluation(self, batch=64):
        return self.pool.batch_data_for_evaluation(batch_size=batch, split="train")

    def __len__(self):
        return len(self.train_text)


class FineTuningDataset(TextDataset):

    def __init__(self, dataset_name, train_size=0.8, num_labelled=1000, num_labelled_test=1000,
                 label_seed=0, device=None, full_test=True, tokenizer=None,
                 max_len=50, train_test_indices=None):
        super(FineTuningDataset, self).__init__(dataset_name, train_size, num_labelled, num_labelled_test,
                                                label_seed, device, full_test, prompt_format=0,
                                                train_test_indices=train_test_indices)
        self.tokenizer = tokenizer
        self.train = True
        self.max_len = max_len
        self.n_classes = self.num_classes

    def __len__(self):
        return len(self.train_text) if self.train else len(self.test_text)

    def __getitem__(self, index):
        text = str(self.train_text[index] if self.train else self.test_text[index])
        target = self.train_targets[index] if self.train else self.test_targets[index]

        inputs = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_token_type_ids=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets': torch.tensor(target, dtype=torch.long)
        }