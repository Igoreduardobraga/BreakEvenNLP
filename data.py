# data.py

import torch
import copy
import os
import pickle
import math
import numpy as np
from datasets import load_dataset
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
import pandas as pd
from torch.utils.data import RandomSampler, DataLoader, Dataset
from transformers import BertModel, BertTokenizer
from functools import lru_cache

class SeededRandomSampler(RandomSampler):

    def __init__(self, dataset, replacement=False, num_samples=None, seed=0):
        old_state = torch.get_rng_state()
        torch.manual_seed(seed)
        self.state = torch.get_rng_state()
        torch.set_rng_state(old_state)
        super(SeededRandomSampler, self).__init__(dataset, replacement, num_samples)
        self.dataset = dataset

    def __iter__(self):
        size = len(self.dataset)

        old_state = torch.get_rng_state()
        torch.set_rng_state(self.state)

        if self.replacement:
            iterator = iter(torch.randint(high=size, size=(self.num_samples,), dtype=torch.int64).tolist())
        else:
            iterator = iter(torch.randperm(size).tolist())

        self.state = torch.get_rng_state()
        torch.set_rng_state(old_state)
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

    def __init__(self, name, batch_size, dataset, shuffle_train_seed=0):
        self.name = name
        self.batch_size = batch_size
        self.shuffle_train_seed = shuffle_train_seed
        self.train_dataset = copy.deepcopy(dataset)
        self.train_dataset.train = True

        self.test_dataset = copy.deepcopy(dataset)
        self.test_dataset.train = False

    def trainloader(self):
        sampler = SeededRandomSampler(self.train_dataset, seed=self.shuffle_train_seed)
        trainloader = DataLoader(self.train_dataset, batch_size=self.batch_size, sampler=sampler, pin_memory=True)
        return trainloader

    def testloader(self):
        testloader = DataLoader(self.test_dataset, batch_size = 64, shuffle=False, pin_memory=True)
        return testloader


class TextDataset(Dataset):

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

        t, y, c = load_text_and_targets(self.dataset_name, self.prompt_format)
        self.text, self.targets, self.classes = list(t), list(y), list(c)
        self.num_classes = len(self.classes)

        self.split_train_test(train_test_indices=train_test_indices)
        self.select_labelled_data()


    def split_train_test(self, train_test_indices=None):
        if train_test_indices is None:
            raise ValueError("train_test_indices must be provided when using external KFold.")
        else:
            self.train_indices, self.test_indices = train_test_indices

        self.train_text = [self.text[idx] for idx in self.train_indices]
        self.train_targets = [self.targets[idx] for idx in self.train_indices]

        self.test_text = [self.text[idx] for idx in self.test_indices]
        self.test_targets = [self.targets[idx] for idx in self.test_indices] 
        


    def select_labelled_data(self):
        if self.num_labelled > 0:

            old_state = torch.get_rng_state()
            torch.manual_seed(self.label_seed)

            to_select = math.ceil(self.num_labelled / self.num_classes)

            targets = np.array(self.train_targets)

            texts = []
            labels = []
            train_indices = []
            for cls in range(self.num_classes):
                inds = np.argwhere(targets == cls).reshape(-1)
                indices = torch.randperm(len(inds))
                inds = inds[indices]
                inds = inds[:to_select]
                train_indices.extend(inds)
                for idx in inds:
                    texts.append(self.train_text[idx])
                    labels.append(self.train_targets[idx])
            indices = torch.randperm(len(labels))
            self.train_text = [texts[idx] for idx in indices]
            self.train_targets = [labels[idx] for idx in indices]
            self.train_indices = train_indices

            print(f'Number of selected Train samples: {len(self.train_targets)}')

            torch.set_rng_state(old_state)
        
        if not self.full_test and self.num_labelled_test > 0:

            old_state = torch.get_rng_state()
            torch.manual_seed(self.label_seed)

            to_select = math.ceil(self.num_labelled_test / self.num_classes)

            targets = np.array(self.test_targets)

            texts = []
            labels = []
            test_indices = []
            for cls in range(self.num_classes):
                inds = np.argwhere(targets == cls).reshape(-1)
                indices = torch.randperm(len(inds))
                inds = inds[indices]
                inds = inds[:to_select]
                test_indices.extend(inds)
                for idx in inds:
                    texts.append(self.test_text[idx])
                    labels.append(self.test_targets[idx])
            indices = torch.randperm(len(labels))
            self.test_text = [texts[idx] for idx in indices]
            self.test_targets = [labels[idx] for idx in indices]
            self.test_indices = test_indices
            print(len(self.test_text))

            print(f'Number of selected Test samples: {len(self.test_targets)}')

            torch.set_rng_state(old_state)

    def prepare_dataset_keywords(self):
        from prompter import PromptFormatter
        formatter = PromptFormatter(model_name="seq2seq", prompt_format=self.prompt_format)
        info = formatter.get_task_info(self.dataset_name, classes=self.classes)
        return info["instruction"], info["sentence_start"], info["answer_start"], info["task_type"]


class ICLDataset(TextDataset):

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
        texts, targets = self.__choose_shots()
        texts, targets = self.__sample_reorder(texts, targets)
        instructions, context_samples = self.__prepare_prompt(texts, targets)
        return instructions, context_samples

    
    def batch_data_for_evaluation(self, batch=64):
        start_idx = 0
        end_idx = batch

        while start_idx < len(self.test_text):
            data = self.test_text[start_idx : end_idx]
            labels = self.test_targets[start_idx : end_idx]

            yield data, labels

            start_idx = end_idx
            end_idx += batch


    def __len__(self):
        return len(self.test_text)
    

    def __choose_shots(self):
        to_choose = int(self.num_shots)
        
        old_state = torch.get_rng_state()
        torch.manual_seed(self.choice_seed)

        targets = np.array(self.train_targets)

        texts = []
        labels = []
        for cls in range(self.num_classes):
            inds = np.argwhere(targets == cls).reshape(-1)
            indices = torch.randperm(len(inds))
            inds = inds[indices]
            inds = inds[:to_choose]
            for idx in inds:
                texts.append(self.train_text[idx])
                labels.append(self.train_targets[idx])

        torch.set_rng_state(old_state)

        return texts, labels

    def __sample_reorder(self, texts, targets):
        old_state = torch.get_rng_state()
        torch.manual_seed(self.order_seed)

        indices = torch.randperm(len(texts))
        texts = [texts[idx] for idx in indices]
        targets = [targets[idx] for idx in indices]

        torch.set_rng_state(old_state)

        return texts, targets

    def __prepare_prompt(self, texts, targets):
        instruction, sentence_start, answer_start, task_type = self.prepare_dataset_keywords()
        instructions = {
            'instruction': instruction,
            'sentence_start': sentence_start,
            'answer_start': answer_start,
            'task_type': task_type
        }

        context_samples = [(texts[idx], self.classes[targets[idx]]) for idx in range(len(targets))]
        return instructions, context_samples

    


class SimilarityICLDataset(ICLDataset):

    def __init__(self, dataset_name, train_size=0.8, num_labelled=1000, num_labelled_test=1000,
                 label_seed=0, device=None, full_test=True, num_shots=4,
                 num_classes=2, choice_seed=0, order_seed=0, model_name='flan-t5',
                 prompt_format=0, train_test_indices=None):
        super(SimilarityICLDataset, self).__init__(dataset_name, train_size, num_labelled, num_labelled_test,
                                                   label_seed, device, full_test, num_shots,
                                                   num_classes, choice_seed, order_seed, model_name,
                                                   prompt_format, train_test_indices)
        self.num_shots = num_shots
        self.choice_seed = choice_seed
        self.order_seed = order_seed
        with open(os.path.join('data', f'{dataset_name}_embeddings.pkl'), 'rb') as file:
            self.embeddings = pickle.load(file)
        self.model_name = model_name
        self.instructions, self.context_samples = self.prepare_dataset_for_use()


    def prepare_dataset_for_use(self):
        texts, targets = self.__choose_shots()
        texts, targets = self.__sample_reorder(texts, targets)
        prompts, targets = self.__prepare_prompt(texts, targets)
        return prompts, targets

    
    def batch_data_for_evaluation(self, batch=64):
        start_idx = 0
        end_idx = batch

        while start_idx < len(self.prompts):
            data = self.prompts[start_idx : end_idx]
            labels = self.targets[start_idx : end_idx]

            yield data, labels

            start_idx = end_idx
            end_idx += batch
    

    def __choose_shots(self):
        to_choose = int(self.num_shots / self.num_classes)

        test_embeddings = self.embeddings[self.test_indices]
        train_true_embeddings = self.embeddings[self.true_indices]
        train_false_embeddings = self.embeddings[self.false_indices]

        texts = []
        targets = []
        old_state = torch.get_rng_state()
        torch.manual_seed(self.choice_seed)
        indices_true = cosine_similarity(test_embeddings, train_true_embeddings).argsort()[::-1][:, :to_choose]
        indices_false = cosine_similarity(test_embeddings, train_false_embeddings).argsort()[::-1][:, :to_choose]
        for true_idx, false_idx in zip(indices_true, indices_false):
            temp_texts = [self.train_text[self.used_true_indices[idx]] for idx in true_idx]
            temp_texts.extend([self.train_text[self.used_false_indices[idx]] for idx in false_idx])
            texts.append(temp_texts)
        targets = [self.train_targets[self.used_true_indices[idx]] for idx in indices_true[0]]
        targets.extend([self.train_targets[self.used_false_indices[idx]] for idx in indices_false[0]])
        torch.set_rng_state(old_state)

        return texts, targets


class PromptDataset(TextDataset):
    def __init__(self, dataset_name, train_size=0.8, num_labelled=1000, num_labelled_test=1000,
                 label_seed=0, device=None, full_test=True, model_name='flan-t5',
                 prompt_format=0, train_test_indices=None):
        super(PromptDataset, self).__init__(dataset_name, train_size, num_labelled, num_labelled_test,
                                            label_seed, device, full_test, prompt_format,
                                            train_test_indices)
        self.model_name = model_name
        self.instructions, self.context_samples = self.prepare_dataset_for_use()

    def prepare_dataset_for_use(self):
        instructions, context_samples = self.__prepare_prompt()
        return instructions, context_samples

    
    def batch_data_for_evaluation(self, batch=64):
        start_idx = 0
        end_idx = batch

        while start_idx < len(self.test_text):
            data = self.test_text[start_idx : end_idx]
            labels = self.test_targets[start_idx : end_idx]

            yield data, labels

            start_idx = end_idx
            end_idx += batch


    def __len__(self):
        return len(self.test_text)

    
    def __prepare_prompt(self):
        instruction, sentence_start, answer_start, task_type = self.prepare_dataset_keywords()
        instructions = {
            'instruction': instruction,
            'sentence_start': sentence_start,
            'answer_start': answer_start,
            'task_type': task_type
        }

        context_samples = []
        return instructions, context_samples


class InstructionTuningDataset(ICLDataset):
    def __init__(self, dataset_name, train_size=0.8, num_labelled=1000, num_labelled_test=1000,
                 label_seed=0, device=None, full_test=True, model_name='flan-t5',
                 prompt_format=0, train_test_indices=None):
        super(InstructionTuningDataset, self).__init__(dataset_name, train_size, num_labelled, num_labelled_test,
                                                       label_seed, device, full_test, 0, 0, 0, 0,
                                                       model_name, prompt_format, train_test_indices)
        self.model_name = model_name
        self.instructions, self.context_samples = self.prepare_dataset_for_use()

    def prepare_dataset_for_use(self):
        instructions, context_samples = self.__prepare_prompt(self.train_text, self.train_targets)
        return instructions, context_samples

    
    def batch_data_for_evaluation(self, batch=64):
        start_idx = 0
        end_idx = batch

        while start_idx < len(self.prompts):
            data = self.prompts[start_idx : end_idx]
            labels = self.targets[start_idx : end_idx]

            yield data, labels

            start_idx = end_idx
            end_idx += batch
    
    def __len__(self):
        return len(self.prompts)

    def __prepare_prompt(self, texts, targets):
        instruction, sentence_start, answer_start, task_type = self.prepare_dataset_keywords()
        instructions = {
            'instruction': instruction,
            'sentence_start': sentence_start,
            'answer_start': answer_start,
            'task_type': task_type
        }

        context_samples = [(texts[idx], self.classes[targets[idx]]) for idx in range(len(targets))]
        return instructions, context_samples


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