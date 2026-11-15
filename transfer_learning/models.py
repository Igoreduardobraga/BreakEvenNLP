import os
import sys
import random

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rng import RNGController, RNGStream, RNGSnapshot

try:
    import numpy as np
except ImportError:
    np = None

try:
    import torch
    import torch.nn as nn
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False
    torch = None
    class DummyModule:
        def __init__(self, *args, **kwargs):
            pass
    class nn:
        Module = DummyModule


try:
    from transformers import BertModel, RobertaModel, AutoModel, DebertaV2Model
except ImportError:
    BertModel = None
    RobertaModel = None
    AutoModel = None
    DebertaV2Model = None



class DeterministicModel():
    """
    Backward-compatibility wrapper delegating to RNGController and RNGStream.
    """

    def __init__(self, dropout_seed: int = 0):
        self.dropout_stream = RNGStream(seed=dropout_seed)

    def set_rng_state(self, seed: int):
        snapshot = RNGController.capture_state()
        RNGController.seed_all(seed)
        return snapshot

    def restore_rng_state(self, states):
        if isinstance(states, RNGSnapshot):
            states.restore()
        else:
            old_torch_state, old_torch_cuda_state, old_numpy_state, old_random_state = states
            snapshot = RNGSnapshot(
                python_state=old_random_state,
                numpy_state=old_numpy_state,
                torch_state=old_torch_state,
                cuda_state=old_torch_cuda_state
            )
            snapshot.restore()

    def get_rng_state(self):
        return RNGController.capture_state()


class BERTBase(nn.Module, DeterministicModel):

    def __init__(self, n_classes, init_seed=0, dropout_seed=0, trainable=True):
        self.name = 'bert-base'
        super(BERTBase, self).__init__()
        DeterministicModel.__init__(self, dropout_seed=dropout_seed)

        with RNGController.isolate(init_seed):
            self.bert = BertModel.from_pretrained('bert-base-uncased', return_dict=False)
            if not trainable:
                for param in self.bert.parameters():
                    param.requires_grad = False
            self.dropout = torch.nn.Dropout(p=0.3)
            self.output = torch.nn.Linear(self.bert.config.hidden_size, n_classes)

        self.dropout_states = self.dropout_stream._state

    def forward(self, input_ids, attention_mask, token_type_ids):
        _, bert_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        with self.dropout_stream:
            output = self.dropout(bert_output)
        output = self.output(output)
        self.dropout_states = self.dropout_stream._state
        return output


class RoBERTaBase(nn.Module, DeterministicModel):

    def __init__(self, n_classes, init_seed=0, dropout_seed=0, trainable=True):
        self.name = 'roberta-base'
        super(RoBERTaBase, self).__init__()
        DeterministicModel.__init__(self, dropout_seed=dropout_seed)

        with RNGController.isolate(init_seed):
            self.bert = RobertaModel.from_pretrained('roberta-base', return_dict=False)
            if not trainable:
                for param in self.bert.parameters():
                    param.requires_grad = False
            self.dropout = torch.nn.Dropout(p=0.3)
            self.output = torch.nn.Linear(self.bert.config.hidden_size, n_classes)

        self.dropout_states = self.dropout_stream._state

    def forward(self, input_ids, attention_mask, token_type_ids):
        _, bert_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        with self.dropout_stream:
            output = self.dropout(bert_output)
        output = self.output(output)
        self.dropout_states = self.dropout_stream._state
        return output


class DeBERTaBase(nn.Module, DeterministicModel):

    def __init__(self, n_classes, init_seed=0, dropout_seed=0, trainable=True):
        self.name = 'deberta-v3-base'
        super(DeBERTaBase, self).__init__()
        DeterministicModel.__init__(self, dropout_seed=dropout_seed)

        with RNGController.isolate(init_seed):
            loader = AutoModel or DebertaV2Model
            self.deberta = loader.from_pretrained('microsoft/deberta-v3-base', use_safetensors=True, return_dict=False)
            if not trainable:
                for param in self.deberta.parameters():
                    param.requires_grad = False
            config = self.deberta.config
            pooler_size = getattr(config, 'pooler_hidden_size', config.hidden_size)
            self.pooler = torch.nn.Linear(config.hidden_size, pooler_size)
            torch.nn.init.normal_(self.pooler.weight, mean=0.0, std=config.initializer_range)
            torch.nn.init.zeros_(self.pooler.bias)
            self.dropout = torch.nn.Dropout(p=0.3)
            self.output = torch.nn.Linear(pooler_size, n_classes)

        self.dropout_states = self.dropout_stream._state

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        kwargs = {'input_ids': input_ids, 'attention_mask': attention_mask}
        if token_type_ids is not None:
            kwargs['token_type_ids'] = token_type_ids
        outputs = self.deberta(**kwargs)
        cls_output = outputs[0][:, 0, :]
        pooled = torch.nn.functional.gelu(self.pooler(cls_output))
        with self.dropout_stream:
            output = self.dropout(pooled)
        output = self.output(output)
        self.dropout_states = self.dropout_stream._state
        return output