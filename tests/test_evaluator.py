import unittest
from types import SimpleNamespace

from evaluator import ModelEvaluator, _CausalLMAdapter, _ChoiceConstraint, _parse_results


class DummyEncoded:
    def __init__(self, batch_size):
        self.batch_size = batch_size

    def to(self, device):
        return self


class DummyTokenizer:
    def __call__(self, texts, **kwargs):
        return DummyEncoded(len(texts))

    def batch_decode(self, token_ids, skip_special_tokens=True):
        return list(token_ids)


class DummySeq2SeqModel:
    def __init__(self, canned_outputs):
        self.canned_outputs = canned_outputs

    def generate(self, **kwargs):
        return self.canned_outputs


class DummyCausalInputs:
    def __init__(self, input_ids):
        self.input_ids = input_ids
        self.shape = (len(input_ids), len(input_ids[0]))

    def to(self, device):
        return self

    def __getitem__(self, idx):
        return self.input_ids[idx]


class DummyCausalTokenizer:
    def __init__(self):
        self.eos_token_id = 99
        self.pad_token_id = 99

    def convert_tokens_to_ids(self, token):
        return 99

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return " | ".join(f"{m['role']}: {m['content']}" for m in messages)

    def __call__(self, texts, return_tensors=None, **kwargs):
        if isinstance(texts, str):
            texts = [texts]
        # Simulate 4 tokens in prompt
        return DummyCausalInputs([[10, 11, 12, 13] for _ in texts])

    def decode(self, token_ids, skip_special_tokens=True):
        mapping = {20: 'positive', 21: 'negative', 22: 'ambiguous response'}
        return " ".join(mapping.get(t, f"tok_{t}") for t in token_ids)


class DummyCausalModel:
    def __init__(self, generated_tokens_per_call):
        self.generated_tokens_per_call = list(generated_tokens_per_call)
        self.call_idx = 0

    def generate(self, input_ids=None, **kwargs):
        # Generates prompt tokens + generated token
        tokens = self.generated_tokens_per_call[self.call_idx % len(self.generated_tokens_per_call)]
        self.call_idx += 1
        full_seq = []
        for seq in input_ids:
            full_seq.append(list(seq) + list(tokens))
        return full_seq


class DummyDataset:
    def __init__(self, texts=None, targets=None, classes=None):
        self.classes = classes or ['negative', 'positive']
        self.test_text = texts if texts is not None else ['I love this', 'I hate this']
        self.test_targets = targets if targets is not None else [1, 0]
        self.instructions = {
            'instruction': "Analyze the sentiment. Options: 'negative', 'positive'.",
            'sentence_start': 'Text',
            'answer_start': 'Answer',
            'task_type': 'sentiment analysis'
        }
        self.context_samples = [('loved it', 'positive'), ('hated it', 'negative')]

    def batch_data_for_evaluation(self, batch_size=64):
        for i in range(0, len(self.test_text), batch_size):
            yield self.test_text[i:i + batch_size], self.test_targets[i:i + batch_size]


class TestParseResults(unittest.TestCase):

    def test_nested_class_names_resolve_to_longest_match(self):
        classes = ["not entailment", "entailment"]
        self.assertEqual(_parse_results("not entailment", classes), 0)
        self.assertEqual(_parse_results("Entailment.", classes), 1)

    def test_two_distinct_answers_stay_ambiguous(self):
        classes = ["not entailment", "entailment"]
        self.assertEqual(_parse_results("not entailment, entailment", classes), -1)
        self.assertEqual(_parse_results("positive or negative", ["negative", "positive"]), -1)

    def test_overlapping_but_not_nested_names(self):
        classes = ["Playlist", "Rate Book", "Book Restaurant"]
        self.assertEqual(_parse_results("Rate Book", classes), 1)
        self.assertEqual(_parse_results("Book Restaurant", classes), 2)
        self.assertEqual(_parse_results("2", classes), 1)


class TestModelEvaluatorSeq2Seq(unittest.TestCase):

    def test_seq2seq_never_truncates_the_prompt(self):
        calls = []

        class RecordingTokenizer(DummyTokenizer):
            def __call__(self, texts, **kwargs):
                calls.append(kwargs)
                return super().__call__(texts, **kwargs)

        evaluator = ModelEvaluator(model=DummySeq2SeqModel(['positive', 'negative']),
                                   tokenizer=RecordingTokenizer(), model_name='flan-t5',
                                   batch_size=2, device='cpu')
        evaluator.evaluate(DummyDataset(), mode='icl')
        self.assertTrue(calls)
        self.assertTrue(all(c.get('truncation') is False for c in calls))

    def test_seq2seq_prompting_returns_standardized_tuple(self):
        tokenizer = DummyTokenizer()
        model = DummySeq2SeqModel(canned_outputs=['positive', 'negative'])
        dataset = DummyDataset()

        evaluator = ModelEvaluator(
            model=model,
            tokenizer=tokenizer,
            model_name='flan-t5',
            batch_size=2,
            prompt_format=0,
            device='cpu'
        )

        golden, predicted, decodeds = evaluator.evaluate(dataset, mode='prompting')

        self.assertEqual(golden, [1, 0])
        self.assertEqual(predicted, [1, 0])
        self.assertEqual(decodeds, ['positive', 'negative'])

    def test_seq2seq_icl_evaluation(self):
        tokenizer = DummyTokenizer()
        model = DummySeq2SeqModel(canned_outputs=['positive', 'negative'])
        dataset = DummyDataset()

        evaluator = ModelEvaluator(
            model=model,
            tokenizer=tokenizer,
            model_name='flan-t5',
            batch_size=2,
            prompt_format=0,
            device='cpu'
        )

        golden, predicted, decodeds = evaluator.evaluate(dataset, mode='icl')

        self.assertEqual(golden, [1, 0])
        self.assertEqual(predicted, [1, 0])
        self.assertEqual(len(decodeds), 2)

    def test_seq2seq_with_batch_encoding_dictionary_unpacking(self):
        from collections import UserDict

        class MockBatchEncoding(UserDict):
            def to(self, device):
                return self

        class MockTokenizer:
            def __call__(self, texts, **kwargs):
                return MockBatchEncoding({'input_ids': [[1, 2, 3] for _ in texts], 'attention_mask': [[1, 1, 1] for _ in texts]})

            def batch_decode(self, token_ids, skip_special_tokens=True):
                return ['positive' for _ in token_ids]

        class MockModel:
            def generate(self, **kwargs):
                self.received_kwargs = kwargs
                return [[10] for _ in kwargs['input_ids']]

        model = MockModel()
        evaluator = ModelEvaluator(
            model=model,
            tokenizer=MockTokenizer(),
            model_name='flan-t5',
            batch_size=2,
            prompt_format=0,
            device='cpu'
        )
        golden, predicted, decodeds = evaluator.evaluate(DummyDataset(), mode='prompting')
        self.assertIn('input_ids', model.received_kwargs)
        self.assertIn('attention_mask', model.received_kwargs)
        self.assertEqual(predicted, [1, 1])

    def test_seq2seq_parsing_failure_returns_negative_one(self):
        tokenizer = DummyTokenizer()
        model = DummySeq2SeqModel(canned_outputs=['completely ambiguous text', 'positive'])
        dataset = DummyDataset()

        evaluator = ModelEvaluator(
            model=model,
            tokenizer=tokenizer,
            model_name='flan-t5',
            batch_size=2,
            prompt_format=0,
            device='cpu'
        )

        golden, predicted, decodeds = evaluator.evaluate(dataset, mode='prompting')

        self.assertEqual(golden, [1, 0])
        self.assertEqual(predicted, [-1, 1])
        self.assertEqual(decodeds, ['completely ambiguous text', 'positive'])


class TestModelEvaluatorCausalLM(unittest.TestCase):

    def test_causallm_prompting_with_token_slicing(self):
        tokenizer = DummyCausalTokenizer()
        model = DummyCausalModel(generated_tokens_per_call=[[20], [21]])
        dataset = DummyDataset()

        evaluator = ModelEvaluator(
            model=model,
            tokenizer=tokenizer,
            model_name='llama3',
            batch_size=1,
            prompt_format=0,
            device='cpu'
        )

        golden, predicted, decodeds = evaluator.evaluate(dataset, mode='prompting')

        self.assertEqual(golden, [1, 0])
        self.assertEqual(predicted, [1, 0])
        self.assertEqual(decodeds, ['positive', 'negative'])

    def test_causallm_icl_evaluation(self):
        tokenizer = DummyCausalTokenizer()
        model = DummyCausalModel(generated_tokens_per_call=[[20], [21]])
        dataset = DummyDataset()

        evaluator = ModelEvaluator(
            model=model,
            tokenizer=tokenizer,
            model_name='mistral',
            batch_size=1,
            prompt_format=0,
            device='cpu'
        )

        golden, predicted, decodeds = evaluator.evaluate(dataset, mode='icl')

        self.assertEqual(golden, [1, 0])
        self.assertEqual(predicted, [1, 0])
        self.assertEqual(decodeds, ['positive', 'negative'])

    def test_causallm_parsing_failure(self):
        tokenizer = DummyCausalTokenizer()
        model = DummyCausalModel(generated_tokens_per_call=[[22]])  # 'ambiguous response'
        dataset = DummyDataset(texts=['uncertain sample'], targets=[0])

        evaluator = ModelEvaluator(
            model=model,
            tokenizer=tokenizer,
            model_name='zephyr',
            batch_size=1,
            prompt_format=0,
            device='cpu'
        )

        golden, predicted, decodeds = evaluator.evaluate(dataset, mode='prompting')

        self.assertEqual(golden, [0])
        self.assertEqual(predicted, [-1])
        self.assertEqual(decodeds, ['ambiguous response'])


class MockChatCompletionMessage:
    def __init__(self, content):
        self.content = content


class MockChatChoice:
    def __init__(self, content):
        self.message = MockChatCompletionMessage(content)


class MockChatResponse:
    def __init__(self, content):
        self.choices = [MockChatChoice(content)]


class MockOpenAIClient:
    def __init__(self, responses, fail_first_n=0):
        self.responses = list(responses)
        self.fail_first_n = fail_first_n
        self.calls = 0
        self.chat = self
        self.completions = self

    def create(self, **kwargs):
        self.calls += 1
        if self.fail_first_n > 0:
            self.fail_first_n -= 1
            raise RuntimeError("Temporary API network glitch")
        resp = self.responses.pop(0)
        return MockChatResponse(resp)


class TestModelEvaluatorAPI(unittest.TestCase):

    def test_api_adapter_prompting_evaluation(self):
        client = MockOpenAIClient(responses=['positive', 'negative'])
        dataset = DummyDataset()

        evaluator = ModelEvaluator(
            model=client,
            tokenizer=None,
            model_name='chatgpt',
            batch_size=1,
            prompt_format=0
        )

        golden, predicted, decodeds = evaluator.evaluate(dataset, mode='prompting')

        self.assertEqual(golden, [1, 0])
        self.assertEqual(predicted, [1, 0])
        self.assertEqual(decodeds, ['positive', 'negative'])

    def test_api_adapter_retry_on_failure(self):
        client = MockOpenAIClient(responses=['positive'], fail_first_n=1)
        dataset = DummyDataset(texts=['retry me'], targets=[1])

        evaluator = ModelEvaluator(
            model=client,
            tokenizer=None,
            model_name='chatgpt',
            batch_size=1,
            prompt_format=0
        )

        golden, predicted, decodeds = evaluator.evaluate(dataset, mode='prompting')

        self.assertEqual(golden, [1])
        self.assertEqual(predicted, [1])
        self.assertEqual(client.calls, 2)

    def test_api_adapter_partial_save_and_resumption(self):
        import tempfile
        import shutil
        import os

        tmp_dir = tempfile.mkdtemp()
        try:
            client = MockOpenAIClient(responses=['positive', 'negative'])
            dataset = DummyDataset()

            evaluator = ModelEvaluator(
                model=client,
                tokenizer=None,
                model_name='chatgpt',
                batch_size=1,
                prompt_format=0
            )

            # First run: processes both samples and saves partial files
            golden, predicted, decodeds = evaluator.evaluate(dataset, mode='prompting', partial_save_path=tmp_dir)
            self.assertEqual(predicted, [1, 0])
            self.assertEqual(client.calls, 2)

            # Second run with same partial_save_path: should read from disk without calling API
            client2 = MockOpenAIClient(responses=[])
            evaluator2 = ModelEvaluator(
                model=client2,
                tokenizer=None,
                model_name='chatgpt',
                batch_size=1,
                prompt_format=0
            )
            golden2, predicted2, decodeds2 = evaluator2.evaluate(dataset, mode='prompting', partial_save_path=tmp_dir)
            self.assertEqual(predicted2, [1, 0])
            self.assertEqual(client2.calls, 0)
        finally:
            shutil.rmtree(tmp_dir)

    def test_evaluation_result_captures_inputs_and_prompts(self):
        """Verifies that evaluate() returns an EvaluationResult with inputs and prompts while supporting 3-tuple unpacking."""
        dataset = DummyDataset()
        evaluator = ModelEvaluator(
            model=DummyCausalModel(generated_tokens_per_call=[[20], [21]]),
            tokenizer=DummyCausalTokenizer(),
            model_name='llama3',
            batch_size=1,
            prompt_format=0,
            device='cpu'
        )

        res = evaluator.evaluate(dataset, mode='prompting')
        # 1. Backward-compatible 3-tuple unpacking
        golden, predicted, decodeds = res
        self.assertEqual(len(res), 3)
        self.assertEqual(golden, [1, 0])
        self.assertEqual(predicted, [1, 0])
        self.assertEqual(decodeds, ['positive', 'negative'])

        # 2. Rich qualitative attributes
        self.assertTrue(hasattr(res, 'prompts'))
        self.assertTrue(hasattr(res, 'inputs'))
        self.assertEqual(len(res.inputs), 2)
        self.assertEqual(len(res.prompts), 2)
        self.assertEqual(res.inputs[0], "I love this")
        self.assertEqual(res.inputs[1], "I hate this")
        self.assertIn("I love this", res.prompts[0])

    def test_modern_models_adapter_dispatch(self):
        """Verifies ModelEvaluator resolves qwen, phi, and gemma correctly into CausalLM adapter."""
        for m_name in ('qwen_4b', 'qwen_9b', 'phi_mini', 'gemma_12b'):
            evaluator = ModelEvaluator(
                model=DummyCausalModel(generated_tokens_per_call=[[20]]),
                tokenizer=DummyCausalTokenizer(),
                model_name=m_name,
                batch_size=1
            )
            expected_family = 'qwen' if 'qwen' in m_name else ('phi' if 'phi' in m_name else 'gemma')
            self.assertEqual(evaluator.adapter.model_family, expected_family)

    def test_deberta_base_forward_shape(self):
        """Verifies DeBERTaBase forward pass extracts [CLS] at index 0 and outputs logits of shape (batch, n_classes)."""
        import torch
        from transfer_learning.models import DeBERTaBase

        class DummyBackbone(torch.nn.Module):
            def __init__(self, hidden_size):
                super().__init__()
                self.config = type('Config', (), {'hidden_size': hidden_size})()
            def forward(self, input_ids, attention_mask=None, token_type_ids=None):
                return [torch.randn(input_ids.shape[0], input_ids.shape[1], self.config.hidden_size)]

        # Instantiate DeBERTaBase and mock backbone for deterministic fast execution
        model = DeBERTaBase(n_classes=3, init_seed=42)
        hidden_size = model.deberta.config.hidden_size
        model.deberta = DummyBackbone(hidden_size)

        dummy_ids = torch.ones(2, 4, dtype=torch.long)
        dummy_mask = torch.ones(2, 4, dtype=torch.long)
        dummy_types = torch.zeros(2, 4, dtype=torch.long)

        logits = model(dummy_ids, dummy_mask, dummy_types)
        self.assertEqual(logits.shape, (2, 3))


class _Encoded(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def to(self, device):
        return self


class VocabTokenizer:
    """Fixed vocabulary where id 0 ends the answer. With strip_start, decode
    drops the answer's leading space as SentencePiece does."""
    VOCAB = ['<end>', 'maybe', 'positive', 'negative', 'pos', 'itive', ' positive',
             'Book', ' Restaurant', 'Rate', ' Book', 'x']

    def __init__(self, strip_start=False):
        self.strip_start = strip_start
        self.eos_token_id = self.pad_token_id = 0

    def __len__(self):
        return len(self.VOCAB)

    def decode(self, ids, skip_special_tokens=False):
        text = "".join(self.VOCAB[i] for i in ids if not (skip_special_tokens and i == 0))
        return text.lstrip() if self.strip_start else text

    def batch_decode(self, sequences, skip_special_tokens=False):
        return [self.decode(list(s), skip_special_tokens) for s in sequences]

    def __call__(self, text, return_tensors=None, add_special_tokens=True, **kwargs):
        x = self.VOCAB.index('x')
        if isinstance(text, list):
            return _Encoded(input_ids=[[x] for _ in text], attention_mask=[[1] for _ in text])
        return _Encoded(input_ids=[[x]] if return_tensors else [x])


def _ids(*tokens):
    return [VocabTokenizer.VOCAB.index(t) for t in tokens]


class ConstrainedGreedyModel:
    """Emits its best-ranked token among those generate()'s constraint allows;
    a seq2seq decoder starts from token 0."""

    def __init__(self, ranking, seq2seq=False):
        self.ranking, self.seq2seq = ranking, seq2seq

    def generate(self, input_ids=None, max_new_tokens=10, prefix_allowed_tokens_fn=None, **kwargs):
        import torch
        outputs = []
        for prompt in input_ids:
            seq = [0] if self.seq2seq else list(prompt)
            for _ in range(max_new_tokens):
                allowed = (prefix_allowed_tokens_fn(0, torch.tensor(seq))
                           if prefix_allowed_tokens_fn else self.ranking)
                token = next(t for t in self.ranking if t in allowed)
                seq.append(token)
                if token == 0:
                    break
            outputs.append(seq)
        return outputs


class TestChoiceConstraint(unittest.TestCase):

    def test_answer_stays_a_class_prefix_and_ends_only_on_a_whole_class(self):
        c = _ChoiceConstraint(VocabTokenizer(), ['negative', 'positive'], end_ids=[0])
        self.assertEqual(c.allowed([]), sorted(_ids('positive', 'negative', 'pos')))
        self.assertEqual(c.allowed(_ids('pos')), _ids('itive'))
        self.assertEqual(c.allowed(_ids('pos', 'itive')), [0])

    def test_first_token_space_follows_decode(self):
        bpe = _ChoiceConstraint(VocabTokenizer(), ['positive'], end_ids=[0])
        spm = _ChoiceConstraint(VocabTokenizer(strip_start=True), ['positive'], end_ids=[0])
        self.assertNotIn(_ids(' positive')[0], bpe.allowed([]))
        self.assertIn(_ids(' positive')[0], spm.allowed([]))

    def test_class_inside_a_longer_class_may_end_or_continue(self):
        c = _ChoiceConstraint(VocabTokenizer(), ['Book', 'Book Restaurant', 'Rate Book'], end_ids=[0])
        self.assertEqual(c.allowed(_ids('Book')), sorted([0] + _ids(' Restaurant')))
        self.assertEqual(c.allowed(_ids('Rate')), _ids(' Book'))

    def test_prefix_fn_skips_the_prompt(self):
        import torch
        c = _ChoiceConstraint(VocabTokenizer(), ['negative', 'positive'], end_ids=[0])
        fn = c.prefix_allowed_tokens_fn(prompt_len=2)
        self.assertEqual(fn(0, torch.tensor(_ids('x', 'x', 'pos'))), _ids('itive'))


class TestGuidedDecodingHF(unittest.TestCase):
    # The model would rather say "maybe" or " positive" than a bare class name.
    RANKING = _ids('maybe', ' positive', 'pos', 'itive', '<end>')

    def _evaluate(self, model_name, tokenizer, decoding, seq2seq=False):
        evaluator = ModelEvaluator(model=ConstrainedGreedyModel(self.RANKING, seq2seq=seq2seq),
                                   tokenizer=tokenizer, model_name=model_name, batch_size=2, device='cpu')
        return evaluator.evaluate(DummyDataset(), mode='prompting', decoding=decoding)

    def test_causal_guided_answers_are_class_names(self):
        _, free, _ = self._evaluate('qwen', VocabTokenizer(), 'free')
        self.assertEqual(free, [-1, -1])
        _, guided, decoded = self._evaluate('qwen', VocabTokenizer(), 'guided')
        self.assertEqual(guided, [1, 1])
        self.assertEqual(decoded, ['positive', 'positive'])

    def test_seq2seq_guided_answers_are_class_names(self):
        tokenizer = VocabTokenizer(strip_start=True)
        _, free, _ = self._evaluate('flan-t5', tokenizer, 'free', seq2seq=True)
        self.assertEqual(free, [-1, -1])
        _, guided, decoded = self._evaluate('flan-t5', tokenizer, 'guided', seq2seq=True)
        self.assertEqual(guided, [1, 1])
        self.assertEqual(decoded, ['positive', 'positive'])


class CharTensorTokenizer:
    """One id per ASCII character, returned as tensors like a HF tokenizer;
    id 0 pads and ends the answer."""
    pad_token_id = eos_token_id = 0

    def __call__(self, text, return_tensors=None, **kwargs):
        import torch
        ids = torch.tensor([[ord(c) % 128 for c in text]])
        return SimpleNamespace(input_ids=ids, attention_mask=torch.ones_like(ids))

    def convert_tokens_to_ids(self, token):
        return None

    def decode(self, ids, skip_special_tokens=False):
        return "".join(chr(int(i)) for i in ids if not (skip_special_tokens and int(i) == 0))


class FirstTokenModel:
    """Answers each row with its first prompt token and records the batch
    shapes; batches above max_rows run out of memory."""

    def __init__(self, max_rows=None):
        self.max_rows = max_rows
        self.shapes = []

    def generate(self, input_ids=None, attention_mask=None, **kwargs):
        import torch
        self.shapes.append(tuple(input_ids.shape))
        if self.max_rows and len(input_ids) > self.max_rows:
            raise torch.cuda.OutOfMemoryError("CUDA out of memory")
        first = torch.stack([row[mask.bool()][0] for row, mask in zip(input_ids, attention_mask)])
        return torch.cat([input_ids, first[:, None]], dim=1)


class TestCausalBatching(unittest.TestCase):
    """HF evaluation of causal models (instruction tuning's tuned adapter)
    generates several prompts per call, left-padded."""

    GEN = {'max_new_tokens': 1, 'do_sample': False, 'pad_token_id': 0}
    PROMPTS = ["a", "bbbb", "cc", "dddddd"]

    def _adapter(self, model, budget=None):
        adapter = _CausalLMAdapter(model, CharTensorTokenizer(), 'qwen', batch_size=4, device='cpu')
        if budget:
            adapter.MAX_BATCH_TOKENS = budget
        return adapter

    def test_long_prompts_split_under_the_token_budget(self):
        model = FirstTokenModel()
        answers = self._adapter(model, budget=8)._generate(self.PROMPTS, self.GEN, None)
        self.assertEqual(answers, ["a", "b", "c", "d"])
        self.assertEqual(model.shapes, [(2, 4), (1, 2), (1, 6)])

    def test_out_of_memory_halves_the_batch(self):
        model = FirstTokenModel(max_rows=1)
        answers = self._adapter(model, budget=64)._generate(self.PROMPTS, self.GEN, None)
        self.assertEqual(answers, ["a", "b", "c", "d"])
        self.assertEqual([rows for rows, _ in model.shapes], [4, 2, 1, 1, 2, 1, 1])

    def _assert_batches_match_one_at_a_time(self, config, device='cpu'):
        import torch
        from transformers import AutoModelForCausalLM
        torch.manual_seed(0)
        model = AutoModelForCausalLM.from_config(config).to(device).eval()
        texts = ["Great.", "A long and winding review of a film that never quite finds its footing.",
                 "Dull", "Funny, warm and smart: the best comedy of the year, by a wide margin."]
        dataset = DummyDataset(texts=texts, targets=[1, 0, 0, 1])
        decoded = {}
        for batch_size in (1, 4):
            evaluator = ModelEvaluator(model=model, tokenizer=CharTensorTokenizer(), model_name='qwen',
                                       batch_size=batch_size, device=device)
            decoded[batch_size] = evaluator.evaluate(dataset, mode='icl', decoding='free')[2]
        self.assertEqual(decoded[4], decoded[1])

    def test_llama_answers_match_one_at_a_time(self):
        from transformers import LlamaConfig
        self._assert_batches_match_one_at_a_time(LlamaConfig(
            vocab_size=128, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
            num_attention_heads=2, num_key_value_heads=1))

    def test_qwen3_5_gated_deltanet_ignores_the_left_padding(self):
        try:
            from transformers import Qwen3_5TextConfig
        except ImportError:
            self.skipTest("transformers without Qwen 3.5")
        import importlib.util
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if device == 'cpu' and importlib.util.find_spec('fla') is not None:
            self.skipTest("flash-linear-attention kernels need a GPU")
        self._assert_batches_match_one_at_a_time(Qwen3_5TextConfig(
            vocab_size=128, hidden_size=32, intermediate_size=64, num_hidden_layers=4,
            num_attention_heads=2, num_key_value_heads=1, head_dim=16, linear_num_key_heads=2,
            linear_num_value_heads=2, linear_key_head_dim=16, linear_value_head_dim=16), device)


if __name__ == '__main__':
    unittest.main()
