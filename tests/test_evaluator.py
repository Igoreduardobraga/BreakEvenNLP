import unittest
from evaluator import ModelEvaluator


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


class TestModelEvaluatorSeq2Seq(unittest.TestCase):

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
        # Mocking generation: sample 0 outputs [20] ('positive'), sample 1 outputs [21] ('negative')
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
            prompt_format=0
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


if __name__ == '__main__':
    unittest.main()
