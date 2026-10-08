"""Unit tests for the vLLM engine path (offline, stubbed — no GPU/weights needed)."""
import sys
import types
import unittest


class FakeOutput:
    def __init__(self, text):
        self.text = text


class FakeRequestOutput:
    def __init__(self, prompt, text):
        self.prompt = prompt
        self.outputs = [FakeOutput(text)]


class FakeLLM:
    last_instance = None

    def __init__(self, model, quantization=None, dtype=None, seed=0, **budget):
        self.model = model
        self.quantization = quantization
        self.dtype = dtype
        self.seed = seed
        self.budget = budget
        self.canned = ["positive", "negative"]
        self.calls = []
        FakeLLM.last_instance = self

    def get_tokenizer(self):
        raise RuntimeError("no tokenizer in stub")

    def generate(self, prompts, params):
        self.calls.append((list(prompts), params))
        return [FakeRequestOutput(p, self.canned[i % len(self.canned)])
                for i, p in enumerate(prompts)]


class FakeSamplingParams:
    def __init__(self, temperature=1.0, max_tokens=16, seed=None, **kwargs):
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.seed = seed
        self.extra = kwargs


def install_stub():
    mod = types.ModuleType("vllm")
    mod.LLM = FakeLLM
    mod.SamplingParams = FakeSamplingParams
    sys.modules["vllm"] = mod


class DummyDataset:
    def __init__(self):
        self.classes = ["negative", "positive"]
        self.test_text = ["I love this", "I hate this"]
        self.test_targets = [1, 0]
        self.instructions = {
            "instruction": "Analyze. Options: 'negative', 'positive'.",
            "sentence_start": "Text", "answer_start": "Answer",
            "task_type": "sentiment analysis",
        }
        self.context_samples = [("loved it", "positive")]

    def batch_data_for_evaluation(self, batch_size=64):
        for i in range(0, len(self.test_text), batch_size):
            yield self.test_text[i:i + batch_size], self.test_targets[i:i + batch_size]


class TestVLLMAdapter(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        install_stub()

    def test_dispatch_resolves_vllm_adapter(self):
        from evaluator import ModelEvaluator, _VLLMAdapter
        ev = ModelEvaluator(model="mistralai/Mistral-7B", tokenizer=None,
                            model_name="mistral", engine="vllm", seed=7)
        self.assertIsInstance(ev.adapter, _VLLMAdapter)
        self.assertEqual(ev.adapter.seed, 7)
        self.assertEqual(ev.quant_format, "bitsandbytes")

    def test_free_decoding_greedy_no_guided_choice(self):
        from evaluator import ModelEvaluator
        ev = ModelEvaluator(model="x", tokenizer=None, model_name="llama3", engine="vllm")
        golden, predicted, decodeds = ev.evaluate(DummyDataset(), mode="prompting")
        self.assertEqual(golden, [1, 0])
        self.assertEqual(predicted, [1, 0])
        prompts, params = FakeLLM.last_instance.calls[-1]
        self.assertEqual(params.temperature, 0.0)
        self.assertEqual(params.max_tokens, 10)
        self.assertNotIn("guided_choice", params.extra)

    def test_guided_decoding_passes_choices(self):
        from evaluator import ModelEvaluator
        ev = ModelEvaluator(model="x", tokenizer=None, model_name="qwen_4b", engine="vllm")
        golden, predicted, decodeds = ev.evaluate(DummyDataset(), mode="icl", decoding="guided")
        self.assertEqual(predicted, [1, 0])
        _, params = FakeLLM.last_instance.calls[-1]
        self.assertEqual(params.extra["guided_choice"], ["negative", "positive"])

    def test_guided_on_hf_raises(self):
        from evaluator import ModelEvaluator
        e = ModelEvaluator.__new__(ModelEvaluator)
        e.adapter = object()
        with self.assertRaises(ValueError):
            ModelEvaluator.evaluate(e, DummyDataset(), decoding="guided")

    def test_vllm_rejects_non_causal(self):
        from evaluator import ModelEvaluator
        with self.assertRaises(NotImplementedError):
            ModelEvaluator(model="x", tokenizer=None, model_name="flan-t5", engine="vllm")
        with self.assertRaises(NotImplementedError):
            ModelEvaluator(model="x", tokenizer=None, model_name="chatgpt", engine="vllm")

    def test_invalid_engine_and_decoding(self):
        from evaluator import ModelEvaluator
        with self.assertRaises(ValueError):
            ModelEvaluator(model="x", tokenizer=None, model_name="mistral", engine="trt")
        ev = ModelEvaluator(model="x", tokenizer=None, model_name="mistral", engine="vllm")
        with self.assertRaises(ValueError):
            ev.evaluate(DummyDataset(), decoding="beam")

    def test_budgets_cover_all_families(self):
        from evaluator import _VLLMAdapter
        for fam in ("llama2", "llama3", "mistral", "zephyr", "qwen", "phi", "gemma"):
            b = _VLLMAdapter.BUDGETS[fam]
            self.assertLessEqual(b["gpu_memory_utilization"], 0.90)
            self.assertGreaterEqual(b["max_model_len"], 1024)

    def test_missing_vllm_raises_helpful_error(self):
        import evaluator as evmod
        saved = sys.modules.pop("vllm", None)
        try:
            with self.assertRaises(ImportError) as ctx:
                evmod._VLLMAdapter(model_id="x", model_family="mistral")
            self.assertIn("vllm-bnb-plugin", str(ctx.exception))
        finally:
            if saved is not None:
                sys.modules["vllm"] = saved


if __name__ == "__main__":
    unittest.main()
