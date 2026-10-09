"""Parity gate HF vs vLLM (Q2/Q9): runs only on demand with real weights + GPU.

Run on the RTX 4090 machine with:
    VLLM_PARITY=1 .venv/bin/python -m pytest tests/test_vllm_parity.py -q

Protocol: 40 fixed sst2 test prompts (zero-shot + ICL), greedy on both engines,
criterion = parsed labels 100% identical; raw decoded strings reported with
whitespace tolerance. Any label divergence fails.
"""
import os
import unittest

PARITY_N = 100

PROMPTS = [
    "This movie was a complete triumph.",
    "A dull, lifeless waste of time.",
    "Brilliant acting, terrible script.",
    "I loved every minute of it.",
    "Painfully boring and predictable.",
]


@unittest.skipUnless(os.environ.get("VLLM_PARITY") == "1", "needs VLLM_PARITY=1 + GPU + weights")
class TestVLLMParity(unittest.TestCase):

    def test_parity_sst2_sample(self):
        from evaluator import ModelEvaluator
        from sample_pool import SamplePool

        texts = tuple(PROMPTS * 20)[:PARITY_N]
        targets = tuple([1, 0, 0, 1, 0] * 20)[:PARITY_N]
        pool = SamplePool(
            dataset_name="sst2",
            train_test_indices=(list(range(60)), list(range(60, 100))),
            num_labelled=0, full_test=True,
            texts=texts, targets=targets, classes=("negative", "positive"),
        )
        pool.context_samples = pool.get_shots(num_shots=2, choice_seed=0, order_seed=0)

        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        import torch
        model_id = "mistralai/Mistral-7B-Instruct-v0.1"
        tok = AutoTokenizer.from_pretrained(model_id)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        tok.padding_side = "left"
        # transformers>=5 removed the load_in_4bit kwarg: use quantization_config
        # (same NF4/double-quant/bf16 setup as main.py).
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        hf_model = AutoModelForCausalLM.from_pretrained(
            model_id, quantization_config=bnb_config, device_map="auto")
        hf_model.eval()

        hf = ModelEvaluator(model=hf_model, tokenizer=tok, model_name="mistral",
                            batch_size=1, engine="hf")
        vl = None  # built after the HF model is freed (see below)

        hf_results = {}
        for mode in ("prompting", "icl"):
            g1, p1, d1 = hf.evaluate(pool, mode=mode)
            hf_results[mode] = (list(g1), list(p1), list(d1))

        # Free the HF weights before vLLM init: co-residency OOMs (5GB HF +
        # 0.85*24GB reservation > 24GB). 0.70 is plenty for 2048 ctx + short prompts.
        import gc
        del hf, hf_model, tok
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

        vl = ModelEvaluator(model=model_id, tokenizer=None, model_name="mistral",
                            engine="vllm", seed=42,
                            vllm_kwargs={"budget_override": {"gpu_memory_utilization": 0.70}})

        for mode in ("prompting", "icl"):
            g1, p1, d1 = hf_results[mode]
            g2, p2, d2 = vl.evaluate(pool, mode=mode)
            self.assertEqual(g1, g2)
            self.assertEqual(p1, p2, f"label divergence in mode={mode}")
            norm = lambda s: " ".join(str(s).split())  # noqa: E731
            mism = sum(1 for a, b in zip(d1, d2) if norm(a) != norm(b))
            print(f"[{mode}] raw-decoded mismatches: {mism}/{len(d1)} "
                  f"(labels must match exactly)")


if __name__ == "__main__":
    unittest.main()
