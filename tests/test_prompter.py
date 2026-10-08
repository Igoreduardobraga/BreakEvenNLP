# tests/test_prompter.py
"""
Unit tests for the deep PromptFormatter module.
Tests core task catalog, zero-shot prompting across all model families,
and chat template formatting without external network or GPU dependencies.
"""

import unittest
from prompter import PromptFormatter


class TestPromptFormatterCatalog(unittest.TestCase):
    """Tests that the task catalog correctly provides dataset-specific metadata."""

    def setUp(self):
        self.prompter = PromptFormatter(model_name="flan-t5", prompt_format=0)

    def test_sst2_task_info(self):
        info = self.prompter.get_task_info("sst2")
        self.assertEqual(info["task_type"], "sentiment analysis")
        self.assertEqual(info["sentence_start"], "Text")
        self.assertEqual(info["answer_start"], "Answer")
        self.assertIn("'negative', 'positive'", info["instruction"])

    def test_sst2_custom_classes(self):
        info = self.prompter.get_task_info("sst2", classes=["terrible", "great"])
        self.assertIn("'terrible', 'great'", info["instruction"])

    def test_cola_task_info(self):
        info = self.prompter.get_task_info("cola")
        self.assertEqual(info["task_type"], "grammatical acceptability")
        self.assertIn("grammatically acceptable", info["instruction"])

    def test_mrpc_task_info(self):
        info = self.prompter.get_task_info("mrpc")
        self.assertEqual(info["task_type"], "semantic equivalence")
        self.assertEqual(info["sentence_start"], "Sentences")

    def test_boolq_task_info(self):
        info = self.prompter.get_task_info("boolq")
        self.assertEqual(info["task_type"], "question answering")
        self.assertEqual(info["sentence_start"], "")
        self.assertIn("Read the passage and answer", info["instruction"])

    def test_trec_task_info(self):
        info = self.prompter.get_task_info("trec")
        self.assertEqual(info["task_type"], "question classification")

    def test_ag_news_task_info(self):
        info = self.prompter.get_task_info("ag_news")
        self.assertEqual(info["task_type"], "news classification")

    def test_snips_task_info(self):
        info = self.prompter.get_task_info("snips")
        self.assertEqual(info["task_type"], "intent detection")

    def test_db_pedia_task_info(self):
        info = self.prompter.get_task_info("db_pedia")
        self.assertEqual(info["task_type"], "topic classification")

    def test_unknown_dataset_fallback(self):
        info = self.prompter.get_task_info("unknown_corpus", classes=["alpha", "beta"])
        self.assertEqual(info["task_type"], "classification")
        self.assertIn("'alpha', 'beta'", info["instruction"])


class TestPromptFormatterZeroShot(unittest.TestCase):
    """Tests zero-shot prompt generation across all supported model families."""

    def test_seq2seq_flan_t5_format_0(self):
        prompter = PromptFormatter(model_name="flan-t5", prompt_format=0)
        prompt = prompter.format("Great movie!", dataset_name="sst2")
        self.assertIn("Analyze the sentiment of the text.", prompt)
        self.assertIn("Text: Great movie!", prompt)
        self.assertTrue(prompt.endswith("Answer: "))

    def test_seq2seq_flan_t5_format_1(self):
        prompter = PromptFormatter(model_name="flan-t5", prompt_format=1)
        prompt = prompter.format("Great movie!", dataset_name="sst2")
        self.assertEqual(prompt, "Great movie!\nAnalyze the sentiment of the text. Options: 'negative', 'positive'. ")

    def test_llama2_format_0(self):
        prompter = PromptFormatter(model_name="llama2", prompt_format=0)
        prompt = prompter.format("Great movie!", dataset_name="sst2")
        self.assertIn("<s>[INST] <<SYS>>", prompt)
        self.assertIn("sentiment analysis", prompt)
        self.assertIn("<</SYS>>", prompt)
        self.assertIn("Input: Great movie!\nAnswer: [/INST]", prompt)

    def test_mistral_format_0_native(self):
        prompter = PromptFormatter(model_name="mistral", prompt_format=0)
        prompt = prompter.format("Great movie!", dataset_name="sst2")
        self.assertIn("<s>[INST]", prompt)
        self.assertIn("Do not explain. Answer only with the class name.", prompt)
        self.assertIn("Text: Great movie!", prompt)
        self.assertIn("[/INST]", prompt)

    def test_zephyr_format_0_native(self):
        prompter = PromptFormatter(model_name="zephyr", prompt_format=0)
        prompt = prompter.format("Great movie!", dataset_name="sst2")
        self.assertIn("<|system|>", prompt)
        self.assertIn("Answer with the class name only.", prompt)
        self.assertIn("<|user|>\nText: Great movie!</s>", prompt)
        self.assertIn("<|assistant|>\n", prompt)

    def test_llama3_format_0_native(self):
        prompter = PromptFormatter(model_name="llama3", prompt_format=0)
        prompt = prompter.format("Great movie!", dataset_name="sst2")
        self.assertIn("<|start_header_id|>system<|end_header_id|>", prompt)
        self.assertIn("You are an expert classifier.", prompt)
        self.assertIn("<|start_header_id|>user<|end_header_id|>", prompt)
        self.assertIn("Text: Great movie!", prompt)
        self.assertIn("<|start_header_id|>assistant<|end_header_id|>", prompt)

    def test_chatgpt_format_0(self):
        prompter = PromptFormatter(model_name="chatgpt", prompt_format=0)
        messages = prompter.format("Great movie!", dataset_name="sst2")
        self.assertIsInstance(messages, list)
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[1]["role"], "user")
        self.assertIn("Text: Great movie!\nAnswer: ", messages[1]["content"])


class TestPromptFormatterFewShotICL(unittest.TestCase):
    """Tests few-shot (ICL) demonstration injection across model families."""

    def setUp(self):
        self.shots = [("I loved it", "positive"), ("I hated it", "negative")]

    def test_seq2seq_icl(self):
        prompter = PromptFormatter(model_name="flan-t5", prompt_format=0)
        prompt = prompter.format("Great movie!", dataset_name="sst2", shots=self.shots)
        self.assertIn("Text: I loved it\nAnswer: positive\n", prompt)
        self.assertIn("Text: I hated it\nAnswer: negative\n", prompt)
        self.assertTrue(prompt.endswith("Text: Great movie!\nAnswer: "))

    def test_llama2_icl(self):
        prompter = PromptFormatter(model_name="llama2", prompt_format=0)
        prompt = prompter.format("Great movie!", dataset_name="sst2", shots=self.shots)
        self.assertIn("<s>[INST] I loved it [/INST] positive </s>", prompt)
        self.assertIn("<s>[INST] I hated it [/INST] negative </s>", prompt)
        self.assertIn("Input: Great movie!\nAnswer: [/INST]", prompt)

    def test_mistral_icl_native(self):
        prompter = PromptFormatter(model_name="mistral", prompt_format=0)
        prompt = prompter.format("Great movie!", dataset_name="sst2", shots=self.shots)
        self.assertIn("[INST] Text: I loved it [/INST] positive</s>", prompt)
        self.assertIn("[INST] Text: Great movie! [/INST]", prompt)

    def test_zephyr_icl_native(self):
        prompter = PromptFormatter(model_name="zephyr", prompt_format=0)
        prompt = prompter.format("Great movie!", dataset_name="sst2", shots=self.shots)
        self.assertIn("<|user|>\nText: I loved it</s>\n<|assistant|>\npositive</s>", prompt)
        self.assertIn("<|user|>\nText: Great movie!</s>\n<|assistant|>\n", prompt)

    def test_llama3_icl_native(self):
        prompter = PromptFormatter(model_name="llama3", prompt_format=0)
        prompt = prompter.format("Great movie!", dataset_name="sst2", shots=self.shots)
        self.assertIn("<|start_header_id|>user<|end_header_id|>\n\nText: I loved it<|eot_id|>", prompt)
        self.assertIn("<|start_header_id|>assistant<|end_header_id|>\n\npositive<|eot_id|>", prompt)
        self.assertIn("Text: Great movie!\nBased on the text above, determine the sentiment analysis. Answer:<|eot_id|>", prompt)

    def test_chatgpt_icl(self):
        prompter = PromptFormatter(model_name="chatgpt", prompt_format=0)
        messages = prompter.format("Great movie!", dataset_name="sst2", shots=self.shots)
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[1]["role"], "user")
        self.assertIn("Text: I loved it\nAnswer: positive\n", messages[1]["content"])
        self.assertIn("Text: Great movie!\nAnswer: ", messages[1]["content"])


class TestPromptFormatterBatch(unittest.TestCase):
    """Tests batch formatting capability."""

    def test_format_batch(self):
        prompter = PromptFormatter(model_name="flan-t5", prompt_format=0)
        texts = ["Text one", "Text two"]
        batch = prompter.format_batch(texts, dataset_name="sst2")
        self.assertEqual(len(batch), 2)
        self.assertIn("Text: Text one", batch[0])
        self.assertIn("Text: Text two", batch[1])


class TestPromptFormatterInstructionTuning(unittest.TestCase):
    """Tests instruction tuning SFT formatting and response templates."""

    def setUp(self):
        self.samples = [("Superb film", "positive"), ("Awful waste of time", "negative")]

    def test_flan_t5_instruction_tuning(self):
        prompter = PromptFormatter(model_name="flan-t5", prompt_format=0)
        prompts = prompter.format_instruction_tuning(self.samples, dataset_name="sst2")
        self.assertEqual(len(prompts), 2)
        self.assertIn("Analyze the sentiment of the text.", prompts[0])
        self.assertIn("Text: Superb film\nAnswer: positive", prompts[0])
        self.assertIn("Text: Awful waste of time\nAnswer: negative", prompts[1])

    def test_mistral_instruction_tuning(self):
        prompter = PromptFormatter(model_name="mistral", prompt_format=0)
        prompts = prompter.format_instruction_tuning(self.samples, dataset_name="sst2")
        self.assertEqual(len(prompts), 2)
        self.assertTrue(prompts[0].startswith("<s>[INST]"))
        self.assertTrue(prompts[0].endswith("[/INST] positive</s>"))
        self.assertIn("Text: Superb film", prompts[0])

    def test_zephyr_instruction_tuning(self):
        prompter = PromptFormatter(model_name="zephyr", prompt_format=0)
        prompts = prompter.format_instruction_tuning(self.samples, dataset_name="sst2")
        self.assertEqual(len(prompts), 2)
        self.assertIn("<|system|>\nAnalyze the sentiment", prompts[0])
        self.assertIn("<|user|>\nText: Superb film</s>", prompts[0])
        self.assertIn("<|assistant|>\npositive</s>", prompts[0])

    def test_llama3_instruction_tuning(self):
        prompter = PromptFormatter(model_name="llama3", prompt_format=0)
        prompts = prompter.format_instruction_tuning(self.samples, dataset_name="sst2")
        self.assertEqual(len(prompts), 2)
        self.assertIn("<|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant.", prompts[0])
        self.assertIn("<|start_header_id|>user<|end_header_id|>\n\nAnalyze the sentiment", prompts[0])
        self.assertIn("<|start_header_id|>assistant<|end_header_id|>\n\npositive<|eot_id|>", prompts[0])

    def test_response_templates(self):
        self.assertEqual(PromptFormatter("flan-t5").get_response_template(), "Answer:")
        self.assertEqual(PromptFormatter("mistral").get_response_template(), "[/INST]")
        self.assertEqual(PromptFormatter("zephyr").get_response_template(), "<|assistant|>")
        self.assertEqual(PromptFormatter("llama3").get_response_template(), "<|start_header_id|>assistant<|end_header_id|>\n\n")


class TestDataKeywordsDelegation(unittest.TestCase):
    """Verifies data.py delegates prepare_dataset_keywords to PromptFormatter."""

    def test_text_dataset_keywords_delegation(self):
        import sys
        from unittest.mock import MagicMock

        class BaseDataset:
            pass

        # Mock third-party dependencies if not installed locally
        mocked_modules = [
            'torch', 'torch.utils', 'torch.utils.data', 'datasets',
            'transformers', 'pandas', 'numpy', 'sklearn',
            'sklearn.metrics', 'sklearn.metrics.pairwise', 'sklearn.model_selection'
        ]
        originals = {}
        for mod in mocked_modules:
            if mod not in sys.modules:
                originals[mod] = None
                mock = MagicMock()
                if mod == 'torch.utils.data':
                    mock.Dataset = BaseDataset
                sys.modules[mod] = mock

        if hasattr(sys.modules.get('torch'), 'utils'):
            sys.modules['torch'].utils.data.Dataset = BaseDataset

        try:
            from data import TextDataset

            class DummyTextDataset(TextDataset):
                def __init__(self, dataset_name, prompt_format=0):
                    self.dataset_name = dataset_name
                    self.prompt_format = prompt_format
                    self.classes = ['negative', 'positive']

            ds = DummyTextDataset("sst2", prompt_format=0)
            instruction, s_start, a_start, task_type = ds.prepare_dataset_keywords()
            self.assertIn("Analyze the sentiment", instruction)
            self.assertEqual(s_start, "Text")
            self.assertEqual(a_start, "Answer")
            self.assertEqual(task_type, "sentiment analysis")
        finally:
            for mod, orig in originals.items():
                if orig is None:
                    sys.modules.pop(mod, None)


class TestModernModelPrompting(unittest.TestCase):
    """Verifies PromptFormatter correctly supports modern models (Qwen, Phi, Gemma)."""

    def test_qwen_formatting(self):
        prompter = PromptFormatter(model_name="qwen_4b", prompt_format=0)
        self.assertEqual(prompter.model_family, "qwen")
        self.assertEqual(prompter.get_response_template(), "<|im_start|>assistant\n")

        # Zero-shot
        prompt = prompter.format("Great movie!", dataset_name="sst2")
        self.assertIn("<|im_start|>user", prompt)
        self.assertIn("Great movie!", prompt)
        self.assertIn("<|im_start|>assistant", prompt)

        # Few-shot ICL
        shots = [("Bad film", "negative")]
        icl_prompt = prompter.format("Loved it", dataset_name="sst2", shots=shots)
        self.assertIn("Bad film", icl_prompt)
        self.assertIn("Loved it", icl_prompt)

        # SFT instruction tuning
        sft_samples = [("Great movie!", "positive")]
        sft_prompts = prompter.format_instruction_tuning(sft_samples, dataset_name="sst2")
        self.assertEqual(len(sft_prompts), 1)
        self.assertIn("<|im_start|>assistant\npositive<|im_end|>", sft_prompts[0])

    def test_phi_formatting(self):
        prompter = PromptFormatter(model_name="phi_mini", prompt_format=0)
        self.assertEqual(prompter.model_family, "phi")
        self.assertEqual(prompter.get_response_template(), "<|assistant|>\n")

        # Zero-shot
        prompt = prompter.format("Awesome acting!", dataset_name="sst2")
        self.assertIn("<|user|>", prompt)
        self.assertIn("<|assistant|>", prompt)

        # SFT instruction tuning
        sft_samples = [("Awesome acting!", "positive")]
        sft_prompts = prompter.format_instruction_tuning(sft_samples, dataset_name="sst2")
        self.assertIn("<|assistant|>\npositive<|end|>", sft_prompts[0])

    def test_gemma_formatting(self):
        prompter = PromptFormatter(model_name="gemma_26b", prompt_format=0)
        self.assertEqual(prompter.model_family, "gemma")
        self.assertEqual(prompter.get_response_template(), "<start_of_turn>model\n")

        # Zero-shot
        prompt = prompter.format("Terrible plot.", dataset_name="sst2")
        self.assertIn("<start_of_turn>user", prompt)
        self.assertIn("<start_of_turn>model", prompt)

        # SFT instruction tuning
        sft_samples = [("Terrible plot.", "negative")]
        sft_prompts = prompter.format_instruction_tuning(sft_samples, dataset_name="sst2")
        self.assertIn("<start_of_turn>model\nnegative<end_of_turn>", sft_prompts[0])


if __name__ == "__main__":
    unittest.main()



