# tests/test_prompter.py
"""
Unit tests for the deep PromptFormatter module.
Tests core task catalog, zero-shot prompting across all model families,
and chat template formatting without external network or GPU dependencies.
"""

import unittest
from unittest.mock import MagicMock
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
        self.assertIn("<|start_header_id|>system<|end_header_id|>\n\nAnalyze the sentiment of the text. "
                      "Options: 'negative', 'positive'. Do not explain. Answer only with the class name.<|eot_id|>",
                      prompt)
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
        self.assertIn("<|start_header_id|>user<|end_header_id|>\n\nText: Great movie!<|eot_id|>", prompt)

    def test_demonstrations_are_formatted_like_the_query(self):
        # A demonstration's user turn must be exactly the query's turn for the
        # same text, edge whitespace included (every GLUE SST-2 text ends in a
        # space). BoolQ has no sentence prefix, which used to differ too.
        tok = MagicMock()
        tok.apply_chat_template.side_effect = lambda messages, **kw: messages
        for family in ("qwen", "llama3", "phi", "gemma", "mistral", "zephyr"):
            for dataset in ("sst2", "boolq"):
                with self.subTest(family=family, dataset=dataset):
                    messages = PromptFormatter(model_name=family, tokenizer=tok).format(
                        "same text ", dataset_name=dataset, shots=[("same text ", "positive")])
                    user_turns = [m["content"] for m in messages if m["role"] == "user"]
                    self.assertEqual(user_turns[-2], user_turns[-1])

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
    """Instruction tuning trains on exactly the prompt that is evaluated."""

    FAMILIES = {
        "flan-t5": "",
        "mistral": "</s>",
        "zephyr": "</s>",
        "llama3": "<|eot_id|>",
        "llama2": "</s>",
        "qwen": "<|im_end|>",
        "phi": "<|end|>",
        "gemma": "<turn|>",
    }

    def setUp(self):
        self.samples = [("Superb film", "positive"), ("Awful waste of time", "negative")]

    def test_training_prompt_is_inference_prompt(self):
        for model, end in self.FAMILIES.items():
            with self.subTest(model=model):
                prompter = PromptFormatter(model_name=model, prompt_format=0)
                pairs = prompter.format_instruction_tuning(self.samples, dataset_name="sst2")
                self.assertEqual(len(pairs), 2)
                for (text, label), (prompt, completion) in zip(self.samples, pairs):
                    self.assertEqual(prompt, prompter.format(text, dataset_name="sst2"))
                    self.assertEqual(completion, f"{label}{end}")

    def test_training_prompt_does_not_contain_label(self):
        prompter = PromptFormatter(model_name="flan-t5", prompt_format=0)
        prompt, completion = prompter.format_training_pair("Superb film", "positive", dataset_name="sst2")
        self.assertEqual(completion, "positive")
        self.assertTrue(prompt.endswith("Text: Superb film\nAnswer: "))

    def test_chat_template_is_shared_with_inference(self):
        tok = MagicMock()
        tok.apply_chat_template.side_effect = lambda m, **kw: "|".join(x["content"] for x in m) + "<gen>"
        prompter = PromptFormatter(model_name="qwen", prompt_format=0, tokenizer=tok)
        prompt, completion = prompter.format_training_pair("Superb film", "positive", dataset_name="sst2")
        self.assertTrue(prompt.endswith("<gen>"))
        self.assertEqual(prompt, prompter.format("Superb film", dataset_name="sst2"))
        self.assertEqual(completion, "positive<|im_end|>")
        self.assertFalse(tok.apply_chat_template.call_args.kwargs["enable_thinking"])


class TestModernModelPrompting(unittest.TestCase):
    """Verifies PromptFormatter correctly supports modern models (Qwen, Phi, Gemma)."""

    def test_qwen_formatting(self):
        prompter = PromptFormatter(model_name="qwen_4b", prompt_format=0)
        self.assertEqual(prompter.model_family, "qwen")

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

    def test_phi_formatting(self):
        prompter = PromptFormatter(model_name="phi_mini", prompt_format=0)
        self.assertEqual(prompter.model_family, "phi")

        # Zero-shot
        prompt = prompter.format("Awesome acting!", dataset_name="sst2")
        self.assertIn("<|user|>", prompt)
        self.assertIn("<|assistant|>", prompt)

    def test_gemma_formatting(self):
        prompter = PromptFormatter(model_name="gemma_26b", prompt_format=0)
        self.assertEqual(prompter.model_family, "gemma")

        # Zero-shot
        prompt = prompter.format("Terrible plot.", dataset_name="sst2")
        self.assertIn("<|turn>user", prompt)
        self.assertIn("<|turn>model", prompt)


class TestChatFamiliesUseTheDatasetInstruction(unittest.TestCase):
    """Llama-3, Qwen, Phi and Gemma get the dataset's own instruction and the
    same messages; only the chat template differs."""

    FAMILIES = ("qwen", "phi", "gemma", "llama3")

    def _messages(self, family, dataset, shots=None):
        tok = MagicMock()
        tok.apply_chat_template.side_effect = lambda messages, **kw: messages
        return PromptFormatter(model_name=family, tokenizer=tok).format(
            "some text", dataset_name=dataset, shots=shots)

    def test_the_system_turn_is_the_dataset_instruction(self):
        for dataset in PromptFormatter.DEFAULT_CLASSES:
            instruction = PromptFormatter(model_name="qwen").get_task_info(dataset)["instruction"]
            for family in self.FAMILIES:
                with self.subTest(family=family, dataset=dataset):
                    system, user = self._messages(family, dataset)
                    self.assertEqual(system["role"], "system")
                    self.assertTrue(system["content"].startswith(instruction))
                    self.assertTrue(user["content"].endswith("some text"))

    def test_boolq_asks_to_answer_the_question(self):
        system, _ = self._messages("qwen", "boolq")
        self.assertIn("Read the passage and answer the question with Yes or No.", system["content"])
        self.assertNotIn("question answering", system["content"])

    def test_the_four_families_get_the_same_messages(self):
        shots = [("loved it", "positive"), ("hated it", "negative")]
        for dataset in ("sst2", "boolq", "snips"):
            expected = self._messages("qwen", dataset, shots)
            for family in self.FAMILIES[1:]:
                with self.subTest(family=family, dataset=dataset):
                    self.assertEqual(self._messages(family, dataset, shots), expected)

    def test_icl_only_adds_demonstrations_to_the_zero_shot_prompt(self):
        zero_shot = self._messages("phi", "sst2")
        icl = self._messages("phi", "sst2", shots=[("loved it", "positive")])
        self.assertEqual(icl[0], zero_shot[0])
        self.assertEqual(icl[-1], zero_shot[-1])
        self.assertEqual([m["role"] for m in icl[1:-1]], ["user", "assistant"])


class TestDuplicateBos(unittest.TestCase):

    class _Tok:
        bos_token, bos_token_id = "<s>", 1

        def __init__(self, adds_bos):
            self.adds_bos = adds_bos

        def __call__(self, text):
            return {"input_ids": ([1] if self.adds_bos else []) + [42]}

        def apply_chat_template(self, messages, **kwargs):
            return "<s>[INST] " + messages[-1]["content"] + " [/INST]"

    def test_template_bos_dropped_when_tokenizer_adds_one(self):
        for family in ("mistral", "llama2"):
            prompt = PromptFormatter(model_name=family, tokenizer=self._Tok(adds_bos=True)).format(
                "Nice.", dataset_name="sst2")
            self.assertFalse(prompt.startswith("<s>"), family)
            self.assertTrue(prompt.startswith("[INST]"), family)

    def test_template_bos_kept_when_tokenizer_does_not_add_one(self):
        prompt = PromptFormatter(model_name="mistral", tokenizer=self._Tok(adds_bos=False)).format(
            "Nice.", dataset_name="sst2")
        self.assertTrue(prompt.startswith("<s>"))


if __name__ == "__main__":
    unittest.main()



