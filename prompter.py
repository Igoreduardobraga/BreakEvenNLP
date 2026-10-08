# prompter.py
"""
Deep PromptFormatter module for BreakEvenNLP.
Encapsulates task instructions, options, few-shot ICL formatting,
model-specific chat templates, and instruction tuning training sequences.
"""

import copy
from typing import Any, Dict, List, Optional, Tuple, Union


class PromptFormatter:
    """
    Deep module responsible for formatting prompts across datasets,
    model families, and evaluation/training modes (prompting, icl, instruction_tuning).
    """

    DEFAULT_CLASSES = {
        "sst2": ("negative", "positive"),
        "cola": ("No", "Yes"),
        "mrpc": ("No", "Yes"),
        "rte": ("not entailment", "entailment"),
        "boolq": ("No", "Yes"),
        "trec": ("Expression", "Entity", "Description", "Human", "Location", "Number"),
        "ag_news": ("World", "Sports", "Business", "Science and Technology"),
        "snips": ("Playlist", "Weather", "Event", "Music", "Creative Work", "Rate Book", "Book Restaurant"),
        "db_pedia": (
            "Company", "Educational Institution", "Artist", "Athlete", "Office Holder",
            "Transportation", "Building", "Natural Place", "Village", "Animal",
            "Plant", "Album", "Film", "Written Work"
        ),
    }

    def __init__(self, model_name: str, prompt_format: int = 0, tokenizer: Any = None):
        self.model_name = str(model_name).lower()
        self.prompt_format = prompt_format
        self.tokenizer = tokenizer
        self.model_family = self._resolve_model_family(self.model_name)

    @staticmethod
    def _resolve_model_family(name: str) -> str:
        name = name.lower()
        if "t5" in name:
            return "seq2seq"
        elif "qwen" in name:
            return "qwen"
        elif "phi" in name:
            return "phi"
        elif "gemma" in name:
            return "gemma"
        elif "llama-3" in name or "llama3" in name:
            return "llama3"
        elif "llama-2" in name or "llama2" in name:
            return "llama2"
        elif "mistral" in name:
            return "mistral"
        elif "zephyr" in name:
            return "zephyr"
        elif "chatgpt" in name or "gpt-" in name or "openai" in name:
            return "chatgpt"
        return "seq2seq"

    def get_task_info(
        self,
        dataset_name: str,
        classes: Optional[Union[List[str], Tuple[str, ...]]] = None,
        custom_instruction: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Returns a dictionary with instructions, delimiter tags, options, and task type.
        """
        if custom_instruction is not None and isinstance(custom_instruction, dict):
            resolved_classes = tuple(classes) if classes is not None else tuple(custom_instruction.get("classes", ()))
            if not resolved_classes:
                resolved_classes = self.DEFAULT_CLASSES.get(str(dataset_name).lower(), ("Option 1", "Option 2"))
            options_str = ", ".join([f"'{c}'" for c in resolved_classes])
            return {
                "instruction": custom_instruction.get("instruction", f"Classify the text using these options: {options_str}."),
                "sentence_start": custom_instruction.get("sentence_start", "Text"),
                "answer_start": custom_instruction.get("answer_start", "Answer"),
                "task_type": custom_instruction.get("task_type", "classification"),
                "classes": list(resolved_classes),
                "options_str": options_str,
            }

        d_name = str(dataset_name).lower()

        if classes is None:
            if d_name == "sst2" and self.prompt_format in [3]:
                resolved_classes = ("terrible", "great")
            elif d_name == "cola" and self.prompt_format in [2]:
                resolved_classes = ("Yes", "No")
            elif d_name == "cola" and self.prompt_format in [3]:
                resolved_classes = ("not acceptable", "acceptable")
            elif d_name == "mrpc" and self.prompt_format in [2]:
                resolved_classes = ("Yes", "No")
            elif d_name == "mrpc" and self.prompt_format in [3]:
                resolved_classes = ("not equivalent", "equivalent")
            else:
                resolved_classes = self.DEFAULT_CLASSES.get(d_name, ("Option 1", "Option 2"))
        else:
            resolved_classes = tuple(classes)

        options_str = ", ".join([f"'{c}'" for c in resolved_classes])

        sentence_start = "Text"
        answer_start = "Answer"
        task_type = "classification"

        if d_name == "sst2":
            instruction = f"Analyze the sentiment of the text. Options: {options_str}."
            task_type = "sentiment analysis"
        elif d_name == "cola":
            instruction = f"Determine if the sentence is grammatically acceptable. Options: {options_str}."
            task_type = "grammatical acceptability"
        elif d_name == "mrpc":
            instruction = f"Determine if the two sentences are semantically equivalent. Options: {options_str}."
            sentence_start = "Sentences"
            task_type = "semantic equivalence"
        elif d_name == "boolq":
            instruction = "Read the passage and answer the question with Yes or No."
            sentence_start = ""
            task_type = "question answering"
        elif d_name == "ag_news":
            instruction = f"Classify the news article into one of these topics: {options_str}."
            task_type = "news classification"
        elif d_name == "trec":
            instruction = f"Classify the question type: {options_str}."
            task_type = "question classification"
        elif d_name == "snips":
            instruction = f"Identify the intent of the user command. Options: {options_str}."
            task_type = "intent detection"
        elif d_name == "db_pedia":
            instruction = f"Classify the topic of the text: {options_str}."
            task_type = "topic classification"
        elif d_name == "rte":
            instruction = f"Determine if the premise entails the hypothesis. Options: {options_str}."
            task_type = "natural language inference"
        else:
            instruction = f"Classify the text using these options: {options_str}."

        return {
            "instruction": instruction,
            "sentence_start": sentence_start,
            "answer_start": answer_start,
            "task_type": task_type,
            "classes": list(resolved_classes),
            "options_str": options_str,
        }

    def format(
        self,
        text: str,
        dataset_name: str,
        classes: Optional[Union[List[str], Tuple[str, ...]]] = None,
        shots: Optional[List[Tuple[str, str]]] = None,
        custom_instruction: Optional[Dict[str, Any]] = None,
    ) -> Union[str, List[Dict[str, str]]]:
        """
        Formats a single sample for inference (zero-shot or few-shot).
        Returns a prompt string for local models or a messages list for ChatGPT.
        """
        info = self.get_task_info(dataset_name, classes=classes, custom_instruction=custom_instruction)
        sample_str = text.strip()

        if self.model_family == "seq2seq":
            return self._format_seq2seq(sample_str, info, shots)
        elif self.model_family == "qwen":
            return self._format_qwen(sample_str, info, shots)
        elif self.model_family == "phi":
            return self._format_phi(sample_str, info, shots)
        elif self.model_family == "gemma":
            return self._format_gemma(sample_str, info, shots)
        elif self.model_family == "llama2":
            return self._format_llama2(sample_str, info, shots)
        elif self.model_family == "mistral":
            return self._format_mistral(sample_str, info, shots)
        elif self.model_family == "zephyr":
            return self._format_zephyr(sample_str, info, shots)
        elif self.model_family == "llama3":
            return self._format_llama3(sample_str, info, shots)
        elif self.model_family == "chatgpt":
            return self._format_chatgpt(sample_str, info, shots)

        return self._format_seq2seq(sample_str, info, shots)

    def format_batch(
        self,
        texts: List[str],
        dataset_name: str,
        classes: Optional[Union[List[str], Tuple[str, ...]]] = None,
        shots: Optional[List[Tuple[str, str]]] = None,
        custom_instruction: Optional[Dict[str, Any]] = None,
    ) -> List[Any]:
        """
        Formats a batch of samples for fast batched inference.
        """
        return [
            self.format(t, dataset_name, classes=classes, shots=shots, custom_instruction=custom_instruction)
            for t in texts
        ]

    def get_response_template(self) -> str:
        """
        Returns the response template delimiter token for DataCollatorForCompletionOnlyLM.
        """
        if self.model_family == "seq2seq":
            return "Answer:"
        elif self.model_family == "qwen":
            return "<|im_start|>assistant\n"
        elif self.model_family == "phi":
            return "<|assistant|>\n"
        elif self.model_family == "gemma":
            return "<start_of_turn>model\n"
        elif self.model_family == "mistral":
            return "[/INST]"
        elif self.model_family == "zephyr":
            return "<|assistant|>"
        elif self.model_family == "llama3":
            return "<|start_header_id|>assistant<|end_header_id|>\n\n"
        elif self.model_family == "llama2":
            return "[/INST]"
        return "Answer:"

    def format_instruction_tuning(
        self,
        samples: List[Tuple[str, str]],
        dataset_name: str,
        classes: Optional[Union[List[str], Tuple[str, ...]]] = None,
        custom_instruction: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        """
        Formats (input_text, target_label) sample pairs into full training prompt sequences
        for supervised fine-tuning (SFT).
        """
        info = self.get_task_info(dataset_name, classes=classes, custom_instruction=custom_instruction)
        instruction = info["instruction"]
        sentence_start = info["sentence_start"]
        answer_start = info["answer_start"]
        task_type = info["task_type"]
        d_name = str(dataset_name).lower()

        final_prompts = []

        if self.model_family == "seq2seq":
            for sample in samples:
                text = sample[0].strip()
                label = sample[1].strip()
                if self.prompt_format == 0:
                    p = f"{instruction}\n{sentence_start}: {text}\n{answer_start}: {label}"
                elif d_name == "snips" and self.prompt_format == 3:
                    p = f"User: {text}\n{instruction} {answer_start}: {label}"
                else:
                    p = f"{text}\n{instruction} {answer_start}: {label}"
                final_prompts.append(p)
            return final_prompts

        elif self.model_family == "mistral":
            for sample in samples:
                text = sample[0].strip()
                label = sample[1].strip()
                if self.prompt_format == 0:
                    user_text = f"{instruction}\n{sentence_start}: {text}"
                else:
                    user_text = f"{text} {instruction}"
                full_text = f"<s>[INST] {user_text} [/INST] {label}</s>"
                final_prompts.append(full_text)
            return final_prompts

        elif self.model_family == "zephyr":
            for sample in samples:
                text = sample[0].strip()
                label = sample[1].strip()
                if self.prompt_format == 0:
                    user_text = f"{sentence_start}: {text}"
                else:
                    user_text = f"{text} {instruction}"
                full_text = (
                    f"<|system|>\n{instruction}</s>\n"
                    f"<|user|>\n{user_text}</s>\n"
                    f"<|assistant|>\n{label}</s>"
                )
                final_prompts.append(full_text)
            return final_prompts

        elif self.model_family == "llama3":
            system_header = "<|start_header_id|>system<|end_header_id|>\n\n"
            user_header = "<|start_header_id|>user<|end_header_id|>\n\n"
            assistant_header = "<|start_header_id|>assistant<|end_header_id|>\n\n"
            eot = "<|eot_id|>"
            sys_msg = f"You are a helpful assistant. Follow the instruction exactly. Determine the {task_type}."

            for sample in samples:
                text = sample[0].strip()
                label = sample[1].strip()
                if self.prompt_format == 0:
                    user_input = f"{instruction}\n{sentence_start}: {text}\n{answer_start}: "
                else:
                    user_input = f"{text} {instruction}"
                full_text = (
                    f"{system_header}{sys_msg}{eot}"
                    f"{user_header}{user_input}{eot}"
                    f"{assistant_header}{label}{eot}"
                )
                final_prompts.append(full_text)
            return final_prompts

        elif self.model_family == "qwen":
            sys_msg = f"You are a helpful assistant. Follow the instruction exactly. Determine the {task_type}."
            for sample in samples:
                text = sample[0].strip()
                label = sample[1].strip()
                if self.prompt_format == 0:
                    user_input = f"{instruction}\n{sentence_start}: {text}\n{answer_start}: "
                else:
                    user_input = f"{text} {instruction}"
                full_text = (
                    f"<|im_start|>system\n{sys_msg}<|im_end|>\n"
                    f"<|im_start|>user\n{user_input}<|im_end|>\n"
                    f"<|im_start|>assistant\n{label}<|im_end|>"
                )
                final_prompts.append(full_text)
            return final_prompts

        elif self.model_family == "phi":
            for sample in samples:
                text = sample[0].strip()
                label = sample[1].strip()
                if self.prompt_format == 0:
                    user_input = f"{instruction}\n{sentence_start}: {text}\n{answer_start}: "
                else:
                    user_input = f"{text} {instruction}"
                full_text = (
                    f"<|user|>\n{user_input}<|end|>\n"
                    f"<|assistant|>\n{label}<|end|>"
                )
                final_prompts.append(full_text)
            return final_prompts

        elif self.model_family == "gemma":
            for sample in samples:
                text = sample[0].strip()
                label = sample[1].strip()
                if self.prompt_format == 0:
                    user_input = f"{instruction}\n{sentence_start}: {text}\n{answer_start}: "
                else:
                    user_input = f"{text} {instruction}"
                full_text = (
                    f"<start_of_turn>user\n{user_input}<end_of_turn>\n"
                    f"<start_of_turn>model\n{label}<end_of_turn>"
                )
                final_prompts.append(full_text)
            return final_prompts

        else:
            sys_msg = f"You are a helpful assistant. Your task is {task_type}. Answer ONLY with one of the valid class names. Do not explain."
            for sample in samples:
                text = sample[0].strip()
                label = sample[1].strip()
                if self.prompt_format == 0:
                    user_query = f"{instruction}\n\nInput: {text}\nAnswer:"
                else:
                    user_query = f"{text}\n{instruction}"
                full_text = f"<s>[INST] <<SYS>>\n{sys_msg}\n<</SYS>>\n\n{user_query} [/INST] {label} </s>"
                final_prompts.append(full_text)
            return final_prompts

    def _format_seq2seq(self, sample: str, info: Dict[str, Any], shots: Optional[List[Tuple[str, str]]] = None) -> str:
        instruction = info["instruction"]
        s_start = info["sentence_start"]
        a_start = info["answer_start"]

        if shots and len(shots) > 0:
            if self.prompt_format == 0:
                base = f"{instruction}\n"
                for s in shots:
                    prefix = f"{s_start}: " if s_start else ""
                    base += f"{prefix}{s[0].strip()}\n{a_start}: {s[1].strip()}\n"
                prefix = f"{s_start}: " if s_start else ""
                return f"{base}{prefix}{sample}\n{a_start}: "
            else:
                base = ""
                for s in shots:
                    base += f"{s[0].strip()}\n{instruction} {s[1].strip()}\n"
                return f"{base}{sample}\n{instruction} "

        if self.prompt_format == 0:
            prefix = f"{s_start}: " if s_start else ""
            return f"{instruction}\n{prefix}{sample}\n{a_start}: "
        else:
            return f"{sample}\n{instruction} "

    def _format_llama2(self, sample: str, info: Dict[str, Any], shots: Optional[List[Tuple[str, str]]] = None) -> str:
        instruction = info["instruction"]
        task_type = info["task_type"]
        sys_msg = (
            f"You are a helpful assistant. Your task is {task_type}. "
            f"Answer ONLY with one of the valid class names. Do not explain."
        )

        prompt_history = ""
        if shots and len(shots) > 0:
            for s in shots:
                prompt_history += f"<s>[INST] {s[0]} [/INST] {s[1]} </s>"

        if self.prompt_format == 0:
            user_query = f"{instruction}\n\nInput: {sample}\nAnswer:"
        else:
            user_query = f"{sample}\n{instruction}"

        if shots and len(shots) > 0:
            return f"<s>[INST] <<SYS>>\n{sys_msg}\n<</SYS>>\n\n{prompt_history}{user_query} [/INST]"
        else:
            return f"<s>[INST] <<SYS>>\n{sys_msg}\n<</SYS>>\n\n{user_query} [/INST]"

    def _format_mistral(self, sample: str, info: Dict[str, Any], shots: Optional[List[Tuple[str, str]]] = None) -> str:
        instruction = info["instruction"]
        s_start = info["sentence_start"]
        prefix = f"{s_start}: " if s_start else ""
        instruction_text = f"{instruction} Do not explain. Answer only with the class name."

        messages = []
        if shots and len(shots) > 0:
            messages.append({"role": "user", "content": instruction_text})
            messages.append({"role": "assistant", "content": "Understood."})
            for s in shots:
                if self.prompt_format == 0:
                    content = f"{prefix}{s[0]}"
                else:
                    content = f"{s[0]} {instruction}"
                messages.append({"role": "user", "content": content})
                messages.append({"role": "assistant", "content": s[1]})

            if self.prompt_format == 0:
                content = f"{prefix}{sample}"
            else:
                content = f"{sample} {instruction}"
            messages.append({"role": "user", "content": content})
        else:
            if self.prompt_format == 0:
                content = f"{instruction_text}\n\n{prefix}{sample}"
            else:
                content = f"{sample} {instruction}"
            messages.append({"role": "user", "content": content})

        if hasattr(self.tokenizer, "apply_chat_template"):
            return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        # Native fallback
        if shots and len(shots) > 0:
            native_str = f"<s>[INST] {messages[0]['content']} [/INST] {messages[1]['content']}</s>"
            for i in range(2, len(messages) - 1, 2):
                native_str += f"[INST] {messages[i]['content']} [/INST] {messages[i+1]['content']}</s>"
            native_str += f"[INST] {messages[-1]['content']} [/INST]"
            return native_str
        else:
            return f"<s>[INST] {messages[0]['content']} [/INST]"

    def _format_zephyr(self, sample: str, info: Dict[str, Any], shots: Optional[List[Tuple[str, str]]] = None) -> str:
        instruction = info["instruction"]
        s_start = info["sentence_start"]
        prefix = f"{s_start}: " if s_start else ""
        instruction_text = f"{instruction} Answer with the class name only."

        messages = [{"role": "system", "content": instruction_text}]
        if shots and len(shots) > 0:
            for s in shots:
                if self.prompt_format == 0:
                    user_content = f"{prefix}{s[0]}"
                else:
                    user_content = f"{s[0]} {instruction}"
                messages.append({"role": "user", "content": user_content})
                messages.append({"role": "assistant", "content": s[1]})

        if self.prompt_format == 0:
            content = f"{prefix}{sample}"
        else:
            content = f"{sample} {instruction}"
        messages.append({"role": "user", "content": content})

        if hasattr(self.tokenizer, "apply_chat_template"):
            return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        # Native fallback
        native_str = f"<|system|>\n{instruction_text}</s>\n"
        for i in range(1, len(messages) - 1, 2):
            native_str += f"<|user|>\n{messages[i]['content']}</s>\n<|assistant|>\n{messages[i+1]['content']}</s>\n"
        native_str += f"<|user|>\n{messages[-1]['content']}</s>\n<|assistant|>\n"
        return native_str

    def _format_llama3(self, sample: str, info: Dict[str, Any], shots: Optional[List[Tuple[str, str]]] = None) -> str:
        instruction = info["instruction"]
        task_type = info["task_type"]
        options_str = info["options_str"]
        s_start = info["sentence_start"]
        prefix = f"{s_start}: " if s_start else ""

        sys_msg = (
            f"You are an expert classifier. Your task is to determine the {task_type}. "
            f"You must answer with exactly one of these options: {options_str}. "
            f"Do not explain. Do not output the number, only the class name."
        )

        messages = [{"role": "system", "content": sys_msg}]
        if shots and len(shots) > 0:
            for s in shots:
                if self.prompt_format == 0:
                    u_content = f"{prefix}{s[0]}"
                else:
                    u_content = f"{s[0]} {instruction}"
                messages.append({"role": "user", "content": u_content})
                messages.append({"role": "assistant", "content": s[1]})

        if self.prompt_format == 0:
            q_content = f"Text: {sample}\nBased on the text above, determine the {task_type}. Answer:"
        else:
            q_content = f"{sample} {instruction} "
        messages.append({"role": "user", "content": q_content})

        if hasattr(self.tokenizer, "apply_chat_template"):
            return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        # Native fallback
        header_sys = "<|start_header_id|>system<|end_header_id|>\n\n"
        header_user = "<|start_header_id|>user<|end_header_id|>\n\n"
        header_asst = "<|start_header_id|>assistant<|end_header_id|>\n\n"
        eot = "<|eot_id|>"

        native_str = f"{header_sys}{sys_msg}{eot}"
        for i in range(1, len(messages) - 1, 2):
            native_str += f"{header_user}{messages[i]['content']}{eot}{header_asst}{messages[i+1]['content']}{eot}"
        native_str += f"{header_user}{messages[-1]['content']}{eot}{header_asst}"
        return native_str

    def _format_chatgpt(self, sample: str, info: Dict[str, Any], shots: Optional[List[Tuple[str, str]]] = None) -> List[Dict[str, str]]:
        instruction = info["instruction"]
        s_start = info["sentence_start"]
        a_start = info["answer_start"]
        prefix = f"{s_start}: " if s_start else ""

        if shots and len(shots) > 0:
            if self.prompt_format == 0:
                user_content = f"{instruction}\n"
                for s in shots:
                    user_content += f"{prefix}{s[0].strip()}\n{a_start}: {s[1].strip()}\n"
                user_content += f"{prefix}{sample}\n{a_start}: "
            else:
                user_content = ""
                for s in shots:
                    user_content += f"{s[0].strip()}\n{instruction} {s[1].strip()}\n"
                user_content += f"{sample}\n{instruction} "
        else:
            if self.prompt_format == 0:
                user_content = f"{instruction}\n{prefix}{sample}\n{a_start}: "
            else:
                user_content = f"{sample}\n{instruction} "

        return [
            {"role": "system", "content": "You are a helpful assistant that follows all the instructions."},
            {"role": "user", "content": user_content},
        ]

    def _format_qwen(self, sample: str, info: Dict[str, Any], shots: Optional[List[Tuple[str, str]]] = None) -> str:
        instruction = info["instruction"]
        task_type = info["task_type"]
        options_str = info["options_str"]
        s_start = info["sentence_start"]
        prefix = f"{s_start}: " if s_start else ""

        sys_msg = (
            f"You are a helpful assistant. Your task is to determine the {task_type}. "
            f"Answer with exactly one of these options: {options_str}. "
            f"Do not explain. Answer only with the class name."
        )

        messages = [{"role": "system", "content": sys_msg}]
        if shots and len(shots) > 0:
            for s in shots:
                if self.prompt_format == 0:
                    u_content = f"{prefix}{s[0]}"
                else:
                    u_content = f"{s[0]} {instruction}"
                messages.append({"role": "user", "content": u_content})
                messages.append({"role": "assistant", "content": s[1]})

        if self.prompt_format == 0:
            q_content = f"Text: {sample}\nBased on the text above, determine the {task_type}. Answer:"
        else:
            q_content = f"{sample} {instruction} "
        messages.append({"role": "user", "content": q_content})

        if hasattr(self.tokenizer, "apply_chat_template"):
            return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        native_str = f"<|im_start|>system\n{sys_msg}<|im_end|>\n"
        for i in range(1, len(messages) - 1, 2):
            native_str += f"<|im_start|>user\n{messages[i]['content']}<|im_end|>\n<|im_start|>assistant\n{messages[i+1]['content']}<|im_end|>\n"
        native_str += f"<|im_start|>user\n{messages[-1]['content']}<|im_end|>\n<|im_start|>assistant\n"
        return native_str

    def _format_phi(self, sample: str, info: Dict[str, Any], shots: Optional[List[Tuple[str, str]]] = None) -> str:
        instruction = info["instruction"]
        task_type = info["task_type"]
        options_str = info["options_str"]
        s_start = info["sentence_start"]
        prefix = f"{s_start}: " if s_start else ""

        instruction_text = (
            f"Determine the {task_type}. Options: {options_str}. "
            f"Answer only with the class name."
        )

        messages = []
        if shots and len(shots) > 0:
            for s in shots:
                if self.prompt_format == 0:
                    u_content = f"{instruction_text}\n{prefix}{s[0]}"
                else:
                    u_content = f"{s[0]} {instruction}"
                messages.append({"role": "user", "content": u_content})
                messages.append({"role": "assistant", "content": s[1]})

        if self.prompt_format == 0:
            q_content = f"{instruction_text}\n{prefix}{sample}"
        else:
            q_content = f"{sample} {instruction}"
        messages.append({"role": "user", "content": q_content})

        if hasattr(self.tokenizer, "apply_chat_template"):
            return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        native_str = ""
        for i in range(0, len(messages) - 1, 2):
            native_str += f"<|user|>\n{messages[i]['content']}<|end|>\n<|assistant|>\n{messages[i+1]['content']}<|end|>\n"
        native_str += f"<|user|>\n{messages[-1]['content']}<|end|>\n<|assistant|>\n"
        return native_str

    def _format_gemma(self, sample: str, info: Dict[str, Any], shots: Optional[List[Tuple[str, str]]] = None) -> str:
        instruction = info["instruction"]
        task_type = info["task_type"]
        options_str = info["options_str"]
        s_start = info["sentence_start"]
        prefix = f"{s_start}: " if s_start else ""

        instruction_text = (
            f"Analyze the text to determine the {task_type}. Options: {options_str}. "
            f"Answer only with the exact class name."
        )

        messages = []
        if shots and len(shots) > 0:
            for s in shots:
                if self.prompt_format == 0:
                    u_content = f"{instruction_text}\n{prefix}{s[0]}"
                else:
                    u_content = f"{s[0]} {instruction}"
                messages.append({"role": "user", "content": u_content})
                messages.append({"role": "assistant", "content": s[1]})

        if self.prompt_format == 0:
            q_content = f"{instruction_text}\n{prefix}{sample}"
        else:
            q_content = f"{sample} {instruction}"
        messages.append({"role": "user", "content": q_content})

        if hasattr(self.tokenizer, "apply_chat_template"):
            return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        native_str = ""
        for i in range(0, len(messages) - 1, 2):
            native_str += f"<start_of_turn>user\n{messages[i]['content']}<end_of_turn>\n<start_of_turn>model\n{messages[i+1]['content']}<end_of_turn>\n"
        native_str += f"<start_of_turn>user\n{messages[-1]['content']}<end_of_turn>\n<start_of_turn>model\n"
        return native_str
