# evaluator.py
"""
Deep ModelEvaluator module for BreakEvenNLP.
Encapsulates model generation, prompt formatting, and result parsing across model architectures.
"""

import re
import copy
import os
import pickle
import time


def _parse_results(text, classes):
    """
    Parses generated text to match target class names or numeric indices.
    Returns the zero-based class index, or -1 if unparseable or ambiguous.
    """
    t = str(text).strip().lower()
    candidates = set()

    # Numeric response: try to find a number 1..N
    m = re.match(r'^\s*(\d{1,2})\s*\)?\s*$', t)
    if m:
        k = int(m.group(1)) - 1
        if 0 <= k < len(classes):
            candidates.add(k)

    # Textual response: try to find the name of the class
    for idx, cls in enumerate(classes):
        cls_pat = r'\b' + re.escape(str(cls).lower()) + r'\b'
        if re.search(cls_pat, t):
            candidates.add(idx)

    # Returns the index found if exactly one unambiguous candidate matched
    if len(candidates) == 1:
        return next(iter(candidates))
    return -1


class _Seq2SeqAdapter:
    """Internal adapter for encoder-decoder models (e.g. Flan-T5)."""

    def __init__(self, model, tokenizer, batch_size=64, prompt_format=0, max_new_tokens=10, device='cuda'):
        self.model = model
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.prompt_format = prompt_format
        self.max_new_tokens = max_new_tokens
        self.device = device

    def _prepare_prompt(self, dataset, batch_samples):
        instructions = dataset.instructions
        prompts = []
        for sample in batch_samples:
            if self.prompt_format == 0:
                p = f"{instructions['instruction']}\n{instructions['sentence_start']}: {sample.strip()}\n{instructions['answer_start']}: "
            else:
                p = f"{sample.strip()}\n{instructions['instruction']} "
            prompts.append(p)
        return prompts

    def _prepare_icl(self, dataset, batch_samples):
        instructions = dataset.instructions
        context_samples = getattr(dataset, 'context_samples', [])
        if self.prompt_format == 0:
            base = f"{instructions['instruction']}\n"
            for s in context_samples:
                base += f"{instructions['sentence_start']}: {s[0].strip()}\n{instructions['answer_start']}: {s[1].strip()}\n"
        else:
            base = ""
            for s in context_samples:
                base += f"{s[0].strip()}\n{instructions['instruction']} {s[1].strip()}\n"

        prompts = []
        for sample in batch_samples:
            if self.prompt_format == 0:
                p = f"{base}{instructions['sentence_start']}: {sample.strip()}\n{instructions['answer_start']}: "
            else:
                p = f"{base}{sample.strip()}\n{instructions['instruction']} "
            prompts.append(p)
        return prompts

    def evaluate(self, dataset, mode='prompting'):
        golden = []
        predicted = []
        decodeds = []

        for data, labels in dataset.batch_data_for_evaluation(self.batch_size):
            if mode == 'icl':
                batch_prompts = self._prepare_icl(dataset, data)
            else:
                batch_prompts = self._prepare_prompt(dataset, data)

            encoded = self.tokenizer(batch_prompts, return_tensors='pt', padding='longest', truncation=True)
            if hasattr(encoded, 'to'):
                encoded = encoded.to(self.device)

            out = self.model.generate(
                **encoded if isinstance(encoded, dict) else {'input_ids': encoded},
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                num_beams=1
            )
            decoded = self.tokenizer.batch_decode(out, skip_special_tokens=True)

            decodeds.extend(decoded)
            for text in decoded:
                pred = _parse_results(text, dataset.classes)
                predicted.append(pred)

            golden.extend(labels)

        return golden, predicted, decodeds


class _CausalLMAdapter:
    """Internal adapter for causal autoregressive models (LLaMA-2, LLaMA-3, Mistral, Zephyr)."""

    def __init__(self, model, tokenizer, model_family, batch_size=1, prompt_format=0, max_new_tokens=10, generation_config=None, device='cuda'):
        self.model = model
        self.tokenizer = tokenizer
        self.model_family = model_family
        self.batch_size = batch_size
        self.prompt_format = prompt_format
        self.max_new_tokens = max_new_tokens
        self.generation_config = generation_config
        self.device = device

    def _build_messages(self, dataset, sample, mode='prompting'):
        instructions = dataset.instructions
        context_samples = getattr(dataset, 'context_samples', [])

        if self.model_family == 'llama3':
            options_str = ", ".join([f"'{c}'" for c in dataset.classes])
            system_message = {
                "role": "system",
                "content": (
                    f"You are an expert classifier. Your task is to determine the {instructions['task_type']}. "
                    f"You must answer with exactly one of these options: {options_str}. "
                    f"Do not explain. Do not output the number, only the class name."
                )
            }
            messages = [system_message]
            if mode == 'icl':
                for s in context_samples:
                    if self.prompt_format == 0:
                        u_content = f"{instructions['sentence_start']}: {s[0]}"
                    else:
                        u_content = f"{s[0]} {instructions['instruction']}"
                    messages.append({'role': 'user', 'content': u_content})
                    messages.append({'role': 'assistant', 'content': s[1]})

            if self.prompt_format == 0:
                q_content = f"Text: {sample}\nBased on the text above, determine the {instructions['task_type']}. Answer:"
            else:
                q_content = f"{sample} {instructions['instruction']} "
            messages.append({'role': 'user', 'content': q_content})
            return messages

        elif self.model_family == 'mistral':
            instruction_text = f"{instructions['instruction']} Do not explain. Answer only with the class name."
            messages = []
            if mode == 'icl':
                messages.append({'role': 'user', 'content': instruction_text})
                messages.append({'role': 'assistant', 'content': 'Understood.'})
                for s in context_samples:
                    if self.prompt_format == 0:
                        content = f"{instructions['sentence_start']}: {s[0]}"
                    else:
                        content = f"{s[0]} {instructions['instruction']}"
                    messages.append({'role': 'user', 'content': content})
                    messages.append({'role': 'assistant', 'content': s[1]})

            if self.prompt_format == 0:
                if mode == 'prompting':
                    content = f"{instruction_text}\n\n{instructions['sentence_start']}: {sample}"
                else:
                    content = f"{instructions['sentence_start']}: {sample}"
            else:
                content = f"{sample} {instructions['instruction']}"
            messages.append({'role': 'user', 'content': content})
            return messages

        elif self.model_family == 'zephyr':
            instruction_text = f"{instructions['instruction']} Answer with the class name only."
            messages = [{'role': 'system', 'content': instruction_text}]
            if mode == 'icl':
                for s in context_samples:
                    if self.prompt_format == 0:
                        user_content = f"{instructions['sentence_start']}: {s[0]}"
                    else:
                        user_content = f"{s[0]} {instructions['instruction']}"
                    messages.append({'role': 'user', 'content': user_content})
                    messages.append({'role': 'assistant', 'content': s[1]})

            if self.prompt_format == 0:
                content = f"{instructions['sentence_start']}: {sample}"
            else:
                content = f"{sample} {instructions['instruction']}"
            messages.append({'role': 'user', 'content': content})
            return messages

        elif self.model_family == 'llama2':
            sys_msg = (
                f"You are a helpful assistant. Your task is {instructions['task_type']}. "
                f"Answer ONLY with one of the valid class names. Do not explain."
            )
            prompt_history = ""
            if mode == 'icl':
                for s in context_samples:
                    prompt_history += f"<s>[INST] {s[0]} [/INST] {s[1]} </s>"

            if self.prompt_format == 0:
                user_query = f"{instructions['instruction']}\n\nInput: {sample}\nAnswer:"
            else:
                user_query = f"{sample}\n{instructions['instruction']}"

            if mode == 'icl' and len(context_samples) > 0:
                final_prompt = f"<s>[INST] <<SYS>>\n{sys_msg}\n<</SYS>>\n\n{prompt_history}{user_query} [/INST]"
            else:
                final_prompt = f"<s>[INST] <<SYS>>\n{sys_msg}\n<</SYS>>\n\n{user_query} [/INST]"
            return final_prompt

        raise NotImplementedError(f"Unsupported causal model family: {self.model_family}")

    def evaluate(self, dataset, mode='prompting'):
        golden = []
        predicted = []
        decodeds = []

        terminators = []
        if hasattr(self.tokenizer, 'eos_token_id') and self.tokenizer.eos_token_id is not None:
            terminators.append(self.tokenizer.eos_token_id)
        if hasattr(self.tokenizer, 'convert_tokens_to_ids'):
            eot_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
            if eot_id is not None and eot_id not in terminators:
                terminators.append(eot_id)

        for data, labels in dataset.batch_data_for_evaluation(self.batch_size):
            for sample in data:
                msgs = self._build_messages(dataset, sample, mode)

                if isinstance(msgs, str):
                    prompt_str = msgs
                elif hasattr(self.tokenizer, 'apply_chat_template'):
                    prompt_str = self.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                else:
                    prompt_str = str(msgs)

                inputs = self.tokenizer(prompt_str, return_tensors="pt", padding=True, truncation=True)
                if hasattr(inputs, 'to'):
                    inputs = inputs.to(self.device)

                if hasattr(inputs, 'input_ids'):
                    input_len = inputs.input_ids.shape[-1] if hasattr(inputs.input_ids, 'shape') else len(inputs.input_ids[0])
                    input_kwargs = {'input_ids': inputs.input_ids}
                    if hasattr(inputs, 'attention_mask'):
                        input_kwargs['attention_mask'] = inputs.attention_mask
                else:
                    input_len = inputs.shape[-1] if hasattr(inputs, 'shape') else len(inputs[0])
                    input_kwargs = {'input_ids': inputs}

                gen_kwargs = {
                    'max_new_tokens': self.max_new_tokens,
                    'do_sample': False,
                }
                if terminators:
                    gen_kwargs['eos_token_id'] = terminators if len(terminators) > 1 else terminators[0]
                if hasattr(self.tokenizer, 'pad_token_id') and self.tokenizer.pad_token_id is not None:
                    gen_kwargs['pad_token_id'] = self.tokenizer.pad_token_id
                elif hasattr(self.tokenizer, 'eos_token_id') and self.tokenizer.eos_token_id is not None:
                    gen_kwargs['pad_token_id'] = self.tokenizer.eos_token_id

                if self.generation_config is not None:
                    gen_kwargs['generation_config'] = self.generation_config

                out = self.model.generate(**input_kwargs, **gen_kwargs)

                if hasattr(out, '__getitem__'):
                    raw_gen = out[0]
                    if hasattr(raw_gen, '__getitem__'):
                        new_tokens = raw_gen[input_len:]
                    else:
                        new_tokens = raw_gen
                else:
                    new_tokens = out

                decoded_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
                decoded_clean = decoded_text.replace('.', '').replace('"', '').strip()

                decodeds.append(decoded_clean)
                pred = _parse_results(decoded_clean, dataset.classes)
                predicted.append(pred)

            golden.extend(labels)

        return golden, predicted, decodeds


class _APIAdapter:
    """Internal adapter for external API models (ChatGPT)."""

    def __init__(self, client=None, model_id='gpt-3.5-turbo', prompt_format=0, max_retries=5, retry_delay=0.01):
        self.client = client
        self.model_id = model_id
        self.prompt_format = prompt_format
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    def _get_client(self):
        if self.client is not None:
            return self.client
        import openai
        api_key = os.environ.get('OPEN_AI_KEY')
        organization = os.environ.get('OPEN_AI_ORGANISATION')
        self.client = openai.OpenAI(api_key=api_key, organization=organization)
        return self.client

    def _prepare_prompt(self, dataset, sample):
        instructions = dataset.instructions
        if self.prompt_format == 0:
            return f"{instructions['instruction']}\n{instructions['sentence_start']}: {sample.strip()}\n{instructions['answer_start']}: "
        else:
            return f"{sample.strip()}\n{instructions['instruction']} "

    def _prepare_icl(self, dataset, sample):
        instructions = dataset.instructions
        context_samples = getattr(dataset, 'context_samples', [])
        if self.prompt_format == 0:
            prompt = f"{instructions['instruction']}\n"
            for s in context_samples:
                prompt += f"{instructions['sentence_start']}: {s[0].strip()}\n{instructions['answer_start']}: {s[1].strip()}\n"
            prompt += f"{instructions['sentence_start']}: {sample.strip()}\n{instructions['answer_start']}: "
        else:
            prompt = ""
            for s in context_samples:
                prompt += f"{s[0].strip()}\n{instructions['instruction']} {s[1].strip()}\n"
            prompt += f"{sample.strip()}\n{instructions['instruction']} "
        return prompt

    def evaluate(self, dataset, mode='prompting', partial_save_path=None):
        client = self._get_client()

        if partial_save_path:
            os.makedirs(partial_save_path, exist_ok=True)

        golden = []
        predicted = []
        decodeds = []

        sample_idx = 0
        for data, labels in dataset.batch_data_for_evaluation(1):
            sample = data[0]
            label = labels[0]

            pickle_file = os.path.join(partial_save_path, f"{sample_idx}.pkl") if partial_save_path else None

            if pickle_file and os.path.exists(pickle_file):
                with open(pickle_file, 'rb') as f:
                    saved_result = pickle.load(f)
                decoded = saved_result['predicted']
            else:
                prompt = self._prepare_icl(dataset, sample) if mode == 'icl' else self._prepare_prompt(dataset, sample)

                attempts = 0
                while True:
                    try:
                        response = client.chat.completions.create(
                            model=self.model_id,
                            messages=[
                                {"role": "system", "content": "You are a helpful assistant that follows all the instructions."},
                                {"role": "user", "content": prompt}
                            ],
                            temperature=0,
                            max_tokens=30
                        )
                        decoded = response.choices[0].message.content
                        break
                    except Exception as e:
                        attempts += 1
                        if attempts >= self.max_retries:
                            raise e
                        time.sleep(self.retry_delay)

                if pickle_file:
                    with open(pickle_file, 'wb') as f:
                        pickle.dump({'predicted': decoded, 'real': label}, f)

            decodeds.append(decoded)
            pred = _parse_results(decoded, dataset.classes)
            predicted.append(pred)
            golden.append(label)

            sample_idx += 1

        return golden, predicted, decodeds


class ModelEvaluator:
    """
    Deep ModelEvaluator module presenting a single, uniform interface
    to evaluate language models across tasks and architectures.
    """

    def __init__(
        self,
        model,
        tokenizer,
        model_name: str,
        batch_size: int = 64,
        prompt_format: int = 0,
        max_new_tokens: int = 10,
        generation_config=None,
        device: str = "cuda"
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.model_name = str(model_name).lower()
        self.batch_size = batch_size
        self.prompt_format = prompt_format
        self.max_new_tokens = max_new_tokens
        self.generation_config = generation_config
        self.device = device

        self.adapter = self._resolve_adapter()

    def _resolve_adapter(self):
        if 'flan-t5' in self.model_name:
            return _Seq2SeqAdapter(
                model=self.model,
                tokenizer=self.tokenizer,
                batch_size=self.batch_size,
                prompt_format=self.prompt_format,
                max_new_tokens=self.max_new_tokens,
                device=self.device
            )
        elif 'chatgpt' in self.model_name:
            return _APIAdapter(
                client=self.model if self.model is not None and not isinstance(self.model, str) else None,
                model_id=self.model_name if self.model_name != 'chatgpt' else 'gpt-3.5-turbo',
                prompt_format=self.prompt_format
            )
        elif any(fam in self.model_name for fam in ('llama3', 'mistral', 'zephyr', 'llama2')):
            family = 'llama3' if 'llama3' in self.model_name else (
                'mistral' if 'mistral' in self.model_name else (
                    'zephyr' if 'zephyr' in self.model_name else 'llama2'
                )
            )
            return _CausalLMAdapter(
                model=self.model,
                tokenizer=self.tokenizer,
                model_family=family,
                batch_size=self.batch_size,
                prompt_format=self.prompt_format,
                max_new_tokens=self.max_new_tokens,
                generation_config=self.generation_config,
                device=self.device
            )
        else:
            raise NotImplementedError(f"Model family for '{self.model_name}' not yet supported in ModelEvaluator.")

    def evaluate(self, dataset, mode: str = 'prompting', partial_save_path=None):
        """
        Evaluates the model on the provided dataset.
        
        Args:
            dataset: A dataset object providing `batch_data_for_evaluation`, `classes`, and prompt metadata.
            mode: Operating mode ('prompting' for zero-shot, 'icl' for in-context learning).
            partial_save_path: Optional path to save and resume partial results (for API models).
            
        Returns:
            Tuple of (golden_labels, predicted_labels, decoded_texts).
        """
        if mode not in ('prompting', 'icl'):
            raise ValueError(f"Unknown evaluation mode '{mode}'. Supported modes: 'prompting', 'icl'.")

        if isinstance(self.adapter, _APIAdapter):
            return self.adapter.evaluate(dataset, mode=mode, partial_save_path=partial_save_path)
        return self.adapter.evaluate(dataset, mode=mode)
