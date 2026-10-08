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
from prompter import PromptFormatter


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


class EvaluationResult(tuple):
    """
    3-tuple subclass containing (golden, predicted, decodeds) with qualitative metadata attributes.
    Maintains 100% backward compatibility with code expecting a (golden, predicted, decodeds) tuple.
    """
    def __new__(cls, golden, predicted, decodeds, prompts=None, inputs=None):
        instance = super().__new__(cls, (golden, predicted, decodeds))
        instance.golden = golden
        instance.predicted = predicted
        instance.decodeds = decodeds
        instance.prompts = list(prompts) if prompts is not None else []
        instance.inputs = list(inputs) if inputs is not None else []
        return instance


class _Seq2SeqAdapter:
    """Internal adapter for encoder-decoder models (e.g. Flan-T5)."""

    def __init__(self, model, tokenizer, prompter=None, batch_size=64, prompt_format=0, max_new_tokens=10, device='cuda'):
        self.model = model
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.prompt_format = prompt_format
        self.max_new_tokens = max_new_tokens
        self.device = device
        self.prompter = prompter or PromptFormatter(model_name='flan-t5', prompt_format=prompt_format, tokenizer=tokenizer)

    def evaluate(self, dataset, mode='prompting'):
        golden = []
        predicted = []
        decodeds = []
        prompts = []
        inputs = []

        shots = getattr(dataset, 'context_samples', None) if mode == 'icl' else None
        dataset_name = getattr(dataset, 'dataset_name', 'sst2')
        custom_inst = getattr(dataset, 'instructions', None)

        for data, labels in dataset.batch_data_for_evaluation(self.batch_size):
            batch_prompts = self.prompter.format_batch(
                data,
                dataset_name=dataset_name,
                classes=dataset.classes,
                shots=shots,
                custom_instruction=custom_inst
            )
            inputs.extend(data)
            prompts.extend(batch_prompts)

            encoded = self.tokenizer(batch_prompts, return_tensors='pt', padding='longest', truncation=True)
            if hasattr(encoded, 'to'):
                encoded = encoded.to(self.device)

            if hasattr(encoded, 'items'):
                gen_inputs = {k: v for k, v in encoded.items() if k in ('input_ids', 'attention_mask')}
            elif hasattr(encoded, 'input_ids'):
                gen_inputs = {'input_ids': encoded.input_ids}
            elif isinstance(encoded, dict):
                gen_inputs = encoded
            else:
                gen_inputs = {'input_ids': encoded}

            out = self.model.generate(
                **gen_inputs,
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

        return EvaluationResult(golden, predicted, decodeds, prompts=prompts, inputs=inputs)


class _CausalLMAdapter:
    """Internal adapter for causal autoregressive models (LLaMA-2, LLaMA-3, Mistral, Zephyr)."""

    def __init__(self, model, tokenizer, model_family, prompter=None, batch_size=1, prompt_format=0, max_new_tokens=10, generation_config=None, device='cuda'):
        self.model = model
        self.tokenizer = tokenizer
        self.model_family = model_family
        self.batch_size = batch_size
        self.prompt_format = prompt_format
        self.max_new_tokens = max_new_tokens
        self.generation_config = generation_config
        self.device = device
        self.prompter = prompter or PromptFormatter(model_name=model_family, prompt_format=prompt_format, tokenizer=tokenizer)

    def evaluate(self, dataset, mode='prompting'):
        golden = []
        predicted = []
        decodeds = []
        prompts = []
        inputs = []

        terminators = []
        if hasattr(self.tokenizer, 'eos_token_id') and self.tokenizer.eos_token_id is not None:
            if isinstance(self.tokenizer.eos_token_id, list):
                terminators.extend(self.tokenizer.eos_token_id)
            else:
                terminators.append(self.tokenizer.eos_token_id)
        if hasattr(self.tokenizer, 'convert_tokens_to_ids'):
            for t_token in ("<|eot_id|>", "<|im_end|>", "<|end|>", "<end_of_turn>"):
                tid = self.tokenizer.convert_tokens_to_ids(t_token)
                if tid is not None and not isinstance(tid, str) and tid not in terminators:
                    terminators.append(tid)

        shots = getattr(dataset, 'context_samples', None) if mode == 'icl' else None
        dataset_name = getattr(dataset, 'dataset_name', 'sst2')
        custom_inst = getattr(dataset, 'instructions', None)

        for data, labels in dataset.batch_data_for_evaluation(self.batch_size):
            for sample in data:
                formatted = self.prompter.format(
                    sample,
                    dataset_name=dataset_name,
                    classes=dataset.classes,
                    shots=shots,
                    custom_instruction=custom_inst
                )

                if isinstance(formatted, str):
                    prompt_str = formatted
                elif hasattr(self.tokenizer, 'apply_chat_template'):
                    prompt_str = self.tokenizer.apply_chat_template(formatted, tokenize=False, add_generation_prompt=True)
                else:
                    prompt_str = str(formatted)

                inputs.append(sample)
                prompts.append(prompt_str)

                inputs_tok = self.tokenizer(prompt_str, return_tensors="pt", padding=True, truncation=True)
                if hasattr(inputs_tok, 'to'):
                    inputs_tok = inputs_tok.to(self.device)

                if hasattr(inputs_tok, 'input_ids'):
                    input_len = inputs_tok.input_ids.shape[-1] if hasattr(inputs_tok.input_ids, 'shape') else len(inputs_tok.input_ids[0])
                    input_kwargs = {'input_ids': inputs_tok.input_ids}
                    if hasattr(inputs_tok, 'attention_mask'):
                        input_kwargs['attention_mask'] = inputs_tok.attention_mask
                else:
                    input_len = inputs_tok.shape[-1] if hasattr(inputs_tok, 'shape') else len(inputs_tok[0])
                    input_kwargs = {'input_ids': inputs_tok}

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

        return EvaluationResult(golden, predicted, decodeds, prompts=prompts, inputs=inputs)


class _APIAdapter:
    """Internal adapter for external API models (ChatGPT)."""

    def __init__(self, client=None, model_id='gpt-3.5-turbo', prompter=None, prompt_format=0, max_retries=5, retry_delay=0.01):
        self.client = client
        self.model_id = model_id
        self.prompt_format = prompt_format
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.prompter = prompter or PromptFormatter(model_name=model_id, prompt_format=prompt_format)

    def _get_client(self):
        if self.client is not None:
            return self.client
        import openai
        api_key = os.environ.get('OPEN_AI_KEY')
        organization = os.environ.get('OPEN_AI_ORGANISATION')
        self.client = openai.OpenAI(api_key=api_key, organization=organization)
        return self.client

    def evaluate(self, dataset, mode='prompting', partial_save_path=None):
        client = self._get_client()

        if partial_save_path:
            os.makedirs(partial_save_path, exist_ok=True)

        golden = []
        predicted = []
        decodeds = []
        prompts = []
        inputs = []

        shots = getattr(dataset, 'context_samples', None) if mode == 'icl' else None
        dataset_name = getattr(dataset, 'dataset_name', 'sst2')
        custom_inst = getattr(dataset, 'instructions', None)

        sample_idx = 0
        for data, labels in dataset.batch_data_for_evaluation(1):
            sample = data[0]
            label = labels[0]

            pickle_file = os.path.join(partial_save_path, f"{sample_idx}.pkl") if partial_save_path else None

            if pickle_file and os.path.exists(pickle_file):
                with open(pickle_file, 'rb') as f:
                    saved_result = pickle.load(f)
                decoded = saved_result['predicted']
                prompt_repr = ""
            else:
                messages = self.prompter.format(
                    sample,
                    dataset_name=dataset_name,
                    classes=dataset.classes,
                    shots=shots,
                    custom_instruction=custom_inst
                )
                prompt_repr = str(messages)
                if isinstance(messages, str):
                    messages = [
                        {"role": "system", "content": "You are a helpful assistant that follows all the instructions."},
                        {"role": "user", "content": messages}
                    ]

                attempts = 0
                while True:
                    try:
                        response = client.chat.completions.create(
                            model=self.model_id,
                            messages=messages,
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

            inputs.append(sample)
            prompts.append(prompt_repr)
            decodeds.append(decoded)
            pred = _parse_results(decoded, dataset.classes)
            predicted.append(pred)
            golden.append(label)

            sample_idx += 1

        return EvaluationResult(golden, predicted, decodeds, prompts=prompts, inputs=inputs)


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
        self.prompter = PromptFormatter(model_name=self.model_name, prompt_format=self.prompt_format, tokenizer=self.tokenizer)

        self.adapter = self._resolve_adapter()

    def _resolve_adapter(self):
        if 'flan-t5' in self.model_name:
            return _Seq2SeqAdapter(
                model=self.model,
                tokenizer=self.tokenizer,
                prompter=self.prompter,
                batch_size=self.batch_size,
                prompt_format=self.prompt_format,
                max_new_tokens=self.max_new_tokens,
                device=self.device
            )
        elif 'chatgpt' in self.model_name:
            return _APIAdapter(
                client=self.model if self.model is not None and not isinstance(self.model, str) else None,
                model_id=self.model_name if self.model_name != 'chatgpt' else 'gpt-3.5-turbo',
                prompter=self.prompter,
                prompt_format=self.prompt_format
            )
        elif any(fam in self.model_name for fam in ('llama3', 'mistral', 'zephyr', 'llama2', 'qwen', 'phi', 'gemma')):
            family = 'qwen' if 'qwen' in self.model_name else (
                'phi' if 'phi' in self.model_name else (
                    'gemma' if 'gemma' in self.model_name else (
                        'llama3' if 'llama3' in self.model_name else (
                            'mistral' if 'mistral' in self.model_name else (
                                'zephyr' if 'zephyr' in self.model_name else 'llama2'
                            )
                        )
                    )
                )
            )
            return _CausalLMAdapter(
                model=self.model,
                tokenizer=self.tokenizer,
                model_family=family,
                prompter=self.prompter,
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
