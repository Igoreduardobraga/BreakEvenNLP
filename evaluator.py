"""
Deep ModelEvaluator module for BreakEvenNLP.
Encapsulates model generation, prompt formatting, and result parsing across model architectures.
"""

import re
import copy
import os
import pickle
import time
import weakref
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
    spans = {}
    for idx, cls in enumerate(classes):
        cls_pat = r'\b' + re.escape(str(cls).lower()) + r'\b'
        found = [m.span() for m in re.finditer(cls_pat, t)]
        if found:
            spans[idx] = found

    # A match nested in a longer class name does not count: "not entailment" is not "entailment".
    def _inside(span, other):
        return other[0] <= span[0] and span[1] <= other[1] and span != other

    for idx, found in spans.items():
        nested = all(
            any(_inside(s, o) for j, other in spans.items() if j != idx for o in other)
            for s in found
        )
        if not nested:
            candidates.add(idx)

    if len(candidates) == 1:
        return next(iter(candidates))
    return -1


def _per_sample(values):
    """Copy of per-sample metadata: a list, or for instruction tuning a dict
    with one list per evaluation mode (list() of it would keep only the keys)."""
    if values is None:
        return []
    return dict(values) if isinstance(values, dict) else list(values)


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
        instance.prompts = _per_sample(prompts)
        instance.inputs = _per_sample(inputs)
        return instance


_TOKEN_SURFACES = weakref.WeakKeyDictionary()


def _token_surfaces(tokenizer):
    """Every token's text as tokenizer.decode renders it at the start of the
    answer and after another token, as {text: [ids]} maps. They differ for
    SentencePiece, whose decode drops the leading space of the first token."""
    if tokenizer not in _TOKEN_SURFACES:
        ids = list(range(len(tokenizer)))
        anchor = tokenizer("a", add_special_tokens=False)["input_ids"][-1:]
        head = tokenizer.decode(anchor)
        start, after = {}, {}
        for tid, first, following in zip(ids, tokenizer.batch_decode([[i] for i in ids]),
                                         tokenizer.batch_decode([anchor + [i] for i in ids])):
            start.setdefault(first, []).append(tid)
            if following.startswith(head):
                after.setdefault(following[len(head):], []).append(tid)
        _TOKEN_SURFACES[tokenizer] = start, after
    return _TOKEN_SURFACES[tokenizer]


class _ChoiceConstraint:
    """Greedy decoding restricted to the class names on HF, the counterpart of
    vLLM's guided choice: a token is allowed only if the answer stays a prefix
    of some class name, and the answer may end (end_ids) only once it is a
    whole class name."""

    def __init__(self, tokenizer, classes, end_ids):
        self.tokenizer = tokenizer
        self.classes = [str(c) for c in classes]
        self.end_ids = list(end_ids)
        self.start, self.after = _token_surfaces(tokenizer)
        self._allowed = {}

    def allowed(self, answer_ids):
        text = self.tokenizer.decode(answer_ids) if answer_ids else ""
        key = (text, not answer_ids)
        if key not in self._allowed:
            surfaces = self.after if answer_ids else self.start
            allowed = set(self.end_ids) if text in self.classes else set()
            for name in self.classes:
                if name.startswith(text):
                    rest = name[len(text):]
                    for k in range(1, len(rest) + 1):
                        allowed.update(surfaces.get(rest[:k], ()))
            # A finished answer (followed by the end token or padding) may only end.
            self._allowed[key] = sorted(allowed) or self.end_ids
        return self._allowed[key]

    def prefix_allowed_tokens_fn(self, prompt_len):
        """For generate(): the answer is what follows the first prompt_len ids."""
        return lambda batch_id, ids: self.allowed(ids[prompt_len:].tolist())


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

    def evaluate(self, dataset, mode='prompting', decoding='free'):
        golden = []
        predicted = []
        decodeds = []
        prompts = []
        inputs = []

        shots = getattr(dataset, 'context_samples', None) if mode == 'icl' else None
        dataset_name = getattr(dataset, 'dataset_name', 'sst2')
        custom_inst = getattr(dataset, 'instructions', None)
        guided = {}
        if decoding == 'guided':
            # The decoder starts from its start token; the answer follows it.
            constraint = _ChoiceConstraint(self.tokenizer, dataset.classes, [self.tokenizer.eos_token_id])
            guided['prefix_allowed_tokens_fn'] = constraint.prefix_allowed_tokens_fn(1)

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

            encoded = self.tokenizer(batch_prompts, return_tensors='pt', padding='longest', truncation=False)
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
                num_beams=1,
                **guided
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

    MAX_BATCH_TOKENS = 8192

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

    def evaluate(self, dataset, mode='prompting', decoding='free'):
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
            unk_id = getattr(self.tokenizer, 'unk_token_id', None)
            for t_token in ("<|eot_id|>", "<|im_end|>", "<|end|>", "<end_of_turn>", "<turn|>"):
                tid = self.tokenizer.convert_tokens_to_ids(t_token)
                if tid is not None and not isinstance(tid, str) and tid != unk_id and tid not in terminators:
                    terminators.append(tid)

        shots = getattr(dataset, 'context_samples', None) if mode == 'icl' else None
        dataset_name = getattr(dataset, 'dataset_name', 'sst2')
        custom_inst = getattr(dataset, 'instructions', None)
        constraint = _ChoiceConstraint(self.tokenizer, dataset.classes, terminators) if decoding == 'guided' else None

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

        for data, labels in dataset.batch_data_for_evaluation(self.batch_size):
            batch_prompts = []
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
                batch_prompts.append(prompt_str)

            inputs.extend(data)
            prompts.extend(batch_prompts)
            for decoded_text in self._generate(batch_prompts, gen_kwargs, constraint):
                decoded_clean = decoded_text.replace('.', '').replace('"', '').strip()
                decodeds.append(decoded_clean)
                predicted.append(_parse_results(decoded_clean, dataset.classes))

            golden.extend(labels)

        return EvaluationResult(golden, predicted, decodeds, prompts=prompts, inputs=inputs)

    def _generate(self, prompt_strs, gen_kwargs, constraint):
        """Answers in prompt order. Each prompt is tokenized alone, as at batch
        size 1, and the rows are left-padded, so batching only adds positions
        the attention mask hides."""
        rows = []
        for prompt_str in prompt_strs:
            encoded = self.tokenizer(prompt_str, return_tensors="pt", padding=True, truncation=True)
            ids = (encoded.input_ids if hasattr(encoded, 'input_ids') else encoded)[0]
            rows.append(ids.tolist() if hasattr(ids, 'tolist') else list(ids))

        answers, chunk = [], []
        for row in rows:
            width = max(len(r) for r in chunk + [row])
            if chunk and (len(chunk) + 1) * width > self.MAX_BATCH_TOKENS:
                answers.extend(self._generate_rows(chunk, gen_kwargs, constraint))
                chunk = []
            chunk.append(row)
        if chunk:
            answers.extend(self._generate_rows(chunk, gen_kwargs, constraint))
        return answers

    def _generate_rows(self, rows, gen_kwargs, constraint):
        import torch
        width = max(len(r) for r in rows)
        pad_id = gen_kwargs.get('pad_token_id')
        pad_id = (pad_id[0] if isinstance(pad_id, list) else pad_id) or 0
        input_ids = torch.tensor([[pad_id] * (width - len(r)) + r for r in rows], device=self.device)
        attention_mask = torch.tensor([[0] * (width - len(r)) + [1] * len(r) for r in rows], device=self.device)
        kwargs = dict(gen_kwargs)
        if constraint is not None:
            # Left padding puts every answer right after the same width.
            kwargs['prefix_allowed_tokens_fn'] = constraint.prefix_allowed_tokens_fn(width)
        try:
            out = self.model.generate(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        except torch.cuda.OutOfMemoryError:
            if len(rows) == 1:
                raise
            out = None
        if out is None:
            # Retried outside the except block, so the failed call's tensors are freed.
            torch.cuda.empty_cache()
            half = len(rows) // 2
            return (self._generate_rows(rows[:half], gen_kwargs, constraint)
                    + self._generate_rows(rows[half:], gen_kwargs, constraint))
        return [self.tokenizer.decode(seq[width:], skip_special_tokens=True).strip() for seq in out]


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


class _VLLMAdapter:
    """Offline vLLM adapter for causal models (Q6: em-processo, mesma costura).

    Requires `vllm` + `vllm-bnb-plugin` (optional group: `uv sync --group vllm`).
    Greedy decoding (temperature=0) with vLLM-level seed preserves the
    determinism contract; parity with the HF path is gated by
    tests/test_vllm_parity.py (Q2/Q9).
    """

    BUDGETS = {
        'llama2': {'gpu_memory_utilization': 0.85, 'max_model_len': 4096, 'max_num_seqs': 16},
        'llama3': {'gpu_memory_utilization': 0.85, 'max_model_len': 8192, 'max_num_seqs': 32},
        'mistral': {'gpu_memory_utilization': 0.85, 'max_model_len': 8192, 'max_num_seqs': 32},
        'zephyr': {'gpu_memory_utilization': 0.85, 'max_model_len': 8192, 'max_num_seqs': 32},
        'qwen': {'gpu_memory_utilization': 0.85, 'max_model_len': 8192, 'max_num_seqs': 32},
        'phi': {'gpu_memory_utilization': 0.85, 'max_model_len': 8192, 'max_num_seqs': 32},
        'gemma': {'gpu_memory_utilization': 0.90, 'max_model_len': 6144, 'max_num_seqs': 8},
    }

    QUANT_FORMAT = "bitsandbytes"

    def __init__(self, model_id, model_family, prompter=None, prompt_format=0,
                 max_new_tokens=10, seed=0, quantization="bitsandbytes",
                 budget_override=None):
        try:
            from vllm import LLM
        except ImportError as e:
            raise ImportError(
                "vLLM engine requires 'vllm' and 'vllm-bnb-plugin' "
                "(on the 4090 machine: .venv/bin/pip install "
                "'vllm>=0.11' vllm-bnb-plugin)."
            ) from e
        from vllm import LLM
        self.model_id = model_id
        self.model_family = model_family
        self.max_new_tokens = max_new_tokens
        self.seed = seed
        self.quantization = quantization
        self.quant_format = self.QUANT_FORMAT if quantization else "none"
        budget = dict(self.BUDGETS.get(model_family, self.BUDGETS['llama3']))
        if budget_override:
            budget.update(budget_override)
        self.budget = budget
        self.llm = LLM(
            model=model_id,
            quantization=quantization,
            dtype="bfloat16",
            seed=seed,
            **budget
        )
        self.prompter = prompter or PromptFormatter(model_name=model_family, prompt_format=prompt_format)
        try:
            self.prompter.tokenizer = self.llm.get_tokenizer()
        except Exception:
            pass

    def _build_sampling_params(self, dataset, decoding: str, seed=None):
        """Builds SamplingParams honouring guided decoding across vLLM versions.

        vLLM >= 0.10 removed the `guided_choice` SamplingParams kwarg in favour
        of `structured_outputs=StructuredOutputsParams(choice=[...])`.
        """
        from vllm import SamplingParams
        import inspect
        common = dict(
            temperature=0.0,
            max_tokens=self.max_new_tokens,
            seed=self.seed if seed is None else seed,
        )
        if decoding != 'guided':
            return SamplingParams(**common)

        if 'guided_choice' in inspect.signature(SamplingParams).parameters:
            return SamplingParams(guided_choice=list(dataset.classes), **common)
        from vllm.sampling_params import StructuredOutputsParams
        return SamplingParams(
            structured_outputs=StructuredOutputsParams(choice=list(dataset.classes)),
            **common,
        )

    def evaluate(self, dataset, mode='prompting', decoding='free', seed=None):
        if decoding not in ('free', 'guided'):
            raise ValueError(f"Unknown decoding '{decoding}'. Use 'free' or 'guided'.")
        shots = getattr(dataset, 'context_samples', None) if mode == 'icl' else None
        dataset_name = getattr(dataset, 'dataset_name', 'sst2')
        custom_inst = getattr(dataset, 'instructions', None)

        test_texts, test_labels = [], []
        for data, labels in dataset.batch_data_for_evaluation(1):
            test_texts.extend(data)
            test_labels.extend(labels)

        params = self._build_sampling_params(dataset, decoding, seed=seed)
        prompts = [
            self._to_string(self.prompter.format(
                sample, dataset_name=dataset_name, classes=dataset.classes,
                shots=shots, custom_instruction=custom_inst))
            for sample in test_texts
        ]
        outputs = self.llm.generate(prompts, params)

        golden, predicted, decodeds = [], [], []
        for sample, label, out in zip(test_texts, test_labels, outputs):
            decoded = out.outputs[0].text.strip()
            decodeds.append(decoded)
            predicted.append(_parse_results(decoded, dataset.classes))
            golden.append(label)
        return EvaluationResult(golden, predicted, decodeds, prompts=prompts, inputs=test_texts)

    def close(self):
        """Shuts the engine core down explicitly; left to interpreter
        finalization, a long-lived engine hangs the process at exit."""
        engine = getattr(getattr(self.llm, 'llm_engine', None), 'engine_core', None)
        if engine is not None and hasattr(engine, 'shutdown'):
            engine.shutdown()

    @staticmethod
    def _to_string(formatted):
        if isinstance(formatted, str):
            return formatted
        return str(formatted)


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
        device: str = "cuda",
        engine: str = "hf",
        seed: int = 0,
        vllm_kwargs=None
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.model_name = str(model_name).lower()
        self.batch_size = batch_size
        self.prompt_format = prompt_format
        self.max_new_tokens = max_new_tokens
        self.generation_config = generation_config
        self.device = device
        self.engine = engine
        self.seed = seed
        self.vllm_kwargs = vllm_kwargs or {}
        if engine not in ('hf', 'vllm'):
            raise ValueError(f"Unknown engine '{engine}'. Use 'hf' or 'vllm'.")
        self.prompter = PromptFormatter(model_name=self.model_name, prompt_format=self.prompt_format, tokenizer=self.tokenizer)

        self.adapter = self._resolve_adapter()

    @property
    def quant_format(self) -> str:
        if self.engine == 'vllm':
            return getattr(self.adapter, 'quant_format', 'bitsandbytes')
        if 'flan-t5' in self.model_name:
            return 'fp32'
        if 'chatgpt' in self.model_name:
            return 'api'
        return 'bnb-4bit-nf4'

    def close(self):
        """Releases engine resources (vLLM only; no-op for the other adapters)."""
        if hasattr(self.adapter, 'close'):
            self.adapter.close()

    def _resolve_adapter(self):
        if self.engine == 'vllm':
            family = self._causal_family()
            if family is None:
                raise NotImplementedError(
                    f"vLLM engine supports causal families only; '{self.model_name}' "
                    "must use engine='hf'."
                )
            model_id = self.model if isinstance(self.model, str) else self.model_name
            return _VLLMAdapter(
                model_id=model_id,
                model_family=family,
                prompter=self.prompter,
                prompt_format=self.prompt_format,
                max_new_tokens=self.max_new_tokens,
                seed=self.seed,
                **self.vllm_kwargs
            )
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
        elif self._causal_family() is not None:
            family = self._causal_family()
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

    @staticmethod
    def _causal_family_for(name: str):
        for fam in ('qwen', 'phi', 'gemma', 'llama3', 'mistral', 'zephyr', 'llama2'):
            if fam in name:
                return fam
        return None

    def _causal_family(self):
        return self._causal_family_for(self.model_name)

    def evaluate(self, dataset, mode: str = 'prompting', partial_save_path=None, decoding: str = 'free',
                 seed=None):
        """Evaluates the model on the provided dataset.

        Args:
            dataset: A dataset object providing `batch_data_for_evaluation`, `classes`, and prompt metadata.
            mode: Operating mode ('prompting' for zero-shot, 'icl' for in-context learning).
            partial_save_path: Optional path to save and resume partial results (for API models).
            decoding: 'free' (unconstrained) or 'guided' (answer restricted to the class names:
                vLLM structured outputs, or token masking on HF).
            seed: Per-call sampling seed (vLLM only), so one engine can serve many folds.

        Returns:
            Tuple of (golden_labels, predicted_labels, decoded_texts).
        """
        if mode not in ('prompting', 'icl'):
            raise ValueError(f"Unknown evaluation mode '{mode}'. Supported modes: 'prompting', 'icl'.")
        if decoding not in ('free', 'guided'):
            raise ValueError(f"Unknown decoding '{decoding}'. Use 'free' or 'guided'.")
        if decoding == 'guided' and isinstance(self.adapter, _APIAdapter):
            raise ValueError("Guided decoding is not available for API models.")

        if isinstance(self.adapter, _APIAdapter):
            return self.adapter.evaluate(dataset, mode=mode, partial_save_path=partial_save_path)
        if isinstance(self.adapter, _VLLMAdapter):
            return self.adapter.evaluate(dataset, mode=mode, decoding=decoding, seed=seed)
        return self.adapter.evaluate(dataset, mode=mode, decoding=decoding)
