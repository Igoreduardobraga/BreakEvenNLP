# main.py

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments, get_linear_schedule_with_warmup, EarlyStoppingCallback
from datasets import Dataset
from data import ICLDataset, FineTuningDataset, DatasetLoader, PromptDataset, SimilarityICLDataset, InstructionTuningDataset, TextDataset, load_text_and_targets, SeededRandomSampler
from transfer_learning.models import BERTBase, RoBERTaBase
import re
import random
import pickle
import argparse
import torch
import numpy as np
import os
import copy
import json
from peft import LoraConfig, PeftModelForCausalLM, prepare_model_for_kbit_training
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from torch.utils.data import DataLoader
import time
import shutil
import torch.nn.functional as F


from trl import SFTTrainer, DataCollatorForCompletionOnlyLM, SFTConfig

def compute_macro_f1(golden, predicted, failed_label=-1):
    total_samples = len(predicted)
    failed_preds = predicted.count(failed_label)
    
    if total_samples > 0:
        failure_rate = (failed_preds / total_samples) * 100
        print(f"\n[METRICS LOG] Total Amostras: {total_samples} | Falhas de Parsing (-1): {failed_preds} ({failure_rate:.2f}%)")
    
    y_true = np.array(golden)
    y_pred = np.array(predicted)
    valid_labels = np.unique(y_true)
    
    score = f1_score(
        y_true=y_true, 
        y_pred=y_pred, 
        average='macro', 
        labels=valid_labels,
        zero_division=0
    )
    
    return score

def parse_results(text, classes):
    t = text.strip().lower()
    candidates = set()

    # Numeric response: try to find a number 1..N
    m = re.match(r'^\s*(\d{1,2})\s*\)?\s*$', t)
    if m:
        k = int(m.group(1)) - 1
        if 0 <= k < len(classes):
            candidates.add(k)

    # Textual response: try to find the name of the class
    for idx, cls in enumerate(classes):
        cls_pat = r'\b' + re.escape(cls.lower()) + r'\b'
        if re.search(cls_pat, t):
            candidates.add(idx)

    # Returns the index found
    if len(candidates) == 1:
        return next(iter(candidates))
    return -1

def prepare_flan_t5_prompt(dataset, test_data):
    instructions = dataset.instructions
    context_samples = dataset.context_samples
    if PROMPT_FORMAT == 0:
        prompt = ''
        prompt += f'{instructions["instruction"]}\n'
    elif DATASET == 'snips' and PROMPT_FORMAT == 3:
        prompt = ''
    else:
        prompt = ''

    final_prompts = []
    for sample in test_data:
        new_prompt = copy.deepcopy(prompt)
        if PROMPT_FORMAT == 0:
            new_prompt += f'{instructions["sentence_start"]}: {sample.strip()}\n{instructions["answer_start"]}: '
        elif DATASET == 'snips' and PROMPT_FORMAT == 3:
            new_prompt += f'User: {sample.strip()}\n{instructions["instruction"]} '
        else:
            new_prompt += f'{sample.strip()}\n{instructions["instruction"]} '
        final_prompts.append(new_prompt)

    return final_prompts


def prepare_flan_t5_icl(dataset, test_data):
    instructions = dataset.instructions
    context_samples = dataset.context_samples
    if PROMPT_FORMAT == 0:
        prompt = ''
        prompt += f'{instructions["instruction"]}\n'
        for sample in context_samples:
            prompt += f'{instructions["sentence_start"]}: {sample[0].strip()}\n{instructions["answer_start"]}: {sample[1].strip()}\n'
    elif DATASET == 'snips' and PROMPT_FORMAT == 3:
        prompt = ''
        for sample in context_samples:
            prompt += f'User: {sample[0].strip()}\n{instructions["instruction"]} {sample[1].strip()}\n'
    else:
        prompt = ''
        for sample in context_samples:
            prompt += f'{sample[0].strip()}\n{instructions["instruction"]} {sample[1].strip()}\n'


    final_prompts = []
    for sample in test_data:
        new_prompt = copy.deepcopy(prompt)
        if PROMPT_FORMAT == 0:
            new_prompt += f'{instructions["sentence_start"]}: {sample.strip()}\n{instructions["answer_start"]}: '
        elif DATASET == 'snips' and PROMPT_FORMAT == 3:
            new_prompt += f'User: {sample.strip()}\n{instructions["instruction"]} '
        else:
            new_prompt += f'{sample.strip()}\n{instructions["instruction"]} '
        final_prompts.append(new_prompt)

    return final_prompts


def run_flan_t5(dataset, model, tokenizer, mode):
    
    golden = []
    predicted = []
    decodeds = []
    for data, labels in dataset.batch_data_for_evaluation(BATCH_SIZE):
        final_prompts = prepare_flan_t5_icl(dataset, data) if mode == 'icl' else prepare_flan_t5_prompt(dataset, data)
        
        encoded = tokenizer(final_prompts, return_tensors='pt', padding='longest', truncation=True).to('cuda')
        out = model.generate(**encoded, max_new_tokens=10, do_sample=False, num_beams=1)
        decoded = tokenizer.batch_decode(out, skip_special_tokens=True)

        decodeds.extend(decoded)
        
        predicted_labels = []
        for text in decoded:
            pred = parse_results(text, dataset.classes)
            predicted_labels.append(pred)

        predicted.extend(predicted_labels)
        golden.extend(labels)
    return golden, predicted, decodeds


def prepare_llama2_prompt(dataset, test_data):
    instructions = dataset.instructions
    context_samples = dataset.context_samples
    if PROMPT_FORMAT == 0:
        prompt = f'<s>[INST] <<SYS>>\nYou are a helpful assistant that will follow every instruction from the user\n<</SYS>>\n\n{instructions["instruction"]} [/INST] Ok, I will determine the {instructions["task_type"]} of the Sentences you will give me using only the options provided! </s>'
    else:
        prompt = ''

    final_prompts = []
    for sample in test_data:
        new_prompt = copy.deepcopy(prompt)
        if PROMPT_FORMAT == 0:
            new_prompt += f'<s>[INST] {instructions["sentence_start"]}: {sample.strip()} [/INST] {instructions["answer_start"]}: '
        else:
            prompt += f'<s>[INST] {sample.strip()}; {instructions["instruction"]} '
        final_prompts.append(new_prompt)

    return final_prompts


def prepare_llama2_icl(dataset, test_data):
    instructions = dataset.instructions
    context_samples = dataset.context_samples
    if PROMPT_FORMAT == 0:
        prompt = f'<s>[INST] <<SYS>>\nYou are a helpful assistant that will follow every instruction from the user\n<</SYS>>\n\n{instructions["instruction"]} [/INST] Ok, I will determine the {instructions["task_type"]} of the Sentences you will give me using only the options provided! </s>'
        for sample in context_samples:
            prompt += f'<s>[INST] {instructions["sentence_start"]}: {sample[0].strip()} [/INST] {instructions["answer_start"]}: {sample[1].strip()} </s>'
    else:
        prompt = ''
        for sample in context_samples:
            prompt += f'<s>[INST] {sample[0].strip()}; {instructions["instruction"]} [/INST] {sample[1].strip()} </s>'

    final_prompts = []
    for sample in test_data:
        new_prompt = copy.deepcopy(prompt)
        if PROMPT_FORMAT == 0:
            new_prompt += f'<s>[INST] {instructions["sentence_start"]}: {sample.strip()} [/INST] {instructions["answer_start"]}: '
        else:
            prompt += f'<s>[INST] {sample.strip()}; {instructions["instruction"]} '
        final_prompts.append(new_prompt)

    return final_prompts


def run_llama2(dataset, model, tokenizer, mode):
    golden = []
    predicted = []
    decodeds = []
    for data, labels in dataset.batch_data_for_evaluation(BATCH_SIZE):
        final_prompts = prepare_llama2_icl(dataset, data) if mode == 'icl' else prepare_llama2_prompt(dataset, data)

        encoded = tokenizer(final_prompts, return_tensors='pt', padding='longest').to('cuda')
        out = model.generate(**encoded, max_new_tokens=10, do_sample=False, num_beams=1, generation_config=generation_config)
        decoded = tokenizer.batch_decode(out, skip_special_tokens=True)

        decodeds.extend(decoded)
        
        predicted_labels = []
        for text in decoded:
            text = text.split('[/INST]')[-1].lower()
            pred = parse_results(text, dataset.classes)
            predicted_labels.append(pred)

        predicted.extend(predicted_labels)
        golden.extend(labels)
    return golden, predicted, decodeds


def run_llama2_optimized(dataset, model, tokenizer, mode):
    instructions = dataset.instructions
    context_samples = dataset.context_samples
    
    sys_msg = (
        f"You are a helpful assistant. Your task is {instructions['task_type']}. "
        f"Answer ONLY with one of the valid class names. Do not explain."
    )
    
    # Format: <s>[INST] <<SYS>>\n{sys}\n<</SYS>>\n\n{user} [/INST]
    prompt_history = ""
    if mode == 'icl':
        for s in context_samples:
            prompt_history += f"<s>[INST] {s[0]} [/INST] {s[1]} </s>"

    golden = []
    predicted = []
    decodeds = []

    for data, labels in dataset.batch_data_for_evaluation(BATCH_SIZE):
        batch_prompts = []
        
        for sample in data:
            if PROMPT_FORMAT == 0:
                user_query = f"{instructions['instruction']}\n\nInput: {sample}\nAnswer:"
            else:
                user_query = f"{sample}\n{instructions['instruction']}"
            
            if mode == 'icl' and len(context_samples) > 0:
                final_prompt = f"<s>[INST] <<SYS>>\n{sys_msg}\n<</SYS>>\n\n{prompt_history}{user_query} [/INST]"
            else:
                final_prompt = f"<s>[INST] <<SYS>>\n{sys_msg}\n<</SYS>>\n\n{user_query} [/INST]"
            
            batch_prompts.append(final_prompt)

        inputs = tokenizer(batch_prompts, return_tensors='pt', padding=True, truncation=True).to('cuda')
        
        out = model.generate(
            **inputs, 
            max_new_tokens=10, 
            do_sample=False,
            temperature=None,
            top_p=None
        )
        
        for i, generated_ids in enumerate(out):
            input_len = inputs.input_ids[i].shape[0]
            new_tokens = generated_ids[input_len:]
            text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            
            decodeds.append(text)
            
            pred = parse_results(text, dataset.classes)
            predicted.append(pred)
            
        golden.extend(labels)
        
    return golden, predicted, decodeds

def run_llama3(dataset, model, tokenizer, mode):
    instructions = dataset.instructions
    context_samples = dataset.context_samples

    options_str = ", ".join([f"'{c}'" for c in dataset.classes])

    system_message = {
        "role": "system", 
        "content": (
            f"You are an expert classifier. Your task is to determine the {instructions['task_type']}. "
            f"You must answer with exactly one of these options: {options_str}. "
            f"Do not explain. Do not output the number, only the class name."
        )
    }

    messages_prefix = [system_message]
    
    if mode == 'icl':
        for s in context_samples:
            if PROMPT_FORMAT == 0:
                user_content = f'{instructions["sentence_start"]}: {s[0]}'
            else:
                user_content = f'{s[0]} {instructions["instruction"]}'
            
            messages_prefix.append({'role': 'user', 'content': user_content})
            messages_prefix.append({'role': 'assistant', 'content': s[1]})

    golden, predicted, decodeds = [], [], []

    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    debug_count = 0 

    for data, labels in dataset.batch_data_for_evaluation(1):
        for sample in data:
            msgs = copy.deepcopy(messages_prefix)

            if PROMPT_FORMAT == 0:
                content = f"Text: {sample}\nBased on the text above, determine the {instructions['task_type']}. Answer:"
            else:
                content = f'{sample} {instructions["instruction"]} '
            
            msgs.append({'role': 'user', 'content': content})

            prompt_str = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            
            inputs = tokenizer(prompt_str, return_tensors="pt", padding=True, truncation=True).to('cuda')

            out = model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=10,
                eos_token_id=terminators,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=False
            )
            
            response_ids = out[0][inputs.input_ids.shape[-1]:]
            decoded_text = tokenizer.decode(response_ids, skip_special_tokens=True).strip()
            
            decoded_clean = decoded_text.replace('.', '').replace('"', '').strip()

            decodeds.append(decoded_clean)
            
            if debug_count < 5:
                print(f"\n--- DEBUG [{debug_count}] ---")
                print(f"Input: ...{str(sample)[-50:]}")
                print(f"Output Modelo: '{decoded_clean}'")
                debug_count += 1

            pred = parse_results(decoded_clean, dataset.classes)
            predicted.append(pred)
            golden.extend(labels)

    return golden, predicted, decodeds


def run_mistral(dataset, model, tokenizer, mode):
    instructions = dataset.instructions
    context_samples = dataset.context_samples

    # Prefix Construction (Instructions + Examples)
    instruction_text = f"{instructions['instruction']} Do not explain. Answer only with the class name."
    
    messages_prefix = []
    
    if mode == 'icl':
        messages_prefix.append({'role': 'user', 'content': instruction_text})
        messages_prefix.append({'role': 'assistant', 'content': 'Understood.'})

        for s in context_samples:
            
            if PROMPT_FORMAT == 0:
                content = f"{instructions['sentence_start']}: {s[0]}"
            else:
                content = f"{s[0]} {instructions['instruction']}"

            messages_prefix.append({'role': 'user', 'content': content})
            messages_prefix.append({'role': 'assistant', 'content': s[1]})
    else:
        pass

    golden, predicted, decodeds = [], [], []

    for data, labels in dataset.batch_data_for_evaluation(1):
        for sample in data:
            msgs = copy.deepcopy(messages_prefix)
            
            if PROMPT_FORMAT == 0:
                if mode == 'prompting':
                     content = f"{instruction_text}\n\n{instructions['sentence_start']}: {sample}"
                else:
                     content = f"{instructions['sentence_start']}: {sample}"
            else:
                content = f"{sample} {instructions['instruction']}"

            msgs.append({'role': 'user', 'content': content})

            inputs = tokenizer.apply_chat_template(
                msgs, 
                return_tensors="pt", 
                add_generation_prompt=True,
                tokenize=True
            ).to('cuda')

            out = model.generate(
                inputs,
                max_new_tokens=10, # Short to capture only the class name
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
            
            decoded = tokenizer.decode(out[0][inputs.shape[-1]:], skip_special_tokens=True).strip()
            
            decodeds.append(decoded)
            
            pred = parse_results(decoded, dataset.classes)
            predicted.append(pred)
            
            golden.extend(labels)

    return golden, predicted, decodeds


def run_zephyr(dataset, model, tokenizer, mode):
    instructions = dataset.instructions
    context_samples = dataset.context_samples

    instruction_text = f"{instructions['instruction']} Answer with the class name only."

    messages_prefix = []
    
    messages_prefix.append({'role': 'system', 'content': instruction_text})

    if mode == 'icl':
        for s in context_samples:
            if PROMPT_FORMAT == 0:
                user_content = f"{instructions['sentence_start']}: {s[0]}"
            else:
                user_content = f"{s[0]} {instructions['instruction']}"
            
            messages_prefix.append({'role': 'user', 'content': user_content})
            messages_prefix.append({'role': 'assistant', 'content': s[1]})

    golden, predicted = [], []
    decodeds = []

    for data, labels in dataset.batch_data_for_evaluation(1):
        for sample in data:
            msgs = copy.deepcopy(messages_prefix)
            
            if PROMPT_FORMAT == 0:
                content = f"{instructions['sentence_start']}: {sample}"
            else:
                content = f"{sample} {instructions['instruction']}"
                
            msgs.append({'role': 'user', 'content': content})

            inputs = tokenizer.apply_chat_template(
                msgs, 
                return_tensors="pt", 
                add_generation_prompt=True
            ).to('cuda')

            out = model.generate(
                inputs, 
                max_new_tokens=10, 
                do_sample=False, 
                pad_token_id=tokenizer.eos_token_id
            )
            
            decoded = tokenizer.decode(out[0][inputs.shape[-1]:], skip_special_tokens=True).strip()
            decodeds.append(decoded)

            pred = parse_results(decoded, dataset.classes)
            predicted.append(pred)
            
            golden.extend(labels)

    return golden, predicted, decodeds


def prepare_chatgpt_prompt(dataset, test_data):
    instructions = dataset.instructions
    context_samples = dataset.context_samples
    sample = test_data
    if PROMPT_FORMAT == 0:
        prompt = ''
        prompt += f'{instructions["instruction"]}\n{instructions["sentence_start"]}: {sample.strip()}\n{instructions["answer_start"]}: '
    elif DATASET == 'snips' and PROMPT_FORMAT == 3:
        prompt = f'User: {sample.strip()}\n{instructions["instruction"]} '
    else:
        prompt = f'{sample.strip()}\n{instructions["instruction"]} '
    
    return prompt


def prepare_chatgpt_icl(dataset, test_data):
    instructions = dataset.instructions
    context_samples = dataset.context_samples
    sample = test_data
    if PROMPT_FORMAT == 0:
        prompt = ''
        prompt += f'{instructions["instruction"]}\n'
        for sample in context_samples:
            prompt += f'{instructions["sentence_start"]}: {sample[0].strip()}\n{instructions["answer_start"]}: {sample[1].strip()}\n'
        prompt += f'{instructions["sentence_start"]}: {sample.strip()}\n{instructions["answer_start"]}: '
    elif DATASET == 'snips' and PROMPT_FORMAT == 3:
        prompt = ''
        for sample in context_samples:
            prompt += f'User: {sample[0].strip()}\n{instructions["instruction"]} {sample[1].strip()}\n'
        prompt += f'User: {sample.strip()}\n{instructions["instruction"]} '
    else:
        prompt = ''
        for sample in context_samples:
            prompt += f'{sample[0].strip()}\n{instructions["instruction"]} {sample[1].strip()}\n'
        prompt += f'{sample.strip()}\n{instructions["instruction"]} '

    return prompt

def run_chatgpt(dataset, investigation_path):
    import openai
    api_key = os.environ['OPEN_AI_KEY']
    organization = os.environ['OPEN_AI_ORGANISATION']

    client = openai.OpenAI(api_key=api_key, organization=organization)

    partial_save_path = os.path.join(investigation_path, 'partial')
    if not os.path.exists(partial_save_path):
        os.makedirs(partial_save_path)

    golden = []
    predicted = []
    decodeds = []
    idx = 0
    processed = len(os.listdir(partial_save_path))
    for data, labels in dataset.batch_data_for_evaluation(1):
        if idx < processed:
            print(f'Skipping idx: {idx}')
            idx += 1
            continue

        if idx >= len(dataset):
            print('Processed whole dataset')
            break
        print(f'Running idx {idx}!')
        prompt = prepare_chatgpt_icl(dataset, data[0]) if EXPERIMENT_TYPE == 'icl' else prepare_chatgpt_prompt(dataset, data[0])
        print(prompt)

        def request_with_checks(prompt):
            success = False
            count = 0
            while not success:
                if count > 0:
                    print(f'Retrying again. Current number of retries: {count}')
                if count >= 5:
                    raise Exception('Too many attempts')
                try:
                    time.sleep(0.5)
                    response = client.chat.completions.create(
                        model='gpt-3.5-turbo',
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant that follows all the instructions."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0,
                        max_tokens=30,
                    )
                    success = True
                    break
                except Exception as e:
                    print(e)
                    time.sleep(5)
                    count += 1
            return response

        response = request_with_checks(prompt)
        decoded = response.choices[0].message.content 
        result = {'predicted': decoded, 'real': labels[0]}
        with open(os.path.join(partial_save_path, f'{idx}.pkl'), 'wb') as file:
            pickle.dump(result, file)
        idx += 1

    for idx, label in enumerate(dataset.test_targets):

        with open(os.path.join(partial_save_path, f'{idx}.pkl'), 'rb') as file:
            result = pickle.load(file) 

        decoded = result['predicted']
        predicted_label = parse_results(decoded, dataset.classes)
        predicted.append(predicted_label)
        golden.append(label)
        decodeds.append(decoded)
    
    return golden, predicted, decodeds



def prompt_icl_experiment(randomness_factor_seeds, model, tokenizer, experiment='icl',
                          investigation_path=None, train_test_indices=None):
    if 'icl' in experiment:
        dataset_constr = SimilarityICLDataset if experiment == 'icl_similarity' else ICLDataset 
        dataset = dataset_constr(
            dataset_name=DATASET,
            train_size=args.train_size,
            num_labelled=args.num_labelled,
            num_labelled_test=args.num_labelled_test,
            label_seed=randomness_factor_seeds['label_choice'],
            device=device,
            full_test=FULL_TEST,
            num_shots=args.num_shots,
            num_classes=args.num_classes,
            choice_seed=randomness_factor_seeds['sample_choice'],
            order_seed=randomness_factor_seeds['sample_order'],
            model_name=MODEL,
            prompt_format=PROMPT_FORMAT,
            train_test_indices=train_test_indices
        )
    else:
        dataset = PromptDataset(
            dataset_name=DATASET,
            train_size=args.train_size,
            num_labelled=args.num_labelled,
            num_labelled_test=args.num_labelled_test,
            label_seed=randomness_factor_seeds['label_choice'],
            device=device,
            full_test=FULL_TEST,
            model_name=MODEL,
            prompt_format=PROMPT_FORMAT,
            train_test_indices=train_test_indices
        )

    torch.manual_seed(randomness_factor_seeds['model_randomness'])    
    torch.cuda.manual_seed(randomness_factor_seeds['model_randomness'])
    torch.cuda.manual_seed_all(randomness_factor_seeds['model_randomness'])
    np.random.seed(randomness_factor_seeds['model_randomness'])
    random.seed(randomness_factor_seeds['model_randomness'])

    if MODEL == 'chatgpt':
        return run_chatgpt(dataset, investigation_path)
    else:
        return ICL_MODEL_RUN[f'{MODEL}_{MODEL_SIZE}'](dataset, model, tokenizer, experiment)


def prepare_instruction_tuning_flan_t5(dataset):
    instructions = dataset.instructions
    context_samples = dataset.context_samples
    if PROMPT_FORMAT == 0:
        prompt = ''
        prompt += f'{instructions["instruction"]}\n'
    elif DATASET == 'snips' and PROMPT_FORMAT == 3:
        prompt = ''
    else:
        prompt = ''

    final_prompts = []
    for sample in context_samples:
        new_prompt = copy.deepcopy(prompt)
        if PROMPT_FORMAT == 0:
            new_prompt += f'{instructions["sentence_start"]}: {sample[0].strip()}\n{instructions["answer_start"]}: {sample[1].strip()}'
        elif DATASET == 'snips' and PROMPT_FORMAT == 3:
            new_prompt += f'User: {sample[0].strip()}\n{instructions["instruction"]} {instructions["answer_start"]}: {sample[1].strip()}'
        else:
            new_prompt += f'{sample[0].strip()}\n{instructions["instruction"]} {instructions["answer_start"]}: {sample[1].strip()}'
        final_prompts.append(new_prompt)

    return final_prompts

def prepare_instruction_tuning_mistral(dataset):
    instructions = dataset.instructions
    context_samples = dataset.context_samples
    
    instruction_text = instructions["instruction"]
    
    final_prompts = []
    for sample in context_samples:
        
        if PROMPT_FORMAT == 0:
            user_text = f"{instruction_text}\n{instructions['sentence_start']}: {sample[0].strip()}"
        else:
            user_text = f"{sample[0].strip()} {instruction_text}"

        full_text = f"<s>[INST] {user_text} [/INST] {sample[1].strip()}</s>"
        final_prompts.append(full_text)

    return final_prompts

def prepare_instruction_tuning_zephyr(dataset):
    instructions = dataset.instructions
    context_samples = dataset.context_samples
    
    instruction_text = instructions["instruction"]

    final_prompts = []
    for sample in context_samples:
        if PROMPT_FORMAT == 0:
            user_text = f"{instructions['sentence_start']}: {sample[0].strip()}"
        else:
            user_text = f"{sample[0].strip()} {instructions['instruction']}"

        full_text = (
            f"<|system|>\n{instruction_text}</s>\n"
            f"<|user|>\n{user_text}</s>\n"
            f"<|assistant|>\n{sample[1].strip()}</s>"
        )
        final_prompts.append(full_text)

    return final_prompts

def prepare_instruction_tuning_llama3(dataset):
    instructions = dataset.instructions
    context_samples = dataset.context_samples
    
    system_header = "<|start_header_id|>system<|end_header_id|>\n\n"
    user_header = "<|start_header_id|>user<|end_header_id|>\n\n"
    assistant_header = "<|start_header_id|>assistant<|end_header_id|>\n\n"
    eot = "<|eot_id|>"
    
    sys_msg = f"You are a helpful assistant. Follow the instruction exactly. Determine the {instructions['task_type']}."
    
    final_prompts = []
    for sample in context_samples:
        if PROMPT_FORMAT == 0:
            user_input = f'{instructions["instruction"]}\n{instructions["sentence_start"]}: {sample[0].strip()}\n{instructions["answer_start"]}: '
        else:
            user_input = f'{sample[0].strip()} {instructions["instruction"]}'
            
        label_text = sample[1].strip()
        
        full_text = (
            f"{system_header}{sys_msg}{eot}"
            f"{user_header}{user_input}{eot}"
            f"{assistant_header}{label_text}{eot}"
        )
        final_prompts.append(full_text)

    return final_prompts


def instruction_tuning_experiment(randomness_factor_seeds, model_name, tokenizer, investigation_path,
                                  train_test_indices=None):
    dataset = InstructionTuningDataset(
        dataset_name=DATASET,
        train_size=args.train_size,
        num_labelled=args.num_labelled,
        num_labelled_test=args.num_labelled_test,
        label_seed=randomness_factor_seeds['label_choice'],
        device=device,
        full_test=FULL_TEST,
        model_name=MODEL,
        prompt_format=PROMPT_FORMAT,
        train_test_indices=train_test_indices
    )

    if MODEL == 'flan-t5':
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name).cuda()
        prompts = prepare_instruction_tuning_flan_t5(dataset)
        response_template = "Answer:"
    elif MODEL in ['mistral', 'zephyr', 'llama3']:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        
        target_modules = ['up_proj', 'down_proj', 'gate_proj', 'k_proj', 'q_proj', 'v_proj', 'o_proj']

        peft_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=target_modules
        )
        
        model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config, device_map='auto')
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'right'

        model = prepare_model_for_kbit_training(model)
        
        if MODEL == 'mistral':
            prompts = prepare_instruction_tuning_mistral(dataset)
            response_template = "[/INST]"
        elif MODEL == 'zephyr':
            prompts = prepare_instruction_tuning_zephyr(dataset)
            response_template = "<|assistant|>"
        elif MODEL == 'llama3':
            prompts = prepare_instruction_tuning_llama3(dataset)
            response_template = "<|start_header_id|>assistant<|end_header_id|>\n\n"
            
    all_idx = np.arange(len(prompts))
    train_idx, val_idx = train_test_split(
        all_idx, test_size=0.1,
        random_state=randomness_factor_seeds['sample_order'],
        shuffle=True
    )

    train_prompts = [prompts[i] for i in train_idx]
    train_labels  = [dataset.train_targets[i] for i in train_idx]
    val_prompts   = [prompts[i] for i in val_idx]
    val_labels    = [dataset.train_targets[i] for i in val_idx]
        
    if MODEL == 'flan-t5':
        steps_per_epoch_cap = 250
        max_examples_per_epoch = steps_per_epoch_cap * 4
        if len(train_prompts) > max_examples_per_epoch:
            rng = np.random.default_rng(randomness_factor_seeds['sample_order'])
            keep = rng.choice(len(train_prompts), size=max_examples_per_epoch, replace=False)
            keep.sort()
            train_prompts = [train_prompts[i] for i in keep]
            train_labels  = [train_labels[i]  for i in keep]
            
    train_ds = Dataset.from_dict({'prompt': train_prompts, 'label': train_labels})
    val_ds   = Dataset.from_dict({'prompt': val_prompts,   'label': val_labels})

    training_args = SFTConfig(
        output_dir=investigation_path,
        per_device_train_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        num_train_epochs=NUM_EPOCHS,
        logging_strategy="no",
        save_strategy="epoch", 
        eval_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        save_total_limit=1,
        greater_is_better=False,
        gradient_accumulation_steps=1,
        optim="paged_adamw_8bit" if MODEL in ['mistral', 'zephyr'] else 'adamw_torch',
        lr_scheduler_type="linear",
        warmup_ratio=0.1,
        dataset_text_field="prompt",
        max_seq_length=512,
        packing=False,
    )

    collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        tokenizer=tokenizer,
        peft_config=peft_config if MODEL in ['mistral', 'zephyr'] else None,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5, early_stopping_threshold=0.0)]
    )

    trainer.train()
    if MODEL in ['mistral', 'zephyr']:
       model = trainer.model.merge_and_unload()
    model.eval()
    
    if MODEL in ['mistral', 'zephyr']:
        tokenizer.padding_side = 'left'

    golden = {'prompting': None, 'icl': None}
    predicted = {'prompting': None, 'icl': None}
    decoded = {'prompting': None, 'icl': None}
    for key in ['prompting', 'icl']:
        golden[key], predicted[key], decoded[key] = prompt_icl_experiment(
            randomness_factor_seeds, model, tokenizer, key, investigation_path=investigation_path,
            train_test_indices=train_test_indices
        )
        score = compute_macro_f1(golden[key], predicted[key], failed_label=-1)
        print(score)
        with open(os.path.join(investigation_path, f'{key}_results.json'), 'w') as file:
            json.dump({'real': golden[key], 'predicted': predicted[key]}, file)

    return golden, predicted, decoded

def make_train_val_loaders(dataset, batch_size, shuffle_seed):
    indices = list(range(len(dataset.train_text)))
    y = np.array(dataset.train_targets)
    n_total = len(indices)
    n_classes = dataset.n_classes  
    test_size_ratio = 0.1          

    n_val = int(n_total * test_size_ratio)

    if n_val < n_classes:
        print(f"Warning: Training set is too small for validation split")
        print("Using the full dataset for training and skipping early stopping")
        tr_idx = indices  
        val_idx = []      
    else:
        tr_idx, val_idx = train_test_split(
            indices, 
            test_size=test_size_ratio,
            random_state=shuffle_seed, 
            stratify=y
        )

    train_ds = copy.copy(dataset);  train_ds.train = True
    val_ds   = copy.copy(dataset);  val_ds.train   = True

    train_ds.train_text   = [dataset.train_text[i]   for i in tr_idx]
    train_ds.train_targets= [dataset.train_targets[i]for i in tr_idx]

    val_ds.train_text     = [dataset.train_text[i]   for i in val_idx]
    val_ds.train_targets  = [dataset.train_targets[i]for i in val_idx]

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              sampler=SeededRandomSampler(train_ds, seed=shuffle_seed),
                              pin_memory=True)
    
    val_loader   = DataLoader(val_ds, batch_size=64, shuffle=False, pin_memory=True)
    
    return train_loader, val_loader

def ft_experiment(randomness_factor_seeds, train_test_indices=None):
    tokenizer = AutoTokenizer.from_pretrained(model_name, return_dict=False)
    dataset = FineTuningDataset(
        dataset_name=DATASET,
        train_size=args.train_size,
        num_labelled=args.num_labelled,
        num_labelled_test=args.num_labelled_test,
        label_seed=randomness_factor_seeds['label_choice'],
        device=device,
        full_test=FULL_TEST,
        tokenizer=tokenizer,
        max_len=MAX_LEN,
        train_test_indices=train_test_indices
    )
    loader = DatasetLoader(DATASET, BATCH_SIZE, dataset, randomness_factor_seeds['sample_order'])
    trainloader, valloader = make_train_val_loaders(dataset, BATCH_SIZE, randomness_factor_seeds['sample_order'])
    testloader  = loader.testloader()

    net = FT_MODELS[MODEL](dataset.n_classes, randomness_factor_seeds['model_initialisation'], randomness_factor_seeds['model_randomness'], True)
    net.cuda()
    optimizer = torch.optim.AdamW(params=net.parameters(), lr=LEARNING_RATE)
    
    num_epochs = min(args.num_epochs, 10)
    total_steps = max(1, len(trainloader) * num_epochs)
    warmup_steps = max(1, int(0.1 * total_steps))
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    loss_fn = torch.nn.CrossEntropyLoss()
    
    best_val = float('inf')
    patience = 5
    pat = 0

    for epoch in range(num_epochs):
        net.train()
        for batch_idx, data in enumerate(trainloader):
            optimizer.zero_grad()
            
            ids = data['ids'].to(device, dtype=torch.long)
            mask = data['mask'].to(device, dtype=torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
            targets = data['targets'].to(device, dtype=torch.long)

            outputs = net(ids, mask, token_type_ids)
            loss = loss_fn(outputs, targets)
            loss.backward()
            
            optimizer.step()
            scheduler.step()
            
        net.eval()
        val_loss = 0.0
        val_n = 0
        with torch.no_grad():
            for data in valloader:
                ids = data['ids'].to(device, dtype=torch.long)
                mask = data['mask'].to(device, dtype=torch.long)
                token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
                targets = data['targets'].to(device, dtype=torch.long)
                logits = net(ids, mask, token_type_ids)
                val_loss += loss_fn(logits, targets).item() * targets.size(0)
                val_n += targets.size(0)
        val_loss /= max(1, val_n)

        if val_loss < best_val - 1e-4:
            best_val = val_loss
            best_state = copy.deepcopy(net.state_dict())
            pat = 0
        else:
            pat += 1
            if pat >= patience:
                break
    
    if 'best_state' in locals():
        net.load_state_dict(best_state)
    
    golden = []
    predictions = []
    
    net.eval()
    with torch.no_grad():
        for batch_idx, data in enumerate(testloader):
            ids = data['ids'].to(device, dtype=torch.long)
            mask = data['mask'].to(device, dtype=torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
            targets = data['targets'].to(device, dtype=torch.long)

            outputs = net(ids, mask, token_type_ids)

            _, predicted = torch.max(outputs.data, 1)
            predictions.extend(predicted.tolist())
            golden.extend(targets.tolist())
    return golden, predictions

parser = argparse.ArgumentParser()
# Meta
parser.add_argument('--experiment_name', default='investigation_experiments', type=str, help='Directory to save experiments to')
parser.add_argument('--configuration_name', default='stability', type=str, help='Further distinction for the save directory')
parser.add_argument('--experiment_type', default='icl', type=str, choices=['finetuning', 'prompting', 'icl', 'icl_similarity', 'instruction_tuning', 'instruction_tuning_steps'], help='Type of experiment to run')
parser.add_argument('--full_test', default=1, type=int, help='Whether to use whole test dataset (Yes (default): 1; No: 0). If "No" and "num_labelled_test" is not set then uses same number of labelled samples as defined by num_labelled')
parser.add_argument('--regenerate', default=0, type=int, help='Whether to calculate every result again or continue from checkpoint (Yes: 1; No (default): 0).')
# General training args
parser.add_argument('--factor', default='golden_model', type=str, choices=['golden_model', 'data_split', 'label_choice', 'sample_choice', 'sample_order', 'model_initialisation', 'model_randomness'], help='Randomness factor to investigate.')
parser.add_argument('--num_shots', default=4, type=int, help='Number of samples to use as in-context examples in in-context learning or in different tasks in meta-learning.')
parser.add_argument('--dataset', default='sst2', type=str, choices=['sst2', 'mrpc', 'cola', 'rte', 'boolq', 'trec', 'ag_news', 'db_pedia', 'snips'], help='Dataset to use for investigation.')
parser.add_argument('--num_classes', default=2, type=int, help='Number of classes in dataset.')
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--train_size', default=0.8, type=float)
parser.add_argument('--num_labelled', default=1000, type=int)
parser.add_argument('--num_labelled_test', default=1000, type=int)
parser.add_argument('--model', default='flan-t5', type=str, choices=['bert', 'roberta', 'flan-t5', 'llama2', 'chatgpt', 'protonet', 'maml', 'fomaml', 'reptile', 'mistral', 'zephyr', 'lora_bert', 'lora_roberta', 'llama3'])
parser.add_argument('--model_size', default='base', type=str, choices=['base', '8b'])
parser.add_argument('--lr', default=1e-5, type=float)
parser.add_argument('--num_epochs', default=5, type=int, help='Total number of epochs to train for')
parser.add_argument('--max_len', default=20, type=int, help='Maximal length of input for fine-tuning experiments')
parser.add_argument('--prompt_format', default=0, type=int, help='Which prompt format to use')
# K Fold
parser.add_argument('--rskf_splits', default=10, type=int, help='Number of folds for RepeatedStratifiedKFold (K).')
parser.add_argument('--rskf_repeats', default=1, type=int, help='Number of repeats for RepeatedStratifiedKFold (R).')
parser.add_argument('--rskf_seed', default=27, type=int, help='Random state for RepeatedStratifiedKFold.')

parser.add_argument('-f')
args = parser.parse_args()

device = torch.device('cuda')
FT_MODELS = {
    'bert':  BERTBase,
    'roberta':  RoBERTaBase,
}

ICL_MODELS = {
    'flan-t5_base': 'google/flan-t5-base',
    'llama2_base': 'meta-llama/Llama-2-13b-chat-hf',
    'mistral_base': 'mistralai/Mistral-7B-Instruct-v0.1',
    'zephyr_base': 'HuggingFaceH4/zephyr-7b-alpha',
    'llama3_8b': 'meta-llama/Meta-Llama-3-8B-Instruct'
}

ICL_MODEL_RUN = {
    'flan-t5_base': run_flan_t5,
    'llama2_base': run_llama2_optimized,
    'chatgpt_base': run_chatgpt,
    'mistral_base': run_mistral,
    'zephyr_base': run_zephyr,
    'llama3_8b': run_llama3
}

EXPERIMENT_TYPE = args.experiment_type
FULL_TEST = args.full_test == 1
MAX_LEN = args.max_len
PROMPT_FORMAT = args.prompt_format

torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)
np.random.seed(0)
random.seed(0)
os.environ['PYTHONHASHSEED'] = '0'

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = not torch.backends.cudnn.deterministic

MODEL = args.model
MODEL_SIZE = args.model_size
FACTOR = args.factor
DATASET = args.dataset
RESULTS_PATH = os.path.join('results', f'{args.experiment_name}', f'{EXPERIMENT_TYPE}_{MODEL}_{MODEL_SIZE}', args.configuration_name, DATASET, FACTOR)
if not os.path.exists(RESULTS_PATH):
    os.makedirs(RESULTS_PATH)

BATCH_SIZE = args.batch_size
NUM_EPOCHS = args.num_epochs
LEARNING_RATE = args.lr


if MODEL == 'chatgpt':
    model_name = MODEL

elif EXPERIMENT_TYPE in ['instruction_tuning', 'instruction_tuning_steps']:
    model_name = ICL_MODELS[f'{MODEL}_{MODEL_SIZE}']
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = 'right'

elif EXPERIMENT_TYPE in ('icl', 'prompting', 'icl_similarity'):
    model_name = ICL_MODELS[f'{MODEL}_{MODEL_SIZE}']

    if MODEL == 'llama2':
        access_token = os.environ['HUGGINGFACE_TOKEN']
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, token=access_token)
        
        tokenizer.padding_side = 'left'
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", load_in_4bit=True, token=access_token)
        model.config.pad_token_id = tokenizer.pad_token_id
        generation_config = model.generation_config
        generation_config.num_beams = 1
        generation_config.max_new_tokens = 10
        generation_config.do_sample = False
        generation_config.temperature = None
        model.eval()
    elif MODEL in ['mistral', 'zephyr', 'llama3']:
        access_token = None
        if MODEL == 'llama3':
            access_token = os.environ['HUGGINGFACE_TOKEN']
        model = AutoModelForCausalLM.from_pretrained(model_name, load_in_4bit=True, device_map="auto", token=access_token)
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=access_token)
        tokenizer.padding_side = 'left'
        
        if tokenizer.pad_token is None:
            if MODEL == 'llama3':
                tokenizer.pad_token = tokenizer.eos_token
            else:
                tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                model.resize_token_embeddings(len(tokenizer))
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.padding_side = 'right'
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name).cuda()
        model.eval()

else:
    model_name = f'{MODEL}-{MODEL_SIZE}{"-uncased" if MODEL == "bert" else ""}'
    
total_runs = args.rskf_repeats * args.rskf_splits
run_seeds_path = os.path.join(RESULTS_PATH, 'run_seeds.pkl')

if os.path.exists(run_seeds_path) and args.regenerate == 0:
    with open(run_seeds_path, 'rb') as file:
        print(f'Loading existing run seeds:')
        run_seeds = pickle.load(file)
        if len(run_seeds) != total_runs:
             print(f"Warning: Number of saved seeds ({len(run_seeds)}) doesn't match expected runs ({total_runs}). Regenerating.")
             random.seed(args.rskf_seed)
             run_seeds = [random.randint(1, 100000) for _ in range(total_runs)]
             with open(run_seeds_path, 'wb') as file:
                 pickle.dump(run_seeds, file)
else:
    print(f'Generating new run seeds:')
    random.seed(args.rskf_seed)
    run_seeds = [random.randint(1, 100000) for _ in range(total_runs)]
    print(f'Saving new run seeds:')
    with open(run_seeds_path, 'wb') as file:
        pickle.dump(run_seeds, file)

print(f'Using seeds: {run_seeds}')

_, all_targets, _ = load_text_and_targets(DATASET, PROMPT_FORMAT)
all_targets = np.array(all_targets)
n_samples = len(all_targets)

rskf = RepeatedStratifiedKFold(
    n_splits=args.rskf_splits,
    n_repeats=args.rskf_repeats,
    random_state=args.rskf_seed
)

print(f'Running RSKF with {args.rskf_repeats} repeats × {args.rskf_splits} folds (total {args.rskf_repeats * args.rskf_splits}).')

fold_counter = 0
for split_idx, (train_idx, test_idx) in enumerate(rskf.split(np.zeros(n_samples), all_targets)):

    r = split_idx // args.rskf_splits
    k = split_idx %  args.rskf_splits

    fold_path = os.path.join(RESULTS_PATH, f'repeat_{r}_fold_{k}')
    if not os.path.exists(fold_path):
        os.makedirs(fold_path)


    if os.path.exists(os.path.join(fold_path, 'results.json')) and args.regenerate == 0:
        print(f'RSKF repeat {r}, fold {k} already exists. Skipping!')
        continue


    current_run_seed = run_seeds[split_idx]
    print(f'Running RSKF repeat {r}, fold {k} | train={len(train_idx)} test={len(test_idx)} | Seed: {current_run_seed}')

    randomness_factor_seeds = {
        'label_choice':       current_run_seed,
        'sample_choice':      current_run_seed,
        'sample_order':       current_run_seed,
        'model_initialisation': current_run_seed,
        'model_randomness':   current_run_seed
    }

    if EXPERIMENT_TYPE in ['finetuning']:
        print('Running fine-tuning experiments!')
        golden, predicted = ft_experiment(randomness_factor_seeds, train_test_indices=(train_idx, test_idx))
        decodeds = None
    elif EXPERIMENT_TYPE in ['instruction_tuning', 'instruction_tuning_steps']:
        golden, predicted, decodeds = instruction_tuning_experiment(
            randomness_factor_seeds, model_name, tokenizer, fold_path, train_test_indices=(train_idx, test_idx)
        )
        f1_prompting = compute_macro_f1(golden['prompting'], predicted['prompting'], failed_label=-1)
        f1_icl       = compute_macro_f1(golden['icl'],       predicted['icl'],       failed_label=-1)
    elif MODEL == 'chatgpt':
        golden, predicted, decodeds = prompt_icl_experiment(
            randomness_factor_seeds, None, None, EXPERIMENT_TYPE, investigation_path=fold_path,
            train_test_indices=(train_idx, test_idx)
        )
    else:
        ret = prompt_icl_experiment(
            randomness_factor_seeds, model, tokenizer, EXPERIMENT_TYPE, investigation_path=fold_path,
            train_test_indices=(train_idx, test_idx)
        )
        if isinstance(ret, tuple) and len(ret) == 3:
            golden, predicted, decodeds = ret
        else:
            golden, predicted = ret
            decodeds = None
    
    if(EXPERIMENT_TYPE not in ['instruction_tuning', 'instruction_tuning_steps']):
        f1_macro = compute_macro_f1(golden, predicted, failed_label=-1)
        print(f1_macro)
        results = {'f1_macro': float(f1_macro), **randomness_factor_seeds}
    else:
        results = {'f1_prompting': float(f1_prompting), 'f1_macro_icl': float(f1_icl), **randomness_factor_seeds}
        
    results['real'] = golden
    results['predicted'] = predicted
    results['base_model'] = model_name
    results['rskf_repeat'] = int(r)
    results['rskf_fold'] = int(k)
    results['run_seed_used'] = current_run_seed
    if decodeds is not None:
        results['decodeds'] = decodeds

    with open(os.path.join(fold_path, 'results.json'), 'w') as file:
        json.dump(results, file)
      
    # Clean checkpoints  
    if EXPERIMENT_TYPE in ['instruction_tuning', 'instruction_tuning_steps']:
        try:
            for item_name in os.listdir(fold_path):
                item_path = os.path.join(fold_path, item_name)
                if os.path.isdir(item_path) and item_name.startswith('checkpoint-'):
                    shutil.rmtree(item_path)
        except Exception as e:
            print(f"Error while trying to remove checkpoint: {e}")

    fold_counter += 1
