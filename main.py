# main.py

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments, get_linear_schedule_with_warmup
from datasets import Dataset
from data import ICLDataset, FineTuningDataset, DatasetLoader, PromptDataset, SimilarityICLDataset, InstructionTuningDataset, TextDataset, load_text_and_targets
from transfer_learning.models import BERTBase, RoBERTaBase
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
from sklearn.model_selection import RepeatedStratifiedKFold
import time
import torch.nn.functional as F

from trl import SFTTrainer, DataCollatorForCompletionOnlyLM, SFTConfig
        

def parse_results(text, classes):
    pred = -1
    if DATASET in ['cola', 'mrpc'] and PROMPT_FORMAT in [3]:
        for idx, cls in enumerate(classes):
            if (cls.lower() in text.lower()) or (str(idx) in text):
                pred = idx
                break
    else:
        for idx, cls in enumerate(classes):
            if (cls.lower() in text.lower()) or (str(idx) in text):
                if pred == -1:
                    pred = idx
                else:
                    pred = -1
                    break
    return pred
        

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


def run_flan_t5(dataset, model, tokenizer):
    
    golden = []
    predicted = []
    decodeds = []
    for data, labels in dataset.batch_data_for_evaluation(BATCH_SIZE):
        final_prompts = prepare_flan_t5_icl(dataset, data) if EXPERIMENT_TYPE == 'icl' else prepare_flan_t5_prompt(dataset, data)
        
        encoded = tokenizer(final_prompts, return_tensors='pt', padding='longest', truncation=True).to('cuda')
        out = model.generate(**encoded, max_new_tokens=10)
        decoded = tokenizer.batch_decode(out, skip_special_tokens=True)

        print(decoded)
        decodeds.extend(decoded)
        
        predicted_labels = []
        for text in decoded:
            pred = parse_results(text, dataset.classes)
            predicted_labels.append(pred)
        print(predicted_labels)
        print(labels)

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


def run_llama2(dataset, model, tokenizer):
    golden = []
    predicted = []
    decodeds = []
    for data, labels in dataset.batch_data_for_evaluation(BATCH_SIZE):
        final_prompts = prepare_llama2_icl(dataset, data) if EXPERIMENT_TYPE == 'icl' else prepare_llama2_prompt(dataset, data)

        encoded = tokenizer(final_prompts, return_tensors='pt', padding='longest').to('cuda')
        out = model.generate(**encoded, max_new_tokens=20, do_sample=False, num_beams=1, generation_config=generation_config)
        decoded = tokenizer.batch_decode(out, skip_special_tokens=True)

        print(decoded)
        decodeds.extend(decoded)
        
        predicted_labels = []
        for text in decoded:
            text = text.split('[/INST]')[-1].lower()
            pred = parse_results(text, dataset.classes)
            predicted_labels.append(pred)
        print(predicted_labels)
        print(labels)

        predicted.extend(predicted_labels)
        golden.extend(labels)
    return golden, predicted, decodeds


def run_mistral(dataset, model, tokenizer):
    instructions = dataset.instructions
    context_samples = dataset.context_samples
    if PROMPT_FORMAT == 0:
        messages = [
            {'role': 'user', 'content': instructions['instruction']}, 
            {'role': 'assistant', 'content': f'Ok, I will determine the {instructions["task_type"]} of the Sentences you will give me using only the options provided!'}
        ]
        for sample in context_samples:
            messages.append({'role': 'user', 'content': sample[0]})
            messages.append({'role': 'assistant', 'content': sample[1]})
    else:
        messages = []
        for sample in context_samples:
            messages.append({'role': 'user', 'content': f'{sample[0]} {instructions["instruction"]} '})
            messages.append({'role': 'assistant', 'content': sample[1]})
    golden = []
    predicted = []
    for data, labels in dataset.batch_data_for_evaluation(1):
        for sample in data:
            temp_messages = copy.deepcopy(messages)
            if PROMPT_FORMAT == 0:
                temp_messages.append({'role': 'user', 'content': sample})
            else:
                temp_messages.append({'role': 'user', 'content': f'{sample} {instructions["instruction"]} '})
        encoded = tokenizer.apply_chat_template(temp_messages,return_tensors="pt", tokenize=True, add_generation_prompt=True).to('cuda')
        out = model.generate(encoded, max_new_tokens=10, do_sample=False, pad_token_id=tokenizer.pad_token_id)
        decoded = tokenizer.batch_decode(out)

        print(decoded)
        
        predicted_labels = []
        for text in decoded:
            text = text.split('[/INST]')[-1]
            pred = parse_results(text, dataset.classes)
            predicted_labels.append(pred)
        print(predicted_labels)
        print(labels)

        predicted.extend(predicted_labels)
        golden.extend(labels)
    return golden, predicted

def run_zephyr(dataset, model, tokenizer):
    instructions = dataset.instructions
    context_samples = dataset.context_samples
    if PROMPT_FORMAT == 0:
        messages = [
            {'role': 'user', 'content': instructions['instruction']}, 
        ]
        for sample in context_samples:
            messages.append({'role': 'user', 'content': sample[0]})
            messages.append({'role': 'assistant', 'content': sample[1]})
    else:
        messages = []
        for sample in context_samples:
            messages.append({'role': 'user', 'content': f'{sample[0]} {instructions["instruction"]} '})
            messages.append({'role': 'assistant', 'content': sample[1]})
    golden = []
    predicted = []
    for data, labels in dataset.batch_data_for_evaluation(1):
        for sample in data:
            temp_messages = copy.deepcopy(messages)
            if PROMPT_FORMAT == 0:
                temp_messages.append({'role': 'user', 'content': sample})
            else:
                temp_messages.append({'role': 'user', 'content': f'{sample} {instructions["instruction"]} '})
        encoded = tokenizer.apply_chat_template(temp_messages,return_tensors="pt", tokenize=True, add_generation_prompt=True).to('cuda')
        out = model.generate(encoded, max_new_tokens=10, do_sample=False, pad_token_id=tokenizer.pad_token_id)
        decoded = tokenizer.batch_decode(out)

        print(decoded)
        
        predicted_labels = []
        for text in decoded:
            text = text.split('<|assistant|>')[-1]
            pred = parse_results(text, dataset.classes)
            predicted_labels.append(pred)
        print(predicted_labels)
        print(labels)

        predicted.extend(predicted_labels)
        golden.extend(labels)
    return golden, predicted


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
        return ICL_MODEL_RUN[f'{MODEL}_{MODEL_SIZE}'](dataset, model, tokenizer)


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
    if PROMPT_FORMAT == 0:
        prompt = f'<s> [INST] {instructions["instruction"]}\n'
    else:
        prompt = '<s> [INST] '

    final_prompts = []
    for sample in context_samples:
        new_prompt = copy.deepcopy(prompt)
        if PROMPT_FORMAT == 0:
            new_prompt += f'{sample[0].strip()} [/INST] {sample[1].strip()}</s>'
        else:
            new_prompt += f'{sample[0].strip()} {instructions["instruction"]} [/INST] {sample[1].strip()}</s>'
        final_prompts.append(new_prompt)

    return final_prompts


def prepare_instruction_tuning_zephyr(dataset):
    instructions = dataset.instructions
    context_samples = dataset.context_samples
    if PROMPT_FORMAT == 0:
        prompt = f'<|user|> {instructions["instruction"]}\n'
    else:
        prompt = '<|user|> '

    final_prompts = []
    for sample in context_samples:
        new_prompt = copy.deepcopy(prompt)
        if PROMPT_FORMAT == 0:
            new_prompt += f'{sample[0].strip()} </s> <|assistant|> {sample[1].strip()}</s>'
        else:
            new_prompt += f'{sample[0].strip()} {instructions["instruction"]} </s> <|assistant|> {sample[1].strip()}</s>'
        final_prompts.append(new_prompt)

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
    elif MODEL in ['mistral', 'zephyr']:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        peft_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=['up_proj', 'down_proj', 'gate_proj', 'k_proj', 'q_proj', 'v_proj', 'o_proj']
        )
        
        model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config, device_map='auto')
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'right'

        model = prepare_model_for_kbit_training(model)
        
        if MODEL in 'mistral':
            prompts = prepare_instruction_tuning_mistral(dataset)
            response_template = "[/INST]"
        else:
            prompts = prepare_instruction_tuning_zephyr(dataset)
            response_template = "<|assistant|>"

    tuning_dataset = Dataset.from_dict({'prompt': prompts, 'label': dataset.train_targets})

    if 'steps' in EXPERIMENT_TYPE:
        max_steps = 150 if MODEL == 'flan-t5' else 600
    else:
        max_steps = -1

    training_args = SFTConfig(
        output_dir=investigation_path,
        per_device_train_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        num_train_epochs=NUM_EPOCHS,
        logging_strategy="no",
        save_strategy="no",
        max_steps=max_steps,
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
        train_dataset=tuning_dataset,
        data_collator=collator,
        tokenizer=tokenizer,
        peft_config=peft_config if MODEL in ['mistral', 'zephyr'] else None,
    )

    trainer.train()
    if MODEL in ['mistral', 'zephyr']:
       model = trainer.model.merge_and_unload()
    model.eval()

    golden = {'prompting': None, 'icl': None}
    predicted = {'prompting': None, 'icl': None}
    decoded = {'prompting': None, 'icl': None}
    for key in ['prompting', 'icl']:
        golden[key], predicted[key], decoded[key] = prompt_icl_experiment(
            randomness_factor_seeds, model, tokenizer, key, investigation_path=investigation_path,
            train_test_indices=train_test_indices
        )
        score = f1_score(np.array(golden[key]), np.array(predicted[key]), average='macro')
        print(score)
        with open(os.path.join(investigation_path, f'{key}_results.json'), 'w') as file:
            json.dump({'real': golden[key], 'predicted': predicted[key]}, file)

    return golden, predicted, decoded


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
    trainloader = loader.trainloader()
    testloader = loader.testloader()

    net = FT_MODELS[MODEL](dataset.n_classes, randomness_factor_seeds['model_initialisation'], randomness_factor_seeds['model_randomness'], True)
    net.cuda()
    optimizer = torch.optim.AdamW(params=net.parameters(), lr=LEARNING_RATE)
    
    total_steps = len(trainloader) * NUM_EPOCHS
    warmup_steps = max(1, int(0.1 * total_steps))
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    loss_fn = torch.nn.CrossEntropyLoss()

    for epoch in range(NUM_EPOCHS):
        net.train()
        for batch_idx, data in enumerate(trainloader):
            ids = data['ids'].to(device, dtype=torch.long)
            mask = data['mask'].to(device, dtype=torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
            targets = data['targets'].to(device, dtype=torch.long)

            optimizer.zero_grad()
            outputs = net(ids, mask, token_type_ids)
            loss = loss_fn(outputs, targets)

            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
    
    golden = []
    predictions = []
    
    net.eval()
    for batch_idx, data in enumerate(testloader):
        ids = data['ids'].to(device, dtype=torch.long)
        mask = data['mask'].to(device, dtype=torch.long)
        token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)

        outputs = net(ids, mask, token_type_ids)

        _, predicted = torch.max(outputs.data, 1)
        predictions.extend(predicted.tolist())
        golden.extend(data['targets'].tolist())
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
parser.add_argument('--model', default='flan-t5', type=str, choices=['bert', 'roberta', 'flan-t5', 'llama2', 'chatgpt', 'protonet', 'maml', 'fomaml', 'reptile', 'mistral', 'zephyr', 'lora_bert', 'lora_roberta'])
parser.add_argument('--model_size', default='base', type=str, choices=['base'])
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
    'zephyr_base': 'HuggingFaceH4/zephyr-7b-alpha'
}

ICL_MODEL_RUN = {
    'flan-t5_base': run_flan_t5,
    'llama2_base': run_llama2,
    'chatgpt_base': run_chatgpt,
    'mistral_base': run_mistral,
    'zephyr_base': run_zephyr,
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

BATCH_SIZE = args.batch_size # 64
NUM_EPOCHS = args.num_epochs # 5
LEARNING_RATE = args.lr # 1e-5


if MODEL == 'chatgpt':
    model_name = MODEL

elif EXPERIMENT_TYPE in ['instruction_tuning', 'instruction_tuning_steps']:
    model_name = ICL_MODELS[f'{MODEL}_{MODEL_SIZE}']
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = 'left'

elif EXPERIMENT_TYPE in ('icl', 'prompting', 'icl_similarity'):
    model_name = ICL_MODELS[f'{MODEL}_{MODEL_SIZE}']

    if MODEL == 'llama2':
        access_token = os.environ['HUGGINGFACE_TOKEN']
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, token=access_token)
        
        tokenizer.padding_side = 'left'
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", load_in_4bit=True, token=access_token)
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            model.resize_token_embeddings(len(tokenizer))
        generation_config = model.generation_config
        generation_config.num_beams = 1
        generation_config.max_new_tokens = 4
        generation_config.do_sample = False
        generation_config.temperature = None
        model.eval()
    elif MODEL in ['mistral', 'zephyr']:
        model = AutoModelForCausalLM.from_pretrained(model_name, load_in_4bit=True, device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.padding_side = 'left'
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer))
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.padding_side = 'left'
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

    print(np.mean(np.array(golden) == np.array(predicted)))

    results = copy.deepcopy(randomness_factor_seeds)
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

    fold_counter += 1
