# main.py

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments, get_linear_schedule_with_warmup, EarlyStoppingCallback
from datasets import Dataset
from data import ICLDataset, FineTuningDataset, DatasetLoader, PromptDataset, SimilarityICLDataset, InstructionTuningDataset, TextDataset, load_text_and_targets, SeededRandomSampler
from transfer_learning.models import BERTBase, RoBERTaBase, DeBERTaBase
from evaluator import ModelEvaluator, _parse_results as parse_results, EvaluationResult
from prompter import PromptFormatter
from rng import RNGController, RNGStream
from result_store import ResultStore
from sustainability import SustainabilityTracker
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

    RNGController.seed_all(randomness_factor_seeds['model_randomness'])

    mode = 'icl' if 'icl' in experiment else 'prompting'
    partial_path = os.path.join(investigation_path, 'partial') if investigation_path else None

    if ENGINE == 'vllm':
        evaluator = ModelEvaluator(
            model=model_name,
            tokenizer=None,
            model_name=MODEL,
            batch_size=BATCH_SIZE,
            prompt_format=PROMPT_FORMAT,
            max_new_tokens=10,
            engine='vllm',
            seed=randomness_factor_seeds['model_randomness'],
        )
    else:
        evaluator = ModelEvaluator(
            model=model,
            tokenizer=tokenizer,
            model_name=MODEL,
            batch_size=BATCH_SIZE if MODEL == 'flan-t5' else 1,
            prompt_format=PROMPT_FORMAT,
            max_new_tokens=10,
            generation_config=getattr(model, 'generation_config', None) if model is not None else None,
            device=device
        )
    return evaluator.evaluate(dataset, mode=mode, partial_save_path=partial_path, decoding=DECODING)



def instruction_tuning_experiment(randomness_factor_seeds, model_name, tokenizer, investigation_path,
                                  train_test_indices=None):
    if ENGINE == 'vllm':
        raise NotImplementedError(
            "vLLM engine does not yet support evaluating in-memory tuned models. "
            "Run instruction_tuning with --engine hf, or save the merged model "
            "and evaluate it via --experiment_type icl/prompting --engine vllm."
        )
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

    prompter = PromptFormatter(model_name=MODEL, prompt_format=PROMPT_FORMAT)
    prompts = prompter.format_instruction_tuning(
        dataset.context_samples,
        dataset_name=DATASET,
        classes=dataset.classes,
        custom_instruction=getattr(dataset, 'instructions', None)
    )
    response_template = prompter.get_response_template()

    if MODEL == 'flan-t5':
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name).cuda()
    elif MODEL in ['mistral', 'zephyr', 'llama3', 'qwen', 'phi', 'gemma']:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        
        target_modules = ['up_proj', 'down_proj', 'gate_proj', 'k_proj', 'q_proj', 'v_proj', 'o_proj', 'qkv_proj', 'gate_up_proj']

        peft_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=target_modules
        )
        
        access_token = os.environ.get('HUGGINGFACE_TOKEN', None)
        model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config, device_map='auto', token=access_token)
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=access_token)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'right'

        model = prepare_model_for_kbit_training(model)
            
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
        optim="paged_adamw_8bit" if MODEL != 'flan-t5' else 'adamw_torch',
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
        peft_config=peft_config if MODEL != 'flan-t5' else None,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5, early_stopping_threshold=0.0)]
    )

    trainer.train()
    if MODEL != 'flan-t5':
       model = trainer.model.merge_and_unload()
    model.eval()
    
    if MODEL != 'flan-t5':
        tokenizer.padding_side = 'left'

    golden = {'prompting': None, 'icl': None}
    predicted = {'prompting': None, 'icl': None}
    decoded = {'prompting': None, 'icl': None}
    prompts = {'prompting': None, 'icl': None}
    inputs = {'prompting': None, 'icl': None}
    for key in ['prompting', 'icl']:
        eval_res = prompt_icl_experiment(
            randomness_factor_seeds, model, tokenizer, key, investigation_path=investigation_path,
            train_test_indices=train_test_indices
        )
        golden[key], predicted[key], decoded[key] = eval_res[0], eval_res[1], eval_res[2]
        prompts[key] = getattr(eval_res, 'prompts', None)
        inputs[key] = getattr(eval_res, 'inputs', None)
        score = compute_macro_f1(golden[key], predicted[key], failed_label=-1)
        print(score)
        with open(os.path.join(investigation_path, f'{key}_results.json'), 'w') as file:
            json.dump({'real': golden[key], 'predicted': predicted[key]}, file)

    return EvaluationResult(golden, predicted, decoded, prompts=prompts, inputs=inputs)

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
    # DeBERTa-v3 uses SentencePiece: the fast tokenizer conversion drops byte
    # fallback (unknown tokens instead of byte pieces), so use the slow one.
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, return_dict=False, use_fast=(MODEL != 'deberta'))
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
parser.add_argument('--model', default='flan-t5', type=str, choices=['bert', 'roberta', 'deberta', 'flan-t5', 'llama2', 'chatgpt', 'protonet', 'maml', 'fomaml', 'reptile', 'mistral', 'zephyr', 'lora_bert', 'lora_roberta', 'llama3', 'qwen', 'phi', 'gemma'])
parser.add_argument('--model_size', default='base', type=str, choices=['base', '8b', '4b', '9b', 'mini', '26b'])
parser.add_argument('--lr', default=1e-5, type=float)
parser.add_argument('--num_epochs', default=5, type=int, help='Total number of epochs to train for')
parser.add_argument('--max_len', default=20, type=int, help='Maximal length of input for fine-tuning experiments')
parser.add_argument('--prompt_format', default=0, type=int, help='Which prompt format to use')
parser.add_argument('--engine', default='hf', type=str, choices=['hf', 'vllm'], help='Inference engine for prompting/icl (vllm: offline vLLM, causal models only)')
parser.add_argument('--decoding', default='free', type=str, choices=['free', 'guided'], help='Decoding constraint (guided requires --engine vllm)')
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
    'deberta': DeBERTaBase,
}

ICL_MODELS = {
    'flan-t5_base': 'google/flan-t5-base',
    'llama2_base': 'meta-llama/Llama-2-13b-chat-hf',
    'mistral_base': 'mistralai/Mistral-7B-Instruct-v0.1',
    'zephyr_base': 'HuggingFaceH4/zephyr-7b-alpha',
    'llama3_8b': 'meta-llama/Meta-Llama-3-8B-Instruct',
    'qwen_4b': 'Qwen/Qwen3.5-4B',
    'qwen_9b': 'Qwen/Qwen3.5-9B',
    'phi_mini': 'microsoft/Phi-4-mini-instruct',
    'gemma_26b': 'google/gemma-4-26B-A4B',
}

EXPERIMENT_TYPE = args.experiment_type
FULL_TEST = args.full_test == 1
MAX_LEN = args.max_len
PROMPT_FORMAT = args.prompt_format

RNGController.seed_all(0)
os.environ['PYTHONHASHSEED'] = '0'

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = not torch.backends.cudnn.deterministic

MODEL = args.model
MODEL_SIZE = args.model_size
FACTOR = args.factor
DATASET = args.dataset
ENGINE = args.engine
DECODING = args.decoding
if DECODING == 'guided' and ENGINE != 'vllm':
    parser.error("--decoding guided requires --engine vllm")
if ENGINE == 'vllm' and EXPERIMENT_TYPE in ['instruction_tuning', 'instruction_tuning_steps']:
    parser.error("--engine vllm does not support instruction_tuning (in-memory tuned model)")
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

    if ENGINE == 'vllm':
        # Weights load inside _VLLMAdapter; keep only the HF id here so a
        # second copy never sits in VRAM. Fail fast for unsupported families.
        if ModelEvaluator._causal_family_for(MODEL) is None:
            parser.error(f"--engine vllm supports causal families only, not '{MODEL}'")
        model = None
        tokenizer = None

    elif MODEL == 'llama2':
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
    elif MODEL in ['mistral', 'zephyr', 'llama3', 'qwen', 'phi', 'gemma']:
        access_token = os.environ.get('HUGGINGFACE_TOKEN', None)
        model = AutoModelForCausalLM.from_pretrained(model_name, load_in_4bit=True, device_map="auto", token=access_token)
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=access_token)
        tokenizer.padding_side = 'left'
        
        if tokenizer.pad_token is None:
            if hasattr(tokenizer, 'eos_token') and tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
            else:
                tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                model.resize_token_embeddings(len(tokenizer))
        model.eval()
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.padding_side = 'right'
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name).cuda()
        model.eval()

else:
    if MODEL == 'deberta':
        model_name = 'microsoft/deberta-v3-base'
    else:
        model_name = f'{MODEL}-{MODEL_SIZE}{"-uncased" if MODEL == "bert" else ""}'

result_store = ResultStore(
    results_path=RESULTS_PATH,
    experiment_name=args.experiment_name,
    experiment_type=EXPERIMENT_TYPE,
    model_name=model_name,
    dataset=DATASET,
    factor=FACTOR,
    configuration_name=args.configuration_name,
    legacy_json=True,
)
    
total_runs = args.rskf_repeats * args.rskf_splits
run_seeds_path = os.path.join(RESULTS_PATH, 'run_seeds.pkl')
run_seeds = RNGController.generate_run_seeds(
    rskf_seed=args.rskf_seed,
    total_runs=total_runs,
    cache_path=run_seeds_path,
    regenerate=bool(args.regenerate)
)
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


    if result_store.has_fold(r, k) and args.regenerate == 0:
        print(f'RSKF repeat {r}, fold {k} already exists. Skipping!')
        continue


    current_run_seed = run_seeds[split_idx]
    print(f'Running RSKF repeat {r}, fold {k} | train={len(train_idx)} test={len(test_idx)} | Seed: {current_run_seed}')

    randomness_factor_seeds = RNGController.build_factor_seeds(
        base_seed=current_run_seed,
        isolated_factor=FACTOR
    )

    fold_tracker = SustainabilityTracker()
    fold_tracker.start()

    if EXPERIMENT_TYPE in ['finetuning']:
        print('Running fine-tuning experiments!')
        golden, predicted = ft_experiment(randomness_factor_seeds, train_test_indices=(train_idx, test_idx))
        decodeds = None
        prompts = None
        inputs = None
        f1_macro = compute_macro_f1(golden, predicted, failed_label=-1)
        metrics = {'f1_macro': float(f1_macro)}
    elif EXPERIMENT_TYPE in ['instruction_tuning', 'instruction_tuning_steps']:
        eval_res = instruction_tuning_experiment(
            randomness_factor_seeds, model_name, tokenizer, fold_path, train_test_indices=(train_idx, test_idx)
        )
        golden, predicted, decodeds = eval_res[0], eval_res[1], eval_res[2]
        prompts = getattr(eval_res, 'prompts', None)
        inputs = getattr(eval_res, 'inputs', None)
        f1_prompting = compute_macro_f1(golden['prompting'], predicted['prompting'], failed_label=-1)
        f1_icl       = compute_macro_f1(golden['icl'],       predicted['icl'],       failed_label=-1)
        metrics = {'f1_prompting': float(f1_prompting), 'f1_macro_icl': float(f1_icl)}
    else:
        eval_model = None if MODEL == 'chatgpt' else model
        eval_tok = None if MODEL == 'chatgpt' else tokenizer
        eval_res = prompt_icl_experiment(
            randomness_factor_seeds, eval_model, eval_tok, EXPERIMENT_TYPE, investigation_path=fold_path,
            train_test_indices=(train_idx, test_idx)
        )
        golden, predicted, decodeds = eval_res[0], eval_res[1], eval_res[2]
        prompts = getattr(eval_res, 'prompts', None)
        inputs = getattr(eval_res, 'inputs', None)
        f1_macro = compute_macro_f1(golden, predicted, failed_label=-1)
        metrics = {'f1_macro': float(f1_macro)}

    sust_report = fold_tracker.stop()
    duration = sust_report.duration_seconds
    if 'f1_macro' in metrics:
        print(metrics['f1_macro'])

    result_store.record_fold(
        repeat=r,
        fold=k,
        run_seed_used=current_run_seed,
        randomness_factor_seeds=randomness_factor_seeds,
        metrics=metrics,
        real=golden,
        predicted=predicted,
        decodeds=decodeds,
        duration_seconds=duration,
        inputs=inputs,
        prompts=prompts,
        energy_kwh=sust_report.energy_kwh,
        co2_kg=sust_report.co2_kg,
        hardware=sust_report.hardware,
        carbon_intensity=sust_report.carbon_intensity,
        engine=ENGINE,
        quant_format=(
            'bitsandbytes' if ENGINE == 'vllm'
            else ('fp32' if MODEL == 'flan-t5'
                  else ('api' if MODEL == 'chatgpt' else 'bnb-4bit-nf4'))
        ),
        decoding=DECODING,
    )
      
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
