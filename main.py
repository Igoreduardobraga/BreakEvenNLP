# main.py

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments, Trainer, DataCollatorForSeq2Seq, get_linear_schedule_with_warmup, EarlyStoppingCallback
from datasets import Dataset
from sample_pool import effective_num_labelled
from data import ICLDataset, FineTuningDataset, DatasetLoader, PromptDataset, SimilarityICLDataset, InstructionTuningDataset, TextDataset, load_text_and_targets, SeededRandomSampler
from transfer_learning.models import BERTBase, RoBERTaBase, DeBERTaBase
from evaluator import ModelEvaluator, _parse_results as parse_results, EvaluationResult
from prompter import PromptFormatter
from rng import RNGController, RNGStream
from result_store import ResultStore, experiment_dir_name
from sustainability import SustainabilityTracker
import re
import random
import pickle
import argparse
import atexit
import dataclasses
import torch
import numpy as np
import os
import copy
import json
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from torch.utils.data import DataLoader
import time
import shutil
import sys
import traceback
import torch.nn.functional as F


def _require_it_deps():
    """Lazy peft imports: prompting/icl/finetuning must not require them."""
    try:
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    except ImportError as e:
        raise ImportError(f"instruction_tuning requires peft: {e}") from e
    return LoraConfig, get_peft_model, prepare_model_for_kbit_training


IT_MAX_LEN = 1024

# LoRA targets of the causal models: every linear projection of the decoder
# blocks. Qwen 3.5 puts three Gated DeltaNet layers (linear_attn: in_proj_qkv,
# in_proj_z, in_proj_b, in_proj_a, out_proj) before each full-attention layer;
# without their names LoRA reached the attention of only 8 of its 32 layers.
# Phi-4 fuses q/k/v into qkv_proj and gate/up into gate_up_proj.
IT_LORA_TARGETS = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'qkv_proj',
                   'in_proj_qkv', 'in_proj_z', 'in_proj_b', 'in_proj_a', 'out_proj',
                   'gate_proj', 'up_proj', 'down_proj', 'gate_up_proj']


def bnb_4bit_config():
    """4-bit NF4 setup shared by prompting, ICL and IT, and matching vLLM's
    bitsandbytes quantization: the approaches differ only in how the model
    is used, never in how it is quantized. The bare load_in_4bit kwarg meant
    FP4 with float32 compute, and transformers>=5 no longer accepts it."""
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )


def tokenize_it_pairs(pairs, tokenizer, seq2seq, max_len=IT_MAX_LEN):
    """Tokenizes (prompt, completion) pairs so the loss covers only the completion.

    The prompt is tokenized exactly as the evaluator tokenizes it at inference.
    Seq2seq: the encoder sees the prompt and the decoder learns the completion.
    Causal: prompt tokens are masked with -100; overlong prompts lose their
    head so the chat template's assistant header and the answer survive.
    """
    examples = []
    for prompt, completion in pairs:
        prompt_ids = tokenizer(prompt)['input_ids']
        if seq2seq:
            prompt_ids = prompt_ids[:max_len]
            examples.append({'input_ids': prompt_ids,
                             'attention_mask': [1] * len(prompt_ids),
                             'labels': tokenizer(completion)['input_ids']})
            continue
        completion_ids = tokenizer(completion, add_special_tokens=False)['input_ids']
        prompt_ids = prompt_ids[-(max_len - len(completion_ids)):]
        input_ids = prompt_ids + completion_ids
        examples.append({'input_ids': input_ids,
                         'attention_mask': [1] * len(input_ids),
                         'labels': [-100] * len(prompt_ids) + completion_ids})
    return examples


def warmup_ratio_kwargs(ratio):
    """Linear warmup over a fraction of the steps on both transformers lines:
    transformers>=5 dropped warmup_ratio and reads a float warmup_steps below 1
    as that fraction, while before it warmup_steps only counted steps."""
    fields = {f.name for f in dataclasses.fields(TrainingArguments)}
    return {'warmup_ratio': ratio} if 'warmup_ratio' in fields else {'warmup_steps': ratio}


def it_training_args(output_dir, factor_seeds, batch_size, learning_rate, num_epochs,
                     seq2seq, select_best):
    """TrainingArguments of one instruction-tuning fold.

    The Trainer reseeds every RNG when it is built: seed and data_seed carry
    the fold's model-randomness and sample-order seeds, the ones fine-tuning
    uses for dropout and batch order. Without a validation split (select_best
    False) every epoch trains and the final adapter is kept.
    """
    if select_best:
        selection = dict(eval_strategy="epoch", save_strategy="epoch", save_total_limit=1,
                         load_best_model_at_end=True, metric_for_best_model="eval_loss",
                         greater_is_better=False)
    else:
        selection = dict(eval_strategy="no", save_strategy="no")
    return TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        learning_rate=learning_rate,
        num_train_epochs=num_epochs,
        logging_strategy="no",
        gradient_accumulation_steps=1,
        optim="adamw_torch" if seq2seq else "paged_adamw_8bit",
        lr_scheduler_type="linear",
        seed=factor_seeds['model_randomness'],
        data_seed=factor_seeds['sample_order'],
        # PeftModel hides the labels argument from Trainer's signature check;
        # without it eval_loss is never computed for early stopping.
        label_names=['labels'],
        **selection,
        **warmup_ratio_kwargs(0.1),
    )


def it_val_loss(trainer, select_best):
    """Validation loss of the adapter that is evaluated, the LR sweep's criterion.

    With epoch selection it is the best epoch's loss, the epoch
    load_best_model_at_end restores; with the sweep's small holdout it scores
    the final adapter, as fine-tuning does.
    """
    if select_best:
        return float(trainer.state.best_metric)
    return float(trainer.evaluate()['eval_loss'])

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
        # Engine built once in main(); the fold's seed goes per request.
        return vllm_evaluator.evaluate(dataset, mode=mode, decoding=DECODING,
                                       seed=randomness_factor_seeds['model_randomness'])
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
    LoraConfig, get_peft_model, prepare_model_for_kbit_training = _require_it_deps()
    # LoRA weights are created on the GPU before the Trainer seeds anything:
    # seed them per fold so a resumed run reproduces the same adapter.
    RNGController.seed_all(randomness_factor_seeds['model_initialisation'])
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
    elif MODEL in ['mistral', 'zephyr', 'llama3', 'qwen', 'phi', 'gemma']:
        peft_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=IT_LORA_TARGETS
        )
        
        access_token = os.environ.get('HUGGINGFACE_TOKEN', None)
        model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_4bit_config(), device_map='auto', token=access_token)
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=access_token)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, peft_config)

    # Same formatter (and tokenizer/chat template) the evaluator uses, so the
    # tuned prompt is exactly the evaluated zero-shot prompt.
    prompter = PromptFormatter(model_name=MODEL, prompt_format=PROMPT_FORMAT, tokenizer=tokenizer)
    pairs = prompter.format_instruction_tuning(
        dataset.context_samples,
        dataset_name=DATASET,
        classes=dataset.classes,
        custom_instruction=getattr(dataset, 'instructions', None)
    )

    # Same validation rule as fine-tuning: the best epoch is picked only when
    # the split is large enough to rank epochs; below that every epoch trains
    # on all N samples.
    train_idx, val_idx, select_best = split_train_val(
        dataset.train_targets, dataset.num_classes, randomness_factor_seeds['sample_order'],
        holdout_val=args.holdout_val == 1)

    train_pairs = [pairs[i] for i in train_idx]
    val_pairs   = [pairs[i] for i in val_idx]
        
    if MODEL == 'flan-t5':
        steps_per_epoch_cap = 250
        max_examples_per_epoch = steps_per_epoch_cap * 4
        if len(train_pairs) > max_examples_per_epoch:
            rng = np.random.default_rng(randomness_factor_seeds['sample_order'])
            keep = rng.choice(len(train_pairs), size=max_examples_per_epoch, replace=False)
            keep.sort()
            train_pairs = [train_pairs[i] for i in keep]

    seq2seq = MODEL == 'flan-t5'
    train_ds = Dataset.from_list(tokenize_it_pairs(train_pairs, tokenizer, seq2seq))
    val_ds   = Dataset.from_list(tokenize_it_pairs(val_pairs, tokenizer, seq2seq)) if val_pairs else None

    training_args = it_training_args(investigation_path, randomness_factor_seeds, BATCH_SIZE,
                                     LEARNING_RATE, NUM_EPOCHS, seq2seq, select_best)

    # Pads input_ids/attention_mask with the tokenizer and labels with -100;
    # with the T5 model it also builds decoder_input_ids from the labels.
    collator = DataCollatorForSeq2Seq(tokenizer, model=model if seq2seq else None,
                                      label_pad_token_id=-100, padding='longest')

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5, early_stopping_threshold=0.0)] if select_best else None
    )

    trainer.train()
    val_metrics = {'val_loss': it_val_loss(trainer, select_best)} if val_ds is not None else {}
    if args.it_val_only:
        # LR sweep: the choice uses only the validation loss, and the test
        # generations would cost more than training the smaller bands.
        return EvaluationResult({'prompting': [], 'icl': []}, {'prompting': [], 'icl': []},
                                {'prompting': [], 'icl': []}), val_metrics

    # Keep the LoRA adapter unmerged: merging into 4-bit weights re-quantizes
    # them and evaluates a different model than the one that was trained.
    model = trainer.model
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

    return EvaluationResult(golden, predicted, decoded, prompts=prompts, inputs=inputs), val_metrics

# Early stopping needs a validation split large enough to rank epochs: below
# FT_MIN_VAL_SAMPLES (10% of N, i.e. N < 250) the model trains every epoch on
# all N labelled samples, as ICL uses all N. Instruction tuning picks its best
# epoch under the same rule. With holdout_val the LR sweep still holds out 10%
# in that regime, only to score the final model.
FT_VAL_RATIO = 0.1
FT_MIN_VAL_SAMPLES = 25


def split_train_val(targets, n_classes, seed, holdout_val=False):
    """Returns (train_idx, val_idx, early_stopping) over the labelled pool."""
    indices = list(range(len(targets)))
    n_val = int(len(indices) * FT_VAL_RATIO)

    early_stopping = n_val >= max(n_classes, FT_MIN_VAL_SAMPLES)
    if early_stopping or (holdout_val and n_val >= n_classes):
        tr_idx, val_idx = train_test_split(
            indices,
            test_size=FT_VAL_RATIO,
            random_state=seed,
            stratify=np.array(targets)
        )
    else:
        print(f"Validation split of {n_val} samples is too small: training on all "
              f"{len(indices)} samples for every epoch, without early stopping")
        tr_idx, val_idx = indices, []
    return tr_idx, val_idx, early_stopping


def make_train_val_loaders(dataset, batch_size, shuffle_seed, holdout_val=False):
    """Returns (train_loader, val_loader, early_stopping)."""
    tr_idx, val_idx, early_stopping = split_train_val(
        dataset.train_targets, dataset.n_classes, shuffle_seed, holdout_val)

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

    return train_loader, val_loader, early_stopping

def ft_experiment(randomness_factor_seeds, train_test_indices=None):
    # The encoder's internal dropout draws from the global RNG (only the head
    # dropout has its own stream): seed it per fold with the model-randomness
    # factor so each fold is reproducible on its own.
    RNGController.seed_all(randomness_factor_seeds['model_randomness'])
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
    trainloader, valloader, early_stopping = make_train_val_loaders(
        dataset, BATCH_SIZE, randomness_factor_seeds['sample_order'], holdout_val=args.holdout_val == 1)
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

        # Without a validation split there is no signal for early stopping:
        # train every epoch and keep the final weights.
        if len(valloader.dataset) == 0:
            continue

        net.eval()
        val_loss = 0.0
        val_n = 0
        val_golden, val_predicted = [], []
        with torch.no_grad():
            for data in valloader:
                ids = data['ids'].to(device, dtype=torch.long)
                mask = data['mask'].to(device, dtype=torch.long)
                token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
                targets = data['targets'].to(device, dtype=torch.long)
                logits = net(ids, mask, token_type_ids)
                val_loss += loss_fn(logits, targets).item() * targets.size(0)
                val_n += targets.size(0)
                val_golden.extend(targets.tolist())
                val_predicted.extend(logits.argmax(dim=1).tolist())
        val_loss /= max(1, val_n)
        val_f1 = f1_score(val_golden, val_predicted, average='macro', zero_division=0)

        if not early_stopping:
            # Holdout only scores the final model (LR sweep).
            best_val, best_val_f1 = val_loss, val_f1
            continue
        if val_loss < best_val - 1e-4:
            best_val = val_loss
            best_val_f1 = val_f1
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
    # Validation metrics of the returned model (best epoch, or the final one
    # without early stopping), used to pick the learning rate (run_lr_sweep.py).
    val_metrics = {}
    if best_val != float('inf'):
        val_metrics = {'val_loss': float(best_val), 'val_f1_macro': float(best_val_f1)}
    return golden, predictions, val_metrics


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
    'gemma_26b': 'google/gemma-4-26B-A4B-it',
}


def main():
    global args, device, MODEL, MODEL_SIZE, FACTOR, DATASET, \
        EXPERIMENT_TYPE, FULL_TEST, MAX_LEN, PROMPT_FORMAT, BATCH_SIZE, \
        NUM_EPOCHS, LEARNING_RATE, ENGINE, DECODING, RESULTS_PATH, \
        model, tokenizer, model_name, vllm_evaluator
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
    parser.add_argument('--decoding', default='free', type=str, choices=['free', 'guided'], help='Decoding constraint: guided restricts the answer to the class names (vLLM structured outputs, or token masking on HF)')
    # K Fold
    parser.add_argument('--rskf_splits', default=10, type=int, help='Number of folds for RepeatedStratifiedKFold (K).')
    parser.add_argument('--rskf_repeats', default=1, type=int, help='Number of repeats for RepeatedStratifiedKFold (R).')
    parser.add_argument('--rskf_seed', default=27, type=int, help='Random state for RepeatedStratifiedKFold.')
    parser.add_argument('--holdout_val', default=0, type=int, help='Fine-tuning and instruction tuning: hold out 10%% for validation even when it is too small to select epochs, only to score the final model (used by run_lr_sweep.py).')
    parser.add_argument('--it_val_only', default=0, type=int, help='Instruction tuning: record only the validation loss, without evaluating on the test subset (used by run_lr_sweep.py, where the test generations would cost more than the training).')
    parser.add_argument('--results_root', default='results', type=str, help='Root directory for results (the LR sweep writes elsewhere so it never mixes with the main grid).')
    parser.add_argument('--sweep', default=None, type=str, help='JSON list of configurations (dataset, num_labelled, configuration_name, rskf_splits) to run in this process with the model loaded once.')

    parser.add_argument('-f')
    args = parser.parse_args()

    device = torch.device('cuda')

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
    if DECODING == 'guided' and MODEL == 'chatgpt':
        parser.error("--decoding guided is not available for the API model")
    if ENGINE == 'vllm' and EXPERIMENT_TYPE in ['instruction_tuning', 'instruction_tuning_steps']:
        parser.error("--engine vllm does not support instruction_tuning (in-memory tuned model)")
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
            # Load once, outside the timed folds (like the HF models below):
            # rebuilding per fold reloads the weights and can leak VRAM.
            vllm_evaluator = ModelEvaluator(
                model=model_name,
                tokenizer=None,
                model_name=MODEL,
                batch_size=BATCH_SIZE,
                prompt_format=PROMPT_FORMAT,
                max_new_tokens=10,
                engine='vllm',
            )
            atexit.register(vllm_evaluator.close)

        elif MODEL == 'llama2':
            access_token = os.environ['HUGGINGFACE_TOKEN']
            tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, token=access_token)
        
            tokenizer.padding_side = 'left'
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", quantization_config=bnb_4bit_config(), token=access_token)
            model.config.pad_token_id = tokenizer.pad_token_id
            generation_config = model.generation_config
            generation_config.num_beams = 1
            generation_config.max_new_tokens = 10
            generation_config.do_sample = False
            generation_config.temperature = None
            model.eval()
        elif MODEL in ['mistral', 'zephyr', 'llama3', 'qwen', 'phi', 'gemma']:
            access_token = os.environ.get('HUGGINGFACE_TOKEN', None)
            model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_4bit_config(), device_map="auto", token=access_token)
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

    configurations = load_sweep(args.sweep, parser) if args.sweep else [{}]
    failed = []
    for config in configurations:
        try:
            run_configuration(config)
        except Exception:
            if len(configurations) == 1:
                raise
            # One bad configuration must not cost the rest of the sweep the
            # model load; its finished folds are kept and it resumes on rerun.
            traceback.print_exc()
            failed.append(config)
    if failed:
        print(f'{len(failed)}/{len(configurations)} sweep configurations failed: {failed}')
        sys.exit(1)


SWEEP_KEYS = ('dataset', 'num_labelled', 'configuration_name', 'rskf_splits')


def load_sweep(path, parser):
    """Configurations to run with the already loaded model.

    JSON list of objects; each may set any of SWEEP_KEYS and inherits the
    command-line value for the keys it omits.
    """
    with open(path) as f:
        configurations = json.load(f)
    datasets = next(a.choices for a in parser._actions if a.dest == 'dataset')
    for config in configurations:
        unknown = set(config) - set(SWEEP_KEYS)
        if unknown:
            parser.error(f"--sweep: unsupported keys {sorted(unknown)} (allowed: {SWEEP_KEYS})")
        if config.get('dataset', args.dataset) not in datasets:
            parser.error(f"--sweep: unknown dataset {config['dataset']!r}")
    return configurations


def run_configuration(config):
    """Runs every RSKF fold of one (dataset, num_labelled) configuration."""
    global DATASET, RESULTS_PATH
    for key, value in config.items():
        setattr(args, key, value)
    DATASET = args.dataset
    RESULTS_PATH = os.path.join(args.results_root, f'{args.experiment_name}', experiment_dir_name(EXPERIMENT_TYPE, MODEL, MODEL_SIZE, ENGINE, DECODING), args.configuration_name, DATASET, FACTOR)
    if not os.path.exists(RESULTS_PATH):
        os.makedirs(RESULTS_PATH)
    print(f'=== Configuration {args.configuration_name} | dataset={DATASET} | num_labelled={args.num_labelled} ===')

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

    _, all_targets, all_classes = load_text_and_targets(DATASET, PROMPT_FORMAT)
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
            golden, predicted, val_metrics = ft_experiment(randomness_factor_seeds, train_test_indices=(train_idx, test_idx))
            decodeds = None
            prompts = None
            inputs = None
            f1_macro = compute_macro_f1(golden, predicted, failed_label=-1)
            metrics = {'f1_macro': float(f1_macro), **val_metrics}
        elif EXPERIMENT_TYPE in ['instruction_tuning', 'instruction_tuning_steps']:
            eval_res, val_metrics = instruction_tuning_experiment(
                randomness_factor_seeds, model_name, tokenizer, fold_path, train_test_indices=(train_idx, test_idx)
            )
            golden, predicted, decodeds = eval_res[0], eval_res[1], eval_res[2]
            prompts = getattr(eval_res, 'prompts', None)
            inputs = getattr(eval_res, 'inputs', None)
            if args.it_val_only:
                metrics = {'f1_prompting': float('nan'), 'f1_macro_icl': float('nan')}
            else:
                f1_prompting = compute_macro_f1(golden['prompting'], predicted['prompting'], failed_label=-1)
                f1_icl       = compute_macro_f1(golden['icl'],       predicted['icl'],       failed_label=-1)
                metrics = {'f1_prompting': float(f1_prompting), 'f1_macro_icl': float(f1_icl)}
            metrics.update(val_metrics)
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
            gpu_energy_kwh=sust_report.gpu_energy_kwh,
            cpu_energy_kwh=sust_report.cpu_energy_kwh,
            energy_method=sust_report.method,
            engine=ENGINE,
            quant_format=(
                'bitsandbytes' if ENGINE == 'vllm'
                else ('fp32' if MODEL == 'flan-t5'
                      else ('api' if MODEL == 'chatgpt' else 'bnb-4bit-nf4'))
            ),
            decoding=DECODING,
            learning_rate=LEARNING_RATE if EXPERIMENT_TYPE in ('finetuning', 'instruction_tuning', 'instruction_tuning_steps') else None,
            num_labelled_effective=effective_num_labelled(
                args.num_labelled, len(all_classes),
                np.bincount(all_targets[train_idx], minlength=len(all_classes))),
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


if __name__ == '__main__':
    main()
