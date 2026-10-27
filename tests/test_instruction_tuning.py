"""Loss masking, step budget, Trainer config and validation split for
instruction tuning (no pretrained weights; only the Qwen 3.5 loss test uses
a GPU, when there is one)."""
import tempfile
import unittest
from types import SimpleNamespace

from main import (FT_MIN_VAL_SAMPLES, IT_LORA_TARGETS, IT_STEP_BUDGET, completion_loss,
                  it_train_steps, it_training_args, it_val_loss, split_train_val,
                  tokenize_it_pairs)

SEEDS = {'label_choice': 1, 'sample_choice': 2, 'sample_order': 22,
         'model_initialisation': 3, 'model_randomness': 11}


class _CharTokenizer:
    """One id per character; BOS=1 as a special token, EOS=2 for seq2seq."""

    def __init__(self, seq2seq=False):
        self.seq2seq = seq2seq

    def __call__(self, text, add_special_tokens=True):
        ids = [ord(c) for c in text]
        if add_special_tokens:
            ids = ids + [2] if self.seq2seq else [1] + ids
        return {'input_ids': ids}


class TestTokenizeITPairs(unittest.TestCase):

    def test_causal_loss_covers_only_completion(self):
        tok = _CharTokenizer()
        [ex] = tokenize_it_pairs([("ab", "Y.")], tok, seq2seq=False)
        self.assertEqual(ex['input_ids'], [1, ord('a'), ord('b'), ord('Y'), ord('.')])
        self.assertEqual(ex['labels'], [-100, -100, -100, ord('Y'), ord('.')])
        self.assertEqual(ex['attention_mask'], [1] * 5)

    def test_causal_prompt_matches_inference_tokenization(self):
        tok = _CharTokenizer()
        [ex] = tokenize_it_pairs([("prompt", "x")], tok, seq2seq=False)
        n = len(tok("prompt")['input_ids'])
        self.assertEqual(ex['input_ids'][:n], tok("prompt")['input_ids'])

    def test_causal_truncation_keeps_answer_and_prompt_tail(self):
        tok = _CharTokenizer()
        [ex] = tokenize_it_pairs([("abcdef", "Y")], tok, seq2seq=False, max_len=4)
        self.assertEqual(ex['input_ids'], [ord('d'), ord('e'), ord('f'), ord('Y')])
        self.assertEqual(ex['labels'], [-100, -100, -100, ord('Y')])

    def test_seq2seq_encoder_never_sees_label(self):
        tok = _CharTokenizer(seq2seq=True)
        [ex] = tokenize_it_pairs([("Answer: ", "pos")], tok, seq2seq=True)
        self.assertEqual(ex['input_ids'], tok("Answer: ")['input_ids'])
        self.assertEqual(ex['labels'], [ord('p'), ord('o'), ord('s'), 2])
        self.assertNotIn(ord('p'), ex['input_ids'])


class TestITTrainSteps(unittest.TestCase):
    """5 epochs, cut to IT_STEP_BUDGET steps but never below one epoch."""

    def test_n_up_to_1000_trains_every_epoch(self):
        self.assertEqual(it_train_steps(10, 4, 5), 15)
        self.assertEqual(it_train_steps(900, 4, 5), 5 * 225)  # N=1000, 10% held out

    def test_budget_cuts_the_epochs(self):
        self.assertEqual(IT_STEP_BUDGET, 1250)
        self.assertEqual(it_train_steps(2250, 4, 5), IT_STEP_BUDGET)  # N=2500: ~2.2 epochs

    def test_large_n_still_sees_every_sample_once(self):
        self.assertEqual(it_train_steps(9000, 4, 5), 2250)  # N=10000: one epoch

    def test_steps_are_a_multiple_of_the_evaluations(self):
        # SNIPS N=15000: 11731 samples, one epoch of 2933 steps, rounded up so
        # the fifth evaluation falls on the last step.
        self.assertEqual(it_train_steps(11731, 4, 5), 2935)


class TestITTrainingArgs(unittest.TestCase):

    def _args(self, select_best):
        return it_training_args(tempfile.mkdtemp(), SEEDS, batch_size=4, learning_rate=1e-5,
                                num_epochs=5, seq2seq=False, select_best=select_best,
                                max_steps=200)

    def test_trainer_follows_the_fold_seeds(self):
        # The Trainer reseeds on creation; with its default seed (42) every
        # fold would share batch order and dropout.
        args = self._args(select_best=True)
        self.assertEqual(args.seed, SEEDS['model_randomness'])
        self.assertEqual(args.data_seed, SEEDS['sample_order'])

    def test_warmup_is_a_tenth_of_the_steps(self):
        # transformers>=5 has no warmup_ratio; the 10% must hold on both lines.
        self.assertEqual(self._args(select_best=True).get_warmup_steps(200), 20)

    def test_best_checkpoint_is_selected_only_with_a_validation_split(self):
        best = self._args(select_best=True)
        self.assertEqual(best.max_steps, 200)
        self.assertTrue(best.load_best_model_at_end)
        # num_epochs evaluations evenly spaced: once per epoch when every
        # epoch trains.
        self.assertEqual(best.eval_strategy, "steps")
        self.assertEqual(best.eval_steps, 40)
        self.assertEqual(best.save_steps, 40)
        final = self._args(select_best=False)
        self.assertEqual(final.max_steps, 200)
        self.assertFalse(final.load_best_model_at_end)
        self.assertEqual(final.eval_strategy, "no")
        self.assertEqual(final.save_strategy, "no")


class TestITValidationSplit(unittest.TestCase):
    """Instruction tuning holds out validation under the fine-tuning rule."""

    def test_small_pool_trains_on_every_sample(self):
        train, val, select_best = split_train_val([0, 1] * 5, 2, seed=1)
        self.assertFalse(select_best)
        self.assertEqual(sorted(train), list(range(10)))
        self.assertEqual(val, [])

    def test_large_pool_holds_out_a_stratified_tenth(self):
        targets = [0] * 150 + [1] * 100
        train, val, select_best = split_train_val(targets, 2, seed=1)
        self.assertTrue(select_best)
        self.assertEqual(len(val), FT_MIN_VAL_SAMPLES)
        self.assertEqual([targets[i] for i in val].count(1), 10)
        self.assertEqual(len(set(train) | set(val)), len(targets))

    def test_sweep_holdout_scores_a_small_pool_without_selecting_epochs(self):
        # The LR sweep's small band (N=100): 10% held out only to score the
        # final adapter, as in fine-tuning.
        targets = [0, 1] * 50
        train, val, select_best = split_train_val(targets, 2, seed=1, holdout_val=True)
        self.assertFalse(select_best)
        self.assertEqual(len(val), 10)
        self.assertEqual(len(set(train) | set(val)), len(targets))


class _Trainer:
    def __init__(self, best_metric=None, eval_loss=None):
        self.state = SimpleNamespace(best_metric=best_metric)
        self._eval_loss = eval_loss
        self.evaluated = False

    def evaluate(self):
        self.evaluated = True
        return {'eval_loss': self._eval_loss}


class TestITValLoss(unittest.TestCase):
    """The LR sweep ranks learning rates by the loss of the adapter that is kept."""

    def test_epoch_selection_reports_the_best_epoch(self):
        trainer = _Trainer(best_metric=0.25, eval_loss=0.9)
        self.assertEqual(it_val_loss(trainer, select_best=True), 0.25)
        self.assertFalse(trainer.evaluated)

    def test_holdout_scores_the_final_adapter(self):
        trainer = _Trainer(eval_loss=0.4)
        self.assertEqual(it_val_loss(trainer, select_best=False), 0.4)
        self.assertTrue(trainer.evaluated)


class TestCompletionLoss(unittest.TestCase):
    """Logits only at the completion positions give the model's own loss
    (tiny randomly initialised models on CPU)."""

    PAIRS = [("ab", "Y."), ("abcdefg", "No!"), ("abcd", "Yes")]

    def _batch(self):
        import torch
        examples = tokenize_it_pairs(self.PAIRS, _CharTokenizer(), seq2seq=False)
        width = max(len(e['input_ids']) for e in examples)
        return {k: torch.tensor([e[k] + [-100 if k == 'labels' else 0] * (width - len(e[k]))
                                 for e in examples])
                for k in ('input_ids', 'attention_mask', 'labels')}

    def _assert_matches_model_loss(self, config, device='cpu'):
        import torch
        from transformers import AutoModelForCausalLM
        torch.manual_seed(0)
        model = AutoModelForCausalLM.from_config(config).to(device).eval()
        batch = {k: v.to(device) for k, v in self._batch().items()}
        with torch.no_grad():
            expected = model(**batch).loss
            loss, outputs = completion_loss(model, batch)
            summed, _ = completion_loss(model, batch, num_items_in_batch=(batch['labels'] != -100).sum())
        torch.testing.assert_close(loss, expected)
        torch.testing.assert_close(summed, expected)
        # 8 of the 10 positions predict a completion token in some row.
        self.assertEqual(outputs.logits.shape, (3, 8, config.vocab_size))

    def test_llama(self):
        from transformers import LlamaConfig
        self._assert_matches_model_loss(LlamaConfig(
            vocab_size=128, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
            num_attention_heads=2, num_key_value_heads=1))

    def test_qwen3_5_with_gated_deltanet_layers(self):
        try:
            from transformers import Qwen3_5TextConfig
        except ImportError:
            self.skipTest("transformers without Qwen 3.5")
        import importlib.util
        import torch
        # With flash-linear-attention installed, transformers runs the DeltaNet
        # layers on its Triton kernels, which need a GPU.
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if device == 'cpu' and importlib.util.find_spec('fla') is not None:
            self.skipTest("flash-linear-attention kernels need a GPU")
        self._assert_matches_model_loss(Qwen3_5TextConfig(
            vocab_size=128, hidden_size=32, intermediate_size=64, num_hidden_layers=4,
            num_attention_heads=2, num_key_value_heads=1, head_dim=16, linear_num_key_heads=2,
            linear_num_value_heads=2, linear_key_head_dim=16, linear_value_head_dim=16), device)


class TestLoraCoverage(unittest.TestCase):
    """LoRA wraps every linear projection of the decoder blocks of each family
    (models built on the meta device, without weights)."""

    def _assert_every_linear_is_tuned(self, config):
        try:
            from peft import LoraConfig, get_peft_model
        except ImportError:
            self.skipTest("peft not installed")
        import torch
        from transformers import AutoModelForCausalLM
        with torch.device("meta"):
            model = AutoModelForCausalLM.from_config(config)
        head = model.get_output_embeddings()
        linear = {n for n, m in model.named_modules() if isinstance(m, torch.nn.Linear) and m is not head}
        model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, task_type="CAUSAL_LM",
                                                 target_modules=IT_LORA_TARGETS))
        tuned = {n.removeprefix("base_model.model.") for n, m in model.named_modules()
                 if hasattr(m, "lora_A") and hasattr(m, "base_layer")}
        self.assertTrue(linear)
        self.assertEqual(linear - tuned, set())

    def test_qwen3_5_tunes_its_gated_deltanet_layers(self):
        try:
            from transformers import Qwen3_5TextConfig
        except ImportError:
            self.skipTest("transformers without Qwen 3.5")
        config = Qwen3_5TextConfig(num_hidden_layers=4)
        self.assertEqual(config.layer_types.count("linear_attention"), 3)
        self._assert_every_linear_is_tuned(config)

    def test_phi_and_mistral(self):
        from transformers import MistralConfig, Phi3Config
        for config in (Phi3Config(num_hidden_layers=2), MistralConfig(num_hidden_layers=2)):
            with self.subTest(type(config).__name__):
                self._assert_every_linear_is_tuned(config)


if __name__ == '__main__':
    unittest.main()
