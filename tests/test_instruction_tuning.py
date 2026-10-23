"""Loss masking, Trainer config and validation split for instruction tuning
(no model weights, no GPU)."""
import tempfile
import unittest

from main import FT_MIN_VAL_SAMPLES, it_training_args, split_train_val, tokenize_it_pairs

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


class TestITTrainingArgs(unittest.TestCase):

    def _args(self, select_best):
        return it_training_args(tempfile.mkdtemp(), SEEDS, batch_size=4, learning_rate=1e-5,
                                num_epochs=5, seq2seq=False, select_best=select_best)

    def test_trainer_follows_the_fold_seeds(self):
        # The Trainer reseeds on creation; with its default seed (42) every
        # fold would share batch order and dropout.
        args = self._args(select_best=True)
        self.assertEqual(args.seed, SEEDS['model_randomness'])
        self.assertEqual(args.data_seed, SEEDS['sample_order'])

    def test_warmup_is_a_tenth_of_the_steps(self):
        # transformers>=5 has no warmup_ratio; the 10% must hold on both lines.
        self.assertEqual(self._args(select_best=True).get_warmup_steps(200), 20)

    def test_best_epoch_is_selected_only_with_a_validation_split(self):
        best = self._args(select_best=True)
        self.assertTrue(best.load_best_model_at_end)
        self.assertEqual(best.eval_strategy, "epoch")
        final = self._args(select_best=False)
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


if __name__ == '__main__':
    unittest.main()
