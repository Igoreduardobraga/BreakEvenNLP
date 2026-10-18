"""Loss masking for instruction tuning (no model weights, no GPU)."""
import unittest

from main import tokenize_it_pairs


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


if __name__ == '__main__':
    unittest.main()
