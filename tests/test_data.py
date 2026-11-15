"""
The class names data.py gives each dataset (what the LLMs read, ICL shots and
IT targets) must be the prompter's, in the label order of the Hub dataset.
"""

import unittest
from unittest.mock import patch

import data
from prompter import PromptFormatter

# One row with every column load_text_and_targets reads, in each split.
_ROW = {
    "sentence": ["a"], "sentence1": ["a"], "sentence2": ["b"], "question": ["q"], "passage": ["p"],
    "text": ["t"], "content": ["c"], "label": [0], "coarse_label": [0], "category": ["AddToPlaylist"],
}
_SPLITS = {split: _ROW for split in ("train", "validation", "test")}


class TestClassNames(unittest.TestCase):

    def setUp(self):
        data.load_text_and_targets.cache_clear()
        self.addCleanup(data.load_text_and_targets.cache_clear)

    def classes(self, dataset_name):
        with patch("data.load_dataset", return_value=_SPLITS):
            return list(data.load_text_and_targets(dataset_name, 0)[2])

    def test_data_and_prompter_agree(self):
        for name, classes in PromptFormatter.DEFAULT_CLASSES.items():
            with self.subTest(dataset=name):
                self.assertEqual(self.classes(name), list(classes))

    def test_hub_label_order(self):
        # Label 0 of each Hub dataset whose names were once wrong or replaced.
        self.assertEqual(self.classes("rte")[0], "entailment")      # GLUE: entailment, not_entailment
        self.assertEqual(self.classes("trec")[0], "Abbreviation")   # coarse_label: ABBR, ENTY, ...
        self.assertEqual(self.classes("ag_news")[3], "Sci/Tech")    # World, Sports, Business, Sci/Tech


if __name__ == "__main__":
    unittest.main()
