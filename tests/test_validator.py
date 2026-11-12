import copy
import unittest
from tunix.validator import calibrate, rouge_l, token_f1, validate

TEXT = 'City officials opened a new public library today after residents raised funds for its construction.'
CONFIG = dict(dtype='float16', attn='sdpa', batch=1, compile=False, max_new_tokens=128)


def bundle(text=TEXT, split='tuning', count=20, config=None):
    return dict(profile_name='fixture', config=dict(config or CONFIG), docs=[dict(
        doc_id='doc-' + split, split=split, output_text=text,
        n_output_tokens=count, gold_summary=TEXT)])


class ValidatorTests(unittest.TestCase):
    def setUp(self):
        self.base = bundle()
        self.cal = calibrate(self.base, self.base)

    def test_identical_scores_one(self):
        self.assertEqual(token_f1(TEXT, TEXT), 1)
        self.assertEqual(rouge_l(TEXT, TEXT), 1)
        self.assertTrue(validate(self.base, self.base, self.cal)['accepted'])

    def test_invalid_outputs(self):
        cases = [('', 0, 'empty'), ('too short', 2, 'short'),
                 ('spam ' * 20, 20, 'garbage'),
                 (' '.join(reversed(TEXT.split())), 20, 'order_drop'),
                 ('unrelated nonsense words ' * 10, 30, 'overlap_drop')]
        for text, count, reason in cases:
            with self.subTest(reason=reason):
                result = validate(bundle(text, count=count), self.base, self.cal)
                self.assertFalse(result['accepted'])
                self.assertIn(reason, result['reasons'])

    def test_cap64_truncation(self):
        candidate = bundle(count=64, config=dict(CONFIG, max_new_tokens=64))
        self.assertIn('truncation', validate(candidate, self.base, self.cal)['reasons'])

    def test_calibrated_variation(self):
        naive = bundle(TEXT.replace('new', 'modern'))
        cal = calibrate(self.base, naive, margin=0.02)
        self.assertGreater(cal.overlap_drop, 0)
        self.assertTrue(validate(naive, self.base, cal)['accepted'])
        self.assertFalse(validate(bundle('different ' * 20), self.base, cal)['accepted'])

    def test_split_and_identity_fail_closed(self):
        for artifact in ({}, bundle(split='heldout'), dict(self.base, docs=[])):
            self.assertFalse(validate(artifact, self.base, self.cal)['accepted'])
        duplicate = copy.deepcopy(self.base)
        duplicate['docs'] *= 2
        self.assertFalse(validate(duplicate, self.base, self.cal)['accepted'])
        with self.assertRaises(ValueError):
            calibrate(bundle(''), self.base)

    def test_heldout_calibration_leak(self):
        held = bundle(split='heldout')
        held['docs'][0]['doc_id'] = 'doc-tuning'
        self.assertFalse(validate(held, held, self.cal, 'heldout')['accepted'])
