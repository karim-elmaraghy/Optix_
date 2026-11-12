import json
import os
import tempfile
import unittest
from tunix.safe_apply import safe_apply
from tunix.validator import calibrate

C = dict(dtype='float16', attn='sdpa', batch=1, compile=False, max_new_tokens=128)
TEXT = 'City officials opened a new public library today after residents raised funds for its construction.'


def artifact(config, split, text=TEXT, count=20):
    return dict(profile_name='fixture', config=config, docs=[dict(doc_id=split, split=split,
                output_text=text, n_output_tokens=count, gold_summary=TEXT)])


class SafeApplyTests(unittest.TestCase):
    def setUp(self):
        tuning = artifact(C, 'tuning')
        self.cal = calibrate(tuning, tuning)
        self.base = artifact(C, 'heldout')
        self.calls = []

    def run_apply(self, config, evaluator, **kwargs):
        return safe_apply(config, C, self.base, self.cal, self.calls.append, evaluator, **kwargs)

    def result(self, config, text=TEXT, count=20):
        return dict(benchmark=dict(status='success', requested_config=config, actual_config=config),
                    outputs=artifact(config, 'heldout', text, count))

    def test_promote_and_atomic_state(self):
        config = dict(C, batch=4)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'active.json')
            event = self.run_apply(config, self.result, state_path=path)
            self.assertEqual(event['action'], 'promoted')
            with open(path) as stream: self.assertEqual(json.load(stream), config)
        self.assertEqual(self.calls, [config])

    def test_bad_eight_token_profile_rolls_back_and_logs(self):
        config = dict(C, max_new_tokens=8)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'events.jsonl')
            event = self.run_apply(config, lambda c: self.result(c, 'Tiny summary.', 8), log_path=path)
            self.assertEqual(event['action'], 'rolled_back')
            self.assertIn('truncation', event['validation']['reasons'])
            with open(path) as stream: self.assertEqual(json.load(stream)['active_config'], C)
        self.assertEqual(self.calls, [config, C])

    def test_execution_and_truthfulness_failures(self):
        def fail(_): raise RuntimeError('device failure')
        def mismatch(c):
            result = self.result(c)
            result['benchmark']['actual_config'] = dict(c, batch=8)
            return result
        for evaluator in (fail, mismatch):
            self.calls.clear()
            event = self.run_apply(dict(C, batch=4), evaluator)
            self.assertEqual(event['action'], 'rolled_back')
            self.assertEqual(self.calls[-1], C)

    def test_rollback_failure_is_fatal(self):
        def apply(_): raise RuntimeError('cannot load')
        with self.assertRaisesRegex(RuntimeError, 'rollback failed'):
            safe_apply(dict(C, batch=4), C, self.base, self.cal, apply, self.result)
