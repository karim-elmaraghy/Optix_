"""SYNTHETIC CPU fixtures only: no real model, GPU timings or API calls."""
import copy
import json
import math
import sys
import types
import unittest
from unittest.mock import patch

from tunix.agent import optimize
from tunix.integration import make_evaluator, select_winner
from tunix.validator import calibrate
from test_integration import DEFAULT, NAIVE, TEXT, benchmark, outputs


class AdapterTests(unittest.TestCase):
    def setUp(self):
        self.base = outputs('default', DEFAULT)
        self.cal = calibrate(self.base, outputs('naive', NAIVE))
        self.docs = [dict(doc_id=d['doc_id'], split='tuning', gold_summary=d['gold_summary'],
                          prompt='Summarize in 3 sentences: SYNTHETIC ARTICLE') for d in self.base['docs']]
        self.calls = []

    def runner(self, config, docs, **kwargs):
        self.calls.append((copy.deepcopy(config), docs, kwargs))
        name = kwargs['profile_name']
        return json.loads(json.dumps(dict(benchmark=benchmark(name, config),
                                          outputs=outputs(name, config), console={})))

    def test_actual_adapter_connects_to_optimizer_and_winner(self):
        config = dict(DEFAULT, batch=4)
        evaluate = make_evaluator(self.docs, self.base, self.cal, runner=self.runner,
                                  results_dir='SYNTHETIC-NOT-WRITTEN', repeats=3)
        records = optimize([config], evaluate, completion=lambda _: '{}')
        winner = select_winner(records)
        self.assertIs(winner, records[0])
        self.assertEqual(winner['proposal']['source'], 'rule_fallback')
        self.assertEqual(len(winner['result']['validation']['docs']), 20)
        self.assertEqual(self.calls[0][2]['split'], 'tuning')
        self.assertEqual(self.calls[0][2]['repeats'], 3)
        self.assertEqual(self.calls[0][2]['results_dir'], 'SYNTHETIC-NOT-WRITTEN')
        self.assertEqual(set(winner['result']), {'benchmark', 'validation'})

    def test_failed_benchmark_without_outputs_preserves_error(self):
        def fail(c, docs, **kw):
            return dict(benchmark=benchmark(kw['profile_name'], c, 'failed'))
        record = optimize([DEFAULT], make_evaluator(self.docs, self.base, self.cal, runner=fail),
                          completion=lambda _: '{}')[0]
        self.assertIn('execution_failed', record['reasons'])
        self.assertEqual(record['result']['benchmark']['error'], 'SYNTHETIC CUDA out of memory')
        self.assertIsNone(select_winner([record]))

    def test_default_runner_lazily_connects_to_benchmark_module(self):
        module = types.ModuleType('tunix.benchmark')
        module.run_config = self.runner
        evaluate = make_evaluator(self.docs, self.base, self.cal)
        with patch.dict(sys.modules, {'tunix.benchmark': module}):
            self.assertTrue(evaluate(DEFAULT)['validation']['accepted'])
        self.assertEqual(len(self.calls), 1)

    def test_invalid_real_return_artifact_rejected(self):
        for text in ('', 'Too short.', 'garbage ' * 30, ' '.join(reversed(TEXT.split()))):
            def invalid(c, docs, **kw):
                outcome = self.runner(c, docs, **kw)
                outcome['outputs'] = outputs(kw['profile_name'], c, text=text)
                return outcome
            result = make_evaluator(self.docs, self.base, self.cal, runner=invalid)(DEFAULT)
            self.assertFalse(result['validation']['accepted'])
            self.assertTrue(result['validation']['reasons'])

    def test_identity_config_and_document_mismatches_rejected(self):
        for kind in ('runtime', 'output_config', 'profile', 'doc_id'):
            def mismatch(c, docs, **kw):
                outcome = self.runner(c, docs, **kw)
                if kind == 'runtime': outcome['benchmark']['actual_config'] = dict(c, batch=8)
                if kind == 'output_config': outcome['outputs']['config'] = dict(c, batch=8)
                if kind == 'profile': outcome['outputs']['profile_name'] = 'stale'
                if kind == 'doc_id': outcome['outputs']['docs'][0]['doc_id'] = 'stale'
                return outcome
            with self.subTest(kind=kind):
                self.assertFalse(make_evaluator(self.docs, self.base, self.cal, runner=mismatch)(DEFAULT)['validation']['accepted'])

    def test_heldout_workload_cannot_enter_search(self):
        for bad in ([], [dict(d, split='heldout') for d in self.docs], self.docs + self.docs[:1]):
            with self.assertRaises(ValueError):
                make_evaluator(bad, self.base, self.cal, runner=self.runner)
        self.assertEqual(self.calls, [])

    def test_winner_rechecks_gates_and_finite_throughput(self):
        record = optimize([DEFAULT], make_evaluator(self.docs, self.base, self.cal, runner=self.runner),
                          completion=lambda _: '{}')[0]
        for field in ('status', 'quality', 'accepted', 'config'):
            bad = copy.deepcopy(record)
            if field == 'status': bad['result']['benchmark']['status'] = 'failed'
            if field == 'quality': bad['result']['validation']['accepted'] = False
            if field == 'accepted': bad['accepted'] = False
            if field == 'config': bad['result']['benchmark']['actual_config'] = dict(DEFAULT, batch=8)
            self.assertIsNone(select_winner([bad]))
        for rate in (None, math.nan, math.inf, -1, 0, True, '100'):
            bad = copy.deepcopy(record)
            bad['result']['benchmark']['docs_per_sec'] = rate
            self.assertIsNone(select_winner([bad]))
        faster = copy.deepcopy(record)
        faster['result']['benchmark']['docs_per_sec'] = 3.0  # synthetic sentinel
        self.assertIs(select_winner([record, faster]), faster)
        self.assertIsNone(select_winner([{}, dict(accepted=True)]))
