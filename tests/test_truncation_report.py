"""SYNTHETIC regression reproductions of reported counts, not T4 results."""
import json
from pathlib import Path
import tempfile
import unittest

import app
from tunix.agent import optimize
from tunix.application import run_optimization
from tunix.integration import select_winner
from tunix.validator import CalibrationBlocked, calibrate, validate
from test_application import completion, runner, workload
from test_integration import DEFAULT, NAIVE, benchmark, outputs


class TruncationReportTests(unittest.TestCase):
    def test_default_eight_of_twenty_blocks_calibration(self):
        base = outputs('default', DEFAULT)
        for d in base['docs'][:8]:
            d['n_output_tokens'] = 128
        with self.assertRaises(CalibrationBlocked) as caught:
            calibrate(base, outputs('naive', NAIVE))
        self.assertEqual(caught.exception.reasons, ['truncation'])
        self.assertEqual(len(caught.exception.diagnostics), 8)
        self.assertTrue(all(d['baseline'] == 'competent_default' for d in caught.exception.diagnostics))

    def test_reported_candidate_counts_never_accepted_even_with_identical_text(self):
        base = outputs('default', DEFAULT)
        cal = calibrate(base, outputs('naive', NAIVE))
        for config, count in ((dict(DEFAULT, max_new_tokens=64), 10),
                              (dict(DEFAULT, batch=4), 8), (dict(DEFAULT, batch=8), 7)):
            candidate = outputs('candidate', config)
            for d in candidate['docs'][:count]:
                d['n_output_tokens'] = config['max_new_tokens']
            def evaluate(c):
                return dict(benchmark=benchmark('candidate', c), validation=validate(candidate, base, cal))
            history = optimize([config], evaluate, completion=completion)
            self.assertFalse(history[0]['accepted'])
            self.assertIn('truncation', history[0]['reasons'])
            self.assertIsNone(select_winner(history))

    def test_application_persists_explicit_block_and_never_calls_agent(self):
        calls = []
        def truncated(c, docs, **kw):
            result = runner(c, docs, **kw)
            if c == DEFAULT:
                for d in result['outputs']['docs'][:8]:
                    d['n_output_tokens'] = 128
            return result
        def forbidden(messages):
            calls.append(messages)
            raise AssertionError('agent must not run before calibration')
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(CalibrationBlocked):
                run_optimization(tmp, runner=truncated, workload=workload, completion=forbidden)
            session = next(Path(tmp).glob('run_*'))
            status = json.loads((session / 'session.json').read_text())
            self.assertEqual(status['state'], 'blocked')
            self.assertIsNone(status['winner'])
            self.assertEqual(status['reasons'], ['truncation'])
            self.assertIn('Calibration BLOCKED', app.build_dashboard_view(app.load_dashboard_data(session))['banner'])
            self.assertFalse((session / 'agent_log.json').exists())
        self.assertEqual(calls, [])
