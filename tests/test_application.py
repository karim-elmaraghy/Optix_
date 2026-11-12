"""SYNTHETIC combined application fixtures; no GPU numbers or API claims."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import app
from tunix.application import confirm_heldout, run_optimization
from tunix.integration import make_evaluator, select_winner
from tunix.agent import optimize
from tunix.validator import calibrate
from test_integration import DEFAULT, NAIVE, TEXT, benchmark, outputs


def workload():
    return tuple([dict(d, prompt='Summarize SYNTHETIC fixture') for d in outputs('fixture', DEFAULT, split)['docs']]
                 for split in ('tuning', 'heldout'))


def runner(config, docs, **kwargs):
    name, split = kwargs['profile_name'], kwargs['split']
    result = dict(benchmark=benchmark(name, config), outputs=outputs(name, config, split), console={})
    path = Path(kwargs['results_dir'])
    path.mkdir(exist_ok=True, parents=True)
    (path / ('outputs_' + name + '.json')).write_text(json.dumps(result['outputs']), encoding='utf-8')
    return result


def completion(messages):
    return json.dumps(dict(config=json.loads(messages[-1]['content'])['remaining'][0], reason='SYNTHETIC proposal'))


class ApplicationTests(unittest.TestCase):
    def test_complete_synthetic_session_logs_and_heldout(self):
        with tempfile.TemporaryDirectory() as tmp:
            session = run_optimization(tmp, 1, runner=runner, workload=workload, completion=completion, space=[DEFAULT])
            self.assertFalse((Path(tmp) / 'current_session.json').exists())
            self.assertEqual(json.loads((session / 'session.json').read_text())['kind'], 'synthetic')
            data = app.load_dashboard_data(session)
            self.assertFalse(data['has_real_benchmarks'])
            self.assertIn('SAMPLE DATA', app.build_dashboard_view(data)['banner'])
            result = confirm_heldout(session, runner=runner, workload=workload)
            self.assertTrue(result['validation']['accepted'])
            self.assertEqual(len(result['validation']['docs']), 10)
            self.assertIn('not attempted', result['activation'])
            self.assertEqual(len(result['baselines']), 2)

    def test_t5_cannot_be_weakened_by_quality_calibration(self):
        baseline = outputs('default', DEFAULT)
        cal = calibrate(baseline, outputs('naive', NAIVE), margin=0.3)
        def broken(config, docs, **kwargs):
            result = runner(config, docs, **kwargs)
            if config['batch'] > 1:
                for d in result['outputs']['docs']:
                    d['output_text'] = TEXT.replace('City officials opened a new', 'Local neighbors built another large')
            return result
        with tempfile.TemporaryDirectory() as tmp:
            evaluate = make_evaluator(workload()[0], baseline, cal, runner=broken, results_dir=tmp)
            history = optimize([dict(DEFAULT, batch=8)], evaluate, completion=completion)
            self.assertIn('T5_batching_failed', history[0]['reasons'])
            self.assertIsNone(select_winner(history))

    def test_baseline_failure_is_saved_and_search_stops(self):
        def fail(config, docs, **kwargs):
            return dict(benchmark=benchmark(kwargs['profile_name'], config, 'failed'), outputs=outputs('empty', config))
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(RuntimeError, 'baseline failed'):
                run_optimization(tmp, runner=fail, workload=workload, completion=completion)
            session = next(Path(tmp).glob('run_*'))
            self.assertEqual(json.loads((session / 'session.json').read_text())['state'], 'failed')
            self.assertIn('out of memory', json.loads((session / 'benchmark_session.json').read_text())[0]['error'])

    def test_no_key_fails_before_loading_model(self):
        with patch.dict('os.environ', {}, clear=True), patch('tunix.benchmark.run_config') as run:
            with self.assertRaisesRegex(RuntimeError, 'MODEL_API_KEY'):
                run_optimization()
            run.assert_not_called()

    def test_determinism_failure_stops_search(self):
        def unstable(config, docs, **kwargs):
            result = runner(config, docs, **kwargs)
            if kwargs['profile_name'].endswith('_repeat'):
                result['outputs']['docs'][0]['output_text'] = 'SYNTHETIC changed output'
            return result
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(RuntimeError, 'T1 determinism'):
                run_optimization(tmp, runner=unstable, workload=workload, completion=completion)

    def test_failed_heldout_has_no_acceptance_or_activation(self):
        with tempfile.TemporaryDirectory() as tmp:
            session = run_optimization(tmp, 1, runner=runner, workload=workload, completion=completion, space=[DEFAULT])
            def fail(config, docs, **kwargs):
                result = runner(config, docs, **kwargs)
                result['benchmark']['status'] = 'failed'
                result['benchmark']['error'] = 'SYNTHETIC heldout failure'
                return result
            result = confirm_heldout(session, runner=fail, workload=workload)
            self.assertFalse(result['validation']['accepted'])
            self.assertIn('execution_failed', result['validation']['reasons'])
            self.assertIn('not attempted', result['activation'])

    def test_dashboard_reject_flag_beats_explicit_selection(self):
        row = benchmark('fast_b8', dict(DEFAULT, batch=8), throughput=1000.0)
        step = dict(chosen=True, config=row['requested_config'], decision='REJECT', reason='T5_batching_failed')
        self.assertIsNone(app._find_chosen([row], [step]))
        self.assertIsNone(app._find_chosen([dict(row, winner=True)], []))

    def test_optimize_ui_calls_real_application_path(self):
        with patch('tunix.application.run_optimization', return_value=Path('measured-session')) as run, patch('app._read_json', return_value=dict(winner=None)):
            self.assertIn('NO validated winner', app.optimize_action())
            run.assert_called_once_with(app.RESULTS_DIR)

    def test_prompt_version_mismatch_blocks_confirmation_before_gpu_work(self):
        with tempfile.TemporaryDirectory() as tmp:
            session = run_optimization(tmp, 1, runner=runner, workload=workload, completion=completion, space=[DEFAULT])
            with patch('tunix.application.prompt_version', return_value='synthetic-different-version'):
                with self.assertRaisesRegex(ValueError, 'prompt version changed'):
                    confirm_heldout(session, runner=runner, workload=workload)
