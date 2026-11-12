"""SYNTHETIC CPU contract tests, NOT measured GPU results.

All timings/energy below are arbitrary test sentinels. These fixtures must
never be used in results/, a performance table, or a hackathon claim.
Run: python -m unittest discover -s tests -p 'test_*.py' -v
"""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from tunix.agent import optimize
from tunix.safe_apply import safe_apply
from tunix.validator import calibrate, validate

DEFAULT = dict(dtype='float16', attn='sdpa', batch=1, compile=False, max_new_tokens=128)
NAIVE = dict(DEFAULT, dtype='float32', attn='eager')
TEXT = ('City officials opened a new public library on Monday after residents raised funds. '
        'The building offers reading rooms and free classes for local families. '
        'Volunteers will help visitors during the first month of operation.')


def json_copy(value):
    """Exercise JSON serialization exactly as artifact loading does."""
    return json.loads(json.dumps(value))


def outputs(name, config, split='tuning', text=TEXT):
    return json_copy(dict(profile_name=name, config=config, docs=[dict(
        doc_id=f'{split}-{i}', split=split, output_text=text,
        n_output_tokens=8 if config['max_new_tokens'] == 8 else 45,
        gold_summary=TEXT) for i in range(20 if split == 'tuning' else 10)]))


def benchmark(name, config, status='success', throughput=2.0):
    # Complete Issue #2 item, including nullable measurements on failure.
    item = dict(profile_name=name, status=status, error=None,
                requested_config=config, actual_config=config, repeats=3,
                total_time_sec=dict(median=10.0, min=9.9, max=10.1),
                latency_p50_sec=0.5, latency_p95_sec=0.6, docs_per_sec=throughput,
                tokens_per_sec=90.0, joules_per_doc=5.0, idle_power_w=10.0,
                compile_time_sec=0.0, noisy=False)
    if status == 'failed':
        item.update(error='SYNTHETIC CUDA out of memory', actual_config=None,
                    docs_per_sec=None, tokens_per_sec=None, joules_per_doc=None,
                    total_time_sec=dict(median=None, min=None, max=None))
    return json_copy(item)


class SyntheticIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.default = outputs('default_fp16_sdpa_b1', DEFAULT)
        naive = outputs('naive_fp32_eager_b1', NAIVE, text=TEXT.replace('new', 'modern'))
        self.cal = calibrate(self.default, naive, margin=0.02)
        self.heldout = outputs('default_fp16_sdpa_b1', DEFAULT, 'heldout')

    def adapter(self, table, bundles):
        """Integration-owned example: extract ONE item from the saved list.

        Failed profiles may have no output file. Preserve their real error
        and reject them without attempting to load an absent output bundle.
        """
        def evaluate(config):
            matches = [r for r in table if r['requested_config'] == config]
            if len(matches) != 1:
                raise ValueError('need exactly one matching benchmark item')
            result = matches[0]
            if result['status'] != 'success':
                decision = dict(accepted=False, reasons=['execution_failed'])
            else:
                bundle = bundles[result['profile_name']]
                if bundle['config'] != config:
                    decision = dict(accepted=False, reasons=['output_config_mismatch'])
                else:
                    decision = validate(bundle, self.default, self.cal)
            return dict(benchmark=result, validation=decision)
        return evaluate

    def test_full_json_artifacts_connect_and_failed_profile_cannot_win(self):
        failed, good = dict(DEFAULT, batch=8), dict(DEFAULT, batch=4)
        table = json_copy([benchmark('failed', failed, 'failed'), benchmark('good', good)])
        evaluate = self.adapter(table, {'good': outputs('good', good)})
        history = optimize([failed, good], evaluate, completion=lambda _: '{}')
        self.assertFalse(history[0]['accepted'])
        self.assertIn('execution_failed', history[0]['reasons'])
        self.assertEqual(history[0]['result']['benchmark']['error'], 'SYNTHETIC CUDA out of memory')
        # Winner selection belongs to the caller; never sort unfiltered records.
        eligible = [r for r in history if r['accepted']]
        winner = max(eligible, key=lambda r: r['result']['benchmark']['docs_per_sec'])
        self.assertEqual(winner['proposal']['config'], good)

    def test_success_with_invalid_summaries_is_rejected_and_logged(self):
        candidate = dict(DEFAULT, batch=4)
        for text in ('', 'Short summary.', 'nonsense ' * 40, ' '.join(reversed(TEXT.split()))):
            with self.subTest(text=text[:20]), tempfile.TemporaryDirectory() as tmp:
                evaluate = self.adapter([benchmark('bad', candidate)], {'bad': outputs('bad', candidate, text=text)})
                path = os.path.join(tmp, 'rejections.jsonl')
                record = optimize([candidate], evaluate, completion=lambda _: '{}', log_path=path)[0]
                self.assertFalse(record['accepted'])
                self.assertTrue(record['reasons'])
                with open(path) as stream:
                    self.assertEqual(json.load(stream)['reasons'], record['reasons'])

    def test_run_config_mapping_adapter_ignores_console_diagnostics(self):
        # Mirrors PR #5 run_config's {benchmark, outputs, console} return.
        candidate = dict(DEFAULT, batch=4)
        outcome = json_copy(dict(benchmark=benchmark('candidate', candidate),
                                 outputs=outputs('candidate', candidate),
                                 console=dict(recompilations=None, break_even_docs=None,
                                              static_cache_used='not_attempted',
                                              energy_poll_joules_per_doc=None,
                                              energy_crosscheck_ok=None)))
        def evaluate(config):
            self.assertEqual(config, outcome['benchmark']['requested_config'])
            return dict(benchmark=outcome['benchmark'],
                        validation=validate(outcome['outputs'], self.default, self.cal))
        record = optimize([candidate], evaluate, completion=lambda _: '{}')[0]
        self.assertTrue(record['accepted'])
        self.assertNotIn('console', record['result']['benchmark'])

    def test_failed_execution_overrides_even_an_accepted_quality_decision(self):
        def evaluate(config):
            return dict(benchmark=benchmark('failed', config, 'failed'),
                        validation=dict(accepted=True, reasons=[]))
        record = optimize([DEFAULT], evaluate, completion=lambda _: '{}')[0]
        self.assertFalse(record['accepted'])
        self.assertIn('execution_failed', record['reasons'])

    def test_runtime_and_output_config_mismatch_rejected(self):
        candidate = dict(DEFAULT, batch=4)
        result = benchmark('candidate', candidate)
        result['actual_config'] = DEFAULT
        evaluate = self.adapter([result], {'candidate': outputs('candidate', candidate)})
        self.assertIn('config_mismatch', optimize([candidate], evaluate, completion=lambda _: '{}')[0]['reasons'])
        evaluate = self.adapter([benchmark('candidate', candidate)], {'candidate': outputs('candidate', DEFAULT)})
        self.assertIn('output_config_mismatch', optimize([candidate], evaluate, completion=lambda _: '{}')[0]['reasons'])

    def test_real_api_path_reads_environment_with_mock_transport_only(self):
        # A dummy key and fake HTTP transport prove wiring, NOT live API access.
        candidate = dict(DEFAULT, batch=4)
        captured = []
        class Response:
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def read(self):
                proposal = json.dumps(dict(config=candidate, reason='Synthetic integration choice'))
                return json.dumps(dict(choices=[dict(finish_reason='stop', message=dict(content=proposal))])).encode()
        def transport(request, timeout):
            self.assertEqual(request.get_header('Authorization'), 'Bearer SYNTHETIC-KEY-NOT-A-CREDENTIAL')
            self.assertEqual(request.full_url, 'https://api.meta.ai/v1/chat/completions')
            self.assertEqual(timeout, 45)
            captured.append(json.loads(request.data))
            return Response()
        evaluate = self.adapter([benchmark('candidate', candidate)], {'candidate': outputs('candidate', candidate)})
        with patch.dict(os.environ, {'MODEL_API_KEY': 'SYNTHETIC-KEY-NOT-A-CREDENTIAL'}), patch('urllib.request.urlopen', side_effect=transport):
            history = optimize([candidate], evaluate)
        self.assertEqual(len(captured), 1)
        self.assertEqual(history[0]['proposal']['source'], 'meta_api')
        self.assertNotIn('SYNTHETIC-KEY-NOT-A-CREDENTIAL', json.dumps(history))

    def test_unavailable_api_labels_fallback_without_network(self):
        evaluate = self.adapter([benchmark('default', DEFAULT)], {'default': self.default})
        with patch.dict(os.environ, {}, clear=True), patch('urllib.request.urlopen') as network:
            record = optimize([DEFAULT], evaluate)[0]
        network.assert_not_called()
        self.assertEqual(record['proposal']['source'], 'rule_fallback')
        self.assertNotIn('model', record['proposal'])

    def test_bad_profile_rolls_back_runtime_and_persisted_config(self):
        bad = dict(DEFAULT, max_new_tokens=8)
        active = dict(DEFAULT)
        calls = []
        def apply(config):
            active.clear()
            active.update(config)
            calls.append(copy.deepcopy(config))
        def evaluate(config):
            return dict(benchmark=benchmark('bad', config),
                        outputs=outputs('bad', config, 'heldout', 'Only eight tokens of a deliberately truncated summary'))
        with tempfile.TemporaryDirectory() as tmp:
            path, log = os.path.join(tmp, 'active.json'), os.path.join(tmp, 'apply.jsonl')
            event = safe_apply(bad, DEFAULT, self.heldout, self.cal, apply, evaluate, path, log)
            self.assertEqual(event['action'], 'rolled_back')
            self.assertIn('short', event['validation']['reasons'])
            self.assertIn('truncation', event['validation']['reasons'])
            with open(path) as stream: self.assertEqual(json.load(stream), DEFAULT)
            with open(log) as stream: self.assertEqual(json.load(stream)['action'], 'rolled_back')
        self.assertEqual(active, DEFAULT)
        self.assertEqual(calls, [bad, DEFAULT])

    def test_valid_winner_promotes_on_disjoint_heldout(self):
        candidate = dict(DEFAULT, batch=4)
        calls = []
        def evaluate(config):
            return dict(benchmark=benchmark('winner', config), outputs=outputs('winner', config, 'heldout'))
        event = safe_apply(candidate, DEFAULT, self.heldout, self.cal, calls.append, evaluate)
        self.assertEqual(event['action'], 'promoted')
        self.assertEqual(len(event['validation']['docs']), 10)
        self.assertEqual(calls, [candidate])

    def test_imports_do_not_load_gpu_or_optional_dependencies(self):
        script = '''
import sys
class BlockGPU:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch','transformers','datasets','pynvml','openai','gradio'}:
            raise AssertionError('optional dependency imported: ' + fullname)
sys.meta_path.insert(0, BlockGPU())
import tunix.validator, tunix.agent, tunix.safe_apply, tunix.integration
'''
        completed = subprocess.run([sys.executable, '-c', script], cwd=Path(__file__).resolve().parents[1],
                                   capture_output=True, text=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)
