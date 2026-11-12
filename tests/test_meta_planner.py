import json
import hashlib
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tunix.meta_planner import main, plan, v2_evidence
from tests.test_agent import C


class PlannerTests(unittest.TestCase):
    def test_v2_archive_is_read_only_and_checks_hashes_and_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            def save(name, value):
                path = root / name
                path.write_text(json.dumps(value), encoding='utf-8')
                return hashlib.sha256(path.read_bytes()).hexdigest()
            history = [dict(profile_name='default', requested_config=C, docs_per_sec=1.2)]
            bundle = dict(profile_name='default', config=C, docs=[dict(split='tuning')])
            hashes = {'benchmark.json': save('benchmark.json', history),
                      'outputs_default.json': save('outputs_default.json', bundle)}
            save('session.json', dict(kind='measured', prompt_version='v2-chat-template', state='blocked', winner=None))
            save('validation_report.json', dict(prompt_version='v2-chat-template', artifact_sha256=hashes, calibration='blocked'))
            before = {p.name: p.read_bytes() for p in root.iterdir()}
            loaded, context, files = v2_evidence(root / 'benchmark.json')
            self.assertEqual(loaded, history)
            self.assertEqual(context['session']['state'], 'blocked')
            self.assertEqual(files['benchmark']['sha256'], hashes['benchmark.json'])
            self.assertEqual(before, {p.name: p.read_bytes() for p in root.iterdir()})
            save('benchmark.json', [])
            with self.assertRaises(ValueError):
                v2_evidence(root / 'benchmark.json')
            save('benchmark.json', history)
            bundle['docs'][0]['split'] = 'heldout'
            hashes['outputs_default.json'] = save('outputs_default.json', bundle)
            save('validation_report.json', dict(prompt_version='v2-chat-template', artifact_sha256=hashes))
            with self.assertRaises(ValueError):
                v2_evidence(root / 'benchmark.json')

    def test_cannot_write_into_original_archive(self):
        with tempfile.TemporaryDirectory() as tmp, patch('sys.stderr'):
            with self.assertRaises(SystemExit):
                main(['--v2-benchmark', str(Path(tmp) / 'benchmark.json'),
                      '--output', str(Path(tmp) / 'proposal.json')])

    def test_advisory_and_actual_submission(self):
        measured = dict(C, batch=4)
        history = [dict(requested_config=measured, docs_per_sec=1.2, split='tuning')]
        response = dict(content=json.dumps(dict(config=C, reason='Test batch hypothesis')),
                        http_status=200, model='muse-spark-1.3')
        with patch('tunix.meta_planner.meta_response', return_value=response) as call:
            result = plan(history, [C, measured])
        sent = json.loads(call.call_args.args[0][1]['content'])
        self.assertEqual(sent['history'], history)
        self.assertEqual(sent['remaining'], [C])
        self.assertEqual(result['source'], 'meta_api')
        for field in ('benchmarked', 'quality_approved', 'deployed'):
            self.assertFalse(result[field])

    def test_invalid_proposal_never_falls_back(self):
        with patch('tunix.meta_planner.meta_response', return_value=dict(content='{}')):
            with self.assertRaises(ValueError):
                plan([dict(docs_per_sec=1.2)], [C])

    def test_heldout_rejected_before_network(self):
        with patch('tunix.meta_planner.meta_response') as call:
            with self.assertRaises(ValueError):
                plan([dict(split='heldout')], [C])
        call.assert_not_called()

    def test_missing_key_is_unverified(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {}, clear=True):
            output = Path(tmp) / 'ping.json'
            with patch('builtins.print'):
                self.assertEqual(main(['--ping', '--output', str(output)]), 1)
            self.assertEqual(json.loads(output.read_text())['status'], 'unverified')

    def test_error_text_never_exposed(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {'MODEL_API_KEY': 'private-test-key'}):
            output = Path(tmp) / 'ping.json'
            with patch('tunix.meta_planner.meta_response', side_effect=RuntimeError('private-test-key')), patch('builtins.print'):
                self.assertEqual(main(['--ping', '--output', str(output)]), 1)
            self.assertNotIn('private-test-key', output.read_text())
