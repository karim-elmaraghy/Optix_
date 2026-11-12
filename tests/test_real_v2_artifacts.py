"""Offline tests of ORIGINAL user-supplied T4 JSON, not synthetic GPU data.

No model/API runs; these tests verify saved measurements and derived gates.
"""
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import app
from tunix.integration import batching_reasons
from tunix.validator import CalibrationBlocked, calibrate, index_docs, token_f1

ROOT = Path(__file__).resolve().parents[1] / 'results'


def load(name):
    return json.loads((ROOT / name).read_text(encoding='utf-8'))


class RealV2ArtifactsTests(unittest.TestCase):
    def test_real_measurements_display_raw_speedup_and_rejections(self):
        data = app.load_dashboard_data(ROOT)
        self.assertTrue(data['has_real_benchmarks'])
        self.assertEqual(len(data['artifact_validation']['artifact_sha256']), 6)
        view = app.build_dashboard_view(data)
        rows = {r[0]: r for r in view['profile_rows']}
        b8 = rows['fp16_sdpa_b8']
        self.assertEqual(b8[1:4], ['1.982', '19.667', '5.63x'])
        self.assertEqual(b8[6], 'Rejected')
        self.assertIn('T5 PASS (min F1 1.000)', b8[7])
        self.assertIn('Calibration BLOCKED', view['banner'])
        self.assertIn('No configuration selected', view['winner_markdown'])
        self.assertEqual(len(view['rejection_rows']), 5)
        self.assertEqual(data['agent_steps'], [])  # no fake Meta proposals

    def test_actual_baselines_block_and_ui_cannot_report_success(self):
        with self.assertRaises(CalibrationBlocked) as caught:
            calibrate(load('outputs_default_fp16_sdpa_b1.json'), load('outputs_naive_fp32_eager_b1.json'))
        diagnostics = caught.exception.diagnostics
        default_ids = [d['doc_id'] for d in diagnostics if d['baseline'] == 'competent_default']
        naive_ids = [d['doc_id'] for d in diagnostics if d['baseline'] == 'naive']
        self.assertEqual(default_ids, ['cnn-test-1','cnn-test-2','cnn-test-12','cnn-test-15','cnn-test-17'])
        self.assertEqual(naive_ids, ['cnn-test-2','cnn-test-12','cnn-test-15','cnn-test-17'])
        self.assertEqual(caught.exception.reasons, ['truncation'])
        with patch('tunix.application.run_optimization', side_effect=caught.exception):
            message = app.optimize_action()
            self.assertIn('Calibration blocked', message)
            self.assertNotIn('candidate selected', message)
        self.assertIn('no profile was activated', app.safe_apply(app.build_dashboard_view(app.load_dashboard_data(ROOT))['snapshot']))

    def test_t5_pass_is_exactly_reproduced_from_real_outputs(self):
        default = load('outputs_default_fp16_sdpa_b1.json')
        base = index_docs(default, 'tuning')
        for name in ('fp16_sdpa_b4', 'fp16_sdpa_b8'):
            candidate = load('outputs_' + name + '.json')
            self.assertEqual(batching_reasons(candidate, default), [])
            docs = index_docs(candidate, 'tuning')
            self.assertEqual(min(token_f1(docs[k]['output_text'], base[k]['output_text']) for k in base), 1.0)
