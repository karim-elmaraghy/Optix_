"""Functional product-shell checks; no live API or GPU measurements."""
import json
import os
from pathlib import Path
import unittest
import io
import contextlib
from unittest.mock import patch

import app
from tunix import workbench_ui as ui
from tunix.validator import CalibrationBlocked


class WorkbenchTests(unittest.TestCase):
    def test_landing_workbench_and_action_connections(self):
        demo=app.create_app()
        config=demo.get_config_file()
        self.assertIn('Tunix',config['title'])
        components={c['id']:c for c in config['components']}
        def button(prefix):
            return next(c['id'] for c in components.values() if c['type']=='button' and str(c['props'].get('value','')).startswith(prefix))
        def callback(prefix):
            target=button(prefix)
            dep=next(d for d in config['dependencies'] if any(t[0]==target and t[1]=='click' for t in d['targets']))
            return demo.fns[dep['id']].fn,dep
        fn,dep=callback('Open Workbench')
        updates=fn()
        self.assertFalse(updates[0]['visible'])
        self.assertTrue(updates[1]['visible'])
        self.assertEqual(len(dep['outputs']),2)
        fn,dep=callback('Optimize')
        self.assertEqual(len(dep['inputs']),5)
        values=fn(ui.REPLAY,ui.MODEL,ui.WORKLOAD,'','')
        self.assertIn('No Meta request',values[0])
        self.assertEqual(values[1:],('',''))
        self.assertIs(callback('Export measured candidate')[0],ui.export_candidate)
        self.assertIs(callback('Check Apply readiness')[0],app.safe_apply)
        self.assertEqual(len(callback('Run held-out confirmation')[1]['inputs']),2)
        passwords=[c for c in components.values() if c['type']=='textbox' and c['props'].get('type')=='password']
        self.assertEqual(len(passwords),2)
        self.assertTrue(all(not c['props'].get('value') for c in passwords))

    def test_real_replay_statistics_and_detection(self):
        data=app.load_dashboard_data(follow_session=False)
        cards,charts=ui.stats_and_charts(data)
        self.assertIn('5.63\u00d7',cards)
        self.assertIn('81.8%',cards)
        self.assertIn('not approval',cards)
        self.assertIn('Withheld',cards)
        self.assertIn('raw measured comparison',charts)
        with patch('tunix.workbench_ui.subprocess.run') as probe:
            probe.return_value.returncode=0
            probe.return_value.stdout='NVIDIA GeForce RTX 4060\n'
            detected=ui.runtime_status()
            self.assertIn('RTX 4060',detected)
            self.assertIn('Only a connected Tesla T4',detected)
            self.assertIn('server, not your browser',detected)

    def test_replay_never_calls_live_backend(self):
        with patch('tunix.application.run_optimization') as run:
            message=ui.run_configured(ui.REPLAY,ui.MODEL,ui.WORKLOAD,'DUMMY-KEY','DUMMY-HF')
            self.assertIn('No Meta request',message)
            run.assert_not_called()

    def test_credentials_restore_after_block_and_do_not_leak(self):
        blocked=CalibrationBlocked([dict(baseline='competent_default',doc_id='fixture',reasons=['truncation'])])
        captured=io.StringIO()
        with patch.dict(os.environ,{'MODEL_API_KEY':'original','HF_TOKEN':'original-hf'}), patch('tunix.application.run_optimization',side_effect=blocked), contextlib.redirect_stdout(captured), contextlib.redirect_stderr(captured):
            message=ui.run_configured(ui.LIVE,ui.MODEL,ui.WORKLOAD,'DUMMY-KEY','DUMMY-HF')
            self.assertIn('Calibration blocked',message)
            self.assertNotIn('DUMMY',message)
            self.assertEqual(os.environ['MODEL_API_KEY'],'original')
            self.assertEqual(os.environ['HF_TOKEN'],'original-hf')
        self.assertNotIn('DUMMY',captured.getvalue())
        self.assertTrue(app.RUN_LOCK.acquire(blocking=False))
        app.RUN_LOCK.release()

    def test_missing_keys_and_unsupported_models_do_not_run(self):
        with patch.dict(os.environ,{},clear=True),patch('tunix.application.run_optimization') as run:
            self.assertIn('Credentials needed',ui.run_configured(ui.LIVE,ui.MODEL,ui.WORKLOAD,'',''))
            self.assertIn('Unsupported',ui.run_configured(ui.LIVE,'unsupported',ui.WORKLOAD,'',''))
            run.assert_not_called()

    def test_export_is_measured_settings_not_approval(self):
        snapshot=app.build_dashboard_view(app.load_dashboard_data(follow_session=False))['snapshot']
        path=ui.export_candidate(snapshot)
        export=json.loads(Path(path).read_text())
        self.assertFalse(export['quality_approved'])
        self.assertEqual(export['config']['batch'],8)
        self.assertEqual(export['original_repeats'],2)
        self.assertIn('run_config',export['reproduce_python'])
        self.assertNotIn('MODEL_API_KEY',json.dumps(export))

    def test_no_winner_blocks_heldout_and_samples_do_not_claim_t5(self):
        self.assertIn('blocked',ui.confirm_with_token({'chosen':None},''))
        pipeline=ui.pipeline_html(dict(has_real_benchmarks=False))
        self.assertIn('T5 not verified',pipeline)
        self.assertNotIn('T5 consistency passes',pipeline)
