"""CLI dispatch/setup checks: no live GPU or API execution."""
import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tunix import cli
from tunix.validator import CalibrationBlocked


class CliTests(unittest.TestCase):
    def invoke(self,args):
        output=io.StringIO()
        with contextlib.redirect_stdout(output),contextlib.redirect_stderr(output):
            code=cli.main(args)
        return code,output.getvalue()

    def test_init_creates_no_secrets_and_never_overwrites(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'tunix.json'
            self.assertEqual(self.invoke(['init','--config',str(path)])[0],0)
            original=path.read_bytes()
            self.assertEqual(set(json.loads(original)),{'model','workload','results_dir'})
            self.assertEqual(self.invoke(['init','--config',str(path)])[0],2)
            self.assertEqual(path.read_bytes(),original)
            _,results=cli.project_config(path)
            self.assertEqual(results,Path(tmp)/'.tunix'/'results')

    def test_doctor_json_has_no_credential_values(self):
        info=dict(compatible=False,hf_token_set=True,model_api_key_set=True,gpu=[])
        with patch('tunix.cli.inspect_environment',return_value=info):
            code,text=self.invoke(['doctor','--json'])
        self.assertEqual(code,2)
        self.assertEqual(json.loads(text),info)

    def test_benchmark_invokes_real_gate1_entrypoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            def backend(args):
                self.assertEqual(args[:3],['--gate1','--repeats','3'])
                session=Path(args[-1])
                (session/'benchmark_fixture.json').write_text(json.dumps([dict(status='success')]))
                return 0
            with patch('tunix.run.main',side_effect=backend) as run:
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(cli.command_benchmark(root),0)
                run.assert_called_once()
            pointer=json.loads((root/'current_session.json').read_text())
            self.assertEqual(json.loads((root/pointer['directory']/'session.json').read_text())['state'],'measured_not_validated')

    def test_optimize_routes_to_backend_and_returns_blocked(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'tunix.json'
            self.invoke(['init','--config',str(path)])
            error=CalibrationBlocked([dict(baseline='default',doc_id='fixture',reasons=['truncation'])])
            with patch('tunix.cli.require_runtime'),patch('tunix.application.run_optimization',side_effect=error) as run:
                code,text=self.invoke(['optimize','--config',str(path),'--budget','2'])
            self.assertEqual(code,2)
            self.assertIn('Calibration blocked',text)
            run.assert_called_once_with(Path(tmp)/'.tunix'/'results',budget=2)

    def test_report_resolves_project_data_and_replay_is_explicit(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'tunix.json'
            self.invoke(['init','--config',str(path)])
            with patch('tunix.cli.launch_report') as report:
                self.assertEqual(self.invoke(['report','--config',str(path),'--port','7870','--no-browser'])[0],0)
                report.assert_called_once_with(Path(tmp)/'.tunix'/'results',7870,False,replay=False)
            with patch('tunix.cli.launch_report'):
                code,text=self.invoke(['report','--replay','--no-browser'])
                self.assertEqual(code,0)
                self.assertIn('not a new benchmark',text)

    def test_missing_runtime_has_actionable_error_without_benchmark(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'tunix.json'
            self.invoke(['init','--config',str(path)])
            with patch('tunix.cli.inspect_environment',return_value=dict(compatible=False)),patch('tunix.cli.command_benchmark') as run:
                code,text=self.invoke(['benchmark','--config',str(path)])
            self.assertEqual(code,2)
            self.assertIn('Tesla T4',text)
            self.assertIn('pip install',text)
            run.assert_not_called()
