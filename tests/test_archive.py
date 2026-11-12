"""Original real V2 artifact transport tests + deliberately malicious ZIPs.

No synthetic GPU measurements; malformed archives are security fixtures only.
"""
import hashlib
import io
import json
from pathlib import Path
import stat
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import app
from tunix import archive as transport
from tunix import workbench_ui as ui

ROOT=Path(__file__).resolve().parents[1]
NAMES=['benchmark_20261104T211417.json','outputs_default_fp16_sdpa_b1.json','outputs_naive_fp32_eager_b1.json',
       'outputs_fp16_sdpa_b4.json','outputs_fp16_sdpa_b8.json','outputs_fp16_sdpa_b1_cap64.json']


def write_zip(path,entries):
    with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED) as z:
        for name,data in entries:
            z.writestr(name,data)
    return path


def original_zip(path):
    return write_zip(path,[(name,(ROOT/'results'/name).read_bytes()) for name in NAMES])


class ArchiveTests(unittest.TestCase):
    def test_original_v2_byte_preservation_and_gates(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=original_zip(Path(tmp)/'original.zip')
            first=transport.import_archive(path,Path(tmp)/'imports')
            second=transport.import_archive(path,Path(tmp)/'imports')
            self.assertNotEqual(first,second)
            for name in NAMES:
                self.assertEqual((first/'evidence'/name).read_bytes(),(ROOT/'results'/name).read_bytes())
                self.assertEqual((first/name).read_bytes(),(ROOT/'results'/name).read_bytes())
            data=app.load_dashboard_data(first,follow_session=False)
            self.assertFalse(data['errors'])
            self.assertEqual(data['session_status']['kind'],'imported')
            self.assertEqual(data['session_status']['prompt_version'],'v2-chat-template')
            self.assertEqual(data['session_status']['state'],'blocked')
            self.assertIsNone(data['session_status']['winner'])
            b8=data['artifact_validation']['profiles']['fp16_sdpa_b8']
            self.assertTrue(b8['t5']['passed'])
            self.assertEqual(b8['t5']['min_f1'],1.0)
            self.assertIn('truncation',b8['reasons'])
            self.assertFalse(b8['quality_approved'])

    def test_tunix_export_roundtrip_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'tunix.zip'
            transport.export_archive(ROOT/'results',out)
            before=out.read_bytes()
            with self.assertRaises(FileExistsError): transport.export_archive(ROOT/'results',out)
            self.assertEqual(out.read_bytes(),before)
            destination=transport.import_archive(out,Path(tmp)/'imports')
            manifest=json.loads((destination/'evidence'/'tunix_export.json').read_text())
            self.assertEqual(manifest['format'],'tunix-results-v1')
            for name in NAMES:
                self.assertEqual((destination/'evidence'/name).read_bytes(),(ROOT/'results'/name).read_bytes())

    def test_upload_callback_returns_updated_dashboard_and_selection(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(app,'RESULTS_DIR',Path(tmp)/'project-results'):
            path=original_zip(Path(tmp)/'original.zip')
            demo=app.create_app()
            config=demo.get_config_file()
            dep=next(d for d in config['dependencies'] if d['api_name']=='import_benchmark_results')
            result=demo.fns[dep['id']].fn(str(path))
            self.assertEqual(len(result),16)
            selection=result[0]['value']
            self.assertTrue(selection.startswith('import_'))
            self.assertIn((selection),[value for _,value in ui.session_choices()])
            self.assertEqual(result[1]['value'],ui.IMPORTED)
            self.assertIn('Imported GPU results',result[2])
            self.assertIn('5.63\u00d7',result[3])
            self.assertIn('81.8%',result[3])
            self.assertIn('IMPORTED GPU RESULTS',result[-1])
            rows={r[0]:r for r in result[7]}
            self.assertEqual(rows['fp16_sdpa_b8'][3],'5.63x')
            self.assertEqual(rows['fp16_sdpa_b8'][6],'Rejected')
            self.assertIsNone(result[6]['chosen'])
            self.assertIn('not live Meta',result[-1])
            self.assertEqual(ui.session_data(':bundled')['results_dir'],str(ROOT/'results'))
            with self.assertRaises(ValueError): ui.session_data('../outside')

    def test_traversal_symlinks_duplicates_and_unsupported_entries(self):
        bad_names=['../evil.json','/evil.json','C:/evil.json','outputs_../x.json','..\\evil.json','heldout/../session.json','a//session.json','code.py']
        with tempfile.TemporaryDirectory() as tmp:
            for i,name in enumerate(bad_names):
                with self.subTest(name=name):
                    p=write_zip(Path(tmp)/f'bad{i}.zip',[(name,b'{}')])
                    with self.assertRaises(ValueError): transport.import_archive(p,Path(tmp)/'imports')
            info=zipfile.ZipInfo('session.json');info.create_system=3;info.external_attr=(stat.S_IFLNK|0o777)<<16
            for entries in ([(info,b'/outside')],[('session.json',b'{}'),('SESSION.JSON',b'{}')]):
                p=write_zip(Path(tmp)/'bad.zip',entries)
                with self.assertRaises(ValueError): transport.read_archive(p)
            self.assertFalse((Path(tmp)/'imports').exists())

    def test_bombs_oversize_entry_count_and_truncated_zip(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=write_zip(Path(tmp)/'bomb.zip',[('session.json',b' '*500000)])
            with self.assertRaises(ValueError): transport.read_archive(p)
            p=original_zip(Path(tmp)/'original.zip')
            for setting in ('MAX_FILE','MAX_TOTAL','MAX_ZIP','MAX_FILES'):
                with self.subTest(setting=setting),patch.object(transport,setting,1):
                    with self.assertRaises(ValueError): transport.read_archive(p)
            p=Path(tmp)/'truncated.zip';p.write_bytes(b'PK\x03\x04broken')
            with self.assertRaises(ValueError): transport.read_archive(p)

    def test_malformed_json_schema_and_manifest_rejected_before_storage(self):
        originals={name:(ROOT/'results'/name).read_bytes() for name in NAMES}
        variants=[]
        malformed=dict(originals);malformed[NAMES[0]]=b'not JSON';variants.append(malformed)
        duplicate=dict(originals);duplicate['session.json']=b'{"kind":"measured","kind":"synthetic"}';variants.append(duplicate)
        bad=dict(originals);items=json.loads(bad[NAMES[0]]);items[0]['repeats']='two';bad[NAMES[0]]=json.dumps(items).encode();variants.append(bad)
        bad=dict(originals);bad['tunix_export.json']=b'{"format":"tunix-results-v1","files":{}}';variants.append(bad)
        bad=dict(originals);bundle=json.loads(bad[NAMES[1]]);bundle['docs'][0]['n_output_tokens']=True;bad[NAMES[1]]=json.dumps(bundle).encode();variants.append(bad)
        bad=dict(originals);bundle=json.loads(bad[NAMES[1]]);bundle['docs'][0]['output_text']='a.'*600;bad[NAMES[1]]=json.dumps(bundle).encode();variants.append(bad)
        with tempfile.TemporaryDirectory() as tmp:
            for i,files in enumerate(variants):
                p=write_zip(Path(tmp)/f'bad{i}.zip',files.items())
                with self.assertRaises(ValueError): transport.import_archive(p,Path(tmp)/'imports')
            self.assertFalse((Path(tmp)/'imports').exists())

    def test_untrusted_approval_metadata_never_becomes_winner(self):
        with tempfile.TemporaryDirectory() as tmp:
            entries=[(name,(ROOT/'results'/name).read_bytes()) for name in NAMES]
            entries += [('session.json',json.dumps(dict(kind='measured',prompt_version='v2-chat-template',winner={'approved':True})).encode()),
                        ('agent_log.json',b'{"accepted":true,"decision":"PROMOTE"}')]
            imported=transport.import_archive(write_zip(Path(tmp)/'claims.zip',entries),Path(tmp)/'imports')
            data=app.load_dashboard_data(imported,follow_session=False)
            self.assertEqual(data['agent_steps'],[])
            self.assertIsNone(app.build_dashboard_view(data)['snapshot']['chosen'])
            self.assertTrue((imported/'evidence'/'agent_log.json').is_file())

    def test_synthetic_metadata_is_never_upgraded_to_measurement(self):
        with tempfile.TemporaryDirectory() as tmp:
            entries=[(name,(ROOT/'results'/name).read_bytes()) for name in NAMES]
            entries.append(('session.json',b'{"kind":"synthetic","prompt_version":"v2-chat-template"}'))
            imported=transport.import_archive(write_zip(Path(tmp)/'synthetic.zip',entries),Path(tmp)/'imports')
            data=app.load_dashboard_data(imported,follow_session=False)
            self.assertFalse(data['has_real_benchmarks'])
            self.assertEqual(data['session_status']['kind'],'synthetic')
            self.assertIn('synthetic_evidence',data['session_status']['reasons'])
            with self.assertRaises(ValueError): transport.export_archive(imported,Path(tmp)/'not-measured.zip',session=imported)
