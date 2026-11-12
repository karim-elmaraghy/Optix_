"""Bounded, data-only archive transport. Uploaded claims never confer approval.

No extractall, code execution, trusted uploaded decisions or historical writes.
All originals are retained under evidence/ and copied byte-exactly as needed
by the existing dashboard. Derived metadata is computed locally, not imported.
"""
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import re
import stat
import struct
import uuid
import zipfile

from .agent import check_config, execution_reasons
from .integration import batching_reasons
from .validator import CalibrationBlocked, calibrate, index_docs, invalid_reasons, rouge_l, token_f1, tokens, validate

MAX_ZIP = 32 * 1024 * 1024
MAX_FILE = 8 * 1024 * 1024
MAX_TOTAL = 64 * 1024 * 1024
MAX_FILES = 128
MAX_RATIO = 100
MAX_ANALYSIS_CELLS = 50_000_000
METADATA = {'session.json','validation_report.json','confirmation_context.json',
            'agent_log.json','apply_log.jsonl','dashboard_apply_log.jsonl','tunix_export.json'}


def allowed_name(name):
    if not isinstance(name,str) or '\\' in name or ':' in name or '\x00' in name:
        return False
    path=PurePosixPath(name)
    if path.is_absolute() or '..' in path.parts or str(path)!=name:
        return False
    if len(path.parts)>2 or (len(path.parts)==2 and path.parts[0]!='heldout'):
        return False
    leaf=path.name
    return bool(re.fullmatch(r'(?:benchmark|outputs)_[A-Za-z0-9_-]{1,100}\.json',leaf) or
                (len(path.parts)==1 and leaf in METADATA) or name=='heldout/confirmation.json')


def strict_json(raw):
    def pairs(items):
        obj={}
        for k,v in items:
            if k in obj: raise ValueError('duplicate JSON key')
            obj[k]=v
        return obj
    def constant(_): raise ValueError('nonfinite JSON value')
    try:
        return json.loads(raw.decode('utf-8-sig'),object_pairs_hook=pairs,parse_constant=constant)
    except (UnicodeError,RecursionError,json.JSONDecodeError) as exc:
        raise ValueError('invalid UTF-8 JSON') from exc


def parse_payloads(payloads):
    parsed={}; analysis_cells=0
    for name,raw in payloads.items():
        if name.endswith('.jsonl') or PurePosixPath(name).name=='agent_log.json':
            try: value=strict_json(raw)
            except ValueError:
                value=[strict_json(line) for line in raw.splitlines() if line.strip()]
            if not isinstance(value,(dict,list)): raise ValueError('invalid metadata log')
        else:
            value=strict_json(raw)
        parsed[name]=value
        if PurePosixPath(name).name.startswith('benchmark_'):
            if not isinstance(value,list) or not value or len(value)>128: raise ValueError('invalid benchmark list')
            seen=set()
            required={'profile_name','status','error','requested_config','actual_config','repeats','total_time_sec',
                      'latency_p50_sec','latency_p95_sec','docs_per_sec','tokens_per_sec','joules_per_doc','idle_power_w','compile_time_sec','noisy'}
            for item in value:
                if not isinstance(item,dict) or not required.issubset(item): raise ValueError('benchmark schema mismatch')
                profile=item['profile_name']
                if not isinstance(profile,str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,100}',profile) or profile in seen:
                    raise ValueError('invalid/duplicate profile name')
                seen.add(profile)
                check_config(item['requested_config'])
                if item['actual_config'] is not None: check_config(item['actual_config'])
                if item['status'] not in ('success','failed') or type(item['noisy']) is not bool:
                    raise ValueError('invalid benchmark status')
                if type(item['repeats']) is not int or not 1<=item['repeats']<=100:
                    raise ValueError('invalid repeats')
                if item['error'] is not None and not isinstance(item['error'],str): raise ValueError('invalid error')
                times=item['total_time_sec']
                if not isinstance(times,dict) or set(times)!= {'median','min','max'}: raise ValueError('invalid timing schema')
                for number in [*times.values(),*(item[k] for k in ('latency_p50_sec','latency_p95_sec','docs_per_sec','tokens_per_sec','joules_per_doc','idle_power_w','compile_time_sec'))]:
                    if number is not None and (type(number) not in (int,float) or not math.isfinite(number)):
                        raise ValueError('invalid numeric measurement')
        elif PurePosixPath(name).name.startswith('outputs_'):
            if not isinstance(value,dict) or not {'profile_name','config','docs'}.issubset(value): raise ValueError('output schema mismatch')
            check_config(value['config'])
            if PurePosixPath(name).name != 'outputs_'+str(value['profile_name'])+'.json': raise ValueError('output name mismatch')
            docs=value['docs']
            if not isinstance(docs,list) or len(docs)>30: raise ValueError('invalid output count for supported 20/10 workload')
            if docs:
                split=docs[0].get('split') if isinstance(docs[0],dict) else None
                if split not in ('tuning','heldout'): raise ValueError('invalid output split')
                try:
                    for d in docs:
                        if not isinstance(d,dict) or not {'doc_id','split','output_text','n_output_tokens','gold_summary'}.issubset(d):
                            raise ValueError('output document schema mismatch')
                        if type(d['doc_id']) not in (str,int): raise ValueError('invalid document ID')
                        if not isinstance(d['output_text'],str) or not isinstance(d['gold_summary'],str):
                            raise ValueError('invalid text')
                        # Bound quadratic ROUGE-L work as well as ZIP expansion.
                        if len(d['output_text'])>16384 or len(d['gold_summary'])>16384:
                            raise ValueError('summary exceeds supported analysis limits')
                        n_output,n_gold=len(tokens(d['output_text'])),len(tokens(d['gold_summary']))
                        if n_output>512 or n_gold>512: raise ValueError('summary token analysis limit exceeded')
                        analysis_cells+=n_output*(n_gold+512)
                        if analysis_cells>MAX_ANALYSIS_CELLS: raise ValueError('archive exceeds quality analysis work budget')
                    index_docs(value,split)
                except (KeyError,TypeError,AttributeError) as exc:
                    raise ValueError('invalid output document') from exc
    if not any(n.startswith('benchmark_') for n in parsed): raise ValueError('no root benchmark JSON')
    manifest=parsed.get('tunix_export.json')
    if manifest is not None:
        expected={n:hashlib.sha256(raw).hexdigest() for n,raw in payloads.items() if n!='tunix_export.json'}
        if not isinstance(manifest,dict) or manifest.get('format')!='tunix-results-v1' or manifest.get('files')!=expected:
            raise ValueError('export manifest/hash mismatch')
    return parsed


def read_archive(path):
    path=Path(path)
    if not path.is_file() or path.stat().st_size>MAX_ZIP: raise ValueError('ZIP exceeds 32 MiB or is missing')
    # Bound central-directory object allocation before ZipFile parses entries.
    with path.open('rb') as stream:
        stream.seek(max(0,path.stat().st_size-65557)); tail=stream.read(65557)
    offset=tail.rfind(b'PK\x05\x06')
    if offset<0 or len(tail)-offset<22: raise ValueError('missing ZIP end record')
    _,disk,start_disk,disk_count,count,central_size,central_offset,comment_size=struct.unpack('<4s4H2LH',tail[offset:offset+22])
    if (disk or start_disk or disk_count!=count or not 1<=count<=MAX_FILES or
        central_size>MAX_FILES*4096 or central_offset==0xFFFFFFFF or
        central_offset+central_size>path.stat().st_size or offset+22+comment_size!=len(tail)):
        raise ValueError('unsupported ZIP layout or entry count')
    try:
        with zipfile.ZipFile(path) as archive:
            entries=archive.infolist()
            if not 1<=len(entries)<=MAX_FILES: raise ValueError('archive entry count exceeds limit')
            total=0; seen=set(); payloads={}
            for entry in entries:
                name=entry.filename
                if not allowed_name(name) or name.casefold() in seen: raise ValueError('unsafe, unsupported or duplicate archive path')
                seen.add(name.casefold())
                mode=entry.external_attr>>16
                if stat.S_IFMT(mode) not in (0,stat.S_IFREG): raise ValueError('symlinks/directories/special files are forbidden')
                if entry.is_dir() or entry.flag_bits&1 or entry.compress_type not in (zipfile.ZIP_STORED,zipfile.ZIP_DEFLATED):
                    raise ValueError('unsupported archive entry')
                total+=entry.file_size
                if entry.file_size>MAX_FILE or total>MAX_TOTAL or entry.file_size/max(entry.compress_size,1)>MAX_RATIO:
                    raise ValueError('archive size/expansion limit exceeded')
                with archive.open(entry) as stream: raw=stream.read(MAX_FILE+1)
                if len(raw)!=entry.file_size or len(raw)>MAX_FILE: raise ValueError('entry length mismatch')
                payloads[name]=raw
    except (zipfile.BadZipFile,RuntimeError,NotImplementedError,EOFError,OSError) as exc:
        raise ValueError('invalid or unsupported ZIP') from exc
    return payloads,parse_payloads(payloads)


def derived_report(payloads, parsed):
    tables=[n for n in parsed if n.startswith('benchmark_')]
    if len(tables)!=1: raise ValueError('import one benchmark run/table at a time')
    table=parsed[tables[0]]
    by_name={r['profile_name']:r for r in table}
    bundles={v['profile_name']:v for n,v in parsed.items() if n.startswith('outputs_')}
    manifest=parsed.get('tunix_export.json',{})
    supplied=parsed.get('session.json',{})
    if not isinstance(supplied,dict): raise ValueError('invalid session metadata')
    version=manifest.get('prompt_version') or supplied.get('prompt_version')
    if version is not None and (not isinstance(version,str) or not re.fullmatch(r'[A-Za-z0-9_.-]{1,80}',version)):
        raise ValueError('invalid prompt-version metadata')
    # Original six-file V2 ZIP has no metadata; identify only its exact known
    # content hashes, not its filename or a user-provided claim of approval.
    known_path=Path(__file__).resolve().parents[1]/'results'/'validation_report.json'
    if not version and known_path.is_file():
        known=strict_json(known_path.read_bytes()).get('artifact_sha256',{})
        actual={n:hashlib.sha256(b).hexdigest() for n,b in payloads.items() if n.startswith(('benchmark_','outputs_'))}
        if actual==known: version='v2-chat-template'
    synthetic=(supplied.get('kind')=='synthetic' or manifest.get('kind')=='synthetic' or
               'benchmark_sample.json' in payloads)
    default=bundles.get('default_fp16_sdpa_b1'); naive=bundles.get('naive_fp32_eager_b1')
    reasons=[]; diagnostics=[]; calibration=None
    if synthetic: reasons.append('synthetic_evidence')
    if version not in ('v1-raw','v2-chat-template'): reasons.append('prompt_version_unknown')
    if manifest.get('prompt_version') and supplied.get('prompt_version') and manifest['prompt_version']!=supplied['prompt_version']:
        reasons.append('prompt_version_conflict')
    if not default or not naive: reasons.append('missing_baselines')
    else:
        from .profiles import PROFILES
        for name,bundle in [('default_fp16_sdpa_b1',default),('naive_fp32_eager_b1',naive)]:
            if bundle['config']!=PROFILES[name]: reasons.append('baseline_config_mismatch')
            if name not in by_name or execution_reasons(by_name[name],bundle['config']): reasons.append('baseline_execution_failed')
            if len(bundle['docs'])!=20: reasons.append('tuning_document_count_not_20')
        try: calibration=calibrate(default,naive,margin=0.02)
        except CalibrationBlocked as exc:
            reasons.extend(exc.reasons); diagnostics=exc.diagnostics
        except (ValueError,KeyError,TypeError,AttributeError): reasons.append('invalid_baseline_artifacts')
    profiles={}
    for item in table:
        name=item['profile_name']; bundle=bundles.get(name)
        failures=execution_reasons(item,item['requested_config'])
        flagged=[]; scores=[]; gold=[]; t5=None
        if not bundle:
            failures.append('missing_outputs')
        elif bundle['config']!=item['requested_config']:
            failures.append('output_config_mismatch')
        else:
            if not bundle['docs']: failures.append('empty_outputs')
            for d in bundle['docs']:
                flags=invalid_reasons(d,bundle['config'])
                if flags: flagged.append(dict(doc_id=d['doc_id'],reasons=flags))
            failures.extend(r for row in flagged for r in row['reasons'])
            if default and bundle['docs']:
                try:
                    docs=index_docs(bundle,'tuning'); base=index_docs(default,'tuning')
                    if docs.keys()!=base.keys() or any(docs[k]['gold_summary']!=base[k]['gold_summary'] for k in base):
                        failures.append('document_or_gold_mismatch')
                    else:
                        scores=[token_f1(docs[k]['output_text'],base[k]['output_text']) for k in base]
                        gold=[rouge_l(d['output_text'],d['gold_summary']) for d in docs.values()]
                except (ValueError,KeyError,TypeError): failures.append('invalid_tuning_outputs')
            if bundle['config']['batch']>1:
                cfg=dict(bundle['config'],batch=1)
                refs=[b for n,b in bundles.items() if b['config']==cfg and n in by_name and not execution_reasons(by_name[n],cfg)]
                t5_reasons=batching_reasons(bundle,refs[0]) if refs else ['T5_reference_missing']
                failures.extend(t5_reasons)
                t5=dict(passed=not t5_reasons,min_f1=None,mean_f1=None)
                if refs:
                    try:
                        a=index_docs(bundle,'tuning'); b=index_docs(refs[0],'tuning')
                        if a.keys()==b.keys():
                            values=[token_f1(a[k]['output_text'],b[k]['output_text']) for k in a]
                            t5.update(min_f1=min(values),mean_f1=sum(values)/len(values))
                    except (ValueError,KeyError,TypeError):
                        pass
        decision=None
        if calibration is not None and not reasons and bundle:
            decision=validate(bundle,default,calibration)
            failures.extend(decision['reasons'])
        profiles[name]=dict(config=item['requested_config'],reasons=sorted(set(failures)),flagged_docs=flagged,
            doc_count=len(bundle['docs']) if bundle else 0,t5=t5,quality_approved=False,
            tuning_validation=decision,mean_token_f1_vs_default=sum(scores)/len(scores) if scores else None,
            mean_gold_rouge_l=sum(gold)/len(gold) if gold else None,timing_repeats_sufficient=item['repeats']>=3)
    report=dict(source='locally recomputed import validation; not Meta optimization',prompt_version=version,
                artifact_sha256={n:hashlib.sha256(b).hexdigest() for n,b in payloads.items() if n.startswith(('benchmark_','outputs_'))},
                calibration='blocked' if reasons else 'calibrated',profiles=profiles,
                remaining_checks=['No full approval inferred from import; heldout/speed confirmation is not verified',
                                  'T1 independent determinism evidence and T2 console crosscheck not verified']+
                                  (['T3 requires at least 3 repeats; supplied run is insufficient'] if any(r['repeats']<3 for r in table) else []))
    session=dict(kind='synthetic' if synthetic else 'imported',state='blocked' if reasons else 'validation_pending',
                 prompt_version=version or 'unknown',winner=None,reasons=sorted(set(reasons)),diagnostics=diagnostics,
                 source='Imported GPU results — archive evidence, not live Meta optimization',
                 message='Calibration blocked: '+', '.join(sorted(set(reasons))) if reasons else 'Tuning calibration computed; full approval/heldout not established')
    return report,session


def import_archive(path, results_root):
    payloads,parsed=read_archive(path)
    report,session=derived_report(payloads,parsed)
    root=Path(results_root).resolve(); root.mkdir(parents=True,exist_ok=True)
    directory=root/('import_'+uuid.uuid4().hex)
    directory.mkdir()  # exclusive new session, never overwrites historical data
    for name,raw in payloads.items():
        target=directory/'evidence'/name
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(raw)
        if '/' not in name and name.startswith(('benchmark_','outputs_')):
            (directory/name).write_bytes(raw)
    (directory/'validation_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8',newline='\n')
    # Commit marker last: incomplete IO sessions do not appear in the selector.
    (directory/'session.json').write_text(json.dumps(session,indent=2),encoding='utf-8',newline='\n')
    return directory


def export_archive(results_root, output, session=None):
    root=Path(session).resolve() if session else Path(results_root).resolve()
    if not session and (root/'current_session.json').is_file():
        pointer=strict_json((root/'current_session.json').read_bytes()).get('directory','')
        child=(root/pointer).resolve()
        if child.parent!=root or not child.is_dir(): raise ValueError('invalid current session pointer')
        root=child
    # Re-export imported evidence without serializing derived approval claims.
    evidence=root/'evidence' if (root/'evidence').is_dir() else root
    payloads={}; total=0
    for p in list(evidence.glob('*'))+list((evidence/'heldout').glob('*')):
        if p.is_symlink(): raise ValueError('symlinks cannot be exported')
        if p.is_file() and allowed_name(p.relative_to(evidence).as_posix()) and p.name!='tunix_export.json':
            if p.stat().st_size>MAX_FILE: raise ValueError('source file exceeds export limit')
            total+=p.stat().st_size
            if total>MAX_TOTAL or len(payloads)>=MAX_FILES-1: raise ValueError('export input exceeds transport limits')
            payloads[p.relative_to(evidence).as_posix()]=p.read_bytes()
    parsed=parse_payloads(payloads)
    meta=parsed.get('session.json',{})
    if not isinstance(meta,dict) or meta.get('kind')=='synthetic' or 'benchmark_sample.json' in payloads:
        raise ValueError('synthetic session cannot be exported as measured evidence')
    version=meta.get('prompt_version')
    if not version:
        version=derived_report(payloads,parsed)[1]['prompt_version']
    manifest=dict(format='tunix-results-v1',kind=meta.get('kind','reported_measurement'),prompt_version=version,
                  source='Tunix export; original measurements are unchanged; hardware not reverified by export',
                  files={n:hashlib.sha256(b).hexdigest() for n,b in payloads.items()})
    data=io.BytesIO()
    with zipfile.ZipFile(data,'w',compression=zipfile.ZIP_DEFLATED) as archive:
        for name,raw in payloads.items(): archive.writestr(name,raw)
        archive.writestr('tunix_export.json',json.dumps(manifest,indent=2).encode())
    if len(data.getvalue())>MAX_ZIP: raise ValueError('export exceeds 32 MiB')
    # Same bounds as the importer, including expansion ratio.
    with zipfile.ZipFile(io.BytesIO(data.getvalue())) as archive:
        entries=archive.infolist()
        if len(entries)>MAX_FILES or sum(e.file_size for e in entries)>MAX_TOTAL or any(e.file_size>MAX_FILE or e.file_size/max(e.compress_size,1)>MAX_RATIO for e in entries):
            raise ValueError('export exceeds archive transport limits')
    output=Path(output)
    with output.open('xb') as stream: stream.write(data.getvalue())
    return output.resolve()
