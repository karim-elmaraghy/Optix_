"""Developer-installed Tunix workflow; stdlib-only imports for doctor/init.

Editable installation is the MVP delivery: python -m pip install -e .
No browser GPU detection, arbitrary model discovery, secrets in project config,
automatic model activation, or PyPI publishing is implemented.
"""
import argparse
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid

MODEL = 'meta-llama/Llama-3.2-1B-Instruct'
WORKLOAD = 'cnn_dailymail'
SETUP = ('Use a CUDA Tesla T4 (Colab: Runtime > Change runtime type > T4 GPU). '
         'In the active Python environment run: python -m pip install -e "' + str(Path(__file__).resolve().parents[1]) + '[gpu]". '
         'Accept the Hugging Face Llama-3.2-1B-Instruct license and set HF_TOKEN '
         'to a read token. For optimize also set MODEL_API_KEY; never put keys in tunix.json.')


def inspect_environment():
    info = dict(python=sys.version.split()[0], executable=sys.executable, gpu=[],
                cuda_available=False, torch_cuda=None, frameworks={},
                hf_token_set=bool(os.environ.get('HF_TOKEN')),
                model_api_key_set=bool(os.environ.get('MODEL_API_KEY')),
                supported_model=MODEL, supported_workload=WORKLOAD)
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,driver_version',
                                 '--format=csv,noheader,nounits'], capture_output=True,text=True,timeout=5)
        if result.returncode == 0:
            for line in result.stdout.splitlines():
                name,vram,driver = [part.strip() for part in line.rsplit(',',2)]
                info['gpu'].append(dict(name=name,vram_mib=float(vram),driver_version=driver))
    except (OSError,subprocess.TimeoutExpired,ValueError):
        pass
    for name in ('torch','transformers','datasets','nvidia-ml-py','gradio'):
        try: info['frameworks'][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: info['frameworks'][name] = None
    if info['frameworks']['torch']:
        try:
            import torch
            info['cuda_available'] = bool(torch.cuda.is_available())
            info['torch_cuda'] = torch.version.cuda
            if info['cuda_available']:
                info['torch_device_0'] = torch.cuda.get_device_name(0)
        except Exception as exc:
            info['torch_probe_error'] = type(exc).__name__
    info['compatible'] = bool(info['cuda_available'] and 'T4' in info.get('torch_device_0','') and
                              all(info['frameworks'][n] for n in ('torch','transformers','datasets','nvidia-ml-py')))
    return info


def project_config(path):
    path = Path(path).resolve()
    if not path.is_file():
        raise ValueError('No project config. Run tunix init in your project, or supply --config PATH.')
    config = json.loads(path.read_text(encoding='utf-8'))
    if set(config) != {'model','workload','results_dir'} or config['model'] != MODEL or config['workload'] != WORKLOAD:
        raise ValueError('Unsupported project config. MVP supports only T4 / '+MODEL+' / CNN-DailyMail; no arbitrary project model is inspected.')
    if not isinstance(config['results_dir'],str) or not config['results_dir'].strip():
        raise ValueError('results_dir must be a nonempty path')
    return config, (path.parent / config['results_dir']).resolve()


def require_runtime(optimize=False):
    info = inspect_environment()
    if not info['compatible']:
        raise ValueError('Current runtime is not compatible with the supported T4 benchmark. '+SETUP)
    if not info['hf_token_set']:
        raise ValueError('HF_TOKEN is missing. '+SETUP)
    if optimize and not info['model_api_key_set']:
        raise ValueError('MODEL_API_KEY is missing. Set it securely in the terminal environment. '+SETUP)


def command_benchmark(root):
    """Invoke Agent A's real Gate 1 implementation; retain failures as JSON."""
    from .run import main as benchmark_main
    from .application import prompt_version, write_json
    session = root / ('benchmark_' + uuid.uuid4().hex)
    session.mkdir(parents=True)
    write_json(session/'session.json',dict(kind='measured',state='running',prompt_version=prompt_version(),winner=None,
                                         source='tunix benchmark; real execution, not Meta search'))
    write_json(root/'current_session.json',dict(directory=session.name))
    try:
        code = benchmark_main(['--gate1','--repeats','3','--results-dir',str(session)])
    except Exception as exc:
        write_json(session/'session.json',dict(kind='measured',state='failed',prompt_version=prompt_version(),winner=None,
                                             error=type(exc).__name__,source='tunix benchmark failed before completion'))
        raise
    tables = sorted(session.glob('benchmark_*.json'))
    items = json.loads(tables[-1].read_text(encoding='utf-8')) if tables else []
    failed = not items or any(r.get('status') != 'success' for r in items)
    state = 'failed' if code or failed else 'measured_not_validated'
    write_json(session/'session.json',dict(kind='measured',state=state,prompt_version=prompt_version(),winner=None,
                                         source='tunix benchmark; real execution, not Meta search'))
    print('Measured benchmark evidence: '+str(session))
    print('No quality-approved winner. Next: tunix optimize; tunix report.')
    return 2 if code or failed else 0


def launch_report(results, port, browser=True, replay=False):
    # Set before import so the dashboard's default argument and constants both
    # use the developer project's result directory, not the installation repo.
    os.environ['TUNIX_RESULTS_DIR'] = str(results)
    os.environ['TUNIX_REPORT_MODE'] = 'replay' if replay else 'project'
    # A developer's project commonly contains its own app.py. Load our known
    # installation file explicitly; never execute that unrelated project file.
    app_path = Path(__file__).resolve().parents[1]/'app.py'
    spec = importlib.util.spec_from_file_location('app',app_path)
    app = importlib.util.module_from_spec(spec)
    sys.modules['app'] = app
    spec.loader.exec_module(app)
    app.RESULTS_DIR = Path(results)
    # UI passes explicit paths, so importing app previously is safe too.
    demo = app.create_app()
    print(f'Tunix report: http://127.0.0.1:{port} — data: {results}')
    demo.launch(server_name='127.0.0.1',server_port=port,share=False,inbrowser=browser,max_file_size="32mb",
                favicon_path=str(Path(app.__file__).parent/'assets'/'tunix-symbol.svg'),css=app.CSS)


def main(argv=None):
    parser=argparse.ArgumentParser(prog='tunix',description='Local GPU inference optimization agent. MVP: Tesla T4 / Llama 3.2 1B / CNN-DailyMail.')
    sub=parser.add_subparsers(dest='command',required=True)
    doctor=sub.add_parser('doctor',help='Inspect this execution environment, not browser hardware')
    doctor.add_argument('--json',action='store_true',help='Machine-readable diagnostics (no credential values)')
    init=sub.add_parser('init',help='Create credential-free tunix.json in your project')
    init.add_argument('--config',default='tunix.json')
    init.add_argument('--results-dir',default='.tunix/results')
    for name in ('benchmark','optimize','report','export'):
        cmd=sub.add_parser(name,help={'benchmark':'Measure real Gate 1 profiles (3 repeats)','optimize':'Run real baselines + strict quality gates + Meta search','report':'Launch local dashboard for this project','export':'Package the current measured run as a byte-preserving ZIP'}[name])
        cmd.add_argument('--config',default='tunix.json')
        if name=='optimize': cmd.add_argument('--budget',type=int,choices=range(1,9),default=8)
        if name=='export':
            cmd.add_argument('--session',help='Original results/session directory (legacy Colab)')
            cmd.add_argument('--output',default='tunix-results.zip',help='New ZIP path; never overwrites')
        if name=='report':
            cmd.add_argument('--replay',action='store_true',help='Explicitly view bundled measured V2 results instead of project results')
            cmd.add_argument('--port',type=int,default=7860)
            cmd.add_argument('--no-browser',action='store_true')
    args=parser.parse_args(argv)
    try:
        if args.command=='doctor':
            info=inspect_environment()
            if args.json: print(json.dumps(info,indent=2))
            else:
                print('Tunix doctor — execution environment, not browser GPU')
                print(json.dumps(info,indent=2))
                print('Compatible for live T4 workflow: '+str(info['compatible']))
                if not info['compatible']: print(SETUP)
                print('Model/token access and Meta service connectivity are NOT verified by doctor.')
            return 0 if info['compatible'] else 2
        if args.command=='init':
            path=Path(args.config)
            config=dict(model=MODEL,workload=WORKLOAD,results_dir=args.results_dir)
            # Exclusive create: never overwrite developer work or save keys.
            with path.open('x',encoding='utf-8') as stream: json.dump(config,stream,indent=2)
            print('Created '+str(path.resolve())+' (no credentials). Next: tunix doctor; tunix benchmark; tunix optimize; tunix report.')
            return 0
        if args.command=='report' and args.replay:
            results=Path(__file__).resolve().parents[1]/'results'
            print('SAVED V2 REPLAY — not a new benchmark or live Meta API optimization.')
        elif args.command=='export' and args.session:
            results=Path(args.session)
        else:
            _,results=project_config(args.config)
        if args.command=='export':
            from .archive import export_archive
            output=export_archive(results,args.output,session=args.session)
            print('Exported '+str(output)+' - original evidence preserved; no approval or new run implied.')
            return 0
        if args.command=='report':
            if not 1<=args.port<=65535: raise ValueError('port must be in 1..65535')
            launch_report(results,args.port,not args.no_browser,replay=args.replay)
            return 0
        require_runtime(optimize=args.command=='optimize')
        if args.command=='benchmark': return command_benchmark(results)
        from .application import run_optimization
        session=run_optimization(results,budget=args.budget)
        status=json.loads((session/'session.json').read_text(encoding='utf-8'))
        print('Optimization evidence: '+str(session))
        if not status.get('winner'):
            print('No validated tuning candidate; no approval or activation. Inspect tunix report.')
            return 2
        print('Tuning candidate found. Held-out confirmation required; nothing applied. Run tunix report.')
        return 0
    except (ValueError,OSError,ImportError) as exc:
        from .validator import CalibrationBlocked
        if isinstance(exc,CalibrationBlocked):
            print(str(exc)+'. Meta candidate search did not run; nothing approved or applied. Run tunix report.',file=sys.stderr)
        elif isinstance(exc,FileExistsError):
            print('Config already exists; left unchanged.',file=sys.stderr)
        elif isinstance(exc,ImportError):
            print('Missing runtime dependency. '+SETUP,file=sys.stderr)
        else:
            print('Tunix: '+str(exc),file=sys.stderr)
        return 2
    except Exception as exc:
        # External client errors can contain credential-bearing URLs: don't echo.
        print('Tunix stopped ('+type(exc).__name__+'). Check saved result errors and tunix doctor; no approval claimed.',file=sys.stderr)
        return 2


if __name__=='__main__':
    sys.exit(main())
