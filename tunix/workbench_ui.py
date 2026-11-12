"""Product shell around existing, unchanged optimization and quality backend.

Only T4/Llama/CNN-DailyMail are supported. Saved evidence is explicit replay.
No GPU/Gradio imports at module level. Password inputs are request-scoped;
their temporary environment bridge is restored under the GPU operation lock.
"""
import html
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
CSS = (ROOT / 'assets' / 'workbench.css').read_text(encoding='utf-8')
MARK = (ROOT / 'assets' / 'tunix-symbol.svg').read_text(encoding='utf-8')
MODEL = 'meta-llama/Llama-3.2-1B-Instruct'
WORKLOAD = 'CNN/DailyMail summarization · 20 tuning / 10 held-out'
REPLAY = 'Saved T4 results'
LIVE = 'Connected T4 runtime'
IMPORTED = 'Imported GPU results'


def session_choices():
    import app
    choices=[('Bundled V2 replay',':bundled'),('Latest local measurements',':local')]
    if app.RESULTS_DIR.is_dir():
        for p in sorted(app.RESULTS_DIR.iterdir(),key=lambda p:p.stat().st_mtime,reverse=True):
            if p.is_dir() and (p/'session.json').is_file() and p.name.startswith(('import_','run_','benchmark_')):
                label='Imported GPU results' if p.name.startswith('import_') else 'Local measurement'
                try:
                    if json.loads((p/'session.json').read_text(encoding='utf-8')).get('kind')=='synthetic': label='Imported synthetic data'
                except (ValueError,OSError,AttributeError):
                    continue
                choices.append((label+' · '+p.name[-8:],p.name))
    return choices


def session_data(selection):
    import app
    if selection==':bundled': return app.load_dashboard_data(ROOT/'results',follow_session=False)
    if selection==':local': return app.load_dashboard_data(app.RESULTS_DIR)
    allowed={value for _,value in session_choices() if not value.startswith(':')}
    if selection not in allowed: raise ValueError('Unknown session selection')
    path=(app.RESULTS_DIR/selection).resolve()
    if path.parent!=app.RESULTS_DIR.resolve(): raise ValueError('Invalid session path')
    return app.load_dashboard_data(path,follow_session=False)


def view_outputs(selection, current_mode=REPLAY):
    import app
    data=session_data(selection)
    current=app.build_dashboard_view(data)
    cards,graphs=stats_and_charts(data)
    kind=data.get('session_status',{}).get('kind')
    label=('IMPORTED GPU RESULTS · archive evidence, not live Meta optimization' if kind=='imported' else
           'IMPORTED SYNTHETIC DATA · not GPU measurements' if kind=='synthetic' else
           'SAVED T4 REPLAY · no live Meta optimization' if selection==':bundled' else
           'LOCAL MEASUREMENT EVIDENCE · no new benchmark or Meta call implied')
    return (cards,graphs,status_panel(data),current['snapshot'],current['profile_rows'],current['winner_markdown'],current['banner'],current['baseline_rows'],current['rejection_rows'],current['agent_rows'],current['apply_summary'],pipeline_html(data),'<div class="source-note">'+label+'</div>')


def import_upload(path):
    """Actual upload handler: imports then returns the fully refreshed view."""
    import app
    from .archive import import_archive
    directory=import_archive(path,app.RESULTS_DIR)
    return directory.name,view_outputs(directory.name,IMPORTED)


def runtime_status():
    """Detect the server GPU, never imply access to the visitor's machine."""
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
                                capture_output=True, text=True, timeout=3)
        gpu = result.stdout.strip() if result.returncode == 0 else ''
    except (OSError, subprocess.TimeoutExpired):
        gpu = ''
    versions = []
    for package in ('torch', 'transformers'):
        try:
            versions.append(package + ' ' + importlib.metadata.version(package))
        except importlib.metadata.PackageNotFoundError:
            versions.append(package + ' not installed')
    return ('**Server GPU:** ' + (gpu or 'No NVIDIA GPU detected') +
            '\n\n' + ' · '.join(versions) +
            '\n\nOnly a connected Tesla T4 is supported for live execution. This detects the server, not your browser’s GPU.')


def stats_and_charts(data, *, landing=False):
    import app
    items = [r for r in data['benchmarks'] if r.get('status') == 'success']
    by_name = {r['profile_name']: r for r in items}
    base, b8 = by_name.get(app.DEFAULT_PROFILE, {}), by_name.get('fp16_sdpa_b8', {})
    measured = data['has_real_benchmarks']
    speed = app._speedup(b8.get('docs_per_sec'), base.get('docs_per_sec')) if measured else None
    energy = app._percent_saving(b8.get('joules_per_doc'), base.get('joules_per_doc')) if measured else None
    cards = [('RAW THROUGHPUT', f'{speed:.2f}×' if speed else '—', 'Batch 8 vs competent default · not approval'),
             ('ENERGY REDUCTION', f'{energy:.1f}%' if energy is not None else '—', 'Measured joules per document'),
             ('QUALITY APPROVAL', 'Withheld' if data.get('session_status', {}).get('state') == 'blocked' else 'Pending', 'No configuration is approved'),
             ('EVIDENCE', 'Imported' if data.get('session_status',{}).get('kind')=='imported' else 'Measured' if measured else 'Sample',
              'Imported GPU results · not live Meta output' if data.get('session_status',{}).get('kind')=='imported' else 'Saved T4 replay · not live Meta output' if data.get('artifact_validation') else 'Recorded project evidence · not a new run')]
    if landing: cards=cards[:2]
    stats = '<div class="stat-grid'+(' landing-proof' if landing else '')+'">' + ''.join(
        f'<div class="stat"><div class="label">{a}</div><div class="value {"amber" if i==2 else ""}">{b}</div><div class="detail">{c}</div></div>'
        for i, (a,b,c) in enumerate(cards)) + '</div>'
    charts = '<div class="chart-grid">'
    for metric, title, unit in [('docs_per_sec','Throughput','docs/s'), ('joules_per_doc','Energy per document','J/doc')]:
        valid = [r for r in items if app._number(r.get(metric)) is not None]
        maximum = max((r[metric] for r in valid), default=1) or 1
        evidence_label = 'raw measured comparison, not quality approval' if measured else 'SAMPLE VALUES · NOT MEASURED'
        charts += f'<div class="chart-card"><h3>{title}</h3><div class="small-note">{unit} · {evidence_label}</div>'
        for r in valid:
            label = {'naive_fp32_eager_b1':'FP32 · B1', 'default_fp16_sdpa_b1':'FP16 · B1', 'fp16_sdpa_b4':'FP16 · B4', 'fp16_sdpa_b8':'FP16 · B8', 'fp16_sdpa_b1_cap64':'FP16 · cap64'}.get(r['profile_name'], r['profile_name'])
            charts += (f'<div class="bar-row"><span>{html.escape(label)}</span><div class="bar-track">'
                       f'<div class="bar {"emphasis" if r is b8 else ""}" style="width:{max(1,r[metric]/maximum*100):.2f}%"></div></div>'
                       f'<span class="bar-value">{r[metric]:.3f}</span></div>')
        charts += '</div>'
    return stats, charts + '</div>'


def status_panel(data):
    state = data.get('session_status', {})
    measured = data['has_real_benchmarks']
    blocked = state.get('state') == 'blocked'
    t5 = data.get('artifact_validation', {}).get('profiles', {}).get('fp16_sdpa_b8', {}).get('t5', {})
    meta_steps = [s for s in data.get('agent_steps', []) if s.get('source') == 'meta_api']
    rows = [('Baseline', 'Measured' if measured else 'Not measured', 'pass' if measured else ''),
            ('Meta search', 'Recorded Meta steps' if meta_steps else 'Not run / no Meta steps', ''),
            ('T5 batch consistency', 'Pass' if t5.get('passed') else 'Not verified', 'pass' if t5.get('passed') else ''),
            ('Quality gate', 'Approval withheld' if blocked else 'Pending', 'warn'),
            ('Approved winner', 'None', 'warn')]
    diagnostics = state.get('diagnostics', [])
    default_count = sum(d.get('baseline') == 'competent_default' for d in diagnostics)
    naive_count = sum(d.get('baseline') == 'naive' for d in diagnostics)
    details = (f'{default_count} of 20 baseline summaries reached the generation limit, so the results are measured but not approved for deployment. '
               f'FP32 also has {naive_count} invalid outputs. Strict calibration remains BLOCKED; T5 PASS is a separate check.') if blocked else 'No full quality approval is recorded.'
    if blocked and not diagnostics: details=html.escape(state.get('message','Calibration is blocked by missing evidence.'))
    return ('<div class="status-label">CURRENT STATE · SAVED EVIDENCE</div>' +
            ''.join(f'<div class="status-row"><span>{label}</span><span class="pill {css}">{value}</span></div>' for label,value,css in rows) +
            f'<div class="status-explainer">{details}</div>')


def pipeline_html(data):
    profile = data.get('artifact_validation', {}).get('profiles', {}).get('fp16_sdpa_b8', {})
    blocked = data.get('session_status', {}).get('state') == 'blocked'
    stages = [('01 · Measure', 'Baseline recorded' if data['has_real_benchmarks'] else 'Not measured'),
              ('02 · Compare', 'T5 consistency passes' if profile.get('t5', {}).get('passed') else 'T5 not verified'),
              ('03 · Validate', 'Truncation rejected' if 'truncation' in profile.get('reasons', []) else 'Quality pending'),
              ('04 · Approve', 'Calibration blocked' if blocked else 'Approval pending')]
    return '<div class="pipeline">' + '<span class="arrow">→</span>'.join(
        f'<span class="{"blocked" if i>1 else ""}">{label}<b>{state}</b></span>' for i,(label,state) in enumerate(stages)) + '</div>'


def confirm_with_token(snapshot, hf_token):
    import app
    if not snapshot or not snapshot.get('chosen'):
        return 'Held-out is blocked: no quality-validated tuning candidate exists.'
    if not app.RUN_LOCK.acquire(blocking=False):
        return 'Another GPU operation is running.'
    previous = os.environ.get('HF_TOKEN')
    try:
        if hf_token.strip(): os.environ['HF_TOKEN'] = hf_token.strip()
        if not os.environ.get('HF_TOKEN'):
            return 'Re-enter your HF token for held-out execution, or set HF_TOKEN on the server.'
        from .application import confirm_heldout
        result = confirm_heldout(snapshot['results_dir'])
        return ('Held-out quality accepted.' if result['validation']['accepted'] else 'Held-out rejected.') + ' No runtime activation was attempted.'
    except Exception as exc:
        return 'Held-out stopped (' + type(exc).__name__ + '). No approval or activation claimed.'
    finally:
        if previous is None: os.environ.pop('HF_TOKEN', None)
        else: os.environ['HF_TOKEN'] = previous
        app.RUN_LOCK.release()


def run_configured(mode, model, workload, meta_key, hf_token):
    """Secure bridge to existing env-based backend; no stored credential state."""
    import app
    if model != MODEL or workload != WORKLOAD:
        return 'Unsupported selection. Only the tested T4 / Llama 3.2 1B summarization path is available.'
    if mode != LIVE:
        return 'Saved-results replay loaded. No Meta request or GPU experiment was made. Choose Connected T4 runtime for real optimization.'
    if not app.RUN_LOCK.acquire(blocking=False):
        return 'Another GPU operation is running.'
    previous = {k: os.environ.get(k) for k in ('MODEL_API_KEY', 'HF_TOKEN')}
    try:
        if meta_key.strip(): os.environ['MODEL_API_KEY'] = meta_key.strip()
        if hf_token.strip(): os.environ['HF_TOKEN'] = hf_token.strip()
        if not os.environ.get('MODEL_API_KEY'):
            return 'Credentials needed: enter your Meta API key or set MODEL_API_KEY on the server. No experiment started.'
        if not os.environ.get('HF_TOKEN'):
            return 'Gated model access needed: accept the Llama license and enter an HF read token. No experiment started.'
        from .application import run_optimization
        from .validator import CalibrationBlocked
        try:
            session = run_optimization(app.RESULTS_DIR)
            status = json.loads((session/'session.json').read_text(encoding='utf-8'))
            return ('Tuning candidate measured; held-out approval is still required. Nothing applied.' if status.get('winner')
                    else 'Search finished: no validated winner. Nothing applied.')
        except CalibrationBlocked as exc:
            return str(exc) + '. Meta candidate search did not run. No profile was approved or applied.'
        except Exception as exc:
            return 'Execution stopped (' + type(exc).__name__ + '). Verify connected T4, dependencies, model access and saved failure records. No approval claimed.'
    finally:
        for key, value in previous.items():
            if value is None: os.environ.pop(key, None)
            else: os.environ[key] = value
        app.RUN_LOCK.release()


def export_candidate(snapshot):
    """Export measured candidate settings as NOT APPROVED, never a launch/apply action."""
    if not snapshot or not snapshot.get('has_real_benchmarks'):
        return None
    candidates = [r for r in snapshot['benchmarks'] if r.get('status') == 'success' and
                  type(r.get('docs_per_sec')) in (int,float) and r['docs_per_sec'] > 0 and
                  r.get('requested_config') == r.get('actual_config')]
    record = max(candidates, key=lambda r:r['docs_per_sec'], default=None)
    if record is None:
        return None
    payload = dict(model=MODEL, workload=WORKLOAD, prompt_version=snapshot.get('session_status', {}).get('prompt_version'),
                   config=record['requested_config'], quality_approved=False,
                   purpose='Reproduce this measured candidate, not an approved deployment',
                   original_repeats=record.get('repeats'),
                   limitations=['baseline calibration blocked by truncation', 'heldout not confirmed', 'no persistent runtime activation'],
                   instructions='See INTEGRATION.md and V2_VALIDATION.md. Use the same V2 baseline protocol; do not apply as an approved winner.')
    payload['reproduce_python'] = ('from tunix.benchmark import run_config\n'
        'import json\nconfig = json.loads(' + repr(json.dumps(record['requested_config'])) + ')\n'
        'outcome = run_config(config, profile_name="reproduced_candidate", split="tuning", '
        'repeats=' + str(record.get('repeats',2)) + ', results_dir="results/reproduction")\n')
    directory = Path(tempfile.mkdtemp(prefix='tunix-export-'))
    path = directory/'tunix-measured-candidate-NOT-APPROVED.json'
    path.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    return str(path)


def create_workbench():
    import app
    import gradio as gr
    default_mode = LIVE if os.environ.get('TUNIX_REPORT_MODE') == 'project' else REPLAY
    default_selection=':bundled' if default_mode==REPLAY else ':local'
    initial = session_data(default_selection)
    view = app.build_dashboard_view(initial)
    stats, charts = stats_and_charts(initial)
    landing_stats,_ = stats_and_charts(initial,landing=True)
    with gr.Blocks(title='Tunix. · Measure performance. Verify behavior.', analytics_enabled=False) as demo:
        gr.HTML(f'<div class="nav"><div class="brand">{MARK}<span>Tunix.</span></div><div class="nav-note">MEASURE PERFORMANCE. VERIFY BEHAVIOR.</div></div>')
        with gr.Group(visible=True) as landing:
            with gr.Row():
                with gr.Column(scale=6):
                    gr.HTML('<div class="landing-copy"><div class="eyebrow">LOCAL INTELLIGENCE. MEASURABLE IMPROVEMENT.</div><h1>Measure performance.<br>Verify behavior.</h1><p>An AI agent for your GPU runtime. Test inference configurations, compare real measurements, and keep correctness in control.</p></div>')
                    with gr.Row(elem_classes=['hero-actions']):
                        install_button=gr.Button('Install Tunix',variant='primary',elem_classes=['cta'])
                        open_button = gr.Button('Open Workbench  →', variant='secondary', elem_classes=['cta'])
                    gr.HTML('<div class="small-note">Developer-installed CLI · runs on your GPU machine, not inside the browser</div>')
                with gr.Column(scale=5):
                    gr.HTML(f'<div class="art"><div class="art-chip chip-a">↗ &nbsp; Measure the baseline</div><div class="art-chip chip-b">✓ &nbsp; Check correctness</div>{MARK}<div class="art-label">INTELLIGENCE, IN THE LOOP.</div><div class="art-chip chip-c">◎ &nbsp; Test. Compare. Refine.</div></div>')
            with gr.Group(visible=False,elem_classes=['install-section']) as installation:
                gr.Markdown('## Install Tunix\n\nClone the repository and install locally. Tunix is not published on PyPI. The CLI runs on the GPU machine; the workbench inspects its evidence.')
                gr.Code('git clone https://github.com/karim-elmaraghy/Tunix.git\ncd Tunix\npython -m venv .venv\n# Windows / VS Code:\n.\\.venv\\Scripts\\Activate.ps1\n# macOS / Linux: source .venv/bin/activate\npython -m pip install -e .\ntunix doctor\ntunix init\ntunix report --replay --port 7865',language='shell',label='Copy installation commands',interactive=False,buttons=['copy'],show_line_numbers=False,wrap_lines=True,lines=11)
                gr.Markdown('**Supported live MVP:** CUDA Tesla T4 + Llama 3.2 1B Instruct + CNN/DailyMail. Install `python -m pip install -e ".[gpu]"` on the T4 runtime; gated HF access is required. Replay needs no GPU or API key.\n\n[GitHub](https://github.com/karim-elmaraghy/Tunix) · [Windows / VS Code guide](https://github.com/karim-elmaraghy/Tunix/blob/main/docs/DEVELOPER.md) · [Google Colab export → import guide](https://github.com/karim-elmaraghy/Tunix/blob/main/docs/COLAB_TRANSFER.md)')
            gr.HTML('<div class="proof-line">MEASURED EVIDENCE, NOT PROMISES</div>')
            gr.HTML(landing_stats)
            gr.HTML('<div class="small-note">Raw throughput and energy comparisons from saved T4 evidence. Tunix verifies output quality separately; raw gains are never deployment approval.</div>')
        with gr.Group(visible=False) as workbench:
            with gr.Row():
                with gr.Column(scale=8):
                    gr.HTML('<div class="section-intro"><h1>Optimization workbench</h1><p>Configure a run. Let the agent test the trade-offs. Keep correctness in control.</p></div>')
                with gr.Column(scale=1,min_width=110):
                    home_button = gr.Button('← Overview', size='sm')
            gr.HTML('<div class="workflow"><strong>01 Configure</strong><i>→</i><span>02 Optimize</span><i>→</i><span>03 Compare</span><i>→</i><span>04 Validate</span><i>→</i><span>05 Apply</span></div>')
            snapshot = gr.State(view['snapshot'])
            gr.Markdown('### Import your benchmark results\n\nRun Tunix on your GPU, export a results ZIP, then import it here to inspect your measurements.')
            with gr.Row(elem_classes=['import-section']):
                upload=gr.File(label='Import benchmark results · Import ZIP / Browse files · up to 32 MiB',file_types=['.zip'],type='filepath',height=145)
                sessions=gr.Dropdown(session_choices(),value=default_selection,label='Current selected evidence session · includes verified T4 replay',allow_custom_value=False)
            import_status=gr.Textbox(label='Import status',interactive=False,lines=2)
            with gr.Row(equal_height=True):
                with gr.Column(scale=7, elem_classes=['config-card']):
                    gr.Markdown('### Configure your optimization')
                    mode = gr.Radio([REPLAY,LIVE,IMPORTED], value=default_mode, label='Execution environment')
                    with gr.Row():
                        model = gr.Dropdown([( 'Llama 3.2 · 1B Instruct', MODEL)], value=MODEL, label='Supported model', allow_custom_value=False)
                        workload = gr.Dropdown([WORKLOAD], value=WORKLOAD, label='Workload', allow_custom_value=False)
                    hardware = gr.Markdown(runtime_status(), elem_classes=['small-note'])
                    with gr.Accordion('Credentials · required only for live execution', open=False):
                        gr.Markdown('Keys are sent only to this app’s server and the required model services. Use your trusted local/Colab runtime. Blank fields use server environment credentials; no keys are written to exports or result files.', elem_classes=['small-note'])
                        meta = gr.Textbox(label='Meta API key', type='password', placeholder='MODEL_API_KEY')
                        hf = gr.Textbox(label='Hugging Face read token', type='password', placeholder='HF_TOKEN · Llama license required')
                    with gr.Row():
                        optimize_button = gr.Button('Optimize', variant='primary')
                        replay_button = gr.Button('Load verified T4 benchmark', variant='secondary')
                    operation = gr.Textbox(value=('Saved-results mode. No Meta request or GPU experiment has been made.' if default_mode == REPLAY else 'Project results loaded. No new experiment starts until Optimize is requested.'), label='Run status', interactive=False, lines=2)
                with gr.Column(scale=4, elem_classes=['status-card']):
                    status = gr.HTML(status_panel(initial))
                    detect_button = gr.Button('Refresh server environment', size='sm')
            source = gr.HTML('<div class="source-note">'+('SAVED T4 REPLAY · no live Meta optimization' if default_mode == REPLAY else 'PROJECT RUN EVIDENCE · benchmark execution and Meta search are separate; no new run implied')+'</div>')
            stat_cards = gr.HTML(stats)
            with gr.Tabs():
                with gr.Tab('Compare'):
                    chart = gr.HTML(charts)
                    gr.Markdown('### Measured profiles', elem_classes=['section-heading'])
                    profiles = gr.Dataframe(value=view['profile_rows'], headers=['Profile','docs/s','J/doc','Raw vs default','Token F1','Gold ROUGE-L','Quality','Reason'], interactive=False,wrap=True)
                with gr.Tab('Validate & export'):
                    pipeline_view = gr.HTML(pipeline_html(initial))
                    gr.Markdown('T5 tests batching consistency. Completeness is a separate gate. Identical truncated summaries do not qualify as a correct profile.', elem_classes=['small-note'])
                    winner = gr.Markdown(view['winner_markdown'])
                    confirm_button = gr.Button('Run held-out confirmation', variant='secondary')
                    confirmation = gr.Textbox(label='Confirmation status', interactive=False)
                    with gr.Row():
                        export_button = gr.Button('Export measured candidate · not approved')
                        apply_button = gr.Button('Check Apply readiness')
                    download = gr.File(label='Reproducible settings · NOT APPROVED', interactive=False)
                    apply_result = gr.Textbox(label='Apply state', interactive=False)
                with gr.Tab('Results / Evidence'):
                    gr.Markdown('### Original measurements & gate decisions')
                    banner = gr.Markdown(view['banner'])
                    baselines = gr.Dataframe(value=view['baseline_rows'],headers=['Baseline','Run','docs/s','p50','p95','J/doc'],interactive=False,wrap=True)
                    rejections = gr.Dataframe(value=view['rejection_rows'], headers=['Profile','Status','Reason'],interactive=False,wrap=True)
                    agent_log = gr.Dataframe(value=view['agent_rows'],headers=['Hypothesis','Source','Decision','Reason'],interactive=False,wrap=True)
                    apply_summary = gr.Markdown(view['apply_summary'])
                    refresh = gr.Button('Refresh saved evidence')
        gr.HTML('<div class="footer-note"><span>Tunix · intelligence, measured.</span><span>T4 / Llama MVP · no automatic activation</span></div>')

        def render(selection,current_mode=REPLAY):
            return view_outputs(selection,current_mode)
        outputs = [stat_cards,chart,status,snapshot,profiles,winner,banner,baselines,rejections,agent_log,apply_summary,pipeline_view,source]
        open_button.click(lambda:(gr.update(visible=False),gr.update(visible=True)),outputs=[landing,workbench],api_name=False)
        install_button.click(lambda:gr.update(visible=True),outputs=installation,api_name=False)
        home_button.click(lambda:(gr.update(visible=True),gr.update(visible=False)),outputs=[landing,workbench],api_name=False)
        def configured(*args):
            message = run_configured(*args)
            return message, '', ''
        optimize_button.click(configured, inputs=[mode,model,workload,meta,hf],outputs=[operation,meta,hf],api_name=False).then(lambda mode,selected:gr.update(value=':local' if mode==LIVE else selected),inputs=[mode,sessions],outputs=sessions,api_name=False).then(render,inputs=[sessions,mode],outputs=outputs,api_name=False)
        replay_button.click(lambda:('Verified saved T4 evidence loaded. No Meta API request or new GPU run.',gr.update(value=REPLAY),gr.update(value=':bundled')),outputs=[operation,mode,sessions],api_name=False).then(render,inputs=[sessions,mode],outputs=outputs,api_name=False)
        detect_button.click(runtime_status,outputs=hardware,api_name=False)
        refresh.click(render,inputs=[sessions,mode],outputs=outputs,api_name=False)
        mode.change(render,inputs=[sessions,mode],outputs=outputs,api_name=False)
        sessions.change(render,inputs=[sessions,mode],outputs=outputs,api_name=False)
        def uploaded(path):
            try:
                selection,values=import_upload(path)
                message=('Imported synthetic data - NOT measured GPU results.' if values[3]['session_status'].get('kind')=='synthetic' else
                         'Imported GPU results. Original bytes preserved; validation recomputed. No live Meta call or full approval.')
                return (gr.update(choices=session_choices(),value=selection),gr.update(value=IMPORTED),message)+values
            except (ValueError,OSError,TypeError,KeyError) as exc:
                raise gr.Error('Import rejected: '+str(exc)) from None
        upload.upload(uploaded,inputs=upload,outputs=[sessions,mode,import_status]+outputs,api_name='import_benchmark_results')
        demo.load(lambda:gr.update(choices=session_choices()),outputs=sessions,api_name=False)
        confirm_button.click(lambda state,token:(confirm_with_token(state,token),''),inputs=[snapshot,hf],outputs=[confirmation,hf],api_name=False).then(render,inputs=[sessions,mode],outputs=outputs,api_name=False)
        apply_button.click(app.safe_apply,inputs=snapshot,outputs=apply_result,api_name=False)
        export_button.click(export_candidate,inputs=snapshot,outputs=download,api_name=False)
    return demo
