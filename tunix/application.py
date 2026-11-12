"""Application wiring. Real runner by default; test doubles stay in tests.

Each run has its own directory. Benchmark items and agent rejection JSONL
retain their existing schemas. session.json is separate provenance/status.
No implicit activation: run_config owns a temporary benchmark model only.
"""
import json
import os
from pathlib import Path
import uuid

from .agent import execution_reasons, optimize
from .integration import batching_reasons, make_evaluator, select_winner
from .validator import CalibrationBlocked, calibrate, validate


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2), encoding='utf-8')


def prompt_version():
    from . import workload as module
    return getattr(module, 'PROMPT_VERSION', 'v1-raw')


def run_optimization(results_dir='results', budget=8, *, runner=None, workload=None,
                     completion=None, space=None):
    """Run both real tuning baselines and the bounded optimization loop.

    MODEL_API_KEY is required in the UI flow. Individual failed API requests
    retain rule_fallback labels; no key or exception text is printed here.
    """
    synthetic = any(value is not None for value in (runner, workload, completion))
    if completion is None and not os.environ.get('MODEL_API_KEY'):
        raise RuntimeError('MODEL_API_KEY is required; set it in the environment')
    if not synthetic:
        import torch
        if not torch.cuda.is_available() or 'T4' not in torch.cuda.get_device_name(0):
            raise RuntimeError('real optimization requires a CUDA Tesla T4; CPU measurements are not this demo')
    if runner is None:
        from .benchmark import run_config
        runner = run_config
    if workload is None:
        from .workload import load_workload
        workload = load_workload
    from .profiles import DEFAULT_PROFILE, NAIVE_PROFILE, PROFILES, SPACE
    tuning, heldout = workload()
    root = Path(results_dir)
    session = root / ('run_' + uuid.uuid4().hex)
    session.mkdir(parents=True)
    # Dependency-injected runs are explicitly synthetic and never selected by
    # the normal dashboard as measured sessions.
    kind = 'synthetic' if synthetic else 'measured'
    version = prompt_version()
    write_json(session / 'session.json', dict(kind=kind, state='running', gpu_verified=False, prompt_version=version))
    items = []
    def run(config, docs, **kwargs):
        outcome = runner(config, docs, **kwargs)
        items.append(outcome['benchmark'])
        write_json(session / 'benchmark_session.json', items)
        return outcome
    try:
        baselines = {}
        for name in (DEFAULT_PROFILE, NAIVE_PROFILE):
            config = PROFILES[name]
            outcome = run(config, tuning, profile_name=name, split='tuning', repeats=3, results_dir=str(session))
            if execution_reasons(outcome['benchmark'], config):
                raise RuntimeError('baseline failed: ' + name + '; see benchmark JSON error')
            baselines[name] = outcome['outputs']
        repeat = run(PROFILES[DEFAULT_PROFILE], tuning, profile_name=DEFAULT_PROFILE + '_repeat',
                     split='tuning', repeats=3, results_dir=str(session))
        first = {d['doc_id']: d['output_text'] for d in baselines[DEFAULT_PROFILE]['docs']}
        second = {d['doc_id']: d['output_text'] for d in repeat['outputs']['docs']}
        if (execution_reasons(repeat['benchmark'], PROFILES[DEFAULT_PROFILE]) or not first
                or first.keys() != second.keys()
                or sum(first[k] == second[k] for k in first) / len(first) < 0.99):
            raise RuntimeError('T1 determinism failed; search stopped')
        cal = calibrate(baselines[DEFAULT_PROFILE], baselines[NAIVE_PROFILE], margin=0.02)
        evaluate = make_evaluator(tuning, baselines[DEFAULT_PROFILE], cal, runner=run,
                                  results_dir=str(session), repeats=3)
        history = optimize(SPACE if space is None else space, evaluate, budget=budget,
                           completion=completion, log_path=str(session / 'agent_log.json'))
        winner = select_winner(history)
        write_json(session / 'confirmation_context.json', dict(
            default_tuning=baselines[DEFAULT_PROFILE], naive_tuning=baselines[NAIVE_PROFILE],
            default_benchmark=items[0],
            heldout_ids=[d['doc_id'] for d in heldout]))
        write_json(session / 'session.json', dict(kind=kind, state='complete',
                   winner=winner, gpu_verified=False, prompt_version=version,
                   activation='unavailable: no persistent runtime'))
    except Exception as exc:
        status = dict(kind=kind, state='failed', error=type(exc).__name__, winner=None, prompt_version=version)
        if isinstance(exc, CalibrationBlocked):
            status.update(state='blocked', reasons=exc.reasons, diagnostics=exc.diagnostics,
                          message=str(exc))
        write_json(session / 'session.json', status)
        if kind == 'measured':
            write_json(root / 'current_session.json', dict(directory=session.name))
        raise
    if kind == 'measured':
        write_json(root / 'current_session.json', dict(directory=session.name))
    return session


def confirm_heldout(session_dir, *, runner=None, workload=None):
    """Confirm winner + both baselines, without claiming activation/rollback."""
    session = Path(session_dir)
    meta = json.loads((session / 'session.json').read_text(encoding='utf-8'))
    if meta.get('prompt_version', 'v1-raw') != prompt_version():
        raise ValueError('prompt version changed; rerun tuning and both baselines before heldout confirmation')
    winner = meta.get('winner')
    if not winner:
        raise ValueError('no validated winner to confirm')
    if runner is None:
        from .benchmark import run_config
        runner = run_config
    if workload is None:
        from .workload import load_workload
        workload = load_workload
    from .profiles import DEFAULT_PROFILE, NAIVE_PROFILE, PROFILES
    _, heldout = workload()
    context = json.loads((session / 'confirmation_context.json').read_text(encoding='utf-8'))
    if [d['doc_id'] for d in heldout] != context['heldout_ids']:
        raise ValueError('heldout workload changed')
    cal = calibrate(context['default_tuning'], context['naive_tuning'], margin=0.02)
    directory = session / 'heldout'
    directory.mkdir(exist_ok=True)
    outcomes = {}
    for name in (DEFAULT_PROFILE, NAIVE_PROFILE):
        outcomes[name] = runner(PROFILES[name], heldout, profile_name=name, split='heldout',
                                repeats=3, results_dir=str(directory))
    config = winner['proposal']['config']
    name = winner['result']['benchmark']['profile_name']
    outcome = runner(config, heldout, profile_name=name, split='heldout', repeats=3, results_dir=str(directory))
    reasons = execution_reasons(outcome['benchmark'], config)
    for base in (DEFAULT_PROFILE, NAIVE_PROFILE):
        if execution_reasons(outcomes[base]['benchmark'], PROFILES[base]):
            reasons.append('heldout_baseline_failed')
    if outcome['outputs'].get('config') != config:
        reasons.append('output_config_mismatch')
    decision = validate(outcome['outputs'], outcomes[DEFAULT_PROFILE]['outputs'], cal, split='heldout')
    if config['batch'] > 1:
        ref_config = dict(config, batch=1)
        ref = runner(ref_config, heldout, profile_name=name + '_batch1', split='heldout',
                     repeats=3, results_dir=str(directory))
        reasons += (['T5_reference_failed'] if execution_reasons(ref['benchmark'], ref_config)
                    else batching_reasons(outcome['outputs'], ref['outputs']))
    decision['reasons'] = sorted(set(decision['reasons'] + reasons))
    decision['accepted'] = decision['accepted'] and not reasons
    result = dict(validation=decision, benchmark=outcome['benchmark'],
                  baselines={key: value['benchmark'] for key, value in outcomes.items()},
                  activation='not attempted; no persistent runtime', kind=meta['kind'])
    def ratio(a, b):
        x, y = a.get('docs_per_sec'), b.get('docs_per_sec')
        return x / y if isinstance(x, (int, float)) and isinstance(y, (int, float)) and x > 0 and y > 0 else None
    tuning_speedup = ratio(winner['result']['benchmark'], context['default_benchmark'])
    heldout_speedup = ratio(outcome['benchmark'], outcomes[DEFAULT_PROFILE]['benchmark'])
    result.update(tuning_speedup_vs_default=tuning_speedup, heldout_speedup_vs_default=heldout_speedup,
                  speedup_confirmed=bool(tuning_speedup and heldout_speedup and
                                         abs(heldout_speedup / tuning_speedup - 1) <= 0.10),
                  claim_basis='Use heldout measured speedup; no GPU activation or rollback claim')
    write_json(directory / 'confirmation.json', result)
    return result
