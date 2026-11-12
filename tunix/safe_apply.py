"""Transactional config promotion; hardware execution is injected by Agent A.

apply(config) must load/activate that exact config or raise. evaluate(config)
must run heldout documents and return {benchmark: result, outputs: bundle}.
The default is restored on every rejected or exceptional attempt. A rollback
failure raises, never claiming recovery. Persisted config uses atomic replace.

Minimal persistent-runtime integration contract (GPU path not executed here):
Keep ONE runtime holder {bundle, config}, owned by the serving integration.
apply(config) must stop serving, release the previous bundle (important on T4),
load_model(config), run maybe_compile for compile=True, warm up and read back
the actual config. Publish the holder only after exact config checks pass.
On activation failure clear the holder and raise; safe_apply then invokes
apply(default_config), which reloads the default. If that fails, keep serving
disabled and propagate rollback failed. Do not leave a stale bundle published.
The evaluate callback MUST generate heldout outputs from this SAME holder,
returning {benchmark, outputs} with actual runtime readback. PR #5 run_config
loads/unloads a separate model: it is suitable for search, not for validating
the active serving holder. Single-thread the entire safe_apply transaction;
resume serving only after promotion or successful default restoration.
This requires integration-owned lifecycle/generation callbacks; persisting a
JSON config alone is not proof of GPU activation or hardware rollback.
"""
import json
import os
import tempfile
from .agent import check_config, execution_reasons
from .validator import validate


def _persist(path, config):
    parent = os.path.dirname(os.path.abspath(path))
    fd, tmp = tempfile.mkstemp(dir=parent, prefix='.tunix-')
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            json.dump(config, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def safe_apply(config, default_config, default_heldout, calibration, apply,
               evaluate, state_path=None, log_path=None):
    check_config(default_config)
    if default_heldout['config'] != default_config:
        raise ValueError('default artifact config mismatch')
    decision = dict(accepted=False, reasons=[])
    try:
        check_config(config)
        apply(dict(config))
        result = evaluate(dict(config))
        reasons = execution_reasons(result['benchmark'], config)
        if result['outputs']['config'] != config:
            reasons.append('output_config_mismatch')
        decision = validate(result['outputs'], default_heldout, calibration, split='heldout')
        decision['reasons'] = sorted(set(decision['reasons'] + reasons))
        decision['accepted'] = decision['accepted'] and not reasons
        if decision['accepted'] and state_path:
            _persist(state_path, config)
    except Exception as exc:
        decision = dict(accepted=False, reasons=['execution_error'], error=type(exc).__name__)
    if not decision['accepted']:
        try:
            apply(dict(default_config))
            if state_path:
                _persist(state_path, default_config)
        except Exception as exc:
            raise RuntimeError('rollback failed; active configuration unknown') from exc
    event = dict(action='promoted' if decision['accepted'] else 'rolled_back',
                 active_config=dict(config if decision['accepted'] else default_config), validation=decision)
    if log_path:
        with open(log_path, 'a', encoding='utf-8') as stream:
            stream.write(json.dumps(event) + '\n')
    return event
