"""Minimal A+B adapter. No GPU imports until the real runner is called.

Usage after combining PRs #4/#5:
    cal = calibrate(default_tuning, naive_tuning, margin=0.02)
    evaluate = make_evaluator(tuning_docs, default_tuning, cal)
    history = optimize(SPACE, evaluate, budget=8, log_path='experiments.jsonl')
    winner = select_winner(history)  # None if no eligible experiment

Production uses the actual run_config return artifacts, never test fixtures.
The optional runner exists for CPU tests; synthetic fixtures live under tests/
and must never be published as measurements. Proposal provenance stays in the
original history record, including rule_fallback when the API is unavailable.
"""
import copy
import hashlib
import json
import math

from .agent import check_config, execution_reasons
from .validator import index_docs, token_f1, validate


def batching_reasons(candidate, reference):
    """T5 is independent of calibrated quality; never relax its 0.95 floor."""
    try:
        split = candidate['docs'][0]['split']
        a, b = index_docs(candidate, split), index_docs(reference, split)
        if a.keys() != b.keys():
            return ['T5_document_mismatch']
        if any(token_f1(a[k]['output_text'], b[k]['output_text']) < 0.95 for k in a):
            return ['T5_batching_failed']
        return []
    except (KeyError, TypeError, ValueError, IndexError):
        return ['T5_missing_outputs']


def make_evaluator(docs, default, calibration, *, runner=None, results_dir='results', repeats=3):
    """Return optimize's evaluate(config) for TUNING documents only.

    runner follows PR #5 run_config(config, docs, profile_name=..., split=...,
    repeats=..., results_dir=...) -> {benchmark, outputs, console}. The real
    runner saves its real output artifacts. Console diagnostics remain outside
    the shared benchmark schema. Missing/malformed artifacts reject the trial.
    """
    docs, default = copy.deepcopy(docs), copy.deepcopy(default)
    baseline = index_docs(default, 'tuning')
    ids = [d['doc_id'] for d in docs]
    if len(ids) != len(set(ids)) or set(ids) != set(baseline):
        raise ValueError('tuning workload must match baseline document IDs')
    if any(d.get('split') != 'tuning' or d['gold_summary'] != baseline[d['doc_id']]['gold_summary'] for d in docs):
        raise ValueError('tuning workload split/gold mismatch')

    def evaluate(config):
        check_config(config)
        requested = copy.deepcopy(config)
        slug = 'agent_tuning_' + hashlib.sha256(json.dumps(requested, sort_keys=True).encode()).hexdigest()[:16]
        run = runner
        if run is None:
            from .benchmark import run_config
            run = run_config
        outcome = run(copy.deepcopy(requested), copy.deepcopy(docs), profile_name=slug,
                      split='tuning', repeats=repeats, results_dir=results_dir)
        benchmark = outcome['benchmark']
        reasons = execution_reasons(benchmark, requested)
        if benchmark.get('profile_name') != slug:
            reasons.append('profile_mismatch')
        if not reasons:
            bundle = outcome.get('outputs')
            if not isinstance(bundle, dict) or bundle.get('config') != requested:
                reasons.append('output_config_mismatch')
            elif bundle.get('profile_name') != slug:
                reasons.append('profile_mismatch')
        if reasons:
            decision = dict(accepted=False, split='tuning', docs=[], reasons=sorted(set(reasons)))
        else:
            decision = validate(bundle, default, calibration, split='tuning')
            if requested['batch'] > 1:
                reference_config = dict(requested, batch=1)
                reference = run(reference_config, copy.deepcopy(docs), profile_name=slug + '_batch1',
                                split='tuning', repeats=repeats, results_dir=results_dir)
                t5 = execution_reasons(reference['benchmark'], reference_config)
                if t5:
                    t5 = ['T5_reference_failed']
                elif reference['outputs'].get('config') != reference_config:
                    t5 = ['T5_reference_config_mismatch']
                else:
                    t5 = batching_reasons(bundle, reference['outputs'])
                decision['reasons'] = sorted(set(decision['reasons'] + t5))
                decision['accepted'] = decision['accepted'] and not t5
        return dict(benchmark=benchmark, validation=decision)
    return evaluate


def select_winner(history):
    """Best validated docs/s record, or None; retain its proposal provenance.

    Recheck execution and quality decisions rather than trusting the outer
    accepted flag alone. Null/nonfinite/nonpositive throughput cannot win.
    """
    eligible = []
    for record in history:
        try:
            result = record['result']
            benchmark, decision = result['benchmark'], result['validation']
            config = check_config(record['proposal']['config'])
            rate = benchmark['docs_per_sec']
            if (record.get('accepted') is True and not record.get('reasons')
                    and decision.get('accepted') is True and not decision.get('reasons')
                    and not execution_reasons(benchmark, config)
                    and type(rate) in (int, float) and math.isfinite(rate) and rate > 0):
                eligible.append(record)
        except (KeyError, TypeError, ValueError, AttributeError):
            continue
    return max(eligible, key=lambda r: r['result']['benchmark']['docs_per_sec'], default=None)
