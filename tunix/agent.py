"""Meta optimization loop with explicit, auditable fallback provenance.

Integration: optimize(space, evaluate, budget=8). evaluate(config) must execute
the benchmark and return {benchmark: Issue #2 result, validation: gate decision}.
It must never use heldout data for search. No GPU imports are needed here.

Load outputs_<profile>.json as dictionaries. First call
validator.calibrate(default_tuning, naive_tuning, margin=0.02), then wrap
Agent A's run_config(config) to call validator.validate(outputs, default_tuning,
calibration). Pass that adapter to optimize(SPACE, evaluate). Persist the
returned history (or supply log_path for JSONL rejection/provenance records).
PR #5's run_config(config, docs, profile_name=slug, split='tuning', repeats=3)
returns {benchmark, outputs, console}; wrap it rather than passing it directly
as evaluate. Use outcome['benchmark'] and validate(outcome['outputs'], ...).
For a failed outcome return an explicitly rejected validation decision and
preserve benchmark.error; failed output bundles contain no docs. Reject an
outputs.config mismatch before validation. Saved benchmark JSON is a list;
select exactly one matching requested_config/profile, not the entire list.
Keep console diagnostics separate from the approved benchmark JSON schema.
Select a winner only among accepted records using actual measured throughput.
Call safe_apply.safe_apply with a separate HELDOUT execution adapter; do not
reuse tuning artifacts. Calibrated values must be reported with real results.

API protocol: https://dev.meta.ai/docs/protocols/chat-completions
MODEL_API_KEY is required for real calls; META_MODEL optionally overrides the
documented muse-spark-1.3 model. No client dependency beyond the standard library.
"""
import json
import os
import urllib.request


def check_config(c):
    if not isinstance(c, dict) or set(c) != {'dtype', 'attn', 'batch', 'compile', 'max_new_tokens'}:
        raise ValueError('config keys')
    if c['dtype'] not in ('float32', 'float16') or c['attn'] not in ('eager', 'sdpa'):
        raise ValueError('config enum')
    if type(c['compile']) is not bool or type(c['batch']) is not int or c['batch'] not in (1, 4, 8):
        raise ValueError('config types')
    if type(c['max_new_tokens']) is not int or not 1 <= c['max_new_tokens'] <= 4096:
        raise ValueError('token limit')
    return c


def parse_proposal(text, remaining):
    obj = json.loads(text)
    if set(obj) != {'config', 'reason'} or not isinstance(obj['reason'], str) or not obj['reason'].strip():
        raise ValueError('proposal needs config and nonempty reason')
    check_config(obj['config'])
    if obj['config'] not in remaining:
        raise ValueError('config outside remaining search space')
    return obj


def meta_response(messages):
    """Real transport; return only credential-free response metadata and text."""
    key = os.environ.get('MODEL_API_KEY')
    if not key:
        raise RuntimeError('MODEL_API_KEY unavailable')
    body = json.dumps(dict(model=os.environ.get('META_MODEL', 'muse-spark-1.3'),
                           messages=messages, max_completion_tokens=2048)).encode()
    req = urllib.request.Request('https://api.meta.ai/v1/chat/completions', data=body,
                                 headers={'Authorization': 'Bearer ' + key, 'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=45) as response:
        result = json.load(response)
        status = getattr(response, 'status', 200)
    choice = result['choices'][0]
    if choice.get('finish_reason') != 'stop':
        raise ValueError('incomplete Meta response')
    content = choice['message']['content']
    if not isinstance(content, str) or not content.strip():
        raise ValueError('empty Meta response')
    # Do not retain arbitrary headers, error bodies, or the authentication key.
    if key in content:
        raise ValueError('credential in response')
    return dict(content=content, http_status=status, model=result.get('model'),
                requested_model=os.environ.get('META_MODEL', 'muse-spark-1.3'),
                response_id=result.get('id'), finish_reason=choice['finish_reason'])


def meta_completion(messages):
    return meta_response(messages)['content']


def propose(remaining, history, completion=None):
    if not remaining:
        raise ValueError('search exhausted')
    for c in remaining:
        check_config(c)
    messages = [dict(role='system', content='Optimize Llama-3.2-1B-Instruct on T4. Choose one remaining config using measured history. Maximize docs_per_sec subject to accepted validation; no invented measurements. Return ONLY JSON {"config": <exact remaining config>, "reason": "explanation"}.'),
                dict(role='user', content=json.dumps(dict(remaining=remaining, history=history)))]
    try:
        proposal = parse_proposal((completion or meta_completion)(messages), remaining)
        return dict(proposal, source='meta_api', model=os.environ.get('META_MODEL', 'muse-spark-1.3'))
    except Exception as exc:
        # Never log exception text: HTTP/client exceptions can contain secrets.
        return dict(config=dict(remaining[0]), reason='Fixed remaining-space order; Meta unavailable or invalid response.',
                    source='rule_fallback', fallback_error=type(exc).__name__)


def execution_reasons(benchmark, config):
    if benchmark.get('status') != 'success':
        return ['execution_failed']
    if benchmark.get('requested_config') != config or benchmark.get('actual_config') != config:
        return ['config_mismatch']
    return []


def optimize(space, evaluate, budget=8, completion=None, log_path=None):
    """Return measured experiments, preserving individual proposal provenance."""
    remaining = []
    for c in space:
        check_config(c)
        if c not in remaining:
            remaining.append(dict(c))
    history = []
    for _ in range(min(budget, len(remaining))):
        proposal = propose(remaining, history, completion)
        config = proposal['config']
        remaining.remove(config)
        try:
            measured = evaluate(dict(config))
            reasons = execution_reasons(measured['benchmark'], config)
            decision = measured['validation']
            if decision.get('accepted') is not True:
                reasons += decision.get('reasons') or ['quality_rejected']
            record = dict(proposal=proposal, result=measured, accepted=not reasons, reasons=reasons)
        except Exception as exc:
            record = dict(proposal=proposal, accepted=False, reasons=['evaluation_error'], error=type(exc).__name__)
        history.append(record)
        if log_path:
            with open(log_path, 'a', encoding='utf-8') as stream:
                stream.write(json.dumps(record) + '\n')
    return history
