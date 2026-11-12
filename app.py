"""Tunix's results-first Gradio dashboard.

Run with ``python app.py`` after installing Gradio. This module deliberately
keeps GPU and benchmark dependencies out of its import path.
"""

from __future__ import annotations

import json
import hashlib
import math
import os
import subprocess
import threading
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
RESULTS_DIR = Path(os.environ.get('TUNIX_RESULTS_DIR', str(ROOT / 'results'))).resolve()
RUN_LOCK = threading.Lock()
SAMPLE_DIR = RESULTS_DIR / "sample"

NAIVE_PROFILE = "naive_fp32_eager_b1"
DEFAULT_PROFILE = "default_fp16_sdpa_b1"
KNOWN_PROFILES = (
    NAIVE_PROFILE,
    DEFAULT_PROFILE,
    "fp16_sdpa_b4",
    "fp16_sdpa_b8",
    "fp16_sdpa_b4_compile",
    "fp16_sdpa_b1_cap64",
)
TITLE = "Tunix - Find the fastest correct profile for YOUR GPU"
CSS = """
:root {
  --canvas: #F7F6F2;
  --surface: #FFFFFF;
  --ink: #292824;
  --muted: #77756F;
  --line: #E5E2DA;
  --accent: #B66C4B;
  --accent-wash: #F5ECE7;
}
body, .gradio-container {
  background: var(--canvas) !important;
  color: var(--ink) !important;
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif !important;
}
.gradio-container {
  width: 100% !important;
  max-width: none !important;
  padding: 28px 0 64px !important;
}
.gradio-container > .main {
  width: 100% !important;
  max-width: 1440px !important;
}
.gradio-container > main {
  width: calc(100% - 80px) !important;
  max-width: 1360px !important;
  margin-left: auto !important;
  margin-right: auto !important;
}
footer { display: none !important; }
h1, h2, h3, p { color: var(--ink) !important; }
h1 {
  font-family: Georgia, "Times New Roman", serif !important;
  font-size: 2.45rem !important;
  font-weight: 400 !important;
  letter-spacing: -0.035em !important;
  line-height: 1.12 !important;
}
h2 {
  font-family: Georgia, "Times New Roman", serif !important;
  font-size: 1.55rem !important;
  font-weight: 400 !important;
  letter-spacing: -0.02em !important;
}
h3 { font-size: 1rem !important; font-weight: 600 !important; }
.hero { padding: 18px 0 26px; border-bottom: 1px solid var(--line); }
.eyebrow, .section-label {
  color: var(--accent) !important;
  font-size: 0.72rem !important;
  font-weight: 650 !important;
  letter-spacing: 0.12em !important;
  text-transform: uppercase;
}
.subtitle { color: var(--muted) !important; font-size: 1rem; max-width: 660px; }
.section-heading { padding-top: 20px; margin-bottom: 8px; }
.surface, .optix-card, .gr-group {
  background: var(--surface) !important;
  border: 1px solid var(--line) !important;
  border-radius: 5px !important;
  box-shadow: none !important;
}
.optix-card { padding: 18px 20px !important; }
.notice {
  border: 1px solid var(--line);
  border-left: 3px solid var(--accent);
  background: #FBFAF7;
  color: var(--ink);
  padding: 13px 16px;
  margin: 18px 0 8px;
  font-size: 0.92rem;
}
.quiet { color: var(--muted) !important; font-size: 0.88rem !important; }
.workflow-note { color: var(--muted); font-size: 0.88rem; line-height: 1.6; }
.gradio-container button.primary {
  background: var(--accent) !important;
  border: 1px solid var(--accent) !important;
  border-radius: 4px !important;
  color: #fff !important;
  box-shadow: none !important;
  font-weight: 550 !important;
}
.gradio-container button.secondary {
  background: #fff !important;
  border: 1px solid var(--line) !important;
  border-radius: 4px !important;
  color: var(--ink) !important;
  box-shadow: none !important;
}
.gradio-container button:disabled {
  opacity: 0.62 !important;
  cursor: not-allowed !important;
}
.gradio-container input, .gradio-container textarea {
  border-color: var(--line) !important;
  border-radius: 4px !important;
  background: #fff !important;
  color: var(--ink) !important;
}
.gradio-container .table-wrap {
  border: 1px solid var(--line) !important;
  border-radius: 4px !important;
  background: #fff !important;
}
.gradio-container table thead th {
  background: #F8F7F4 !important;
  color: var(--muted) !important;
  font-size: 0.73rem !important;
  font-weight: 600 !important;
  letter-spacing: 0.04em;
  text-transform: uppercase;
}
.gradio-container table td, .gradio-container table th {
  border-color: var(--line) !important;
  color: var(--ink) !important;
}
@media (max-width: 720px) {
  .gradio-container { padding: 20px 0 40px !important; }
  .gradio-container > .main { max-width: none !important; }
  .gradio-container > main {
    width: calc(100% - 32px) !important;
    max-width: none !important;
  }
  h1 { font-size: 2rem !important; }
}
"""


def _read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def _load_benchmark(path: Path) -> list[dict[str, Any]]:
    data = _read_json(path)
    if not isinstance(data, list) or any(not isinstance(item, dict) for item in data):
        raise ValueError(f"{path} must contain a JSON list of benchmark result objects")
    return data


def _load_agent_steps(path: Path) -> list[dict[str, Any]]:
    content = path.read_text(encoding="utf-8")
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        steps = [json.loads(line) for line in content.splitlines() if line.strip()]
    else:
        if isinstance(data, list):
            steps = data
        elif isinstance(data, dict):
            if "steps" in data or "agent_steps" in data:
                steps = data.get("steps", data.get("agent_steps", []))
            else:
                steps = [data]
        else:
            raise ValueError(f"{path} must contain a list or an object with a steps list")
    if not isinstance(steps, list) or any(not isinstance(step, dict) for step in steps):
        raise ValueError(f"{path} steps must be a list of agent step objects")
    return [_normalize_agent_step(step) for step in steps]


def _normalize_agent_step(step: dict[str, Any]) -> dict[str, Any]:
    proposal = step.get("proposal")
    proposal = proposal if isinstance(proposal, dict) else {}
    result = step.get("result")
    result = result if isinstance(result, dict) else {}
    validation = result.get("validation")
    validation = validation if isinstance(validation, dict) else {}
    benchmark = result.get("benchmark")
    benchmark = benchmark if isinstance(benchmark, dict) else {}
    accepted = step.get("accepted")
    decision = step.get("decision")
    if decision is None and isinstance(accepted, bool):
        decision = "PROMOTE" if accepted else "REJECT"
    reasons = step.get("reasons")
    reason = step.get("reason")
    if not reason and isinstance(reasons, list):
        reason = "; ".join(str(item) for item in reasons)
    return {
        **step,
        "config": step.get("config", proposal.get("config")),
        "hypothesis": step.get("hypothesis", proposal.get("hypothesis", proposal.get("reason"))),
        "source": step.get("source", proposal.get("source")),
        "decision": decision,
        "reason": reason or validation.get("error") or ", ".join(validation.get("reasons", [])),
        "benchmark": benchmark,
        "validation": validation,
        "accepted": accepted,
    }


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    records = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        record = json.loads(line)
        if not isinstance(record, dict):
            raise ValueError(f"{path} line {line_number} must be a JSON object")
        records.append(record)
    return records


def load_dashboard_data(results_dir: Path = RESULTS_DIR, *, follow_session=True) -> dict[str, Any]:
    """Load real results when present, otherwise the explicitly marked samples."""
    results_dir = Path(results_dir)
    pointer = results_dir / 'current_session.json'
    if pointer.is_file() and follow_session:
        directory = _read_json(pointer).get('directory', '')
        candidate = (results_dir / directory).resolve()
        if candidate.parent == results_dir.resolve() and candidate.is_dir():
            results_dir = candidate
    sample_dir = results_dir / "sample"
    errors: list[str] = []
    benchmark_files = sorted(
        results_dir.glob("benchmark_*.json"),
        key=lambda path: path.stat().st_mtime_ns,
    )
    benchmark_is_sample = not benchmark_files
    metadata = _read_json(results_dir / 'session.json') if (results_dir / 'session.json').is_file() else {}
    if metadata.get('kind') == 'synthetic':
        benchmark_is_sample = True
    benchmark_path = benchmark_files[-1] if benchmark_files else sample_dir / "benchmark_sample.json"

    try:
        benchmarks = _load_benchmark(benchmark_path)
    except (OSError, json.JSONDecodeError, ValueError) as error:
        benchmarks = []
        errors.append(f"Could not load benchmark results from {benchmark_path}: {error}")

    agent_path = results_dir / "agent_log.json"
    agent_is_sample = not agent_path.is_file()
    if agent_path.is_file():
        selected_agent_path = agent_path
    else:
        selected_agent_path = sample_dir / "agent_log_sample.json"
    try:
        agent_steps = _load_agent_steps(selected_agent_path) if selected_agent_path.is_file() else []
    except (OSError, json.JSONDecodeError, ValueError) as error:
        agent_steps = []
        errors.append(f"Could not load agent log from {selected_agent_path}: {error}")

    try:
        apply_events = _load_jsonl(results_dir / "apply_log.jsonl")
    except (OSError, json.JSONDecodeError, ValueError) as error:
        apply_events = []
        errors.append(f"Could not load safe-apply log: {error}")
    try:
        dashboard_apply_events = _load_jsonl(results_dir / "dashboard_apply_log.jsonl")
    except (OSError, json.JSONDecodeError, ValueError) as error:
        dashboard_apply_events = []
        errors.append(f"Could not load dashboard apply details: {error}")

    artifact_validation = {}
    report_path = results_dir / 'validation_report.json'
    if report_path.is_file():
        try:
            report = _read_json(report_path)
            if benchmark_path.name not in report['artifact_sha256']:
                raise ValueError('artifact validation report does not cover selected benchmark')
            for name, expected in report['artifact_sha256'].items():
                if Path(name).name != name or hashlib.sha256((results_dir / name).read_bytes()).hexdigest() != expected:
                    raise ValueError('artifact validation report is stale')
            artifact_validation = report
        except (OSError, ValueError, KeyError) as error:
            errors.append('Could not verify artifact validation report: ' + str(error))

    return {
        'artifact_validation': artifact_validation,
        'session_status': metadata,
        "confirmation": _read_json(results_dir / 'heldout' / 'confirmation.json')
        if (results_dir / 'heldout' / 'confirmation.json').is_file() else {},
        "benchmarks": benchmarks,
        "agent_steps": agent_steps,
        "apply_events": apply_events,
        "dashboard_apply_events": dashboard_apply_events,
        "benchmark_is_sample": benchmark_is_sample,
        "agent_is_sample": agent_is_sample,
        "has_real_benchmarks": bool(benchmark_files) and not benchmark_is_sample,
        "errors": errors,
        "results_dir": str(results_dir),
    }


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if not math.isfinite(float(value)):
        return None
    return float(value)


def _format_metric(value: Any, suffix: str = "") -> str:
    number = _number(value)
    if number is None:
        return "Not recorded"
    return f"{number:.3f}{suffix}"


def _config_key(config: Any) -> str | None:
    if not isinstance(config, dict):
        return None
    return json.dumps(config, sort_keys=True, separators=(",", ":"))


def _decision(step: dict[str, Any]) -> str:
    value = step.get("decision", step.get("status", ""))
    return str(value).strip().lower().replace("-", "_").replace(" ", "_")


def _source_label(step: dict[str, Any]) -> str:
    source = str(step.get("source", "")).strip().lower()
    if source in {"muse_api", "meta_api"}:
        return "Meta API"
    if source == "rule_fallback":
        return "Rule fallback (not Meta API)"
    return str(step.get("source") or "Source not recorded")


def _agent_decisions(steps: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    decisions: dict[str, dict[str, Any]] = {}
    for step in steps:
        config = step.get("config", step.get("requested_config"))
        key = _config_key(config)
        if key:
            decisions[key] = step
    return decisions


def _profile_decision(
    result: dict[str, Any], decisions: dict[str, dict[str, Any]]
) -> dict[str, Any] | None:
    config = result.get("requested_config", result.get("actual_config"))
    key = _config_key(config)
    return decisions.get(key) if key else None


def _status_and_reason(
    result: dict[str, Any], step: dict[str, Any] | None, artifact_validation=None
) -> tuple[str, str]:
    benchmark_status = str(result.get("status", "")).lower()
    if benchmark_status == "failed":
        return "Failed", str(result.get("error") or result.get("reason") or "Failure reason not recorded")
    report = (artifact_validation or {}).get('profiles', {}).get(result.get('profile_name'), {})
    if report and report.get('config') == result.get('requested_config'):
        reasons = report.get('reasons', [])
        note = ('Artifact validator: ' + ', '.join(reasons) if reasons else
                'Quality calibration is blocked' if (artifact_validation or {}).get('calibration')=='blocked' else
                'Tuning checks computed; full approval and held-out confirmation remain unverified')
        if report.get('flagged_docs'):
            note += '; docs: ' + ', '.join(str(d['doc_id']) for d in report['flagged_docs'])
        if report.get('t5'):
            minimum=report['t5'].get('min_f1')
            note += '; T5 ' + ('PASS' if report['t5']['passed'] else 'FAIL')
            if isinstance(minimum,(float,int)): note += f" (min F1 {minimum:.3f})"
        return ('Rejected' if reasons else 'Measured (not validated)'), note
    decision = _decision(step or {})
    if decision in {"reject", "rejected", "fail", "failed"}:
        return "Rejected", str((step or {}).get("reason") or "Rejection reason not recorded")
    if decision in {"promote", "accepted", "accept", "pass", "passed"}:
        if benchmark_status != "success":
            return "Unverified", "Proposal was marked accepted, but no successful benchmark is recorded"
        return "Accepted", str((step or {}).get("reason") or "Accepted; no additional reason recorded")
    if benchmark_status == "success":
        return "Measured (not validated)", "Benchmark succeeded; no quality decision recorded"
    reason = result.get("error") or result.get("reason")
    if result.get("status") is None:
        return "Status not recorded", str(reason or "No benchmark status recorded")
    return str(result.get("status")), str(reason or "No acceptance decision or reason recorded")


def _find_chosen(
    benchmarks: list[dict[str, Any]], steps: list[dict[str, Any]]
) -> dict[str, Any] | None:
    candidates = []
    for step in steps:
        decision = _decision(step)
        if decision in {'reject', 'rejected', 'failed', 'fail'} or step.get('accepted') is False:
            continue
        validation = step.get('validation', {})
        if validation.get('accepted') is not True or validation.get('reasons'):
            continue
        if step.get("chosen") is not True and step.get("selected") is not True and decision not in {
            "promote",
            "accepted",
            "accept",
        }:
            continue
        result = step.get("benchmark")
        result = result if isinstance(result, dict) else {}
        config = step.get("config", step.get("requested_config"))
        matched = next(
            (
                item
                for item in benchmarks
                if _config_key(config)
                and _config_key(config)
                in {
                    _config_key(item.get("requested_config")),
                    _config_key(item.get("actual_config")),
                }
            ),
            None,
        )
        selected = dict(result or matched or {})
        if selected:
            if str(selected.get("status", "")).lower() != "success":
                continue
            if selected.get('actual_config') != selected.get('requested_config'):
                continue
            selected.setdefault("profile_name", step.get("profile_name"))
            selected.setdefault("requested_config", config)
            selected.setdefault("actual_config", config)
            selected["_agent_validation"] = step.get("validation", {})
            selected["_agent_decision"] = step.get("decision")
            selected["_agent_reason"] = step.get("reason")
            throughput = _number(selected.get("docs_per_sec"))
            candidates.append((step.get("chosen") is True or step.get("selected") is True, throughput, selected))
    explicit = [item for item in candidates if item[0]]
    if explicit:
        return explicit[-1][2]
    measured = [item for item in candidates if item[1] is not None]
    if measured:
        return max(measured, key=lambda item: item[1])[2]

    return None


def _quality_values(result: dict[str, Any] | None) -> tuple[Any, Any]:
    if not result:
        return None, None
    validation = result.get("_agent_validation")
    if isinstance(validation, dict):
        rows = validation.get("docs")
        if isinstance(rows, list) and rows:
            f1 = [_number(row.get("token_f1")) for row in rows if isinstance(row, dict)]
            rouge = [_number(row.get("rouge_l")) for row in rows if isinstance(row, dict)]
            f1 = [value for value in f1 if value is not None]
            rouge = [value for value in rouge if value is not None]
            return (
                sum(f1) / len(f1) if f1 else None,
                sum(rouge) / len(rouge) if rouge else None,
            )
    return result.get("quality_f1"), result.get("rouge_l")


def _heldout_event(snapshot: dict[str, Any], chosen: dict[str, Any] | None) -> dict[str, Any]:
    if not chosen:
        return {}
    config = chosen.get("actual_config") or chosen.get("requested_config")
    for event in reversed(snapshot.get("dashboard_apply_events", [])):
        if _config_key(event.get("candidate_config")) == _config_key(config):
            return event
    for event in reversed(snapshot.get("apply_events", [])):
        if _config_key(event.get("active_config")) == _config_key(config):
            return event
    return {}


def _heldout(result: dict[str, Any] | None) -> dict[str, Any]:
    if not result:
        return {}
    nested = result.get("heldout")
    data = nested if isinstance(nested, dict) else {}
    fields = {
        "status": data.get("status", result.get("heldout_status")),
        "decision": data.get("decision", result.get("heldout_decision")),
        "quality_f1": data.get("quality_f1", result.get("heldout_quality_f1")),
        "rouge_l": data.get("rouge_l", result.get("heldout_rouge_l")),
        "docs_per_sec": data.get("docs_per_sec", result.get("heldout_docs_per_sec")),
        "joules_per_doc": data.get("joules_per_doc", result.get("heldout_joules_per_doc")),
        "reason": data.get("reason", result.get("heldout_reason")),
    }
    if str(result.get("split", "")).lower() == "heldout":
        for field in ("status", "decision", "quality_f1", "rouge_l", "docs_per_sec", "joules_per_doc", "reason"):
            if fields[field] is None:
                fields[field] = result.get(field)
    return fields


def _baseline(benchmarks: list[dict[str, Any]], profile_name: str) -> dict[str, Any] | None:
    return next(
        (result for result in benchmarks if result.get("profile_name") == profile_name),
        None,
    )


def _speedup(candidate: Any, baseline: Any) -> float | None:
    first = _number(candidate)
    second = _number(baseline)
    if first is None or second is None or first <= 0 or second <= 0:
        return None
    return first / second


def _percent_saving(candidate: Any, baseline: Any) -> float | None:
    first = _number(candidate)
    second = _number(baseline)
    if first is None or second is None or second <= 0:
        return None
    return (second - first) / second * 100


def _display_value(value: Any) -> str:
    if value is None:
        return "Not recorded"
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True)
    return str(value)


def _gpu_info() -> tuple[str, str]:
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            check=False,
            text=True,
            timeout=3,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError) as error:
        return f"GPU detection unavailable: {error}", "Benchmark GPU: Not recorded"
    if completed.returncode != 0:
        detail = completed.stderr.strip() or f"nvidia-smi exited with code {completed.returncode}"
        return f"GPU detection unavailable: {detail}", "Benchmark GPU: Not recorded"
    detected = completed.stdout.strip().splitlines()
    return (
        f"Detected runtime GPU: {', '.join(detected) if detected else 'No GPU reported'} "
        "(this is the current runtime, not proof of a benchmark run)",
        "Benchmark GPU: Not recorded",
    )


def _result_metadata(benchmarks: list[dict[str, Any]]) -> tuple[str, str]:
    gpu_values = []
    model_values = []
    for result in benchmarks:
        for field in ("gpu_name", "gpu", "device_name"):
            value = result.get(field)
            if value and str(value) not in gpu_values:
                gpu_values.append(str(value))
        for field in ("model_name", "model", "model_id"):
            value = result.get(field)
            if value and str(value) not in model_values:
                model_values.append(str(value))
    return (
        f"Benchmark GPU: {', '.join(gpu_values) if gpu_values else 'Not recorded'}",
        f"Model: {', '.join(model_values) if model_values else 'Not recorded'}",
    )


def build_dashboard_view(data: dict[str, Any]) -> dict[str, Any]:
    steps = data["agent_steps"]
    benchmarks = list(data["benchmarks"])
    known_configs = {
        _config_key(result.get("requested_config") or result.get("actual_config"))
        for result in benchmarks
    }
    for step in steps:
        result = step.get("benchmark")
        if not isinstance(result, dict) or not result:
            continue
        key = _config_key(result.get("requested_config") or result.get("actual_config"))
        if key and key not in known_configs:
            benchmarks.append(result)
            known_configs.add(key)
    decisions = _agent_decisions(steps)
    naive = _baseline(benchmarks, NAIVE_PROFILE)
    competent = _baseline(benchmarks, DEFAULT_PROFILE)
    chosen = _find_chosen(benchmarks, steps)
    if data['benchmark_is_sample'] or data.get('session_status', {}).get('state') == 'blocked':
        chosen = None

    if data["benchmark_is_sample"]:
        banner = (
            "**SAMPLE DATA - NOT REAL RESULTS**  \n"
            "No real benchmark result set is available. Profiles, measured metrics, "
            "and hold-out confirmation remain in a not-run state."
        )
    elif not data["benchmarks"]:
        banner = "**No valid real benchmark results are available.** See any loading errors below."
    else:
        banner = "**Real benchmark results loaded.** Verify every displayed field against the saved JSON."
        if data["agent_is_sample"]:
            banner += " Agent log: not run or unavailable."
    if data["errors"]:
        banner += "\n\n**Data loading errors:** " + " | ".join(data["errors"])
    session_status = data.get('session_status') or {}
    if session_status.get('state') == 'blocked':
        banner += '\n\n**Calibration BLOCKED:** ' + session_status.get('message', 'invalid baseline outputs') + '. No accepted winner.'
    if data.get('artifact_validation'):
        banner += '\n\n**Artifact checks:** ' + '; '.join(data['artifact_validation'].get('remaining_checks', []))
    if session_status.get('kind') in ('imported','synthetic'):
        banner += ('\n\n**Imported evidence, not live Meta optimization.** Uploaded approval claims are not trusted. '
                   'T5 and completeness are recomputed separately; missing checks remain unverified. No full approval inferred.')
    else:
        banner += ('\n\n**Raw speed is not approval.** Owner-reported V2 batching checks passed T5; V1 failed. '
               'V2 baseline truncation still blocks strict calibration, and held-out approval is pending. '
               'Integrated search requires per-document F1 >= 0.95. '
               'GPU activation/rollback remains unverified.')
    banner += '\n\n**Loaded artifact prompt version:** ' + str(session_status.get('prompt_version', 'unversioned / not verified; do not mix V1 and V2'))
    measured_b8 = next((r for r in benchmarks if r.get('profile_name') == 'fp16_sdpa_b8'), None)
    if not data['benchmark_is_sample'] and data.get('artifact_validation') and measured_b8 and competent:
        raw = _speedup(measured_b8.get('docs_per_sec'), competent.get('docs_per_sec'))
        saving = _percent_saving(measured_b8.get('joules_per_doc'), competent.get('joules_per_doc'))
        if raw is not None and saving is not None:
            approval_state='BLOCKED' if session_status.get('state')=='blocked' else 'PENDING'
            evidence_caption=('imported artifacts; hardware not independently verified' if session_status.get('kind')=='imported' else 'saved Tesla T4 measurements')
            banner = (f'## {raw:.2f}× raw throughput · {saving:.1f}% less energy\n\n'
                      f'**FP16 batch 8 vs competent FP16 batch 1 · {evidence_caption}**\n\n'
                      f'**QUALITY APPROVAL {approval_state} — no approved winner.** '
                      'This is a replay of committed measured evidence, not live Meta API optimization.\n\n'
                      + banner)
        b8_report = data['artifact_validation'].get('profiles', {}).get('fp16_sdpa_b8', {})
        t5 = b8_report.get('t5', {})
        if t5.get('passed') and session_status.get('state') == 'blocked':
            banner += ('\n\n### Validation pipeline\n\n'
                       '**Execution: PASS** → **T5 batch consistency: PASS** → '
                       '**Completeness: REJECTED (truncation)** → **Calibration: BLOCKED** → '
                       '**Held-out: NOT RUN** → **Apply: UNAVAILABLE**')

    runtime_gpu, _ = _gpu_info()
    benchmark_gpu, model = _result_metadata(benchmarks)
    gpu_and_model = f"### Hardware and model\n\n{runtime_gpu}\n\n{benchmark_gpu}\n\n{model}"

    profile_rows = []
    for result in benchmarks:
        step = _profile_decision(result, decisions)
        status, reason = _status_and_reason(result, step, data.get('artifact_validation'))
        quality_source = dict(result)
        if step:
            quality_source["_agent_validation"] = step.get("validation")
        quality_f1, rouge_l = _quality_values(quality_source)
        artifact_profile = data.get('artifact_validation', {}).get('profiles', {}).get(result.get('profile_name'), {})
        if artifact_profile.get('config') == result.get('requested_config'):
            quality_f1 = artifact_profile.get('mean_token_f1_vs_default')
            rouge_l = artifact_profile.get('mean_gold_rouge_l')
        speedup = _speedup(
            result.get("docs_per_sec") if str(result.get("status", "")).lower() == "success" else None,
            competent.get("docs_per_sec")
            if competent and str(competent.get("status", "")).lower() == "success"
            else None,
        )
        profile_rows.append(
            [
                _display_value(result.get("profile_name")),
                _format_metric(result.get("docs_per_sec")),
                _format_metric(result.get("joules_per_doc")),
                f"{speedup:.2f}x" if speedup is not None else "Not recorded",
                _format_metric(quality_f1),
                _format_metric(rouge_l),
                status,
                reason,
            ]
        )
    if not profile_rows:
        profile_rows = [
            [NAIVE_PROFILE, "Not recorded", "Not recorded", "Not recorded", "Not recorded", "Not recorded", "Not run", "No benchmark result exists"],
            [DEFAULT_PROFILE, "Not recorded", "Not recorded", "Not recorded", "Not recorded", "Not recorded", "Not run", "No benchmark result exists"],
        ]
    recorded_names = {str(result.get("profile_name")) for result in benchmarks}
    for profile_name in KNOWN_PROFILES:
        if profile_name not in recorded_names:
            profile_rows.append(
                [
                    profile_name,
                    "Not recorded",
                    "Not recorded",
                    "Not recorded",
                    "Not recorded",
                    "Not recorded",
                    "Not run",
                    "No benchmark result exists for this profile",
                ]
            )

    baseline_rows = []
    for label, profile in (("Naive baseline (fp32/eager/batch 1)", naive), ("Competent default (fp16/SDPA/batch 1)", competent)):
        baseline_rows.append(
            [
                label,
                (
                    "Failed"
                    if str(profile.get("status", "")).lower() == "failed"
                    else "Measured"
                    if str(profile.get("status", "")).lower() == "success"
                    else "Status not recorded"
                )
                if profile
                else "Not run",
                _format_metric(profile.get("docs_per_sec")) if profile else "Not recorded",
                _format_metric(profile.get("latency_p50_sec"), " s") if profile else "Not recorded",
                _format_metric(profile.get("latency_p95_sec"), " s") if profile else "Not recorded",
                _format_metric(profile.get("joules_per_doc"), " J/doc") if profile else "Not recorded",
            ]
        )

    chosen_name = _display_value(chosen.get("profile_name")) if chosen else "No configuration selected"
    chosen_config = chosen.get("actual_config") or chosen.get("requested_config") if chosen else None
    heldout = _heldout(chosen)
    heldout_event = _heldout_event(data, chosen)
    event_validation = heldout_event.get("apply_result", {}).get("validation", {})
    if not isinstance(event_validation, dict):
        event_validation = {}
    event_quality = _quality_values({"_agent_validation": event_validation})
    heldout_benchmark = heldout_event.get("benchmark")
    if not isinstance(heldout_benchmark, dict):
        heldout_benchmark = {}
    heldout["quality_f1"] = heldout.get("quality_f1") or event_quality[0]
    heldout["rouge_l"] = heldout.get("rouge_l") or event_quality[1]
    heldout["docs_per_sec"] = heldout.get("docs_per_sec") or heldout_benchmark.get("docs_per_sec")
    heldout["joules_per_doc"] = heldout.get("joules_per_doc") or heldout_benchmark.get("joules_per_doc")
    if not heldout.get("decision") and event_validation:
        heldout["decision"] = "accepted" if event_validation.get("accepted") is True else "rejected"
        heldout["reason"] = "; ".join(event_validation.get("reasons", []))
    integrated_confirmation = data.get('confirmation') or {}
    if integrated_confirmation:
        confirmed_validation = integrated_confirmation.get('validation', {})
        heldout['decision'] = 'accepted' if confirmed_validation.get('accepted') is True else 'rejected'
        heldout['reason'] = '; '.join(confirmed_validation.get('reasons', []))
        heldout['quality_f1'], heldout['rouge_l'] = _quality_values({'_agent_validation': confirmed_validation})
        heldout['docs_per_sec'] = integrated_confirmation.get('benchmark', {}).get('docs_per_sec')
        heldout['joules_per_doc'] = integrated_confirmation.get('benchmark', {}).get('joules_per_doc')
        if confirmed_validation.get('accepted') is not True:
            chosen_name += ' (tuning candidate only; held-out rejected)'
    heldout_decision = str(heldout.get("decision") or heldout.get("status") or "").lower()
    if any(token in heldout_decision for token in ("accept", "promote", "pass")):
        heldout_summary = "Held-out confirmation: accepted in recorded results."
    elif any(token in heldout_decision for token in ("reject", "fail")):
        heldout_summary = "Held-out confirmation: rejected in recorded results."
    else:
        heldout_summary = "Held-out confirmation: not recorded; do not treat as passed."
    winner_lines = [
        f"**Chosen profile:** {chosen_name}",
        f"**Configuration:** `{json.dumps(chosen_config, sort_keys=True) if chosen_config else 'Not recorded'}`",
        heldout_summary,
        f"**Held-out quality F1:** {_format_metric(heldout.get('quality_f1'))}",
        f"**Held-out ROUGE-L:** {_format_metric(heldout.get('rouge_l'))}",
        f"**Held-out throughput:** {_format_metric(heldout.get('docs_per_sec'), ' docs/s')}",
        f"**Held-out energy:** {_format_metric(heldout.get('joules_per_doc'), ' J/doc')}",
        f"**Held-out reason:** {_display_value(heldout.get('reason') or 'No reason recorded')}",
    ]

    comparisons: list[list[str]] = []
    winner_docs = chosen.get("docs_per_sec") if chosen else None
    winner_energy = chosen.get("joules_per_doc") if chosen else None
    for label, baseline in (("vs naive baseline", naive), ("vs competent default", competent)):
        measured_baseline = baseline if baseline and str(baseline.get("status", "")).lower() == "success" else None
        speedup = _speedup(
            winner_docs if chosen else None,
            measured_baseline.get("docs_per_sec") if measured_baseline else None,
        )
        saving = _percent_saving(
            winner_energy if chosen else None,
            measured_baseline.get("joules_per_doc") if measured_baseline else None,
        )
        comparisons.append(
            [
                label,
                f"{speedup:.3f}x" if speedup is not None else "Not recorded",
                f"{saving:.2f}%" if saving is not None else "Not recorded",
            ]
        )
    default_speedup = _speedup(
        winner_docs if chosen else None,
        competent.get("docs_per_sec")
        if competent and str(competent.get("status", "")).lower() == "success"
        else None,
    )
    if chosen and default_speedup is not None:
        gate = "GATE 1 PASSED" if default_speedup >= 1.2 else "GATE 1 NOT PASSED"
    else:
        gate = "GATE 1 NOT RUN - winner and competent-default measurements are required"
    if integrated_confirmation and integrated_confirmation.get('validation', {}).get('accepted') is not True:
        gate = 'HELD-OUT REJECTED - no final approved winner'
    winner_lines.append(f"**{gate}**")
    confirmation = data.get('confirmation') or {}
    if confirmation:
        winner_lines.append('**Integrated held-out confirmation:** ' +
                            ('accepted' if confirmation.get('validation', {}).get('accepted') is True else 'rejected'))
        winner_lines.append('**Activation:** not attempted; persistent GPU runtime is not connected.')
        winner_lines.append('**T8 speed confirmation:** ' +
                            ('within 10%' if confirmation.get('speedup_confirmed') else 'not confirmed; use held-out speedup only'))
        winner_lines.append('**Held-out speedup vs default:** ' + _format_metric(confirmation.get('heldout_speedup_vs_default'), 'x'))

    rejection_rows = []
    for result in benchmarks:
        step = _profile_decision(result, decisions)
        status, reason = _status_and_reason(result, step, data.get('artifact_validation'))
        if status in {"Rejected", "Failed"}:
            rejection_rows.append([result.get("profile_name", "Profile not recorded"), status, reason])
    if not rejection_rows:
        rejection_rows = [["No rejected or failed configurations recorded", "Not run", "No rejection/failure reason available"]]

    agent_rows = [
        [
            _display_value(step.get("hypothesis")),
            _source_label(step),
            _display_value(step.get("decision", step.get("status"))),
            _display_value(step.get("reason")),
        ]
        for step in steps
    ] or [["No agent steps recorded", "Source not recorded", "Not run", "No agent log entries available"]]

    apply_events = data.get("apply_events", [])
    dashboard_events = data.get("dashboard_apply_events", [])
    last_apply = dashboard_events[-1] if dashboard_events else (
        apply_events[-1] if apply_events else None
    )
    if not last_apply:
        apply_summary = (
            "**No apply attempt recorded.** The dashboard is not connected to an "
            "active inference runtime, so no profile has been applied or restored."
        )
    else:
        event = last_apply.get("apply_result", last_apply)
        action = event.get("action")
        active_config = event.get("active_config")
        if action == "promoted":
            apply_summary = (
                "**Recorded status: promoted.**\n\n"
                f"Active configuration: `{json.dumps(active_config, sort_keys=True) if active_config else 'Not recorded'}`"
            )
        elif action == "rolled_back":
            apply_summary = (
                "**Recorded status: rolled back.** The safe-apply log reports restoration "
                f"of `{json.dumps(active_config, sort_keys=True) if active_config else 'the default configuration'}`."
            )
        else:
            apply_summary = (
                "**Apply status is unverified.** A log entry exists but does not contain "
                "a recognized promoted or rolled-back action."
            )

    return {
        "banner": banner,
        "gpu_and_model": gpu_and_model,
        "profile_rows": profile_rows,
        "baseline_rows": baseline_rows,
        "comparison_rows": comparisons,
        "winner_markdown": "\n\n".join(winner_lines),
        "rejection_rows": rejection_rows,
        "agent_rows": agent_rows,
        "apply_summary": apply_summary,
        "snapshot": {
            **data,
            "benchmarks": benchmarks,
            "chosen": chosen,
            "naive": naive,
            "competent": competent,
            "winner": chosen,
        },
    }


def refresh_dashboard() -> tuple[Any, ...]:
    view = build_dashboard_view(load_dashboard_data())
    return (
        view["banner"],
        view["gpu_and_model"],
        view["profile_rows"],
        view["baseline_rows"],
        view["winner_markdown"],
        view["rejection_rows"],
        view["agent_rows"],
        view["apply_summary"],
        view["snapshot"],
    )


def safe_apply(snapshot: dict[str, Any] | None) -> str:
    """Fail closed until the app provides a real active-model apply callback."""
    if not snapshot or not snapshot.get("has_real_benchmarks"):
        return "Not applied: real benchmark results are not available."
    return (
        "Not applied: the dashboard has no active inference runtime to update. "
        "Agent B's safe_apply requires a real apply(config) callback; no profile "
        "was activated or rolled back."
    )


def optimize_action():
    from tunix.application import run_optimization
    if not RUN_LOCK.acquire(blocking=False):
        return 'Another GPU operation is running.'
    try:
        session = run_optimization(RESULTS_DIR)
        status = _read_json(Path(session) / 'session.json')
        if not status.get('winner'):
            return 'Search finished; NO validated winner. Review rejection reasons in ' + str(session) + '. Nothing applied.'
        return 'Tuning candidate selected; held-out confirmation is still required. Results: ' + str(session) + '. Nothing applied; fallback sources remain labeled.'
    except Exception as exc:
        from tunix.validator import CalibrationBlocked
        if isinstance(exc, CalibrationBlocked):
            return str(exc) + '. No search or winner selection occurred. Baseline diagnostics are saved in session.json.'
        return 'Optimization stopped: ' + type(exc).__name__ + '. Check saved failure/rejection JSON. MODEL_API_KEY must be set; no key is displayed.'
    finally:
        RUN_LOCK.release()


def confirm_action(snapshot):
    from tunix.application import confirm_heldout
    if not snapshot or not snapshot.get('has_real_benchmarks'):
        return 'No measured session to confirm.'
    if not RUN_LOCK.acquire(blocking=False):
        return 'Another GPU operation is running.'
    try:
        result = confirm_heldout(snapshot['results_dir'])
        return ('Held-out confirmation: ' + ('accepted' if result['validation']['accepted'] else 'rejected')
                + '. No runtime activation or GPU rollback was attempted.')
    except Exception as exc:
        return 'Held-out confirmation stopped: ' + type(exc).__name__
    finally:
        RUN_LOCK.release()


def _create_evidence_app() -> Any:
    import gradio as gr

    with gr.Blocks(title=TITLE) as demo:
        with gr.Row(elem_classes=["hero"]):
            with gr.Column(scale=5):
                gr.Markdown("TUNIX  /  INFERENCE WORKBENCH", elem_classes=["eyebrow"])
                gr.Markdown("# Find the fastest correct profile.")
                gr.Markdown(
                    "Compare measured inference profiles, review the evidence, and keep quality in view.",
                    elem_classes=["subtitle"],
                )
            with gr.Column(scale=1, min_width=190):
                refresh = gr.Button("Refresh saved results", variant="secondary")
        banner = gr.Markdown(elem_classes=["notice"])
        snapshot_state = gr.State()

        gr.Markdown("01  /  OVERVIEW", elem_classes=["section-label section-heading"])
        with gr.Row():
            with gr.Column(scale=3, elem_classes=["surface", "optix-card"]):
                gr.Markdown(
                    "### A careful search for better inference\n\n"
                    "Tunix tests configurations against a competent default, "
                    "then checks whether a faster profile preserves output quality. "
                    "A benchmark result is not a quality pass by itself.",
                    elem_classes=["overview-copy"],
                )
                gr.Markdown(
                    "Numbers below come from saved benchmark results. Missing evidence stays unreported.",
                    elem_classes=["quiet"],
                )
            with gr.Column(scale=2, elem_classes=["surface", "optix-card"]):
                gpu = gr.Markdown()

        gr.Markdown("02  /  EXPERIMENTS", elem_classes=["section-label section-heading"])
        gr.Markdown(
            "Compare both baselines first. Speedup is calculated only when the profile and competent default have successful measurements.",
            elem_classes=["quiet"],
        )
        baselines = gr.Dataframe(
            headers=["Reference profile", "Run", "docs/s", "p50 latency", "p95 latency", "Energy / doc"],
            datatype=["str", "str", "str", "str", "str", "str"],
            interactive=False,
            wrap=True,
            elem_classes=["surface", "results-table"],
        )
        profiles = gr.Dataframe(
            headers=[
                "Profile",
                "docs/s",
                "J / doc",
                "Raw vs default (not approval)",
                "Quality F1",
                "ROUGE-L",
                "Validation",
                "Recorded reason",
            ],
            datatype=["str", "str", "str", "str", "str", "str", "str", "str"],
            interactive=False,
            wrap=True,
            elem_classes=["surface", "results-table"],
        )
        gr.Markdown("03  /  OPTIMIZE", elem_classes=["section-label section-heading"])
        with gr.Row():
            with gr.Column(scale=2, elem_classes=["surface", "optix-card"]):
                gr.Markdown("### Request an optimization")
                gr.Markdown(
                    "Runs real Meta API proposals and GPU benchmarks (up to eight experiments plus baselines and batch-1 checks). Set MODEL_API_KEY and HF_TOKEN in the server environment. Failed quality or T5 checks cannot win. No profile is automatically activated.",
                    elem_classes=["workflow-note"],
                )
                optimize_button = gr.Button(
                    "Request Meta optimization",
                    variant="primary",
                    interactive=True,
                )
                optimize_result = gr.Textbox(label='Optimization status', interactive=False)
                gr.Markdown(
                    "Saved proposals appear alongside their source and recorded decision.",
                    elem_classes=["quiet"],
                )
            with gr.Column(scale=3):
                gr.Markdown("### Proposal history")
                agent_log = gr.Dataframe(
                    headers=["Hypothesis", "Source", "Decision", "Reason"],
                    datatype=["str", "str", "str", "str"],
                    interactive=False,
                    wrap=True,
                    elem_classes=["surface", "results-table"],
                )
        gr.Markdown("### Rejected and failed candidates", elem_classes=["section-heading"])
        rejections = gr.Dataframe(
            headers=["Profile", "Status", "Recorded reason"],
            datatype=["str", "str", "str"],
            interactive=False,
            wrap=True,
            elem_classes=["surface", "results-table"],
        )

        gr.Markdown("04  /  APPLY & ROLLBACK", elem_classes=["section-label section-heading"])
        with gr.Row():
            with gr.Column(scale=3, elem_classes=["surface", "optix-card"]):
                gr.Markdown("### Held-out confirmation")
                winner = gr.Markdown()
                confirm_button = gr.Button('Confirm winner on held-out documents')
                confirm_result = gr.Textbox(label='Held-out status', interactive=False)
            with gr.Column(scale=2, elem_classes=["surface", "optix-card"]):
                gr.Markdown("### Apply state")
                apply_summary = gr.Markdown()
                apply_button = gr.Button("Check apply readiness", variant="secondary")
                apply_result = gr.Textbox(
                    label="Readiness",
                    value="No profile has been applied.",
                    interactive=False,
                    lines=2,
                )

        gr.Markdown(
            "Tunix  ·  Evidence from saved results only  ·  GPU and optimizer execution are not implied by this interface",
            elem_classes=["quiet"],
        )

        outputs = [
            banner,
            gpu,
            profiles,
            baselines,
            winner,
            rejections,
            agent_log,
            apply_summary,
            snapshot_state,
        ]
        demo.load(refresh_dashboard, outputs=outputs)
        refresh.click(refresh_dashboard, outputs=outputs)
        optimize_button.click(optimize_action, outputs=optimize_result).then(refresh_dashboard, outputs=outputs)
        confirm_button.click(confirm_action, inputs=snapshot_state, outputs=confirm_result).then(refresh_dashboard, outputs=outputs)
        apply_button.click(safe_apply, inputs=snapshot_state, outputs=apply_result)
    return demo


from tunix.workbench_ui import CSS


def create_app() -> Any:
    from tunix.workbench_ui import create_workbench
    return create_workbench()


if __name__ == "__main__":
    import gradio as gr

    app = create_app()
    app.launch(
        max_file_size="32mb",
        favicon_path=str(ROOT / 'assets' / 'tunix-symbol.svg'),
        server_port=int(os.environ.get('GRADIO_SERVER_PORT', '7860')),
        share=os.environ.get('GRADIO_SHARE', '0') == '1',
        server_name=os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1"),
        theme=gr.themes.Soft(primary_hue="cyan", secondary_hue="slate", neutral_hue="slate"),
        css=CSS,
    )
