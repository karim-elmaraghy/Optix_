# Integrated Tunix demonstration

This branch combines PR #4 (9068273), PR #5 (4eee925), and PR #6
(710db1f) without changing their active branches. Additional application wiring
lives in `optiloop/application.py`, `optiloop/integration.py`, and `app.py`.
No benchmark JSON keys or rejection-log schemas were changed.

## V2 Gate 1 update (original artifacts received and checked)

Muse V2 `6ac0558` is now integrated here, retaining its commit history.
Prompt version is `v2-chat-template`. The original six JSON files from
Karim's Colab archive are now included byte-for-byte under `results/`.
Codex verified these saved measurements offline, without a new GPU run.
See [V2_VALIDATION.md](V2_VALIDATION.md) for exact failures and provenance:

| Profile | docs/s | J/doc | Reported cap flags |
|---|---:|---:|---:|
| FP32 B1 | 0.3162 | 118.28 | not supplied |
| FP16 default B1 | 0.3522 | 108.27 | 5/20 |
| FP16 B4 | 1.2863 | 29.43 | 5/20 |
| FP16 B8 | 1.9822 | 19.67 | 5/20 |
| FP16 B1 cap64 | 0.6205 | 61.18 | 20/20 |

Both V2 batch checks reportedly passed T5 (minimum F1=1.000). B8's raw
speedup is 5.63x and energy reduction approximately 81.8%, **measured but
not quality-approved**. This supersedes the V1 T5 failure only for V2;
do not combine their baselines or calibration. The dashboard computes raw
speedup from the supplied original JSON; no narrative values were inserted
into benchmark JSON or hardcoded dashboard measurement fields.

Strict calibration remains BLOCKED: default B1 has five cap-reaching outputs.
No document is silently excluded and no threshold is weakened. Cap64 remains
rejected. Held-out confirmation has not passed; no approved winner or Apply
demonstration exists. Calibration requires valid default AND naive outputs;
FP32 is also blocked: the original artifacts flag four of its twenty outputs.

Muse was asked on the board to inspect the five baseline outputs and actual
generation-stop evidence, including possible EOS at the last allowed token.
Counts alone cannot distinguish EOS-at-cap from a true cutoff (the counter
includes EOS). Until evidence and any owner-approved resolution are available,
all such flags remain rejections. No prompt/cap/methodology change is made here.

## Final readiness and V2 handoff

Readiness check: the integrated Gradio application was actually launched on
localhost and returned HTTP 200, then was closed. Real artifact loading and
derived rejection reasons are now verified against the six original Colab
JSON files. Their statuses remain measured/rejected, never approved winners.
MODEL_API_KEY was absent, so the live API smoke is **untested**, not passed.

For the blocked-calibration demo, the ORIGINAL downloaded files are already
included in this integration branch. For a new run, copy only ORIGINAL
`benchmark_*.json`, `outputs_*.json` and any actual `agent_log.json` into the
checkout's `results/` directory. Keep V1/V2 evidence in separate directories;
do not overwrite it with synthetic samples. Run `python app.py`, refresh,
show measured-only or rejected rows and raw speeds, explain the reported
truncation/T5 failures and show that no winner is selected. If an actual
integrated run blocks, its saved `session.json` displays Calibration BLOCKED
with reasons. Optimize cannot label a no-winner run successful, and Check
apply readiness always reports no activation/runtime rollback. The results
viewer starts without a GPU or Meta credentials; do not click Optimize merely
to show saved data. If original files are absent, demonstrate sample/not-run
state honestly and mark real-data verification pending.

Muse's V2 commit `6ac0558` is included in this integration branch. It adds workload `messages` and
`PROMPT_VERSION`, and renders the official chat template in the benchmark;
`run_config` arguments and result schemas remain compatible. The adapter
preserves entire workload documents including `article`/`messages`. Restart
the Python process, run the combined CPU suite and start a NEW optimization
session with fresh default and naive baselines. Never reuse V1 calibration
or compare V2 candidates to V1 outputs. Session metadata now records prompt
version, and held-out confirmation refuses cross-version sessions before
GPU work. Existing unversioned sessions are treated as V1. Validator
thresholds, token caps and benchmark methodology were not changed here.

## Historical V1 T4 findings (separate from V2): calibration blocked

Owner-supplied Colab results (not independently rerun by Codex, and not
converted into measured result JSON from this narrative):

| Profile | Reached generation cap | Conservative truncation flag |
|---|---:|---:|
| default_fp16_sdpa_b1 | 8/20 at 128 tokens | 40% |
| fp16_sdpa_b1_cap64 | 10/20 at 64 tokens | 50% |
| fp16_sdpa_b4 | 8/20 at 128 tokens | 40% |
| fp16_sdpa_b8 | 7/20 at 128 tokens | 35% |

Reaching the cap is a conservative flag, not proof that a sentence was cut.
Under the existing strict policy it still rejects the output. **Calibration
cannot proceed with this default baseline:** calibrate() raises
CalibrationBlocked with doc-level truncation diagnostics. The application
records state=blocked and winner=null, displays the reason, and does not call
the search agent. It does not discard those eight documents or calibrate
against only the remaining twelve. No margin or threshold was changed.

Cap64 is not quality-validated. Batch 4 and 8 still fail T5 independently of
truncation; their raw speedups are not approved. **No configuration has passed
all quality requirements.** The honest demo is the integrated measured-results
viewer and explicit blocked/rejected decisions, not an approved speedup claim.
Proceeding with calibration requires owner-approved resolution and new valid
baseline artifacts; this integration does not change token caps, prompts,
decoding or benchmark methodology. Obtain/commit the original Colab artifact
files before displaying these narrative findings as measured dashboard rows.

## Local CPU verification and dashboard

```sh
python -m pip install pytest gradio==6.31.0
python -m pytest tests -q
python app.py
```

Open http://127.0.0.1:7860. This starts the UI without loading GPU libraries.
Real optimization requires the full requirements, model license, T4 and keys.
The optional tiny-Llama batching tests skip if torch/transformers are missing.
Synthetic tests use temporary directories, dummy API transports and fabricated
test sentinels; they are not measurements. No live Meta or GPU test was run by
the integration author.

## Exact Google Colab T4 setup

Accept the Hugging Face license for `meta-llama/Llama-3.2-1B-Instruct` first.
Select **Runtime > Change runtime type > T4 GPU**. Run this Python cell:

```python
import os, sys, subprocess, getpass
from pathlib import Path

subprocess.run(['nvidia-smi'], check=True)
repo = Path('/content/Optix-integration')
if not repo.exists():
    subprocess.run(['git', 'clone', '--branch', 'integration-optiloop',
                    'https://github.com/karim-elmaraghy/Optix.git', str(repo)], check=True)
os.chdir(repo)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-r', 'requirements.txt'], check=True)
os.environ['HF_TOKEN'] = getpass.getpass('HF_TOKEN: ')
os.environ['MODEL_API_KEY'] = getpass.getpass('MODEL_API_KEY: ')
```

Run the CPU suite in the same runtime (this also exercises tiny-model tests
when the installed torch/transformers combination supports them):

```python
subprocess.run([sys.executable, '-m', 'pytest', 'tests', '-q'], check=True)
```

Check the real API without displaying credentials or raw exception text:

```python
from optiloop.agent import meta_completion
try:
    reply = meta_completion([{'role': 'user', 'content': 'Reply OK only.'}])
    assert reply.strip()
except Exception as exc:
    print('Meta smoke failed:', type(exc).__name__)
else:
    print('Meta smoke passed')
```

Launch the dashboard in the notebook (the password protects the shared URL):

```python
import app, gradio as gr
demo = app.create_app()
demo.queue().launch(share=True, auth=('demo', getpass.getpass('Dashboard password: ')),
                    theme=gr.themes.Soft(), css=app.CSS)
```

Standalone local command: `python app.py`. Colab shell alternative after
setting the environment: `GRADIO_SHARE=1 python app.py` (without notebook auth).
The integration author tested localhost startup only, not public sharing.

## Demonstration sequence

1. Start with saved Gate 1 files in `results/` if available; otherwise the
   dashboard explicitly displays sample/not-run state. Do not create result
   files from the supplied narrative numbers. Existing raw T4 batch-4 and
   batch-8 V1 results failed T5. V2 reportedly passed T5 but remains blocked
   by truncation/quality calibration; neither run has an approved winner.
2. Click **Request Meta optimization**. This performs real GPU work: both
   tuning baselines, a default repeat for T1 (>=99% matching output text),
   up to eight Meta-selected experiments, and matching batch-1 references
   for every batched experiment. Each benchmark uses three repeats.
   Missing keys stop before model loading; failed API proposals are visibly
   labeled `rule_fallback`, not presented as real Meta decisions.
3. Only successful, exact-config experiments passing calibrated quality and
   T5 (every document F1 >=0.95) are eligible. No candidate means no winner;
   do not promote the fastest raw measurement. T5 rejection is recorded as
   `T5_batching_failed`, not a weakened quality threshold.
4. Review proposal sources, quality F1/ROUGE-L and rejection reasons. Click
   **Confirm winner on held-out documents**. This reruns winner plus both
   baselines on the ten disjoint held-out documents, three repeats, with
   another matching batch-1 comparison if needed. T8 reports whether the
   speedup is within 10%; otherwise use the held-out speedup only.
5. **Check apply readiness** deliberately reports unavailable. Benchmark
   models are unloaded after each experiment. There is no persistent serving
   runtime to activate or restore. The existing CPU safe-apply tests prove
   callback/state rollback with a bad eight-token fixture; they do not prove
   GPU model rollback. The required runtime lifecycle is specified in
   `optiloop/safe_apply.py`; no automatic GPU activation is performed.
6. Preserve and commit genuine measured JSON before making hackathon claims.
   Keep console energy cross-check/compile diagnostics too. T9 full-grid
   agent-vs-random ablation remains separate and is not implemented by this
   dashboard flow. No agent superiority claim follows from this demo alone.

## Artifact paths and limitations

- Each optimization creates `results/run_<id>/` with unchanged-schema
  `benchmark_session.json`, `outputs_<profile>.json`, and `agent_log.json`
  (JSONL, including rejection records).
- `results/current_session.json` points the dashboard to the latest measured
  session; dependency-injected synthetic runs never update it. `session.json`
  is a separate status/provenance sidecar, not a benchmark schema extension.
- Held-out evidence is in the session's `heldout/confirmation.json` plus
  real output bundles. Held-out runs do not overwrite tuning artifacts.
- `gpu_verified:false` means no manual hardware certification is recorded;
  a saved numerical result is not a claim that the integration author ran it.
- The UI serializes optimization/held-out operations with one process lock.
  This is a single-process hackathon app, not a multi-server GPU service.
