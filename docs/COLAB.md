# Tunix — Google Colab T4 instructions (Gate 1)

> **Prompt V2 (current):** prompts are rendered through the Llama 3.2
> chat template with a generation prompt. Earlier V1 runs fed raw text
> and are preserved as failed-validation results; **never compare V2
> candidates against V1 baselines** — rerun the baselines under V2.
> Verify formatting before benchmarking (prints one rendered prompt;
> must start with `<|begin_of_text|>` once and end with the assistant
> header): `python -c "from tunix.benchmark import load_model, render_prompt; from tunix.profiles import profile_config; from tunix.workload import load_workload; t,_=load_workload(); b=load_model(profile_config('default_fp16_sdpa_b1')); print(repr(render_prompt(b['tokenizer'], t[0])[:400]))"`

Status: GPU code is **written, not run**. Nothing is a result until you run
Gate 1 below on a real T4 and paste the output back. Do not quote any
docs/s or J/doc number that did not come out of your own run's
`results/benchmark_*.json`.

## Before you open Colab (5 minutes, once)

1. Accept the license for `meta-llama/Llama-3.2-1B-Instruct` on Hugging
   Face (the model is gated; without this the download returns 401).
2. Create a Hugging Face access token (read permission is enough).
3. Have the Meta API key for the optimizer (`MODEL_API_KEY`) ready. Gate 1
   itself does not call it, but later gates (T0/T9) do.

## Gate 1 — copy-paste cell

In Colab: **Runtime → Change runtime type → T4 GPU**, then paste this whole
block into one cell and run it. It asks for the two tokens with `getpass`
(they are never printed or committed).

```python
import getpass
import os
import subprocess
import sys

REPO_URL = "https://github.com/karim-elmaraghy/Tunix.git"
BRANCH = "main"

def sh(cmd):
    print("+", cmd)
    subprocess.run(cmd, shell=True, check=True)

# GPU must be a T4 (free tier). If this fails, the runtime is CPU-only:
# stop and re-select a T4 runtime; CPU numbers are not Gate 1 results.
sh("nvidia-smi")

if not os.path.exists("Tunix"):
    sh(f"git clone {REPO_URL}")
os.chdir("Tunix")
sh(f"git fetch origin && git checkout {BRANCH} && git pull --ff-only")

sh(f"{sys.executable} -m pip install -q -r requirements.txt")

os.environ["HF_TOKEN"] = getpass.getpass("HF_TOKEN (Llama license): ")
os.environ["MODEL_API_KEY"] = getpass.getpass("MODEL_API_KEY (Meta API): ")

# Gate 1: naive vs competent default vs batch 4/8 vs token-cap, 20 tuning
# docs, 2 repeats. Expect several minutes (fp32 naive is the slow one).
sh(f"{sys.executable} -m tunix.run --gate1")
```

Optional, in a second cell (compile is slow to warm up; T3 wants its
compile time, recompilations, and break-even reported from the console):

```python
sh(f"{sys.executable} -m tunix.run --gate1 --with-compile")
```

## What Gate 1 prints (check every line)

- `GPU (nvidia-smi): Tesla T4` plus torch/transformers versions (T0).
- One console line per profile with counter vs polling J/doc. If it says
  `WARNING T2 ... differ by >15%`, keep both numbers; do not pick one.
- The table: `docs/s`, `J/doc`, `vs default x`, `vs naive x`, `noisy`.
  Headline comparisons are **vs `default_fp16_sdpa_b1`**; `vs naive` is
  reported separately and labeled (R2). `noisy=yes` means repeats varied
  by >10% — rerun before trusting that row.
- `T5 batching: ... mean F1=... min F1=...`. If the run exits with
  `T5 FAIL`, batching is buggy (padding/mask), not "low quality": stop
  and report the full output instead of working around it.
- `model load: confirmed (...)` and `saved: results/benchmark_<ts>.json`.

## After Gate 1 (same runtime, in order)

```python
# T1 determinism: competent default twice, identical-output fraction.
sh(f"{sys.executable} -m tunix.run --determinism")
# T8 held-out confirmation: replace with the winning profile name.
sh(f"{sys.executable} -m tunix.run --heldout fp16_sdpa_b4")
# T9 grid cache (long: 24 configs). Run once; keep the JSON.
sh(f"{sys.executable} -m tunix.run --grid --repeats 3")
# T11 reproducibility on a fresh runtime:
sh(f"{sys.executable} -m tunix.run --demo")
```

Download `results/benchmark_*.json` and `results/outputs_*.json`
(Files pane → right-click → Download) and paste the full Gate 1 output
back to the team. Every number in the README/dashboard must come from
those JSON files (R4).

## Pass criteria (do not move the goalposts)

- **T7 headroom:** the best *quality-valid* profile beats
  `default_fp16_sdpa_b1` by ≥20% in docs/s or J/doc. Beating only the
  naive fp32 baseline is not a win; say so explicitly if that happens.
- **T1:** determinism ≥99% identical outputs, else fix seeds/greedy first.
- **T5:** batch-4 vs batch-1 F1 ≥ 0.95 per doc, else it is a bug to fix.

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| `401` / gated-model error | Accept the Llama-3.2-1B-Instruct license, then re-enter `HF_TOKEN` |
| `nvidia-smi` fails / no T4 | Runtime → Change runtime type → T4 GPU, then rerun from the top |
| Dataset script error on load | The loader retries the Hub parquet revision automatically; paste the real error if both fail |
| CUDA out of memory | Expected profiles fit a T4; if fp32 batch-8 OOMs it is recorded as `status:"failed"` with the real error — report it, do not delete the row |
| NVML energy is `None` | Colab exposes NVML on T4; if `J/doc` is `n/a`, report timing only and say energy was unavailable |

## Exact Gate 1 instructions (for the owner)

1. Colab → T4 runtime → paste the Gate 1 cell above → run.
2. Paste the **entire output** plus `results/benchmark_*.json` back here.
3. Only then are GPU numbers real; until then: **written, not run**.
