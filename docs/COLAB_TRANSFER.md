# Colab → Tunix workbench: final demo workflow

Only the supported T4 / Llama-3.2-1B-Instruct / CNN-DailyMail path is live.
No model approval, live Meta run or GPU deployment is implied by transferring
data. Import is offline validation of supplied files, not hardware attestation.

## 1. Google Colab: install the local CLI

Select **Runtime > Change runtime type > T4 GPU**. Accept the Llama license
on Hugging Face first. In a fresh notebook run:

```python
import os, sys, subprocess, getpass
from pathlib import Path

repo = Path('/content/Tunix')  # separate checkout; preserves /content/Optix/results
if not repo.exists():
    subprocess.run(['git', 'clone',
                    'https://github.com/karim-elmaraghy/Tunix.git', str(repo)], check=True)
os.chdir(repo)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-e', '.[gpu]'], check=True)
subprocess.run(['tunix', 'doctor'], check=True)
if not Path('tunix.json').exists():
    subprocess.run(['tunix', 'init'], check=True)
os.environ['HF_TOKEN'] = getpass.getpass('HF_TOKEN (read token, accepted Llama license): ')
```

`doctor` checks hardware/framework compatibility, not model entitlement.
Benchmarking does not require MODEL_API_KEY. Real optimization separately
requires it, entered securely with getpass; it still stops if calibration fails.

## 2. Measure, export and download

```python
run = subprocess.run(['tunix', 'benchmark'], check=False)
print('Benchmark exit:', run.returncode)  # 0=measured, 2=failed gate/execution
# Export preserves the current run's actual files even if a profile failed.
# Use a NEW output name for every export; existing ZIPs are never overwritten.
zip_path = '/content/tunix-results.zip'
subprocess.run(['tunix', 'export', '--output', zip_path], check=True)
from google.colab import files
files.download(zip_path)
```

The CLI invokes the existing real Gate 1 runner with three repeats. It does
not change the original committed two-repeat evidence or bypass T5/quality.
No quality-approved winner is inferred from benchmark success.

For your ALREADY completed original V2 run in `/content/Optix/results/`,
export that directory explicitly, without rerunning or modifying measurements:

```python
subprocess.run(['tunix', 'export', '--session', '/content/Optix/results',
                '--output', '/content/tunix-original-v2.zip'], check=True)
files.download('/content/tunix-original-v2.zip')
```

Both this export and the original `optiloop_v2_measured_results.zip` are
accepted. The old six-file ZIP has no embedded prompt metadata; its exact
committed file hashes identify this known V2 dataset. Other metadata-free
archives are labeled unknown prompt version and cannot establish calibration.

## 3. Windows / VS Code: open the workbench and upload

Activate your Python environment, then use PowerShell:

```powershell
cd C:\path\to\Tunix
python -m pip install -e .
# Run once in this checkout/project; skip if tunix.json already exists.
tunix init
tunix report --port 7868
```

If port 7868 is already serving the review preview, open
**http://127.0.0.1:7868** instead of launching another server. A fresh clone
may use any directory; initialize from the developer project you want to use.

1. Click **Open Workbench**.
2. Use **Import benchmark results** to choose the downloaded ZIP.
3. Wait for **Imported GPU results**. The selected **Evidence session**
   changes to the new import. Charts, tables, gate diagnostics and status
   refresh together; no Optimize click is necessary and no API/GPU call occurs.
4. For the original V2 data, see raw **5.63x** throughput / **81.8%** energy
   reduction, T5 PASS, and separate truncation rejection/calibration BLOCKED.
5. Switch **Evidence session** between imported sessions, **Bundled V2 replay**,
   **Latest local measurements** and individual local runs. Imports never
   overwrite that local-current pointer or old sessions.
6. Inspect **Results / Evidence** and **Validate & export**. Imported agent
   logs/approval claims remain originals under `evidence/`; they do not become
   trusted live Meta steps or a winner. No Apply action is performed.

## Transport and validation boundaries

- ZIP ≤32 MiB; ≤128 files; each expanded file ≤8 MiB; total expansion ≤64 MiB;
  maximum expansion ratio 100:1. Summarization text/doc counts are also bounded
  to keep local ROUGE computation finite. Limits are checked before accepting
  evidence; the central-directory count is checked before ZIP entry allocation.
- Only supported JSON/JSONL artifact names and optional `heldout/` artifacts
  are accepted. Traversal, absolute/Windows paths, duplicate case-folded names,
  symlinks/special files, encryption, ZIP64/multidisk layouts, unsupported
  compression, bad JSON/duplicate keys/nonfinite numbers and malformed schemas
  are rejected. No archive code or extraction instructions are executed.
- Every import receives a new `import_<id>` directory. Original entries are
  byte-identical under `evidence/`. Root benchmark/output copies are also
  byte-identical. Locally derived `session.json` and `validation_report.json`
  are distinct from the uploaded originals and bound to their hashes.
- Recomputed evidence includes raw comparisons, output-validity flags,
  token F1, gold ROUGE-L, supported baseline calibration and matched batch-1
  T5 comparisons. Missing references/prompt provenance/checks are explicit.
  A tuning pass never establishes held-out approval or a deployment winner.
- Tunix export includes a hash manifest and available session metadata. Hashes
  prove byte integrity, not that uploaded GPU/model claims are independently
  authenticated. Synthetic metadata remains synthetic; it is never upgraded
  to measured evidence. No credentials/environment secrets are added by export.

Verification: original V2 and Tunix-generated ZIPs were both transported through
the actual Gradio multipart upload + import endpoint and returned updated
dashboard/selection payloads, not merely HTTP 200. No new GPU or Meta run was
performed to validate this transfer. Browser file-picker clicks remain a
manual review item when the browser access host is unavailable.
