# Tunix local agent: VS Code terminal quickstart

Tunix is a locally installed Python CLI and dashboard, not a VS Code extension.
The only supported live MVP is **CUDA Tesla T4 +
meta-llama/Llama-3.2-1B-Instruct + CNN/DailyMail summarization**. It benchmarks
this fixed workload, not arbitrary code or the model already running in your
project. Other GPUs/models are unsupported; compile paths remain experimental.
No package has been published to PyPI. Editable installation is supported.

## Install and run the saved-evidence demo

From your VS Code terminal (PowerShell on Windows):

```powershell
git clone https://github.com/karim-elmaraghy/Tunix.git
cd Tunix
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
tunix --help
tunix doctor
tunix init
tunix report --replay --port 7865
```

On macOS/Linux/Colab use `source .venv/bin/activate` instead of the PowerShell
activation command. A developer already using a virtual environment may skip
creating another one. `doctor` exits 2 on unsupported/missing hardware; this
does not prevent replay. The report opens http://127.0.0.1:7865 locally. Use
`--no-browser` for remote Colab (the existing authenticated notebook launch in
INTEGRATION.md is the sharing option; no automatic public exposure is added).

To install into another project's environment, activate that environment,
then `python -m pip install -e "C:\path\to\Tunix"`. Stay in the developer
project and run `tunix init`. Do not copy Tunix sources into the project.

## Real runtime workflow

On a licensed-model Colab T4 or supported CUDA T4 environment, install GPU
dependencies in the same active environment:

```sh
python -m pip install -e ".[gpu]"
tunix doctor
tunix init
tunix benchmark
tunix optimize --budget 8
tunix report
```

If installed from another project directory, use the absolute checkout path
with the extra: `python -m pip install -e "C:\path\to\Tunix[gpu]"`.
Accept the gated Llama license on Hugging Face. Set `HF_TOKEN` (read access)
and `MODEL_API_KEY` securely in that terminal environment; do not save secrets
in the project config. Example hidden entry in a Python session/Colab:

```python
import getpass, os, subprocess
os.environ['HF_TOKEN'] = getpass.getpass('HF_TOKEN: ')
os.environ['MODEL_API_KEY'] = getpass.getpass('MODEL_API_KEY: ')
subprocess.run(['tunix', 'doctor'], check=False)
subprocess.run(['tunix', 'benchmark'], check=False)
subprocess.run(['tunix', 'optimize'], check=False)
```

The subprocesses inherit credentials; they do not appear in command arguments.
Desktop users may instead enter masked credentials in the connected-T4
workbench. CLI credentials come only from environment variables.

## What each command actually does

- `tunix doctor [--json]`: probes server `nvidia-smi` for name, VRAM and driver;
  probes torch CUDA availability/build version/device 0; reports installed
  framework versions and credential-presence booleans. It does not transmit
  keys or verify model entitlement/API connectivity. It never inspects a
  browser/visitor GPU. Exit 0 means supported hardware/dependencies detected,
  not quality approval; exit 2 means not ready for live execution.
- `tunix init`: exclusively creates `tunix.json` with the fixed supported
  model/workload and `.tunix/results` output path. Never overwrites an existing
  config. `--config PATH` and `--results-dir PATH` customize paths, not models.
- `tunix benchmark`: preflights GPU/dependencies/HF token, invokes the existing
  real Gate 1 runner with three repeats, and preserves actual status/error
  records. Success means measured execution only, never a quality-approved
  winner or Meta search. T5 or execution failure exits 2.
- `tunix optimize`: preflights all requirements including MODEL_API_KEY,
  invokes real baseline measurement/calibration/Meta search/validation via
  `run_optimization()`. Calibration failure stops before Meta candidate
  search. Fallback remains labeled. No candidate/blocked run exits 2;
  a found tuning candidate still needs held-out confirmation, not activation.
- `tunix report`: opens the real dashboard with this project's current
  session/result path. The workbench distinguishes measured benchmark
  evidence, Meta steps (if present), validation and no approved winner.
  `--replay` explicitly switches to the bundled original V2 T4 data; it is
  never presented as a new experiment or live Meta call.
- `tunix export --output tunix-results.zip`: packages the current measured
  session with an integrity manifest and available metadata. Use `--session
  PATH` for an older Colab results directory. Refuses existing output files;
  no GPU/API call or approval is implied. See COLAB_TRANSFER.md for the ZIP
  upload-to-workbench workflow, archive limits and exact notebook cells.

Paths in `tunix.json` resolve relative to that file, independent of the
installation location. Results are stored per run, with a `current_session`
pointer; existing measured JSON/rejection schemas are unchanged. The internal
`tunix` namespace is the only supported package namespace. Exported measured candidate settings are
explicitly NOT APPROVED. No persistent GPU runtime Apply is implemented.

## Current evidence and honest limitations

Bundled V2 records show 5.63x raw throughput and 81.8% lower energy for B8
vs competent B1, with T5 passing. Baseline truncation still blocks strict
calibration; no approved optimization/held-out or rollback result exists.
Original records used two repeats; the CLI's existing Gate 1 invocation uses
the specified three-repeat setting without modifying those records.
The install/CLI/dashboard path is CPU-tested. Live T4 execution of the new
CLI, live Meta access, gated-model entitlement, and GPU activation/rollback
are not claimed verified. See V2_VALIDATION.md for the actual data findings.
