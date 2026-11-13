<p align="center"><img src="assets/tunix-symbol.svg" width="88" alt="Tunix"></p>

# Tunix

**Measure GPU inference. Check quality. Understand the tradeoffs.**

Tunix is a local Python CLI and interactive workbench for benchmarking LLM inference on a Tesla T4. It compares throughput and energy use, validates generated outputs, and makes blocked or rejected experiments visible. Built for the Meta Global AI Developer Hackathon.

[Get started](#quick-start) · [Developer guide](docs/DEVELOPER.md) · [Colab notebook](notebooks/tunix_colab.ipynb) · [Measured evidence](docs/V2_VALIDATION.md)

## What it does

- **Benchmark:** measure latency, throughput, and GPU energy across inference profiles.
- **Validate:** check output completeness, batch consistency, and calibrated quality before approving a candidate.
- **Inspect:** explore original measurements and rejection reasons in a local Gradio workbench.
- **Transport:** export and import results without changing the original measurement bytes.
- **Plan:** ask Meta for advisory experiments using measured evidence and a permitted configuration space.

## Quick start

Use Python 3.10+ in a virtual environment:

```sh
git clone https://github.com/karim-elmaraghy/Tunix.git
cd Tunix
python -m venv .venv
# macOS / Linux
source .venv/bin/activate
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
python -m pip install -e .
tunix init
tunix report --replay --port 7865
```

Open **http://127.0.0.1:7865** and select **Open Workbench**. Saved-results replay needs no GPU or API key. Run `tunix doctor` to inspect live-runtime support; exit code 2 means the environment does not meet the GPU requirements.

## Measured results, with limits

The committed V2 experiment compares FP16 batch 8 with FP16 batch 1 on a Tesla T4 using Llama 3.2 1B Instruct and CNN/DailyMail summarization.

| Measurement | Observed result |
| --- | --- |
| Raw throughput improvement | 5.63× |
| Energy reduction per document | 81.84% |
| Batch consistency (T5) | Passed; minimum F1 1.000 |
| Strict calibration | Blocked by truncated baseline outputs |
| Quality-approved winner | None |

These are raw measurements, **not a validated optimization or general performance guarantee**. The archive has two repeats, below the three-repeat requirement; held-out approval and persistent GPU activation/rollback have not been demonstrated. See the [validation report](docs/V2_VALIDATION.md) and original [results](results/).

## Run a live experiment

The supported MVP is **CUDA Tesla T4 + `meta-llama/Llama-3.2-1B-Instruct` + CNN/DailyMail**. Accept the model license and set `HF_TOKEN` securely. Optimization also requires `MODEL_API_KEY`.

```sh
python -m pip install -e ".[gpu]"
tunix doctor
tunix benchmark
tunix optimize --budget 8
tunix export --output tunix-results.zip
tunix report
```

Optimization stops when calibration fails. It does not silently relax quality gates or deploy a configuration. Arbitrary models and persistent runtime activation are not supported.

The separate [Meta advisory planner](docs/META_PLANNER.md) can propose an experiment from the recorded evidence:

```sh
python -m tunix.meta_planner --v2-benchmark results/benchmark_20261104T211417.json --allowed docs/meta-allowed-v2.json --output meta-proposal-new.json
```

Use a new output filename outside `results/`. A proposal is not a benchmark result or approved winner. Live Meta access was not verified in this cleanup environment.

## Repository layout

| Path | Purpose |
| --- | --- |
| `tunix/` | CLI, benchmarking, validation, planner, and workbench UI |
| `app.py` | Dashboard entry point and result presentation |
| `notebooks/` | Runnable Google Colab workflow |
| `tests/` | CPU regression tests; optional model-dependent tests |
| `assets/` | Tunix visual assets |
| `results/` | Original measured evidence, derived validation, and labeled samples |
| `docs/` | Setup, Colab, validation, and planner guides |
| `docs/archive/` | Historical development notes and retired agent tooling |

The application now lives on `main`. **Tunix** is the current product and Python package name; Optix and OptiLoop are historical names. Old `optiloop` imports must now be changed to `tunix`; the compatibility package has been removed. See the [migration guide](docs/MIGRATION.md). Original evidence and historical Git branches retain their provenance.

## Development

```sh
python -m pip install -e ".[test]"
python -m pytest tests -q
```

Optional model tests skip when their dependencies are unavailable. See [CONTRIBUTING.md](CONTRIBUTING.md) for the workflow. Installation is from source; this project has not been published to PyPI.
