# Tunix on Google Colab

Open [notebooks/tunix_colab.ipynb](../notebooks/tunix_colab.ipynb) in Colab, select a T4 GPU runtime, and run the cells in order. The notebook installs Tunix from `main`, checks the runtime, securely requests a Hugging Face token, benchmarks with three repeats, and exports a uniquely named results ZIP.

Accept the `meta-llama/Llama-3.2-1B-Instruct` license first. Benchmarking needs `HF_TOKEN`; it does not require a Meta key. Optional optimization is disabled by default and requests `MODEL_API_KEY` only when explicitly enabled.

The supported workload is CNN/DailyMail summarization with V2 chat-template prompts. Do not compare V1 and V2 baselines. Current committed measurements are real historical evidence, but calibration remains blocked by truncated baselines; they are not an approved winner. See [V2 validation](V2_VALIDATION.md).

Fresh runs write separate sessions and do not overwrite the committed original evidence. Failed profiles and blocked calibration remain part of the record. A zero benchmark exit code is not quality approval or deployment.

See [Colab export and workbench import](COLAB_TRANSFER.md) for the detailed workflow, and [migration](MIGRATION.md) for updating older notebook cells. Private Colab/Drive copies must be updated separately.
