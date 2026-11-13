# Migrating older notebooks to Tunix

The compatibility package has been removed. Use the [current Colab notebook](../notebooks/tunix_colab.ipynb) for new runs.

| Previous usage | Current usage |
| --- | --- |
| `from optiloop.benchmark import run_config` | `from tunix.benchmark import run_config` |
| `import optiloop.validator` | `import tunix.validator` |
| `python -m optiloop.run` | `python -m tunix.run` |
| `python -m optiloop.meta_planner` | `python -m tunix.meta_planner` |
| `OPTILOOP_MODEL_ID` (low-level benchmark override) | `TUNIX_MODEL_ID` |
| Old Optix clone URLs / integration branches | `https://github.com/karim-elmaraghy/Tunix.git`, branch `main` |

Update saved notebook code cells and rerun `python -m pip install -e .` from the updated checkout. Restart the Python kernel to clear old imported modules. The supported high-level CLI model remains Llama 3.2 1B Instruct on T4; renaming the low-level override does not add model support.

Do not rewrite measured output files or rename original evidence merely to remove historical names. Point export commands at the actual location of an old run. Existing result archives remain supported.

Notebooks saved privately in Colab or Drive are separate copies: this repository update cannot modify them. Replace their code using the mappings above, or open the committed notebook for a fresh run.
