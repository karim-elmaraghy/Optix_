# Meta Experiment Planner (advisory)

Protocol checked against Meta's official documentation on November 4, 2026:
https://dev.meta.ai/docs/cookbook/quickstart-chat-completions and
https://dev.meta.ai/docs/protocols/chat-completions.
The documented model is `muse-spark-1.3`, using Bearer authentication at
`https://api.meta.ai/v1/chat/completions`, a model/messages request, and
`choices[0].message.content`. Responses must finish with `stop` and contain
nonempty text. The transport uses `max_completion_tokens` and default temperature.

Set MODEL_API_KEY securely in the process environment. Never put it in arguments
or evidence files. META_MODEL optionally overrides the requested model.

```powershell
.venv/Scripts/python.exe -m tunix.meta_planner --ping --output meta-ping.json
.venv/Scripts/python.exe -m tunix.meta_planner --history actual-v2-tuning-history.json --allowed permitted-configurations.json --output meta-plan.json
```

History must be a JSON list of actual benchmark rows (or optimizer history records
with result.benchmark). Allowed configurations must be a JSON list using the
existing agent contract: dtype, attn, batch, compile, max_new_tokens. Supply tuning
evidence only. The caller must establish the measurements' authenticity; this
command does not certify evidence as authentic V2 GPU results. No example fixture
is shipped as benchmark evidence.

The planner submits the supplied history and allowed configurations, excludes
previously requested configurations, and requires an exact permitted proposal and
nonempty hypothesis. Output includes submitted messages, evidence file hashes,
raw returned proposal text, validated proposal, requested/returned model IDs,
response ID, HTTP status and UTC timestamp. It never overwrites an existing file.
Protect evidence files from sensitive content before submission.

Successful live calls carry source=meta_api. Failures have source=null and no
proposal; there is no rule fallback in this demonstration. The existing optimizer
retains its explicit rule_fallback behavior. A ping proves connectivity only.
An advisory proposal proves neither benchmark execution nor quality approval nor
deployment. No optimizer, validator, or safe-apply gate is changed.

For the authentic V2 archive, use the read-only adapter:

```powershell
.venv/Scripts/python.exe -m tunix.meta_planner --v2-benchmark "C:\path\to\Tunix\results\benchmark_20261104T211417.json" --allowed docs/meta-allowed-v2.json --output meta-v2-proposal.json
```

Run from the feature checkout with MODEL_API_KEY already set securely in the
environment. This makes one real request when the key is present. Use a fresh
output name for each run. A missing key writes an unverified record without a
network call. The output must be outside the original artifact directory.

The adapter reads benchmark, session, validation report and matching output
bundles. It checks the report's artifact hashes, V2 metadata, profile/config
identity and tuning splits. These checks establish consistency of the supplied
archive, not independent certification of GPU execution. It submits the original
benchmark rows and the session/validation context, including blocked calibration,
with provenance hashes of all read files. It does not submit document text.

`meta-allowed-v2.json` is an explicit 24-configuration snapshot of the integration
checkout's `tunix/profiles.py` SPACE: dtype x attention x batch x token cap,
compile=false. It is configuration data, not benchmark data. Review the permitted
space if that contract changes. The examined session is measured V2, imported
from user-supplied Colab evidence, calibration blocked by truncation, winner=null.
The planner remains usable for advice in that state; execution remains blocked
by the optimizer's existing gate. No workbench UI changes are included.
