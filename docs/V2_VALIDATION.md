# Actual V2 artifact validation

Source: Karim's `optiloop_v2_measured_results.zip`, six original JSON files
copied byte-for-byte into `results/`. Prompt version `v2-chat-template` /
Muse commit `6ac0558` is supplied by Karim; the unchanged benchmark schema
does not itself contain prompt-version metadata. Original SHA-256 hashes
are recorded in `results/validation_report.json`. No GPU run was performed
by Codex; these are offline checks of actual saved Colab measurements.

| Profile | Actual docs/s | Actual J/doc | Truncation flags | T5 |
|---|---:|---:|---:|---|
| FP32 B1 | 0.316154 | 118.279261 | 4/20 | not applicable |
| FP16 default B1 | 0.352181 | 108.274149 | 5/20 | reference |
| FP16 B4 | 1.286347 | 29.432868 | 5/20 | PASS, minimum F1 1.000 |
| FP16 B8 | 1.982246 | 19.666552 | 5/20 | PASS, minimum F1 1.000 |
| FP16 cap64 | 0.620499 | 61.177673 | 20/20 | not applicable |

B8 raw speedup vs competent default: **5.62848x (displayed 5.63x)**.
Energy reduction: **81.84%**. These are raw measured comparisons, not an
approved winner or end-to-end Meta agent result.

## Exact validator findings

- `calibrate(default, naive, margin=0.02)` raises `CalibrationBlocked`.
- Default truncation doc IDs: `cnn-test-1`, `cnn-test-2`, `cnn-test-12`,
  `cnn-test-15`, `cnn-test-17`.
- FP32 truncation doc IDs: `cnn-test-2`, `cnn-test-12`, `cnn-test-15`,
  `cnn-test-17`.
- B4 and B8 have the same flagged IDs as default, and token-overlap F1=1
  for every document vs default. Matching a truncated reference passes
  T5 but cannot establish summary completeness.
- Cap64 flags every document. No empty/short/garbage/invalid-text reason
  was triggered by the existing output checks on any supplied profile.
- Default `cnn-test-1` ends with `Good Samar`; `cnn-test-12` ends with
  `where they faced`; `cnn-test-15` ends with `bellwether for shifting`.
  This is visible incompleteness, beyond a mere theoretical cap warning.
  Original generated token IDs / finish reasons are not in the shared schema.
- Descriptive gold ROUGE-L means: default/B4/B8 0.180473; FP32 0.177388;
  cap64 0.208699. These scores do not override truncation or constitute a
  calibrated pass. Candidate calibrated-drop decisions cannot proceed
  without valid baseline calibration; no placeholder tolerance was invented.
- All benchmark statuses are success and actual configs equal requested
  configs. Noisy flags are false. **Every profile has only two repeats**;
  T3 requires at least three. Held-out, T1 rerun and T2 polling-crosscheck
  console evidence were not included in the archive.

## Working demo and remaining blockers

`python app.py` loads the original measurements, raw 5.63x, derived quality
scores, explicit truncation rejections, T5 passes and blocked calibration.
The derived report is SHA-256-bound to the original files; it is not a Meta
proposal log. There were no agent logs in the archive; none were fabricated.
`results/session.json` is an imported-evidence status sidecar, not a claim
that a new GPU or Meta optimization run completed locally.

Optimize must stop at calibration before any Meta search; Apply remains
unavailable without a persistent inference runtime. The local environment
has no MODEL_API_KEY, so live Meta access is still untested. Held-out approval
and GPU rollback have not passed.

Shortest honest submission path: launch this measured-results dashboard,
show raw throughput/energy and T5 passing, show why all candidates remain
unapproved, and present no winner/agent-superiority/rollback claim. A fully
approved optimization demo additionally needs owner-approved resolution of
actual incomplete baselines, a same-protocol rerun with >=3 repeats, valid
calibration, live Meta search and disjoint held-out confirmation. No quality
threshold or benchmark methodology was changed during artifact validation.
