# OptiLoop — validation tests (run in order; a failed gate stops the line)

Paste this into the Optix_ orchestrator as an addendum to Task Package #001. It replaces the weaker parts of the original acceptance criteria.

## Global rules (every agent)
R1. Greedy decoding only (`do_sample=False`), fixed seeds, identical prompts and `max_new_tokens` across profiles (except the deliberate token-cap profile).
R2. **Two baselines, always reported:** `naive` = fp32 + eager + batch 1, and `competent_default` = fp16 + SDPA + batch 1. The headline claim is measured against `competent_default`. The speedup vs `naive` is reported separately and labeled as such.
R3. Real documents, not synthetic random text: 20 tuning docs + 10 held-out docs from a public news/summarization dataset (CNN/DailyMail or XSum), fixed indices, truncated to 150-250 words. Hold-out docs are never used for tuning.
R4. Every number shown anywhere (dashboard, README, video) comes from a committed results JSON. No hard-coded or estimated numbers. A failed profile is `status:"failed"` with the real error.
R5. The agent that proposes experiments must be a real model call (Muse via the Meta API, `https://api.meta.ai/v1`, key in `MODEL_API_KEY`). A rule-based order may exist only as a labeled fallback and never as the headline "agent."

## Tests

**T0 Environment.** Pass if: GPU name printed from `nvidia-smi`, torch/transformers versions logged, the Llama model loads (HF license accepted, token works), and a one-token Muse API call succeeds.

**T1 Determinism.** Run `competent_default` twice. Pass if >= 99% of documents produce identical output tokens. If not, the quality metric is measuring noise: fix seeds/greedy before continuing.

**T2 Energy meter sanity.** Use the NVML cumulative energy counter (`nvmlDeviceGetTotalEnergyConsumption`, already confirmed working on the T4), not 0.5 s `nvidia-smi` polling. Measure 10 s of idle power first. Report `J/doc = (E_total - P_idle * duration) / n_docs`. Pass if it agrees with integrated power polling within 15%; otherwise report both and flag it.

**T3 Timing validity.** Exclude warmup (2 docs). Synchronize CUDA around timing. At least 3 repeats per profile in shuffled order; report median and min/max; flag any profile where (max-min)/median > 10% as noisy. For `torch.compile`: report compile time separately, the number of recompilations, and the **break-even document count** (compile time divided by per-document time saved). Note that HF generation under compile usually needs a static cache to avoid constant recompiling; test it with and without, and report what actually happened.

**T4 Quality validator (unit tests + calibration).**
- Unit tests: identical output scores 1.0; empty/short output (<10 tokens) is rejected; garbage or shuffled text is rejected; a 64-token truncation is flagged with reason "truncation".
- Metrics: token-overlap F1 vs the `competent_default` output **and** ROUGE-L vs the dataset's gold summary.
- Threshold calibration: measure the fp32-vs-fp16 quality difference. A profile is acceptable if its drop is no larger than that difference plus a stated margin. Do not use an arbitrary 0.85.

**T5 Batching correctness.** Decoder-only batching needs left padding, a pad token (Llama has none by default), and an attention mask. Pass if batch-4 outputs match batch-1 outputs per document (F1 >= 0.95). If not, treat it as a bug to fix, not as a "quality rejection."

**T6 Profile truthfulness.** At runtime, read back and log the actual dtype, attention implementation, batch size, and whether the model was compiled. Fail the profile if any differs from what was requested (silent fallbacks are not allowed).

**T7 Headroom gate (Gate 1).** Run all profiles on the tuning docs. Pass if the best valid profile beats `competent_default` by >= 20% in docs/s or J/doc at acceptable quality. If it only beats `naive`, say so and decide with the owner whether to continue; do not call that a win.

**T8 Held-out confirmation.** Re-run the winner and both baselines on the 10 held-out docs, 3 repeats. Pass if the speedup is within +/-10% of the tuning speedup and quality is above threshold. Otherwise downgrade the claim to the held-out number.

**T9 Agent ablation.** Define a search space of at least 24 configs (dtype x attention x batch size x compile/static cache x max_new_tokens, optionally model size). Run the full grid once and cache the results table. Then compare, on that table, the Muse agent (budget 8 experiments) vs random search (>= 20 random orderings, median) vs the fixed rule order. Metric: best-valid-profile-so-far vs number of experiments. Claim "the agent helps" only if it beats the random median; otherwise report honestly that it did not.

**T10 Safe apply + rollback.** Applying a winning profile re-checks quality on held-out docs and rolls back to the default if the check fails. Prove it with a deliberately bad profile (e.g., `max_new_tokens=8`) that must trigger the rollback.

**T11 Reproducibility.** On a fresh Colab runtime, `python -m optiloop.run --demo` reproduces the headline table within +/-10%. Commit the results JSON, and record GPU name, torch, transformers, and driver versions in the README.

## Claim template (fill only with measured values)
"On a Tesla T4 with Llama-3.2-1B-Instruct, OptiLoop's agent found a profile X times faster than a competent fp16/SDPA default (Y times faster than naive fp32) at quality >= Z, using N experiments vs M for random search; energy per document fell W% (NVML counter). Confirmed on held-out documents."
