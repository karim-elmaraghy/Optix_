"""Tunix CLI runner (Subtask A). GPU paths: written, not run.

Usage (Colab T4, after ``pip install -r requirements.txt``)::

    python -m tunix.run --gate1 [--with-compile] [--repeats 2]
    python -m tunix.run --grid
    python -m tunix.run --heldout fp16_sdpa_b4
    python -m tunix.run --demo
    python -m tunix.run --determinism

Every number printed comes from a measured run in this process or from a
committed results JSON (R4). Failed profiles print as FAILED with the real
error. See COLAB.md for the copy-paste Gate 1 flow.
"""
import argparse
import re
import subprocess
import sys
from collections import Counter

from .benchmark import run_config, run_profiles, save_benchmark
from .profiles import (COMPILE_PROFILE, DEFAULT_PROFILE, GATE1_PROFILES,
                       NAIVE_PROFILE, PROFILES, SPACE, config_slug,
                       profile_config)
from .workload import PROMPT_VERSION, load_workload

BATCH_F1_THRESHOLD = 0.95


def _tokens(text):
    return re.findall(r"\w+", (text or "").lower(), re.UNICODE)


def token_f1(text_a, text_b):
    """Token-overlap F1 (same tokenization as the quality validator)."""
    a, b = _tokens(text_a), _tokens(text_b)
    if not a or not b:
        return 0.0
    overlap = sum((Counter(a) & Counter(b)).values())
    return 2.0 * overlap / (len(a) + len(b))


def print_env_info():
    """T0: GPU name (nvidia-smi first), library versions, CUDA device."""
    gpu_name = None
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30)
        if out.returncode == 0 and out.stdout.strip():
            gpu_name = out.stdout.strip().splitlines()[0]
    except Exception as exc:
        print("nvidia-smi unavailable: %s: %s" % (type(exc).__name__, exc))
    print("GPU (nvidia-smi): %s" % (gpu_name or "unavailable"))
    try:
        import torch
        import transformers
        print("torch: %s" % torch.__version__)
        print("transformers: %s" % transformers.__version__)
        print("cuda available: %s" % torch.cuda.is_available())
        if torch.cuda.is_available():
            print("GPU (torch): %s" % torch.cuda.get_device_name(0))
    except Exception as exc:
        print("torch/transformers unavailable: %s: %s"
              % (type(exc).__name__, exc))


def _fmt(value, digits=3):
    return "n/a" if value is None else ("%.*f" % (digits, value))


def print_table(items, consoles=None):
    """Headline table: profile, docs/s, J/doc, speedups vs both baselines.

    ``consoles`` maps profile name -> benchmark console dict; its
    ``model_loaded`` flag distinguishes "weights never loaded" from
    "weights loaded, generation failed" (the two must not be conflated).
    """
    by_name = {item["profile_name"]: item for item in items}
    default = by_name.get(DEFAULT_PROFILE, {})
    naive = by_name.get(NAIVE_PROFILE, {})

    def speedup(item, baseline):
        mine = item.get("docs_per_sec")
        base = baseline.get("docs_per_sec")
        if not mine or not base:
            return None
        return mine / base

    header = ("%-24s %10s %10s %14s %14s %8s"
              % ("profile", "docs/s", "J/doc", "vs default x",
                 "vs naive x", "noisy"))
    print(header)
    print("-" * len(header))
    for item in items:
        if item["status"] != "success":
            print("%-24s FAILED: %s"
                  % (item["profile_name"], item["error"]))
            continue
        print("%-24s %10s %10s %14s %14s %8s" % (
            item["profile_name"], _fmt(item["docs_per_sec"], 2),
            _fmt(item["joules_per_doc"], 2),
            _fmt(speedup(item, default), 2),
            _fmt(speedup(item, naive), 2), item["noisy"]))
    loaded = [i["profile_name"] for i in items if i["status"] == "success"]
    weights_loaded = bool(consoles) and any(
        c.get("model_loaded") for c in consoles.values())
    if loaded:
        print("model load: confirmed (%s)" % ", ".join(loaded))
    elif weights_loaded:
        print("model load: confirmed (weights loaded); generation failed "
              "for every profile (see errors above)")
    else:
        print("model load: FAILED for every profile (see errors above)")


def check_batching(outputs_by_name,
                   batch_profiles=("fp16_sdpa_b4", "fp16_sdpa_b8")):
    """T5: batched outputs must match batch-1 per doc (F1 >= 0.95).

    Checks every available batched profile (batch 4 and batch 8) against
    the batch-1 competent default. Returns True when the check passes or
    cannot run (missing outputs are already FAILED rows). A mismatch is
    a batching bug, not a quality rejection: callers fail loudly.
    """
    if DEFAULT_PROFILE not in outputs_by_name:
        return True
    single = {d["doc_id"]: d["output_text"]
              for d in outputs_by_name[DEFAULT_PROFILE]["docs"]}
    if not single:
        return True
    ok = True
    for batch_profile in batch_profiles:
        if batch_profile not in outputs_by_name:
            continue
        batched = {d["doc_id"]: d["output_text"]
                   for d in outputs_by_name[batch_profile]["docs"]}
        if not batched:
            continue
        scores = {doc_id: token_f1(batched[doc_id], single[doc_id])
                  for doc_id in single if doc_id in batched}
        worst_id = min(scores, key=scores.get)
        mean_f1 = sum(scores.values()) / len(scores)
        print("T5 batching: %s vs %s mean F1=%.4f min F1=%.4f (worst %s)"
              % (batch_profile, DEFAULT_PROFILE, mean_f1,
                 scores[worst_id], worst_id))
        if scores[worst_id] < BATCH_F1_THRESHOLD:
            print("T5 FAIL: %s output for %s has F1 %.4f < %.2f vs "
                  "batch-1. This is a batching bug (padding/masking), "
                  "not a quality rejection." % (batch_profile, worst_id,
                                                 scores[worst_id],
                                                 BATCH_F1_THRESHOLD),
                  file=sys.stderr)
            ok = False
    return ok


def run_gate1(args, with_compile=False, demo=False):
    """Gate 1 (T7 input): baselines + batch/cap profiles on tuning docs."""
    print_env_info()
    print("prompt format: %s (never compare against V1 baselines)"
          % PROMPT_VERSION)
    tuning, _heldout = load_workload()
    print("tuning docs: %d (150-250 words, cnn_dailymail test)"
          % len(tuning))
    names = list(GATE1_PROFILES)
    if with_compile:
        names.append(COMPILE_PROFILE)
    named = [(name, profile_config(name)) for name in names]
    items, outputs_by_name, consoles = run_profiles(
        named, tuning, split="tuning", repeats=args.repeats,
        results_dir=args.results_dir)
    path = save_benchmark(items, args.results_dir)
    print_table(items, consoles)
    print("saved: %s" % path)
    if demo:
        _print_headline(items)
    if not check_batching(outputs_by_name):
        return 2
    return 0


def _print_headline(items):
    """Measured-only headline summary (no quality claim without T4)."""
    ok = [i for i in items if i["status"] == "success" and i["docs_per_sec"]]
    by_name = {i["profile_name"]: i for i in ok}
    if DEFAULT_PROFILE not in by_name or not ok:
        print("headline: unavailable (competent default did not succeed)")
        return
    best = max(ok, key=lambda i: i["docs_per_sec"])
    default = by_name[DEFAULT_PROFILE]
    naive = by_name.get(NAIVE_PROFILE)
    print("headline (tuning, speed only; quality gate is Subtask C): "
          "best=%s %.3f docs/s = %.2fx vs competent default (%.3f docs/s)"
          % (best["profile_name"], best["docs_per_sec"],
             best["docs_per_sec"] / default["docs_per_sec"],
             default["docs_per_sec"]))
    if naive and naive["docs_per_sec"]:
        print("speedup vs naive (labeled separately, R2): %.2fx"
              % (best["docs_per_sec"] / naive["docs_per_sec"]))


def run_grid(args):
    """Full 24-config SPACE sweep on tuning docs; one cached table."""
    print("prompt format: %s" % PROMPT_VERSION)
    tuning, _heldout = load_workload()
    named = [(config_slug(config), config) for config in SPACE]
    print("grid: %d configs x %d tuning docs, repeats=%d"
          % (len(named), len(tuning), args.repeats))
    items, _outputs, consoles = run_profiles(
        named, tuning, split="tuning", repeats=args.repeats,
        results_dir=args.results_dir)
    path = save_benchmark(items, args.results_dir)
    succeeded = sum(1 for i in items if i["status"] == "success")
    print("grid done: %d/%d succeeded; cached table: %s"
          % (succeeded, len(items), path))
    print_table(items, consoles)
    return 0


def run_heldout(args):
    """Winner + both baselines on the 10 held-out docs (T8)."""
    _tuning, heldout = load_workload()
    names = []
    for name in (args.heldout, DEFAULT_PROFILE, NAIVE_PROFILE):
        if name not in names:
            names.append(name)
    named = [(name, profile_config(name)) for name in names]
    print("heldout: %s on %d docs, repeats=%d"
          % (names, len(heldout), args.repeats))
    items, _outputs, consoles = run_profiles(
        named, heldout, split="heldout", repeats=args.repeats,
        results_dir=args.results_dir)
    path = save_benchmark(items, args.results_dir)
    print_table(items, consoles)
    print("saved: %s" % path)
    return 0


def run_determinism(args):
    """T1: run the competent default twice; identical-output fraction."""
    tuning, _heldout = load_workload()
    config = profile_config(DEFAULT_PROFILE)
    first = run_config(config, tuning, profile_name=DEFAULT_PROFILE,
                       repeats=1, results_dir=args.results_dir)
    second = run_config(config, tuning, profile_name=DEFAULT_PROFILE,
                        repeats=1, results_dir=args.results_dir)
    a = {d["doc_id"]: d["output_text"] for d in first["outputs"]["docs"]}
    b = {d["doc_id"]: d["output_text"] for d in second["outputs"]["docs"]}
    if not a or a.keys() != b.keys():
        print("T1 determinism: FAILED (missing outputs: %s / %s)"
              % (first["benchmark"]["error"],
                 second["benchmark"]["error"]))
        return 2
    identical = sum(1 for key in a if a[key] == b[key])
    fraction = identical / len(a)
    print("T1 determinism: %d/%d identical (%.1f%%); pass >= 99%%: %s"
          % (identical, len(a), 100.0 * fraction, fraction >= 0.99))
    return 0 if fraction >= 0.99 else 2


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="tunix.run",
        description="Tunix benchmark runner (GPU paths: written, "
                    "not run until Colab Gate 1).")
    parser.add_argument("--gate1", action="store_true",
                        help="Gate 1: baselines + batch/cap on tuning docs")
    parser.add_argument("--with-compile", action="store_true",
                        help="with --gate1/--demo: add the compile profile")
    parser.add_argument("--grid", action="store_true",
                        help="run all 24 SPACE configs, cache results table")
    parser.add_argument("--heldout", metavar="PROFILE",
                        help="re-run PROFILE + both baselines on held-out")
    parser.add_argument("--demo", action="store_true",
                        help="reproduce the headline table (T11, fresh run)")
    parser.add_argument("--determinism", action="store_true",
                        help="T1: default twice, identical-output fraction")
    parser.add_argument("--repeats", type=int, default=2,
                        help="timed repeats per profile (default 2)")
    parser.add_argument("--results-dir", default="results",
                        help="where results JSON is written")
    args = parser.parse_args(argv)
    # Known profiles for --heldout validation (fail before GPU work).
    if args.heldout and args.heldout not in PROFILES:
        parser.error("unknown profile %r; known: %s"
                     % (args.heldout, sorted(PROFILES)))
    if args.gate1:
        return run_gate1(args, with_compile=args.with_compile)
    if args.grid:
        return run_grid(args)
    if args.heldout:
        return run_heldout(args)
    if args.demo:
        return run_gate1(args, with_compile=args.with_compile, demo=True)
    if args.determinism:
        return run_determinism(args)
    parser.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
