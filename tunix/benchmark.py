"""Tunix benchmark harness: timing, NVML energy, truthfulness (T2/T3/T6).

GPU paths are WRITTEN, NOT RUN (no GPU in the dev sandbox). They run on a
Colab T4 via ``python -m tunix.run --gate1``; see COLAB.md. Nothing here
invents numbers: a failed profile is ``status="failed"`` with the real
error, and unmeasured metrics stay ``None`` (JSON null).

Result item schema (Interface 3, exact keys, do not extend without owner
approval — see DECISIONS.md)::

    {"profile_name", "status", "error", "requested_config", "actual_config",
     "repeats", "total_time_sec": {"median", "min", "max"},
     "latency_p50_sec", "latency_p95_sec", "docs_per_sec",
     "tokens_per_sec", "joules_per_doc", "idle_power_w",
     "compile_time_sec", "noisy"}

T2/T3 diagnostics with no approved JSON key (polling cross-check,
recompilations, break-even, static-cache outcome) are returned in the
in-memory ``console`` dict and printed; they are never written to JSON.

Integration (Subtask C): wrap ``run_config`` as the ``evaluate`` callable::

    outcome = run_config(config, docs, profile_name=slug, split="tuning")
    # outcome["benchmark"] -> one benchmark JSON item
    # outcome["outputs"]   -> outputs_<profile>.json bundle for validator

Top level imports only the standard library; torch/transformers/pynvml are
imported lazily so this module imports cleanly without GPU libraries.
"""
import json
import os
import random
import statistics
import threading
import time

from .profiles import validate_config
from .workload import MODEL_ID, SEED

WARMUP_DOCS = 2
NOISY_THRESHOLD = 0.10
ENERGY_CROSSCHECK_TOLERANCE = 0.15
POLL_INTERVAL_SEC = 0.5
IDLE_MEASURE_SEC = 10.0

_DTYPE_NAMES = {"torch.float32": "float32", "torch.float16": "float16",
                "float32": "float32", "float16": "float16"}


# --------------------------------------------------------------------------
# Pure helpers (CPU-testable, no GPU libraries)
# --------------------------------------------------------------------------

def summarize_times(times):
    """Median/min/max + noisy flag for repeat pass durations (T3)."""
    if not times:
        return {"median": None, "min": None, "max": None}, False
    median = statistics.median(times)
    noisy = bool(median and (max(times) - min(times)) / median
                 > NOISY_THRESHOLD)
    return {"median": median, "min": min(times), "max": max(times)}, noisy


def percentile(values, pct):
    """Linear-interpolation percentile; ``None`` for empty input."""
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (pct / 100.0) * (len(ordered) - 1)
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    frac = rank - low
    return ordered[low] + frac * (ordered[high] - ordered[low])


def break_even_docs(compile_time_sec, compiled_per_doc_sec,
                    baseline_per_doc_sec):
    """Docs for compile time to pay off; ``None`` if it never does (T3)."""
    if compile_time_sec is None or compiled_per_doc_sec is None \
            or baseline_per_doc_sec is None:
        return None
    saved = baseline_per_doc_sec - compiled_per_doc_sec
    if saved <= 0:
        return None
    return compile_time_sec / saved


def joules_per_doc(total_joules, idle_power_w, duration_sec, n_docs):
    """Idle-subtracted energy per doc; ``None`` if unmeasurable (T2)."""
    if total_joules is None or not n_docs or duration_sec is None:
        return None
    idle = (idle_power_w or 0.0) * duration_sec
    return (total_joules - idle) / n_docs


def count_output_tokens(row_ids, eos_id):
    """Generated length: tokens up to + including first EOS, else all."""
    count = 0
    for token in row_ids:
        count += 1
        if eos_id is not None and token == eos_id:
            break
    return count


def _result_item(profile_name, status, error, requested, actual, repeats):
    return {
        "profile_name": profile_name,
        "status": status,
        "error": error,
        "requested_config": requested,
        "actual_config": actual,
        "repeats": repeats,
        "total_time_sec": {"median": None, "min": None, "max": None},
        "latency_p50_sec": None,
        "latency_p95_sec": None,
        "docs_per_sec": None,
        "tokens_per_sec": None,
        "joules_per_doc": None,
        "idle_power_w": None,
        "compile_time_sec": None,
        "noisy": False,
    }


# --------------------------------------------------------------------------
# NVML energy (cumulative counter primary, power polling cross-check)
# --------------------------------------------------------------------------

class EnergyMeter:
    """NVML cumulative-energy meter with an integrated-polling estimate.

    Primary (T2): ``nvmlDeviceGetTotalEnergyConsumption`` (millijoules).
    Cross-check: sample ``nvmlDeviceGetPowerUsage`` every 0.5 s during the
    timed window and integrate. Unavailable NVML -> all readings ``None``.
    """

    def __init__(self):
        self._nvml = None
        self._handle = None
        self.available = False
        try:
            import pynvml
            pynvml.nvmlInit()
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            self._nvml = pynvml
            self.available = True
        except Exception:
            self.available = False

    def total_energy_joules(self):
        """Cumulative GPU energy in joules, or ``None``."""
        if not self.available:
            return None
        try:
            millijoules = self._nvml.nvmlDeviceGetTotalEnergyConsumption(
                self._handle)
            return millijoules / 1000.0
        except Exception:
            return None

    def power_watts(self):
        if not self.available:
            return None
        try:
            return self._nvml.nvmlDeviceGetPowerUsage(self._handle) / 1000.0
        except Exception:
            return None

    def measure_idle_power(self, seconds=IDLE_MEASURE_SEC):
        """Mean idle watts over ``seconds`` from the cumulative counter."""
        start = self.total_energy_joules()
        if start is None:
            return None
        time.sleep(seconds)
        end = self.total_energy_joules()
        if end is None:
            return None
        return (end - start) / seconds

    def start_polling(self):
        """Begin 0.5 s power polling; returns a stop callable -> joules."""
        samples = []
        stop = threading.Event()

        def _poll():
            while not stop.is_set():
                watts = self.power_watts()
                if watts is not None:
                    samples.append((time.perf_counter(), watts))
                stop.wait(POLL_INTERVAL_SEC)

        thread = threading.Thread(target=_poll, daemon=True)
        thread.start()

        def _stop():
            stop.set()
            thread.join(timeout=2.0)
            total = 0.0
            for (t0, w0), (t1, _w1) in zip(samples, samples[1:]):
                total += w0 * (t1 - t0)
            return total if samples else None

        return _stop

    def shutdown(self):
        if self.available:
            try:
                self._nvml.nvmlShutdown()
            except Exception:
                pass
            self.available = False


# --------------------------------------------------------------------------
# Model lifecycle + generation (GPU code: written, not run)
# --------------------------------------------------------------------------

def _sync(torch, device):
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()


def load_model(config, model_id=None, device=None):
    """Load tokenizer + model for ``config``; returns a bundle dict.

    Batching setup (T5): left padding for decoder-only generation, pad
    token falls back to the EOS token (Llama ships no pad token), and
    callers must pass the attention mask to ``generate``.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_id = model_id or os.environ.get("TUNIX_MODEL_ID", MODEL_ID)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = {"float32": torch.float32,
             "float16": torch.float16}[config["dtype"]]
    token = os.environ.get("HF_TOKEN") or None
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=token)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=dtype, attn_implementation=config["attn"],
        token=token)
    model.to(device)
    model.eval()
    return {"model": model, "tokenizer": tokenizer, "device": device,
            "model_id": model_id, "compiled": False,
            "compile_time_sec": None, "static_cache": "not_attempted"}


def maybe_compile(bundle, config, sample_prompts):
    """Apply ``torch.compile`` when requested; time the first call (T3).

    Tries a static cache first (HF generation under compile otherwise
    recompiles); falls back to the default cache on any error and records
    what actually happened in ``bundle["static_cache"]``. Returns nothing;
    mutates ``bundle``. Compile failures propagate (honest failure).
    """
    if not config["compile"]:
        return
    import torch

    model = bundle["model"]
    compiled = torch.compile(model)
    start = time.perf_counter()
    # First compiled call pays the compile cost; keep the graph warm.
    _generate_batch(bundle, compiled, sample_prompts[:1], config)
    bundle["compile_time_sec"] = time.perf_counter() - start
    bundle["model"] = compiled
    bundle["compiled"] = True
    try:
        from transformers import StaticCache
        cache = StaticCache(
            config=model.config, max_batch_size=config["batch"],
            max_cache_len=1024, device=bundle["device"],
            dtype=next(model.parameters()).dtype)
        _generate_batch(bundle, compiled, sample_prompts[:1], config,
                        past_key_values=cache)
        bundle["static_cache"] = "static"
    except Exception:
        bundle["static_cache"] = "dynamic_fallback"


def render_prompt(tokenizer, doc):
    """Format one workload doc via the official chat template (V2).

    Falls back to wrapping the article for docs built before V2 (no
    ``messages`` field). Same rendered prompt for every profile (R1).
    """
    messages = doc.get("messages")
    if messages is None:
        from .workload import build_messages
        messages = build_messages(doc["article"])
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)


def build_generate_kwargs(config, pad_token_id):
    """Extra ``model.generate`` kwargs, WITHOUT ``attention_mask``.

    The tokenizer batch already carries ``attention_mask`` inside
    ``inputs``; repeating it here makes ``generate(**inputs, **kwargs)``
    raise ``TypeError: got multiple values for keyword argument
    'attention_mask'``. The mask is supplied exactly once, via ``inputs``
    (left padding + pad token are set up in ``load_model``).
    """
    return {
        "max_new_tokens": config["max_new_tokens"],
        "do_sample": False,
        "pad_token_id": pad_token_id,
    }


def _generate_batch(bundle, model, prompts, config, past_key_values=None):
    """One greedy generation call; returns (texts, token_counts, max_batch).

    Used for every batch size (1, 4, 8) and for the compile warmup, so the
    single ``generate`` call below is the only generation path.
    """
    import torch

    tokenizer = bundle["tokenizer"]
    device = bundle["device"]
    # Prompts arrive chat-template rendered (they already carry BOS and
    # the generation prompt); do not add special tokens a second time.
    inputs = tokenizer(prompts, return_tensors="pt", padding=True,
                       add_special_tokens=False)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    prompt_len = inputs["input_ids"].shape[1]
    kwargs = build_generate_kwargs(config, tokenizer.pad_token_id)
    if past_key_values is not None:
        kwargs["past_key_values"] = past_key_values
    with torch.no_grad():
        output = model.generate(**inputs, **kwargs)
    rows = output[:, prompt_len:].tolist()
    eos_id = tokenizer.eos_token_id
    texts, counts = [], []
    for row in rows:
        n_tokens = count_output_tokens(row, eos_id)
        counts.append(n_tokens)
        texts.append(tokenizer.decode(row[:n_tokens],
                                       skip_special_tokens=True))
    return texts, counts, inputs["input_ids"].shape[0]


def _readback_actual(bundle, config, max_batch_seen):
    """Actual dtype/attn/batch/compiled read back at runtime (T6)."""
    model = bundle["model"]
    base = getattr(model, "_orig_mod", model)
    dtype = _DTYPE_NAMES.get(str(next(base.parameters()).dtype))
    attn = (getattr(base.config, "_attn_implementation", None)
            or getattr(base.config, "attn_implementation", None)
            or config["attn"])
    return {
        "dtype": dtype,
        "attn": attn,
        "batch": max_batch_seen,
        "compile": bundle["compiled"],
        "max_new_tokens": config["max_new_tokens"],
    }


def _count_recompilations():
    """Dynamo recompile count if counters are exposed, else ``None``."""
    try:
        from torch._dynamo.utils import counters
        frames = counters.get("frames", {})
        total = frames.get("total", None)
        return int(total) if total is not None else None
    except Exception:
        return None


def _batches(docs, batch_size):
    for start in range(0, len(docs), batch_size):
        yield docs[start:start + batch_size]


def _run_pass(bundle, docs, config, rendered=None):
    """One full timed pass; returns (records, elapsed, max_batch_seen).

    ``records`` maps doc_id -> (output_text, n_output_tokens). CUDA is
    synchronized around the whole pass (T3); per-doc latency uses the
    enclosing batch time divided by the batch size. ``rendered`` maps
    doc_id -> chat-template prompt (rendered once per run_config call).
    """
    import torch

    device = bundle["device"]
    records, latencies = {}, []
    max_batch_seen = 0
    _sync(torch, device)
    start = time.perf_counter()
    for chunk in _batches(docs, config["batch"]):
        if rendered is None:
            prompts = [render_prompt(bundle["tokenizer"], doc)
                       for doc in chunk]
        else:
            prompts = [rendered[doc["doc_id"]] for doc in chunk]
        batch_start = time.perf_counter()
        texts, counts, seen = _generate_batch(
            bundle, bundle["model"], prompts, config)
        _sync(torch, device)
        per_doc = (time.perf_counter() - batch_start) / len(chunk)
        latencies.extend([per_doc] * len(chunk))
        max_batch_seen = max(max_batch_seen, seen)
        for doc, text, n_tokens in zip(chunk, texts, counts):
            records[doc["doc_id"]] = (text, n_tokens)
    _sync(torch, device)
    elapsed = time.perf_counter() - start
    return records, elapsed, latencies, max_batch_seen


def run_config(config, docs=None, profile_name="custom", split="tuning",
               repeats=3, results_dir="results", seed=SEED, device=None,
               model_id=None, meter=None, idle_power_w="measure"):
    """Benchmark one config; returns {benchmark, outputs, console}.

    Loads the model once, runs ``WARMUP_DOCS`` untimed warmup docs, then
    ``repeats`` timed passes. Writes ``results/outputs_<profile>.json``
    (empty docs on failure, so stale successes never pass validation).
    Any exception becomes ``status="failed"`` with the real error (R4).
    """
    validate_config(config)
    requested = dict(config)
    item = _result_item(profile_name, "failed", None, requested, None,
                        repeats)
    outputs = {"profile_name": profile_name, "config": dict(requested),
               "docs": []}
    console = {"recompilations": None, "break_even_docs": None,
               "static_cache_used": None, "energy_poll_joules_per_doc": None,
               "energy_crosscheck_ok": None, "model_loaded": False}
    own_meter = meter is None
    meter = meter or EnergyMeter()
    bundle = None
    torch = None
    try:
        import torch
        if docs is None:
            from .workload import load_workload
            tuning, heldout = load_workload()
            docs = tuning if split == "tuning" else heldout
        if not docs:
            raise ValueError("no documents for split %r" % split)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        bundle = load_model(requested, model_id=model_id, device=device)
        # Weights are in memory from here on; a later failure is an
        # inference failure, not a load failure (report stays honest).
        console["model_loaded"] = True
        # V2: render every doc once through the official chat template;
        # the identical rendered prompts are used for all profiles (R1).
        rendered = {doc["doc_id"]: render_prompt(bundle["tokenizer"], doc)
                    for doc in docs}
        prompts = [rendered[doc["doc_id"]] for doc in docs]
        maybe_compile(bundle, requested, prompts)
        console["static_cache_used"] = bundle["static_cache"]
        console["recompilations"] = _count_recompilations()
        # Untimed warmup (excluded from every timed number, T3).
        for chunk in _batches(docs[:WARMUP_DOCS], requested["batch"]):
            _generate_batch(bundle, bundle["model"],
                            [rendered[d["doc_id"]] for d in chunk],
                            requested)
        if idle_power_w == "measure":
            idle_power_w = meter.measure_idle_power()
        item["idle_power_w"] = idle_power_w
        energy_start = meter.total_energy_joules()
        stop_polling = meter.start_polling()
        pass_times, latencies, last_records = [], [], {}
        max_batch_seen = 0
        try:
            for _ in range(repeats):
                records, elapsed, pass_lat, seen = _run_pass(
                    bundle, docs, requested, rendered)
                pass_times.append(elapsed)
                latencies.extend(pass_lat)
                last_records = records
                max_batch_seen = max(max_batch_seen, seen)
        finally:
            poll_joules = stop_polling()
        energy_end = meter.total_energy_joules()
        duration = sum(pass_times)
        n_timed = len(docs) * repeats
        total_tokens = sum(n for _t, n in last_records.values())
        actual = _readback_actual(bundle, requested, max_batch_seen)
        item["actual_config"] = actual
        if actual != requested:
            raise RuntimeError(
                "profile truthfulness mismatch (T6): requested %s, "
                "actual %s" % (requested, actual))
        summary, noisy = summarize_times(pass_times)
        item["total_time_sec"] = summary
        item["noisy"] = noisy
        median = summary["median"]
        item["latency_p50_sec"] = percentile(latencies, 50)
        item["latency_p95_sec"] = percentile(latencies, 95)
        item["docs_per_sec"] = (len(docs) / median) if median else None
        item["tokens_per_sec"] = (total_tokens / median) if median else None
        counter_joules = None
        if energy_start is not None and energy_end is not None:
            counter_joules = joules_per_doc(
                energy_end - energy_start, idle_power_w, duration, n_timed)
        item["joules_per_doc"] = counter_joules
        poll_per_doc = joules_per_doc(poll_joules, idle_power_w, duration,
                                      n_timed)
        console["energy_poll_joules_per_doc"] = poll_per_doc
        if counter_joules and poll_per_doc is not None:
            console["energy_crosscheck_ok"] = bool(
                abs(counter_joules - poll_per_doc) / counter_joules
                <= ENERGY_CROSSCHECK_TOLERANCE)
        item["compile_time_sec"] = bundle["compile_time_sec"]
        item["status"] = "success"
        outputs["docs"] = [{
            "doc_id": doc["doc_id"],
            "split": split,
            "output_text": last_records[doc["doc_id"]][0],
            "n_output_tokens": last_records[doc["doc_id"]][1],
            "gold_summary": doc["gold_summary"],
        } for doc in docs]
    except Exception as exc:
        item["status"] = "failed"
        item["error"] = "%s: %s" % (type(exc).__name__, exc)
        outputs["docs"] = []
    finally:
        if bundle is not None:
            try:
                del bundle["model"]
                if torch is not None and torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
        if own_meter:
            meter.shutdown()
    _write_outputs(results_dir, outputs)
    _print_console(profile_name, item, console)
    return {"benchmark": item, "outputs": outputs, "console": console}


def _write_outputs(results_dir, outputs):
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir,
                        "outputs_%s.json" % outputs["profile_name"])
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(outputs, handle, indent=2)


def _print_console(profile_name, item, console):
    """T2/T3 diagnostics that have no approved JSON key (DECISIONS.md)."""
    print("[%s] status=%s docs/s=%s J/doc(counter)=%s J/doc(poll)=%s "
          "crosscheck_ok=%s idle_W=%s compile_s=%s recompilations=%s "
          "static_cache=%s break_even_docs=%s noisy=%s" % (
              profile_name, item["status"], item["docs_per_sec"],
              item["joules_per_doc"],
              console["energy_poll_joules_per_doc"],
              console["energy_crosscheck_ok"], item["idle_power_w"],
              item["compile_time_sec"], console["recompilations"],
              console["static_cache_used"], console["break_even_docs"],
              item["noisy"]))
    if console["energy_crosscheck_ok"] is False:
        print("[%s] WARNING T2: NVML counter vs polling differ by >15%%; "
              "report both." % profile_name)


def run_profiles(named_configs, docs, split="tuning", repeats=3,
                 results_dir="results", seed=SEED):
    """Run ``(name, config)`` pairs in seeded shuffled order (T3).

    Shares one idle-power measurement across profiles. Returns
    ``(items, outputs_by_name, consoles_by_name)``; per-profile failures
    are captured as failed items, never raised.
    """
    order = list(named_configs)
    random.Random(seed).shuffle(order)
    meter = EnergyMeter()
    idle_power_w = meter.measure_idle_power()
    items, outputs_by_name, consoles = [], {}, {}
    baselines_per_doc = {}
    try:
        for name, config in order:
            outcome = run_config(
                config, docs, profile_name=name, split=split,
                repeats=repeats, results_dir=results_dir, seed=seed,
                meter=meter, idle_power_w=idle_power_w)
            items.append(outcome["benchmark"])
            outputs_by_name[name] = outcome["outputs"]
            consoles[name] = outcome["console"]
            item = outcome["benchmark"]
            median = item["total_time_sec"]["median"]
            if item["status"] == "success" and median:
                baselines_per_doc[name] = median / len(docs)
    finally:
        meter.shutdown()
    # Break-even needs an uncompiled twin; fill it in where both ran (T3).
    by_name = {item["profile_name"]: item for item in items}
    for name, config in named_configs:
        if not config["compile"] or name not in baselines_per_doc:
            continue
        twin = dict(config, compile=False)
        for other_name, other_config in named_configs:
            if other_config == twin and other_name in baselines_per_doc:
                consoles[name]["break_even_docs"] = break_even_docs(
                    by_name[name]["compile_time_sec"],
                    baselines_per_doc[name],
                    baselines_per_doc[other_name])
    items.sort(key=lambda it: [n for n, _c in named_configs].index(
        it["profile_name"]))
    return items, outputs_by_name, consoles


def save_benchmark(items, results_dir="results"):
    """Write ``results/benchmark_<ts>.json``; returns the path."""
    os.makedirs(results_dir, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%S")
    path = os.path.join(results_dir, "benchmark_%s.json" % stamp)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(items, handle, indent=2)
    return path
