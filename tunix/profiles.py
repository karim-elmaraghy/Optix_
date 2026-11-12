"""Tunix profile definitions: named baselines + 24-config search space.

Shared config schema (Interface, do not alter):
``{"dtype": "float32|float16", "attn": "eager|sdpa", "batch": int,
"compile": bool, "max_new_tokens": int}``

Two baselines are always reported (OPTILOOP_TESTS.md R2):
``naive_fp32_eager_b1`` (fp32 + eager + batch 1) and
``default_fp16_sdpa_b1`` (fp16 + SDPA + batch 1, the competent default).
Headline speedups are measured against the competent default; the speedup
vs naive is reported separately and labeled as such.

Greedy decoding only (``do_sample=False``), fixed seed, identical prompts
and ``max_new_tokens`` across profiles except the deliberate cap profile
``fp16_sdpa_b1_cap64``. Pure data module: no GPU-library imports.
"""
import itertools

MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
SEED = 42
DEFAULT_MAX_NEW_TOKENS = 128

DTYPES = ("float32", "float16")
ATTNS = ("eager", "sdpa")
BATCHES = (1, 4, 8)
TOKEN_CAPS = (128, 64)

_CONFIG_KEYS = {"dtype", "attn", "batch", "compile", "max_new_tokens"}


def make_config(dtype, attn, batch, compile=False,
                max_new_tokens=DEFAULT_MAX_NEW_TOKENS):
    """Build one validated config dict in the shared schema."""
    config = {
        "dtype": dtype,
        "attn": attn,
        "batch": batch,
        "compile": compile,
        "max_new_tokens": max_new_tokens,
    }
    validate_config(config)
    return config


def validate_config(config):
    """Raise ``ValueError`` unless ``config`` matches the shared schema."""
    if not isinstance(config, dict) or set(config) != _CONFIG_KEYS:
        raise ValueError("config keys must be %s" % sorted(_CONFIG_KEYS))
    if config["dtype"] not in DTYPES:
        raise ValueError("dtype must be one of %s" % (DTYPES,))
    if config["attn"] not in ATTNS:
        raise ValueError("attn must be one of %s" % (ATTNS,))
    if type(config["batch"]) is not int or config["batch"] not in BATCHES:
        raise ValueError("batch must be one of %s" % (BATCHES,))
    if type(config["compile"]) is not bool:
        raise ValueError("compile must be a bool")
    if type(config["max_new_tokens"]) is not int \
            or not 1 <= config["max_new_tokens"] <= 4096:
        raise ValueError("max_new_tokens must be an int in [1, 4096]")
    return config


# Named profiles. Insertion order is the canonical display order.
PROFILES = {
    "naive_fp32_eager_b1": make_config("float32", "eager", 1),
    "default_fp16_sdpa_b1": make_config("float16", "sdpa", 1),
    "fp16_sdpa_b4": make_config("float16", "sdpa", 4),
    "fp16_sdpa_b8": make_config("float16", "sdpa", 8),
    "fp16_sdpa_b4_compile": make_config("float16", "sdpa", 4, compile=True),
    "fp16_sdpa_b1_cap64": make_config(
        "float16", "sdpa", 1, max_new_tokens=64),
}

NAIVE_PROFILE = "naive_fp32_eager_b1"
DEFAULT_PROFILE = "default_fp16_sdpa_b1"

# Gate 1 profile set (compile profile only with --with-compile).
GATE1_PROFILES = [
    "naive_fp32_eager_b1",
    "default_fp16_sdpa_b1",
    "fp16_sdpa_b4",
    "fp16_sdpa_b8",
    "fp16_sdpa_b1_cap64",
]
COMPILE_PROFILE = "fp16_sdpa_b4_compile"

# 24-config search grid: dtype x attn x batch x max_new_tokens, no compile.
SPACE = [
    make_config(dtype, attn, batch, compile=False, max_new_tokens=cap)
    for dtype, attn, batch, cap
    in itertools.product(DTYPES, ATTNS, BATCHES, TOKEN_CAPS)
]


def config_slug(config):
    """Filesystem-safe profile name for a grid config."""
    validate_config(config)
    slug = "%s_%s_b%d_cap%d" % (
        config["dtype"], config["attn"], config["batch"],
        config["max_new_tokens"])
    if config["compile"]:
        slug += "_compile"
    return slug


def profile_config(name):
    """Config for a named profile; ``KeyError`` lists known names."""
    try:
        return dict(PROFILES[name])
    except KeyError:
        raise KeyError("unknown profile %r; known: %s"
                       % (name, sorted(PROFILES))) from None
