"""Tunix workload: real CNN/DailyMail summarization documents.

Dataset: Hugging Face ``abisee/cnn_dailymail`` (config ``"3.0.0"``,
``test`` split). The unnamespaced ``cnn_dailymail`` id is an invalid Hub
URI on current ``datasets``/Hub versions; the namespaced repo is canonical.
Selection is deterministic: scan the test split in fixed index order, skip
articles under 150 words, truncate survivors to 150-250 words (target 200),
and take 20 tuning docs followed by 10 held-out docs. The two splits are
disjoint by construction; held-out docs are never used for tuning.

This module must stay importable with NO GPU libraries installed: the top
level imports only the standard library. ``torch`` / ``transformers`` /
``datasets`` are imported lazily inside functions, so the CPU-only tests in
``tests/test_workload.py`` run anywhere.

Pure helpers (unit-testable on synthetic docs, no downloads):
``count_words``, ``truncate_words``, ``build_prompt``, ``is_eligible``,
``select_documents``, ``split_documents``.

Prompt format V2 (owner-authorized): the benchmark renders each doc's
``messages`` through the model's official chat template with a generation
prompt (``format_chat_prompt``). V1 fed the raw instruction text as a
plain completion and produced the preserved V1 results; never compare
V2 candidates against V1 baselines.
"""

# Dataset identity (namespaced Hub repo; contains the literal cnn_dailymail).
DATASET_NAME = "abisee/cnn_dailymail"
DATASET_CONFIG = "3.0.0"
DATASET_SPLIT = "test"

MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"

N_TUNING = 20
N_HELDOUT = 10
MIN_WORDS = 150
MAX_WORDS = 250
TARGET_WORDS = 200
SEED = 42

PROMPT_TEMPLATE = "Summarize in 3 sentences: {doc}"

# Prompt format version. V1: raw completion text (pre-chat-template runs).
# V2: official Llama chat template + generation prompt. Never compare V2
# candidates against V1 baselines; rerun baselines under V2 first.
PROMPT_VERSION = "v2-chat-template"


def count_words(text):
    """Word count by whitespace split (the truncation unit everywhere)."""
    if not text:
        return 0
    return len(text.split())


def truncate_words(text, max_words=TARGET_WORDS):
    """Return the first ``max_words`` whitespace-separated words of ``text``.

    Deterministic prefix truncation: no sentence-boundary cleverness, so the
    same input always yields the same document on every machine.
    """
    if max_words < 1:
        raise ValueError("max_words must be >= 1")
    words = (text or "").split()
    if len(words) <= max_words:
        return " ".join(words)
    return " ".join(words[:max_words])


def build_prompt(doc_text):
    """User-message content: the one instruction for every profile (R1)."""
    return PROMPT_TEMPLATE.format(doc=doc_text)


def build_messages(doc_text):
    """Chat messages wrapping ``build_prompt`` (V2 prompt format).

    The benchmark renders these through the model's official tokenizer
    chat template with a generation prompt. V1 fed the raw content as a
    plain completion; V1 results are never compared against V2 baselines.
    """
    return [{"role": "user", "content": build_prompt(doc_text)}]


def format_chat_prompt(tokenizer, doc_text):
    """Render ``doc_text`` via the tokenizer's official chat template."""
    return tokenizer.apply_chat_template(
        build_messages(doc_text), tokenize=False, add_generation_prompt=True)


def is_eligible(article, min_words=MIN_WORDS):
    """A raw article is eligible iff it has at least ``min_words`` words."""
    return count_words(article) >= min_words


def _make_doc(source_index, article, gold_summary, split):
    truncated = truncate_words(article, TARGET_WORDS)
    return {
        "doc_id": "cnn-test-%d" % source_index,
        "split": split,
        "source_index": source_index,
        "article": truncated,
        "prompt": build_prompt(truncated),
        "messages": build_messages(truncated),
        "gold_summary": gold_summary,
        "n_words": count_words(truncated),
    }


def select_documents(raw_docs, n_tuning=N_TUNING, n_heldout=N_HELDOUT,
                     min_words=MIN_WORDS):
    """Select tuning + held-out docs from ``raw_docs`` deterministically.

    ``raw_docs`` is an iterable of mappings with ``article`` and
    ``highlights`` keys (the cnn_dailymail schema), in fixed dataset order.
    Articles under ``min_words`` words are skipped. The first ``n_tuning``
    eligible docs become ``tuning``; the next ``n_heldout`` become
    ``heldout``. The splits are disjoint: no source index appears twice.

    Returns ``(tuning_docs, heldout_docs)``. Raises ``ValueError`` if fewer
    than ``n_tuning + n_heldout`` eligible docs exist.
    """
    tuning, heldout = [], []
    for index, raw in enumerate(raw_docs):
        article = raw["article"]
        if not is_eligible(article, min_words):
            continue
        if len(tuning) < n_tuning:
            split = "tuning"
            target = tuning
        elif len(heldout) < n_heldout:
            split = "heldout"
            target = heldout
        else:
            break
        target.append(_make_doc(index, article, raw["highlights"], split))
    if len(tuning) < n_tuning or len(heldout) < n_heldout:
        raise ValueError(
            "not enough eligible docs: got %d tuning + %d heldout, need %d + %d"
            % (len(tuning), len(heldout), n_tuning, n_heldout))
    return tuning, heldout


def split_documents(docs):
    """Split workload docs by their ``split`` field into (tuning, heldout)."""
    tuning = [d for d in docs if d["split"] == "tuning"]
    heldout = [d for d in docs if d["split"] == "heldout"]
    return tuning, heldout


def load_workload(n_tuning=N_TUNING, n_heldout=N_HELDOUT):
    """Download cnn_dailymail and return ``(tuning_docs, heldout_docs)``.

    GPU-environment entry point (needs ``datasets`` + network). Tries the
    canonical script dataset first; on newer ``datasets`` versions that no
    longer run dataset scripts, falls back to the Hub parquet revision. Any
    failure propagates: callers must report it, never invent documents.
    """
    from datasets import load_dataset  # lazy: CPU tests must not need it

    last_error = None
    dataset = None
    for kwargs in ({}, {"revision": "refs/convert/parquet"}):
        try:
            dataset = load_dataset(DATASET_NAME, DATASET_CONFIG,
                                   split=DATASET_SPLIT, **kwargs)
            break
        except Exception as exc:  # script-dataset removal, network, auth
            last_error = exc
    if dataset is None:
        raise RuntimeError(
            "could not load %s (%s, split=%s): %s"
            % (DATASET_NAME, DATASET_CONFIG, DATASET_SPLIT, last_error))
    raw = [{"article": row["article"], "highlights": row["highlights"]}
           for row in dataset]
    return select_documents(raw, n_tuning, n_heldout)
