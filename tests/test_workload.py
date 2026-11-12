"""CPU-only workload tests (Subtask A). No GPU libraries required.

Must pass with NO torch / transformers / datasets installed: every test
uses synthetic in-memory documents. Covers deterministic selection,
tuning/held-out disjointness, word-count bounds, skipping, truncation,
prompt building, and gold-summary preservation.

Run: python -m pytest tests/test_workload.py -x -q
"""
import pytest

from tunix.workload import (
    DATASET_CONFIG,
    DATASET_NAME,
    DATASET_SPLIT,
    MAX_WORDS,
    MIN_WORDS,
    TARGET_WORDS,
    build_prompt,
    count_words,
    is_eligible,
    select_documents,
    split_documents,
    truncate_words,
)


def make_article(n_words, tag="doc"):
    return " ".join("%s%d" % (tag, i) for i in range(n_words))


def synthetic_raw(n_docs=45):
    """Raw cnn_dailymail-shaped rows; every 7th is too short (100 words)."""
    raw = []
    for i in range(n_docs):
        n_words = 100 if i % 7 == 0 else 180 + (i % 40)
        raw.append({
            "article": make_article(n_words, tag="a%d_" % i),
            "highlights": "Gold summary %d." % i,
        })
    return raw


def test_dataset_identity():
    # Namespaced Hub repo: bare "cnn_dailymail" is an invalid HF URI.
    assert DATASET_NAME == "abisee/cnn_dailymail"
    assert DATASET_CONFIG == "3.0.0"
    assert DATASET_SPLIT == "test"


def test_count_words():
    assert count_words("") == 0
    assert count_words("one two  three") == 3
    assert count_words(None) == 0


def test_truncate_words():
    text = make_article(500)
    out = truncate_words(text)
    assert count_words(out) == TARGET_WORDS
    assert out.split()[0] == "doc0"
    assert truncate_words("a b c", max_words=10) == "a b c"
    with pytest.raises(ValueError):
        truncate_words("a b", max_words=0)


def test_build_prompt():
    assert build_prompt("TEXT") == "Summarize in 3 sentences: TEXT"


def test_is_eligible():
    assert is_eligible(make_article(MIN_WORDS))
    assert not is_eligible(make_article(MIN_WORDS - 1))


def test_deterministic_selection():
    raw = synthetic_raw()
    first = select_documents(raw)
    second = select_documents(raw)
    assert first == second  # same input -> identical docs, every run


def test_tuning_heldout_disjoint():
    tuning, heldout = select_documents(synthetic_raw())
    assert len(tuning) == 20
    assert len(heldout) == 10
    tuning_ids = {d["doc_id"] for d in tuning}
    heldout_ids = {d["doc_id"] for d in heldout}
    assert tuning_ids.isdisjoint(heldout_ids)  # disjoint doc ids
    tuning_src = {d["source_index"] for d in tuning}
    heldout_src = {d["source_index"] for d in heldout}
    assert tuning_src.isdisjoint(heldout_src)  # disjoint source rows
    assert all(d["split"] == "tuning" for d in tuning)
    assert all(d["split"] == "heldout" for d in heldout)


def test_word_count_bounds():
    tuning, heldout = select_documents(synthetic_raw())
    for doc in tuning + heldout:
        assert MIN_WORDS <= doc["n_words"] <= MAX_WORDS
        assert doc["n_words"] == count_words(doc["article"])


def test_short_articles_are_skipped():
    raw = synthetic_raw()
    tuning, heldout = select_documents(raw)
    used = {d["source_index"] for d in tuning + heldout}
    short_indices = {i for i, row in enumerate(raw)
                     if count_words(row["article"]) < MIN_WORDS}
    assert used.isdisjoint(short_indices)


def test_long_articles_truncated_to_target():
    raw = [{"article": make_article(500, tag="x"),
            "highlights": "Gold."}] * 30
    tuning, _heldout = select_documents(raw)
    assert all(d["n_words"] == TARGET_WORDS for d in tuning)


def test_prompt_and_gold_preserved():
    raw = synthetic_raw()
    tuning, _heldout = select_documents(raw)
    for doc in tuning:
        assert doc["prompt"] == build_prompt(doc["article"])
        assert doc["prompt"].startswith("Summarize in 3 sentences: ")
        source = raw[doc["source_index"]]
        assert doc["gold_summary"] == source["highlights"]


def test_selection_order_is_dataset_order():
    tuning, heldout = select_documents(synthetic_raw())
    indices = [d["source_index"] for d in tuning + heldout]
    assert indices == sorted(indices)  # fixed scan order, no shuffling


def test_split_documents_roundtrip():
    tuning, heldout = select_documents(synthetic_raw())
    tuning2, heldout2 = split_documents(tuning + heldout)
    assert tuning2 == tuning
    assert heldout2 == heldout


def test_not_enough_docs_raises():
    with pytest.raises(ValueError):
        select_documents(synthetic_raw(n_docs=10))
