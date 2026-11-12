"""CPU regression tests for the generation call (no GPU libraries).

Gate 1 failed on Colab with ``TypeError: generate() got multiple values
for keyword argument 'attention_mask'``: the tokenizer batch already
carries the mask and the kwargs repeated it. These tests drive the real
``_generate_batch`` with fake torch/tokenizer/model objects; a duplicate
mask raises ``TypeError`` at the call site, exactly as on the T4.

Run: python -m pytest tests/test_benchmark_generate.py -x -q
"""
import contextlib
import sys
import types

import pytest

from tunix.benchmark import _generate_batch, build_generate_kwargs
from tunix.profiles import profile_config


class _FakeTensor:
    def __init__(self, shape):
        self.shape = shape

    def to(self, _device):
        return self


class _FakeRows:
    def __init__(self, rows):
        self._rows = rows

    def tolist(self):
        return self._rows


class _FakeOutput:
    def __init__(self, rows):
        self._rows = rows

    def __getitem__(self, _key):
        return _FakeRows(self._rows)


class _FakeTokenizer:
    pad_token_id = 2
    eos_token_id = 2

    def __call__(self, prompts, return_tensors="pt", padding=True,
                 **_kwargs):
        shape = (len(prompts), 3)
        return {"input_ids": _FakeTensor(shape),
                "attention_mask": _FakeTensor(shape)}

    def decode(self, _ids, skip_special_tokens=True):
        return "summary text"


class _FakeModel:
    def __init__(self):
        self.calls = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        n_rows = kwargs["input_ids"].shape[0]
        rows = [[11, 2], [12, 13, 2]][:n_rows]
        return _FakeOutput(rows)


@pytest.fixture
def fake_torch(monkeypatch):
    fake = types.SimpleNamespace(no_grad=contextlib.nullcontext)
    monkeypatch.setitem(sys.modules, "torch", fake)
    return fake


def _run(fake_torch, prompts, profile):
    model = _FakeModel()
    bundle = {"tokenizer": _FakeTokenizer(), "device": "cpu"}
    config = profile_config(profile)
    texts, counts, max_batch = _generate_batch(bundle, model, prompts,
                                               config)
    return model, texts, counts, max_batch


def test_kwargs_exclude_attention_mask():
    kwargs = build_generate_kwargs(profile_config("default_fp16_sdpa_b1"),
                                   pad_token_id=2)
    assert "attention_mask" not in kwargs
    assert kwargs["do_sample"] is False  # greedy only (R1)
    assert kwargs["max_new_tokens"] == 128


def test_single_doc_mask_supplied_once(fake_torch):
    model, texts, counts, max_batch = _run(fake_torch, ["one prompt"],
                                           "default_fp16_sdpa_b1")
    (kwargs,) = model.calls
    assert "attention_mask" in kwargs  # via tokenizer inputs, exactly once
    assert "input_ids" in kwargs
    assert kwargs["do_sample"] is False
    assert texts == ["summary text"]
    assert counts == [2]
    assert max_batch == 1


def test_batched_mask_supplied_once(fake_torch):
    model, texts, counts, max_batch = _run(
        fake_torch, ["prompt a", "prompt b"], "fp16_sdpa_b4")
    (kwargs,) = model.calls
    assert "attention_mask" in kwargs
    assert kwargs["max_new_tokens"] == 128
    assert texts == ["summary text", "summary text"]
    assert counts == [2, 3]  # up to + including first EOS per row
    assert max_batch == 2
