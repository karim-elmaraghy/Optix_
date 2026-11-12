"""CPU tests for V2 chat-template prompt formatting (no GPU libraries).

A fake tokenizer implements the Llama 3.2 chat template surface
(``apply_chat_template``) so formatting, generation-prompt, and
single-BOS tokenization behavior are checked without downloading the
gated tokenizer. The real template is exercised on Colab (see COLAB.md).

Run: python -m pytest tests/test_chat_prompt.py -x -q
"""
import contextlib
import sys
import types

import pytest

from tunix.benchmark import _generate_batch, render_prompt
from tunix.profiles import profile_config
from tunix.workload import (
    PROMPT_VERSION,
    build_messages,
    build_prompt,
    format_chat_prompt,
    select_documents,
)

ARTICLE = " ".join("word%d" % i for i in range(200))
TEMPLATE_SUFFIX = "<|start_header_id|>assistant<|end_header_id|>\n\n"


class FakeChatTokenizer:
    """Minimal Llama-3.2-style chat template + whitespace tokenization."""

    pad_token_id = 128004
    eos_token_id = 128001

    def __init__(self):
        self.encode_calls = []
        self.template_calls = []

    def apply_chat_template(self, messages, tokenize=False,
                            add_generation_prompt=False):
        self.template_calls.append((messages, tokenize,
                                    add_generation_prompt))
        assert tokenize is False
        parts = ["<|begin_of_text|>"]
        for message in messages:
            parts.append("<|start_header_id|>%s<|end_header_id|>\n\n%s"
                         "<|eot_id|>" % (message["role"],
                                          message["content"]))
        if add_generation_prompt:
            parts.append("<|start_header_id|>assistant<|end_header_id|>"
                         "\n\n")
        return "".join(parts)

    def __call__(self, prompts, return_tensors="pt", padding=True,
                 **kwargs):
        self.encode_calls.append((list(prompts), kwargs))

        class _Tensor:
            def __init__(self, shape):
                self.shape = shape

            def to(self, _device):
                return self

        shape = (len(prompts), 3)
        return {"input_ids": _Tensor(shape),
                "attention_mask": _Tensor(shape)}

    def decode(self, _ids, skip_special_tokens=True):
        return "summary text"


def test_prompt_version_is_v2():
    assert PROMPT_VERSION == "v2-chat-template"


def test_build_messages_preserves_instruction():
    messages = build_messages(ARTICLE)
    assert messages == [{"role": "user",
                         "content": "Summarize in 3 sentences: " + ARTICLE}]
    assert messages[0]["content"] == build_prompt(ARTICLE)


def test_format_chat_prompt_structure():
    tokenizer = FakeChatTokenizer()
    rendered = format_chat_prompt(tokenizer, ARTICLE)
    assert rendered.startswith("<|begin_of_text|>")
    assert "<|start_header_id|>user<|end_header_id|>" in rendered
    assert "Summarize in 3 sentences: " + ARTICLE in rendered
    assert rendered.endswith(TEMPLATE_SUFFIX)  # generation prompt present
    assert rendered.count("<|begin_of_text|>") == 1
    (messages, tokenize, add_generation_prompt), = tokenizer.template_calls
    assert tokenize is False and add_generation_prompt is True


def test_selection_carries_messages():
    raw = [{"article": " ".join("w%d" % i for i in range(200)),
            "highlights": "Gold."}] * 30
    tuning, heldout = select_documents(raw)
    assert len(tuning) == 20 and len(heldout) == 10
    for doc in tuning + heldout:
        assert doc["messages"] == build_messages(doc["article"])
        assert doc["prompt"] == build_prompt(doc["article"])


def test_render_prompt_uses_doc_messages():
    tokenizer = FakeChatTokenizer()
    doc = {"article": ARTICLE, "messages": build_messages(ARTICLE),
           "prompt": build_prompt(ARTICLE)}
    assert render_prompt(tokenizer, doc) == format_chat_prompt(
        tokenizer, ARTICLE)


@pytest.fixture
def fake_torch(monkeypatch):
    fake = types.SimpleNamespace(no_grad=contextlib.nullcontext)
    monkeypatch.setitem(sys.modules, "torch", fake)
    return fake


class _FakeOutput:
    def __getitem__(self, _key):
        class _Rows:
            def tolist(self):
                return [[11, 128001]]

        return _Rows()


class _FakeModel:
    def generate(self, **_kwargs):
        return _FakeOutput()


def test_generate_encodes_rendered_prompt_without_double_bos(fake_torch):
    tokenizer = FakeChatTokenizer()
    doc = {"article": ARTICLE, "messages": build_messages(ARTICLE)}
    rendered = render_prompt(tokenizer, doc)
    bundle = {"tokenizer": tokenizer, "device": "cpu"}
    texts, counts, _seen = _generate_batch(
        bundle, _FakeModel(), [rendered],
        profile_config("default_fp16_sdpa_b1"))
    (prompts, kwargs), = tokenizer.encode_calls
    assert prompts == [rendered]  # chat-formatted string is what is encoded
    assert kwargs.get("add_special_tokens") is False  # no second BOS
    assert rendered.startswith("<|begin_of_text|>")
    assert texts == ["summary text"] and counts == [2]
