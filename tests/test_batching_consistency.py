"""CPU regression tests for T5 batching consistency (tiny model).

Gate 1 on a T4 reported batch-4 vs batch-1 F1 as low as 0.5871 for one
doc. These tests drive the production ``_generate_batch`` with a tiny
seeded Llama and a stub tokenizer, and require batched generation to
reproduce batch-1 outputs exactly (text and token counts) for batch 4
and batch 8. They catch padding / attention-mask / position / slicing
regressions. Skipped when torch/transformers are not installed, so the
bare CPU suite still runs without GPU libraries.

Run: python -m pytest tests/test_batching_consistency.py -x -q
"""
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from transformers import LlamaConfig, LlamaForCausalLM  # noqa: E402

from tunix.benchmark import _generate_batch  # noqa: E402

PAD_ID = 0
EOS_ID = 2
_LENGTHS = (6, 13, 9, 17, 7, 12, 10, 15)


def _prompt_ids():
    rng = torch.Generator().manual_seed(23)
    return {"prompt-%d" % i: torch.randint(3, 500, (n,), generator=rng)
            .tolist() for i, n in enumerate(_LENGTHS)}


class _StubTokenizer:
    """Maps prompt strings to fixed id sequences; left-pads batches."""

    pad_token_id = PAD_ID
    eos_token_id = EOS_ID

    def __init__(self, prompt_ids):
        self._ids = prompt_ids

    def __call__(self, prompts, return_tensors="pt", padding=True,
                 **_kwargs):
        seqs = [self._ids[p] for p in prompts]
        width = max(len(s) for s in seqs)
        ids = [[PAD_ID] * (width - len(s)) + s for s in seqs]
        mask = [[0] * (width - len(s)) + [1] * len(s) for s in seqs]
        return {"input_ids": torch.tensor(ids),
                "attention_mask": torch.tensor(mask)}

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(str(i) for i in ids)


def _tiny_model(dtype):
    torch.manual_seed(0)
    config = LlamaConfig(vocab_size=512, hidden_size=128,
                         intermediate_size=256, num_hidden_layers=2,
                         num_attention_heads=4, num_key_value_heads=4,
                         max_position_embeddings=512)
    return LlamaForCausalLM(config).to(dtype).eval()


@pytest.mark.parametrize("batch_size", (4, 8))
def test_batched_matches_batch1_exactly(batch_size):
    prompt_ids = _prompt_ids()
    prompts = list(prompt_ids)
    bundle = {"tokenizer": _StubTokenizer(prompt_ids), "device": "cpu"}
    model = _tiny_model(torch.float32)
    config = {"max_new_tokens": 48}
    singles = {}
    for prompt in prompts:
        texts, counts, _seen = _generate_batch(bundle, model, [prompt],
                                               config)
        singles[prompt] = (texts[0], counts[0])
    for start in range(0, len(prompts), batch_size):
        chunk = prompts[start:start + batch_size]
        texts, counts, seen = _generate_batch(bundle, model, chunk, config)
        assert seen == len(chunk)
        for prompt, text, count in zip(chunk, texts, counts):
            assert (text, count) == singles[prompt], (
                "batch %d diverged from batch 1 for %s"
                % (batch_size, prompt))


def test_padded_logits_match_unpadded():
    """Masking/positions: padded row logits must match the single row."""
    prompt_ids = _prompt_ids()
    seqs = list(prompt_ids.values())
    tokenizer = _StubTokenizer(prompt_ids)
    model = _tiny_model(torch.float32)
    with torch.no_grad():
        singles = [model(input_ids=torch.tensor([s])).logits[0, -1]
                   for s in seqs]
        batch = tokenizer(list(prompt_ids), return_tensors="pt",
                          padding=True)
        batched = model(**batch).logits
    for i in range(len(seqs)):
        diff = (batched[i, -1] - singles[i]).abs().max().item()
        assert diff < 1e-5, "row %d logits diverged: %g" % (i, diff)
