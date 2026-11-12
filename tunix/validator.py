"""CPU quality gates. Calibrate only on tuning data; score heldout separately.

Metrics use lowercase Unicode word tokens (no stemming). Calibration permits
the observed fp32/fp16 mean difference plus an explicit absolute F1 margin.
These lexical checks detect regressions, not factual correctness.
"""
import math
import re
from collections import Counter
from dataclasses import asdict, dataclass


def tokens(text):
    return re.findall(r"\w+", text.lower(), re.UNICODE)


def token_f1(a, b):
    a, b = tokens(a), tokens(b)
    if not a or not b:
        return 0.0
    return 2 * sum((Counter(a) & Counter(b)).values()) / (len(a) + len(b))


def rouge_l(a, b):
    a, b = tokens(a), tokens(b)
    if not a or not b:
        return 0.0
    row = [0] * (len(b) + 1)
    for x in a:
        prev = 0
        for j, y in enumerate(b, 1):
            old = row[j]
            row[j] = prev + 1 if x == y else max(row[j], row[j - 1])
            prev = old
    return 2 * row[-1] / (len(a) + len(b))


def index_docs(bundle, split):
    docs = bundle['docs']
    if not docs or any(d['split'] != split for d in docs):
        raise ValueError('empty or mixed/wrong split')
    indexed = {d['doc_id']: d for d in docs}
    if len(indexed) != len(docs):
        raise ValueError('duplicate doc_id')
    for d in docs:
        if not isinstance(d['output_text'], str) or not isinstance(d['gold_summary'], str):
            raise ValueError('invalid text field')
        if type(d['n_output_tokens']) is not int or d['n_output_tokens'] < 0:
            raise ValueError('invalid n_output_tokens')
    return indexed


@dataclass(frozen=True)
class Calibration:
    overlap_drop: float
    gold_drop: float
    order_drop: float
    margin: float
    tuning_ids: tuple


class CalibrationBlocked(ValueError):
    """Invalid baseline artifacts cannot establish a quality tolerance."""
    def __init__(self, diagnostics):
        self.diagnostics = diagnostics
        self.reasons = sorted({r for d in diagnostics for r in d['reasons']})
        super().__init__('Calibration blocked: invalid baseline outputs (' + ', '.join(self.reasons) + ')')


def calibrate(default, naive, margin=0.02):
    if not math.isfinite(margin) or not 0 <= margin < 1:
        raise ValueError('margin must be finite in [0,1)')
    base, fp32 = index_docs(default, 'tuning'), index_docs(naive, 'tuning')
    if base.keys() != fp32.keys():
        raise ValueError('calibration document mismatch')
    diagnostics = []
    for label, docs, config in (('competent_default', base, default['config']),
                                ('naive', fp32, naive['config'])):
        for key, doc in docs.items():
            reasons = invalid_reasons(doc, config)
            if reasons:
                diagnostics.append(dict(baseline=label, doc_id=key, reasons=reasons))
    if diagnostics:
        raise CalibrationBlocked(diagnostics)
    overlap, gold, order = [], [], []
    for key, d in base.items():
        n = fp32[key]
        if d['gold_summary'] != n['gold_summary'] or not tokens(d['gold_summary']):
            raise ValueError('gold mismatch or empty gold')
        if invalid_reasons(d, default['config']) or invalid_reasons(n, naive['config']):
            raise ValueError('invalid calibration output')
        overlap.append(1 - token_f1(d['output_text'], n['output_text']))
        order.append(1 - rouge_l(d['output_text'], n['output_text']))
        gold.append(abs(rouge_l(d['output_text'], d['gold_summary']) -
                        rouge_l(n['output_text'], d['gold_summary'])))
    return Calibration(sum(overlap)/len(overlap), sum(gold)/len(gold),
                       sum(order)/len(order), margin, tuple(base))


def invalid_reasons(doc, config):
    text = doc['output_text']
    words = tokens(text)
    reasons = []
    if not text.strip():
        reasons.append('empty')
    if doc['n_output_tokens'] < 10 or len(words) < 10:
        reasons.append('short')
    if doc['n_output_tokens'] >= config['max_new_tokens']:
        reasons.append('truncation')
    if words and (len(set(words))/len(words) < 0.25 or
                  sum(c.isalpha() for c in text)/max(len(text), 1) < 0.5):
        reasons.append('garbage')
    if any(t in text for t in ('<|', '\ufffd')):
        reasons.append('invalid_text')
    return reasons


def validate(candidate, default, calibration, split='tuning'):
    """Return JSON-serializable decision; fail closed on malformed artifacts."""
    try:
        base, cand = index_docs(default, split), index_docs(candidate, split)
        if base.keys() != cand.keys():
            raise ValueError('document set mismatch')
        if split == 'heldout' and set(base).intersection(calibration.tuning_ids):
            raise ValueError('heldout overlaps calibration')
        rows = []
        for key, d in cand.items():
            b = base[key]
            if d['gold_summary'] != b['gold_summary'] or not tokens(b['gold_summary']):
                raise ValueError('gold mismatch or empty gold')
            if invalid_reasons(b, default['config']):
                raise ValueError('invalid default output')
            reasons = invalid_reasons(d, candidate['config'])
            overlap = token_f1(d['output_text'], b['output_text'])
            gold = rouge_l(d['output_text'], b['gold_summary'])
            order = rouge_l(d['output_text'], b['output_text'])
            # Word-bag equality alone accepts shuffled summaries. Require the
            # same calibrated retention for ordered subsequences as overlap.
            limit = calibration.overlap_drop + calibration.margin
            if 1 - overlap > limit + 1e-12:
                reasons.append('overlap_drop')
            if 1 - order > calibration.order_drop + calibration.margin + 1e-12:
                reasons.append('order_drop')
            if rouge_l(b['output_text'], b['gold_summary']) - gold > calibration.gold_drop + calibration.margin + 1e-12:
                reasons.append('gold_rouge_drop')
            rows.append(dict(doc_id=key, token_f1=overlap, rouge_l=gold, reasons=reasons))
        return dict(accepted=all(not r['reasons'] for r in rows), split=split,
                    calibration=asdict(calibration), docs=rows,
                    reasons=sorted({reason for r in rows for reason in r['reasons']}))
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        return dict(accepted=False, split=split, docs=[], reasons=['invalid_artifact'], error=str(exc))
