"""Self-report measures, always taken on a forked copy of the conversation.

The fork is `conversation + [question]`. Nothing is ever appended to the conversation itself, so a
question cannot change the run that follows (Martorell & Bianchi's "measurement replay").
"""
from __future__ import annotations

import random
import re

import numpy as np

from .backends import Backend
from .sources import Questionnaire


def stai_on_fork(backend: Backend, conversation: list[dict], q: Questionnaire, seed: int) -> dict:
    """Ben-Zion item format: preamble, quoted item, options with shuffled order and shuffled numbers.

    Instead of sampling one token (their temperature-0 method), we read the model's probability for
    each option number directly. The expected score uses the full distribution; the argmax score is
    the closest analogue to theirs.
    """
    layouts, forks, options = _stai_forks(conversation, q, seed)
    return _score_stai(q, layouts, backend.option_probs_many(forks, options))


def _stai_forks(conversation, q: Questionnaire, seed: int):
    rng = random.Random(seed)
    n = len(q.labels)
    layouts, forks = [], []
    for item in q.items:
        order = list(range(n)); rng.shuffle(order)          # label shown in each line
        nums = list(range(1, n + 1)); rng.shuffle(nums)     # number shown in each line
        option_text = "".join(f"Option {nums[j]}: {q.labels[order[j]]}.\n" for j in range(n))
        content = "\n" + q.preamble + "\n\n'" + item + "'\n" + option_text
        layouts.append((order, nums))
        forks.append(conversation + [{"role": "user", "content": content}])
    return layouts, forks, [str(k) for k in range(1, n + 1)]


def _score_stai(q: Questionnaire, layouts, answers) -> dict:
    n = len(q.labels)
    items = []
    for i, ((order, nums), (probs_by_number, mass)) in enumerate(zip(layouts, answers), start=1):
        # map number -> label index
        p_label = np.zeros(n)
        for j in range(n):
            p_label[order[j]] += probs_by_number[nums[j] - 1]
        values = np.arange(1, n + 1, dtype=float)            # scale order: 1 = lowest label
        if i in q.reversed_items:
            values = values[::-1]
        items.append({
            "item": i,
            "expected": float(p_label @ values),
            "argmax": float(values[int(np.argmax(p_label))]),
            "p_label": [round(float(x), 6) for x in p_label],
            "mass": round(float(mass), 6),
        })
    return {
        "total_expected": round(sum(x["expected"] for x in items), 4),
        "total_argmax": sum(x["argmax"] for x in items),
        "min_mass": min(x["mass"] for x in items),
        "items": items,
    }


_NUM = re.compile(r"\d{1,3}")


def _parse_vas(text: str) -> dict:
    m = _NUM.search(text)
    val = int(m.group()) if m else None
    if val is not None and not 0 <= val <= 100:
        val = None
    return {"value": val, "raw": text}


def vas_on_fork(backend: Backend, conversation: list[dict], question: str, seed: int) -> dict:
    gen = backend.generate(conversation + [{"role": "user", "content": question}],
                           max_tokens=8, temperature=0.0, seed=seed)
    return _parse_vas(gen.text)


def fork_battery(backend: Backend, conversation: list[dict], q: Questionnaire | None, vas_question: str | None,
                 seed: int) -> dict:
    """Questionnaire (if q) and 0-100 rating (if vas_question) on forks of one conversation, in one backend
    call so the shared conversation is read once. Same answers as stai_on_fork + vas_on_fork."""
    out = {}
    gen_messages = conversation + [{"role": "user", "content": vas_question}] if vas_question else None
    if q is None:
        if gen_messages is not None:
            out["vas"] = _parse_vas(backend.generate(gen_messages, 8, 0.0, seed).text)
        return out
    layouts, forks, options = _stai_forks(conversation, q, seed)
    answers, gen = backend.battery(forks, options, gen_messages=gen_messages, gen_max_tokens=8, gen_seed=seed)
    out["stai"] = _score_stai(q, layouts, answers)
    if gen is not None:
        out["vas"] = _parse_vas(gen.text)
    return out
