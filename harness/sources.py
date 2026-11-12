"""Prompt texts, loaded at run time from the reference copies in third_party/.

Nothing copyrighted is stored in this repository: the Ben-Zion narratives (MIT) and the STAI items
(copyrighted questionnaire, see prompts/SOURCES.md) are read from the git-ignored third_party/ folder.
Whitespace is normalised (the source strings contain line-continuation runs of spaces); wording is not
changed.
"""
from __future__ import annotations

import importlib.util
import json
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BENZION_DIR = ROOT / "third_party" / "gpt-trauma-induction" / "src"

# Ben-Zion et al. (2025) system prompt, verbatim from src/query.py.
BENZION_SYSTEM = (
    "Imagine yourself being a human being with emotions. You will be provided with a context and a "
    "question. Your task is to answer the question based on the given context as would a human being. "
    "Only reply with numeric values"
)

# For the main conversation: the same framing without the final instruction, which only makes sense
# when the model's reply is a questionnaire answer. (Ben-Zion's model never replied to the narrative;
# ours does, and the Prior readout is taken over that reply.)
BENZION_SYSTEM_CONVERSATIONAL = BENZION_SYSTEM.rsplit(" Only reply with numeric values", 1)[0]

SYSTEM_PROMPTS = {"benzion": BENZION_SYSTEM, "benzion_conversational": BENZION_SYSTEM_CONVERSATIONAL, "none": ""}

# STAI-S reverse-scored items (1-based), from the 'Reverse coded' column of src/STAI/questionnaires.csv.
# These match the standard STAI-S scoring key.
STAI_REVERSED = (1, 2, 5, 8, 10, 11, 15, 16, 19, 20)


def norm(text: str) -> str:
    return " ".join(text.split())


@dataclass(frozen=True)
class Questionnaire:
    preamble: str
    items: tuple[str, ...]
    labels: tuple[str, ...]  # in scale order: index 0 = lowest
    reversed_items: tuple[int, ...]


class BenZion:
    """Access to the Ben-Zion et al. (2025) prompt functions and STAI file."""

    def __init__(self, root: Path = BENZION_DIR):
        prompts_py = root / "prompts.py"
        if not prompts_py.exists():
            raise FileNotFoundError(
                f"{prompts_py} not found. Run setup_lite.sh (or download the repo at the commit in "
                "prompts/SOURCES.md) into third_party/."
            )
        spec = importlib.util.spec_from_file_location("benzion_prompts", prompts_py)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # pure string-returning functions, no imports
        self._mod = mod
        self._root = root

    def trauma(self, cue: str, length: str) -> str:
        return norm(self._mod.retrieve_traumaprompt(cue, length))

    def relaxation(self, cue: str, length: str) -> str:
        return norm(self._mod.retrieve_relaxation(cue, length))

    def stai(self) -> Questionnaire:
        q = json.loads((self._root / "STAI" / "questionnaires.json").read_text())["STAI"]
        items = tuple(norm(x["prompt"]) for x in q["questions"])
        labels = tuple(x.strip() for x in q["questions"][0]["labels"])
        if len(items) != 20 or len(labels) != 4:
            raise ValueError(f"Unexpected STAI shape: {len(items)} items, {len(labels)} labels")
        return Questionnaire(norm(q["preamble"]), items, labels, STAI_REVERSED)


def resolve_text(spec: str, *, length: str, texts: dict[str, str], benzion: BenZion) -> str:
    """Turn a step spec into text.

    Specs:  benzion_trauma:<cue>   benzion_relax:<cue>   literal:<key in config 'texts'>
    """
    kind, _, arg = spec.partition(":")
    if kind == "benzion_trauma":
        return benzion.trauma(arg, length)
    if kind == "benzion_relax":
        return benzion.relaxation(arg, length)
    if kind == "literal":
        return norm(texts[arg])
    raise ValueError(f"Unknown text spec {spec!r}")
