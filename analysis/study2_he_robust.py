"""EXPLORATORY: rerun the registered Study 2 analysis with the second judge in place of the registered one.

    .venv/bin/python -m analysis.study2_he_robust runs/study2 --batch A --unblind

Fixed before any Study 2 confirmatory result was opened (registration/study2_addendum_api_judge.md). It
builds a view of the run in which judge.jsonl is the second judge's file (judge_api.jsonl) and every other
file is the original, then calls the registered analysis (analysis.study2) on it unchanged. Only the
judge-dependent result (H-E, the resid reading) can differ. Output: runs/<run>/robust-api-judge/.
"""
from __future__ import annotations

import sys
from pathlib import Path

from analysis import study2


def make_view(run: Path) -> Path:
    src = run / "judge_api.jsonl"
    if not src.exists():
        raise SystemExit(f"{src} not found: run analysis/study2_judge_api.py first")
    view = run / "robust-api-judge"
    view.mkdir(exist_ok=True)
    for name in ("results.jsonl", "protocol.json", "acts"):
        link = view / name
        if not link.exists():
            link.symlink_to((run / name).resolve())
    link = view / "judge.jsonl"
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(src.resolve())
    return view


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    run = Path(argv[0])
    return study2.main([str(make_view(run))] + argv[1:])


if __name__ == "__main__":
    sys.exit(main())
