"""Study 2 pilot report (plan section 3, check P1). Pilot seeds only; not confirmatory.

    .venv/bin/python -m analysis.study2_pilot runs/study2-pilot

P1: the probe-turn reading (the reply to the fixed neutral question, on a discarded fork) must rise
by at least 0.5 SD from baseline to after the narrative, with a 95% interval above zero. The same
rise is shown for the reply reading and for neutral passages, plus mechanical checks and a few
probe-turn replies to read.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from analysis import confirmatory as C

P1_MIN_RISE = 0.5
CP = ("baseline", "post_induction", "post_regulation")


def load(run: Path, gate0_root: Path):
    """-> rows[model] = list of {arm, story, z: {reading: [3]}, probe_replies: [3], finish: [3]}."""
    out, probes = defaultdict(list), {}
    for line in (run / "results.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        m = r["model"]
        if m not in probes:
            probes[m] = C.load_probes(m, gate0_root)
        p, z = probes[m], np.load(run / r["acts_file"])
        zs = {"reply": [float(p["mb"].z(z[f"{c}__first64"][p["layer"]][None])[0]) for c in CP],
              "probe_turn": [float(p["mb"].z(z[f"{c}__probe"][p["layer"]][None])[0]) for c in CP]}
        out[m].append({"arm": r["arm"], "story": r["arm"].endswith("_story"), "z": zs,
                       "probe_replies": [r["checkpoints"][c]["probe_turn"]["reply"] for c in CP],
                       "tokens": [r["checkpoints"][c]["probe_turn"]["reply_tokens"] for c in CP],
                       "nonfinite": sum(r["checkpoints"][c]["probe_turn"]["nonfinite"]
                                        + r["checkpoints"][c].get("nonfinite_acts", 0) for c in CP)})
    return out


def rise(rows, reading: str, story: bool, rng) -> tuple[float, float, float, int]:
    d = np.array([r["z"][reading][1] - r["z"][reading][0] for r in rows if r["story"] == story])
    if len(d) == 0:
        return (np.nan, np.nan, np.nan, 0)
    bs = [rng.choice(d, len(d)).mean() for _ in range(4000)]
    return (float(d.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)), len(d))


def report(run: Path, gate0_root: Path) -> tuple[str, dict]:
    data, rng, verdicts = load(run, gate0_root), np.random.default_rng(0), {}
    md = ["# Study 2 pilot: check P1", "", f"Run `{run.name}`, pilot seeds, not confirmatory.", ""]
    for m, rows in data.items():
        md += [f"## {C.short(m)} ({len(rows)} conversations)", "",
               "| Reading | Rise after a narrative (SD) | 95% interval | n | Rise after a neutral passage (SD) | n |",
               "| --- | --- | --- | --- | --- | --- |"]
        for reading in ("reply", "probe_turn"):
            s, n = rise(rows, reading, True, rng), rise(rows, reading, False, rng)
            md.append(f"| {reading} | {s[0]:+.2f} | [{s[1]:+.2f}, {s[2]:+.2f}] | {s[3]} | {n[0]:+.2f} | {n[3]} |")
        s = rise(rows, "probe_turn", True, rng)
        ok = bool(s[0] >= P1_MIN_RISE and s[1] > 0)
        verdicts[m] = ok
        bad = sum(r["nonfinite"] for r in rows)
        short = sum(t < 8 for r in rows for t in r["tokens"])
        md += ["", f"**P1 {'PASS' if ok else 'FAIL'}** (needs a rise of at least {P1_MIN_RISE} SD with the "
               f"interval above zero).", "",
               f"Mechanics: {bad} non-finite values; {short} probe-turn replies shorter than 8 tokens; "
               f"median probe-turn reply {int(np.median([t for r in rows for t in r['tokens']]))} tokens.", "",
               "Probe-turn replies after a narrative (first three conversations):", ""]
        for r in [r for r in rows if r["story"]][:3]:
            md.append(f"- {r['probe_replies'][1][:220]}")
        md += ["", "Probe-turn replies at baseline (first two):", ""]
        md += [f"- {r['probe_replies'][0][:220]}" for r in rows[:2]] + [""]
    return "\n".join(md), verdicts


def choose(runs: list[Path], gate0_root: Path) -> str:
    """Selection rule, fixed before the candidate pilots ran (plan section 3b): candidates are taken in
    the order given (most resistant to echoing the story or the regulation text first); the first one
    that passes P1 and whose rise after a neutral passage is within +-0.25 SD is used."""
    rng, lines, chosen = np.random.default_rng(0), [], None
    for run in runs:
        for m, rows in load(run, gate0_root).items():
            s, n = rise(rows, "probe_turn", True, rng), rise(rows, "probe_turn", False, rng)
            ok = bool(s[0] >= P1_MIN_RISE and s[1] > 0 and abs(n[0]) <= 0.25)
            intro = np.mean([r["probe_replies"][0].lower().startswith(("okay", "sure", "here")) for r in rows])
            lines.append(f"| {run.name} | {C.short(m)} | {s[0]:+.2f} [{s[1]:+.2f}, {s[2]:+.2f}] | {n[0]:+.2f} | "
                         f"{intro:.0%} | {'pass' if ok else 'fail'} |")
            if ok and chosen is None:
                chosen = run.name
    head = ["| Candidate run | Model | Rise after a narrative (SD) | Rise after a neutral passage | "
            "Replies opening with filler words | Rule |", "| --- | --- | --- | --- | --- | --- |"]
    return "\n".join(head + lines + ["", f"**Chosen: {chosen or 'none passes; the first question stays and H-B stays secondary'}**"])


def main(argv=None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    if args and args[0] == "--choose":
        print(choose([Path(a) for a in args[1:]], C.ROOT / "runs"))
        return 0
    run = Path(args[0]) if args else C.ROOT / "runs" / "study2-pilot"
    if "pilot" not in run.name and not run.name.startswith("fake"):
        print("This report is for pilot runs only. The confirmatory run has its own analysis.")
        return 2
    md, verdicts = report(run, Path(args[1]) if len(args) > 1 else C.ROOT / "runs")
    (run / "pilot_p1.md").write_text(md + "\n")
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
