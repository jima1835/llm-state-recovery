"""LaTeX tables for Study 2, written from the analysis output (no numbers are typed by hand).

    .venv/bin/python -m analysis.study2_tables runs/study2 --batch A
    .venv/bin/python -m analysis.study2_tables runs/study2 --batch all

Reads <run>/analysis-batch-<b>/study2.json (registered analysis) and, if present,
<run>/robust-api-judge/analysis-batch-<b>/study2.json (second judge; only its H-E row is used, as the
addendum says). Writes paper/tables/study2_tests.tex and paper/tables/study2_specific.tex.
Reporting only: nothing here changes a test or a decision.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NAMES = {
    "gemma-3-4b-it-4bit": "Gemma 3 4B",
    "Llama-3.2-3B-Instruct-4bit": "Llama 3.2 3B",
    "Meta-Llama-3.1-8B-Instruct-4bit": "Llama 3.1 8B",
    "gemma-3-12b-it-4bit": "Gemma 3 12B",
    "Qwen3-14B-4bit": "Qwen3 14B",
    "Mistral-Small-24B-Instruct-2501-4bit": "Mistral Small 24B",
    "gemma-3-27b-it-4bit": "Gemma 3 27B",
}
TESTS = ("HA", "HC", "HD", "HE")
KINDS = (("relax", "Relaxation"), ("filler", "Filler"), ("calm", "Calm description"),
         ("voice", "Voice instruction"), ("report", "Report instruction"))


def name(model: str) -> str:
    short = model.split("/")[-1]
    return NAMES.get(short, short)


def num(x: float, nd: int = 2) -> str:
    s = f"{x:.{nd}f}"
    if s.startswith("-"):
        s = "$-$" + s[1:]
    return s


def ci(c) -> str:
    return f"[{num(c[0])}, {num(c[1])}]"


def cell(t: dict) -> str:
    s = num(t["estimate"])
    return f"\\textbf{{{s}}}" if t.get("confirmed") else s


def tests_table(reg: dict, rob: dict | None) -> str:
    models = list(reg["models"])
    cols = "l" + "r" * 7
    # Plain-language labels first, the plan's test codes beneath them.
    head = (" & Relaxation & Relaxation vs & Voice instruction & H-A, wording & H-A, wording & H-A on the & Probe-turn \\\\\n"
            " & vs filler & calm description & vs relaxation & removed & removed, second judge & probe turn & rise \\\\\n"
            "Model & (H-A) & (H-C) & (H-D) & (H-E) & (H-E) & (H-B) & \\\\")
    lines = ["\\begin{tabular}{" + cols + "}", "\\toprule", head, "\\midrule"]
    for m in models:
        r = reg["models"][m]
        t = r["tests"]
        he2 = num(rob["models"][m]["tests"]["HE"]["estimate"]) if rob and m in rob["models"] else "--"
        hb = cell(t["HB"]) if t["HB"].get("in_family") else num(t["HB"]["estimate"]) + "*"
        lines.append(" & ".join([name(m)] + [cell(t[k]) for k in TESTS] + [he2, hb, num(r["probe_rise"][0])])
                     + " \\\\")
        if m == models[0]:
            he2ci = ci(rob["models"][m]["tests"]["HE"]["ci"]) if rob and m in rob["models"] else ""
            lines.append(" & ".join(["\\quad 95\\% interval"] + [ci(t[k]["ci"]) for k in TESTS]
                                    + [he2ci, ci(t["HB"]["ci"]), ci(r["probe_rise"][1:3]) if len(r["probe_rise"]) >= 3 else ""])
                         + " \\\\")
    lines.append("\\midrule")
    a = reg["across"]
    row = ["Mean of models"] + [num(a[k]["mean"]) for k in TESTS]
    row += [num(sum(rob["models"][m]["tests"]["HE"]["estimate"] for m in models) / len(models)) if rob else "--"]
    row += [num(a["HB"]["mean"]), ""]
    lines.append(" & ".join(row) + " \\\\")
    lines.append(" & ".join(["\\quad interval, models resampled"] + [ci(a[k]["ci_models"]) for k in TESTS]
                            + ["", ci(a["HB"]["ci_models"]), ""]) + " \\\\")
    pos = ["\\quad models positive"] + [f"{a[k]['positive']} of {a[k]['n']}" for k in TESTS]
    pos += [f"{sum(rob['models'][m]['tests']['HE']['estimate'] > 0 for m in models)} of {len(models)}" if rob else "--"]
    pos += [f"{a['HB']['positive']} of {a['HB']['n']}", ""]
    lines.append(" & ".join(pos) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines) + "\n"


def specific_table(reg: dict) -> str:
    models = list(reg["models"])
    lines = ["\\begin{tabular}{l" + "r" * len(KINDS) + "r}", "\\toprule",
             "Model & " + " & ".join(k[1] for k in KINDS) + " & Rise after a narrative \\\\", "\\midrule"]
    for reading, label in (("reply", "Reply reading"), ("probe", "Probe-turn reading")):
        lines.append(f"\\multicolumn{{{len(KINDS) + 2}}}{{l}}{{\\emph{{{label}}}}} \\\\")
        for m in models:
            e = reg["models"][m]["extras"]
            vals = [num(e[f"specific_{reading}_{k}"]["estimate"]) for k, _ in KINDS]
            lines.append(" & ".join([name(m)] + vals + [num(e[f"rise_{reading}"]["estimate"])]) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", type=Path)
    ap.add_argument("--batch", default="A")
    ap.add_argument("--out", type=Path, default=ROOT / "paper" / "tables")
    args = ap.parse_args(argv)
    reg_f = args.run / f"analysis-batch-{args.batch}" / "study2.json"
    if not reg_f.exists():
        raise SystemExit(f"{reg_f} not found: run analysis.study2 first")
    reg = json.loads(reg_f.read_text())
    rob_f = args.run / "robust-api-judge" / f"analysis-batch-{args.batch}" / "study2.json"
    rob = json.loads(rob_f.read_text()) if rob_f.exists() else None
    args.out.mkdir(parents=True, exist_ok=True)
    note = f"% Written by analysis/study2_tables.py from {reg_f.relative_to(args.run.parent.parent) if args.run.is_absolute() else reg_f}; do not edit by hand.\n"
    (args.out / "study2_tests.tex").write_text(note + tests_table(reg, rob))
    (args.out / "study2_specific.tex").write_text(note + specific_table(reg))
    print(f"wrote {args.out / 'study2_tests.tex'} and {args.out / 'study2_specific.tex'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
