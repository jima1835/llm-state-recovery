"""Appendix tables and examples, written from the analysis output and the run files. No number or
reply is typed by hand.

    .venv/bin/python scripts/make_appendix.py

Writes to paper/tables/:
  app_models.tex          model IDs, the layer of the reading, conversations per study
  app_study1_models.tex   Study 1: the six registered tests per model and the size of the induction
  app_study2_drops.tex    Study 2: mean drop of the reply reading by kind of text, after a narrative
                          and after a neutral passage
  app_hd_posthoc.tex      Study 2, post hoc: H-D with each text's no-narrative baseline
  app_calm_posthoc.tex    Study 2, post hoc: calm description against filler
  app_texts.tex           the texts we wrote and the judge prompt, copied from the run's own protocol
  app_examples.tex        raw replies of Gemma 3 4B, chosen by a fixed seed (the same conversations
                          that scripts/show_replies.py prints first)
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from analysis import confirmatory as C  # noqa: E402  (only to score the example replies)

OUT = ROOT / "paper" / "tables"
RUN = ROOT / "runs" / "study2"
MODELS = {  # ordered by parameter count
    "mlx-community/Llama-3.2-3B-Instruct-4bit": "Llama 3.2 3B",
    "mlx-community/gemma-3-4b-it-4bit": "Gemma 3 4B",
    "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit": "Llama 3.1 8B",
    "mlx-community/gemma-3-12b-it-4bit": "Gemma 3 12B",
    "mlx-community/Qwen3-14B-4bit": "Qwen3 14B",
    "mlx-community/Mistral-Small-24B-Instruct-2501-4bit": "Mistral Small 24B",
    "mlx-community/gemma-3-27b-it-4bit": "Gemma 3 27B",
}
KINDS = (("relax", "Relaxation"), ("filler", "Filler"), ("calm", "Calm description"),
         ("voice", "Voice instruction"), ("report", "Report instruction"))
ASCII = {"\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"', "\u2026": "...", "\u2014": "---",
         "\u2013": "--", "\u00a0": " ", "\u2011": "-"}
SPECIAL = {"\\": r"\textbackslash{}", "{": r"\{", "}": r"\}", "$": r"\$", "&": r"\&", "#": r"\#", "_": r"\_",
           "%": r"\%", "^": r"\^{}", "~": r"\~{}"}


def num(x: float, d: int = 2) -> str:
    s = f"{x:.{d}f}"
    if float(s) == 0:
        s = s.lstrip("-")
    return ("$-$" + s[1:]) if s.startswith("-") else s


def tex(text: str) -> str:
    """Plain text -> LaTeX, ASCII only: typographic characters mapped, other non-ASCII dropped."""
    text = " ".join(text.split())
    text = "".join(ASCII.get(ch, ch) for ch in text)
    text = "".join(ch for ch in text if ord(ch) < 128)
    text = "".join(SPECIAL.get(ch, ch) for ch in text)
    parts = text.split('"')                     # straight double quotes -> ``opening and closing''
    return "".join(p + ("" if i == len(parts) - 1 else ("``" if i % 2 == 0 else "''")) for i, p in enumerate(parts))


def words(text: str, n: int) -> str:
    w = text.split()
    return " ".join(w[:n]) + (" [...]" if len(w) > n else "")


def write(name: str, source: str, body: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(f"% Written by scripts/make_appendix.py from {source}; do not edit by hand.\n{body}")
    print("wrote", (OUT / name).relative_to(ROOT))


def main() -> int:
    s1 = json.loads((ROOT / "runs" / "analysis-confirmatory" / "confirmatory.json").read_text())
    ext = next(v for k, v in s1["sets"].items() if k.startswith("Extended set"))
    s2 = json.loads((RUN / "analysis-batch-all" / "study2.json").read_text())
    protocol = json.loads((RUN / "protocol.json").read_text())["config"]
    assert set(MODELS) == set(s2["models"]) == set(ext["per_model"])

    # ---- models
    rows = [r"\begin{tabular}{lllrr}", r"\toprule",
            r"Model & File (Hugging Face ID) & Layer read & Study 1 & Study 2 \\", r"\midrule"]
    for m, name in MODELS.items():
        n1 = sum(s1["info"]["counts"][m].values())
        rows.append(f"{name} & \\texttt{{{tex(m)}}} & {s1['info']['layers'][m]} & {n1} & 600 \\\\")
    rows += [r"\bottomrule", r"\end{tabular}"]
    write("app_models.tex", "runs/analysis-confirmatory/confirmatory.json", "\n".join(rows) + "\n")

    # ---- Study 1 per model
    rows = [r"\begin{tabular}{lrrrrrrrr}", r"\toprule",
            r"Model & H1a & H1b & H2a & H2b & Return a & Return b & Rise, reading & Rise, questionnaire \\",
            r"\midrule"]
    for m, name in MODELS.items():
        v = ext["per_model"][m]
        rows.append(f"{name} & " + " & ".join(num(v[f"T{i}"]) for i in range(1, 7))
                    + f" & {num(v['E_int'])} & {num(v['E_rep'], 1)} \\\\")
    rows += [r"\midrule", "Mean of seven & " + " & ".join(num(ext["tests"][f"T{i}"]["estimate"]) for i in range(1, 7))
             + f" & {num(ext['extras']['E_int']['estimate'])} & {num(ext['extras']['E_rep']['estimate'], 1)} \\\\",
             r"\bottomrule", r"\end{tabular}"]
    write("app_study1_models.tex", "runs/analysis-confirmatory/confirmatory.json", "\n".join(rows) + "\n")

    # ---- Study 2 drops by arm
    rows = [r"\begin{tabular}{l" + "rr" * len(KINDS) + "}", r"\toprule",
            " & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{lab}}}" for _, lab in KINDS) + r" \\",
            "Model" + " & narr. & neutral" * len(KINDS) + r" \\", r"\midrule"]
    for m, name in MODELS.items():
        e = s2["models"][m]["extras"]
        rows.append(name + "".join(f" & {num(e[f'drop_reply_{k}_story']['estimate'])} & "
                                   f"{num(e[f'drop_reply_{k}_neutral']['estimate'])}" for k, _ in KINDS) + r" \\")
    rows += [r"\bottomrule", r"\end{tabular}"]
    write("app_study2_drops.tex", "runs/study2/analysis-batch-all/study2.json", "\n".join(rows) + "\n")

    # ---- post hoc H-D with a baseline
    hd_file = RUN / "analysis-batch-all" / "posthoc_hd.json"
    if hd_file.exists():
        hd = json.loads(hd_file.read_text())
        rows = [r"\begin{tabular}{lrlr}", r"\toprule",
                r"Model & Relaxation $-$ voice instruction & 95\% interval & Pre-specified H-D \\", r"\midrule"]
        for m, name in MODELS.items():
            h = hd["models"][m]
            rows.append(f"{name} & {num(h['estimate'])} & [{num(h['ci'][0])}, {num(h['ci'][1])}] & "
                        f"{num(s2['models'][m]['tests']['HD']['estimate'])} \\\\")
        rows += [r"\midrule", f"Mean of seven & {num(hd['mean'])} & [{num(hd['ci_models'][0])}, {num(hd['ci_models'][1])}] & "
                 f"{num(s2['across']['HD']['mean'])} \\\\", r"\bottomrule", r"\end{tabular}"]
        write("app_hd_posthoc.tex", "runs/study2/analysis-batch-all/posthoc_hd.json", "\n".join(rows) + "\n")
    else:
        print("skipped app_hd_posthoc.tex: run analysis/study2_posthoc_hd.py first")

    # ---- post hoc: calm description against filler
    calm_file = RUN / "analysis-batch-all" / "posthoc_calm.json"
    if calm_file.exists():
        cf = json.loads(calm_file.read_text())
        rows = [r"\begin{tabular}{lrl}", r"\toprule",
                r"Model & Calm description $-$ filler & 95\% interval \\", r"\midrule"]
        for m, name in MODELS.items():
            h = cf["models"][m]
            rows.append(f"{name} & {num(h['estimate'])} & [{num(h['ci'][0])}, {num(h['ci'][1])}] \\\\")
        rows += [r"\midrule", f"Mean of seven & {num(cf['mean'])} & [{num(cf['ci_models'][0])}, {num(cf['ci_models'][1])}] \\\\",
                 r"\bottomrule", r"\end{tabular}"]
        write("app_calm_posthoc.tex", "runs/study2/analysis-batch-all/posthoc_calm.json", "\n".join(rows) + "\n")
    else:
        print("skipped app_calm_posthoc.tex: run analysis/study2_posthoc_calm.py first")

    # ---- the texts we wrote
    t = protocol["texts"]
    items = [("Voice instruction", "voice_instruction"), ("Report instruction", "report_instruction"),
             ("Unrelated question (probe turn)", "probe_question"), ("Calm description 1", "calm_1"),
             ("Calm description 2", "calm_2"), ("Calm description 3", "calm_3"), ("Filler 2", "filler_2"),
             ("Filler 3", "filler_3"), ("Neutral passage 2", "neutral_2"), ("Neutral passage 3", "neutral_3"),
             ("Judge prompt", "judge_prompt")]
    body = "\\begin{description}\\small\n" + "".join(
        f"\\item[{lab}] {tex(t[key])}\n" for lab, key in items if key in t) + "\\end{description}\n"
    write("app_texts.tex", "runs/study2/protocol.json", body)

    # ---- raw replies, Gemma 3 4B, fixed seed (same choice as scripts/show_replies.py)
    model = "mlx-community/gemma-3-4b-it-4bit"
    probe = C.load_probes(model, ROOT / "runs")
    allrows = [json.loads(l) for l in (RUN / "results.jsonl").read_text().splitlines() if l.strip()]
    cps = ("baseline", "post_induction", "post_regulation")
    blocks = []
    for arm, label in (("relax_story", "Relaxation after a narrative"),
                       ("relax_neutral", "Relaxation after a neutral passage"),
                       ("filler_story", "Filler after a narrative"),
                       ("voice_story", "Voice instruction after a narrative")):
        pick = [r for r in allrows if "gemma-3-4b" in r["model"].lower() and r["arm"] == arm]
        random.Random(0).shuffle(pick)
        r = pick[0]
        z = np.load(RUN / r["acts_file"])
        reading = [float(probe["mb"].z(z[f"{c}__first64"][probe["layer"]][None])[0]) for c in cps]
        total = [r["checkpoints"][c]["stai"]["total_expected"] for c in cps]
        replies = [m["content"] for m in r["transcript"] if m["role"] == "assistant"]
        f = r["factors"]
        # plain names for the texts, induction turn first ("literal:filler_3" -> "filler 3")
        names = {"trauma": "narrative", "neutral": "neutral passage", "relax": "relaxation script", "filler": "filler"}
        order = ["trauma", "neutral", "relax", "filler"]
        cues = {k.replace("_cue", ""): str(v).split(":")[-1].replace("neutral_", "").replace("filler_", "").replace("_", " ")
                for k, v in f.items() if k.endswith("_cue")}
        cue = ", ".join(f"{names.get(k, k)} {cues[k]}" for k in sorted(cues, key=lambda k: order.index(k) if k in order else 9))
        blocks.append(
            # ragged right: the conversation id cannot be hyphenated and would run into the margin
            f"\\paragraph{{{label}}} {{\\raggedright ({tex(cue)}; seed {r['seed']}; conversation \\texttt{{{r['cell_id']}}}). "
            f"Reply reading at the three checkpoints: {num(reading[0])}, {num(reading[1])}, {num(reading[2])}. "
            f"Questionnaire total: {total[0]:.1f}, {total[1]:.1f}, {total[2]:.1f}.\\par}}\n"
            f"\\begin{{quote}}\\small\n\\textbf{{Reply to the induction turn.}} {tex(words(replies[1], 45))}\n\n"
            f"\\textbf{{Reply to the regulation turn.}} {tex(words(replies[2], 70))}\n\\end{{quote}}\n")
    write("app_examples.tex", "runs/study2/results.jsonl and the stored activations", "\n".join(blocks))
    return 0


if __name__ == "__main__":
    sys.exit(main())
