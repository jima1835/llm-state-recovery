"""Number check: every Study 2 number in the abstract, introduction, Section 5 and Discussion is
recomputed from the analysis output and must appear in the text as printed. Deterministic, offline.
Section 5 is the home of every number; the abstract, introduction and Discussion repeat a few anchors,
and each anchor they print is required there too.

    .venv/bin/python scripts/number_check.py

Each line below is one claim: a value computed here from the output files, formatted as the paper prints
it, and the files in which that string must occur. A number that is in the text but was edited by hand
away from its source fails here. It does not check Sections 3 and 4 (Study 1 and the validation), whose
numbers were checked against their reports on 15 to 18 November (TRACKING.md).
Exit code 1 on any failure.
"""
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RUN = ROOT / "runs" / "study2"
A = RUN / "analysis-batch-all"
reg = json.loads((A / "study2.json").read_text())
rob = json.loads((RUN / "robust-api-judge" / "analysis-batch-all" / "study2.json").read_text())
hd = json.loads((A / "posthoc_hd.json").read_text())
calm = json.loads((A / "posthoc_calm.json").read_text())

G4, L32, L31, G12, QW, MS, G27 = (
    "mlx-community/gemma-3-4b-it-4bit", "mlx-community/Llama-3.2-3B-Instruct-4bit",
    "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit", "mlx-community/gemma-3-12b-it-4bit",
    "mlx-community/Qwen3-14B-4bit", "mlx-community/Mistral-Small-24B-Instruct-2501-4bit",
    "mlx-community/gemma-3-27b-it-4bit")
MODELS = [G4, L32, L31, G12, QW, MS, G27]


def text(name):
    """A source file as the reader sees its numbers: comments, $ and backslash-percent removed."""
    s = (ROOT / "paper" / name).read_text()
    s = re.sub(r"(?<!\\)%.*", "", s)
    return " ".join(s.replace("$", "").replace("\\%", "%").split())


main_tex = text("main.tex")
i, j = main_tex.index("\\begin{abstract}"), main_tex.index("\\end{abstract}")
k, l = main_tex.index("\\section{Discussion"), main_tex.index("\\section{Related work}")
T = {"abstract": main_tex[i:j], "discussion": main_tex[k:l], "limits": main_tex[l:],
     "intro": text("sections/01_intro_draft0.tex"), "s5": text("sections/05_study2.tex")}

f2 = lambda x: f"{x:.2f}".replace("-0.00", "0.00")
f1 = lambda x: f"{x:.1f}"
iv = lambda c, f=f2: f"[{f(c[0])}, {f(c[1])}]"
test = lambda m, t, r=reg: r["models"][m]["tests"][t]
ex = lambda m, key: reg["models"][m]["extras"][key]["estimate"]
fails, n = [], 0


def need(what, string, *where):
    global n
    for w in where:
        n += 1
        if string not in T[w]:
            fails.append(f"{what}: expected '{string}' in {w}")


# ---- H-A
ha = reg["across"]["HA"]
need("H-A mean", f2(ha["mean"]) + " SD units", "intro", "s5", "discussion")
need("H-A mean, abstract", f2(ha["mean"]) + " standard deviations", "abstract")
need("H-A interval, lower", f2(ha["ci_models"][0]) + " to " + f2(ha["ci_models"][1]), "abstract", "intro", "s5")
need("H-A upper bound as 'at most'", f2(ha["ci_models"][1]) + " SD units", "limits")
need("H-A positive count", {5: "five of seven"}[ha["positive"]], "s5")
t = test(G4, "HA")
need("H-A Gemma 3 4B", f"{f2(t['estimate'])} {iv(t['ci'])}", "s5")
need("H-A Gemma 3 4B p", f"{t['p']:.3f}", "s5"); need("H-A Gemma 3 4B corrected p", f"{t['p_bh']:.3f}", "s5")
others = [test(m, "HA")["estimate"] for m in MODELS if m != G4]
need("H-A other six, range", f"{f2(min(others))} to {f2(max(others))}", "s5")
assert sum(test(m, "HA")["confirmed"] for m in MODELS) == 1 and test(G4, "HA")["confirmed"]
need("H-A mixed model", f"{f2(reg['mixed']['HA']['estimate'])} (standard error {f2(reg['mixed']['HA']['se'])})", "s5")

# ---- H-C
t = test(G4, "HC")
need("H-C Gemma 3 4B", f"{f2(t['estimate'])} {iv(t['ci'])}", "s5")
hc = sorted(test(m, "HC")["estimate"] for m in MODELS if m not in (G4, G27))
need("H-C five negative, range", f"{f2(hc[0])} to {f2(hc[-1])}", "s5"); assert hc[-1] < 0
need("H-C Gemma 3 27B", f"({f2(test(G27, 'HC')['estimate'])})", "s5")

# ---- H-D, pre-specified and post hoc
d = sorted(abs(test(m, "HD")["estimate"]) for m in MODELS)
need("H-D range", f"{f2(d[0])} to {f2(d[-1])} SD", "s5")
assert all(test(m, "HD")["estimate"] < 0 for m in MODELS) and sum(test(m, "HD")["confirmed"] for m in MODELS) == 5
need("H-D confirmed in five", "confirmed in five models", "s5")
need("H-D interval", f"{f2(reg['across']['HD']['ci_models'][0])} to {f2(reg['across']['HD']['ci_models'][1])}", "s5")
need("H-D post hoc mean", f2(hd["mean"]) + " SD units", "s5")
need("H-D post hoc interval", f"{f2(hd['ci_models'][0])} to {f2(hd['ci_models'][1])}", "s5")
assert hd["positive"] == 5
above = sorted(v["estimate"] for v in hd["models"].values() if v["ci"][0] > 0)
assert len(above) == 4
need("H-D post hoc, four above zero", f"({f2(above[0])} to {f2(above[-1])})", "s5")
need("H-D post hoc Gemma 3 12B", f"({f2(hd['models'][G12]['estimate'])})", "s5")
need("H-D post hoc Qwen3 14B", f"({f2(hd['models'][QW]['estimate'])})", "s5")
need("H-D post hoc Gemma 3 4B", f"({f2(hd['models'][G4]['estimate'])})", "discussion")

# ---- H-E, both judges
need("H-E registered, Gemma 3 4B", f"({f2(test(G4, 'HE')['estimate'])})", "s5")
need("H-E registered, Llama 3.1 8B", f"({f2(test(L31, 'HE')['estimate'])})", "s5")
assert {m for m in MODELS if test(m, "HE")["confirmed"]} == {G4, L31}
for m, lab in ((G4, "Gemma 3 4B"), (L31, "Llama 3.1 8B"), (G27, "Gemma 3 27B")):
    t = test(m, "HE", rob)
    need(f"H-E second judge, {lab}", f"{f2(t['estimate'])} {iv(t['ci'])}", "s5")
assert [m for m in MODELS if test(m, "HE", rob)["ci"][0] > 0] == [G27]
t = test(G27, "HE")
need("H-E registered, Gemma 3 27B", f"{f2(t['estimate'])} {iv(t['ci'])}", "s5")
for r, lab in ((reg, "registered"), (rob, "second")):
    a = r["across"]["HE"]
    need(f"H-E mean, {lab} judge", f"{f2(a['mean'])} {iv(a['ci_models'])}", "s5")
ja = {(json.loads(x)["model"], json.loads(x)["cell_id"]): json.loads(x)["scores"] for x in (RUN / "judge.jsonl").read_text().splitlines() if x.strip()}
jb = {(json.loads(x)["model"], json.loads(x)["cell_id"]): json.loads(x)["scores"] for x in (RUN / "judge_api.jsonl").read_text().splitlines() if x.strip()}
rs = []
for m in MODELS:
    xs = [(ja[key][c], jb[key][c]) for key in ja if key[0] == m for c in ja[key]]
    rs.append(float(np.corrcoef(*zip(*xs))[0, 1]))
need("judge agreement", f"{f2(min(rs))} to {f2(max(rs))}", "s5")

# ---- probe turn and rises
pr = sorted(reg["models"][m]["probe_rise"][0] for m in MODELS if m != G27)
need("probe-turn rise, six models", f"{f2(pr[0])} to {f2(pr[-1])} SD units", "s5", "discussion")
need("probe-turn rise, Gemma 3 27B", f"by {f2(abs(reg['models'][G27]['probe_rise'][0]))}", "s5")
rise = sorted(ex(m, "rise_reply") for m in MODELS)
need("rise on the reply itself", f"{f2(rise[0])} to {f2(rise[-1])}", "s5")
need("rise, in the summary sections", f"{f2(rise[0])} to {f2(rise[-1])}", "intro", "discussion")
for m in (G4, MS):
    t = test(m, "HB"); assert reg["models"][m]["p1"]
    need("H-B in family", f"{f2(t['estimate'])} {iv(t['ci'])}", "s5")
assert sum(reg["models"][m]["p1"] for m in MODELS) == 2
a = reg["across"]["HB"]
need("H-B mean", f"{f2(a['mean'])} {iv(a['ci_models'])}", "s5")

# ---- drops with no narrative, and the gap shares
rn = {m: ex(m, "drop_reply_relax_neutral") for m in MODELS}
six = sorted(v for m, v in rn.items() if m != G12)
need("relaxation after a neutral passage, six models", f"{f2(six[0])} to {f2(six[-1])} SD units", "intro", "s5")
need("relaxation after a neutral passage, abstract", f"{f2(six[0])} to {f2(six[-1])} standard deviations", "abstract")
need("relaxation after a neutral passage, Gemma 3 12B", f"({f2(rn[G12])}", "s5")
fn = {m: ex(m, "drop_reply_filler_neutral") for m in MODELS}
flat = [v for v in fn.values() if abs(v) < 0.1]; up = sorted(-v for v in fn.values() if v <= -0.1)
assert len(flat) == 2 and len(up) == 5
need("filler after a neutral passage, flat", f"within {f2(max(abs(v) for v in flat))}", "s5")
need("filler after a neutral passage, rises", f"{f2(up[0])} to {f2(up[-1])}", "s5")
share = {m: (rn[m] - fn[m]) / (ex(m, "drop_reply_relax_story") - ex(m, "drop_reply_filler_story")) for m in MODELS}
need("gap share, Gemma 3 4B", f"{share[G4]:.0%}", "s5")
need("gap share, other six at least", f"{min(v for m, v in share.items() if m != G4):.0%} or more", "s5")
fs = sorted(ex(m, "specific_reply_filler") / ex(m, "rise_reply") for m in MODELS)
need("filler undoes share of the rise", f"{fs[0]:.0%}".rstrip("%") + f" to {fs[-1]:.0%}", "s5")
fsm = {m: ex(m, "specific_reply_filler") / ex(m, "rise_reply") for m in MODELS}
assert min(fsm, key=fsm.get) == G4
need("filler share, validated model", f"{fsm[G4]:.0%} in the validated model", "s5")
need("filler share, other six at least", f"{min(v for m, v in fsm.items() if m != G4):.0%} or more in the other six", "s5")
assert sum(ex(m, "specific_reply_voice") < ex(m, "specific_reply_filler") for m in MODELS) == 6

# ---- questionnaire (descriptive)
qa = {m: reg["models"][m]["extras"]["A_stai"] for m in MODELS}
six = sorted(v["estimate"] for m, v in qa.items() if m != MS)
need("questionnaire contrast, six models", f"{f1(six[0])} and {f1(six[-1])} points", "s5")
need("questionnaire contrast, Mistral Small 24B", f"{f1(qa[MS]['estimate'])} points {iv(qa[MS]['ci'], f1)}", "s5")
need("questionnaire contrast bound", f"within {f1(max(abs(v) for v in six))} points", "discussion")
qr = sorted(ex(m, "rise_stai") for m in MODELS)
need("questionnaire rise", f"{qr[0]:.0f} to {qr[-1]:.0f}", "s5", "discussion")

# ---- post hoc: calm description against filler
need("calm vs filler mean", f2(calm["mean"]) + " SD units", "s5")
need("calm vs filler interval", f"{f2(calm['ci_models'][0])} to {f2(calm['ci_models'][1])}", "s5")
assert calm["positive"] == 5
above = sorted(v["estimate"] for v in calm["models"].values() if v["ci"][0] > 0)
assert len(above) == 3
need("calm vs filler, three above zero", f"{f2(above[0])} to {f2(above[-1])}", "s5")
c = calm["models"][G4]
need("calm vs filler, Gemma 3 4B", f"{f2(c['estimate'])} {iv(c['ci'])}", "s5")
# the Discussion says the calm description exceeds filler in five of the other six models
assert sum(v["estimate"] > 0 for m, v in calm["models"].items() if m != G4) == 5
# the Discussion and introduction say filler raises the reading after a neutral passage in five models (asserted above)
need("H-A Gemma 3 4B, Discussion", f"({f2(test(G4, 'HA')['estimate'])}, confirmed", "discussion")
need("H-C Gemma 3 4B, Discussion", f"({f2(test(G4, 'HC')['estimate'])})", "discussion")
need("H-A upper bound, Discussion", f"reaches {f2(ha['ci_models'][1])} SD units", "discussion")

# ---- counts
n1 = json.loads((ROOT / "runs" / "analysis-confirmatory" / "confirmatory.json").read_text())["info"]["n_rows"]
n2 = sum(1 for x in (RUN / "results.jsonl").read_text().splitlines() if x.strip())
need("conversations in both studies", f"{n1 + n2:,}", "intro")

print(f"{n} checks, {len(fails)} failed")
for f in fails:
    print("FAIL:", f)
print("RESULT:", "FAIL" if fails else "PASS")
sys.exit(1 if fails else 0)
