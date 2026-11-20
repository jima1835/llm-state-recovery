"""Build and audit the data record (Zenodo) from the local run folders.

    .venv/bin/python scripts/export_release.py                      # build release/, run checks 1-4 and 6
    .venv/bin/python scripts/export_release.py --tar                # also write release.tar.gz and its MD5
    .venv/bin/python scripts/export_release.py --acceptance         # also check 5: a clean tree + the export
    .venv/bin/python scripts/export_release.py --check-only --acceptance --rerun-analysis

The record unpacks at the repository root: everything is under runs/ with the layout the scripts and
the README commands expect, plus DATA_README.md, data-licenses/ and MANIFEST.md5.

What goes in (tiers 2 and 3 of the release plan):
  Study 1  runs/wave1, runs/wave2 (registered, 1,365 conversations), runs/wave2-8bit (195, 8-bit copy):
           results.jsonl, judge_api.jsonl, protocol.json, readings.csv, and acts/<id>.npz per conversation
  Study 2  runs/study2 (4,200 conversations): results.jsonl, judge.jsonl, judge_api.jsonl, protocol.json,
           readings.csv, acts/<id>.npz, analysis-batch-A/ and analysis-batch-all/, and the outputs of the
           second-judge view robust-api-judge/analysis-batch-*/ (the view's links are rebuilt by
           analysis/study2_he_robust.py on the reader's machine)
  Study 1  outputs runs/analysis-confirmatory, -sensitivity, -wording, -wording-8bit
  Gate 0   runs/gate0-<model>/: reports, protocol, replies (texts.jsonl), judge and steering files; for
           every model used in Study 1 or 2 also acts/pole_train.npz and acts/pole_select.npz (all
           layers: the layer is reselected from them) and direction.npz (the registered direction)

A conversation's acts/<id>.npz keeps the file name and array shapes of the original, so the registered
analysis and scripts/recompute_reading.py read it unchanged, but holds the registered layer only:
every other layer is NaN. Code that reads a different layer gets NaN and excludes the conversation
loudly instead of producing a number. Values at the registered layer are the original float32.

What stays out: the questionnaire's item text (never written to results.jsonl: the questionnaire runs
on a discarded copy and only its scores are stored), run logs (they contain local paths), the other
layers' activations (about 30 GB), the Gate 0 test-set activations, pilots, and the links in
robust-api-judge/.

Checks, each printed PASS or FAIL:
  1. no questionnaire item text: no item in any field the model did not write, no string with three or
     more items, no stretch of the preamble (needs the local licensed copy under third_party/)
  2. no secrets, home-directory paths, user or machine name, own email address or red-line terms
  3. no file from third_party/ (byte-identical or a 10-word stretch of Wekenborg's texts), except the
     Ben-Zion MIT licence shipped in data-licenses/
  4. row counts: 1,365 + 195 Study 1 and 4,200 Study 2 conversations, seven models x 600, judges and
     readings complete, every activation file present
  5. (--acceptance) a clean tree of the repository with only the export unpacked: scripts/check_all.sh
     prints ALL CHECKS PASS and scripts/show_replies.py runs; with --rerun-analysis the README's four
     analysis commands regenerate the exported reports
  6. size and MANIFEST.md5 (MD5, bytes, path per file: compare with the Zenodo draft's file list)
Exit code 1 if any check fails.
"""
from __future__ import annotations

import argparse
import csv
import getpass
import hashlib
import importlib.util
import io
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import tarfile
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from analysis import confirmatory as C  # noqa: E402  (registered code, used read-only to fit the directions)

RUNS = ROOT / "runs"
STUDY1 = ("wave1", "wave2", "wave2-8bit")
STUDY2 = "study2"
ANALYSIS_DIRS = ("analysis-confirmatory", "analysis-sensitivity", "analysis-wording", "analysis-wording-8bit")
RUN_FILES = ("results.jsonl", "judge.jsonl", "judge_api.jsonl", "protocol.json")
GATE0_SKIP_SUFFIX = (".log",)
POLE_FILES = ("pole_train.npz", "pole_select.npz")
EXPECT = {"wave1+wave2": 1365, "wave2-8bit": 195, "study2": 4200, "study2_models": 7, "study2_per_model": 600}
THIRD = ROOT / "third_party"
STAI_JSON = THIRD / "gpt-trauma-induction" / "src" / "STAI" / "questionnaires.json"
BENZION_LICENSE = THIRD / "gpt-trauma-induction" / "LICENSE"
WEKENBORG = THIRD / "LLMAffect"

results_log: list[tuple[str, bool, str]] = []


def report(name: str, ok: bool, detail: str = "") -> None:
    results_log.append((name, ok, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"\n       {detail}" if detail else ""), flush=True)


def jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def norm(s: str) -> str:
    s = s.lower().replace("\u2019", "'")
    s = re.sub(r"[^a-z0-9' ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


# ---------------------------------------------------------------- build

def layer_only(z: dict, dst: Path, layer: int) -> None:
    """Write one conversation's activations, keeping `layer` and setting every other layer to NaN."""
    out = {}
    for k, a in z.items():
        if a.dtype.kind not in "iuf":
            raise SystemExit(f"{dst.name}: unexpected non-numeric array {k}")
        if k == "layers" or a.ndim != 2:
            out[k] = a
            continue
        b = np.full(a.shape, np.nan, dtype=a.dtype)
        b[layer] = a[layer]
        out[k] = b
    dst.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dst, **out)


def copy(src: Path, dst: Path) -> None:
    if src.is_symlink():
        raise SystemExit(f"refusing to export a link: {src.relative_to(ROOT)}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def judges(run: Path) -> dict:
    return {name: {j["cell_id"]: j["scores"] for j in jsonl(run / name)}
            for name in ("judge.jsonl", "judge_api.jsonl") if (run / name).exists()}


def readings_rows(r: dict, z, p: dict, judge: dict, study2: bool) -> list[dict]:
    """The registered reading (and the whole-reply and probe readings) at each checkpoint of one conversation."""
    out, L = [], p["layer"]
    for c, cp in r["checkpoints"].items():
        rec = {"model": r["model"], "cell_id": r["cell_id"], "arm": r["arm"], "checkpoint": c, "layer": L,
               "reply_reading": float(p["mb"].z(z[f"{c}__first64"][L][None])[0]),
               "whole_reply_reading": float(p["mb"].z(z[c][L][None])[0])}
        if study2:
            rec["probe_reading"] = float(p["mb"].z(z[f"{c}__probe"][L][None])[0])
        st = cp.get("stai", {})
        rec["stai_total_expected"] = st.get("total_expected")
        rec["stai_total_argmax"] = st.get("total_argmax")
        rec["vas"] = cp.get("vas", {}).get("value")
        rec["judge_local"] = (judge.get("judge.jsonl", {}).get(r["cell_id"]) or {}).get(c)
        rec["judge_api"] = (judge.get("judge_api.jsonl", {}).get(r["cell_id"]) or {}).get(c)
        out.append(rec)
    return out


def write_csv(path: Path, rows: list[dict]) -> None:
    cols = list(rows[0])
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else (f"{r[k]:.10g}" if isinstance(r[k], float) else r[k]))
                        for k in cols})


def build(out: Path) -> None:
    if out.exists():
        shutil.rmtree(out)
    (out / "runs").mkdir(parents=True)
    probes: dict = {}
    models_used: set[str] = set()

    for name in STUDY1 + (STUDY2,):
        run, dst = RUNS / name, out / "runs" / name
        rows = jsonl(run / "results.jsonl")
        for f in RUN_FILES:
            if (run / f).exists():
                copy(run / f, dst / f)
        for r in rows:
            m = r["model"]
            models_used.add(m)
            if m not in probes:
                probes[m] = C.load_probes(m, RUNS)
        print(f"  {name}: {len(rows)} conversations, writing the registered layer of each activation file", flush=True)
        judge, readings = judges(run), []
        for i, r in enumerate(rows):
            p = probes[r["model"]]
            with np.load(run / r["acts_file"]) as zf:
                z = {k: zf[k] for k in zf.files}
            layer_only(z, dst / r["acts_file"], p["layer"])
            readings += readings_rows(r, z, p, judge, name == STUDY2)
            if (i + 1) % 600 == 0:
                print(f"    {i + 1} of {len(rows)}", flush=True)
        write_csv(dst / "readings.csv", readings)
        if name == STUDY2:
            for sub in ("analysis-batch-A", "analysis-batch-all"):
                for f in sorted((run / sub).glob("*")):
                    copy(f, dst / sub / f.name)
                for f in sorted((run / "robust-api-judge" / sub).glob("*")):
                    copy(f, dst / "robust-api-judge" / sub / f.name)

    for name in ANALYSIS_DIRS:
        for f in sorted((RUNS / name).glob("*")):
            copy(f, out / "runs" / name / f.name)

    used_short = {C.short(m) for m in models_used}
    for g in sorted(RUNS.glob("gate0-*")):
        if not g.is_dir():
            continue
        dst = out / "runs" / g.name
        for f in sorted(g.iterdir()):
            if f.is_file() and not f.name.endswith(GATE0_SKIP_SUFFIX):
                copy(f, dst / f.name)
        short = g.name[len("gate0-"):]
        if short in used_short:
            for f in POLE_FILES:
                copy(g / "acts" / f, dst / "acts" / f)
            m = next(m for m in models_used if C.short(m) == short)
            p = probes[m]
            np.savez_compressed(dst / "direction.npz", layer=p["layer"],
                                **{f"{k}_{a}": np.asarray(getattr(p[k], a)) for k in ("mb", "mean")
                                   for a in ("w", "mid", "scale")})

    (out / "data-licenses").mkdir()
    shutil.copy2(BENZION_LICENSE, out / "data-licenses" / "benzion-gpt-trauma-induction-MIT.txt")
    (out / "DATA_README.md").write_text(DATA_README)


DATA_README = """# Data record: LLM state recovery

Data for "Take a Deep Breath? Relaxation Prompts Calm Language-Model Replies With or Without Prior
Anxiety Induction" (Jiajun Ma, 2026). Code: https://github.com/jima1835/llm-state-recovery.

**Content note.** The induction turns are Ben-Zion et al.'s narratives of accidents, disasters,
violence and combat, and many replies express fear in the first person.

## Use

Unpack at the root of a clone of the code repository; everything lands under `runs/`. Then
`scripts/check_all.sh` reruns every check and `python3 scripts/show_replies.py` prints conversations
(see the repository README).

## Layout

| Path | Contents |
| --- | --- |
| `runs/wave1/`, `runs/wave2/` | Study 1 (registered): 1,365 conversations |
| `runs/wave2-8bit/` | Study 1, Gemma 3 4B at 8 bits: 195 conversations |
| `runs/study2/` | Study 2 (pre-specified): 4,200 conversations, seven models x 600 |
| `runs/<run>/results.jsonl` | one line per conversation: arm and factors, seed, the full transcript (system prompt, user turns, replies), per-step token counts, at each checkpoint the questionnaire scores (total and per-item probabilities, item numbers only), the 0-100 rating and, in Study 2, the answer to the unrelated question |
| `runs/<run>/judge.jsonl`, `judge_api.jsonl` | the local judge's (Llama 3.1 8B) and the second judge's ratings of each reply |
| `runs/<run>/readings.csv` | per conversation and checkpoint: the registered reply reading, the whole-reply reading, in Study 2 the reading on the unrelated question, questionnaire totals, rating, both judges |
| `runs/<run>/acts/<id>.npz` | per conversation, residual-stream means at the registered layer (first 64 reply tokens, whole reply, and in Study 2 the unrelated question); arrays keep every layer's row, all NaN except the registered layer |
| `runs/<run>/protocol.json` | the run's configuration and protocol hash |
| `runs/study2/analysis-*/`, `runs/analysis-*/` | the analysis reports the paper cites |
| `runs/gate0-<model>/` | Gate 0 validation: report, replies, judge and steering files; for the models in Study 1 or 2, `acts/pole_train.npz` and `acts/pole_select.npz` (all layers) and `direction.npz` (layer, unit direction, midpoint and scale) |
| `MANIFEST.md5` | MD5, size in bytes and path of every other file |

Not included: the questionnaire's item text (the STAI is copyrighted by Mind Garden; only scores and
item numbers are stored), activations at other layers (about 30 GB; recomputable with one forward pass
per reply), run logs and pilots.

## Licences

- Our data, readings and analysis outputs: the licence of this Zenodo record.
- The narratives, relaxation scripts and filler texts inside the transcripts are from Ben-Zion et al.
  (2025), https://github.com/akjagadish/gpt-trauma-induction, under the MIT licence
  (`data-licenses/benzion-gpt-trauma-induction-MIT.txt`).
- Model replies were generated by Gemma 3 (Gemma Terms of Use), Llama 3.1 and 3.2 (Llama Community
  Licence; built with Llama), Qwen3 and Mistral Small (Apache 2.0) and, for the second judge's ratings,
  an OpenAI model (OpenAI terms of use). Their use is subject to those terms.
"""


# ---------------------------------------------------------------- checks

def exported_files(out: Path) -> list[Path]:
    return sorted(p for p in out.rglob("*") if p.is_file())


def text_payloads(path: Path):
    """Yield (label, object) for the text inside a file: parsed JSON, or a str. NPZ: their string arrays."""
    if path.suffix == ".npz":
        import zipfile
        with zipfile.ZipFile(path) as zf:
            text_keys = []
            for name in zf.namelist():
                with zf.open(name) as fh:              # read the array header only, not the data
                    version = np.lib.format.read_magic(fh)
                    if version == (1, 0):
                        _, _, dtype = np.lib.format.read_array_header_1_0(fh)
                    elif version == (2, 0):
                        _, _, dtype = np.lib.format.read_array_header_2_0(fh)
                    else:                              # rare: read the array to learn its type
                        dtype = np.load(path)[name[:-4]].dtype
                if dtype.kind in "USO":
                    text_keys.append(name[:-4] if name.endswith(".npy") else name)
        if text_keys:
            z = np.load(path)
            for k in text_keys:
                s = str(z[k])
                try:
                    yield f"{k}", json.loads(s)
                except json.JSONDecodeError:
                    yield f"{k}", s
        return
    raw = path.read_text(errors="replace")
    if path.suffix == ".jsonl":
        for i, line in enumerate(raw.splitlines()):
            if line.strip():
                yield f"line {i + 1}", json.loads(line)
    elif path.suffix == ".json":
        yield "", json.loads(raw)
    else:
        yield "", raw


def strings(o, model_written=False, key=""):
    """Yield (str, written_by_a_model, key) for every string in a JSON object."""
    if isinstance(o, dict):
        for k, v in o.items():
            mw = (k == "content" and o.get("role") == "assistant") or k in ("reply", "text")
            yield from strings(v, mw, k)
    elif isinstance(o, list):
        for v in o:
            yield from strings(v, model_written, key)
    elif isinstance(o, str):
        yield o, model_written, key


def check_stai(out: Path, files: list[Path]) -> None:
    if not STAI_JSON.exists():
        report("1. no questionnaire item text", False, f"cannot run: {STAI_JSON.relative_to(ROOT)} not found")
        return
    q = json.loads(STAI_JSON.read_text())["STAI"]
    items = [re.compile(r"\b" + re.escape(norm(x["prompt"])) + r"\b") for x in q["questions"]]
    pre = norm(q["preamble"]).split()
    pre_win = [" ".join(pre[i:i + 7]) for i in range(max(1, len(pre) - 6))]
    bad, incidental = [], Counter()
    for f in files:
        rel = f.relative_to(out)
        for label, obj in text_payloads(f):
            for s, model_written, key in strings(obj):
                n = norm(s)
                hits = [i for i, rx in enumerate(items, 1) if rx.search(n)]
                if any(w in n for w in pre_win):
                    bad.append(f"{rel} {label} [{key}]: a stretch of the preamble")
                if not hits:
                    continue
                if not model_written:
                    bad.append(f"{rel} {label} [{key}]: items {hits} in a field the model did not write")
                elif len(hits) >= 3:
                    bad.append(f"{rel} {label} [{key}]: {len(hits)} items in one reply")
                else:
                    for h in hits:
                        incidental[h] += 1
    note = (f"incidental: {sum(incidental.values())} replies contain one or two of the items' short "
            f"phrases in the model's own words (by item number: {dict(sorted(incidental.items()))})")
    report("1. no questionnaire item text", not bad, "; ".join(bad[:10]) if bad else note)


def redline_patterns() -> list[re.Pattern]:
    """The red-line terms live in the local, unpublished scripts/arxiv_check.py; none are typed here."""
    p = ROOT / "scripts" / "arxiv_check.py"
    if not p.exists():
        return []
    src = p.read_text()
    m = re.search(r"# Rights and project red lines.*?\n(.*?)\n\]", src, re.S)
    pats = re.findall(r'\(r"([^"]+)",', m.group(1)) if m else []
    return [re.compile(x, re.I) for x in pats if "STAI" not in x]


def check_private(out: Path, files: list[Path]) -> None:
    user = getpass.getuser()
    host = socket.gethostname().split(".")[0]
    try:
        email = subprocess.run(["git", "-C", str(ROOT), "config", "user.email"], capture_output=True, text=True).stdout.strip()
    except OSError:
        email = ""
    pats = {
        "OpenAI key": re.compile(r"sk-[A-Za-z0-9_\-]{20,}"), "HF token": re.compile(r"hf_[A-Za-z0-9]{30,}"),
        "GitHub token": re.compile(r"gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{20,}"),
        "AWS key": re.compile(r"AKIA[0-9A-Z]{16}"), "private key": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
        "home path": re.compile(r"/Users/[A-Za-z0-9_.\-]+|/home/[a-z][a-z0-9_\-]+|C:\\\\Users"),
        "user name": re.compile(re.escape(user), re.I),
        "machine name": re.compile(re.escape(host), re.I) if len(host) > 3 else None,
        "own email": re.compile(re.escape(email), re.I) if email else None,
    }
    reds = redline_patterns()
    bad, emails = [], Counter()
    any_email = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
    for f in files:
        rel = f.relative_to(out)
        for label, obj in text_payloads(f):
            for s, _, key in (strings(obj) if not isinstance(obj, str) else [(obj, False, "")]):
                for name, rx in pats.items():
                    if rx is not None and rx.search(s):
                        bad.append(f"{rel} {label}: {name}")
                for rx in reds:
                    if rx.search(s):
                        bad.append(f"{rel} {label}: red-line term")
                emails.update(m.group(0).split("@")[1].lower() for m in any_email.finditer(s))
    note = (f"red-line list: {len(reds)} patterns from the local arxiv_check.py" if reds else
            "red-line list not found (scripts/arxiv_check.py is local only); terms not checked")
    if emails:
        note += f"; e-mail-like strings in model text, by domain: {dict(emails.most_common(8))}"
    report("2. no secrets, home paths, user or machine name, own email, red lines", not bad and bool(reds),
           "; ".join(sorted(set(bad))[:10]) if bad else note)


def windows(text: str, n: int = 10) -> set[str]:
    w = norm(text).split()
    return {" ".join(w[i:i + n]) for i in range(max(0, len(w) - n + 1))}


def check_third_party(out: Path, files: list[Path]) -> None:
    def md5(p):
        return hashlib.md5(p.read_bytes()).hexdigest()
    allowed = {out / "data-licenses" / "benzion-gpt-trauma-induction-MIT.txt"}
    third = {md5(p): p for p in THIRD.rglob("*") if p.is_file() and p.name != ".DS_Store"} if THIRD.exists() else {}
    bad = [f"{f.relative_to(out)} = {third[h].relative_to(ROOT)}" for f in files
           if f not in allowed and (h := md5(f)) in third]
    wek = set()
    for p in WEKENBORG.rglob("*"):
        if p.is_file() and p.suffix in (".txt", ".py", ".md", ".json", ".csv"):
            wek |= windows(p.read_text(errors="replace"))
    incidental = 0
    for f in files:
        for label, obj in text_payloads(f):
            for s, model_written, key in (strings(obj) if not isinstance(obj, str) else [(obj, False, "")]):
                if len(s) > 60 and windows(s) & wek:
                    if model_written:      # stock phrases a model also wrote in Wekenborg's conversation log
                        incidental += 1
                    else:
                        bad.append(f"{f.relative_to(out)} {label} [{key}]: a 10-word stretch of Wekenborg's repository")
    detail = "; ".join(bad[:10]) if bad else (f"{len(third)} third_party files and {len(wek)} Wekenborg "
                                              "10-word stretches compared; only the Ben-Zion licence is shipped; "
                                              f"{incidental} model replies share a stock 10-word phrase with "
                                              "Wekenborg's conversation log (model-written, not copied)")
    report("3. nothing from third_party/ except the Ben-Zion licence", not bad and bool(third), detail)


def check_counts(out: Path) -> None:
    r = out / "runs"
    problems, n = [], {}
    for name in STUDY1 + (STUDY2,):
        rows = jsonl(r / name / "results.jsonl")
        n[name] = len(rows)
        ids = [x["cell_id"] for x in rows]
        if len(set(ids)) != len(ids):
            problems.append(f"{name}: duplicate conversation ids")
        missing = [x["acts_file"] for x in rows if not (r / name / x["acts_file"]).exists()]
        if missing:
            problems.append(f"{name}: {len(missing)} activation files missing")
        for j in ("judge.jsonl", "judge_api.jsonl"):
            if (RUNS / name / j).exists():
                jid = {x["cell_id"] for x in jsonl(r / name / j)}
                if jid != set(ids):
                    problems.append(f"{name}/{j}: {len(jid ^ set(ids))} conversations differ from results.jsonl")
        with (r / name / "readings.csv").open() as f:
            rd = list(csv.DictReader(f))
        if len(rd) != sum(len(x["checkpoints"]) for x in rows):
            problems.append(f"{name}/readings.csv: {len(rd)} rows")
        if any(v == "nan" for row in rd for k, v in row.items() if k.endswith("reading")):
            problems.append(f"{name}/readings.csv: non-finite reading")
        if name == STUDY2:
            per = Counter(x["model"] for x in rows)
            if len(per) != EXPECT["study2_models"] or set(per.values()) != {EXPECT["study2_per_model"]}:
                problems.append(f"study2 per model: {dict(per)}")
    s1 = n["wave1"] + n["wave2"]
    if s1 != EXPECT["wave1+wave2"]:
        problems.append(f"Study 1: {s1}")
    if n["wave2-8bit"] != EXPECT["wave2-8bit"]:
        problems.append(f"8-bit: {n['wave2-8bit']}")
    if n[STUDY2] != EXPECT["study2"]:
        problems.append(f"Study 2: {n[STUDY2]}")
    models = {json.loads(x)["model"] for x in (r / STUDY2 / "results.jsonl").read_text().splitlines() if x.strip()}
    for m in models:
        g = r / f"gate0-{C.short(m)}"
        if not all((g / "acts" / f).exists() for f in POLE_FILES) or not (g / "direction.npz").exists():
            problems.append(f"{g.name}: pole activations or direction missing")
    report("4. row counts", not problems,
           "; ".join(problems) if problems else
           f"Study 1 {s1} + 8-bit {n['wave2-8bit']}, Study 2 {n[STUDY2]} (7 x 600), judges, readings and activations complete")


def clean_tree(dst: Path, mode: str) -> str:
    """A copy of the repository as the public tip will hold it, without runs/."""
    if mode == "HEAD":
        subprocess.run(["git", "clone", "-q", "--no-local", str(ROOT), str(dst)], check=True)
        return "commit " + subprocess.run(["git", "-C", str(dst), "rev-parse", "--short", "HEAD"],
                                          capture_output=True, text=True).stdout.strip()
    ls = lambda *a: subprocess.run(["git", "-C", str(ROOT), "ls-files", "-z", *a], capture_output=True,
                                   text=True, check=True).stdout.split("\0")
    tracked = set(ls()) - set(ls("-ci", "--exclude-standard")) - set(ls("-d"))
    files = sorted((tracked | set(ls("-o", "--exclude-standard"))) - {""})
    for f in files:
        if (ROOT / f).is_file():
            (dst / f).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / f, dst / f)
    return f"working tree as it would be committed ({len(files)} files; not a commit)"


def strip_volatile(o):
    if isinstance(o, dict):
        return {k: strip_volatile(v) for k, v in o.items()
                if k not in ("code_version", "created", "generated", "commit", "timestamp", "run")}
    if isinstance(o, list):
        return [strip_volatile(v) for v in o]
    return o


def diff_json(a, b, path="", tol=1e-9, out=None):
    out = [] if out is None else out
    if isinstance(a, dict) and isinstance(b, dict):
        for k in set(a) | set(b):
            if k not in a or k not in b:
                out.append(f"{path}.{k}: only in one")
            else:
                diff_json(a[k], b[k], f"{path}.{k}", tol, out)
    elif isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        for i, (x, y) in enumerate(zip(a, b)):
            diff_json(x, y, f"{path}[{i}]", tol, out)
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        if not (abs(a - b) <= tol * max(1.0, abs(a)) or (a != a and b != b)):
            out.append(f"{path}: {a} vs {b}")
    elif a != b:
        out.append(f"{path}: differs")
    return out


def check_acceptance(out: Path, tar: Path | None, mode: str, rerun: bool) -> None:
    with tempfile.TemporaryDirectory(prefix="release-acceptance-") as tmp:
        tree = Path(tmp) / "repo"
        tree.mkdir()
        what = clean_tree(tree, mode)
        if tar is not None and tar.exists():
            with tarfile.open(tar) as t:
                t.extractall(tree, filter="data")
            what += f", {tar.name} unpacked"
        else:
            shutil.copytree(out / "runs", tree / "runs", dirs_exist_ok=True)
            what += ", export copied in"
        # A reader runs setup_lite.sh, which makes .venv and clones the two reference repos into third_party/.
        # Here both are linked from this machine (same pinned copies); nothing is written through the links.
        for name in (".venv", "third_party"):
            if not (tree / name).exists():
                (tree / name).symlink_to(ROOT / name)
        what += "; .venv and third_party/ linked as setup_lite.sh would provide them"
        env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
        r = subprocess.run(["bash", "scripts/check_all.sh"], cwd=tree, capture_output=True, text=True, env=env)
        tail = "\n".join(r.stdout.strip().splitlines()[-25:])
        (out.parent / f"{out.name}-acceptance.log").write_text(r.stdout + "\n--- stderr\n" + r.stderr)
        ok = r.returncode == 0 and "ALL CHECKS PASS" in r.stdout
        report("5a. clean tree + export: scripts/check_all.sh", ok,
               what + ("" if ok else f"\n{tail}\n       full log: {out.name}-acceptance.log"))
        s1 = subprocess.run([sys.executable, "scripts/show_replies.py"], cwd=tree, capture_output=True, text=True)
        s2 = subprocess.run([sys.executable, "scripts/show_replies.py", "gemma-3-4b", "relax_story", "-n", "1"],
                            cwd=tree, capture_output=True, text=True)
        ok = s1.returncode == 0 and s2.returncode == 0 and len(s1.stdout.splitlines()) >= 70 and s2.stdout.strip()
        report("5b. clean tree + export: scripts/show_replies.py", bool(ok),
               f"{len(s1.stdout.splitlines())} model-arm lines listed; one conversation printed"
               if ok else (s1.stderr + s2.stderr)[-800:])
        if not rerun:
            return
        py = str(tree / ".venv" / "bin" / "python")
        cmds = [([py, "-m", "analysis.confirmatory", "runs/wave1", "runs/wave2", "--unblind"],
                 "runs/analysis-confirmatory/confirmatory.json"),
                ([py, "-m", "analysis.study2", "runs/study2", "--batch", "all", "--unblind"],
                 "runs/study2/analysis-batch-all/study2.json"),
                ([py, "-m", "analysis.study2_he_robust", "runs/study2", "--batch", "all", "--unblind"],
                 "runs/study2/robust-api-judge/analysis-batch-all/study2.json"),
                ([py, "analysis/study2_tables.py", "runs/study2", "--batch", "all"], None)]
        for cmd, target in cmds:
            before = json.loads((tree / target).read_text()) if target else None
            r = subprocess.run(cmd, cwd=tree, capture_output=True, text=True, env=env)
            label = "5c. rerun: " + " ".join(cmd[1:])
            if r.returncode != 0:
                report(label, False, (r.stdout + r.stderr)[-800:])
                continue
            if target is None:
                diff = subprocess.run(["git", "-C", str(ROOT), "diff", "--no-index", "--stat", "paper/tables",
                                       str(tree / "paper" / "tables")], capture_output=True, text=True).stdout
                report(label, True, "ran; tables written" + (f" (differ from paper/tables: {diff.strip()})" if diff.strip() else ", identical to paper/tables"))
                continue
            after = json.loads((tree / target).read_text())
            d = diff_json(strip_volatile(before), strip_volatile(after))
            report(label, not d, f"{target} regenerated identically" if not d else "; ".join(d[:8]))


def manifest(out: Path) -> None:
    files = [f for f in exported_files(out) if f.name != "MANIFEST.md5"]
    lines, per = [], defaultdict(lambda: [0, 0])
    for f in files:
        b = f.read_bytes()
        rel = f.relative_to(out).as_posix()
        lines.append(f"{hashlib.md5(b).hexdigest()}  {len(b):>12d}  {rel}")
        top = "/".join(rel.split("/")[:2]) if rel.startswith("runs/") else rel.split("/")[0]
        per[top][0] += 1
        per[top][1] += len(b)
    (out / "MANIFEST.md5").write_text("\n".join(lines) + "\n")
    total = sum(v[1] for v in per.values())
    rows = "\n".join(f"       {k:55s} {v[0]:6d} files {v[1] / 1e6:10.1f} MB" for k, v in sorted(per.items()))
    report("6. size and listing", True, f"{len(files)} files, {total / 1e6:.1f} MB; listing in "
           f"{out.name}/MANIFEST.md5\n{rows}")


def make_tar(out: Path) -> Path:
    tar = out.parent / f"{out.name}.tar.gz"
    with tarfile.open(tar, "w:gz") as t:
        for f in exported_files(out):
            info = t.gettarinfo(str(f), arcname=f.relative_to(out).as_posix())
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            with f.open("rb") as fh:
                t.addfile(info, fh)
    h = hashlib.md5(tar.read_bytes()).hexdigest()
    print(f"\n  {tar.name}: {tar.stat().st_size / 1e6:.1f} MB, MD5 {h}")
    return tar


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=ROOT / "release")
    ap.add_argument("--check-only", action="store_true", help="audit an existing export, do not rebuild")
    ap.add_argument("--tar", action="store_true", help="write <out>.tar.gz (tar owner fields blanked)")
    ap.add_argument("--acceptance", action="store_true", help="check 5: clean tree + export")
    ap.add_argument("--tree", choices=("auto", "HEAD", "worktree"), default="auto",
                    help="clean tree from the last commit, or from the working tree as it would be committed")
    ap.add_argument("--rerun-analysis", action="store_true", help="with --acceptance: rerun the README's analysis")
    a = ap.parse_args(argv)
    out = a.out.resolve()
    if not a.check_only:
        print(f"Building {out} ...", flush=True)
        build(out)
    elif not out.exists():
        raise SystemExit(f"{out} does not exist")
    manifest(out)
    files = exported_files(out)
    print()
    check_stai(out, files)
    check_private(out, files)
    check_third_party(out, files)
    check_counts(out)
    tar = make_tar(out) if a.tar else None
    if a.acceptance:
        mode = a.tree
        if mode == "auto":
            dirty = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=no"],
                                   capture_output=True, text=True).stdout.strip()
            mode = "worktree" if dirty else "HEAD"
        check_acceptance(out, tar, mode, a.rerun_analysis)
    failed = [n for n, ok, _ in results_log if not ok]
    print("\nEXPORT CHECKS:", "ALL PASS" if not failed else "FAILED: " + ", ".join(failed))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
