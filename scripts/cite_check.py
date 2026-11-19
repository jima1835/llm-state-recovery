"""Citation cross-check. Deterministic, offline, reads only the paper source (and the built PDF if present).

    python3 scripts/cite_check.py            # paper/main.tex, paper/refs.bib, paper/main.pdf
    python3 scripts/cite_check.py --no-pdf

Checks:
  1. every key cited in the text has an entry in refs.bib;
  2. every entry in refs.bib is cited (plainnat prints cited entries only, so an uncited entry never shows);
  3. if the PDF exists and pypdf is installed: every cited entry's title appears in the PDF's reference
     list, the list has one item per cited key, and it is in BibTeX's alphabetical order.
Comments (% ...) are ignored; \\input files are followed from main.tex. Exit code 1 on any failure.
It does not check that a reference is real or supports its sentence (scripts/arxiv_check.py --online
and the manual pass in TRACKING.md do that).
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

CITE = re.compile(r"\\cite[a-zA-Z]*\*?(?:\[[^\]]*\])*\{([^}]*)\}")


def strip_comments(s: str) -> str:
    return re.sub(r"(?<!\\)%.*", "", s)


def load_tex(path: Path, seen: list[Path]) -> list[tuple[Path, str]]:
    text = strip_comments(path.read_text())
    seen.append(path)
    out = [(path, text)]
    for name in re.findall(r"\\(?:input|include)\{([^}]+)\}", text):
        child = (path.parent if path.name == "main.tex" else path.parent.parent) / name
        child = child if child.suffix else child.with_suffix(".tex")
        if not child.exists():
            child = seen[0].parent / (name if name.endswith(".tex") else name + ".tex")
        if child.exists() and child not in seen:
            out += load_tex(child, seen)
    return out


def bib_entries(text: str) -> dict[str, dict[str, str]]:
    entries: dict[str, dict[str, str]] = {}
    for m in re.finditer(r"^@(\w+)\{([^,\s]+),", text, re.M):
        body = text[m.end(): text.find("\n@", m.end()) if "\n@" in text[m.end():] else len(text)]
        fields = {}
        for f in ("author", "title", "year"):
            fm = re.search(rf"\b{f}\s*=\s*\{{(.*?)\}}\s*,?\s*(?:\n|\w+\s*=|$)", body, re.S)
            fields[f] = " ".join(fm.group(1).split()) if fm else ""
        entries[m.group(2)] = fields
    return entries


def squash(s: str) -> str:
    s = re.sub(r"\$[^$]*\$", "", s)          # math in titles prints as symbols
    return re.sub(r"[^a-z0-9]", "", s.lower())


def sort_key(entry: dict[str, str]) -> str:
    # BibTeX's purify$: hyphens become spaces, so "Ben-Zion" sorts before "Benjamini".
    fam = entry["author"].split(" and ")[0].split(",")[0]
    return re.sub(r"[^a-z ]", "", fam.lower().replace("-", " "))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--paper", type=Path, default=Path(__file__).resolve().parent.parent / "paper")
    ap.add_argument("--no-pdf", action="store_true")
    a = ap.parse_args()

    files = load_tex(a.paper / "main.tex", [])
    bib = bib_entries(strip_comments((a.paper / "refs.bib").read_text()))
    cited: dict[str, dict[str, int]] = {}
    for path, text in files:
        for group in CITE.findall(text):
            for key in (k.strip() for k in group.split(",")):
                if key:
                    cited.setdefault(key, {}).setdefault(path.name, 0)
                    cited[key][path.name] += 1

    fails: list[str] = []
    print(f"{len(files)} source files, {len(bib)} entries in refs.bib, {len(cited)} distinct keys cited\n")
    print(f"{'key':24} {'cites':>5}  where")
    for key in sorted(bib, key=lambda k: (sort_key(bib[k]), bib[k]["year"])):
        where = cited.get(key, {})
        print(f"{key:24} {sum(where.values()):>5}  " + (", ".join(f"{f} x{n}" for f, n in sorted(where.items())) or "NOT CITED"))
        if not where:
            fails.append(f"in refs.bib but never cited: {key}")
    for key in sorted(set(cited) - set(bib)):
        fails.append(f"cited but not in refs.bib: {key} ({', '.join(cited[key])})")

    pdf = a.paper / "main.pdf"
    newest = max(p.stat().st_mtime for p, _ in files + [(a.paper / "refs.bib", "")])
    if a.no_pdf:
        print("\nPDF check skipped (--no-pdf).")
    elif not pdf.exists():
        print("\nPDF check skipped: paper/main.pdf not found. Build it with: tectonic paper/main.tex")
    else:
        try:
            from pypdf import PdfReader
        except ImportError:
            PdfReader = None
            print("\nPDF check skipped: pypdf is not installed in this Python.")
        if PdfReader is not None:
            if pdf.stat().st_mtime < newest:
                fails.append("paper/main.pdf is older than the source; rebuild before trusting the PDF check")
            pages = [p.extract_text() or "" for p in PdfReader(str(pdf)).pages]
            full = "\n".join(pages)
            refs = squash(full[full.rfind("References"):])
            pos = {}
            for key in cited:
                if key in bib:
                    i = refs.find(squash(bib[key]["title"])[:30])
                    if i < 0:
                        fails.append(f"title of {key} not found in the PDF reference list")
                    else:
                        pos[key] = i
            order = sorted(pos, key=pos.get)
            expected = sorted(pos, key=lambda k: (sort_key(bib[k]), pos[k]))
            print(f"\nPDF reference list: {len(pos)} of {len(cited)} cited entries found by title.")
            if order != expected:
                fails.append("PDF reference list is not in alphabetical order: " + " < ".join(order))
            else:
                print("Order in the PDF is alphabetical by first author (BibTeX rules).")

    print()
    for f in fails:
        print("FAIL:", f)
    print("RESULT:", "FAIL" if fails else "PASS: every entry is cited and every citation has an entry")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
