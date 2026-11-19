"""Build the arXiv source package from paper/.

    .venv/bin/python scripts/arxiv_pack.py          # writes arxiv/, arxiv.tar.gz, arxiv-preview.pdf

Rebuild after every paper edit; never edit arxiv/ by hand. What it does:
  1. Copies only main.tex, sections/*.tex, tables/*.tex and figures/*.pdf (never project_context.md,
     formulas.*, other PDFs, .claude/).
  2. Strips LaTeX comments, since arXiv publishes the source. A comment-only line is removed; a trailing
     comment is cut back to a bare "%" so line-end spacing is unchanged. "\\%" is kept.
  3. Adds \\pdfoutput=1 in the first lines (arXiv's pdflatex hint; guarded so tectonic still builds).
  4. Builds once with tectonic (offline cache) to make main.bbl, then builds the stripped package again
     from scratch to check that it compiles, and writes arxiv-preview.pdf beside the folder.
  5. Checks: ASCII-only source (arXiv runs pdflatex), no \\todo left, no undefined references or citations.
     Problems are listed at the end; the exit code is 1 if any was found.
arXiv runs pdflatex, not tectonic, so the preview is close but not identical. Check arXiv's own PDF
before announcing.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PAPER = ROOT / "paper"
OUT = ROOT / "arxiv"
TARBALL = ROOT / "arxiv.tar.gz"
PREVIEW = ROOT / "arxiv-preview.pdf"
PDFOUTPUT = "\\ifdefined\\pdftexversion\\pdfoutput=1\\fi % arXiv: \\pdfoutput=1\n"


def sources() -> list[Path]:
    files = [PAPER / "main.tex"]
    files += sorted((PAPER / "sections").glob("*.tex"))
    files += sorted((PAPER / "tables").glob("*.tex"))
    return files


def figures() -> list[Path]:
    """Figure PDFs written by scripts/make_figures.py; copied as they are."""
    return sorted((PAPER / "figures").glob("*.pdf"))


def strip_comments(text: str) -> str:
    out = []
    for line in text.splitlines():
        m = re.search(r"(?<!\\)%", line)
        if m is None:
            out.append(line)
        elif line[: m.start()].strip() == "":
            continue  # comment-only line: drop it, so no blank line (paragraph break) appears
        else:
            out.append(line[: m.start()] + "%")
    return "\n".join(out) + "\n"


def tectonic(workdir: Path, keep: bool) -> subprocess.CompletedProcess:
    env = dict(os.environ, TECTONIC_CACHE_DIR=str(ROOT / ".tectonic-cache"))
    cmd = ["tectonic", "-C", "main.tex", "--keep-logs"] + (["--keep-intermediates"] if keep else [])
    return subprocess.run(cmd, cwd=workdir, env=env, capture_output=True, text=True)


def main() -> int:
    problems: list[str] = []

    with tempfile.TemporaryDirectory() as tmp:
        # 1. Build the original once to get the .bbl
        build = Path(tmp) / "orig"
        for f in sources() + figures() + [PAPER / "refs.bib"]:
            dest = build / f.relative_to(PAPER)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, dest)
        r = tectonic(build, keep=True)
        if r.returncode != 0 or not (build / "main.bbl").exists():
            print(r.stdout[-3000:], r.stderr[-3000:], sep="\n")
            print("tectonic failed on paper/main.tex; nothing written.")
            return 1

        # 2. Package: stripped sources + bbl
        if OUT.exists():
            shutil.rmtree(OUT)
        for f in sources():
            rel = f.relative_to(PAPER)
            text = strip_comments(f.read_text())
            if rel.name == "main.tex":
                text = PDFOUTPUT + text
            dest = OUT / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(text)
        for f in figures():
            dest = OUT / f.relative_to(PAPER)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, dest)
        shutil.copy2(build / "main.bbl", OUT / "main.bbl")

        # 3. Check the shipped .bbl covers every citation the build records (bibitems may wrap lines)
        cited = set(re.findall(r"\\citation\{([^}]+)\}", (build / "main.aux").read_text()))
        cited = {k.strip() for c in cited for k in c.split(",")}
        have = set(re.findall(r"\\bibitem(?:\[.*?\])?\{([^}]+)\}", (OUT / "main.bbl").read_text(), re.S))
        if cited - have:
            problems.append("cited but missing from main.bbl: " + ", ".join(sorted(cited - have)))

        # 4. Rebuild the stripped package. Tectonic reruns BibTeX itself and ignores a shipped .bbl when
        #    no .bib is present (arXiv does not), so refs.bib is added to this test copy only.
        check = Path(tmp) / "check"
        shutil.copytree(OUT, check)
        shutil.copy2(PAPER / "refs.bib", check / "refs.bib")
        r = tectonic(check, keep=False)
        if r.returncode != 0:
            print(r.stdout[-3000:], r.stderr[-3000:], sep="\n")
            problems.append("the stripped package does not compile")
        else:
            shutil.copy2(check / "main.pdf", PREVIEW)
            log = (check / "main.log").read_text(errors="replace") if (check / "main.log").exists() else ""
            for pat, what in [(r"Reference `[^']+' on page", "undefined reference"),
                              (r"Citation `[^']+' on page", "undefined citation")]:
                hits = sorted(set(re.findall(pat, log)))
                if hits:
                    problems.append(f"{what}: " + "; ".join(hits[:5]))

    # 5. Content checks on the package
    todos = []
    for f in sorted(OUT.rglob("*.tex")):
        for i, line in enumerate(f.read_text().splitlines(), 1):
            if any(ord(ch) > 127 for ch in line):
                problems.append(f"non-ASCII character: {f.relative_to(OUT)}:{i}")
            if "\\todo{" in line and "\\newcommand" not in line:
                todos.append(f"{f.relative_to(OUT)}:{i}")
    if todos:
        problems.append(f"{len(todos)} \\todo left: " + ", ".join(todos))

    with tarfile.open(TARBALL, "w:gz") as tar:
        for f in sorted(OUT.rglob("*")):
            if f.is_file():
                tar.add(f, arcname=str(f.relative_to(OUT)))

    files = sorted(str(f.relative_to(OUT)) for f in OUT.rglob("*") if f.is_file())
    print(f"arxiv/ ({len(files)} files): " + ", ".join(files))
    print(f"{TARBALL.name}: {TARBALL.stat().st_size / 1024:.0f} KB   preview: {PREVIEW.name if PREVIEW.exists() else 'none'}")
    if problems:
        print("\nNOT READY:")
        for p in problems:
            print("  - " + p)
        return 1
    print("\nReady to upload: arxiv.tar.gz")
    return 0


if __name__ == "__main__":
    sys.exit(main())
