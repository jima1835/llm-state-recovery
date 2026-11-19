"""Paper figures, drawn from the analysis output. Deterministic: no random numbers, no hand-typed results.

    uv run --no-project --with matplotlib python scripts/make_figures.py            # all figures that have data
    uv run --no-project --with matplotlib python scripts/make_figures.py design     # one figure

Writes vector PDFs to paper/figures/ (and PNG previews with --png):
  design.pdf           the conversation and the two readings (a drawing; no data)
  study1_models.pdf    Study 1 recovery fractions per model and arm, both readings, from
                       runs/analysis-confirmatory/confirmatory.json (the registered analysis output)
Colours: blue is the reply reading, orange the self-report (checked for colour-blind separation);
arms differ by marker shape, so the figures read in grey.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "paper" / "figures"
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, MUTED, RULE, FILL = "#0b0b0b", "#52514e", "#b9b8b2", "#efeeea"
BOLD = {"fontname": "cmb10"}   # Computer Modern Bold, shipped with matplotlib (cmr10 has no bold weight)

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["cmr10", "DejaVu Serif"], "mathtext.fontset": "cm",
    "axes.formatter.use_mathtext": True, "font.size": 8, "axes.unicode_minus": False,
    "pdf.fonttype": 42, "savefig.dpi": 300, "axes.linewidth": 0.6,
})

MODEL_NAMES = {  # display name, ordered by parameter count
    "mlx-community/Llama-3.2-3B-Instruct-4bit": "Llama 3.2 3B",
    "mlx-community/gemma-3-4b-it-4bit": "Gemma 3 4B",
    "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit": "Llama 3.1 8B",
    "mlx-community/gemma-3-12b-it-4bit": "Gemma 3 12B",
    "mlx-community/Qwen3-14B-4bit": "Qwen3 14B",
    "mlx-community/Mistral-Small-24B-Instruct-2501-4bit": "Mistral Small 24B",
    "mlx-community/gemma-3-27b-it-4bit": "Gemma 3 27B",
}


def save(fig, name: str, png: bool) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", pad_inches=0.02)
    if png:
        fig.savefig(OUT / f"{name}.png", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print("wrote", (OUT / f"{name}.pdf").relative_to(ROOT))


# ------------------------------------------------------------------ Figure 1: design
def design(png: bool) -> None:
    W, H = 6.5, 2.72
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")

    gutter, cw, gap = 1.12, 1.17, 0.16           # label gutter, card width, gap between cards
    y_rep, h_out = 0.10, 0.44                    # reply-reading lane
    y_card, h_user, h_reply = 0.86, 0.50, 0.46   # the conversation
    y_self = 2.14                                # self-report lane
    cards = [
        ("Turn 1", "baseline text", "at baseline"),
        ("Turn 2: induction", "traumatic narrative\nor neutral passage", "after induction"),
        ("Turn 3: regulation", "relaxation script, control\ntext or instruction", "after regulation"),
        ("Turn 4 (Study 1 only)", "a second narrative", "after turn 4"),
    ]

    def box(x, y, w, h, fc, ec, ls="-", lw=0.7):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.04",
                                    fc=fc, ec=ec, ls=ls, lw=lw))

    def arrow(p, q, color, ls="-", lw=0.9):
        ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=7, color=color, ls=ls, lw=lw,
                                     shrinkA=0, shrinkB=0))

    # lane labels
    for y, head, sub, col in (
        (y_self + h_out / 2, "Self-report", "questionnaire on a copy\nof the conversation,\nthen discarded", ORANGE),
        (y_card + (h_user + h_reply) / 2, "Conversation", "the model answers\nevery turn in words", MUTED),
        (y_rep + h_out / 2, "Reply reading", "activations while the\nmodel writes its reply", BLUE),
    ):
        ax.add_patch(Rectangle((0.02, y - 0.3), 0.035, 0.6, fc=col, ec="none"))
        ax.text(0.12, y + 0.17, head, fontsize=8.5, color=INK, va="center", **BOLD)
        ax.text(0.12, y - 0.08, sub, fontsize=6.6, color=MUTED, va="center", linespacing=1.15)

    for i, (title, sub, when) in enumerate(cards):
        x = gutter + i * (cw + gap)
        last = i == 3
        # user turn and reply
        box(x, y_card + h_reply, cw, h_user, FILL, RULE, ls="--" if last else "-")
        ax.text(x + cw / 2, y_card + h_reply + h_user - 0.13, title, fontsize=7.6,
                color=INK, ha="center", va="center", **BOLD)
        ax.text(x + cw / 2, y_card + h_reply + 0.16, sub, fontsize=6.6, color=MUTED, ha="center",
                va="center", linespacing=1.1)
        box(x, y_card, cw, h_reply, "white", RULE, ls="--" if last else "-")
        ax.text(x + 0.07, y_card + h_reply - 0.11, "model's reply", fontsize=6.4, color=MUTED, va="center")
        bx, bw, by, bh = x + 0.07, cw - 0.14, y_card + 0.09, 0.12
        ax.add_patch(Rectangle((bx, by), bw, bh, fc="white", ec=RULE, lw=0.5))
        ax.add_patch(Rectangle((bx, by), bw * 64 / 120, bh, fc=BLUE, ec="none", alpha=0.35))
        ax.text(bx + bw * 32 / 120, by + bh / 2 - 0.005, "first 64", fontsize=5.8, color=INK, ha="center", va="center")
        ax.text(bx + bw - 0.03, by + bh / 2 - 0.005, "of 120 tokens", fontsize=5.8, color=MUTED, ha="right", va="center")
        # reply reading
        xr = bx + bw * 32 / 120
        arrow((xr, by - 0.02), (xr, y_rep + h_out + 0.02), BLUE)
        box(x, y_rep, cw, h_out, "#e9f1fb", BLUE)
        ax.text(x + cw / 2, y_rep + h_out - 0.14, "reading " + when, fontsize=7.0, color=INK, ha="center", va="center")
        ax.text(x + cw / 2, y_rep + 0.13, "mean activation, projected", fontsize=6.0, color=MUTED, ha="center", va="center")
        # checkpoint and self-report
        xc = x + cw + gap / 2
        ax.plot([x + cw, xc], [y_card + h_reply / 2] * 2, color=RULE, lw=0.8, solid_capstyle="butt")
        if not last:
            arrow((xc, y_card + h_reply / 2), (x + cw + gap, y_card + h_reply / 2), RULE, lw=0.8)
        ax.plot([xc], [y_card + h_reply / 2], marker="o", ms=3.4, color=ORANGE, zorder=5)
        arrow((xc, y_card + h_reply / 2 + 0.05), (xc, y_self - 0.02), ORANGE, ls=(0, (3, 2)))
        box(x + 0.22, y_self, cw - 0.22 + gap / 2 + 0.02, h_out, "#fdf0ea", ORANGE, ls=(0, (3, 2)))
        xm = x + 0.22 + (cw - 0.22 + gap / 2 + 0.02) / 2
        ax.text(xm, y_self + h_out - 0.14, "20 STAI-s items", fontsize=7.0, color=INK, ha="center", va="center")
        ax.text(xm, y_self + 0.13, "score 20 to 80", fontsize=6.0, color=MUTED, ha="center", va="center")
    xc0 = gutter + cw + gap / 2
    ax.plot([xc0, xc0], [y_card + h_reply / 2 - 0.05, y_card - 0.06], color=MUTED, lw=0.5, ls=(0, (1, 1.5)))
    ax.text(xc0, y_card - 0.12, "checkpoint", fontsize=6.0, color=MUTED, ha="center", va="center")
    save(fig, "design", png)


# ------------------------------------------------------------------ Figure 2: Study 1 per model
def study1_models(png: bool) -> None:
    src = ROOT / "runs" / "analysis-confirmatory" / "confirmatory.json"
    if not src.exists():
        print("skipped study1_models:", src.relative_to(ROOT), "not found"); return
    rep = json.loads(src.read_text())
    ext = next(v for k, v in rep["sets"].items() if k.startswith("Extended set"))
    models = [m for m in MODEL_NAMES if m in ext["per_model"]]
    assert len(models) == len(ext["per_model"]) == 7, "expected the seven extended-set models"
    arms = [  # key suffix, label, marker, filled
        ("relaxation", "Relaxation", "o", True),
        ("neutral_filler", "Filler", "s", False),
        ("instruction_only", "Report instruction", "^", False),
    ]
    panels = [("int", "Reply reading", BLUE), ("rep", "Self-report (questionnaire)", ORANGE)]

    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.75), sharey=True, sharex=True)
    rows = list(range(len(models), 0, -1)) + [-0.35]       # models top to bottom, then the mean
    for ax, (ch, title, col) in zip(axes, panels):
        ax.axvline(0, color=RULE, lw=0.6, zorder=0); ax.axvline(1, color=RULE, lw=0.6, ls=(0, (3, 2)), zorder=0)
        ax.axhline(0.35, color=RULE, lw=0.5, zorder=0)
        for y, m in zip(rows, models + [None]):
            vals = {}
            for key, _, mk, filled in arms:
                if m is None:
                    e = ext["extras"][f"R_{ch}_{key}"]
                    v = e["estimate"]
                    ax.plot(e["ci"], [y, y], color=INK, lw=0.9, zorder=2, solid_capstyle="butt")
                else:
                    v = ext["per_model"][m][f"R_{ch}_{key}"]
                vals[key] = v
            if m is not None:
                ax.plot([min(vals.values()), max(vals.values())], [y, y], color=RULE, lw=0.8, zorder=1)
            for key, _, mk, filled in arms:
                ax.plot([vals[key]], [y], marker=mk, ms=6.0 if filled else (5.2 if mk != "^" else 5.6),
                        mew=0.9, ls="none", zorder=4 if filled else 3,
                        mfc=col if filled else "white", mec="white" if filled else INK)
        ax.set_title(title, fontsize=8.5, loc="left", pad=13, **BOLD)
        ax.set_xlabel("Recovery fraction $R$", fontsize=8)
        ax.tick_params(length=2.5, width=0.5, labelsize=7.5)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.tick_params(axis="y", length=0)
    axes[0].set_yticks(rows)
    axes[0].set_yticklabels([MODEL_NAMES[m] for m in models] + ["Mean of seven"], fontsize=7.8)
    axes[0].set_ylim(-0.95, len(models) + 0.6)
    lo = min(v[f"R_{c}_{k}"] for v in ext["per_model"].values() for c in ("int", "rep") for k, *_ in arms)
    hi = max(v[f"R_{c}_{k}"] for v in ext["per_model"].values() for c in ("int", "rep") for k, *_ in arms)
    axes[0].set_xlim(min(-0.25, lo - 0.1), hi + 0.12)
    for ax in axes:
        ax.text(0, len(models) + 0.66, "no recovery", fontsize=6.2, color=MUTED, ha="center", va="bottom")
        ax.text(1, len(models) + 0.66, "back to baseline", fontsize=6.2, color=MUTED, ha="center", va="bottom")
    handles = [Line2D([], [], marker=mk, ls="none", ms=5.2, mew=0.9, mfc=INK if filled else "white", mec=INK, label=lab)
               for _, lab, mk, filled in arms]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=7.8,
               bbox_to_anchor=(0.56, -0.07), handletextpad=0.3, columnspacing=1.6)
    fig.subplots_adjust(wspace=0.08, left=0.2, right=0.99, top=0.84, bottom=0.2)
    save(fig, "study1_models", png)


FIGURES = {"design": design, "study1_models": study1_models}

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("which", nargs="*", choices=[*FIGURES, []], help="default: all")
    ap.add_argument("--png", action="store_true", help="also write PNG previews")
    a = ap.parse_args()
    for name in a.which or FIGURES:
        FIGURES[name](a.png)
