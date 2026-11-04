"""Paper figures from the result CSVs (no numbers are typed by hand).

Usage (repo root):  python scripts/20_make_figures.py
Writes paper/figures/fig_e1_judges.pdf and paper/figures/fig_rq2_slopes.pdf.
Colours: first three categorical slots of the dataviz reference palette (validated all-pairs);
marker shape is a second encoding so the figures survive greyscale printing.
"""
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper/figures"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
JUDGES = [("j_lm", "J-LM (Agrawal LM judge)", "#2a78d6", "o"),
          ("j_nli", "J-NLI (external NLI judge)", "#eb6834", "s"),
          ("j_str", "J-STR (name overlap, no model)", "#1baf7a", "^")]
MODELS = [("gpt-4", "GPT-4"), ("gpt-3.5-turbo", "GPT-3.5"), ("text-davinci-003", "GPT-3 (davinci)"),
          ("llama-2-70b-chat", "Llama-2-70B"), ("llama-2-13b-chat", "Llama-2-13B"), ("llama-2-7b-chat", "Llama-2-7B")]

plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.linewidth": 0.6, "pdf.fonttype": 42})


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def fig_e1():
    auc = pd.read_csv(ROOT / "results/e1_auroc.csv")
    fig, ax = plt.subplots(figsize=(3.4, 2.7))
    for k, (j, label, color, marker) in enumerate(JUDGES):
        for i, (m, _) in enumerate(MODELS):
            r = auc[(auc["model"] == m) & (auc["judge"] == j)].iloc[0]
            y = i + (k - 1) * 0.24
            ax.plot([r["ci_lo"], r["ci_hi"]], [y, y], color=color, linewidth=1.4, solid_capstyle="round")
            ax.plot(r["auroc"], y, marker=marker, color=color, markersize=4.2, markeredgecolor="white",
                    markeredgewidth=0.6, linestyle="none", label=label if i == 0 else None)
    ax.axvline(0.5, color=MUTED, linewidth=0.8, linestyle=(0, (3, 2)))
    ax.text(0.505, -0.55, "chance", color=MUTED, fontsize=7, va="bottom")
    ax.set_yticks(range(len(MODELS)), [n for _, n in MODELS])
    ax.set_ylim(len(MODELS) - 0.5, -0.8)
    ax.set_xlim(0.45, 0.93)
    ax.set_xlabel("AUROC (95% CI)")
    style(ax)
    ax.legend(loc="upper center", bbox_to_anchor=(0.42, 1.24), ncol=1, frameon=False, fontsize=7,
              handletextpad=0.4, labelcolor=INK)
    fig.savefig(OUT / "fig_e1_judges.pdf", bbox_inches="tight")


def fig_rq2():
    s = pd.read_csv(ROOT / "results/rq2_slopes.csv")
    order = [m for m, _ in MODELS] + ["pooled (model fixed effects)"]
    s = s.set_index("model").loc[order].reset_index()
    names = dict(MODELS) | {"pooled (model fixed effects)": "Pooled (model FE)"}
    fig, ax = plt.subplots(figsize=(3.4, 2.0))
    for i, r in enumerate(s.itertuples()):
        ax.plot([r.ci_lo, r.ci_hi], [i, i], color="#2a78d6", linewidth=1.4, solid_capstyle="round")
        ax.plot(r.slope_per_log10, i, "o", color="#2a78d6", markersize=4.2, markeredgecolor="white",
                markeredgewidth=0.6)
    ax.axvline(0, color=MUTED, linewidth=0.8, linestyle=(0, (3, 2)))
    ax.set_yticks(range(len(s)), [names[m] for m in s["model"]])
    ax.invert_yaxis()
    ax.set_xlabel("Slope per log$_{10}$ topic works (95% CI)")
    style(ax)
    fig.savefig(OUT / "fig_rq2_slopes.pdf", bbox_inches="tight")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    fig_e1()
    fig_rq2()
    print("written", *sorted(p.name for p in OUT.glob("*.pdf")))
