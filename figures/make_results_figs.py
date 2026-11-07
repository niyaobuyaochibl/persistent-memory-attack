"""Figures 2 and 3: cross-model attack lift and attacker strength.

Reads the bootstrap summaries in ../results/summary and writes fig2_crossmodel.pdf,
fig3_strength.pdf, and tab_models.tex (the compact cross-model table). Figure
sizes equal the printed size in acmsmall (text width 5.48 in), so font sizes are
the printed sizes.
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
RES = HERE.parent / "results" / "summary"

plt.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 8,
    "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 8,
    "legend.fontsize": 8, "axes.linewidth": 0.7,
})

TEXT_W = 5.48
RED, GREEN, BLUE, GREY = "#c0392b", "#2f7d4f", "#2e86ab", "#5f6368"
POLICIES = [("raw", "raw memory", RED, "o"), ("ersm", "ERSM", GREEN, "s"),
            ("ersm_invariant", "ERSM-invariant", BLUE, "D")]

SOURCES = {
    "revision": RES / "revision_all.csv",
    "extended": RES / "extended_all.csv",
    "hosted": RES / "hosted_all.csv",
}

# (label, source, experiment prefix, group)
MODELS = [
    ("Qwen3-8B", "revision", "main", "open"),
    ("Llama-3.1-8B", "extended", "llama_v1_main", "open"),
    ("Qwen3-14B", "extended", "qwen14b_v1_main", "open"),
    ("Claude-Haiku-5.5", "hosted", "haiku55_main", "openrouter"),
    ("DeepSeek-v4.1-flash", "hosted", "dsv41flash_main", "openrouter"),
    ("Qwen3.8-27B", "hosted", "qwen38_27b_main", "openrouter"),
    ("GPT-6-Luna", "hosted", "gpt6luna_main", "chatgpt"),
    ("GPT-6.1-Sol", "hosted", "gpt61sol_main", "chatgpt"),
]
DOMAINS = [("ml1m", "MovieLens-1M"), ("cds", "Amazon CDs")]


def load(path: Path) -> dict:
    out = {}
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            out[(r["experiment"], r["metric"], r["mode"])] = r
    return out


DATA = {k: load(p) for k, p in SOURCES.items()}


def stat(source: str, experiment: str, metric: str, mode: str) -> tuple[float, float, float]:
    r = DATA[source][(experiment, metric, mode)]
    return float(r["mean"]), float(r["ci_low"]), float(r["ci_high"])


def fig_crossmodel() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(TEXT_W, 2.75), sharey=True)
    n = len(MODELS)
    offsets = {"raw": 0.24, "ersm": 0.0, "ersm_invariant": -0.24}
    for ax, (dom, title) in zip(axes, DOMAINS):
        for i, (name, src, prefix, _) in enumerate(MODELS):
            y = n - 1 - i
            for mode, lab, color, marker in POLICIES:
                m, lo, hi = stat(src, f"{prefix}_{dom}", "attack_lift", mode)
                ax.errorbar(m, y + offsets[mode], xerr=[[m - lo], [hi - m]], fmt=marker,
                            ms=3.6, color=color, ecolor=color, elinewidth=0.9, capsize=1.6,
                            label=lab if i == 0 else None)
        ax.axvline(0, color=GREY, lw=0.7, ls="--", zorder=0)
        for y_sep in (n - 3.5, n - 6.5):
            ax.axhline(y_sep, color="#d0d0d0", lw=0.6, zorder=0)
        ax.set_title(title, fontsize=8.5)
        ax.set_xlabel("attack lift (95% CI)")
        ax.grid(axis="x", color="#ececec", lw=0.5, zorder=0)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_xlim(-0.03, 0.25)
    axes[1].set_xlim(-0.03, 0.42)
    labels = [m[0] + ("$^\\ast$" if m[3] == "chatgpt" else "") for m in MODELS]
    axes[0].set_yticks(range(n))
    axes[0].set_yticklabels(labels[::-1])
    axes[0].set_ylim(-0.6, n - 0.4)
    axes[0].tick_params(axis="y", length=0)
    handles, labs = axes[0].get_legend_handles_labels()
    fig.legend(handles, labs, loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.56, 1.0), handletextpad=0.3, columnspacing=1.4)
    fig.tight_layout(rect=(0, 0, 1, 0.93), w_pad=1.2)
    fig.savefig(HERE / "fig2_crossmodel.pdf")
    fig.savefig(HERE / "fig2_crossmodel.png", dpi=200)


WEAK = [("weak_f1_ml1m", "1"), ("weak_f3_ml1m", "3"), ("weak_f6_ml1m", "6"),
        ("weak_f12_ml1m", "12"), ("weak_pin1_ml1m", "pinned slot\n($f{=}1$)"),
        ("weak_k1_ml1m", "$k{=}1$,\nfull pool")]


def fig_strength() -> None:
    fig, ax = plt.subplots(figsize=(4.1, 2.05))
    xs = [0, 1, 2, 3, 4.7, 6.0]
    offsets = {"raw": -0.12, "ersm": 0.0, "ersm_invariant": 0.12}
    for mode, lab, color, marker in POLICIES:
        pts = [stat("revision", e, "attack_lift", mode) for e, _ in WEAK]
        px = [x + offsets[mode] for x in xs]
        ax.plot(px[:4], [p[0] for p in pts[:4]], color=color, lw=1.0)
        ax.errorbar(px, [p[0] for p in pts],
                    yerr=[[p[0] - p[1] for p in pts], [p[2] - p[0] for p in pts]],
                    fmt=marker, ms=3.6, color=color, elinewidth=0.9, capsize=1.6, label=lab)
    ax.axhline(0, color=GREY, lw=0.7, ls="--", zorder=0)
    ax.axvline(3.85, color="#d0d0d0", lw=0.6, zorder=0)
    ax.set_xticks(xs)
    ax.set_xticklabels([w[1] for w in WEAK])
    ax.annotate("promoted candidates $f$ of 12 ($k{=}2$)", xy=(1.5, 0), xycoords=("data", "axes fraction"),
                xytext=(0, -24), textcoords="offset points", ha="center", va="top", fontsize=7.5)
    ax.set_ylabel("attack lift (95% CI)")
    ax.set_ylim(-0.03, 0.17)
    ax.grid(axis="y", color="#ececec", lw=0.5, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(loc="upper left", frameon=False, handletextpad=0.3, borderaxespad=0.2)
    fig.tight_layout()
    fig.savefig(HERE / "fig3_strength.pdf")
    fig.savefig(HERE / "fig3_strength.png", dpi=200)


def cell(m: float, lo: float, hi: float) -> str:
    v = f"{m:+.3f}" if round(m, 3) != 0 else "0.000"
    dag = "^\\dagger" if (lo > 0 or hi < 0) else ""
    return f"${v}{dag}$"


def table_models() -> str:
    groups = {"open": "\\emph{Open-weight, vLLM, greedy decoding (3 seeds)}",
              "openrouter": "\\emph{Hosted via OpenRouter, temperature 0 (3 seeds)}",
              "chatgpt": "\\emph{Hosted via ChatGPT plan, default sampling (1 seed)}"}
    lines, last = [], None
    for name, src, prefix, grp in MODELS:
        if grp != last:
            if last is not None:
                lines.append("\\midrule")
            lines.append(f"\\multicolumn{{9}}{{l}}{{{groups[grp]}}} \\\\")
            last = grp
        row = [name]
        for dom, _ in DOMAINS:
            exp = f"{prefix}_{dom}"
            for mode in ("raw", "ersm", "ersm_invariant"):
                row.append(cell(*stat(src, exp, "attack_lift", mode)))
            row.append(cell(*stat(src, exp, "relevance_clean_vs_none", "ersm")))
        lines.append(" & ".join(row) + " \\\\")
    return "\n".join(lines)


def report() -> None:
    for name, src, prefix, _ in MODELS:
        for dom, _ in DOMAINS:
            exp = f"{prefix}_{dom}"
            vals = {mode: stat(src, exp, "attack_lift", mode)[0]
                    for mode in ("none", "raw", "ersm", "ersm_invariant")}
            red = {mode: 1 - vals[mode] / vals["raw"] for mode in ("ersm", "ersm_invariant")}
            rel = {mode: stat(src, exp, "relevance_clean_vs_none", mode)
                   for mode in ("raw", "ersm", "ersm_invariant")}
            print(f"{name:20s} {dom:5s} lift none {vals['none']:+.3f} raw {vals['raw']:+.3f} "
                  f"ersm {vals['ersm']:+.3f} ({red['ersm']:.1%}) inv {vals['ersm_invariant']:+.3f} "
                  f"({red['ersm_invariant']:.1%}) | drel raw {rel['raw'][0]:+.4f} "
                  f"ersm {rel['ersm'][0]:+.4f} [{rel['ersm'][1]:+.4f},{rel['ersm'][2]:+.4f}] "
                  f"inv {rel['ersm_invariant'][0]:+.4f}")


if __name__ == "__main__":
    fig_crossmodel()
    fig_strength()
    (HERE / "tab_models.tex").write_text(table_models(), encoding="utf-8")
    report()
