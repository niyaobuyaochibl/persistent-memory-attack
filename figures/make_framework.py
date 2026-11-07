"""Figure 1: the early-exposure attack loop and where ERSM intervenes.

Pure matplotlib so it builds anywhere; writes fig0_framework.{pdf,png}.
Coordinates are in tenths of an inch so font sizes are the printed sizes.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parent

plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})

RED = "#c0392b"
BLUE = "#2e86ab"
GREEN = "#2f7d4f"
GREY = "#5f6368"
FACE = {"red": "#fbe9eb", "blue": "#e6f2f8", "green": "#e8f4ec", "grey": "#f4f5f6"}

W, H = 16.0, 10.0
COLS = (0.5, 22.0, 43.5)
TOP, BOT = 25.0, 9.0
BODY = 8.0
TITLE = 9.5


def box(ax, x, y, title, body, w=W, h=H, face="grey", edge=GREY, ls="-"):
    ax.add_patch(
        FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.8",
                       facecolor=FACE[face], edgecolor=edge, linewidth=1.2, linestyle=ls)
    )
    ax.text(x + w / 2, y + h - 1.2, title, ha="center", va="top", fontsize=TITLE,
            fontweight="bold", color=edge if edge != GREY else "black")
    if body:
        ax.text(x + w / 2, y + h - 3.9, body, ha="center", va="top", fontsize=BODY,
                linespacing=1.25)


def arrow(ax, a, b, color=GREY, lw=1.3, ls="-"):
    ax.add_patch(
        FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=10, color=color,
                        linewidth=lw, linestyle=ls, shrinkA=0, shrinkB=0)
    )


def label(ax, x, y, text, color=GREY, ha="center", **kw):
    ax.text(x, y, text, ha=ha, va="center", fontsize=7.5, color=color, **kw)


def badge(ax, x, y, text):
    ax.text(x, y, text, ha="center", va="center", fontsize=7.5, color="white",
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.25", facecolor=RED, edgecolor="none"))


def main() -> None:
    fig = plt.figure(figsize=(6.0, 4.85))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 60)
    ax.set_ylim(-1.5, 47)
    ax.axis("off")
    left, mid, right = COLS

    box(ax, mid - 3, 38.5, "Attacker", "controls exposure in rounds $1..k$ only",
        w=W + 6, h=7.5, face="red", edge=RED)
    box(ax, left, TOP, "LLM agent $\\pi$",
        "$S_t=\\pi(C_t, M_t)$\nweights, prompt and\ndecoding untouched",
        face="blue", edge=BLUE)
    box(ax, mid, TOP, "Slate $S_t$", "$m$ items chosen from\ncandidates $C_t$, shown\nto the user")
    box(ax, right, TOP, "User",
        "clicks only shown items:\n$\\Pr[c_t{=}i]\\propto\\pi_i\\,g(r_{u,i})$\nexposure $\\times$ preference")
    box(ax, right, BOT, "Memory write $\\phi$",
        "$M_{t+1}=\\phi(M_t,S_t,c_t)$\nevent log, LLM facts,\nor mem0 vector store")
    box(ax, mid, BOT, "Persistent memory",
        "$M_t$ outlives the attack:\nstill tilted towards $G$\nafter round $k$")
    box(ax, left, BOT, "Memory read",
        "serialized into prompt\nor retrieved top-$k$\nmemories")

    yt, yb = TOP + H / 2, BOT + H / 2
    arrow(ax, (left + W, yt), (mid, yt))
    label(ax, (left + W + mid) / 2, yt + 1.4, "$t>k$")
    arrow(ax, (mid + W, yt), (right, yt))
    label(ax, (mid + W + right) / 2, yt + 1.4, "exposure")
    x_click = right + 5.5
    arrow(ax, (x_click, TOP), (x_click, BOT + H))
    label(ax, x_click - 0.8, (TOP + BOT + H) / 2, "click $c_t$", ha="right")
    arrow(ax, (right, yb), (mid + W, yb))
    arrow(ax, (mid, yb), (left + W, yb))
    x_read = left + 5.5
    arrow(ax, (x_read, BOT + H), (x_read, TOP))
    arrow(ax, (mid + W / 2, 38.5), (mid + W / 2, TOP + H), color=RED, lw=1.5)
    label(ax, mid + W / 2 + 1.2, 36.7, "saturate early slates with $G$", color=RED,
          ha="left")

    label(ax, mid + W / 2, (TOP + BOT + H) / 2,
          "self-reinforcing loop: promoted items are shown,\n"
          "clicked, written back, and recommended again",
          color=RED, style="italic")
    badge(ax, right + W - 4.6, BOT + H, "write surface")
    badge(ax, left + W - 4.6, BOT + H, "read surface")

    ax.add_patch(
        FancyBboxPatch((left, -1.0), right + W - left, 7.0,
                       boxstyle="round,pad=0,rounding_size=0.8", facecolor=FACE["green"],
                       edgecolor=GREEN, linewidth=1.3, linestyle="--")
    )
    ax.text(30, 2.5, "ERSM\n(ours)", ha="center", va="center", fontsize=TITLE,
            fontweight="bold", color=GREEN)
    ax.text(left + 1.0, 2.5,
            "read side: attribute profile\nwithout item identities,\n"
            "anchored to pre-session profile",
            ha="left", va="center", fontsize=BODY, linespacing=1.25)
    ax.text(right + W - 1.0, 2.5,
            "write side: slate-residualized\nevidence (+ catalogue IPS);\n"
            "write gate for fact/mem0",
            ha="right", va="center", fontsize=BODY, linespacing=1.25)
    arrow(ax, (left + W / 2 + 3, 6.0), (left + W / 2 + 3, BOT), color=GREEN, lw=1.6)
    arrow(ax, (right + W / 2 + 3, 6.0), (right + W / 2 + 3, BOT), color=GREEN, lw=1.6)

    for suffix in ("pdf", "png"):
        fig.savefig(OUT / f"fig0_framework.{suffix}", dpi=250)


if __name__ == "__main__":
    main()
