"""Generate LaTeX tables for the three extended experiments from all.csv.

usage:
  python scripts/make_extended_tables.py --csv results/revision_summary/all.csv
  python scripts/make_extended_tables.py --csv results/revision_summary/all.csv --out manuscript/
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

SIG = {True: r"^\dagger", False: ""}

POLICY_ORDER = [
    "none", "raw", "raw_recent", "eam",
    "ersm_ips_only", "ersm_slate_only", "ersm", "ersm_invariant",
]

POLICY_LABEL = {
    "none":             r"\mode{none}",
    "raw":              r"\mode{raw}",
    "raw_recent":       r"\mode{raw\_recent}",
    "eam":              r"\mode{eam}",
    "ersm_ips_only":    r"\mode{ersm\_ips\_only}",
    "ersm_slate_only":  r"\mode{ersm\_slate\_only}",
    "ersm":             r"\mode{ersm}",
    "ersm_invariant":   r"\mode{ersm\_invariant}",
}


def _sig(row: pd.Series) -> str:
    """Dagger when the 95% CI excludes zero (p < .05)."""
    if pd.isna(row["p"]):
        return ""
    return r"^\dagger" if float(row["p"]) < 0.05 else ""


def _lift(row: pd.Series) -> str:
    v = float(row["mean"])
    s = _sig(row)
    return f"${v:+.3f}{s}$"


def _pct(row: pd.Series) -> str:
    return f"{float(row['mean']):.3f}"


def _percent(row: pd.Series) -> str:
    return f"{100 * float(row['mean']):.1f}\\%"


def _get(df: pd.DataFrame, mode: str, metric: str) -> pd.Series | None:
    sub = df[(df["mode"] == mode) & (df["metric"] == metric)]
    return sub.iloc[0] if len(sub) else None


def table_kuairec(df: pd.DataFrame) -> str:
    exp = "kuairec_main"
    d = df[df["experiment"] == exp]
    if d.empty:
        return "% kuairec_main: no data yet\n"

    modes = [m for m in POLICY_ORDER if m in d["mode"].values]
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\caption{KuaiRec fully-observed feedback (200 users $\times$ 3 seeds,"
                 r" Qwen3-8B). Utilities and relevance are within-user percentiles of the logged"
                 r" watch ratios; no utility model is fitted or assumed. Columns as in"
                 r" Table~\ref{tab:main}; $^\dagger$: 95\% CI excludes zero.}")
    lines.append(r"\label{tab:kuairec}")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{lrrr}")
    lines.append(r"\toprule")
    lines.append(r"Policy & lift & reduction & relevance \\")
    lines.append(r"\midrule")
    for mode in modes:
        lbl = POLICY_LABEL.get(mode, r"\mode{" + mode + "}")
        r_lift = _get(d, mode, "attack_lift")
        r_red  = _get(d, mode, "relative_reduction_vs_raw")
        r_rel  = _get(d, mode, "relevance_clean")
        lift = _lift(r_lift) if r_lift is not None else "--"
        red  = _percent(r_red) if r_red is not None else "--"
        rel  = _pct(r_rel)  if r_rel  is not None else "--"
        lines.append(f"{lbl} & {lift} & {red} & {rel} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines) + "\n"


def table_multimodel(df: pd.DataFrame) -> str:
    """One table with Llama-3.1-8B and Qwen3-14B side by side on ML-1M and CDs."""
    model_exps = {
        "Llama-3.1-8B":  ("llama_v1_main_ml1m", "llama_v1_main_cds"),
        "Qwen3-14B":     ("qwen14b_v1_main_ml1m", "qwen14b_v1_main_cds"),
    }
    # check we have data
    available = {tag: (df["experiment"] == tag).any()
                 for exps in model_exps.values() for tag in exps}
    if not any(available.values()):
        return "% multi-model: no data yet\n"

    modes = [m for m in ["none", "raw", "raw_recent", "ersm_slate_only", "ersm", "ersm_invariant"]
             if any((df[df["experiment"] == exp]["mode"] == m).any()
                    for exps in model_exps.values() for exp in exps)]

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\caption{Replication with two further agent models (200 users $\times$ 3 seeds"
                 r" per dataset; cf.\ Table~\ref{tab:main} for Qwen3-8B). Attack lift"
                 r" ($^\dagger$: 95\% CI excludes zero) and clean-arm relevance.}")
    lines.append(r"\label{tab:multimodel}")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{l" + "rr" * 4 + "}")
    lines.append(r"\toprule")
    lines.append(r"& \multicolumn{4}{c}{Llama-3.1-8B} & \multicolumn{4}{c}{Qwen3-14B} \\")
    lines.append(r"\cmidrule(lr){2-5}\cmidrule(lr){6-9}")
    lines.append(r"& \multicolumn{2}{c}{ML-1M} & \multicolumn{2}{c}{CDs}"
                 r" & \multicolumn{2}{c}{ML-1M} & \multicolumn{2}{c}{CDs} \\")
    lines.append(r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}"
                 r"\cmidrule(lr){6-7}\cmidrule(lr){8-9}")
    lines.append(r"Policy & lift & rel. & lift & rel. & lift & rel. & lift & rel. \\")
    lines.append(r"\midrule")

    exp_order = [("Llama-3.1-8B", "llama_v1_main_ml1m"),
                 ("Llama-3.1-8B", "llama_v1_main_cds"),
                 ("Qwen3-14B",    "qwen14b_v1_main_ml1m"),
                 ("Qwen3-14B",    "qwen14b_v1_main_cds")]

    for mode in modes:
        lbl = POLICY_LABEL.get(mode, r"\mode{" + mode + "}")
        cells = []
        for _, exp_tag in exp_order:
            d = df[df["experiment"] == exp_tag]
            r_lift = _get(d, mode, "attack_lift")
            r_rel  = _get(d, mode, "relevance_clean")
            cells.append(_lift(r_lift) if r_lift is not None else "--")
            cells.append(_pct(r_rel)  if r_rel  is not None else "--")
        lines.append(f"{lbl} & " + " & ".join(cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines) + "\n"


HOSTED_MODELS = [
    ("Claude-Haiku-5.5", "haiku55"),
    ("DeepSeek-v4.1-flash", "dsv41flash"),
    ("Qwen3.8-27B", "qwen38_27b"),
    ("GPT-6-Luna", "gpt6luna"),
    ("GPT-6.1-Sol", "gpt61sol"),
]


def table_hosted(df: pd.DataFrame) -> str:
    """Hosted models: one block of policies per model, ML-1M and CDs side by side."""
    models = [(label, tag) for label, tag in HOSTED_MODELS
              if (df["experiment"] == f"{tag}_main_ml1m").any()]
    if not models:
        return "% hosted models: no data yet\n"
    lines = [
        r"\begin{table}[t]",
        r"\caption{Hosted agent models (200 users per dataset; 3 seeds for the OpenRouter"
        r" models at temperature 0, 1 seed for GPT-6-Luna at default sampling; reasoning"
        r" disabled). Columns as in Table~\ref{tab:main}; $^\dagger$: 95\% CI excludes zero.}",
        r"\label{tab:hosted}",
        r"\small",
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"& \multicolumn{3}{c}{ML-1M} & \multicolumn{3}{c}{Amazon CDs} \\",
        r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
        r"Policy & lift & reduction & relevance & lift & reduction & relevance \\",
    ]
    for label, tag in models:
        lines.append(r"\midrule")
        lines.append(r"\multicolumn{7}{l}{\emph{" + label + r"}} \\")
        for mode in ["none", "raw", "ersm", "ersm_invariant"]:
            cells = []
            for ds in ("ml1m", "cds"):
                d = df[df["experiment"] == f"{tag}_main_{ds}"]
                r_lift = _get(d, mode, "attack_lift")
                r_red = _get(d, mode, "relative_reduction_vs_raw")
                r_rel = _get(d, mode, "relevance_clean")
                cells.append(_lift(r_lift) if r_lift is not None else "--")
                cells.append(_percent(r_red) if r_red is not None else "--")
                cells.append(_pct(r_rel) if r_rel is not None else "--")
            lines.append(f"{POLICY_LABEL[mode]} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="results/revision_summary/all.csv")
    parser.add_argument("--out", default=None,
                        help="directory to write .tex files; prints to stdout if omitted")
    args = parser.parse_args()
    df = pd.read_csv(args.csv)
    tables = {
        "tab_kuairec.tex":    table_kuairec(df),
        "tab_multimodel.tex": table_multimodel(df),
        "tab_hosted.tex":     table_hosted(df),
    }
    for name, tex in tables.items():
        if args.out:
            path = Path(args.out) / name
            path.write_text(tex, encoding="utf-8")
            print(f"wrote {path}")
        else:
            print(f"\n%% ===== {name} =====")
            print(tex)


if __name__ == "__main__":
    main()
