"""Aggregate every revision experiment into one table.

usage: python scripts/aggregate_revision.py [--results results/revision] [--out results/revision_summary]
Seeds of the same experiment (rev_<name>_seed<k>) are pooled; users are the
bootstrap clusters. Holm correction is applied to the defense-vs-raw p-values
within each experiment.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import re

import numpy as np
import pandas as pd

from stickyrec.revision_analysis import load_runs, revision_table, unit_table


def holm(p_values: pd.Series) -> pd.Series:
    p = p_values.to_numpy(dtype=float)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(p) - rank) * p[index]))
        adjusted[index] = running
    return pd.Series(adjusted, index=p_values.index)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results/revision")
    parser.add_argument("--out", default="results/revision_summary")
    args = parser.parse_args()
    groups: dict[str, list[str]] = defaultdict(list)
    for run in sorted(Path(args.results).glob("rev_*_seed*")):
        if (run / "llm_summary.csv").exists():
            groups[re.sub(r"_seed\d+$", "", run.name[4:])].append(str(run))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tables = []
    for name, runs in sorted(groups.items()):
        units = unit_table(load_runs(runs))
        units.to_csv(out / f"{name}.units.csv", index=False)
        table = revision_table(units, name)
        table["seeds"] = len(runs)
        mask = table["metric"] == "lift_reduction_vs_raw"
        table.loc[mask, "p_holm"] = holm(table.loc[mask, "p"])
        tables.append(table)
        print(f"{name}: {len(runs)} seeds, {units['user_id'].nunique()} users")
    summary = pd.concat(tables, ignore_index=True)
    summary.to_csv(out / "all.csv", index=False)
    lifts = summary[summary["metric"].isin(["attack_lift", "relative_reduction_vs_raw", "relevance_clean"])]
    pivot = lifts.pivot_table(index=["experiment", "mode"], columns="metric", values="mean")
    with pd.option_context("display.width", 200, "display.max_rows", 500):
        print(pivot.round(3).to_string())


if __name__ == "__main__":
    main()
