"""Revision analysis: attack lift, defense effect, and relevance per memory mode.

Pools any number of run directories (seeds) and resamples *users* as clusters,
so a user that appears under several seeds contributes one resampling unit.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd


def load_runs(run_dirs: list[str], start_round: int | None = None) -> pd.DataFrame:
    frames = []
    for run_dir in run_dirs:
        path = Path(run_dir) / "llm_trajectories.jsonl"
        config = json.loads((Path(run_dir) / "llm_config.json").read_text(encoding="utf-8"))
        first = int(config["intervention_rounds"]) if start_round is None else start_round
        with path.open(encoding="utf-8") as handle:
            frame = pd.DataFrame(json.loads(line) for line in handle)
        frame = frame[frame["round_idx"] >= first].copy()
        frame["run"] = Path(run_dir).name
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


AUDIT_COLUMNS = (
    "mem_title_share",
    "mem_genre_share",
    "ret_title_share",
    "ret_any_share",
    "mem_items",
)


def unit_table(frame: pd.DataFrame) -> pd.DataFrame:
    """One row per (seed, user, mode) with arm contrasts and pooled relevance."""
    audit = [c for c in AUDIT_COLUMNS if c in frame.columns]
    per_arm = (
        frame.groupby(["seed", "user_id", "memory_mode", "condition"])[
            ["target_share", "rank_relevance", "mean_utility", *audit]
        ]
        .mean()
        .unstack("condition")
    )
    out = pd.DataFrame(index=per_arm.index)
    out["lift"] = per_arm[("target_share", "head")] - per_arm[("target_share", "tail")]
    out["share_attack"] = per_arm[("target_share", "head")]
    out["share_clean"] = per_arm[("target_share", "tail")]
    out["relevance"] = per_arm["rank_relevance"].mean(axis=1)
    out["relevance_clean"] = per_arm[("rank_relevance", "tail")]
    out["utility"] = per_arm["mean_utility"].mean(axis=1)
    for column in audit:
        out[f"{column}_attack"] = per_arm[(column, "head")]
        out[f"{column}_lift"] = per_arm[(column, "head")] - per_arm[(column, "tail")]
    return out.reset_index()


def clustered_bootstrap(
    values: pd.Series,
    clusters: pd.Series,
    seed: int,
    n_bootstrap: int = 5000,
    statistic=np.mean,
) -> tuple[float, float, float, float]:
    """Mean, 95% CI and two-sided sign p-value, resampling whole clusters."""
    frame = pd.DataFrame({"v": values.to_numpy(), "c": clusters.to_numpy()}).dropna()
    if frame.empty:
        return (float("nan"),) * 4
    sums = frame.groupby("c")["v"].agg(["sum", "count"])
    s = sums["sum"].to_numpy()
    n = sums["count"].to_numpy()
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(s), size=(n_bootstrap, len(s)))
    means = s[draws].sum(axis=1) / n[draws].sum(axis=1)
    point = float(s.sum() / n.sum())
    low, high = np.quantile(means, [0.025, 0.975])
    p_value = min(1.0, 2.0 * min(float(np.mean(means <= 0.0)), float(np.mean(means >= 0.0))))
    return point, float(low), float(high), p_value


def ratio_bootstrap(
    numerator: pd.Series,
    denominator: pd.Series,
    clusters: pd.Series,
    seed: int,
    n_bootstrap: int = 5000,
) -> tuple[float, float, float]:
    frame = pd.DataFrame(
        {"a": numerator.to_numpy(), "b": denominator.to_numpy(), "c": clusters.to_numpy()}
    ).dropna()
    sums = frame.groupby("c")[["a", "b"]].sum()
    a, b = sums["a"].to_numpy(), sums["b"].to_numpy()
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(a), size=(n_bootstrap, len(a)))
    ratios = a[draws].sum(axis=1) / b[draws].sum(axis=1)
    low, high = np.quantile(ratios, [0.025, 0.975])
    return float(a.sum() / b.sum()), float(low), float(high)


def revision_table(units: pd.DataFrame, label: str, seed: int = 20261103) -> pd.DataFrame:
    rows = []
    modes = list(dict.fromkeys(units["memory_mode"]))
    audit_values = [
        c for c in units.columns if c.endswith("_lift") or c.endswith("_attack")
        if c not in ("share_attack",)
    ]
    wide = units.pivot_table(
        index=["seed", "user_id"],
        columns="memory_mode",
        values=["lift", "relevance", "relevance_clean", "utility", *audit_values],
    )
    users = wide.index.get_level_values("user_id").to_series(index=wide.index)

    def add(metric: str, mode: str, values: pd.Series, offset: int) -> None:
        mean, low, high, p = clustered_bootstrap(values, users, seed + offset)
        rows.append(
            {
                "experiment": label,
                "metric": metric,
                "mode": mode,
                "mean": mean,
                "ci_low": low,
                "ci_high": high,
                "p": p,
                "users": int(users[values.notna()].nunique()),
                "pairs": int(values.notna().sum()),
            }
        )

    for index, mode in enumerate(modes):
        add("attack_lift", mode, wide[("lift", mode)], index)
        add("relevance", mode, wide[("relevance", mode)], 50 + index)
        add("relevance_clean", mode, wide[("relevance_clean", mode)], 70 + index)
        for offset, column in enumerate(audit_values):
            if (column, mode) in wide and wide[(column, mode)].notna().any():
                add(column, mode, wide[(column, mode)], 300 + 20 * offset + index)
    if "raw" in modes:
        for index, mode in enumerate(m for m in modes if m != "raw"):
            diff = wide[("lift", "raw")] - wide[("lift", mode)]
            add("lift_reduction_vs_raw", mode, diff, 100 + index)
            share, low, high = ratio_bootstrap(
                diff, wide[("lift", "raw")], users, seed + 150 + index
            )
            rows.append(
                {
                    "experiment": label,
                    "metric": "relative_reduction_vs_raw",
                    "mode": mode,
                    "mean": share,
                    "ci_low": low,
                    "ci_high": high,
                    "p": float("nan"),
                    "users": int(users.nunique()),
                    "pairs": int(diff.notna().sum()),
                }
            )
    if "none" in modes:
        for index, mode in enumerate(m for m in modes if m != "none"):
            diff = wide[("relevance", mode)] - wide[("relevance", "none")]
            add("relevance_vs_none", mode, diff, 200 + index)
            diff_clean = wide[("relevance_clean", mode)] - wide[("relevance_clean", "none")]
            add("relevance_clean_vs_none", mode, diff_clean, 250 + index)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", required=True, help="run dirs or globs")
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run_dirs = sorted({p for pattern in args.runs for p in glob.glob(pattern)})
    if not run_dirs:
        raise SystemExit("no run directories matched")
    units = unit_table(load_runs(run_dirs))
    table = revision_table(units, args.label)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output, index=False)
    units.to_csv(Path(args.output).with_suffix(".units.csv"), index=False)
    with pd.option_context("display.width", 200, "display.max_rows", 500):
        print(table.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
