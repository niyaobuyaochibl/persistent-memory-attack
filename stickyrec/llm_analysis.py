from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .analysis import paired_bootstrap


def personalization_table(
    trajectory_paths: list[str | Path],
    drift_only: bool = True,
    pool_conditions: bool = True,
    bootstrap_seed: int = 20260614,
) -> pd.DataFrame:
    """Post-drift personalization: mean slate relevance to the *current*
    preference, per memory mode, with paired-bootstrap contrasts.

    ``mean_utility`` is the slate's mean rank-utility against the hidden
    profile at that round (the post-drift profile once drift is active), so a
    policy that tracks the user's shifted taste scores higher. With
    ``pool_conditions`` (default) the two intervention arms are averaged so the
    paired unit is (seed, user), matching the ``60 paired runs`` convention used
    elsewhere; otherwise each (seed, user, arm) is a separate unit. Contrasts of
    interest: ersm-raw (better memory), raw-none (memory helps), ersm-none.
    """
    rows = []
    for path in trajectory_paths:
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                rows.append(json.loads(line))
    frame = pd.DataFrame(rows)
    if drift_only:
        frame = frame[frame["drift_progress"] > 0.0].copy()
    if frame.empty:
        raise ValueError(
            "No drift-active rounds found; run with drift_mode!=stable "
            "or set drift_only=False."
        )
    index_keys = ["seed", "user_id"] if pool_conditions else [
        "seed",
        "user_id",
        "condition",
    ]
    unit = (
        frame.groupby([*index_keys, "memory_mode"])["mean_utility"]
        .mean()
        .reset_index()
    )
    pivot = unit.pivot_table(
        index=index_keys,
        columns="memory_mode",
        values="mean_utility",
    )
    modes = list(pivot.columns)
    out = []
    for index, mode in enumerate(modes):
        values = pivot[mode].dropna().to_numpy()
        out.append(
            {
                "row": f"relevance[{mode}]",
                "mode": mode,
                "mean": float(values.mean()),
                "ci_low": float("nan"),
                "ci_high": float("nan"),
                "p": float("nan"),
                "pairs": int(len(values)),
            }
        )
    contrasts = [("ersm", "raw"), ("ersm", "none"), ("raw", "none")]
    for index, (left, right) in enumerate(contrasts):
        if left not in modes or right not in modes:
            continue
        diffs = (pivot[left] - pivot[right]).dropna().to_numpy()
        if len(diffs) == 0:
            continue
        low, high, p_value = paired_bootstrap(diffs, seed=bootstrap_seed + index)
        out.append(
            {
                "row": f"{left}-{right}",
                "mode": f"{left}-{right}",
                "mean": float(np.mean(diffs)),
                "ci_low": low,
                "ci_high": high,
                "p": p_value,
                "pairs": int(len(diffs)),
            }
        )
    return pd.DataFrame(out)


def analyze_llm_user_metrics(
    frame: pd.DataFrame,
    bootstrap_seed: int = 20260612,
) -> pd.DataFrame:
    frame = frame.copy()
    frame["utility_mean"] = (
        frame["head_cumulative_utility"]
        + frame["tail_cumulative_utility"]
    ) / 2.0
    value_columns = ["path_dependence", "utility_mean"]
    if {
        "head_post_drift_utility",
        "tail_post_drift_utility",
    }.issubset(frame.columns):
        frame["post_drift_utility_mean"] = (
            frame["head_post_drift_utility"]
            + frame["tail_post_drift_utility"]
        ) / 2.0
        value_columns.append("post_drift_utility_mean")
    comparisons = [
        ("raw_minus_eam_path", "path_dependence", "raw", "eam"),
        ("raw_minus_no_memory_path", "path_dependence", "raw", "none"),
        ("raw_minus_summary_path", "path_dependence", "raw", "summary"),
        ("eam_minus_no_memory_path", "path_dependence", "eam", "none"),
        ("eam_minus_no_memory_utility", "utility_mean", "eam", "none"),
        ("raw_minus_ersm_path", "path_dependence", "raw", "ersm"),
        ("ersm_minus_no_memory_path", "path_dependence", "ersm", "none"),
        ("ersm_minus_no_memory_utility", "utility_mean", "ersm", "none"),
        ("eam_minus_ersm_path", "path_dependence", "eam", "ersm"),
        (
            "ersm_no_residual_minus_ersm_path",
            "path_dependence",
            "ersm_no_residual",
            "ersm",
        ),
        (
            "raw_minus_ersm_consolidated_path",
            "path_dependence",
            "raw",
            "ersm_consolidated",
        ),
        (
            "ersm_consolidated_minus_no_memory_path",
            "path_dependence",
            "ersm_consolidated",
            "none",
        ),
        (
            "ersm_consolidated_minus_no_memory_utility",
            "utility_mean",
            "ersm_consolidated",
            "none",
        ),
        (
            "ersm_minus_ersm_consolidated_path",
            "path_dependence",
            "ersm",
            "ersm_consolidated",
        ),
        (
            "raw_minus_ersm_invariant_path",
            "path_dependence",
            "raw",
            "ersm_invariant",
        ),
        (
            "ersm_invariant_minus_no_memory_path",
            "path_dependence",
            "ersm_invariant",
            "none",
        ),
        (
            "ersm_invariant_minus_no_memory_utility",
            "utility_mean",
            "ersm_invariant",
            "none",
        ),
        (
            "ersm_minus_ersm_invariant_path",
            "path_dependence",
            "ersm",
            "ersm_invariant",
        ),
        (
            "ersm_invariant_minus_raw_postdrift_utility",
            "post_drift_utility_mean",
            "ersm_invariant",
            "raw",
        ),
        (
            "ersm_invariant_minus_ersm_postdrift_utility",
            "post_drift_utility_mean",
            "ersm_invariant",
            "ersm",
        ),
        (
            "ersm_minus_raw_postdrift_utility",
            "post_drift_utility_mean",
            "ersm",
            "raw",
        ),
        (
            "ersm_invariant_minus_no_memory_postdrift_utility",
            "post_drift_utility_mean",
            "ersm_invariant",
            "none",
        ),
    ]
    rows = []
    scenario_columns = [
        "model",
        "dataset",
        "drift_mode",
        "feedback_model",
    ]
    for scenario_index, (scenario, group) in enumerate(
        frame.groupby(scenario_columns, sort=True)
    ):
        pivot = group.pivot_table(
            index=["seed", "user_id"],
            columns="memory_mode",
            values=value_columns,
            aggfunc="first",
        )
        for comparison_index, (name, metric, left, right) in enumerate(
            comparisons
        ):
            if (metric, left) not in pivot or (metric, right) not in pivot:
                continue
            differences = (
                pivot[(metric, left)] - pivot[(metric, right)]
            ).dropna()
            if len(differences) == 0:
                continue
            low, high, p_value = paired_bootstrap(
                differences.to_numpy(),
                seed=bootstrap_seed
                + scenario_index * 100
                + comparison_index,
            )
            seed_means = differences.groupby(level="seed").mean()
            rows.append(
                {
                    **dict(zip(scenario_columns, scenario)),
                    "comparison": name,
                    "left_mode": left,
                    "right_mode": right,
                    "metric": metric,
                    "pairs": int(len(differences)),
                    "seeds": int(len(seed_means)),
                    "mean_difference": float(differences.mean()),
                    "ci_low": low,
                    "ci_high": high,
                    "pooled_user_bootstrap_p": p_value,
                    "seed_mean_sd": float(seed_means.std(ddof=1)),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("user_metrics", nargs="+")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    combined = pd.concat(
        [pd.read_csv(path) for path in args.user_metrics],
        ignore_index=True,
    )
    combined.to_csv(output_dir / "combined_llm_user_metrics.csv", index=False)
    effects = analyze_llm_user_metrics(combined)
    effects.to_csv(output_dir / "llm_paired_effects.csv", index=False)
    print(effects.to_string(index=False))


if __name__ == "__main__":
    main()
