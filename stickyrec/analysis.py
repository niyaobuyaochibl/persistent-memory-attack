from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def paired_bootstrap(
    differences: np.ndarray,
    seed: int,
    n_bootstrap: int = 5000,
) -> tuple[float, float, float]:
    values = np.asarray(differences, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return float("nan"), float("nan"), float("nan")
    if np.all(values == values[0]):
        p_value = 0.0 if values[0] != 0.0 else 1.0
        return float(values[0]), float(values[0]), p_value
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(n_bootstrap, len(values)))
    means = values[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    p_value = min(
        1.0,
        2.0 * min(float(np.mean(means <= 0.0)), float(np.mean(means >= 0.0))),
    )
    return float(low), float(high), p_value


def analyze_user_metrics(
    frame: pd.DataFrame,
    bootstrap_seed: int = 20260612,
) -> pd.DataFrame:
    frame = frame.copy()
    frame["utility_mean"] = (
        frame["head_cumulative_utility"] + frame["tail_cumulative_utility"]
    ) / 2.0
    frame["post_regret_mean"] = (
        frame["head_post_drift_mean_regret"]
        + frame["tail_post_drift_mean_regret"]
    ) / 2.0
    comparisons = [
        ("raw_minus_eam_path", "path_dependence", "raw_memory", "eam"),
        (
            "raw_minus_delayed_path",
            "path_dependence",
            "raw_memory",
            "eam_delayed",
        ),
        ("eam_minus_no_memory_utility", "utility_mean", "eam", "no_memory"),
        (
            "no_memory_minus_eam_regret",
            "post_regret_mean",
            "no_memory",
            "eam",
        ),
    ]
    rows = []
    scenario_columns = ["drift_mode", "feedback_model"]
    pair_columns = ["seed", "user_id"]
    for scenario_index, (scenario, group) in enumerate(
        frame.groupby(scenario_columns, sort=True)
    ):
        for comparison_index, (name, metric, left, right) in enumerate(comparisons):
            pivot = group.pivot_table(
                index=pair_columns,
                columns="policy",
                values=metric,
                aggfunc="first",
            )
            if left not in pivot or right not in pivot:
                continue
            differences = (pivot[left] - pivot[right]).dropna()
            if len(differences) == 0:
                continue
            low, high, p_value = paired_bootstrap(
                differences.to_numpy(),
                seed=bootstrap_seed + scenario_index * 100 + comparison_index,
            )
            seed_means = differences.groupby(level="seed").mean()
            rows.append(
                {
                    "drift_mode": scenario[0],
                    "feedback_model": scenario[1],
                    "comparison": name,
                    "left_policy": left,
                    "right_policy": right,
                    "metric": metric,
                    "positive_means": f"{left} > {right}",
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
    parser.add_argument("user_metrics", help="combined_user_metrics.csv")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    effects = analyze_user_metrics(pd.read_csv(args.user_metrics))
    effects.to_csv(output_dir / "paired_effects.csv", index=False)
    print(effects.to_string(index=False))


if __name__ == "__main__":
    main()
