from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path

import pandas as pd
import yaml

from .config import load_config
from .runner import run_experiment


def load_batch_config(path: str | Path) -> tuple[dict, dict]:
    batch_path = Path(path)
    with batch_path.open("r", encoding="utf-8") as handle:
        batch = yaml.safe_load(handle)
    if not isinstance(batch, dict):
        raise ValueError("Batch config must be a mapping")
    for key in ("base_config", "output_root", "seeds", "feedback_models"):
        if key not in batch:
            raise ValueError(f"Missing batch config key: {key}")
    base_path = Path(batch["base_config"])
    if not base_path.is_absolute():
        base_path = batch_path.parent / base_path
    base = load_config(base_path)
    return batch, base


def run_batch(path: str | Path) -> dict[str, pd.DataFrame]:
    batch, base = load_batch_config(path)
    output_root = Path(batch["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    overrides = batch.get("overrides", {})

    summary_frames = []
    user_frames = []
    drift_modes = batch.get("drift_modes", [base.get("drift_mode", "stable")])
    for drift_mode in drift_modes:
        for feedback_model in batch["feedback_models"]:
            for seed in batch["seeds"]:
                config = deepcopy(base)
                config.update(overrides)
                config["seed"] = int(seed)
                config["feedback_model"] = str(feedback_model)
                config["drift_mode"] = str(drift_mode)
                run_name = f"{drift_mode}_{feedback_model}_seed{seed}"
                config["output_dir"] = str(output_root / run_name)
                outputs = run_experiment(config)

                summary = outputs["summary"].copy()
                summary.insert(0, "seed", int(seed))
                summary.insert(0, "feedback_model", str(feedback_model))
                summary.insert(0, "drift_mode", str(drift_mode))
                summary_frames.append(summary)

                users = outputs["user_metrics"].copy()
                users.insert(0, "seed", int(seed))
                users.insert(0, "feedback_model", str(feedback_model))
                user_frames.append(users)

    run_summaries = pd.concat(summary_frames, ignore_index=True)
    user_metrics = pd.concat(user_frames, ignore_index=True)
    aggregate = (
        run_summaries.groupby(
            ["drift_mode", "feedback_model", "policy"], sort=False
        )
        .agg(
            seeds=("seed", "nunique"),
            path_dependence_mean=("path_dependence_mean", "mean"),
            path_dependence_seed_std=("path_dependence_mean", "std"),
            memory_amplification_mean=("memory_amplification", "mean"),
            memory_amplification_seed_std=("memory_amplification", "std"),
            post_popularity_gap_mean=("post_popularity_gap_mean", "mean"),
            intervention_utility_gap_mean=(
                "intervention_utility_gap_mean",
                "mean",
            ),
            intervention_recommended_utility_gap_mean=(
                "intervention_recommended_utility_gap_mean",
                "mean",
            ),
            head_cumulative_utility_mean=(
                "head_cumulative_utility_mean",
                "mean",
            ),
            tail_cumulative_utility_mean=(
                "tail_cumulative_utility_mean",
                "mean",
            ),
            head_transition_regret_mean=(
                "head_transition_regret_mean",
                "mean",
            ),
            tail_transition_regret_mean=(
                "tail_transition_regret_mean",
                "mean",
            ),
            head_post_drift_regret_mean=(
                "head_post_drift_regret_mean",
                "mean",
            ),
            tail_post_drift_regret_mean=(
                "tail_post_drift_regret_mean",
                "mean",
            ),
            head_recovery_rate_mean=("head_recovery_rate_mean", "mean"),
            tail_recovery_rate_mean=("tail_recovery_rate_mean", "mean"),
            head_adaptation_delay_mean=(
                "head_adaptation_delay_mean",
                "mean",
            ),
            tail_adaptation_delay_mean=(
                "tail_adaptation_delay_mean",
                "mean",
            ),
        )
        .reset_index()
    )
    contrast_rows = []
    pivot = run_summaries.pivot_table(
        index=["drift_mode", "feedback_model", "seed"],
        columns="policy",
        values="path_dependence_mean",
        aggfunc="first",
    )
    required = {
        "no_memory",
        "raw_memory",
        "raw_no_early_write",
        "raw_reset",
        "eam",
    }
    if required.issubset(pivot.columns):
        for (drift_mode, feedback_model, seed), row in pivot.iterrows():
            contrast_rows.append(
                {
                    "drift_mode": drift_mode,
                    "feedback_model": feedback_model,
                    "seed": int(seed),
                    "raw_total_effect": row["raw_memory"] - row["no_memory"],
                    "direct_early_write_effect": (
                        row["raw_memory"] - row["raw_no_early_write"]
                    ),
                    "post_feedback_carryover": (
                        row["raw_no_early_write"] - row["no_memory"]
                    ),
                    "effect_removed_by_reset": (
                        row["raw_memory"] - row["raw_reset"]
                    ),
                    "eam_effect_reduction": row["raw_memory"] - row["eam"],
                    "eam_residual_effect": row["eam"] - row["no_memory"],
                }
            )
    contrasts = pd.DataFrame(contrast_rows)
    if len(contrasts):
        contrast_summary = (
            contrasts.groupby(["drift_mode", "feedback_model"], sort=False)
            .agg(
                seeds=("seed", "nunique"),
                raw_total_effect=("raw_total_effect", "mean"),
                direct_early_write_effect=("direct_early_write_effect", "mean"),
                post_feedback_carryover=("post_feedback_carryover", "mean"),
                effect_removed_by_reset=("effect_removed_by_reset", "mean"),
                eam_effect_reduction=("eam_effect_reduction", "mean"),
                eam_residual_effect=("eam_residual_effect", "mean"),
            )
            .reset_index()
        )
    else:
        contrast_summary = pd.DataFrame()

    run_summaries.to_csv(output_root / "run_summaries.csv", index=False)
    user_metrics.to_csv(output_root / "combined_user_metrics.csv", index=False)
    aggregate.to_csv(output_root / "aggregate.csv", index=False)
    contrasts.to_csv(output_root / "causal_contrasts_per_seed.csv", index=False)
    contrast_summary.to_csv(output_root / "causal_contrasts.csv", index=False)
    with (output_root / "batch_config.json").open("w", encoding="utf-8") as handle:
        json.dump(batch, handle, indent=2)
    return {
        "run_summaries": run_summaries,
        "user_metrics": user_metrics,
        "aggregate": aggregate,
        "causal_contrasts": contrast_summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Path to a batch YAML config")
    args = parser.parse_args()
    outputs = run_batch(args.config)
    print(outputs["aggregate"].to_string(index=False))
    if len(outputs["causal_contrasts"]):
        print("\nCausal contrasts")
        print(outputs["causal_contrasts"].to_string(index=False))


if __name__ == "__main__":
    main()
