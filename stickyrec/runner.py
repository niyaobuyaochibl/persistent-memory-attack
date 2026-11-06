from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import sys

import numpy as np
import pandas as pd

from .config import load_config
from .data import build_semantic_partition, load_dataset, select_users
from .environment import CounterfactualRecEnv, Observation, StepResult
from .memory import PolicyContext, make_policy
from .metrics import bootstrap_mean_ci, condition_metrics, trajectory_distance


def _json_record(
    dataset: str,
    policy_name: str,
    condition: str,
    observation: Observation,
    result: StepResult,
    candidate_utilities: np.ndarray,
    candidate_popularities: np.ndarray,
    recovery_threshold: float,
) -> dict:
    return {
        "dataset": dataset,
        "policy": policy_name,
        "condition": condition,
        "user_id": int(result.user_id),
        "round_idx": int(result.round_idx),
        "candidate_ids": observation.candidate_ids.astype(int).tolist(),
        "candidate_utility_mean": float(candidate_utilities.mean()),
        "candidate_utility_std": float(candidate_utilities.std()),
        "candidate_popularity_mean": float(candidate_popularities.mean()),
        "oracle_utility_mean": float(
            np.sort(candidate_utilities)[-len(result.recommended_ids) :].mean()
        ),
        "recommendation_utility_mean": float(result.utilities.mean()),
        "utility_regret": float(
            np.sort(candidate_utilities)[-len(result.recommended_ids) :].mean()
            - result.utilities.mean()
        ),
        "recovered": bool(
            np.sort(candidate_utilities)[-len(result.recommended_ids) :].mean()
            - result.utilities.mean()
            <= recovery_threshold
        ),
        "recommended_ids": result.recommended_ids.astype(int).tolist(),
        "clicked_item": (
            None if result.clicked_item is None else int(result.clicked_item)
        ),
        "reward": float(result.reward),
        "utilities": result.utilities.astype(float).tolist(),
        "popularities": result.popularities.astype(float).tolist(),
        "intervention_active": bool(result.intervention_active),
        "drift_mode": result.drift_mode,
        "drift_active": bool(result.drift_active),
        "drift_progress": float(result.drift_progress),
    }


def run_experiment(config: dict) -> dict[str, pd.DataFrame]:
    bundle = load_dataset(
        config["data_root"],
        config["dataset"],
        min_rating=float(config.get("min_rating", 4.0)),
    )
    users = select_users(
        bundle,
        n_users=int(config["n_users"]),
        min_history=int(config.get("min_history", 20)),
        seed=int(config["seed"]),
    )
    assignments, centroids = build_semantic_partition(
        bundle.embeddings,
        n_clusters=int(config.get("semantic_clusters", 12)),
        seed=int(config["seed"]),
    )
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    trajectory_path = output_dir / "trajectories.jsonl"
    drift_mode = str(config.get("drift_mode", "stable"))
    recovery_threshold = float(config.get("recovery_regret_threshold", 0.02))

    grouped_records: dict[tuple[str, int, str], list[dict]] = defaultdict(list)
    with trajectory_path.open("w", encoding="utf-8") as trajectory_file:
        for user_id in users:
            context = PolicyContext(
                base_profile=bundle.user_profile(user_id),
                history_items=bundle.histories[user_id],
                embeddings=bundle.embeddings,
                popularity=bundle.popularity,
                cluster_assignments=assignments,
                cluster_centroids=centroids,
            )
            candidate_plans = {}
            for condition in config["conditions"]:
                plan_env = CounterfactualRecEnv(
                    bundle, config, preference_anchors=centroids
                )
                plan_env.reset(user_id, condition, drift_mode=drift_mode)
                candidate_plans[condition] = plan_env.candidate_plan
            for policy_name in config["policies"]:
                for condition in config["conditions"]:
                    policy = make_policy(policy_name, config)
                    policy.reset(context)
                    env = CounterfactualRecEnv(
                        bundle, config, preference_anchors=centroids
                    )
                    observation = env.reset(
                        user_id,
                        condition,
                        drift_mode=drift_mode,
                        candidate_plan=candidate_plans[condition],
                    )
                    while observation is not None:
                        candidate_utilities = env.hidden_utility(
                            observation.candidate_ids
                        )
                        candidate_popularities = bundle.popularity[
                            observation.candidate_ids
                        ]
                        action = policy.recommend(
                            observation, int(config["recommendation_size"])
                        )
                        next_observation, result = env.step(action)
                        policy.observe(observation, result)
                        record = _json_record(
                            bundle.name,
                            policy_name,
                            condition,
                            observation,
                            result,
                            candidate_utilities,
                            candidate_popularities,
                            recovery_threshold,
                        )
                        trajectory_file.write(json.dumps(record) + "\n")
                        grouped_records[(policy_name, user_id, condition)].append(
                            record
                        )
                        observation = next_observation

    user_rows = []
    intervention_rounds = int(config["intervention_rounds"])
    for policy_name in config["policies"]:
        for user_id in users:
            by_condition = {
                condition: grouped_records[(policy_name, user_id, condition)]
                for condition in config["conditions"]
            }
            head = by_condition["head"]
            tail = by_condition["tail"]
            post_head = [
                row for row in head if row["round_idx"] >= intervention_rounds
            ]
            post_tail = [
                row for row in tail if row["round_idx"] >= intervention_rounds
            ]
            row = {
                "policy": policy_name,
                "user_id": user_id,
                "drift_mode": drift_mode,
                "path_dependence": trajectory_distance(
                    head, tail, start_round=intervention_rounds
                ),
                "post_popularity_gap": abs(
                    condition_metrics(post_head)["mean_popularity"]
                    - condition_metrics(post_tail)["mean_popularity"]
                ),
                "intervention_utility_gap": abs(
                    float(
                        np.mean(
                            [
                                record["candidate_utility_mean"]
                                for record in head[:intervention_rounds]
                            ]
                        )
                    )
                    - float(
                        np.mean(
                            [
                                record["candidate_utility_mean"]
                                for record in tail[:intervention_rounds]
                            ]
                        )
                    )
                ),
                "intervention_recommended_utility_gap": abs(
                    float(
                        np.mean(
                            [
                                utility
                                for record in head[:intervention_rounds]
                                for utility in record["utilities"]
                            ]
                        )
                    )
                    - float(
                        np.mean(
                            [
                                utility
                                for record in tail[:intervention_rounds]
                                for utility in record["utilities"]
                            ]
                        )
                    )
                ),
            }
            for condition, records in by_condition.items():
                metrics = condition_metrics(
                    records,
                    adaptation_patience=int(config.get("adaptation_patience", 2)),
                )
                for key, value in metrics.items():
                    row[f"{condition}_{key}"] = value
            user_rows.append(row)

    user_metrics = pd.DataFrame(user_rows)
    no_memory_by_user = (
        user_metrics.loc[user_metrics["policy"] == "no_memory"]
        .set_index("user_id")["path_dependence"]
    )
    user_metrics["memory_amplification"] = user_metrics.apply(
        lambda row: row["path_dependence"]
        - float(no_memory_by_user.loc[row["user_id"]]),
        axis=1,
    )
    summary_rows = []
    for policy_index, (policy_name, frame) in enumerate(
        user_metrics.groupby("policy", sort=False)
    ):
        pd_mean = float(frame["path_dependence"].mean())
        ma_mean = float(frame["memory_amplification"].mean())
        pd_low, pd_high = bootstrap_mean_ci(
            frame["path_dependence"],
            seed=int(config["seed"]) + policy_index * 2,
        )
        ma_low, ma_high = bootstrap_mean_ci(
            frame["memory_amplification"],
            seed=int(config["seed"]) + policy_index * 2 + 1,
        )
        summary_rows.append(
            {
                "policy": policy_name,
                "users": len(frame),
                "path_dependence_mean": pd_mean,
                "path_dependence_std": float(
                    frame["path_dependence"].std(ddof=1)
                ),
                "path_dependence_ci_low": pd_low,
                "path_dependence_ci_high": pd_high,
                "memory_amplification": ma_mean,
                "memory_amplification_ci_low": ma_low,
                "memory_amplification_ci_high": ma_high,
                "post_popularity_gap_mean": float(
                    frame["post_popularity_gap"].mean()
                ),
                "intervention_utility_gap_mean": float(
                    frame["intervention_utility_gap"].mean()
                ),
                "intervention_recommended_utility_gap_mean": float(
                    frame["intervention_recommended_utility_gap"].mean()
                ),
                "head_cumulative_utility_mean": float(
                    frame["head_cumulative_utility"].mean()
                ),
                "tail_cumulative_utility_mean": float(
                    frame["tail_cumulative_utility"].mean()
                ),
                "head_mean_popularity": float(
                    frame["head_mean_popularity"].mean()
                ),
                "tail_mean_popularity": float(
                    frame["tail_mean_popularity"].mean()
                ),
                "head_transition_regret_mean": float(
                    frame["head_transition_mean_regret"].mean()
                ),
                "tail_transition_regret_mean": float(
                    frame["tail_transition_mean_regret"].mean()
                ),
                "head_post_drift_regret_mean": float(
                    frame["head_post_drift_mean_regret"].mean()
                ),
                "tail_post_drift_regret_mean": float(
                    frame["tail_post_drift_mean_regret"].mean()
                ),
                "head_recovery_rate_mean": float(
                    frame["head_preference_recovery_rate"].mean()
                ),
                "tail_recovery_rate_mean": float(
                    frame["tail_preference_recovery_rate"].mean()
                ),
                "head_adaptation_delay_mean": float(
                    frame["head_adaptation_delay"].mean()
                ),
                "tail_adaptation_delay_mean": float(
                    frame["tail_adaptation_delay"].mean()
                ),
            }
        )
    summary = pd.DataFrame(summary_rows)
    user_metrics.to_csv(output_dir / "user_metrics.csv", index=False)
    summary.to_csv(output_dir / "summary.csv", index=False)

    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "dataset": bundle.name,
        "drift_mode": drift_mode,
        "n_items": bundle.n_items,
        "embedding_dim": bundle.embedding_dim,
        "selected_users": users,
        "config": config,
    }
    with (output_dir / "run_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    return {"user_metrics": user_metrics, "summary": summary}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Path to a YAML experiment config")
    args = parser.parse_args()
    config = load_config(args.config)
    outputs = run_experiment(config)
    print(outputs["summary"].to_string(index=False))


if __name__ == "__main__":
    main()

