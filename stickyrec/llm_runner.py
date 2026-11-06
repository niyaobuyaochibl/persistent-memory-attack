from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from .config import load_config
from .data import DatasetBundle, build_semantic_partition, load_dataset, select_users
from .environment import CounterfactualRecEnv, Observation, StepResult
from .llm_agent import StructuredLLMPolicy, generate_many, make_backend
from .memory import PolicyContext
from .metadata import ItemMetadata, load_metadata
from .metrics import trajectory_distance


@dataclass
class Episode:
    user_id: int
    memory_mode: str
    condition: str
    policy: StructuredLLMPolicy
    env: CounterfactualRecEnv
    observation: Observation | None
    result: StepResult | None = None
    next_observation: Observation | None = None


def _noisy_genre_metadata(
    metadata: dict[int, ItemMetadata], noise: float, seed: int
) -> dict[int, ItemMetadata]:
    """Replace each item's genre set, with probability ``noise``, by one
    uniformly drawn genre from the catalogue vocabulary."""
    vocabulary = sorted({g for item in metadata.values() for g in item.attributes})
    rng = np.random.default_rng(seed)
    output = {}
    for item_idx in sorted(metadata):
        item = metadata[item_idx]
        attributes = item.attributes
        if rng.random() < noise:
            attributes = (vocabulary[int(rng.integers(len(vocabulary)))],)
        output[item_idx] = ItemMetadata(
            item.item_idx, item.external_id, item.title, attributes
        )
    return output


def _cluster_theme_metadata(
    config: dict,
    bundle: DatasetBundle,
    metadata: dict[int, ItemMetadata],
    backend,
    output_dir: Path,
) -> dict[int, ItemMetadata]:
    """Attribute-free catalogue: replace curated genres by embedding clusters.

    Items are clustered on their text embeddings; each cluster gets a short
    theme name written by the LLM from exemplar titles only, so no curated
    attribute ever enters the prompt or the ERSM profile.
    """
    n_clusters = int(config.get("attribute_clusters", 20))
    assignments, centroids = build_semantic_partition(
        bundle.embeddings,
        n_clusters=n_clusters,
        seed=int(config.get("attribute_cluster_seed", 0)),
    )
    exemplars: dict[int, list[str]] = {}
    for cluster_id in range(n_clusters):
        members = np.array(
            [
                i
                for i in np.flatnonzero(assignments == cluster_id)
                if int(i) in metadata and bundle.item_counts[i] > 0
            ],
            dtype=np.int64,
        )
        if len(members) == 0:
            exemplars[cluster_id] = []
            continue
        order = np.argsort(-(bundle.embeddings[members] @ centroids[cluster_id]))
        exemplars[cluster_id] = [
            metadata[int(i)].title for i in members[order[:10]]
        ]
    prompts = [
        "These catalogue items belong to one group:\n"
        + "\n".join(f"- {title}" for title in exemplars[cluster_id])
        + "\nName the shared theme in 1 to 3 words. "
        'Return JSON only: {"name": "theme"}'
        for cluster_id in range(n_clusters)
    ]
    names: dict[int, str] = {}
    used: set[str] = set()
    for cluster_id, response in enumerate(generate_many(backend, prompts)):
        try:
            name = " ".join(str(json.loads(response)["name"]).split()[:3])
        except (json.JSONDecodeError, KeyError, TypeError):
            name = ""
        name = name or f"Theme {cluster_id}"
        if name.lower() in used:
            name = f"{name} {cluster_id}"
        used.add(name.lower())
        names[cluster_id] = name
    with (output_dir / "attribute_clusters.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                str(k): {"name": names[k], "exemplars": exemplars[k]}
                for k in range(n_clusters)
            },
            handle,
            indent=2,
            ensure_ascii=False,
        )
    return {
        item_idx: ItemMetadata(
            item.item_idx,
            item.external_id,
            item.title,
            (names[int(assignments[item_idx])],),
        )
        for item_idx, item in metadata.items()
    }


def _policy_metadata(config, bundle, metadata, backend, output_dir):
    source = str(config.get("attribute_source", "genre"))
    if source == "genre":
        return metadata
    if source == "noisy_genre":
        return _noisy_genre_metadata(
            metadata,
            float(config.get("genre_noise", 0.2)),
            int(config.get("attribute_noise_seed", 0)),
        )
    if source == "cluster":
        return _cluster_theme_metadata(config, bundle, metadata, backend, output_dir)
    raise ValueError(f"Unknown attribute_source: {source}")


def _make_policy(config: dict, backend, metadata, memory_mode: str, store=None):
    return StructuredLLMPolicy(
        backend,
        metadata,
        memory_mode=memory_mode,
        history_limit=int(config.get("history_limit", 12)),
        eam_decay=float(config.get("eam_decay", 0.8)),
        eam_propensity_floor=float(config.get("eam_propensity_floor", 0.2)),
        eam_reference_propensity=float(
            config.get("eam_reference_propensity", 0.5)
        ),
        ersm_top_genres=int(config.get("ersm_top_genres", 6)),
        ersm_commit_threshold=int(config.get("ersm_commit_threshold", 2)),
        recency_window=int(config.get("recency_window", 3)),
        candidate_observability=str(config.get("candidate_observability", "full")),
        ersm_user_tolerance=float(config.get("ersm_user_tolerance", 2.0)),
        ersm_user_prior=float(config.get("ersm_user_prior", 3.0)),
        fact_limit=int(config.get("fact_limit", 6)),
        fact_words=int(config.get("fact_words", 12)),
        memory_store=store,
        retrieve_k=int(config.get("retrieve_k", 5)),
    )


def _parallel(workers: int, fn, items: list) -> list:
    if workers <= 1 or len(items) <= 1:
        return [fn(item) for item in items]
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(workers) as pool:
        return list(pool.map(fn, items))


def run_llm_experiment(config: dict, backend=None, store_factory=None) -> pd.DataFrame:
    bundle = load_dataset(
        config["data_root"],
        config["dataset"],
        min_rating=float(config.get("min_rating", 4.0)),
        embedding_name=str(config.get("embedding_name", "all-MiniLM-L6-v2")),
    )
    metadata = load_metadata(
        config["data_root"],
        config["dataset"],
        config.get("movies_path"),
    )
    users = select_users(
        bundle,
        n_users=int(config["n_users"]),
        min_history=int(config.get("min_history", 20)),
        seed=int(config["seed"]),
    )
    assignments, centroids = build_semantic_partition(
        bundle.embeddings,
        n_clusters=int(config.get("semantic_clusters", 16)),
        seed=int(config["seed"]),
    )
    # Per-item primary genre ids, used only by the genre-aligned adaptive
    # attack. Items without metadata get a distinct sentinel id each so they
    # never accidentally form a shared genre cluster.
    item_genres = None
    if str(config.get("attack_cluster_mode", "embedding")) == "genre":
        genre_to_id: dict[str, int] = {}
        item_genres = np.full(bundle.n_items, -1, dtype=np.int64)
        for item_idx in range(bundle.n_items):
            md = metadata.get(item_idx)
            genre = md.attributes[0] if (md and md.attributes) else None
            if genre is None:
                item_genres[item_idx] = -(item_idx + 1)
                continue
            item_genres[item_idx] = genre_to_id.setdefault(genre, len(genre_to_id))
    if backend is None:
        backend = make_backend(config)

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    policy_metadata = _policy_metadata(config, bundle, metadata, backend, output_dir)
    trajectory_path = output_dir / "llm_trajectories.jsonl"
    trace_path = output_dir / "llm_traces.jsonl"
    keep_prompts = bool(config.get("trace_prompts", True))
    top_k = int(config["recommendation_size"])
    drift_mode = str(config.get("drift_mode", "stable"))
    chunk = int(config.get("user_chunk", 20))
    workers = int(config.get("io_workers", 1))
    if store_factory is None and any(m.startswith("mem0") for m in config["memory_modes"]):
        from .mem0_store import Mem0Factory

        store_factory = Mem0Factory(config)
    grouped: dict[tuple[str, int, str], list[dict]] = defaultdict(list)
    trace_rows = []
    attack_meta: dict[int, dict] = {}
    started = time.time()
    with trajectory_path.open("w", encoding="utf-8") as trajectory_file:
        for chunk_start in range(0, len(users), chunk):
            episodes: list[Episode] = []
            for user_id in users[chunk_start : chunk_start + chunk]:
                context = PolicyContext(
                    base_profile=bundle.user_profile(user_id),
                    history_items=bundle.histories[user_id],
                    embeddings=bundle.embeddings,
                    popularity=bundle.popularity,
                    cluster_assignments=assignments,
                    cluster_centroids=centroids,
                )
                plans = {}
                for condition in config["conditions"]:
                    plan_env = CounterfactualRecEnv(
                        bundle, config, preference_anchors=centroids,
                        item_genres=item_genres,
                    )
                    plan_env.reset(user_id, condition, drift_mode=drift_mode)
                    plans[condition] = plan_env.candidate_plan
                    if plan_env.attack_target is not None:
                        attack_meta[int(user_id)] = {
                            "target_item": int(plan_env.attack_target),
                            "cluster": [int(i) for i in plan_env.attack_cluster],
                        }
                for memory_mode in config["memory_modes"]:
                    for condition in config["conditions"]:
                        store = None
                        if memory_mode.startswith("mem0"):
                            store = store_factory(
                                f"s{config['seed']}_u{user_id}_{memory_mode}_{condition}"
                            )
                        policy = _make_policy(
                            config, backend, policy_metadata, memory_mode, store
                        )
                        policy.reset(context)
                        if int(user_id) in attack_meta:
                            policy.set_audit(attack_meta[int(user_id)]["cluster"])
                        env = CounterfactualRecEnv(
                            bundle, config, preference_anchors=centroids,
                            item_genres=item_genres,
                        )
                        observation = env.reset(
                            user_id,
                            condition,
                            drift_mode=drift_mode,
                            candidate_plan=plans[condition],
                        )
                        episodes.append(
                            Episode(
                                int(user_id), memory_mode, condition,
                                policy, env, observation,
                            )
                        )

            active = list(episodes)
            while active:
                prompts = _parallel(
                    workers,
                    lambda ep: ep.policy.recommendation_prompt(ep.observation, top_k),
                    active,
                )
                responses = generate_many(backend, prompts)
                actions = {}
                for ep, prompt, response in zip(active, prompts, responses):
                    action = ep.policy.apply_recommendation(
                        ep.observation, top_k, prompt, response
                    )
                    actions[id(ep)] = action
                    ep.next_observation, ep.result = ep.env.step(action)
                memory_jobs = [
                    (ep, ep.policy.memory_prompt(ep.observation, ep.result))
                    for ep in active
                ]
                memory_jobs = [(ep, p) for ep, p in memory_jobs if p is not None]
                memory_responses = dict(
                    zip(
                        (id(ep) for ep, _ in memory_jobs),
                        generate_many(backend, [p for _, p in memory_jobs]),
                    )
                )
                _parallel(
                    workers,
                    lambda ep: ep.policy.observe(
                        ep.observation,
                        ep.result,
                        memory_response=memory_responses.get(id(ep)),
                    ),
                    active,
                )
                for ep in active:
                    result = ep.result
                    shown = result.recommended_ids.astype(int)
                    cluster = set(map(int, ep.env.attack_cluster))
                    record = {
                        "model": config["model_name"],
                        "dataset": config["dataset"],
                        "drift_mode": result.drift_mode,
                        "feedback_model": str(config["feedback_model"]),
                        "seed": int(config["seed"]),
                        "memory_mode": ep.memory_mode,
                        "condition": ep.condition,
                        "user_id": ep.user_id,
                        "round_idx": int(result.round_idx),
                        "recommended_ids": shown.tolist(),
                        "agent_ids": actions[id(ep)].astype(int).tolist(),
                        "clicked_item": (
                            None
                            if result.clicked_item is None
                            else int(result.clicked_item)
                        ),
                        "reward": float(result.reward),
                        "mean_utility": float(result.utilities.mean()),
                        "rank_relevance": float(
                            ep.env.rank_relevance(shown, result.round_idx).mean()
                        ),
                        "mean_popularity": float(result.popularities.mean()),
                        "drift_progress": float(result.drift_progress),
                        "target_item": (
                            int(ep.env.attack_target)
                            if ep.env.attack_target is not None
                            else -1
                        ),
                        "target_share": (
                            float(np.mean([int(i) in cluster for i in shown]))
                            if cluster
                            else float("nan")
                        ),
                    }
                    record.update(ep.policy.audit())
                    trajectory_file.write(json.dumps(record) + "\n")
                    grouped[(ep.memory_mode, ep.user_id, ep.condition)].append(record)
                    ep.observation = ep.next_observation
                active = [ep for ep in active if ep.observation is not None]

            for ep in episodes:
                for trace in ep.policy.trace:
                    row = {
                        "model": config["model_name"],
                        "dataset": config["dataset"],
                        "drift_mode": drift_mode,
                        "feedback_model": str(config["feedback_model"]),
                        "seed": int(config["seed"]),
                        "memory_mode": ep.memory_mode,
                        "condition": ep.condition,
                        "user_id": ep.user_id,
                        **trace,
                    }
                    if not keep_prompts:
                        row.pop("prompt", None)
                    trace_rows.append(row)
            print(
                f"[{time.time() - started:7.1f}s] users "
                f"{min(chunk_start + chunk, len(users))}/{len(users)} done",
                flush=True,
            )

    with trace_path.open("w", encoding="utf-8") as trace_file:
        for row in trace_rows:
            trace_file.write(json.dumps(row) + "\n")

    if attack_meta:
        with (output_dir / "attack_meta.json").open("w", encoding="utf-8") as handle:
            json.dump({str(k): v for k, v in attack_meta.items()}, handle)

    summary_rows = []
    user_metric_rows = []
    start_round = int(config["intervention_rounds"])
    for memory_mode in config["memory_modes"]:
        user_path = []
        utilities = defaultdict(list)
        for user_id in users:
            by_condition = {
                condition: grouped[(memory_mode, user_id, condition)]
                for condition in config["conditions"]
            }
            path_dependence = trajectory_distance(
                by_condition["head"],
                by_condition["tail"],
                start_round=start_round,
            )
            user_path.append(path_dependence)
            user_utilities = {
                condition: float(
                    np.sum([record["reward"] for record in records])
                )
                for condition, records in by_condition.items()
            }
            for condition, value in user_utilities.items():
                utilities[condition].append(value)
            user_post_drift = {
                condition: float(
                    np.sum(
                        [
                            record["reward"]
                            for record in records
                            if record.get("drift_progress", 0.0) > 0.0
                        ]
                    )
                )
                for condition, records in by_condition.items()
            }
            user_metric_rows.append(
                {
                    "model": config["model_name"],
                    "dataset": config["dataset"],
                    "drift_mode": drift_mode,
                    "feedback_model": str(config["feedback_model"]),
                    "seed": int(config["seed"]),
                    "memory_mode": memory_mode,
                    "user_id": int(user_id),
                    "path_dependence": path_dependence,
                    "head_cumulative_utility": user_utilities["head"],
                    "tail_cumulative_utility": user_utilities["tail"],
                    "head_post_drift_utility": user_post_drift["head"],
                    "tail_post_drift_utility": user_post_drift["tail"],
                }
            )
        mode_traces = [
            row
            for row in trace_rows
            if row["memory_mode"] == memory_mode
            and row["event"] == "recommend"
        ]
        recommendation_failures = sum(
            row["parse_error"] is not None for row in mode_traces
        )
        memory_traces = [
            row
            for row in trace_rows
            if row["memory_mode"] == memory_mode
            and row["event"] in {"memory_summary", "memory_facts"}
        ]
        memory_failures = sum(
            row["parse_error"] is not None for row in memory_traces
        )
        summary_rows.append(
            {
                "model": config["model_name"],
                "dataset": config["dataset"],
                "drift_mode": drift_mode,
                "feedback_model": str(config["feedback_model"]),
                "seed": int(config["seed"]),
                "memory_mode": memory_mode,
                "users": len(users),
                "path_dependence_mean": float(np.mean(user_path)),
                "head_cumulative_utility_mean": float(np.mean(utilities["head"])),
                "tail_cumulative_utility_mean": float(np.mean(utilities["tail"])),
                "recommendation_calls": len(mode_traces),
                "parse_failures": recommendation_failures,
                "parse_failure_rate": recommendation_failures
                / max(len(mode_traces), 1),
                "memory_summary_calls": len(memory_traces),
                "memory_summary_parse_failures": memory_failures,
                "memory_summary_parse_failure_rate": memory_failures
                / max(len(memory_traces), 1),
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_dir / "llm_summary.csv", index=False)
    pd.DataFrame(user_metric_rows).to_csv(
        output_dir / "llm_user_metrics.csv", index=False
    )
    with (output_dir / "llm_config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    args = parser.parse_args()
    config = load_config(args.config)
    print(run_llm_experiment(config).to_string(index=False))


if __name__ == "__main__":
    main()
