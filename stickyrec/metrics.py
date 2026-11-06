from __future__ import annotations

from collections import Counter
from typing import Iterable

import numpy as np


def jaccard_distance(left: Iterable[int], right: Iterable[int]) -> float:
    left_set = set(map(int, left))
    right_set = set(map(int, right))
    union = left_set | right_set
    if not union:
        return 0.0
    return 1.0 - len(left_set & right_set) / len(union)


def trajectory_distance(
    left: list[dict],
    right: list[dict],
    start_round: int,
) -> float:
    left_by_round = {
        int(record["round_idx"]): record
        for record in left
        if int(record["round_idx"]) >= start_round
    }
    right_by_round = {
        int(record["round_idx"]): record
        for record in right
        if int(record["round_idx"]) >= start_round
    }
    rounds = sorted(set(left_by_round) & set(right_by_round))
    if not rounds:
        return float("nan")
    distances = [
        jaccard_distance(
            left_by_round[round_idx]["recommended_ids"],
            right_by_round[round_idx]["recommended_ids"],
        )
        for round_idx in rounds
    ]
    return float(np.mean(distances))


def gini(values: Iterable[int]) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    if len(array) == 0 or np.all(array == 0):
        return 0.0
    array = np.sort(array)
    n = len(array)
    index = np.arange(1, n + 1)
    return float(np.sum((2 * index - n - 1) * array) / (n * array.sum()))


def hhi(item_ids: Iterable[int]) -> float:
    counts = np.asarray(list(Counter(map(int, item_ids)).values()), dtype=np.float64)
    if not len(counts):
        return 0.0
    shares = counts / counts.sum()
    return float(np.square(shares).sum())


def condition_metrics(
    records: list[dict],
    tail_threshold: float = 0.2,
    adaptation_patience: int = 2,
) -> dict:
    recommendations = [
        int(item_id)
        for record in records
        for item_id in record["recommended_ids"]
    ]
    popularities = [
        float(value)
        for record in records
        for value in record["popularities"]
    ]
    rewards = [float(record["reward"]) for record in records]
    regrets = [float(record.get("utility_regret", np.nan)) for record in records]
    def progress(record: dict) -> float:
        if "drift_progress" in record:
            return float(record["drift_progress"])
        return 1.0 if record.get("drift_active", False) else 0.0

    pre_drift = [
        float(record.get("utility_regret", np.nan))
        for record in records
        if progress(record) == 0.0
    ]
    transition = [
        float(record.get("utility_regret", np.nan))
        for record in records
        if 0.0 < progress(record) < 1.0
    ]
    post_drift = [record for record in records if progress(record) >= 1.0]
    post_regrets = [float(record["utility_regret"]) for record in post_drift]
    recovered = [bool(record.get("recovered", False)) for record in post_drift]
    adaptation_delay = float("nan")
    if post_drift:
        adaptation_delay = float(len(post_drift) + 1)
        for index in range(max(len(recovered) - adaptation_patience + 1, 0)):
            if all(recovered[index : index + adaptation_patience]):
                adaptation_delay = float(index)
                break
    return {
        "cumulative_utility": float(np.sum(rewards)),
        "mean_popularity": float(np.mean(popularities)),
        "tail_exposure_rate": float(
            np.mean(np.asarray(popularities) <= tail_threshold)
        ),
        "exposure_gini": gini(Counter(recommendations).values()),
        "exposure_hhi": hhi(recommendations),
        "clicks": int(sum(record["clicked_item"] is not None for record in records)),
        "mean_utility_regret": float(np.nanmean(regrets)),
        "pre_drift_mean_regret": (
            float(np.nanmean(pre_drift)) if pre_drift else float("nan")
        ),
        "transition_mean_regret": (
            float(np.nanmean(transition)) if transition else float("nan")
        ),
        "post_drift_mean_regret": (
            float(np.nanmean(post_regrets)) if post_regrets else float("nan")
        ),
        "preference_recovery_rate": (
            float(np.mean(recovered)) if recovered else float("nan")
        ),
        "adaptation_delay": adaptation_delay,
    }



def bootstrap_mean_ci(
    values: Iterable[float],
    seed: int,
    n_bootstrap: int = 2000,
    confidence: float = 0.95,
) -> tuple[float, float]:
    array = np.asarray(list(values), dtype=np.float64)
    if len(array) == 0:
        return float("nan"), float("nan")
    if np.all(array == array[0]):
        return float(array[0]), float(array[0])
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(array), size=(n_bootstrap, len(array)))
    means = array[indices].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    low, high = np.quantile(means, [alpha, 1.0 - alpha])
    return float(low), float(high)
