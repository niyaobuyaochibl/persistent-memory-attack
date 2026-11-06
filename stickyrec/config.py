from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


REQUIRED_KEYS = {
    "data_root",
    "dataset",
    "output_dir",
    "seed",
    "n_users",
    "rounds",
    "intervention_rounds",
    "candidate_size",
    "recommendation_size",
}


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a mapping: {config_path}")

    missing = sorted(REQUIRED_KEYS - config.keys())
    if missing:
        raise ValueError(f"Missing config keys: {', '.join(missing)}")
    if config["intervention_rounds"] >= config["rounds"]:
        raise ValueError("intervention_rounds must be smaller than rounds")
    if config["recommendation_size"] > config["candidate_size"]:
        raise ValueError("recommendation_size cannot exceed candidate_size")
    if config["dataset"] not in {"ml-1m", "amazon_books", "amazon_cds", "yelp", "kuairec"}:
        raise ValueError("dataset must be ml-1m, amazon_books, amazon_cds, yelp, or kuairec")
    drift_mode = config.get("drift_mode", "stable")
    if drift_mode not in {"stable", "gradual", "abrupt"}:
        raise ValueError("drift_mode must be stable, gradual, or abrupt")
    if drift_mode != "stable":
        drift_start = int(config.get("drift_start_round", 4))
        if drift_start < config["intervention_rounds"]:
            raise ValueError("drift_start_round must not precede intervention end")
        if drift_start >= config["rounds"]:
            raise ValueError("drift_start_round must be smaller than rounds")
        if int(config.get("drift_duration", 4)) < 1:
            raise ValueError("drift_duration must be positive")
        magnitude = float(config.get("drift_magnitude", 0.8))
        if not 0.0 <= magnitude <= 1.0:
            raise ValueError("drift_magnitude must lie in [0, 1]")
        target_similarity = float(config.get("drift_target_similarity", 0.1))
        if not -1.0 <= target_similarity <= 1.0:
            raise ValueError("drift_target_similarity must lie in [-1, 1]")
    return config

