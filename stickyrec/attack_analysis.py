"""Early-exposure manipulation attack analysis.

Given attack-mode trajectory logs, reconstruct each user's promoted target
cluster from the logged ``target_item`` and quantify, per memory mode:

* attack lift = post-attack target-cluster share in the attack arm ("head")
  minus the clean arm ("tail"), averaged over users and seeds; and
* the defense effect = reduction in attack lift relative to raw memory.

All contrasts use the user-level paired bootstrap from :mod:`stickyrec.analysis`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .analysis import paired_bootstrap
from .data import load_dataset


def _target_cluster(
    embeddings: np.ndarray, target_item: int, cluster_size: int
) -> set[int]:
    sims = embeddings @ embeddings[target_item]
    return set(np.argsort(-sims)[:cluster_size].tolist())


def _load_logged_clusters(trajectory_paths: list[str | Path]) -> dict[int, set[int]]:
    """Read per-run ``attack_meta.json`` sidecars (exact attack clusters).

    The environment may build the promoted cluster either from embedding
    neighbours or from a target genre (adaptive attack). Reading the logged
    membership keeps the analysis correct regardless of how it was formed.
    """
    clusters: dict[int, set[int]] = {}
    for path in trajectory_paths:
        sidecar = Path(path).parent / "attack_meta.json"
        if not sidecar.exists():
            continue
        with sidecar.open(encoding="utf-8") as handle:
            meta = json.load(handle)
        for entry in meta.values():
            target = int(entry["target_item"])
            clusters[target] = set(int(i) for i in entry["cluster"])
    return clusters


def load_attack_frame(
    trajectory_paths: list[str | Path],
    data_root: str,
    dataset: str,
    cluster_size: int = 80,
    start_round: int = 2,
) -> pd.DataFrame:
    bundle = load_dataset(data_root, dataset)
    rows = []
    for path in trajectory_paths:
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                rows.append(json.loads(line))
    frame = pd.DataFrame(rows)
    frame = frame[frame["round_idx"] >= start_round].copy()
    logged = _load_logged_clusters(trajectory_paths)
    clusters = {
        int(t): logged.get(
            int(t), _target_cluster(bundle.embeddings, int(t), cluster_size)
        )
        for t in frame["target_item"].unique()
        if int(t) >= 0
    }
    frame["target_share"] = frame.apply(
        lambda r: float(
            np.mean([i in clusters[int(r["target_item"])] for i in r["recommended_ids"]])
        ),
        axis=1,
    )
    return frame


def attack_lift_table(
    frame: pd.DataFrame, bootstrap_seed: int = 20260612
) -> pd.DataFrame:
    grouped = (
        frame.groupby(["seed", "user_id", "memory_mode", "condition"])[
            "target_share"
        ]
        .mean()
        .reset_index()
    )
    pivot = grouped.pivot_table(
        index=["seed", "user_id", "memory_mode"],
        columns="condition",
        values="target_share",
    )
    pivot["lift"] = pivot["head"] - pivot["tail"]
    lift = pivot["lift"].reset_index().pivot_table(
        index=["seed", "user_id"], columns="memory_mode", values="lift"
    )
    modes = list(lift.columns)
    rows = []
    for index, mode in enumerate(modes):
        diffs = lift[mode].dropna().to_numpy()
        low, high, p_value = paired_bootstrap(diffs, seed=bootstrap_seed + index)
        rows.append(
            {
                "comparison": f"attack_lift[{mode}]",
                "mode": mode,
                "mean": float(diffs.mean()),
                "ci_low": low,
                "ci_high": high,
                "p": p_value,
                "pairs": int(len(diffs)),
            }
        )
    if "raw" in modes:
        for index, mode in enumerate(m for m in modes if m != "raw"):
            diffs = (lift["raw"] - lift[mode]).dropna().to_numpy()
            low, high, p_value = paired_bootstrap(
                diffs, seed=bootstrap_seed + 100 + index
            )
            rows.append(
                {
                    "comparison": f"defense[raw-{mode}]",
                    "mode": mode,
                    "mean": float(diffs.mean()),
                    "ci_low": low,
                    "ci_high": high,
                    "p": p_value,
                    "pairs": int(len(diffs)),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("trajectories", nargs="+")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--dataset", default="ml-1m")
    parser.add_argument("--cluster-size", type=int, default=80)
    parser.add_argument("--start-round", type=int, default=2)
    parser.add_argument("--output")
    args = parser.parse_args()
    frame = load_attack_frame(
        args.trajectories,
        args.data_root,
        args.dataset,
        args.cluster_size,
        start_round=args.start_round,
    )
    table = attack_lift_table(frame)
    if args.output:
        table.to_csv(args.output, index=False)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
