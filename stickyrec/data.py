from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pickle
from typing import Iterable

import numpy as np
import pandas as pd


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return matrix / norms


def _to_numpy(value: object) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value, dtype=np.float32)


@dataclass(frozen=True)
class DatasetBundle:
    name: str
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    embeddings: np.ndarray
    popularity: np.ndarray
    item_counts: np.ndarray
    histories: dict[int, np.ndarray]
    # Fully observed datasets (KuaiRec): user_idx -> per-item observed feedback,
    # NaN where the user's response to the item is unknown.
    feedback: dict[int, np.ndarray] | None = None

    @property
    def n_items(self) -> int:
        return int(self.embeddings.shape[0])

    @property
    def embedding_dim(self) -> int:
        return int(self.embeddings.shape[1])

    @property
    def user_ids(self) -> list[int]:
        return sorted(self.histories)

    def user_profile(self, user_id: int) -> np.ndarray:
        items = self.histories[user_id]
        profile = self.embeddings[items].mean(axis=0)
        norm = np.linalg.norm(profile)
        return profile / norm if norm > 0 else profile


def _standardize_interactions(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    if "user_idx" not in frame or "item_idx" not in frame:
        raise ValueError("Expected user_idx and item_idx columns")
    if "rating" not in frame:
        frame["rating"] = 1.0
    return frame[["user_idx", "item_idx", "rating"]]


def _load_embeddings(path: Path) -> np.ndarray:
    with path.open("rb") as handle:
        raw = pickle.load(handle)
    if not isinstance(raw, dict) or not raw:
        raise ValueError(f"Expected non-empty embedding dictionary: {path}")

    n_items = max(int(item_id) for item_id in raw) + 1
    first = _to_numpy(next(iter(raw.values())))
    matrix = np.zeros((n_items, first.shape[0]), dtype=np.float32)
    for item_id, vector in raw.items():
        matrix[int(item_id)] = _to_numpy(vector)
    return _normalize_rows(matrix)


def _percentile_ranks(counts: np.ndarray) -> np.ndarray:
    order = np.argsort(counts, kind="stable")
    ranks = np.empty_like(order, dtype=np.float32)
    if len(order) == 1:
        ranks[order] = 0.5
        return ranks
    ranks[order] = np.linspace(0.0, 1.0, len(order), dtype=np.float32)
    return ranks


def _build_histories(
    train: pd.DataFrame,
    n_items: int,
    min_rating: float,
) -> tuple[dict[int, np.ndarray], np.ndarray]:
    positive = train[train["rating"] >= min_rating]
    histories: dict[int, np.ndarray] = {}
    for user_id, group in positive.groupby("user_idx", sort=False):
        items = group["item_idx"].astype(int).to_numpy()
        items = items[(items >= 0) & (items < n_items)]
        if len(items):
            histories[int(user_id)] = np.unique(items)

    counts = np.zeros(n_items, dtype=np.int64)
    item_ids, item_counts = np.unique(
        train["item_idx"].astype(int).to_numpy(), return_counts=True
    )
    valid = (item_ids >= 0) & (item_ids < n_items)
    counts[item_ids[valid]] = item_counts[valid]
    return histories, counts


def load_dataset(
    data_root: str | Path,
    dataset: str,
    min_rating: float = 4.0,
    embedding_name: str = "all-MiniLM-L6-v2",
) -> DatasetBundle:
    root = Path(data_root)
    dataset_root = root / "datasets" / dataset
    embedding_path = root / "embeddings" / dataset / f"{embedding_name}.pkl"

    frames = {}
    for split in ("train", "val", "test"):
        with (dataset_root / f"{split}.pkl").open("rb") as handle:
            frames[split] = _standardize_interactions(pickle.load(handle))

    embeddings = _load_embeddings(embedding_path)
    histories, counts = _build_histories(frames["train"], len(embeddings), min_rating)
    popularity = _percentile_ranks(counts)
    feedback = None
    feedback_path = dataset_root / "feedback.pkl"
    if feedback_path.exists():
        with feedback_path.open("rb") as handle:
            feedback = {int(u): np.asarray(v, dtype=np.float32) for u, v in pickle.load(handle).items()}
    return DatasetBundle(
        name=dataset,
        train=frames["train"],
        val=frames["val"],
        test=frames["test"],
        embeddings=embeddings,
        popularity=popularity,
        item_counts=counts,
        histories=histories,
        feedback=feedback,
    )


def select_users(
    bundle: DatasetBundle,
    n_users: int,
    min_history: int,
    seed: int,
) -> list[int]:
    eligible = [
        user_id
        for user_id, history in bundle.histories.items()
        if len(history) >= min_history
        and (bundle.feedback is None or user_id in bundle.feedback)
    ]
    if len(eligible) < n_users:
        raise ValueError(
            f"Requested {n_users} users but only {len(eligible)} meet min_history"
        )
    rng = np.random.default_rng(seed)
    return sorted(rng.choice(eligible, size=n_users, replace=False).tolist())


def build_semantic_partition(
    embeddings: np.ndarray,
    n_clusters: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if n_clusters < 2 or n_clusters > len(embeddings):
        raise ValueError("n_clusters must be between 2 and n_items")
    rng = np.random.default_rng(seed)
    anchor_ids = rng.choice(len(embeddings), size=n_clusters, replace=False)
    centroids = embeddings[anchor_ids].copy()

    for _ in range(2):
        assignments = np.argmax(embeddings @ centroids.T, axis=1)
        for cluster_id in range(n_clusters):
            members = embeddings[assignments == cluster_id]
            if len(members):
                centroid = members.mean(axis=0)
                norm = np.linalg.norm(centroid)
                centroids[cluster_id] = centroid / norm if norm > 0 else centroid
    assignments = np.argmax(embeddings @ centroids.T, axis=1)
    return assignments.astype(np.int32), _normalize_rows(centroids)

