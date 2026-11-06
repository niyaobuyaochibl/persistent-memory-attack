"""Build the ML-1M and Amazon CDs datasets in the layout stickyrec.data expects.

    python scripts/prepare_data.py ml1m --raw ml-1m --data-root ../niche_user_study/data
    python scripts/prepare_data.py cds --raw amazon_2023 --data-root ../niche_user_study/data

Writes datasets/<name>/{train,val,test}.pkl and embeddings/<name>/all-MiniLM-L6-v2.pkl
under --data-root; CDs also writes item_metadata.pkl and stats.json.

ML-1M (GroupLens ml-1m.zip): items with at least 5 ratings; each user's ratings sorted
by time and split 80/10/10. The original split broke timestamp ties in an unrecorded
order, so data/splits/ml1m_split_overrides.csv lists the 1% of (user, movie) pairs whose
split differs from a stable sort; applying it reproduces the paper's split exactly.

Amazon CDs (Amazon Reviews 2023, CDs_and_Vinyl): the official 5-core timestamp split
(benchmark/5core/timestamp/CDs_and_Vinyl.{train,valid,test}.csv[.gz]) and the item
metadata (raw/meta_categories/meta_CDs_and_Vinyl.jsonl[.gz]); the 6,000 most-rated training items
with a title and genre, and users with at least 30 positive (>= 4) training ratings.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

EMBEDDER = "sentence-transformers/all-MiniLM-L6-v2"
OVERRIDES = Path(__file__).resolve().parents[1] / "data" / "splits" / "ml1m_split_overrides.csv"
COLUMNS = ["user_id", "item_id", "rating", "timestamp", "user_idx", "item_idx"]


def encode(texts: list[str]) -> dict[int, np.ndarray]:
    from sentence_transformers import SentenceTransformer

    vectors = SentenceTransformer(EMBEDDER).encode(
        texts, batch_size=256, show_progress_bar=True, convert_to_numpy=True
    )
    return {i: vectors[i].astype(np.float32) for i in range(len(texts))}


def write(data_root: Path, name: str, splits: dict[str, pd.DataFrame], embeddings: dict) -> None:
    out = data_root / "datasets" / name
    out.mkdir(parents=True, exist_ok=True)
    for split, frame in splits.items():
        with (out / f"{split}.pkl").open("wb") as handle:
            pickle.dump(frame[COLUMNS].reset_index(drop=True), handle)
        print(f"{name} {split}: {len(frame)} rows")
    emb_dir = data_root / "embeddings" / name
    emb_dir.mkdir(parents=True, exist_ok=True)
    with (emb_dir / "all-MiniLM-L6-v2.pkl").open("wb") as handle:
        pickle.dump(embeddings, handle)
    print(f"{name} embeddings: {len(embeddings)}")


def prepare_ml1m(raw: Path, data_root: Path) -> None:
    ratings = pd.read_csv(raw / "ratings.dat", sep="::", engine="python",
                          names=["user_id", "item_id", "rating", "timestamp"])
    counts = ratings["item_id"].value_counts()
    ratings = ratings[ratings["item_id"].isin(counts[counts >= 5].index)]
    ratings = ratings.sort_values(["user_id", "timestamp"], kind="stable")
    position = ratings.groupby("user_id").cumcount()
    size = ratings.groupby("user_id")["user_id"].transform("size")
    split = np.where(position < (size * 0.8).astype(int), "train",
                     np.where(position < (size * 0.9).astype(int), "val", "test"))
    overrides = pd.read_csv(OVERRIDES)
    fixed = dict(zip(zip(overrides["user_id"], overrides["item_id"]), overrides["split"]))
    ratings["split"] = [fixed.get(key, s) for key, s in
                        zip(zip(ratings["user_id"], ratings["item_id"]), split)]

    items = sorted(ratings["item_id"].unique())
    users = sorted(ratings["user_id"].unique())
    ratings["item_idx"] = ratings["item_id"].map({item: i for i, item in enumerate(items)})
    ratings["user_idx"] = ratings["user_id"].map({user: i for i, user in enumerate(users)})
    splits = {s: ratings[ratings["split"] == s] for s in ("train", "val", "test")}

    movies = pd.read_csv(raw / "movies.dat", sep="::", engine="python", encoding="latin-1",
                         names=["item_id", "title", "genres"]).set_index("item_id")
    texts = [f"{movies.at[item, 'title']}. Genres: {', '.join(movies.at[item, 'genres'].split('|'))}"
             for item in items]
    write(data_root, "ml-1m", splits, encode(texts))


def _is_boilerplate(token: str) -> bool:
    low = token.lower()
    return ("today's deals" in low or "todays deals" in low or "$" in token
            or low.startswith("cds ") or low in {"cds & vinyl", "all genres"})


def _open_text(path: Path):
    """Open ``path`` or, if absent, its ``.gz`` download."""
    if path.exists():
        return path.open(encoding="utf-8")
    import gzip

    return gzip.open(path.with_name(path.name + ".gz"), "rt", encoding="utf-8")


def prepare_cds(raw: Path, data_root: Path, item_cap: int = 6000, min_rating: float = 4.0) -> None:
    split_dir = raw / "benchmark" / "5core" / "timestamp"
    frames = {}
    for split, name in (("train", "train"), ("val", "valid"), ("test", "test")):
        with _open_text(split_dir / f"CDs_and_Vinyl.{name}.csv") as handle:
            frames[split] = pd.read_csv(handle).rename(columns={"parent_asin": "item_id"})
    counts = frames["train"]["item_id"].value_counts()
    candidates = set(counts.index)
    meta: dict[str, tuple[str, tuple[str, ...]]] = {}
    with _open_text(raw / "raw" / "meta_categories" / "meta_CDs_and_Vinyl.jsonl") as handle:
        for line in handle:
            record = json.loads(line)
            asin = record.get("parent_asin")
            if asin not in candidates:
                continue
            title = (record.get("title") or "").strip()
            genres = tuple([c for c in (record.get("categories") or [])[1:]
                            if c and not _is_boilerplate(c)][:2])
            if title and genres:
                meta[asin] = (title[:160], genres)

    kept = [asin for asin in counts.index if asin in meta][:item_cap]
    item_idx = {asin: i for i, asin in enumerate(kept)}
    train = frames["train"][frames["train"]["item_id"].isin(item_idx)]
    positive = train[train["rating"] >= min_rating].groupby("user_id")["item_id"].nunique()
    users = sorted(positive[positive >= 30].index)
    user_idx = {user: i for i, user in enumerate(users)}

    splits = {}
    for split, frame in frames.items():
        frame = frame[frame["item_id"].isin(item_idx) & frame["user_id"].isin(user_idx)].copy()
        frame["item_idx"] = frame["item_id"].map(item_idx).astype(int)
        frame["user_idx"] = frame["user_id"].map(user_idx).astype(int)
        splits[split] = frame

    metadata = {item_idx[asin]: meta[asin] for asin in kept}
    texts = [f"{metadata[i][0]} [{', '.join(metadata[i][1])}]" for i in range(len(kept))]
    write(data_root, "amazon_cds", splits, encode(texts))
    out = data_root / "datasets" / "amazon_cds"
    with (out / "item_metadata.pkl").open("wb") as handle:
        pickle.dump(metadata, handle)
    (out / "stats.json").write_text(json.dumps({"num_users": len(users), "num_items": len(kept)}))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=["ml1m", "cds"])
    parser.add_argument("--raw", required=True, help="ml-1m/ directory, or the Amazon Reviews 2023 root")
    parser.add_argument("--data-root", default="../niche_user_study/data")
    args = parser.parse_args()
    prepare = prepare_ml1m if args.dataset == "ml1m" else prepare_cds
    prepare(Path(args.raw), Path(args.data_root))


if __name__ == "__main__":
    main()
