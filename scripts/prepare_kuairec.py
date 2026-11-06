"""Convert KuaiRec into the stickyrec dataset layout with fully observed feedback.

Candidates come from the small matrix, where (almost) every user-video watch ratio
is logged, so the environment can answer any recommendation with the user's real
response. Pre-session histories are the same users' strong positives in the big
matrix on videos outside the small matrix; popularity counts come from the big
matrix.

usage: python scripts/prepare_kuairec.py --raw KuaiRec/data --data-root DIR --embedder DIR
"""

from __future__ import annotations

import argparse
from pathlib import Path
import pickle
import re

import numpy as np
import pandas as pd

UNKNOWN = {"", "UNKNOWN", "nan"}


def clean(text: object) -> str:
    text = "" if pd.isna(text) else str(text)
    text = re.sub(r"#\S+", " ", text)
    text = re.sub(r"@\S+", " ", text)
    return " ".join(text.split())


def title_of(row) -> str:
    for field in ("caption", "manual_cover_text"):
        text = clean(row.get(field))
        if text and text not in UNKNOWN:
            return text[:30]
    tags = clean(str(row.get("topic_tag", "")).strip("[]").replace(",", " "))
    return tags[:30] if tags and tags not in UNKNOWN else f"Video {row['video_id']}"


def attributes_of(row) -> tuple[str, ...]:
    out = []
    for field in ("first_level_category_name", "second_level_category_name"):
        value = "" if pd.isna(row.get(field)) else str(row.get(field)).strip()
        if value not in UNKNOWN and value not in out:
            out.append(value)
    return tuple(out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--embedder", required=True)
    parser.add_argument("--embedding-name", default="paraphrase-multilingual-MiniLM-L12-v2")
    parser.add_argument("--positive-ratio", type=float, default=2.0)
    args = parser.parse_args()
    raw = Path(args.raw)

    small = pd.read_csv(raw / "small_matrix.csv", usecols=["user_id", "video_id", "watch_ratio"])
    big = pd.read_csv(raw / "big_matrix.csv", usecols=["user_id", "video_id", "watch_ratio"])
    captions = pd.read_csv(raw / "kuairec_caption_category.csv", engine="python")
    small_items = set(small["video_id"].unique())
    small_users = sorted(small["user_id"].unique())

    videos = sorted(set(big["video_id"]) | small_items)
    item_index = {int(v): i for i, v in enumerate(videos)}
    other_users = sorted(set(big["user_id"]) - set(small_users))
    user_index = {int(u): i for i, u in enumerate(list(small_users) + other_users)}

    positive = (big["watch_ratio"] >= args.positive_ratio) & ~big["video_id"].isin(small_items)
    train = pd.DataFrame({
        "user_idx": big["user_id"].map(user_index).astype(np.int64),
        "item_idx": big["video_id"].map(item_index).astype(np.int64),
        "item_id": big["video_id"].astype(str),
        "rating": np.where(positive, 5.0, 1.0),
    })
    empty = train.iloc[0:0]

    feedback = {}
    for user_id, group in small.groupby("user_id"):
        row = np.full(len(videos), np.nan, dtype=np.float32)
        row[group["video_id"].map(item_index).to_numpy()] = group["watch_ratio"].to_numpy(np.float32)
        feedback[user_index[int(user_id)]] = row

    captions["video_id"] = pd.to_numeric(captions["video_id"], errors="coerce")
    captions = captions.dropna(subset=["video_id"]).astype({"video_id": np.int64})
    captions = captions.drop_duplicates("video_id").set_index("video_id")
    metadata, texts = {}, []
    for video in videos:
        row = captions.loc[video].to_dict() if video in captions.index else {}
        row["video_id"] = video
        title, attrs = title_of(row), attributes_of(row)
        metadata[item_index[int(video)]] = (title, attrs)
        third = "" if pd.isna(row.get("third_level_category_name")) else str(row.get("third_level_category_name"))
        tags = clean(str(row.get("topic_tag", "")).strip("[]").replace(",", " "))
        texts.append(" ".join([*attrs, third if third not in UNKNOWN else "", clean(row.get("caption")), tags]).strip() or title)

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(args.embedder, device="cuda")
    vectors = model.encode(texts, batch_size=256, normalize_embeddings=True, show_progress_bar=False)

    out = Path(args.data_root) / "datasets" / "kuairec"
    out.mkdir(parents=True, exist_ok=True)
    for name, frame in (("train", train), ("val", empty), ("test", empty)):
        with (out / f"{name}.pkl").open("wb") as handle:
            pickle.dump(frame, handle)
    with (out / "item_metadata.pkl").open("wb") as handle:
        pickle.dump(metadata, handle)
    with (out / "feedback.pkl").open("wb") as handle:
        pickle.dump(feedback, handle)
    emb_dir = Path(args.data_root) / "embeddings" / "kuairec"
    emb_dir.mkdir(parents=True, exist_ok=True)
    with (emb_dir / f"{args.embedding_name}.pkl").open("wb") as handle:
        pickle.dump({i: vectors[i] for i in range(len(videos))}, handle)

    counts = train["item_idx"].value_counts()
    small_idx = [item_index[int(v)] for v in small_items]
    history_len = train[(train["rating"] >= 4) & (train["user_idx"] < len(small_users))].groupby("user_idx").size()
    observed = np.mean([np.mean(~np.isnan(feedback[u][small_idx])) for u in feedback])
    n_attrs = len({a for _, attrs in metadata.values() for a in attrs})
    print(f"items {len(videos)} (small {len(small_items)}), small users {len(small_users)}, all users {len(user_index)}")
    print(f"small items with big-matrix popularity: {np.mean([i in counts.index for i in small_idx]):.3f}")
    print(f"observed share of small matrix per user: {observed:.4f}")
    print(f"pre-session history length (watch_ratio>={args.positive_ratio}, outside small matrix): "
          f"users {len(history_len)}, median {history_len.median():.0f}, min {history_len.min()}, "
          f">=30: {(history_len >= 30).sum()}")
    print(f"attributes (category names): {n_attrs}; items without attributes: "
          f"{sum(1 for _, a in metadata.values() if not a)}")
    print("example items:", [metadata[i] for i in small_idx[:5]])


if __name__ == "__main__":
    main()
