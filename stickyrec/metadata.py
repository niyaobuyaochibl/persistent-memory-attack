from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pickle

import pandas as pd


@dataclass(frozen=True)
class ItemMetadata:
    item_idx: int
    external_id: str
    title: str
    attributes: tuple[str, ...]

    def compact_text(self) -> str:
        attributes = ", ".join(self.attributes) if self.attributes else "unknown"
        return f"{self.title} [{attributes}]"


def internal_to_external_map(
    data_root: str | Path,
    dataset: str,
) -> dict[int, str]:
    dataset_root = Path(data_root) / "datasets" / dataset
    pairs = []
    for split in ("train", "val", "test"):
        with (dataset_root / f"{split}.pkl").open("rb") as handle:
            frame = pickle.load(handle)
        if not isinstance(frame, pd.DataFrame):
            raise ValueError(f"Expected DataFrame in {split}.pkl")
        if "item_id" not in frame or "item_idx" not in frame:
            raise ValueError(
                f"{dataset} does not retain item_id to item_idx mapping"
            )
        pairs.append(frame[["item_idx", "item_id"]])
    mapping_frame = pd.concat(pairs, ignore_index=True).drop_duplicates()
    conflicts = mapping_frame.groupby("item_idx")["item_id"].nunique()
    if bool((conflicts > 1).any()):
        raise ValueError("One internal item index maps to multiple external IDs")
    return {
        int(row.item_idx): str(row.item_id)
        for row in mapping_frame.itertuples(index=False)
    }


def load_pickled_metadata(
    data_root: str | Path,
    dataset: str,
) -> dict[int, ItemMetadata]:
    """Load item metadata for datasets that ship an ``item_metadata.pkl``.

    The pickle maps ``item_idx -> (title, (genre, ...))``. External IDs are
    recovered from the interaction frames via ``internal_to_external_map`` so
    that downstream consumers keep a stable handle on each catalogue item.
    """
    metadata_path = Path(data_root) / "datasets" / dataset / "item_metadata.pkl"
    with metadata_path.open("rb") as handle:
        raw = pickle.load(handle)
    if not isinstance(raw, dict) or not raw:
        raise ValueError(f"Expected non-empty metadata dict: {metadata_path}")

    external_map = internal_to_external_map(data_root, dataset)
    output: dict[int, ItemMetadata] = {}
    for item_idx, value in raw.items():
        title, attributes = value
        output[int(item_idx)] = ItemMetadata(
            item_idx=int(item_idx),
            external_id=str(external_map.get(int(item_idx), item_idx)),
            title=str(title),
            attributes=tuple(attributes),
        )
    return output


def load_metadata(
    data_root: str | Path,
    dataset: str,
    movies_path: str | Path | None = None,
) -> dict[int, ItemMetadata]:
    """Dispatch to the right metadata loader for ``dataset``."""
    if dataset == "ml-1m":
        if movies_path is None:
            raise ValueError("ml-1m metadata requires movies_path")
        return load_ml1m_metadata(data_root, movies_path)
    return load_pickled_metadata(data_root, dataset)


def load_ml1m_metadata(
    data_root: str | Path,
    movies_path: str | Path,
) -> dict[int, ItemMetadata]:
    internal_map = internal_to_external_map(data_root, "ml-1m")
    external_metadata: dict[str, tuple[str, tuple[str, ...]]] = {}
    with Path(movies_path).open("r", encoding="latin-1") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            parts = raw_line.rstrip("\n").split("::")
            if len(parts) != 3:
                raise ValueError(f"Malformed movies.dat line {line_number}")
            external_id, title, genres = parts
            attributes = tuple(value for value in genres.split("|") if value)
            external_metadata[external_id] = (title, attributes)

    output = {}
    for item_idx, external_id in internal_map.items():
        value = external_metadata.get(external_id)
        if value is None:
            continue
        title, attributes = value
        output[item_idx] = ItemMetadata(
            item_idx=item_idx,
            external_id=external_id,
            title=title,
            attributes=attributes,
        )
    if not output:
        raise ValueError("No processed ML-1M items matched movies.dat")
    return output
