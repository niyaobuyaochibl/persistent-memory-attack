"""Adapter from the open-source mem0 library to the policy's memory_store API.

Each episode gets its own mem0 ``Memory`` (separate vector store and history
DB) so arms and modes never share state; the embedding model is loaded once
and shared. mem0's own extraction and update prompts are used unchanged.
"""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import threading


def normalise_extraction(response):
    """Coerce ``{"memory": ["fact", ...]}`` into mem0's ``[{"id", "text"}]`` items.

    Returns (response, repaired). Content is never changed, only the shape.
    """
    if not isinstance(response, str):
        return response, False
    from mem0.memory.utils import remove_code_blocks

    try:
        data = json.loads(remove_code_blocks(response), strict=False)
    except (json.JSONDecodeError, TypeError):
        return response, False
    items = data.get("memory") if isinstance(data, dict) else None
    if not isinstance(items, list):
        return response, False
    fixed, repaired = [], False
    for index, item in enumerate(items):
        if isinstance(item, dict) and item.get("text"):
            fixed.append(item)
            continue
        repaired = True
        if isinstance(item, dict):
            text = next((item[k] for k in ("memory", "fact", "content") if item.get(k)), "")
        else:
            text = item
        if text:
            fixed.append({"id": str(index), "text": str(text)})
    if not repaired:
        return response, False
    data["memory"] = fixed
    return json.dumps(data), True


class Mem0Store:
    def __init__(self, memory, user_id: str):
        self.memory = memory
        self.user_id = user_id

    @staticmethod
    def _texts(result) -> list[str]:
        items = result.get("results", []) if isinstance(result, dict) else result
        return [str(item.get("memory", "")) for item in items if item.get("memory")]

    def add(self, text: str) -> None:
        self.memory.add([{"role": "user", "content": text}], user_id=self.user_id)

    def search(self, query: str, k: int) -> list[str]:
        return self._texts(
            self.memory.search(query, top_k=k, filters={"user_id": self.user_id})
        )

    def all(self) -> list[str]:
        return self._texts(
            self.memory.get_all(filters={"user_id": self.user_id}, top_k=10_000)
        )


class Mem0Factory:
    def __init__(self, config: dict):
        self.config = config
        self.root = Path(config.get("mem0_dir", "results/mem0_runs")) / Path(
            config["output_dir"]
        ).name
        if self.root.exists():
            shutil.rmtree(self.root)
        self.root.mkdir(parents=True)
        self._embedder = None
        self._lock = threading.Lock()

    def _memory_config(self, key: str) -> dict:
        dims = int(self.config.get("mem0_embedding_dims", 384))
        return {
            "llm": {
                "provider": "openai",
                "config": {
                    "model": self.config["served_model_name"],
                    "openai_base_url": self.config["openai_base_url"],
                    "api_key": "EMPTY",
                    "temperature": 0.0,
                    "max_tokens": int(self.config.get("mem0_max_tokens", 512)),
                },
            },
            "embedder": {
                "provider": "huggingface",
                "config": {
                    "model": self.config["mem0_embedder"],
                    "embedding_dims": dims,
                },
            },
            "vector_store": {
                "provider": "faiss",
                "config": {
                    "collection_name": "mem",
                    "path": str(self.root / key / "faiss"),
                    "embedding_model_dims": dims,
                },
            },
            "history_db_path": str(self.root / key / "history.db"),
        }

    def __call__(self, key: str) -> Mem0Store:
        from mem0 import Memory
        from mem0.utils import factory

        with self._lock:
            if self._embedder is None:
                memory = Memory.from_config(self._memory_config(key))
                self._embedder = memory.embedding_model
            else:
                original = factory.EmbedderFactory.create
                factory.EmbedderFactory.create = lambda *args, **kwargs: self._embedder
                try:
                    memory = Memory.from_config(self._memory_config(key))
                finally:
                    factory.EmbedderFactory.create = original
        self._wrap_llm(memory, key)
        return Mem0Store(memory, key)

    def _wrap_llm(self, memory, key: str) -> None:
        generate = memory.llm.generate_response

        def wrapped(*args, **kwargs):
            response, repaired = normalise_extraction(generate(*args, **kwargs))
            if repaired:
                with self._lock, (self.root / "repairs.log").open("a") as log:
                    log.write(key + "\n")
            return response

        memory.llm.generate_response = wrapped
