"""CPU smoke test: run one tiny seed of every revision experiment with a fake LLM.

usage: python scripts/smoke_revision.py [--users 3] [--only PATTERN]
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import time

from stickyrec.config import load_config
from stickyrec.llm_runner import run_llm_experiment
from stickyrec.revision_analysis import load_runs, revision_table, unit_table


class FakeBackend:
    """Deterministic stand-in that returns schema-valid JSON for each prompt type."""

    def generate(self, prompt: str) -> str:
        return self.generate_batch([prompt])[0]

    def generate_batch(self, prompts: list[str]) -> list[str]:
        out = []
        for prompt in prompts:
            if '"selected_ids"' in prompt:
                ids = re.findall(r"^- (\d+):", prompt, flags=re.M)
                k = int(re.search(r"Select exactly (\d+)", prompt).group(1))
                digest = hashlib.md5(prompt.encode()).digest()
                order = sorted(ids, key=lambda i: hashlib.md5(digest + i.encode()).digest())
                out.append(json.dumps({"selected_ids": [int(i) for i in order[:k]]}))
            elif '"facts"' in prompt:
                out.append(json.dumps({"facts": ["Likes the clicked item style"]}))
            elif '"name"' in prompt:
                out.append(json.dumps({"name": "Shared Theme"}))
            else:
                out.append("{}")
        return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--users", type=int, default=3)
    parser.add_argument("--only", default="")
    parser.add_argument("--configs", default="configs/revision")
    parser.add_argument("--out", default="results/smoke_revision")
    args = parser.parse_args()
    seen = set()
    backend = FakeBackend()
    for path in sorted(Path(args.configs).glob("rev_*.yaml")):
        experiment = re.sub(r"_seed\d+$", "", path.stem)
        if experiment in seen or args.only not in path.stem:
            continue
        seen.add(experiment)
        config = load_config(path)
        config["n_users"] = args.users
        config["output_dir"] = f"{args.out}/{path.stem}"
        started = time.time()
        run_llm_experiment(config, backend=backend)
        table = revision_table(unit_table(load_runs([config["output_dir"]])), experiment)
        lift = table[table["metric"] == "attack_lift"].set_index("mode")["mean"].round(3)
        print(f"OK {experiment:28s} {time.time() - started:6.1f}s lift={lift.to_dict()}", flush=True)


if __name__ == "__main__":
    main()
