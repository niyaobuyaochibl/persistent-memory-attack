"""Run every revision config with one shared LLM backend.

usage: python scripts/run_revision_suite.py [--list configs/revision/ALL.txt] [--only SUBSTR ...]
Completed runs (llm_summary.csv present) are skipped, so the suite is resumable.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import time
import traceback

from stickyrec.config import load_config
from stickyrec.llm_agent import make_backend
from stickyrec.llm_runner import run_llm_experiment

PRIORITY = ("main_", "gate_", "weak_", "tail_", "attr_", "hard_", "fact_", "mem0_")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", default="configs/revision")
    parser.add_argument("--only", nargs="*", default=[])
    parser.add_argument("--backend", default="vllm", help="run only configs using this backend")
    parser.add_argument("--n-users", type=int, default=0, help="override for quick checks")
    args = parser.parse_args()
    paths = sorted(Path(args.configs).glob("rev_*.yaml"))
    paths.sort(key=lambda p: next((i for i, tag in enumerate(PRIORITY) if p.stem.startswith("rev_" + tag)), 99))
    if args.only:
        paths = [p for p in paths if any(s in p.stem for s in args.only)]
    backend = None
    backend_key = None
    for path in paths:
        config = load_config(path)
        if str(config.get("backend")) != args.backend:
            continue
        if args.n_users:
            config["n_users"] = args.n_users
            config["output_dir"] = config["output_dir"].replace("/revision/", "/revision_check/")
        if (Path(config["output_dir"]) / "llm_summary.csv").exists():
            print(f"SKIP {path.stem}", flush=True)
            continue
        model_key = (config.get("backend"), config.get("served_model_name"), config.get("model_name"))
        if backend is None or model_key != backend_key:
            backend, backend_key = make_backend(config), model_key
        started = time.time()
        print(f"START {path.stem}", flush=True)
        try:
            summary = run_llm_experiment(config, backend=backend)
            failures = summary[["memory_mode", "parse_failure_rate"]].to_dict("records")
            print(f"DONE {path.stem} {time.time() - started:.0f}s parse={failures}", flush=True)
            if getattr(backend, "usage", None):
                print(f"USAGE {path.stem} {dict(backend.usage)}", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"FAIL {path.stem}\n{traceback.format_exc()}", flush=True)
            if type(exc).__name__ == "UsageLimit":
                print("STOP plan quota exhausted; re-run the same command later to resume", flush=True)
                raise SystemExit(3)


if __name__ == "__main__":
    main()
