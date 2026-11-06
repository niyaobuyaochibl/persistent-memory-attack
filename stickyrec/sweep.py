from __future__ import annotations

import argparse
from copy import deepcopy
from itertools import product
from pathlib import Path

import pandas as pd
import yaml

from .config import load_config
from .runner import run_experiment


def _pareto_mask(frame: pd.DataFrame) -> list[bool]:
    mask = []
    for index, row in frame.iterrows():
        dominated = False
        for other_index, other in frame.iterrows():
            if index == other_index:
                continue
            no_worse = (
                other["stable_utility"] >= row["stable_utility"]
                and other["abrupt_post_regret"] <= row["abrupt_post_regret"]
            )
            strictly_better = (
                other["stable_utility"] > row["stable_utility"]
                or other["abrupt_post_regret"] < row["abrupt_post_regret"]
            )
            if no_worse and strictly_better:
                dominated = True
                break
        mask.append(not dominated)
    return mask


def run_sweep(path: str | Path) -> dict[str, pd.DataFrame]:
    sweep_path = Path(path)
    with sweep_path.open("r", encoding="utf-8") as handle:
        sweep = yaml.safe_load(handle)
    base_path = Path(sweep["base_config"])
    if not base_path.is_absolute():
        base_path = sweep_path.parent / base_path
    base = load_config(base_path)
    output_root = Path(sweep["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    grid = sweep["parameter_grid"]
    parameter_names = list(grid)
    combinations = list(product(*(grid[name] for name in parameter_names)))

    frames = []
    for combination_index, values in enumerate(combinations):
        parameters = dict(zip(parameter_names, values))
        for drift_mode in sweep.get("drift_modes", ["stable", "abrupt"]):
            for feedback_model in sweep.get("feedback_models", ["deterministic"]):
                for seed in sweep.get("seeds", [base["seed"]]):
                    config = deepcopy(base)
                    config.update(sweep.get("overrides", {}))
                    config.update(parameters)
                    config["seed"] = int(seed)
                    config["drift_mode"] = drift_mode
                    config["feedback_model"] = feedback_model
                    run_name = (
                        f"grid{combination_index:02d}_{drift_mode}_"
                        f"{feedback_model}_seed{seed}"
                    )
                    config["output_dir"] = str(output_root / run_name)
                    summary = run_experiment(config)["summary"].copy()
                    summary.insert(0, "seed", int(seed))
                    summary.insert(0, "feedback_model", feedback_model)
                    summary.insert(0, "drift_mode", drift_mode)
                    summary.insert(0, "combination", combination_index)
                    for name, value in reversed(list(parameters.items())):
                        summary.insert(1, name, value)
                    frames.append(summary)

    runs = pd.concat(frames, ignore_index=True)
    runs.to_csv(output_root / "sweep_runs.csv", index=False)
    eam = runs[runs["policy"] == "eam"].copy()
    eam["condition_utility"] = (
        eam["head_cumulative_utility_mean"]
        + eam["tail_cumulative_utility_mean"]
    ) / 2.0
    eam["condition_post_regret"] = (
        eam["head_post_drift_regret_mean"]
        + eam["tail_post_drift_regret_mean"]
    ) / 2.0
    rows = []
    group_keys = ["combination", *parameter_names]
    for key, frame in eam.groupby(group_keys, sort=True, dropna=False):
        key_values = key if isinstance(key, tuple) else (key,)
        row = dict(zip(group_keys, key_values))
        stable = frame[frame["drift_mode"] == "stable"]
        abrupt = frame[frame["drift_mode"] == "abrupt"]
        row.update(
            stable_utility=float(stable["condition_utility"].mean()),
            abrupt_post_regret=float(abrupt["condition_post_regret"].mean()),
            stable_path_dependence=float(stable["path_dependence_mean"].mean()),
            abrupt_path_dependence=float(abrupt["path_dependence_mean"].mean()),
        )
        rows.append(row)
    candidates = pd.DataFrame(rows)
    eligible = (
        (candidates["stable_path_dependence"] <= 0.05)
        & (candidates["abrupt_path_dependence"] <= 0.05)
    )
    candidates["pareto"] = False
    candidates.loc[eligible, "pareto"] = _pareto_mask(candidates.loc[eligible])
    candidates = candidates.sort_values(
        ["pareto", "abrupt_post_regret", "stable_utility"],
        ascending=[False, True, False],
    )
    candidates.to_csv(output_root / "eam_candidates.csv", index=False)
    return {"runs": runs, "candidates": candidates}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Path to a YAML sweep config")
    args = parser.parse_args()
    outputs = run_sweep(args.config)
    print(outputs["candidates"].to_string(index=False))


if __name__ == "__main__":
    main()
