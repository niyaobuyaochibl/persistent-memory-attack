"""Robustness of extended-model lifts to parse-failure fallbacks.

For each experiment and mode, recompute the attack lift after dropping every
(seed, user) pair whose recommendation calls under that mode ever failed to
parse (those rounds fall back to the memoryless policy).
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1] / "results" / "revision_summary" / "extended"
failures = pd.read_csv(ROOT / "parse_failures.csv")
rng = np.random.default_rng(20261105)

rows = []
for exp, group in failures.groupby("experiment"):
    units = pd.read_csv(ROOT / "ext_out" / f"{exp}.units.csv")
    merged = units.merge(group, on=["seed", "user_id", "memory_mode"], how="left")
    for mode, sub in merged.groupby("memory_mode"):
        clean = sub[sub["failures"].fillna(0) == 0]
        boot = [clean["lift"].sample(len(clean), replace=True, random_state=rng.integers(1 << 31)).mean()
                for _ in range(1000)]
        rows.append({
            "experiment": exp,
            "mode": mode,
            "pairs_with_failure": int((sub["failures"] > 0).sum()),
            "pairs": len(sub),
            "lift_all": sub["lift"].mean(),
            "lift_clean": clean["lift"].mean(),
            "ci_low": np.quantile(boot, 0.025),
            "ci_high": np.quantile(boot, 0.975),
        })

out = pd.DataFrame(rows)
out.to_csv(ROOT / "parse_robustness.csv", index=False)
with pd.option_context("display.width", 200):
    print(out.round(4).to_string(index=False))
