"""Paired user-clustered contrasts between two memory policies in one experiment."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stickyrec.revision_analysis import clustered_bootstrap  # noqa: E402

PAIRS = [
    ("tail_full_ml1m", "ersm_ips_only", "ersm_slate_only"),
    ("tail_full_ml1m", "ersm_ips_only", "ersm"),
    ("tail_f3_ml1m", "ersm_ips_only", "ersm_slate_only"),
    ("main_ml1m", "ersm_ips_only", "ersm_slate_only"),
    ("main_cds", "ersm_ips_only", "ersm_slate_only"),
    ("main_ml1m", "ersm_ips_only", "ersm_no_residual"),
    ("main_cds", "ersm_ips_only", "ersm_no_residual"),
    ("main_cds", "ersm", "ersm_invariant"),
    ("main_ml1m", "raw_recent", "raw"),
    ("main_cds", "raw_recent", "raw"),
    ("gate_ml1m", "raw_gate", "raw"),
    ("gate_cds", "raw_gate", "raw"),
    ("mem0_ml1m", "mem0_ersm", "mem0"),
    ("fact_ml1m", "factmem_ersm", "factmem"),
    ("fact_ml1m", "factmem", "raw"),
]

for exp, a, b in PAIRS:
    u = pd.read_csv(f"results/revision_summary/{exp}.units.csv")
    w = u.pivot_table(index=["seed", "user_id"], columns="memory_mode", values="lift").dropna(subset=[a, b])
    diff = (w[a] - w[b]).reset_index(drop=False)
    m, lo, hi, p = clustered_bootstrap(w[a] - w[b], pd.Series(w.index.get_level_values("user_id")), seed=7)
    print(f"{exp:16s} {a:16s} - {b:16s} {m:+.3f} [{lo:+.3f},{hi:+.3f}] p={p:.4f}")
