"""Print every revision experiment as lift / reduction / clean relevance rows."""
import sys

import pandas as pd

s = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else "results/revision_summary/all.csv")
METRICS = sys.argv[2].split(",") if len(sys.argv) > 2 else [
    "attack_lift", "relative_reduction_vs_raw", "relevance_clean", "relevance_clean_vs_none",
]


def fmt(r):
    if pd.isna(r["p"]):
        return f"{r['mean']:.3f} [{r['ci_low']:.3f},{r['ci_high']:.3f}]"
    holm = f" holm={r['p_holm']:.3f}" if pd.notna(r.get("p_holm")) else ""
    return f"{r['mean']:+.3f} [{r['ci_low']:+.3f},{r['ci_high']:+.3f}] p={r['p']:.3f}{holm}"


for exp in s.experiment.unique():
    d = s[s.experiment == exp]
    print("=====", exp, "users", d.users.max(), "seeds", d.seeds.max())
    for metric in METRICS:
        for _, r in d[d.metric == metric].iterrows():
            print(f"  {metric:28s} {r['mode']:18s} {fmt(r)}")
