"""Generate the revision experiment configs (E1-E7) under configs/revision/.

usage: python scripts/make_revision_configs.py [--model-name PATH] [--root DIR]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

DATASETS = {
    "ml1m": {
        "dataset": "ml-1m",
        "movies_path": "{root}/agentic_popularity_lockin/data/raw/ml-1m/movies.dat",
    },
    "cds": {"dataset": "amazon_cds"},
    "kuairec": {
        "dataset": "kuairec",
        "utility_mode": "observed",
        "embedding_name": "paraphrase-multilingual-MiniLM-L12-v2",
    },
}

BASE = {
    "data_root": "{root}/niche_user_study/data",
    "backend": "vllm",
    "trace_prompts": False,
    "user_chunk": 50,
    "min_history": 30,
    "rounds": 8,
    "intervention_rounds": 2,
    "candidate_size": 12,
    "recommendation_size": 3,
    "utility_target_quantile": 0.55,
    "head_quantile": 0.8,
    "tail_quantile": 0.2,
    "balanced_low": 0.4,
    "balanced_high": 0.6,
    "feedback_model": "deterministic",
    "click_threshold": 0.5,
    "position_penalty": 0.015,
    "semantic_clusters": 16,
    "drift_mode": "stable",
    "attack_mode": True,
    "recency_window": 3,
    "attack_target_relevance": 0.6,
    "attack_cluster_size": 80,
    "attack_slate": 12,
    "attack_pool_target": 3,
    "history_limit": 8,
    "max_new_tokens": 128,
    "eam_decay": 0.8,
    "eam_propensity_floor": 0.2,
    "eam_reference_propensity": 0.5,
    "ersm_top_genres": 6,
    "ersm_commit_threshold": 2,
    "conditions": ["head", "tail"],
}

CORE = ["none", "raw", "ersm", "ersm_invariant"]
ABLATION = [
    "none",
    "raw",
    "raw_recent",
    "eam",
    "ersm_no_residual",
    "ersm_ips_only",
    "ersm_slate_only",
    "ersm",
    "ersm_user",
    "ersm_invariant",
]


def experiments() -> list[tuple[str, str, list[int], int, list[str], dict]]:
    """(name, dataset key, seeds, n_users, memory modes, overrides)."""
    out = []
    for ds in ("ml1m", "cds"):
        out.append((f"main_{ds}", ds, [42, 123, 2027], 200, ABLATION, {}))
    weak = {
        "f1": {"attack_injection": "replace", "attack_slate": 1},
        "f3": {"attack_injection": "replace", "attack_slate": 3},
        "f6": {"attack_injection": "replace", "attack_slate": 6},
        "f12": {"attack_injection": "replace", "attack_slate": 12},
        "pin1": {"attack_injection": "replace", "attack_slate": 1, "attack_pin": 1},
        "k1": {"intervention_rounds": 1},
    }
    for tag, overrides in weak.items():
        out.append((f"weak_{tag}_ml1m", "ml1m", [42, 123], 100, CORE, overrides))
    tail_modes = [
        "none", "raw", "ersm_ips_only", "ersm_slate_only", "ersm", "ersm_user", "ersm_invariant",
    ]
    out.append(
        ("tail_full_ml1m", "ml1m", [42, 123], 100, tail_modes,
         {"attack_target_popularity": "tail"})
    )
    out.append(
        ("tail_f3_ml1m", "ml1m", [42, 123], 100, tail_modes,
         {"attack_target_popularity": "tail", "attack_injection": "replace", "attack_slate": 3})
    )
    attributes = {
        "cluster": {"attribute_source": "cluster", "attribute_clusters": 20},
        "noise20": {"attribute_source": "noisy_genre", "genre_noise": 0.2},
        "noise40": {"attribute_source": "noisy_genre", "genre_noise": 0.4},
    }
    for tag, overrides in attributes.items():
        for ds in ("ml1m", "cds"):
            out.append((f"attr_{tag}_{ds}", ds, [42, 123], 100, CORE, overrides))
    for ds in ("ml1m", "cds"):
        out.append(
            (f"hard_{ds}", ds, [42, 123], 100, CORE,
             {"candidate_size": 20, "attack_slate": 20, "attack_pool_mode": "hard"})
        )
    out.append(
        ("fact_ml1m", "ml1m", [42, 123], 100,
         ["none", "raw", "factmem", "factmem_ersm", "ersm"], {})
    )
    for ds in ("ml1m", "cds"):
        out.append(
            (f"gate_{ds}", ds, [42, 123, 2027], 200,
             ["none", "raw", "raw_gate", "ersm"], {})
        )
    out.append(
        ("mem0_ml1m", "ml1m", [42, 123], 100,
         ["none", "raw", "mem0", "mem0_ersm", "ersm"],
         {
             "backend": "openai",
             "served_model_name": "qwen3-8b-awq",
             "openai_base_url": "http://127.0.0.1:8000/v1",
             "io_workers": 48,
             "user_chunk": 25,
             "retrieve_k": 5,
             "mem0_embedder": "sentence-transformers/all-MiniLM-L6-v2",
             "mem0_dir": "{root}/mem0_runs",
         })
    )
    return out


def model_experiments(tag: str) -> list[tuple[str, str, list[int], int, list[str], dict]]:
    """Main experiment at full scale for another agent model."""
    modes = ["none", "raw", "raw_recent", "ersm_slate_only", "ersm", "ersm_invariant"]
    return [
        (f"{tag}_main_{ds}", ds, [42, 123, 2027], 200, modes, {}) for ds in ("ml1m", "cds")
    ]


def kuairec_experiments() -> list[tuple[str, str, list[int], int, list[str], dict]]:
    """Fully observed feedback: clicks and relevance are KuaiRec watch ratios."""
    modes = [
        "none", "raw", "raw_recent", "eam", "ersm_ips_only", "ersm_slate_only", "ersm",
        "ersm_invariant",
    ]
    return [("kuairec_main", "kuairec", [42, 123, 2027], 200, modes, {})]


def api_experiments(
    tag: str, served_model: str, provider: str = "", quantization: str = ""
) -> list[tuple[str, str, list[int], int, list[str], dict]]:
    """Main experiment for a hosted model behind OpenRouter, thinking disabled.

    A pinned provider (and precision) keeps every call of a run on one serving stack.
    """
    extra_body: dict = {"reasoning": {"enabled": False}}
    if provider:
        extra_body["provider"] = {"order": [provider], "allow_fallbacks": False}
        if quantization:
            extra_body["provider"]["quantizations"] = [quantization]
    hosted = {
        "backend": "openai",
        "served_model_name": served_model,
        "openai_base_url": "https://openrouter.ai/api/v1",
        "openai_api_key_env": "OPENROUTER_API_KEY",
        "openai_extra_body": extra_body,
        "openai_max_retries": 6,
        "io_workers": 32,
    }
    modes = ["none", "raw", "ersm", "ersm_invariant"]
    return [
        (f"{tag}_main_{ds}", ds, [42, 123, 2027], 200, modes, hosted) for ds in ("ml1m", "cds")
    ]


def plan_experiments(
    tag: str, served_model: str, effort: str, root: str
) -> list[tuple[str, str, list[int], int, list[str], dict]]:
    """Main experiment for a GPT model billed to a ChatGPT plan (default sampling)."""
    plan = {
        "backend": "chatgpt_plan",
        "served_model_name": served_model,
        "plan_effort": effort or None,
        "plan_cache": f"{root}/agentic_popularity_lockin/results/plan_cache/{tag}.jsonl",
        "io_workers": 4,
    }
    modes = ["none", "raw", "ersm", "ersm_invariant"]
    return [
        (f"{tag}_main_{ds}", ds, [42, 123, 2027], 200, modes, plan) for ds in ("ml1m", "cds")
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="Qwen/Qwen3-8B-AWQ")
    parser.add_argument("--root", default="..", help="directory holding niche_user_study/ and this repo")
    parser.add_argument("--out", default="configs/revision")
    parser.add_argument("--set", default="revision", choices=["revision", "model", "kuairec", "api", "plan"])
    parser.add_argument("--tag", default="", help="model tag for --set model / api / plan")
    parser.add_argument("--served-model", default="", help="model id for --set api / plan")
    parser.add_argument("--provider", default="", help="pin one OpenRouter provider for --set api")
    parser.add_argument("--quantization", default="", help="required provider precision for --set api")
    parser.add_argument("--effort", default="", help="reasoning effort for --set plan ('' = omit)")
    args = parser.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    plan = {
        "revision": experiments,
        "model": lambda: model_experiments(args.tag),
        "kuairec": kuairec_experiments,
        "api": lambda: api_experiments(args.tag, args.served_model, args.provider, args.quantization),
        "plan": lambda: plan_experiments(args.tag, args.served_model, args.effort, args.root),
    }[args.set]()
    names = []
    for name, ds, seeds, n_users, modes, overrides in plan:
        for seed in seeds:
            config = {k: v for k, v in BASE.items()}
            config.update(DATASETS[ds])
            config.update(overrides)
            run = f"rev_{name}_seed{seed}"
            config.update(
                {
                    "output_dir": f"{{root}}/agentic_popularity_lockin/results/revision/{run}",
                    "model_name": args.model_name,
                    "seed": seed,
                    "n_users": n_users,
                    "memory_modes": modes,
                }
            )
            config = {
                k: v.format(root=args.root) if isinstance(v, str) else v
                for k, v in config.items()
            }
            with (out_dir / f"{run}.yaml").open("w", encoding="utf-8") as handle:
                yaml.safe_dump(config, handle, sort_keys=False)
            names.append(run)
    (out_dir / "ALL.txt").write_text("\n".join(names) + "\n", encoding="utf-8")
    print(f"wrote {len(names)} configs to {out_dir}")


if __name__ == "__main__":
    main()
