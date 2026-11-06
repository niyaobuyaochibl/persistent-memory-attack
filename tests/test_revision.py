from __future__ import annotations

from collections import deque
import json
import unittest

import numpy as np
import pandas as pd

from stickyrec.environment import CounterfactualRecEnv, Observation, StepResult
from stickyrec.llm_agent import StructuredLLMPolicy
from stickyrec.metadata import ItemMetadata
from stickyrec.revision_analysis import clustered_bootstrap, revision_table
from tests.test_core import synthetic_bundle, synthetic_config
from tests.test_llm_agent import ScriptedBackend, policy_context


def genre_metadata(genres: dict[int, tuple[str, ...]]) -> dict[int, ItemMetadata]:
    return {
        item_id: ItemMetadata(item_id, str(item_id), f"Title {item_id}", attrs)
        for item_id, attrs in genres.items()
    }


def step(slate: list[int], clicked: int, round_idx: int = 0) -> StepResult:
    return StepResult(
        user_id=0,
        round_idx=round_idx,
        recommended_ids=np.array(slate, dtype=np.int64),
        clicked_item=clicked,
        reward=1.0,
        utilities=np.ones(len(slate)),
        popularities=np.zeros(len(slate)),
        intervention_active=round_idx == 0,
    )


class ErsmComponentTests(unittest.TestCase):
    metadata = genre_metadata(
        {0: ("Drama",), 1: ("Comedy",), 2: ("Comedy",), 3: ("Action",)}
    )

    def policy(self, mode: str, **kwargs) -> StructuredLLMPolicy:
        policy = StructuredLLMPolicy(
            ScriptedBackend([]), self.metadata, memory_mode=mode,
            history_limit=0, **kwargs,
        )
        policy.reset(policy_context())
        return policy

    def test_ips_only_weights_clicked_genre_without_residual(self) -> None:
        policy = self.policy("ersm_ips_only")
        record = policy._update_genre_scores(1, step([1, 2, 3], 1))
        self.assertFalse(record["residualized"])
        self.assertTrue(record["catalogue_ips"])
        # Item 1 has popularity 0.8: propensity 0.84, weight 0.5 / 0.84.
        self.assertAlmostEqual(policy.genre_scores["Comedy"], 0.5 / 0.84, places=5)
        self.assertNotIn("Action", policy.genre_scores)

    def test_slate_only_zeroes_fully_saturated_slate(self) -> None:
        policy = self.policy("ersm_slate_only")
        record = policy._update_genre_scores(1, step([1, 2], 1))
        self.assertTrue(record["residualized"])
        self.assertFalse(record["catalogue_ips"])
        self.assertEqual(record["evidence"]["Comedy"], 0.0)
        self.assertEqual(policy.genre_scores.get("Comedy", 0.0), 0.0)

    def test_slate_only_credits_choice_against_mixed_slate(self) -> None:
        policy = self.policy("ersm_slate_only")
        policy._update_genre_scores(1, step([1, 3], 1))
        self.assertAlmostEqual(policy.genre_scores["Comedy"], 0.5)
        self.assertAlmostEqual(policy.genre_scores["Action"], -0.5)

    def test_user_propensity_discounts_repeated_exposure(self) -> None:
        policy = self.policy("ersm_user", ersm_user_tolerance=1.0, ersm_user_prior=1.0)
        first = policy._update_genre_scores(1, step([1, 3], 1))
        for round_idx in range(1, 4):
            last = policy._update_genre_scores(1, step([1, 2, 3], 1, round_idx))
        self.assertFalse(first["catalogue_ips"])
        self.assertTrue(first["user_propensity"])
        self.assertLess(last["confidence"], first["confidence"])
        self.assertLess(last["confidence"], 1.0)

    def test_user_propensity_ignores_catalogue_popularity(self) -> None:
        # Item 3 is long-tail (popularity 0.1): catalogue IPS gives full weight,
        # but saturating the user's slates with Action still discounts it.
        ips = self.policy("ersm")
        user = self.policy("ersm_user", ersm_user_tolerance=1.0, ersm_user_prior=1.0)
        for round_idx in range(3):
            ips_record = ips._update_genre_scores(3, step([3, 1], 3, round_idx))
            user_record = user._update_genre_scores(3, step([3, 1], 3, round_idx))
        self.assertEqual(ips_record["confidence"], 1.0)
        self.assertLess(user_record["confidence"], 1.0)


class FactMemoryTests(unittest.TestCase):
    metadata = genre_metadata(
        {0: ("Drama",), 1: ("Comedy",), 2: ("Comedy",), 3: ("Action",)}
    )

    def test_factmem_stores_llm_facts(self) -> None:
        backend = ScriptedBackend(['{"facts": ["Likes comedies", "Enjoys light films"]}'])
        policy = StructuredLLMPolicy(backend, self.metadata, memory_mode="factmem")
        policy.reset(policy_context())
        observation = Observation(0, 0, np.array([1, 2, 3]), True, "head")
        policy.observe(observation, step([1, 3], 1))
        self.assertEqual(policy.facts, ["Likes comedies", "Enjoys light films"])
        prompt = policy.recommendation_prompt(observation, 2)
        self.assertIn("Stored preference facts", prompt)
        self.assertIn("Likes comedies", prompt)

    def test_factmem_parse_failure_falls_back(self) -> None:
        backend = ScriptedBackend(["oops"])
        policy = StructuredLLMPolicy(backend, self.metadata, memory_mode="factmem")
        policy.reset(policy_context())
        observation = Observation(0, 0, np.array([1, 2, 3]), True, "head")
        policy.observe(observation, step([1, 3], 1))
        self.assertEqual(policy.parse_failures, 1)
        self.assertEqual(len(policy.facts), 1)
        self.assertIn("Title 1", policy.facts[0])

    def test_ersm_gate_blocks_saturated_click_without_llm_call(self) -> None:
        backend = ScriptedBackend([])
        policy = StructuredLLMPolicy(
            backend, self.metadata, memory_mode="factmem_ersm", history_limit=0
        )
        policy.reset(policy_context())
        observation = Observation(0, 0, np.array([1, 2]), True, "head")
        result = step([1, 2], 1)
        self.assertIsNone(policy.memory_prompt(observation, result))
        policy.observe(observation, result)
        self.assertEqual(backend.prompts, [])
        self.assertFalse(policy.trace[-1]["memory_record"]["admitted"])

    def test_ersm_gate_admits_discriminative_click(self) -> None:
        backend = ScriptedBackend(['{"facts": ["Likes comedies"]}'])
        policy = StructuredLLMPolicy(
            backend, self.metadata, memory_mode="factmem_ersm", history_limit=0
        )
        policy.reset(policy_context())
        observation = Observation(0, 0, np.array([2, 3]), True, "head")
        result = step([2, 3], 2)
        self.assertIsNotNone(policy.memory_prompt(observation, result))
        policy.observe(observation, result)
        self.assertEqual(policy.facts, ["Likes comedies"])


class Mem0ModeTests(unittest.TestCase):
    metadata = genre_metadata(
        {0: ("Drama",), 1: ("Comedy",), 2: ("Comedy",), 3: ("Action",)}
    )

    class FakeStore:
        def __init__(self):
            self.items: list[str] = []
            self.queries: list[str] = []

        def add(self, text: str) -> None:
            self.items.append(text)

        def search(self, query: str, k: int) -> list[str]:
            self.queries.append(query)
            return self.items[-k:]

        def all(self) -> list[str]:
            return list(self.items)

    def test_mem0_requires_store(self) -> None:
        with self.assertRaises(ValueError):
            StructuredLLMPolicy(ScriptedBackend([]), self.metadata, memory_mode="mem0")

    def test_mem0_writes_clicks_and_injects_retrieved_memories(self) -> None:
        store = self.FakeStore()
        policy = StructuredLLMPolicy(
            ScriptedBackend([]), self.metadata, memory_mode="mem0",
            memory_store=store, retrieve_k=2,
        )
        policy.reset(policy_context())
        observation = Observation(0, 0, np.array([1, 2, 3]), True, "head")
        policy.observe(observation, step([1, 3], 1))
        self.assertEqual(len(store.items), 1)
        self.assertIn("Title 1", store.items[0])
        prompt = policy.recommendation_prompt(observation, 2)
        self.assertIn("Retrieved memories", prompt)
        self.assertIn("Title 3", store.queries[0])

    def test_mem0_ersm_gate_skips_saturated_click(self) -> None:
        store = self.FakeStore()
        policy = StructuredLLMPolicy(
            ScriptedBackend([]), self.metadata, memory_mode="mem0_ersm",
            memory_store=store, history_limit=0,
        )
        policy.reset(policy_context())
        observation = Observation(0, 0, np.array([1, 2]), True, "head")
        policy.observe(observation, step([1, 2], 1))
        self.assertEqual(store.items, [])
        self.assertFalse(policy.trace[-1]["memory_record"]["admitted"])
        policy.observe(observation, step([2, 3], 2, 1))
        self.assertEqual(len(store.items), 1)

    def test_raw_gate_keeps_identities_but_skips_saturated_click(self) -> None:
        policy = StructuredLLMPolicy(
            ScriptedBackend([]), self.metadata, memory_mode="raw_gate", history_limit=0,
        )
        policy.reset(policy_context())
        observation = Observation(0, 0, np.array([1, 2, 3]), True, "head")
        policy.observe(observation, step([1, 2], 1))
        self.assertEqual(policy.raw_events, [])
        self.assertFalse(policy.trace[-1]["memory_record"]["admitted"])
        policy.observe(observation, step([2, 3], 2, 1))
        self.assertEqual(len(policy.raw_events), 1)
        self.assertIn("Title 2", policy.memory_text)
        self.assertIsNone(policy.memory_prompt(observation, step([2, 3], 2, 1)))

    def test_audit_counts_promoted_titles_and_genre(self) -> None:
        metadata = {
            0: ItemMetadata(0, "0", "Matrix, The (1999)", ("Sci-Fi",)),
            1: ItemMetadata(1, "1", "Alien (1979)", ("Sci-Fi", "Horror")),
            2: ItemMetadata(2, "2", "Clueless (1995)", ("Comedy",)),
            3: ItemMetadata(3, "3", "Heat (1995)", ("Action",)),
        }
        store = self.FakeStore()
        policy = StructuredLLMPolicy(
            ScriptedBackend([]), metadata, memory_mode="mem0", memory_store=store,
        )
        policy.reset(policy_context())
        policy.set_audit([0, 1])
        store.items = ["Loved The Matrix", "Enjoys sci-fi films", "Likes comedies"]
        policy.last_retrieved = ["Loved The Matrix", "Likes comedies"]
        audit = policy.audit()
        self.assertEqual(audit["mem_items"], 3)
        self.assertAlmostEqual(audit["mem_title_share"], 1 / 3)
        self.assertAlmostEqual(audit["mem_genre_share"], 1 / 3)
        self.assertAlmostEqual(audit["ret_title_share"], 0.5)

    def test_normalize_title(self) -> None:
        from stickyrec.llm_agent import normalize_title

        self.assertEqual(normalize_title("Matrix, The (1999)"), "the matrix")
        self.assertEqual(normalize_title("Heat (1995)"), "heat")

    def test_audit_empty_for_structured_modes(self) -> None:
        policy = StructuredLLMPolicy(
            ScriptedBackend([]), self.metadata, memory_mode="ersm"
        )
        policy.reset(policy_context())
        policy.set_audit([1, 2])
        self.assertEqual(policy.audit(), {})

    def test_normalise_extraction_repairs_shape_only(self) -> None:
        try:
            import mem0  # noqa: F401
        except ImportError:
            self.skipTest("mem0 not installed")
        from stickyrec.mem0_store import normalise_extraction

        good = '{"memory": [{"id": "0", "text": "Likes jazz"}]}'
        self.assertEqual(normalise_extraction(good), (good, False))
        fixed, repaired = normalise_extraction('{"memory": ["Likes jazz", {"fact": "Dislikes pop"}]}')
        self.assertTrue(repaired)
        self.assertEqual(
            json.loads(fixed)["memory"],
            [{"id": "0", "text": "Likes jazz"}, {"id": "1", "text": "Dislikes pop"}],
        )
        self.assertEqual(normalise_extraction("not json"), ("not json", False))


class AggregateTests(unittest.TestCase):
    def test_holm_is_monotone_and_capped(self) -> None:
        from scripts.aggregate_revision import holm

        adjusted = holm(pd.Series([0.01, 0.04, 0.03, 0.5]))
        self.assertEqual(adjusted.round(4).tolist(), [0.04, 0.09, 0.09, 0.5])


class TwoPhaseApiTests(unittest.TestCase):
    def test_apply_recommendation_matches_recommend(self) -> None:
        metadata = genre_metadata({i: ("Genre",) for i in range(4)})
        observation = Observation(0, 0, np.array([1, 2, 3]), True, "head")
        one = StructuredLLMPolicy(
            ScriptedBackend(['{"selected_ids": [3, 1]}']), metadata, memory_mode="raw"
        )
        one.reset(policy_context())
        two = StructuredLLMPolicy(ScriptedBackend([]), metadata, memory_mode="raw")
        two.reset(policy_context())
        prompt = two.recommendation_prompt(observation, 2)
        np.testing.assert_array_equal(
            one.recommend(observation, 2),
            two.apply_recommendation(observation, 2, prompt, '{"selected_ids": [3, 1]}'),
        )
        self.assertEqual(one.trace[0]["prompt"], two.trace[0]["prompt"])


def attack_config(**overrides) -> dict:
    config = synthetic_config()
    config.update(
        {
            "attack_mode": True,
            "attack_cluster_size": 20,
            "candidate_size": 6,
            "recommendation_size": 3,
            "attack_slate": 2,
            "attack_pool_target": 2,
            "rounds": 4,
            "intervention_rounds": 2,
        }
    )
    config.update(overrides)
    return config


class AttackEnvironmentTests(unittest.TestCase):
    def test_replace_injection_differs_only_in_injected_slots(self) -> None:
        config = attack_config(attack_injection="replace")
        bundle = synthetic_bundle()
        head = CounterfactualRecEnv(bundle, config)
        tail = CounterfactualRecEnv(bundle, config)
        head.reset(0, "head")
        tail.reset(0, "tail")
        cluster = set(head.attack_cluster.tolist())
        head_pool = head._candidate_plan[0]
        tail_pool = tail._candidate_plan[0]
        self.assertEqual(len(set(head_pool.tolist()) & cluster), 2)
        self.assertEqual(len(set(tail_pool.tolist()) & cluster), 0)
        self.assertEqual(int(np.sum(head_pool != tail_pool)), 2)

    def test_sponsored_pin_forces_cluster_item_into_attack_slate(self) -> None:
        config = attack_config(attack_injection="replace", attack_slate=1, attack_pin=1)
        bundle = synthetic_bundle()
        env = CounterfactualRecEnv(bundle, config)
        observation = env.reset(0, "head")
        cluster = set(env.attack_cluster.tolist())
        organic = [int(i) for i in observation.candidate_ids if int(i) not in cluster][:3]
        _, result = env.step(np.array(organic))
        shown = result.recommended_ids.tolist()
        self.assertEqual(sum(i in cluster for i in shown), 1)
        self.assertEqual(shown[:2], organic[:2])

    def test_sponsored_pin_inactive_on_clean_arm_and_after_attack(self) -> None:
        config = attack_config(attack_injection="replace", attack_slate=1, attack_pin=1)
        bundle = synthetic_bundle()
        clean = CounterfactualRecEnv(bundle, config)
        observation = clean.reset(0, "tail")
        action = observation.candidate_ids[:3]
        _, result = clean.step(action)
        np.testing.assert_array_equal(result.recommended_ids, action)

    def test_tail_target_restricts_cluster_to_long_tail(self) -> None:
        config = attack_config(attack_target_popularity="tail", tail_quantile=0.4)
        bundle = synthetic_bundle()
        env = CounterfactualRecEnv(bundle, config)
        env.reset(0, "head")
        self.assertTrue(
            np.all(bundle.popularity[env.attack_cluster] <= 0.4)
        )

    def test_hard_pool_draws_high_relevance_filler(self) -> None:
        bundle = synthetic_bundle()
        hard = CounterfactualRecEnv(
            bundle, attack_config(attack_pool_mode="hard", hard_pool_quantile=0.8)
        )
        easy = CounterfactualRecEnv(bundle, attack_config())
        hard.reset(0, "head")
        easy.reset(0, "head")
        cluster = set(hard.attack_cluster.tolist())

        def filler_relevance(env: CounterfactualRecEnv) -> float:
            items = [
                int(i)
                for plan in env._candidate_plan[2:]
                for i in plan
                if int(i) not in cluster
            ]
            return float(env.hidden_utility(np.array(items)).mean())

        self.assertGreater(filler_relevance(hard), filler_relevance(easy))

    def test_rank_relevance_is_catalogue_percentile(self) -> None:
        bundle = synthetic_bundle()
        env = CounterfactualRecEnv(bundle, attack_config())
        env.reset(0, "head")
        sims = bundle.embeddings @ env.hidden_profile()
        best, worst = int(np.argmax(sims)), int(np.argmin(sims))
        ranks = env.rank_relevance(np.array([best, worst]))
        self.assertAlmostEqual(float(ranks[0]), 1.0)
        self.assertAlmostEqual(float(ranks[1]), 1.0 / bundle.n_items)


class RevisionAnalysisTests(unittest.TestCase):
    def test_clustered_bootstrap_resamples_users_not_rows(self) -> None:
        values = pd.Series([1.0, 1.0, 1.0, 0.0])
        clusters = pd.Series([0, 0, 0, 1])
        mean, low, high, _ = clustered_bootstrap(values, clusters, seed=1, n_bootstrap=2000)
        self.assertAlmostEqual(mean, 0.75)
        self.assertAlmostEqual(low, 0.0)
        self.assertAlmostEqual(high, 1.0)

    def test_revision_table_reports_defense_reduction(self) -> None:
        rows = []
        for seed in (1, 2):
            for user in range(6):
                for mode, lift in (("none", 0.0), ("raw", 0.4), ("ersm", 0.1)):
                    rows.append(
                        {
                            "seed": seed, "user_id": user, "memory_mode": mode,
                            "lift": lift + 0.01 * user, "relevance": 0.6,
                            "relevance_clean": 0.6, "utility": 0.7,
                        }
                    )
        table = revision_table(pd.DataFrame(rows), "t")
        reduction = table[
            (table["metric"] == "lift_reduction_vs_raw") & (table["mode"] == "ersm")
        ].iloc[0]
        self.assertAlmostEqual(reduction["mean"], 0.3)
        self.assertEqual(reduction["users"], 6)
        self.assertEqual(reduction["pairs"], 12)


if __name__ == "__main__":
    unittest.main()
