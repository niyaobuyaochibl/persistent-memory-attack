from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from stickyrec.data import DatasetBundle, build_semantic_partition
from stickyrec.environment import CounterfactualRecEnv
from stickyrec.memory import (
    DelayedCommitmentPolicy,
    EAMPolicy,
    ExposureNormalizedPolicy,
    LinUCBPolicy,
    MemoryResetPolicy,
    NoEarlyWritePolicy,
    PolicyContext,
    RawMemoryPolicy,
    RecencyWeightedPolicy,
    SlidingWindowPolicy,
    ThompsonSamplingPolicy,
)
from stickyrec.metrics import (
    bootstrap_mean_ci,
    condition_metrics,
    jaccard_distance,
    trajectory_distance,
)


def synthetic_bundle() -> DatasetBundle:
    rng = np.random.default_rng(7)
    embeddings = rng.normal(size=(120, 8)).astype(np.float32)
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
    train_rows = []
    histories = {}
    for user_id in range(3):
        history = np.arange(user_id * 10, user_id * 10 + 10)
        histories[user_id] = history
        for item_id in history:
            train_rows.append((user_id, int(item_id), 5.0))
    frame = pd.DataFrame(train_rows, columns=["user_idx", "item_idx", "rating"])
    counts = np.arange(1, 121, dtype=np.int64)
    popularity = np.linspace(0.0, 1.0, 120, dtype=np.float32)
    return DatasetBundle(
        name="synthetic",
        train=frame,
        val=frame.iloc[:0],
        test=frame.iloc[:0],
        embeddings=embeddings,
        popularity=popularity,
        item_counts=counts,
        histories=histories,
    )


def synthetic_config() -> dict:
    return {
        "seed": 3,
        "rounds": 4,
        "intervention_rounds": 1,
        "candidate_size": 12,
        "recommendation_size": 3,
        "utility_target_quantile": 0.3,
        "head_quantile": 0.75,
        "tail_quantile": 0.25,
        "balanced_low": 0.35,
        "balanced_high": 0.65,
        "feedback_model": "deterministic",
        "click_threshold": 0.0,
        "position_penalty": 0.0,
    }


class MetricTests(unittest.TestCase):
    def test_jaccard_distance(self) -> None:
        self.assertAlmostEqual(jaccard_distance([1, 2], [2, 3]), 2.0 / 3.0)

    def test_bootstrap_constant_values(self) -> None:
        self.assertEqual(bootstrap_mean_ci([0.25] * 5, seed=1), (0.25, 0.25))

    def test_trajectory_distance_ignores_intervention(self) -> None:
        left = [
            {"round_idx": 0, "recommended_ids": [1]},
            {"round_idx": 1, "recommended_ids": [2, 3]},
        ]
        right = [
            {"round_idx": 0, "recommended_ids": [9]},
            {"round_idx": 1, "recommended_ids": [2, 3]},
        ]
        self.assertEqual(trajectory_distance(left, right, start_round=1), 0.0)

    def test_adaptation_metrics_require_consecutive_recovery(self) -> None:
        records = []
        for round_idx, (drift_active, regret, recovered) in enumerate(
            [
                (False, 0.01, True),
                (True, 0.08, False),
                (True, 0.02, True),
                (True, 0.01, True),
            ]
        ):
            records.append(
                {
                    "round_idx": round_idx,
                    "recommended_ids": [round_idx],
                    "popularities": [0.5],
                    "reward": 0.5,
                    "clicked_item": round_idx,
                    "utility_regret": regret,
                    "recovered": recovered,
                    "drift_active": drift_active,
                }
            )
        metrics = condition_metrics(records, adaptation_patience=2)
        self.assertAlmostEqual(metrics["preference_recovery_rate"], 2.0 / 3.0)
        self.assertEqual(metrics["adaptation_delay"], 1.0)

    def test_gradual_transition_is_separate_from_recovery(self) -> None:
        records = []
        for round_idx, (progress, regret, recovered) in enumerate(
            [
                (0.0, 0.01, True),
                (0.25, 0.02, True),
                (0.75, 0.05, False),
                (1.0, 0.08, False),
                (1.0, 0.02, True),
                (1.0, 0.01, True),
            ]
        ):
            records.append(
                {
                    "round_idx": round_idx,
                    "recommended_ids": [round_idx],
                    "popularities": [0.5],
                    "reward": 0.5,
                    "clicked_item": round_idx,
                    "utility_regret": regret,
                    "recovered": recovered,
                    "drift_active": progress > 0.0,
                    "drift_progress": progress,
                }
            )
        metrics = condition_metrics(records, adaptation_patience=2)
        self.assertAlmostEqual(metrics["transition_mean_regret"], 0.035)
        self.assertAlmostEqual(metrics["post_drift_mean_regret"], 0.11 / 3.0)
        self.assertEqual(metrics["adaptation_delay"], 1.0)


class EnvironmentTests(unittest.TestCase):
    def test_post_intervention_candidates_are_paired(self) -> None:
        bundle = synthetic_bundle()
        head = CounterfactualRecEnv(bundle, synthetic_config())
        tail = CounterfactualRecEnv(bundle, synthetic_config())
        head.reset(0, "head")
        tail.reset(0, "tail")
        for round_idx in range(1, synthetic_config()["rounds"]):
            np.testing.assert_array_equal(
                head._candidate_plan[round_idx],
                tail._candidate_plan[round_idx],
            )

    def test_cached_candidate_plan_is_reused(self) -> None:
        bundle = synthetic_bundle()
        source = CounterfactualRecEnv(bundle, synthetic_config())
        source.reset(0, "head")
        cached = source.candidate_plan
        reused = CounterfactualRecEnv(bundle, synthetic_config())
        reused.reset(0, "head", candidate_plan=cached)
        for left, right in zip(cached, reused.candidate_plan):
            np.testing.assert_array_equal(left, right)

    def test_nonlinear_feedback_runs_across_rounds(self) -> None:
        bundle = synthetic_bundle()
        config = synthetic_config()
        config["feedback_model"] = "nonlinear"
        config["feedback_temperature"] = 0.1
        config["satiation_strength"] = 0.2
        env = CounterfactualRecEnv(bundle, config)
        observation = env.reset(0, "head")
        for _ in range(2):
            next_observation, result = env.step(observation.candidate_ids[:3])
            self.assertEqual(len(result.utilities), 3)
            observation = next_observation

    def test_gradual_and_abrupt_drift_profiles(self) -> None:
        bundle = synthetic_bundle()
        config = synthetic_config()
        config.update(
            {
                "rounds": 7,
                "candidate_size": 6,
                "recommendation_size": 2,
                "drift_start_round": 2,
                "drift_duration": 3,
                "drift_target_similarity": 0.0,
                "drift_magnitude": 1.0,
            }
        )
        base = bundle.user_profile(0)
        anchor = bundle.embeddings[80].astype(np.float64)
        anchor -= float(anchor @ base) * base
        anchor /= np.linalg.norm(anchor)
        anchors = anchor[None, :].astype(np.float32)

        abrupt = CounterfactualRecEnv(bundle, config, preference_anchors=anchors)
        abrupt.reset(0, "head", drift_mode="abrupt")
        np.testing.assert_allclose(abrupt.hidden_profile(1), base, atol=1e-6)
        np.testing.assert_allclose(abrupt.hidden_profile(2), anchor, atol=1e-6)

        gradual = CounterfactualRecEnv(bundle, config, preference_anchors=anchors)
        gradual.reset(0, "head", drift_mode="gradual")
        first = gradual.hidden_profile(2)
        self.assertGreater(float(first @ base), 0.0)
        self.assertGreater(float(first @ anchor), 0.0)
        np.testing.assert_allclose(gradual.hidden_profile(4), anchor, atol=1e-6)

    def test_rank_utility_spans_full_range(self) -> None:
        bundle = synthetic_bundle()
        config = synthetic_config()
        config["utility_mode"] = "rank"
        env = CounterfactualRecEnv(bundle, config)
        env.reset(0, "head")
        all_ids = np.arange(bundle.n_items)
        ranks = env.hidden_utility(all_ids)
        self.assertAlmostEqual(float(ranks.min()), 0.0)
        self.assertAlmostEqual(float(ranks.max()), 1.0)
        sims = bundle.embeddings @ env.hidden_profile()
        self.assertEqual(int(np.argmax(ranks)), int(np.argmax(sims)))

    def test_niche_trap_pool_is_sized_and_paired(self) -> None:
        config = synthetic_config()
        config.update(
            {
                "utility_mode": "rank",
                "pool_mode": "niche_trap",
                "n_gem": 4,
                "n_trap": 4,
                "gem_popularity_max": 0.4,
                "trap_popularity_min": 0.6,
                "gem_relevance_quantile": 0.6,
                "trap_relevance_max": 0.6,
            }
        )
        bundle = synthetic_bundle()
        head = CounterfactualRecEnv(bundle, config)
        tail = CounterfactualRecEnv(bundle, config)
        head.reset(0, "head")
        tail.reset(0, "tail")
        for round_idx in range(1, config["rounds"]):
            pool = head._candidate_plan[round_idx]
            self.assertEqual(len(pool), config["candidate_size"])
            self.assertEqual(len(np.unique(pool)), len(pool))
            np.testing.assert_array_equal(
                pool, tail._candidate_plan[round_idx]
            )

    def test_attack_mode_seeds_target_cluster_and_pairs_pool(self) -> None:
        config = synthetic_config()
        config.update(
            {
                "attack_mode": True,
                "attack_cluster_size": 20,
                "candidate_size": 6,
                "recommendation_size": 3,
                "attack_slate": 3,
                "attack_pool_target": 2,
                "rounds": 4,
                "intervention_rounds": 1,
            }
        )
        bundle = synthetic_bundle()
        head = CounterfactualRecEnv(bundle, config)
        tail = CounterfactualRecEnv(bundle, config)
        head.reset(0, "head")
        tail.reset(0, "tail")
        self.assertIsNotNone(head.attack_target)
        cluster = set(head.attack_cluster.tolist())
        head_int = set(head._candidate_plan[0].tolist())
        tail_int = set(tail._candidate_plan[0].tolist())
        self.assertGreaterEqual(len(head_int & cluster), 3)
        self.assertEqual(len(tail_int & cluster), 0)
        for round_idx in range(1, config["rounds"]):
            np.testing.assert_array_equal(
                head._candidate_plan[round_idx],
                tail._candidate_plan[round_idx],
            )
            pool = set(head._candidate_plan[round_idx].tolist())
            self.assertGreaterEqual(len(pool & cluster), 1)

    def test_genre_aligned_attack_cluster_shares_target_genre(self) -> None:
        config = synthetic_config()
        config.update(
            {
                "attack_mode": True,
                "attack_cluster_mode": "genre",
                "attack_cluster_size": 20,
                "candidate_size": 6,
                "recommendation_size": 3,
                "attack_slate": 3,
                "attack_pool_target": 2,
                "rounds": 4,
                "intervention_rounds": 1,
            }
        )
        bundle = synthetic_bundle()
        item_genres = (np.arange(bundle.n_items) % 4).astype(np.int64)
        env = CounterfactualRecEnv(bundle, config, item_genres=item_genres)
        env.reset(0, "head")
        self.assertIsNotNone(env.attack_target)
        target_genre = int(item_genres[int(env.attack_target)])
        cluster = env.attack_cluster
        self.assertEqual(len(cluster), 20)
        self.assertTrue(
            all(int(item_genres[int(i)]) == target_genre for i in cluster)
        )

    def test_genre_attack_requires_item_genres(self) -> None:
        config = synthetic_config()
        config.update({"attack_mode": True, "attack_cluster_mode": "genre"})
        env = CounterfactualRecEnv(synthetic_bundle(), config)
        with self.assertRaises(ValueError):
            env.reset(0, "head")

    def test_raw_memory_moves_toward_click(self) -> None:
        bundle = synthetic_bundle()
        assignments, centroids = build_semantic_partition(
            bundle.embeddings, n_clusters=4, seed=1
        )
        context = PolicyContext(
            base_profile=bundle.user_profile(0),
            history_items=bundle.histories[0],
            embeddings=bundle.embeddings,
            popularity=bundle.popularity,
            cluster_assignments=assignments,
            cluster_centroids=centroids,
        )
        policy = RawMemoryPolicy(update_rate=0.5)
        policy.reset(context)
        before = policy.profile.copy()
        clicked = 70
        from stickyrec.environment import Observation, StepResult

        observation = Observation(0, 0, np.array([clicked]), True, "head")
        result = StepResult(
            user_id=0,
            round_idx=0,
            recommended_ids=np.array([clicked]),
            clicked_item=clicked,
            reward=1.0,
            utilities=np.array([1.0]),
            popularities=np.array([0.9]),
            intervention_active=True,
        )
        policy.observe(observation, result)
        self.assertEqual(len(policy.memory_records), 1)
        self.assertEqual(policy.memory_records[0].signal_type, "implicit_positive")
        self.assertGreater(
            float(policy.profile @ bundle.embeddings[clicked]),
            float(before @ bundle.embeddings[clicked]),
        )


class PlaceboPolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.bundle = synthetic_bundle()
        assignments, centroids = build_semantic_partition(
            self.bundle.embeddings, n_clusters=4, seed=1
        )
        self.context = PolicyContext(
            base_profile=self.bundle.user_profile(0),
            history_items=self.bundle.histories[0],
            embeddings=self.bundle.embeddings,
            popularity=self.bundle.popularity,
            cluster_assignments=assignments,
            cluster_centroids=centroids,
        )

    def _feedback(self, round_idx: int, intervention: bool):
        from stickyrec.environment import Observation, StepResult

        clicked = 70
        observation = Observation(
            0, round_idx, np.array([clicked]), intervention, "head"
        )
        result = StepResult(
            user_id=0,
            round_idx=round_idx,
            recommended_ids=np.array([clicked]),
            clicked_item=clicked,
            reward=1.0,
            utilities=np.array([1.0]),
            popularities=np.array([0.9]),
            intervention_active=intervention,
        )
        return observation, result

    def test_no_early_write_ignores_treatment_feedback(self) -> None:
        policy = NoEarlyWritePolicy(update_rate=0.5)
        policy.reset(self.context)
        before = policy.profile.copy()
        observation, result = self._feedback(0, True)
        policy.observe(observation, result)
        np.testing.assert_allclose(policy.profile, before)
        self.assertEqual(policy.memory_records, [])

    def test_memory_reset_clears_treatment_state(self) -> None:
        policy = MemoryResetPolicy(update_rate=0.5)
        policy.reset(self.context)
        observation, result = self._feedback(0, True)
        policy.observe(observation, result)
        self.assertFalse(np.allclose(policy.profile, self.context.base_profile))
        post_observation, _ = self._feedback(1, False)
        policy.recommend(post_observation, top_k=1)
        np.testing.assert_allclose(policy.profile, self.context.base_profile)
        self.assertEqual(policy.memory_records, [])

    def test_recent_memory_baselines_update(self) -> None:
        observation, result = self._feedback(0, True)
        for policy in (
            SlidingWindowPolicy(window_size=2, base_strength=2.0),
            RecencyWeightedPolicy(decay=0.7, base_strength=2.0),
        ):
            policy.reset(self.context)
            before = policy.profile.copy()
            policy.observe(observation, result)
            self.assertFalse(np.allclose(policy.profile, before))

    def test_exposure_normalized_policy_updates_cluster_rates(self) -> None:
        policy = ExposureNormalizedPolicy(base_strength=2.0, decay=0.8)
        policy.reset(self.context)
        observation, result = self._feedback(0, True)
        policy.observe(observation, result)
        self.assertEqual(float(policy.exposures.sum()), 1.0)
        self.assertEqual(float(policy.positives.sum()), 1.0)
        self.assertEqual(len(policy.memory_records), 1)

    def test_eam_downweights_popular_implicit_feedback(self) -> None:
        policy = EAMPolicy(
            base_strength=2.0,
            alpha=1.0,
            beta=1.0,
            propensity_floor=0.2,
            reference_propensity=0.5,
        )
        policy.reset(self.context)
        self.assertGreater(
            policy._implicit_confidence(0),
            policy._implicit_confidence(119),
        )
        self.assertLessEqual(policy._implicit_confidence(0), 1.0)

    def test_delayed_eam_quarantines_early_implicit_feedback(self) -> None:
        policy = DelayedCommitmentPolicy(
            base_strength=1.5,
            alpha=1.0,
            beta=1.0,
            quarantine_rounds=2,
        )
        policy.reset(self.context)
        before = policy.profile.copy()
        observation, result = self._feedback(0, True)
        policy.observe(observation, result)
        np.testing.assert_allclose(policy.profile, before)
        self.assertEqual(policy.committed_observations, 0)
        observation, result = self._feedback(2, False)
        policy.observe(observation, result)
        self.assertEqual(policy.committed_observations, 1)
        self.assertFalse(np.allclose(policy.profile, before))

    def test_eam_uses_clicked_item_residual_to_the_slate(self) -> None:
        from stickyrec.environment import Observation, StepResult

        policy = EAMPolicy(
            base_strength=2.0,
            alpha=1.0,
            beta=1.0,
            cluster_strength=0.0,
        )
        policy.reset(self.context)
        recommended = np.array([70, 71, 72], dtype=np.int64)
        observation = Observation(0, 0, recommended, True, "head")
        result = StepResult(
            user_id=0,
            round_idx=0,
            recommended_ids=recommended,
            clicked_item=70,
            reward=1.0,
            utilities=np.ones(3),
            popularities=self.bundle.popularity[recommended],
            intervention_active=True,
        )
        confidence = policy._implicit_confidence(70)
        expected = confidence * (
            self.bundle.embeddings[70]
            - self.bundle.embeddings[recommended].mean(axis=0)
        )
        policy.observe(observation, result)
        np.testing.assert_allclose(policy.residual_vector, expected, atol=1e-7)
        self.assertGreater(policy.residual_weight, 0.0)

    def test_eam_exploration_returns_unique_valid_items(self) -> None:
        policy = EAMPolicy(
            base_strength=2.0,
            alpha=1.0,
            beta=1.0,
            exploration_bonus=0.2,
            exploration_slots=1,
        )
        policy.reset(self.context)
        candidates = np.arange(60, 75, dtype=np.int64)
        from stickyrec.environment import Observation

        observation = Observation(0, 1, candidates, False, "head")
        action = policy.recommend(observation, top_k=3)
        self.assertEqual(len(action), 3)
        self.assertEqual(len(np.unique(action)), 3)
        self.assertTrue(set(action).issubset(set(candidates)))

    def test_linear_bandits_update_and_recommend_valid_items(self) -> None:
        observation, result = self._feedback(0, True)
        for policy in (
            LinUCBPolicy(
                alpha=0.5,
                regularization=1.0,
                base_weight=0.5,
                seed=3,
            ),
            ThompsonSamplingPolicy(
                sample_scale=0.25,
                regularization=1.0,
                base_weight=0.5,
                seed=3,
            ),
        ):
            policy.reset(self.context)
            policy.observe(observation, result)
            candidates = np.arange(60, 75, dtype=np.int64)
            query = type(observation)(
                user_id=0,
                round_idx=1,
                candidate_ids=candidates,
                intervention_active=False,
                exposure_condition="head",
            )
            action = policy.recommend(query, top_k=3)
            self.assertEqual(len(action), 3)
            self.assertEqual(len(np.unique(action)), 3)
            self.assertTrue(set(action).issubset(set(candidates)))
            self.assertGreater(float(np.linalg.norm(policy.b)), 0.0)


if __name__ == "__main__":
    unittest.main()

