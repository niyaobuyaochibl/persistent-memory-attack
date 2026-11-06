from __future__ import annotations

from collections import deque
from pathlib import Path
import pickle
import tempfile
import unittest

import numpy as np
import pandas as pd

from stickyrec.environment import Observation, StepResult
from stickyrec.llm_agent import StructuredLLMPolicy, parse_selected_ids
from stickyrec.memory import PolicyContext
from stickyrec.metadata import (
    ItemMetadata,
    load_metadata,
    load_ml1m_metadata,
    load_pickled_metadata,
)


class ScriptedBackend:
    def __init__(self, responses: list[str]):
        self.responses = deque(responses)
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.responses.popleft()


def policy_context() -> PolicyContext:
    embeddings = np.asarray(
        [[1.0, 0.0], [0.8, 0.2], [0.0, 1.0], [-1.0, 0.0]],
        dtype=np.float32,
    )
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
    return PolicyContext(
        base_profile=np.array([1.0, 0.0], dtype=np.float32),
        history_items=np.array([0], dtype=np.int64),
        embeddings=embeddings,
        popularity=np.array([0.9, 0.8, 0.2, 0.1], dtype=np.float32),
        cluster_assignments=np.array([0, 0, 1, 1], dtype=np.int32),
        cluster_centroids=np.array(
            [[1.0, 0.0], [0.0, 1.0]], dtype=np.float32
        ),
    )


def item_metadata() -> dict[int, ItemMetadata]:
    return {
        item_id: ItemMetadata(item_id, str(item_id), f"Title {item_id}", ("Genre",))
        for item_id in range(4)
    }


class MetadataTests(unittest.TestCase):
    def test_ml1m_external_ids_map_to_internal_indices(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset_root = root / "datasets" / "ml-1m"
            dataset_root.mkdir(parents=True)
            frame = pd.DataFrame(
                {
                    "item_idx": [0, 1],
                    "item_id": [10, 20],
                    "user_idx": [0, 0],
                    "rating": [5, 4],
                }
            )
            for split in ("train", "val", "test"):
                with (dataset_root / f"{split}.pkl").open("wb") as handle:
                    pickle.dump(frame, handle)
            movies = root / "movies.dat"
            movies.write_text(
                "10::First Movie (2000)::Drama|Comedy\n"
                "20::Second Movie (2001)::Action\n",
                encoding="latin-1",
            )
            metadata = load_ml1m_metadata(root, movies)
            self.assertEqual(metadata[0].title, "First Movie (2000)")
            self.assertEqual(metadata[0].attributes, ("Drama", "Comedy"))
            self.assertEqual(metadata[1].external_id, "20")

    def test_pickled_metadata_loads_title_and_genres(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset_root = root / "datasets" / "amazon_cds"
            dataset_root.mkdir(parents=True)
            frame = pd.DataFrame(
                {
                    "item_idx": [0, 1],
                    "item_id": ["B001", "B002"],
                    "user_idx": [0, 0],
                    "rating": [5, 4],
                }
            )
            for split in ("train", "val", "test"):
                with (dataset_root / f"{split}.pkl").open("wb") as handle:
                    pickle.dump(frame, handle)
            payload = {
                0: ("Kind of Blue", ("Jazz", "Bebop")),
                1: ("Thriller", ("Pop",)),
            }
            with (dataset_root / "item_metadata.pkl").open("wb") as handle:
                pickle.dump(payload, handle)

            metadata = load_pickled_metadata(root, "amazon_cds")
            self.assertEqual(metadata[0].title, "Kind of Blue")
            self.assertEqual(metadata[0].attributes, ("Jazz", "Bebop"))
            self.assertEqual(metadata[1].external_id, "B002")

            dispatched = load_metadata(root, "amazon_cds")
            self.assertEqual(dispatched[0].compact_text(), "Kind of Blue [Jazz, Bebop]")


class LLMPolicyTests(unittest.TestCase):
    def test_strict_selection_parser(self) -> None:
        candidates = np.array([1, 2, 3])
        parsed = parse_selected_ids('{"selected_ids": [3, 1]}', candidates, 2)
        np.testing.assert_array_equal(parsed, [3, 1])
        parsed_strings = parse_selected_ids(
            '{"selected_ids": ["3", "1"]}', candidates, 2
        )
        np.testing.assert_array_equal(parsed_strings, [3, 1])
        with self.assertRaises(ValueError):
            parse_selected_ids('{"selected_ids": [1, 1]}', candidates, 2)
        with self.assertRaises(ValueError):
            parse_selected_ids('{"selected_ids": [1, 9]}', candidates, 2)

    def test_policy_records_raw_memory_and_trace(self) -> None:
        backend = ScriptedBackend(['{"selected_ids": [2, 1]}'])
        policy = StructuredLLMPolicy(backend, item_metadata(), memory_mode="raw")
        policy.reset(policy_context())
        observation = Observation(0, 0, np.array([1, 2, 3]), True, "head")
        action = policy.recommend(observation, top_k=2)
        np.testing.assert_array_equal(action, [2, 1])
        result = StepResult(
            user_id=0,
            round_idx=0,
            recommended_ids=action,
            clicked_item=2,
            reward=1.0,
            utilities=np.ones(2),
            popularities=np.array([0.2, 0.8]),
            intervention_active=True,
        )
        policy.observe(observation, result)
        self.assertIn("Clicked Title 2", policy.memory_text)
        self.assertEqual([entry["event"] for entry in policy.trace], ["recommend", "memory_write"])

    def test_raw_recent_drops_early_exposure(self) -> None:
        backend = ScriptedBackend([])
        policy = StructuredLLMPolicy(
            backend,
            item_metadata(),
            memory_mode="raw_recent",
            recency_window=1,
        )
        policy.reset(policy_context())
        for round_idx, clicked in enumerate((1, 2, 3)):
            observation = Observation(
                0, round_idx, np.array([1, 2, 3]), round_idx == 0, "head"
            )
            result = StepResult(
                user_id=0,
                round_idx=round_idx,
                recommended_ids=np.array([clicked]),
                clicked_item=clicked,
                reward=1.0,
                utilities=np.ones(1),
                popularities=np.array([0.5]),
                intervention_active=round_idx == 0,
            )
            policy.observe(observation, result)
        # Only the most recent click survives; the early (attacked) round is gone.
        self.assertIn("Clicked Title 3", policy.memory_text)
        self.assertNotIn("Clicked Title 1", policy.memory_text)
        self.assertEqual(len(policy.raw_events), 3)

    def test_invalid_response_uses_deterministic_fallback(self) -> None:
        backend = ScriptedBackend(["not json"] )
        policy = StructuredLLMPolicy(backend, item_metadata(), memory_mode="none")
        policy.reset(policy_context())
        observation = Observation(0, 0, np.array([1, 2, 3]), True, "head")
        action = policy.recommend(observation, top_k=2)
        np.testing.assert_array_equal(action, [1, 2])
        self.assertEqual(policy.parse_failures, 1)
        self.assertIsNotNone(policy.trace[0]["parse_error"])

    def test_overlong_summary_uses_bounded_fallback(self) -> None:
        words = " ".join(f"word{i}" for i in range(26))
        backend = ScriptedBackend([f'{{"memory": "{words}"}}'])
        policy = StructuredLLMPolicy(
            backend, item_metadata(), memory_mode="summary"
        )
        policy.reset(policy_context())
        observation = Observation(0, 0, np.array([1, 2, 3]), True, "head")
        result = StepResult(
            user_id=0,
            round_idx=0,
            recommended_ids=np.array([1, 2]),
            clicked_item=2,
            reward=1.0,
            utilities=np.ones(2),
            popularities=np.array([0.8, 0.2]),
            intervention_active=True,
        )
        policy.observe(observation, result)
        self.assertEqual(policy.memory_text, "Clicked Title 2 [Genre].")
        self.assertEqual(policy.parse_failures, 1)
        self.assertEqual(policy.trace[0]["event"], "memory_summary")
        self.assertIn("at most 25 words", policy.trace[0]["parse_error"])

    def test_history_anchors_are_selected_by_profile_not_item_id(self) -> None:
        context = policy_context()
        context = PolicyContext(
            base_profile=context.base_profile,
            history_items=np.array([1, 2, 3]),
            embeddings=context.embeddings,
            popularity=context.popularity,
            cluster_assignments=context.cluster_assignments,
            cluster_centroids=context.cluster_centroids,
        )
        policy = StructuredLLMPolicy(
            ScriptedBackend([]),
            item_metadata(),
            memory_mode="none",
            history_limit=1,
        )
        policy.reset(context)
        self.assertIn("Title 1", policy.memory_text)
        self.assertNotIn("Title 2", policy.memory_text)

    def test_eam_records_exposure_confidence_and_decay(self) -> None:
        policy = StructuredLLMPolicy(
            ScriptedBackend([]),
            item_metadata(),
            memory_mode="eam",
            eam_decay=0.5,
        )
        policy.reset(policy_context())
        observation = Observation(0, 0, np.array([1, 2, 3]), True, "head")
        result = StepResult(
            user_id=0,
            round_idx=0,
            recommended_ids=np.array([1, 2]),
            clicked_item=1,
            reward=1.0,
            utilities=np.ones(2),
            popularities=np.array([0.8, 0.2]),
            intervention_active=True,
        )
        policy.observe(observation, result)
        record = policy.eam_records[0]
        self.assertEqual(record["source"], "implicit_click")
        self.assertFalse(record["explicit"])
        self.assertAlmostEqual(record["exposure_prevalence"], 0.8)
        self.assertAlmostEqual(record["confidence"], 0.5 / 0.84)
        rendered = policy._render_eam_memory(current_round=1)
        self.assertIn("weight=0.298", rendered)
        self.assertIn("shown alternatives are context, not dislikes", rendered)

    def test_ersm_renders_genres_not_titles_and_downweights_popular(self) -> None:
        metadata = {
            0: ItemMetadata(0, "0", "Title 0", ("Drama",)),
            1: ItemMetadata(1, "1", "Title 1", ("Comedy",)),
            2: ItemMetadata(2, "2", "Title 2", ("Comedy",)),
            3: ItemMetadata(3, "3", "Title 3", ("Action",)),
        }

        def step(clicked: int) -> StepResult:
            recommended = np.array([1, 2, 3], dtype=np.int64)
            return StepResult(
                user_id=0,
                round_idx=0,
                recommended_ids=recommended,
                clicked_item=clicked,
                reward=1.0,
                utilities=np.ones(3),
                popularities=np.array([0.8, 0.2, 0.1]),
                intervention_active=True,
            )

        observation = Observation(0, 0, np.array([1, 2, 3]), True, "head")

        popular = StructuredLLMPolicy(
            ScriptedBackend([]), metadata, memory_mode="ersm", history_limit=0
        )
        popular.reset(policy_context())
        rec_pop = popular._update_genre_scores(1, step(1))

        unpopular = StructuredLLMPolicy(
            ScriptedBackend([]), metadata, memory_mode="ersm", history_limit=0
        )
        unpopular.reset(policy_context())
        rec_unpop = unpopular._update_genre_scores(2, step(2))

        # Popular click (pop 0.8) gets lower exposure-corrected confidence
        # than an unpopular click (pop 0.2).
        self.assertLess(rec_pop["confidence"], rec_unpop["confidence"])
        self.assertTrue(rec_pop["residualized"])

        rendered = popular._render_ersm_memory()
        self.assertIn("Comedy", rendered)
        self.assertNotIn("Title", rendered)

    def test_ersm_no_residual_does_not_downweight_popularity(self) -> None:
        metadata = {
            item_id: ItemMetadata(item_id, str(item_id), f"Title {item_id}", ("Comedy",))
            for item_id in range(4)
        }
        result = StepResult(
            user_id=0,
            round_idx=0,
            recommended_ids=np.array([1, 2, 3], dtype=np.int64),
            clicked_item=1,
            reward=1.0,
            utilities=np.ones(3),
            popularities=np.array([0.8, 0.2, 0.1]),
            intervention_active=True,
        )
        policy = StructuredLLMPolicy(
            ScriptedBackend([]),
            metadata,
            memory_mode="ersm_no_residual",
            history_limit=0,
        )
        policy.reset(policy_context())
        record = policy._update_genre_scores(1, result)
        self.assertFalse(record["residualized"])
        self.assertEqual(record["confidence"], 1.0)
        self.assertAlmostEqual(policy.genre_scores["Comedy"], 1.0)

    def test_ersm_consolidated_gates_unconfirmed_session_genres(self) -> None:
        metadata = {
            0: ItemMetadata(0, "0", "Title 0", ("Drama",)),
            1: ItemMetadata(1, "1", "Title 1", ("SciFi",)),
            2: ItemMetadata(2, "2", "Title 2", ("Action",)),
            3: ItemMetadata(3, "3", "Title 3", ("Comedy",)),
        }

        def step() -> StepResult:
            return StepResult(
                user_id=0,
                round_idx=0,
                recommended_ids=np.array([1, 2, 3], dtype=np.int64),
                clicked_item=1,
                reward=1.0,
                utilities=np.ones(3),
                popularities=np.array([0.2, 0.1, 0.1]),
                intervention_active=True,
            )

        policy = StructuredLLMPolicy(
            ScriptedBackend([]),
            metadata,
            memory_mode="ersm_consolidated",
            ersm_commit_threshold=2,
        )
        policy.reset(policy_context())
        # History genre (Drama) is pre-confirmed and always eligible.
        self.assertIn("Drama", policy._render_ersm_memory())

        policy._update_genre_scores(1, step())
        rendered_once = policy._render_ersm_memory()
        self.assertNotIn("SciFi", rendered_once)
        self.assertIn("Drama", rendered_once)

        policy._update_genre_scores(1, step())
        rendered_twice = policy._render_ersm_memory()
        self.assertIn("SciFi", rendered_twice)

    def test_ersm_invariant_anchors_base_and_renders_unordered(self) -> None:
        metadata = {
            0: ItemMetadata(0, "0", "Title 0", ("Drama",)),
            1: ItemMetadata(1, "1", "Title 1", ("SciFi",)),
            2: ItemMetadata(2, "2", "Title 2", ("Action",)),
            3: ItemMetadata(3, "3", "Title 3", ("Comedy",)),
        }

        def step() -> StepResult:
            return StepResult(
                user_id=0,
                round_idx=0,
                recommended_ids=np.array([1, 2, 3], dtype=np.int64),
                clicked_item=1,
                reward=1.0,
                utilities=np.ones(3),
                popularities=np.array([0.2, 0.1, 0.1]),
                intervention_active=True,
            )

        policy = StructuredLLMPolicy(
            ScriptedBackend([]),
            metadata,
            memory_mode="ersm_invariant",
        )
        policy.reset(policy_context())
        rendered = policy._render_ersm_memory()
        self.assertIn("unordered set", rendered)
        self.assertIn("Drama", rendered)

        # One weak session click stays below the strongest base genre, so the
        # arm-invariant base profile is unchanged.
        policy._update_genre_scores(1, step())
        self.assertNotIn("SciFi", policy._render_ersm_memory())

        # A repeatedly confirmed session genre clears the base anchor and is
        # admitted, rendered alphabetically after the base genre.
        policy._update_genre_scores(1, step())
        rendered_twice = policy._render_ersm_memory()
        self.assertIn("SciFi", rendered_twice)
        self.assertLess(
            rendered_twice.index("Drama"), rendered_twice.index("SciFi")
        )


if __name__ == "__main__":
    unittest.main()
