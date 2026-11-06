from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .data import DatasetBundle


ExposureCondition = Literal["head", "balanced", "tail"]


def _stable_seed(base_seed: int, user_id: int, round_idx: int, salt: int) -> int:
    return int(
        (base_seed * 1_000_003 + user_id * 10_007 + round_idx * 101 + salt)
        % (2**32 - 1)
    )


@dataclass(frozen=True)
class Observation:
    user_id: int
    round_idx: int
    candidate_ids: np.ndarray
    intervention_active: bool
    exposure_condition: ExposureCondition
    drift_mode: str = "stable"
    drift_active: bool = False
    drift_progress: float = 0.0


@dataclass(frozen=True)
class StepResult:
    user_id: int
    round_idx: int
    recommended_ids: np.ndarray
    clicked_item: int | None
    reward: float
    utilities: np.ndarray
    popularities: np.ndarray
    intervention_active: bool
    drift_mode: str = "stable"
    drift_active: bool = False
    drift_progress: float = 0.0


class CounterfactualRecEnv:
    """Paired multi-round environment with a short initial exposure treatment."""

    def __init__(
        self,
        bundle: DatasetBundle,
        config: dict,
        preference_anchors: np.ndarray | None = None,
        item_genres: np.ndarray | None = None,
    ):
        self.bundle = bundle
        self.config = config
        # Optional per-item primary genre ids (item_idx -> int). Required only
        # for the genre-aligned adaptive attack (attack_cluster_mode="genre").
        self.item_genres = item_genres
        self.user_id: int | None = None
        self.condition: ExposureCondition | None = None
        self.seed = int(config["seed"])
        self.preference_anchors = (
            bundle.embeddings if preference_anchors is None else preference_anchors
        )
        self.round_idx = 0
        self.drift_mode = "stable"
        self.seen: set[int] = set()
        self.profile = np.empty(bundle.embedding_dim, dtype=np.float32)
        self.target_profile = np.empty(bundle.embedding_dim, dtype=np.float32)
        self.clicked_items: list[int] = []
        self._candidate_plan: list[np.ndarray] = []
        self.attack_target: int | None = None
        self.attack_cluster: np.ndarray = np.empty(0, dtype=np.int64)
        self.observed_mask: np.ndarray | None = None
        self.observed_rank: np.ndarray | None = None

    def reset(
        self,
        user_id: int,
        exposure_condition: ExposureCondition,
        drift_mode: str = "stable",
        candidate_plan: list[np.ndarray] | None = None,
    ) -> Observation:
        if drift_mode not in {"stable", "gradual", "abrupt"}:
            raise ValueError(f"Unknown drift mode: {drift_mode}")
        if exposure_condition not in {"head", "balanced", "tail"}:
            raise ValueError(f"Unknown exposure condition: {exposure_condition}")
        self.user_id = int(user_id)
        self.condition = exposure_condition
        self.drift_mode = drift_mode
        self.round_idx = 0
        self.seen = set(map(int, self.bundle.histories[user_id]))
        self.profile = self.bundle.user_profile(user_id)
        self._set_observed_feedback()
        self.target_profile = self._select_target_profile()
        self.clicked_items = []
        self._setup_attack()
        if candidate_plan is None:
            self._candidate_plan = self._build_candidate_plan()
        else:
            if len(candidate_plan) != int(self.config["rounds"]):
                raise ValueError("candidate_plan length must equal configured rounds")
            self._candidate_plan = [
                np.asarray(items, dtype=np.int64).copy() for items in candidate_plan
            ]
        return self._observation()

    @property
    def candidate_plan(self) -> list[np.ndarray]:
        return [items.copy() for items in self._candidate_plan]

    def drift_progress(self, round_idx: int | None = None) -> float:
        round_idx = self.round_idx if round_idx is None else int(round_idx)
        if self.drift_mode == "stable":
            return 0.0
        drift_start = int(self.config.get("drift_start_round", 4))
        if round_idx < drift_start:
            return 0.0
        if self.drift_mode == "abrupt":
            return 1.0
        duration = max(int(self.config.get("drift_duration", 4)), 1)
        return float(min((round_idx - drift_start + 1) / duration, 1.0))

    def hidden_profile(self, round_idx: int | None = None) -> np.ndarray:
        progress = self.drift_progress(round_idx)
        if progress == 0.0:
            return self.profile
        if progress == 1.0:
            return self.target_profile
        mixed = (1.0 - progress) * self.profile + progress * self.target_profile
        norm = np.linalg.norm(mixed)
        return mixed / norm if norm > 0 else mixed

    def _set_observed_feedback(self) -> None:
        # Fully observed mode: the user's utility for an item is the within-user
        # percentile of the real feedback (KuaiRec watch ratio) on that item, so
        # clicks and relevance come from logged responses, not from a model.
        self.observed_mask = None
        self.observed_rank = None
        if self.config.get("utility_mode", "cosine") != "observed":
            return
        if self.bundle.feedback is None or self.user_id not in self.bundle.feedback:
            raise ValueError("utility_mode='observed' needs bundle.feedback for the user")
        if self.drift_mode != "stable":
            raise ValueError("observed feedback cannot express preference drift")
        values = self.bundle.feedback[self.user_id]
        mask = ~np.isnan(values)
        ranks = np.full(len(values), np.nan, dtype=np.float32)
        observed = values[mask]
        order = np.argsort(observed, kind="stable")
        sorted_values = observed[order]
        # Average rank for ties, scaled to [0, 1].
        low = np.searchsorted(sorted_values, observed, side="left")
        high = np.searchsorted(sorted_values, observed, side="right") - 1
        ranks[mask] = ((low + high) / 2.0) / max(len(observed) - 1, 1)
        self.observed_mask = mask
        self.observed_rank = ranks

    def hidden_utility(
        self,
        item_ids: np.ndarray,
        round_idx: int | None = None,
    ) -> np.ndarray:
        if self.observed_rank is not None:
            return self.observed_rank[np.asarray(item_ids, dtype=np.int64)]
        profile = self.hidden_profile(round_idx)
        if self.config.get("utility_mode", "cosine") == "rank":
            # Graded relevance: per-user percentile rank of cosine similarity
            # against the whole catalogue. This spreads the otherwise compressed
            # cosine band ([~0.77, 0.86]) across the full [0, 1] range so that
            # recommendation quality differences are measurable.
            all_sim = self.bundle.embeddings @ profile
            order = np.argsort(all_sim, kind="stable")
            ranks = np.empty(len(all_sim), dtype=np.float32)
            if len(order) > 1:
                ranks[order] = np.linspace(
                    0.0, 1.0, len(order), dtype=np.float32
                )
            else:
                ranks[order] = 0.5
            return ranks[np.asarray(item_ids, dtype=np.int64)]
        similarities = self.bundle.embeddings[item_ids] @ profile
        return np.clip((similarities + 1.0) / 2.0, 0.0, 1.0)

    def rank_relevance(
        self,
        item_ids: np.ndarray,
        round_idx: int | None = None,
    ) -> np.ndarray:
        """Catalogue percentile of each item's similarity to the hidden profile.

        Logged as an evaluation metric only; it never feeds the click model.
        """
        if self.observed_rank is not None:
            return self.observed_rank[np.asarray(item_ids, dtype=np.int64)]
        all_sim = self.bundle.embeddings @ self.hidden_profile(round_idx)
        item_sim = all_sim[np.asarray(item_ids, dtype=np.int64)]
        return np.searchsorted(np.sort(all_sim), item_sim, side="right") / len(
            all_sim
        )

    def _select_target_profile(self) -> np.ndarray:
        similarities = self.preference_anchors @ self.profile
        target_similarity = float(self.config.get("drift_target_similarity", 0.1))
        anchor = self.preference_anchors[
            int(np.argmin(np.abs(similarities - target_similarity)))
        ]
        magnitude = float(self.config.get("drift_magnitude", 0.8))
        target = (1.0 - magnitude) * self.profile + magnitude * anchor
        norm = np.linalg.norm(target)
        return target / norm if norm > 0 else target

    def is_drift_active(self, round_idx: int | None = None) -> bool:
        return self.drift_progress(round_idx) > 0.0

    def _setup_attack(self) -> None:
        # Early-exposure manipulation: an adversary controls the user's first
        # rounds to promote a target content family ("the promoted item and its
        # neighbours") and inflate its share of the long-run feed. We pick a
        # target that is clickable but below the user's best items, so its
        # natural (clean-arm) share is low and any inflation is over-merit
        # amplification rather than genuine relevance.
        self.attack_target = None
        self.attack_cluster = np.empty(0, dtype=np.int64)
        if not self.config.get("attack_mode", False):
            return
        assert self.user_id is not None
        valid = np.flatnonzero(self.bundle.item_counts > 0)
        valid = np.array([i for i in valid if int(i) not in self.seen])
        if self.observed_mask is not None:
            valid = valid[self.observed_mask[valid]]
        if self.config.get("attack_target_popularity", "any") == "tail":
            # Worst case for catalogue-level propensities: the promoted family
            # is long-tail, so popularity-based weights never discount it.
            tail_max = float(self.config.get("tail_quantile", 0.2))
            valid = valid[self.bundle.popularity[valid] <= tail_max]
        if self.observed_rank is not None:
            sims = self.observed_rank[valid]
        else:
            sims = self.bundle.embeddings[valid] @ self.profile
        order = np.argsort(sims)
        ranks = np.empty(len(valid), dtype=np.float32)
        ranks[order] = np.linspace(0.0, 1.0, len(valid), dtype=np.float32)
        target_pct = float(self.config.get("attack_target_relevance", 0.6))
        target_local = int(np.argmin(np.abs(ranks - target_pct)))
        self.attack_target = int(valid[target_local])
        cluster_size = int(self.config.get("attack_cluster_size", 80))
        target_sims = (
            self.bundle.embeddings[valid]
            @ self.bundle.embeddings[self.attack_target]
        )
        cluster_mode = str(self.config.get("attack_cluster_mode", "embedding"))
        if cluster_mode == "genre":
            # Adaptive attacker that knows the defense abstracts memory to
            # genres: build the promoted family from items sharing the target's
            # primary genre so the manipulated signal survives genre folding.
            if self.item_genres is None:
                raise ValueError(
                    "attack_cluster_mode='genre' requires item_genres"
                )
            target_genre = int(self.item_genres[self.attack_target])
            genre_mask = self.item_genres[valid] == target_genre
            genre_local = np.flatnonzero(genre_mask)
            # Prefer the genre members most similar to the target, then top up
            # with embedding neighbours if the genre is too small to fill the
            # cluster (keeps candidate pools correctly sized across arms).
            genre_order = genre_local[np.argsort(-target_sims[genre_local])]
            chosen = genre_order[:cluster_size]
            if len(chosen) < cluster_size:
                fill = np.argsort(-target_sims)
                extra = [i for i in fill if i not in set(chosen.tolist())]
                needed = cluster_size - len(chosen)
                chosen = np.concatenate([chosen, np.array(extra[:needed], dtype=int)])
            cluster_local = chosen
        else:
            cluster_local = np.argsort(-target_sims)[:cluster_size]
        self.attack_cluster = np.sort(valid[cluster_local]).astype(np.int64)

    def step(self, agent_action: np.ndarray) -> tuple[Observation | None, StepResult]:
        if self.user_id is None or self.condition is None:
            raise RuntimeError("reset must be called before step")
        candidates = self._candidate_plan[self.round_idx]
        action = np.asarray(agent_action, dtype=np.int64)
        if len(action) == 0:
            raise ValueError("agent_action must contain at least one item")
        if len(np.unique(action)) != len(action):
            raise ValueError("agent_action contains duplicate items")
        if not set(action).issubset(set(candidates)):
            raise ValueError("agent_action contains items outside the candidate pool")
        action = self._apply_sponsored_pins(action, candidates)

        utilities = self.hidden_utility(action)
        clicked_item = self._simulate_feedback(action, utilities)
        reward = 0.0
        if clicked_item is not None:
            reward = float(utilities[np.flatnonzero(action == clicked_item)[0]])

        result = StepResult(
            user_id=self.user_id,
            round_idx=self.round_idx,
            recommended_ids=action.copy(),
            clicked_item=clicked_item,
            reward=reward,
            utilities=utilities.copy(),
            popularities=self.bundle.popularity[action].copy(),
            intervention_active=self.round_idx
            < int(self.config["intervention_rounds"]),
            drift_mode=self.drift_mode,
            drift_active=self.is_drift_active(self.round_idx),
            drift_progress=self.drift_progress(self.round_idx),
        )
        self.seen.update(map(int, action))
        if clicked_item is not None:
            self.clicked_items.append(int(clicked_item))
        self.round_idx += 1
        next_observation = (
            self._observation()
            if self.round_idx < int(self.config["rounds"])
            else None
        )
        return next_observation, result

    def _apply_sponsored_pins(
        self, action: np.ndarray, candidates: np.ndarray
    ) -> np.ndarray:
        # Sponsored-slot attacker: during the attack rounds of the attack arm,
        # the platform overwrites the trailing slate positions with promoted
        # items, so the user sees them regardless of the agent's choice.
        pins = int(self.config.get("attack_pin", 0))
        if (
            pins <= 0
            or not self.config.get("attack_mode", False)
            or self.condition != "head"
            or self.round_idx >= int(self.config["intervention_rounds"])
        ):
            return action
        cluster = set(map(int, self.attack_cluster))
        already = sum(int(i) in cluster for i in action)
        spare = [
            int(i)
            for i in candidates
            if int(i) in cluster and int(i) not in set(map(int, action))
        ]
        output = action.copy()
        position = len(output) - 1
        for item in spare[: max(min(pins, len(output)) - already, 0)]:
            while position >= 0 and int(output[position]) in cluster:
                position -= 1
            if position < 0:
                break
            output[position] = item
            position -= 1
        return output

    def _observation(self) -> Observation:
        assert self.user_id is not None
        assert self.condition is not None
        return Observation(
            user_id=self.user_id,
            round_idx=self.round_idx,
            candidate_ids=self._candidate_plan[self.round_idx].copy(),
            intervention_active=self.round_idx
            < int(self.config["intervention_rounds"]),
            exposure_condition=self.condition,
            drift_mode=self.drift_mode,
            drift_active=self.is_drift_active(self.round_idx),
            drift_progress=self.drift_progress(self.round_idx),
        )

    def _simulate_feedback(
        self,
        action: np.ndarray,
        utilities: np.ndarray,
    ) -> int | None:
        position_penalty = float(self.config.get("position_penalty", 0.0))
        adjusted = utilities - position_penalty * np.log2(np.arange(len(action)) + 2)
        feedback_model = self.config.get("feedback_model", "deterministic")
        if feedback_model == "deterministic":
            best = int(np.argmax(adjusted))
            if adjusted[best] < float(self.config.get("click_threshold", 0.5)):
                return None
            return int(action[best])
        if feedback_model == "logistic":
            return self._sample_choice(action, adjusted)
        if feedback_model == "nonlinear":
            nonlinear = np.square(np.clip(adjusted, 0.0, 1.0))
            if self.clicked_items:
                clicked_embeddings = self.bundle.embeddings[self.clicked_items]
                similarity = self.bundle.embeddings[action] @ clicked_embeddings.T
                satiation = np.maximum(similarity.max(axis=1), 0.0)
                nonlinear -= float(self.config.get("satiation_strength", 0.15)) * satiation
            threshold = float(self.config.get("click_threshold", 0.5)) ** 2
            return self._sample_choice(action, nonlinear, threshold=threshold)
        raise ValueError(f"Unknown feedback_model: {feedback_model}")

    def _sample_choice(
        self,
        action: np.ndarray,
        scores: np.ndarray,
        threshold: float | None = None,
    ) -> int | None:
        threshold = (
            float(self.config.get("click_threshold", 0.5))
            if threshold is None
            else threshold
        )
        temperature = float(self.config.get("feedback_temperature", 0.1))
        logits = (scores - threshold) / max(temperature, 1e-6)
        click_weights = np.exp(logits - logits.max())
        probabilities = np.append(click_weights, 1.0)
        probabilities /= probabilities.sum()
        rng = np.random.default_rng(
            _stable_seed(self.seed, int(self.user_id), self.round_idx, 991)
        )
        selected = int(rng.choice(len(probabilities), p=probabilities))
        return None if selected == len(action) else int(action[selected])

    def _build_candidate_plan(self) -> list[np.ndarray]:
        assert self.user_id is not None
        intervention_rounds = int(self.config["intervention_rounds"])
        intervention_plans: dict[str, list[np.ndarray]] = {}
        intervention_union = set(self.seen)

        # Reserve every treatment-arm item before drawing post-treatment pools.
        # This keeps later candidate IDs identical without repeating an item that
        # appeared in only one early-exposure arm.
        for condition in ("head", "balanced", "tail"):
            condition_used = set(self.seen)
            condition_plan = []
            for round_idx in range(intervention_rounds):
                candidates = self._intervention_candidates(
                    round_idx, condition_used, condition
                )
                condition_plan.append(candidates)
                condition_used.update(map(int, candidates))
                intervention_union.update(map(int, candidates))
            intervention_plans[condition] = condition_plan

        common_plan = []
        common_used = set(intervention_union)
        for round_idx in range(intervention_rounds, int(self.config["rounds"])):
            candidates = self._common_candidates(round_idx, common_used)
            common_plan.append(candidates)
            common_used.update(map(int, candidates))

        return intervention_plans[self.condition] + common_plan

    def _available_items(self, used: set[int]) -> np.ndarray:
        mask = np.ones(self.bundle.n_items, dtype=bool)
        if used:
            mask[np.fromiter(used, dtype=np.int64)] = False
        mask &= self.bundle.item_counts > 0
        if self.observed_mask is not None:
            mask &= self.observed_mask
        return np.flatnonzero(mask)

    def _intervention_candidates(
        self,
        round_idx: int,
        used: set[int],
        condition: ExposureCondition,
    ) -> np.ndarray:
        assert self.user_id is not None
        if self.config.get("attack_mode", False):
            return self._attack_intervention_candidates(
                round_idx, used, condition
            )
        available = self._available_items(used)
        utility = self.hidden_utility(available, round_idx)
        popularity = self.bundle.popularity[available]

        target_mask = utility >= np.quantile(
            utility, float(self.config.get("utility_target_quantile", 0.5))
        )
        target_pool = available[target_mask]
        target_utility = self.hidden_utility(target_pool, round_idx)
        target_rng = np.random.default_rng(
            _stable_seed(self.seed, self.user_id, round_idx, 17)
        )
        size = int(self.config["candidate_size"])
        replace = len(target_pool) < size
        target_ids = target_rng.choice(target_pool, size=size, replace=replace)
        targets = self.hidden_utility(target_ids, round_idx)

        if condition == "head":
            condition_mask = popularity >= float(self.config["head_quantile"])
        elif condition == "tail":
            condition_mask = popularity <= float(self.config["tail_quantile"])
        else:
            condition_mask = (
                popularity >= float(self.config["balanced_low"])
            ) & (popularity <= float(self.config["balanced_high"]))

        condition_ids = available[condition_mask]
        if len(condition_ids) < size:
            raise ValueError(
                f"Not enough {condition} candidates for user {self.user_id}"
            )
        condition_utility = self.hidden_utility(condition_ids, round_idx)
        selected = self._nearest_unique_matches(
            condition_ids, condition_utility, targets
        )
        shuffle_rng = np.random.default_rng(
            _stable_seed(self.seed, self.user_id, round_idx, 29)
        )
        shuffle_rng.shuffle(selected)
        return selected

    def _attack_intervention_candidates(
        self,
        round_idx: int,
        used: set[int],
        condition: ExposureCondition,
    ) -> np.ndarray:
        assert self.user_id is not None
        available = self._available_items(used)
        size = int(self.config["candidate_size"])
        rng = np.random.default_rng(
            _stable_seed(self.seed, self.user_id, round_idx, 71)
        )
        if self.config.get("attack_injection", "pool") == "replace":
            return self._replace_injection_candidates(
                round_idx, available, size, condition
            )
        if condition == "head":
            # Attack arm: saturate the early slate with target-cluster items so
            # the agent's memory absorbs the promoted content family.
            cluster = np.array(
                [i for i in self.attack_cluster if int(i) not in used],
                dtype=np.int64,
            )
            n_attack = min(int(self.config.get("attack_slate", size)), size)
            n_attack = min(n_attack, len(cluster))
            chosen = list(rng.choice(cluster, size=n_attack, replace=False))
            used_local = set(int(i) for i in chosen)
            filler = np.array(
                [i for i in available if int(i) not in used_local],
                dtype=np.int64,
            )
            remaining = size - len(chosen)
            chosen.extend(
                int(i)
                for i in rng.choice(filler, size=remaining, replace=False)
            )
            selected = np.asarray(chosen, dtype=np.int64)
        else:
            # Clean arm: a neutral balanced slate (no promoted content).
            popularity = self.bundle.popularity[available]
            neutral = available[
                (popularity >= float(self.config.get("balanced_low", 0.4)))
                & (popularity <= float(self.config.get("balanced_high", 0.6)))
            ]
            neutral = neutral[~np.isin(neutral, self.attack_cluster)]
            if len(neutral) < size:
                neutral = available[~np.isin(available, self.attack_cluster)]
            selected = rng.choice(neutral, size=size, replace=False).astype(
                np.int64
            )
        rng.shuffle(selected)
        return selected

    def _replace_injection_candidates(
        self,
        round_idx: int,
        available: np.ndarray,
        size: int,
        condition: ExposureCondition,
    ) -> np.ndarray:
        # Partial-control attacker: both arms draw the same organic pool, and
        # the attack arm swaps only ``attack_slate`` of its slots for promoted
        # items, so the arms differ by exactly the injected fraction.
        assert self.user_id is not None
        organic = available[~np.isin(available, self.attack_cluster)]
        organic_rng = np.random.default_rng(
            _stable_seed(self.seed, self.user_id, round_idx, 73)
        )
        selected = organic_rng.choice(organic, size=size, replace=False)
        if condition == "head":
            cluster = self.attack_cluster[np.isin(self.attack_cluster, available)]
            n_attack = min(int(self.config.get("attack_slate", size)), size, len(cluster))
            inject_rng = np.random.default_rng(
                _stable_seed(self.seed, self.user_id, round_idx, 79)
            )
            slots = inject_rng.choice(size, size=n_attack, replace=False)
            selected[slots] = inject_rng.choice(cluster, size=n_attack, replace=False)
        return selected.astype(np.int64)

    @staticmethod
    def _nearest_unique_matches(
        item_ids: np.ndarray,
        utilities: np.ndarray,
        targets: np.ndarray,
    ) -> np.ndarray:
        remaining = np.ones(len(item_ids), dtype=bool)
        selected: list[int] = []
        for target in np.sort(targets):
            distances = np.abs(utilities - target)
            distances[~remaining] = np.inf
            index = int(np.argmin(distances))
            selected.append(int(item_ids[index]))
            remaining[index] = False
        return np.asarray(selected, dtype=np.int64)

    def _common_candidates(
        self,
        round_idx: int,
        used: set[int],
    ) -> np.ndarray:
        assert self.user_id is not None
        available = self._available_items(used)
        popularity = self.bundle.popularity[available]
        size = int(self.config["candidate_size"])
        rng = np.random.default_rng(
            _stable_seed(self.seed, self.user_id, round_idx, 53)
        )

        if self.config.get("attack_mode", False):
            return self._attack_common_candidates(
                round_idx, available, size, rng
            )

        if self.config.get("pool_mode", "stratified") == "niche_trap":
            return self._niche_trap_candidates(
                round_idx, available, popularity, size, rng
            )

        bands = [
            available[popularity <= 1.0 / 3.0],
            available[
                (popularity > 1.0 / 3.0) & (popularity < 2.0 / 3.0)
            ],
            available[popularity >= 2.0 / 3.0],
        ]
        quotas = [size // 3] * 3
        for index in range(size % 3):
            quotas[index] += 1

        selected = []
        for band, quota in zip(bands, quotas):
            if len(band) < quota:
                raise ValueError("Not enough items for common candidate strata")
            selected.extend(rng.choice(band, size=quota, replace=False).tolist())
        selected_array = np.asarray(selected, dtype=np.int64)
        rng.shuffle(selected_array)
        return selected_array

    def _attack_common_candidates(
        self,
        round_idx: int,
        available: np.ndarray,
        size: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        # Inject a fixed quota of target-cluster items into every post-attack
        # pool so both arms have the same opportunity to recommend the promoted
        # content; any difference in its recommended share is then attributable
        # to the early-exposure manipulation, not to availability.
        n_target = int(self.config.get("attack_pool_target", size // 4))
        cluster = self.attack_cluster[np.isin(self.attack_cluster, available)]
        n_target = min(n_target, len(cluster))
        chosen = list(rng.choice(cluster, size=n_target, replace=False))
        filler = available[~np.isin(available, np.asarray(chosen, dtype=np.int64))]
        remaining = size - len(chosen)
        if self.config.get("attack_pool_mode", "random") == "hard":
            # Hard pool: organic candidates come from the user's own
            # high-relevance band, so a coarse attribute profile cannot
            # separate them and item-level discrimination is required.
            filler = filler[~np.isin(filler, self.attack_cluster)]
            relevance = self.hidden_utility(filler, round_idx)
            band = float(self.config.get("hard_pool_quantile", 0.7))
            filler = filler[relevance >= np.quantile(relevance, band)]
        if len(filler) < remaining:
            raise ValueError("Not enough items for attack pool filler")
        chosen.extend(
            int(i) for i in rng.choice(filler, size=remaining, replace=False)
        )
        selected = np.asarray(chosen, dtype=np.int64)
        rng.shuffle(selected)
        return selected

    def _niche_trap_candidates(
        self,
        round_idx: int,
        available: np.ndarray,
        popularity: np.ndarray,
        size: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        # A popularity-trap stress pool: it mixes niche high-relevance "gems"
        # (which a preference-faithful agent should pick) with popular
        # low-relevance "traps" (which a popularity-locked agent gravitates to),
        # plus neutral filler. This puts popularity and relevance in tension so
        # that lock-in carries a measurable relevance cost.
        relevance = self.hidden_utility(available, round_idx)
        gem_rel = float(self.config.get("gem_relevance_quantile", 0.85))
        gem_pop = float(self.config.get("gem_popularity_max", 0.4))
        trap_pop = float(self.config.get("trap_popularity_min", 0.7))
        trap_rel = float(self.config.get("trap_relevance_max", 0.55))
        n_gem = int(self.config.get("n_gem", size // 3))
        n_trap = int(self.config.get("n_trap", size // 3))

        rel_high = np.quantile(relevance, gem_rel)
        gem_mask = (relevance >= rel_high) & (popularity <= gem_pop)
        trap_mask = (popularity >= trap_pop) & (relevance <= trap_rel)

        chosen: list[int] = []
        used_local: set[int] = set()

        def take(pool_idx: np.ndarray, count: int, by_relevance: bool) -> None:
            pool = available[pool_idx]
            pool = np.array([i for i in pool if int(i) not in used_local])
            if len(pool) == 0 or count <= 0:
                return
            if by_relevance:
                rel = self.hidden_utility(pool, round_idx)
                order = np.argsort(-rel)
                picked = pool[order[:count]]
            else:
                picked = rng.choice(
                    pool, size=min(count, len(pool)), replace=False
                )
            for item in picked:
                chosen.append(int(item))
                used_local.add(int(item))

        take(np.flatnonzero(gem_mask), n_gem, by_relevance=True)
        take(np.flatnonzero(trap_mask), n_trap, by_relevance=False)
        # Fill the rest with neutral items so the pool always reaches `size`.
        remaining = size - len(chosen)
        if remaining > 0:
            filler = np.array(
                [int(i) for i in available if int(i) not in used_local]
            )
            if len(filler) < remaining:
                raise ValueError("Not enough items for niche-trap filler")
            for item in rng.choice(filler, size=remaining, replace=False):
                chosen.append(int(item))
        selected_array = np.asarray(chosen[:size], dtype=np.int64)
        rng.shuffle(selected_array)
        return selected_array

