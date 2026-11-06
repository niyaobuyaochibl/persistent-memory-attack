from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .environment import Observation, StepResult


def _normalize(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    return vector / norm if norm > 0 else vector


@dataclass(frozen=True)
class PolicyContext:
    base_profile: np.ndarray
    history_items: np.ndarray
    embeddings: np.ndarray
    popularity: np.ndarray
    cluster_assignments: np.ndarray
    cluster_centroids: np.ndarray


@dataclass(frozen=True)
class MemoryRecord:
    item_id: int
    round_idx: int
    signal_type: str
    source: str
    explicit: bool
    confidence: float
    exposure_prevalence: float
    cluster_id: int


class Policy(Protocol):
    name: str

    def reset(self, context: PolicyContext) -> None:
        ...

    def recommend(self, observation: Observation, top_k: int) -> np.ndarray:
        ...

    def observe(self, observation: Observation, result: StepResult) -> None:
        ...


class ProfilePolicy:
    name = "profile"

    def __init__(self, popularity_prior_weight: float = 0.0):
        self.popularity_prior_weight = popularity_prior_weight
        self.context: PolicyContext | None = None
        self.profile: np.ndarray | None = None
        self.memory_records: list[MemoryRecord] = []

    def reset(self, context: PolicyContext) -> None:
        self.context = context
        self.profile = context.base_profile.copy()
        self.memory_records = []

    def recommend(self, observation: Observation, top_k: int) -> np.ndarray:
        if self.context is None or self.profile is None:
            raise RuntimeError("reset must be called before recommend")
        items = observation.candidate_ids
        scores = self.context.embeddings[items] @ self.profile
        scores += self.popularity_prior_weight * self.context.popularity[items]
        order = np.argsort(-scores, kind="stable")[:top_k]
        return items[order]

    def _record_feedback(
        self, observation: Observation, result: StepResult
    ) -> None:
        assert self.context is not None
        for item_id in result.recommended_ids:
            clicked = result.clicked_item == int(item_id)
            self.memory_records.append(
                MemoryRecord(
                    item_id=int(item_id),
                    round_idx=int(observation.round_idx),
                    signal_type=(
                        "implicit_positive" if clicked else "implicit_nonpositive"
                    ),
                    source="simulated_feedback",
                    explicit=False,
                    confidence=1.0,
                    exposure_prevalence=float(self.context.popularity[item_id]),
                    cluster_id=int(self.context.cluster_assignments[item_id]),
                )
            )

    def observe(self, observation: Observation, result: StepResult) -> None:
        del observation, result


class NoMemoryPolicy(ProfilePolicy):
    name = "no_memory"


class RawMemoryPolicy(ProfilePolicy):
    name = "raw_memory"

    def __init__(
        self,
        update_rate: float,
        popularity_prior_weight: float = 0.0,
    ):
        super().__init__(popularity_prior_weight)
        self.update_rate = update_rate

    def observe(self, observation: Observation, result: StepResult) -> None:
        self._record_feedback(observation, result)
        if result.clicked_item is None:
            return
        assert self.context is not None
        assert self.profile is not None
        clicked = self.context.embeddings[result.clicked_item]
        self.profile = _normalize(
            (1.0 - self.update_rate) * self.profile + self.update_rate * clicked
        )


class NoEarlyWritePolicy(RawMemoryPolicy):
    """Placebo that ignores all feedback during the exposure intervention."""

    name = "raw_no_early_write"

    def observe(self, observation: Observation, result: StepResult) -> None:
        if observation.intervention_active:
            return
        super().observe(observation, result)


class MemoryResetPolicy(RawMemoryPolicy):
    """Placebo that clears treatment-generated memory before round 1 post-treatment."""

    name = "raw_reset"

    def __init__(
        self,
        update_rate: float,
        popularity_prior_weight: float = 0.0,
    ):
        super().__init__(update_rate, popularity_prior_weight)
        self._post_intervention_reset = False

    def reset(self, context: PolicyContext) -> None:
        super().reset(context)
        self._post_intervention_reset = False

    def recommend(self, observation: Observation, top_k: int) -> np.ndarray:
        if not observation.intervention_active and not self._post_intervention_reset:
            assert self.context is not None
            self.profile = self.context.base_profile.copy()
            self.memory_records = []
            self._post_intervention_reset = True
        return super().recommend(observation, top_k)


class SlidingWindowPolicy(ProfilePolicy):
    name = "sliding_window"

    def __init__(
        self,
        window_size: int,
        base_strength: float,
        popularity_prior_weight: float = 0.0,
    ):
        super().__init__(popularity_prior_weight)
        self.window_size = window_size
        self.base_strength = base_strength
        self.clicked_items: list[int] = []

    def reset(self, context: PolicyContext) -> None:
        super().reset(context)
        self.clicked_items = []

    def observe(self, observation: Observation, result: StepResult) -> None:
        self._record_feedback(observation, result)
        if result.clicked_item is None:
            return
        assert self.context is not None
        self.clicked_items.append(int(result.clicked_item))
        self.clicked_items = self.clicked_items[-self.window_size :]
        memory_vector = self.context.embeddings[self.clicked_items].mean(axis=0)
        self.profile = _normalize(
            self.base_strength * self.context.base_profile + memory_vector
        )


class RecencyWeightedPolicy(ProfilePolicy):
    name = "recency_weighted"

    def __init__(
        self,
        decay: float,
        base_strength: float,
        popularity_prior_weight: float = 0.0,
    ):
        super().__init__(popularity_prior_weight)
        self.decay = decay
        self.base_strength = base_strength
        self.clicked_items: list[int] = []

    def reset(self, context: PolicyContext) -> None:
        super().reset(context)
        self.clicked_items = []

    def observe(self, observation: Observation, result: StepResult) -> None:
        self._record_feedback(observation, result)
        if result.clicked_item is None:
            return
        assert self.context is not None
        self.clicked_items.append(int(result.clicked_item))
        ages = np.arange(len(self.clicked_items) - 1, -1, -1, dtype=np.float64)
        weights = np.power(self.decay, ages)
        weights /= weights.sum()
        memory_vector = weights @ self.context.embeddings[self.clicked_items]
        self.profile = _normalize(
            self.base_strength * self.context.base_profile + memory_vector
        )


class ExposureNormalizedPolicy(ProfilePolicy):
    """Non-Bayesian exposure-normalized profile baseline."""

    name = "exposure_normalized"

    def __init__(
        self,
        base_strength: float,
        decay: float,
        popularity_prior_weight: float = 0.0,
    ):
        super().__init__(popularity_prior_weight)
        self.base_strength = base_strength
        self.decay = decay
        self.exposures = np.empty(0, dtype=np.float64)
        self.positives = np.empty(0, dtype=np.float64)

    def reset(self, context: PolicyContext) -> None:
        super().reset(context)
        n_clusters = len(context.cluster_centroids)
        self.exposures = np.zeros(n_clusters, dtype=np.float64)
        self.positives = np.zeros(n_clusters, dtype=np.float64)

    def observe(self, observation: Observation, result: StepResult) -> None:
        self._record_feedback(observation, result)
        assert self.context is not None
        self.exposures *= self.decay
        self.positives *= self.decay
        for item_id in result.recommended_ids:
            cluster_id = self.context.cluster_assignments[item_id]
            self.exposures[cluster_id] += 1.0
        if result.clicked_item is not None:
            cluster_id = self.context.cluster_assignments[result.clicked_item]
            self.positives[cluster_id] += 1.0
        rates = np.divide(
            self.positives,
            self.exposures,
            out=np.zeros_like(self.positives),
            where=self.exposures > 0,
        )
        memory_vector = rates @ self.context.cluster_centroids
        self.profile = _normalize(
            self.base_strength * self.context.base_profile + memory_vector
        )


class LinearBanditPolicy(ProfilePolicy):
    """Shared implementation for semantic-feature linear bandits."""

    def __init__(
        self,
        regularization: float,
        base_weight: float,
        seed: int,
        popularity_prior_weight: float = 0.0,
    ):
        super().__init__(popularity_prior_weight)
        self.regularization = regularization
        self.base_weight = base_weight
        self.seed = seed
        self.a = np.empty((0, 0), dtype=np.float64)
        self.b = np.empty(0, dtype=np.float64)

    def reset(self, context: PolicyContext) -> None:
        super().reset(context)
        dimension = len(context.cluster_centroids)
        self.a = self.regularization * np.eye(dimension, dtype=np.float64)
        self.b = np.zeros(dimension, dtype=np.float64)

    def _features(self, item_ids: np.ndarray) -> np.ndarray:
        assert self.context is not None
        features = self.context.embeddings[item_ids] @ self.context.cluster_centroids.T
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        return features / norms

    def _base_scores(self, item_ids: np.ndarray) -> np.ndarray:
        assert self.context is not None
        scores = self.base_weight * (
            self.context.embeddings[item_ids] @ self.context.base_profile
        )
        scores += self.popularity_prior_weight * self.context.popularity[item_ids]
        return scores

    def observe(self, observation: Observation, result: StepResult) -> None:
        self._record_feedback(observation, result)
        features = self._features(result.recommended_ids)
        for item_id, feature in zip(result.recommended_ids, features):
            reward = 1.0 if result.clicked_item == int(item_id) else 0.0
            self.a += np.outer(feature, feature)
            self.b += reward * feature


class LinUCBPolicy(LinearBanditPolicy):
    name = "linucb"

    def __init__(
        self,
        alpha: float,
        regularization: float,
        base_weight: float,
        seed: int,
        popularity_prior_weight: float = 0.0,
    ):
        super().__init__(regularization, base_weight, seed, popularity_prior_weight)
        self.alpha = alpha

    def recommend(self, observation: Observation, top_k: int) -> np.ndarray:
        features = self._features(observation.candidate_ids)
        a_inv = np.linalg.inv(self.a)
        theta = a_inv @ self.b
        uncertainty = np.sqrt(
            np.maximum(np.sum((features @ a_inv) * features, axis=1), 0.0)
        )
        scores = self._base_scores(observation.candidate_ids)
        scores += features @ theta + self.alpha * uncertainty
        order = np.argsort(-scores, kind="stable")[:top_k]
        return observation.candidate_ids[order]


class ThompsonSamplingPolicy(LinearBanditPolicy):
    name = "thompson"

    def __init__(
        self,
        sample_scale: float,
        regularization: float,
        base_weight: float,
        seed: int,
        popularity_prior_weight: float = 0.0,
    ):
        super().__init__(regularization, base_weight, seed, popularity_prior_weight)
        self.sample_scale = sample_scale

    def recommend(self, observation: Observation, top_k: int) -> np.ndarray:
        features = self._features(observation.candidate_ids)
        a_inv = np.linalg.inv(self.a)
        mean = a_inv @ self.b
        covariance = self.sample_scale**2 * a_inv
        rng = np.random.default_rng(
            (
                self.seed * 1_000_003
                + observation.user_id * 10_007
                + observation.round_idx * 101
                + 733
            )
            % (2**32 - 1)
        )
        theta = rng.multivariate_normal(mean, covariance)
        scores = self._base_scores(observation.candidate_ids) + features @ theta
        order = np.argsort(-scores, kind="stable")[:top_k]
        return observation.candidate_ids[order]


class EAMPolicy(ProfilePolicy):
    """Exposure-residualized memory that learns choices relative to the slate."""

    name = "eam"

    def __init__(
        self,
        base_strength: float,
        alpha: float,
        beta: float,
        decay: float = 1.0,
        residual_strength: float = 1.0,
        cluster_strength: float = 1.0,
        propensity_floor: float = 0.2,
        reference_propensity: float = 0.5,
        exploration_bonus: float = 0.0,
        exploration_slots: int = 0,
        popularity_prior_weight: float = 0.0,
    ):
        super().__init__(popularity_prior_weight)
        self.base_strength = base_strength
        self.alpha = alpha
        self.beta = beta
        self.decay = decay
        self.residual_strength = residual_strength
        self.cluster_strength = cluster_strength
        self.propensity_floor = propensity_floor
        self.reference_propensity = reference_propensity
        self.exploration_bonus = exploration_bonus
        self.exploration_slots = exploration_slots
        self.exposures = np.empty(0, dtype=np.float64)
        self.positives = np.empty(0, dtype=np.float64)
        self.residual_vector = np.empty(0, dtype=np.float64)
        self.residual_weight = 0.0

    def reset(self, context: PolicyContext) -> None:
        super().reset(context)
        n_clusters = len(context.cluster_centroids)
        self.exposures = np.zeros(n_clusters, dtype=np.float64)
        self.positives = np.zeros(n_clusters, dtype=np.float64)
        self.residual_vector = np.zeros(
            context.embeddings.shape[1], dtype=np.float64
        )
        self.residual_weight = 0.0

    def _implicit_confidence(self, item_id: int) -> float:
        assert self.context is not None
        popularity = float(self.context.popularity[item_id])
        propensity = self.propensity_floor + (1.0 - self.propensity_floor) * popularity
        return min(1.0, self.reference_propensity / max(propensity, 1e-8))

    def recommend(self, observation: Observation, top_k: int) -> np.ndarray:
        if self.context is None or self.profile is None:
            raise RuntimeError("reset must be called before recommend")
        items = observation.candidate_ids
        scores = self.context.embeddings[items] @ self.profile
        scores += self.popularity_prior_weight * self.context.popularity[items]
        slots = min(max(self.exploration_slots, 0), max(top_k - 1, 0))
        if slots == 0 or self.exploration_bonus <= 0.0:
            return items[np.argsort(-scores, kind="stable")[:top_k]]

        exploit_count = top_k - slots
        exploit_order = np.argsort(-scores, kind="stable")[:exploit_count]
        selected_mask = np.zeros(len(items), dtype=bool)
        selected_mask[exploit_order] = True
        cluster_ids = self.context.cluster_assignments[items]
        uncertainty = 1.0 / np.sqrt(
            self.exposures[cluster_ids] + self.alpha + self.beta
        )
        explore_scores = scores + self.exploration_bonus * uncertainty
        explore_scores[selected_mask] = -np.inf
        explore_order = np.argsort(-explore_scores, kind="stable")[:slots]
        return items[np.concatenate([exploit_order, explore_order])]

    def observe(self, observation: Observation, result: StepResult) -> None:
        self._record_feedback(observation, result)
        assert self.context is not None
        self.exposures *= self.decay
        self.positives *= self.decay
        self.residual_vector *= self.decay
        self.residual_weight *= self.decay
        for item_id in result.recommended_ids:
            cluster_id = self.context.cluster_assignments[item_id]
            self.exposures[cluster_id] += 1.0
        if result.clicked_item is not None:
            item_id = int(result.clicked_item)
            cluster_id = self.context.cluster_assignments[item_id]
            confidence = self._implicit_confidence(item_id)
            slate = self.context.embeddings[result.recommended_ids]
            residual = self.context.embeddings[item_id] - slate.mean(axis=0)
            self.positives[cluster_id] += confidence
            self.residual_vector += confidence * residual
            self.residual_weight += confidence
        self.profile = self._posterior_profile()

    def _posterior_profile(self) -> np.ndarray:
        assert self.context is not None
        posterior = (self.positives + self.alpha) / (
            self.exposures + self.alpha + self.beta
        )
        prior_rate = self.alpha / (self.alpha + self.beta)
        cluster_evidence = posterior - prior_rate
        cluster_vector = cluster_evidence @ self.context.cluster_centroids
        if self.residual_weight > 0.0:
            residual_vector = self.residual_vector / self.residual_weight
        else:
            residual_vector = np.zeros_like(self.residual_vector)
        combined = (
            self.base_strength * self.context.base_profile
            + self.residual_strength * residual_vector
            + self.cluster_strength * cluster_vector
        )
        return _normalize(combined)


class DelayedCommitmentPolicy(ProfilePolicy):
    """Exposure-aware memory with delayed commitment and temporal decay."""

    name = "eam_delayed"

    def __init__(
        self,
        base_strength: float,
        alpha: float,
        beta: float,
        decay: float = 1.0,
        base_decay: float = 1.0,
        base_floor: float = 0.5,
        quarantine_rounds: int = 0,
        positive_strength: float = 1.0,
        cluster_strength: float = 1.0,
        propensity_floor: float = 0.2,
        reference_propensity: float = 0.5,
        exploration_bonus: float = 0.0,
        exploration_slots: int = 0,
        popularity_prior_weight: float = 0.0,
    ):
        super().__init__(popularity_prior_weight)
        self.base_strength = base_strength
        self.alpha = alpha
        self.beta = beta
        self.decay = decay
        self.base_decay = base_decay
        self.base_floor = base_floor
        self.quarantine_rounds = quarantine_rounds
        self.positive_strength = positive_strength
        self.cluster_strength = cluster_strength
        self.propensity_floor = propensity_floor
        self.reference_propensity = reference_propensity
        self.exploration_bonus = exploration_bonus
        self.exploration_slots = exploration_slots
        self.exposures = np.empty(0, dtype=np.float64)
        self.positives = np.empty(0, dtype=np.float64)
        self.positive_vector = np.empty(0, dtype=np.float64)
        self.positive_weight = 0.0
        self.committed_observations = 0

    def reset(self, context: PolicyContext) -> None:
        super().reset(context)
        n_clusters = len(context.cluster_centroids)
        self.exposures = np.zeros(n_clusters, dtype=np.float64)
        self.positives = np.zeros(n_clusters, dtype=np.float64)
        self.positive_vector = np.zeros(
            context.embeddings.shape[1], dtype=np.float64
        )
        self.positive_weight = 0.0
        self.committed_observations = 0

    def _implicit_confidence(self, item_id: int) -> float:
        assert self.context is not None
        popularity = float(self.context.popularity[item_id])
        propensity = self.propensity_floor + (1.0 - self.propensity_floor) * popularity
        return min(1.0, self.reference_propensity / max(propensity, 1e-8))

    def recommend(self, observation: Observation, top_k: int) -> np.ndarray:
        if self.context is None or self.profile is None:
            raise RuntimeError("reset must be called before recommend")
        items = observation.candidate_ids
        scores = self.context.embeddings[items] @ self.profile
        scores += self.popularity_prior_weight * self.context.popularity[items]
        slots = min(max(self.exploration_slots, 0), max(top_k - 1, 0))
        if slots == 0 or self.exploration_bonus <= 0.0:
            return items[np.argsort(-scores, kind="stable")[:top_k]]

        exploit_count = top_k - slots
        exploit_order = np.argsort(-scores, kind="stable")[:exploit_count]
        selected_mask = np.zeros(len(items), dtype=bool)
        selected_mask[exploit_order] = True
        cluster_ids = self.context.cluster_assignments[items]
        uncertainty = 1.0 / np.sqrt(
            self.exposures[cluster_ids] + self.alpha + self.beta
        )
        explore_scores = scores + self.exploration_bonus * uncertainty
        explore_scores[selected_mask] = -np.inf
        explore_order = np.argsort(-explore_scores, kind="stable")[:slots]
        return items[np.concatenate([exploit_order, explore_order])]

    def observe(self, observation: Observation, result: StepResult) -> None:
        self._record_feedback(observation, result)
        if observation.round_idx < self.quarantine_rounds:
            return
        assert self.context is not None
        self.committed_observations += 1
        self.exposures *= self.decay
        self.positives *= self.decay
        self.positive_vector *= self.decay
        self.positive_weight *= self.decay
        for item_id in result.recommended_ids:
            cluster_id = self.context.cluster_assignments[item_id]
            self.exposures[cluster_id] += 1.0
        if result.clicked_item is not None:
            item_id = int(result.clicked_item)
            cluster_id = self.context.cluster_assignments[item_id]
            confidence = self._implicit_confidence(item_id)
            self.positives[cluster_id] += confidence
            self.positive_vector += confidence * self.context.embeddings[item_id]
            self.positive_weight += confidence
        self.profile = self._posterior_profile()

    def _posterior_profile(self) -> np.ndarray:
        assert self.context is not None
        posterior = (self.positives + self.alpha) / (
            self.exposures + self.alpha + self.beta
        )
        prior_rate = self.alpha / (self.alpha + self.beta)
        cluster_evidence = posterior - prior_rate
        cluster_vector = cluster_evidence @ self.context.cluster_centroids
        if self.positive_weight > 0.0:
            positive_vector = self.positive_vector / self.positive_weight
        else:
            positive_vector = np.zeros_like(self.positive_vector)
        relative_base = max(
            self.base_decay**self.committed_observations,
            self.base_floor,
        )
        combined = (
            self.base_strength * relative_base * self.context.base_profile
            + self.positive_strength * positive_vector
            + self.cluster_strength * cluster_vector
        )
        return _normalize(combined)


def make_policy(name: str, config: dict) -> ProfilePolicy:
    popularity_weight = float(config.get("popularity_prior_weight", 0.0))
    if name == "no_memory":
        return NoMemoryPolicy(popularity_weight)
    if name == "raw_memory":
        return RawMemoryPolicy(
            update_rate=float(config.get("raw_memory_update_rate", 0.3)),
            popularity_prior_weight=popularity_weight,
        )
    if name == "raw_no_early_write":
        return NoEarlyWritePolicy(
            update_rate=float(config.get("raw_memory_update_rate", 0.3)),
            popularity_prior_weight=popularity_weight,
        )
    if name == "raw_reset":
        return MemoryResetPolicy(
            update_rate=float(config.get("raw_memory_update_rate", 0.3)),
            popularity_prior_weight=popularity_weight,
        )
    if name == "sliding_window":
        return SlidingWindowPolicy(
            window_size=int(config.get("sliding_window_size", 3)),
            base_strength=float(config.get("recent_memory_base_strength", 2.0)),
            popularity_prior_weight=popularity_weight,
        )
    if name == "recency_weighted":
        return RecencyWeightedPolicy(
            decay=float(config.get("recency_decay", 0.7)),
            base_strength=float(config.get("recent_memory_base_strength", 2.0)),
            popularity_prior_weight=popularity_weight,
        )
    if name == "exposure_normalized":
        return ExposureNormalizedPolicy(
            base_strength=float(config.get("exposure_profile_base_strength", 2.0)),
            decay=float(config.get("exposure_profile_decay", 0.8)),
            popularity_prior_weight=popularity_weight,
        )
    if name == "linucb":
        return LinUCBPolicy(
            alpha=float(config.get("linucb_alpha", 0.5)),
            regularization=float(config.get("bandit_regularization", 1.0)),
            base_weight=float(config.get("bandit_base_weight", 0.5)),
            seed=int(config["seed"]),
            popularity_prior_weight=popularity_weight,
        )
    if name == "thompson":
        return ThompsonSamplingPolicy(
            sample_scale=float(config.get("thompson_scale", 0.25)),
            regularization=float(config.get("bandit_regularization", 1.0)),
            base_weight=float(config.get("bandit_base_weight", 0.5)),
            seed=int(config["seed"]),
            popularity_prior_weight=popularity_weight,
        )
    if name == "eam":
        return EAMPolicy(
            base_strength=float(config.get("eam_base_strength", 2.0)),
            alpha=float(config.get("eam_alpha", 1.0)),
            beta=float(config.get("eam_beta", 1.0)),
            decay=float(config.get("eam_decay", 0.7)),
            residual_strength=float(config.get("eam_residual_strength", 1.0)),
            cluster_strength=float(config.get("eam_cluster_strength", 0.5)),
            propensity_floor=float(config.get("eam_propensity_floor", 0.2)),
            reference_propensity=float(
                config.get("eam_reference_propensity", 0.5)
            ),
            exploration_bonus=float(config.get("eam_exploration_bonus", 0.0)),
            exploration_slots=int(config.get("eam_exploration_slots", 0)),
            popularity_prior_weight=popularity_weight,
        )
    if name == "eam_delayed":
        return DelayedCommitmentPolicy(
            base_strength=float(config.get("delayed_eam_base_strength", 1.5)),
            alpha=float(config.get("delayed_eam_alpha", 1.0)),
            beta=float(config.get("delayed_eam_beta", 1.0)),
            decay=float(config.get("delayed_eam_decay", 0.6)),
            base_decay=float(config.get("delayed_eam_base_decay", 0.94)),
            base_floor=float(config.get("delayed_eam_base_floor", 0.6)),
            quarantine_rounds=int(config.get("delayed_eam_quarantine_rounds", 2)),
            positive_strength=float(
                config.get("delayed_eam_positive_strength", 1.25)
            ),
            cluster_strength=float(
                config.get("delayed_eam_cluster_strength", 0.5)
            ),
            propensity_floor=float(
                config.get("delayed_eam_propensity_floor", 0.2)
            ),
            reference_propensity=float(
                config.get("delayed_eam_reference_propensity", 0.5)
            ),
            exploration_bonus=float(
                config.get("delayed_eam_exploration_bonus", 0.05)
            ),
            exploration_slots=int(
                config.get("delayed_eam_exploration_slots", 1)
            ),
            popularity_prior_weight=popularity_weight,
        )
    raise ValueError(f"Unknown policy: {name}")

