from __future__ import annotations

import json
import os
from typing import Protocol

import numpy as np

from .environment import Observation, StepResult
from .memory import NoMemoryPolicy, PolicyContext
from .metadata import ItemMetadata


SYSTEM_PROMPT = (
    "Follow the requested JSON schema exactly. Do not add "
    "markdown fences or explanatory text."
)

ERSM_MODES = {
    "ersm",
    "ersm_no_residual",
    "ersm_ips_only",
    "ersm_slate_only",
    "ersm_user",
    "ersm_consolidated",
    "ersm_invariant",
}
FACT_MODES = {"factmem", "factmem_ersm"}
MEM0_MODES = {"mem0", "mem0_ersm"}
SUPPORTED_MODES = {
    "none",
    "raw",
    "raw_recent",
    "raw_gate",
    "summary",
    "residual",
    "eam",
    *ERSM_MODES,
    *FACT_MODES,
    *MEM0_MODES,
}


class GenerationBackend(Protocol):
    def generate(self, prompt: str) -> str:
        ...


def generate_many(backend: GenerationBackend, prompts: list[str]) -> list[str]:
    if not prompts:
        return []
    batch = getattr(backend, "generate_batch", None)
    if batch is not None:
        return list(batch(prompts))
    return [backend.generate(prompt) for prompt in prompts]


def _chat_text(tokenizer, prompt: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        # Non-Qwen chat templates (e.g. Llama) do not accept enable_thinking.
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


class TransformersBackend:
    """Local Hugging Face chat backend loaded once and shared across policies."""

    def __init__(
        self,
        model_name: str,
        max_new_tokens: int = 128,
        device_map: str = "auto",
        batch_size: int = 16,
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map=device_map,
            low_cpu_mem_usage=True,
        )
        self.model.eval()

    def generate(self, prompt: str) -> str:
        text = _chat_text(self.tokenizer, prompt)
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        with self.torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        output_ids = generated[0, inputs["input_ids"].shape[1] :]
        return self.tokenizer.decode(output_ids, skip_special_tokens=True).strip()

    def generate_batch(self, prompts: list[str]) -> list[str]:
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        outputs: list[str] = []
        for start in range(0, len(prompts), self.batch_size):
            texts = [
                _chat_text(self.tokenizer, prompt)
                for prompt in prompts[start : start + self.batch_size]
            ]
            inputs = self.tokenizer(texts, return_tensors="pt", padding=True).to(
                self.model.device
            )
            with self.torch.inference_mode():
                generated = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            width = inputs["input_ids"].shape[1]
            outputs.extend(
                self.tokenizer.decode(row[width:], skip_special_tokens=True).strip()
                for row in generated
            )
        return outputs


class VLLMBackend:
    """Greedy vLLM backend; batches whole rounds of prompts in one call."""

    def __init__(
        self,
        model_name: str,
        max_new_tokens: int = 128,
        max_model_len: int = 6144,
        gpu_memory_utilization: float = 0.88,
    ):
        from vllm import LLM, SamplingParams

        self.model_name = model_name
        self.llm = LLM(
            model=model_name,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            enable_prefix_caching=True,
            seed=0,
        )
        self.tokenizer = self.llm.get_tokenizer()
        self.sampling = SamplingParams(temperature=0.0, max_tokens=max_new_tokens)

    def generate(self, prompt: str) -> str:
        return self.generate_batch([prompt])[0]

    def generate_batch(self, prompts: list[str]) -> list[str]:
        texts = [_chat_text(self.tokenizer, prompt) for prompt in prompts]
        outputs = self.llm.generate(texts, self.sampling, use_tqdm=False)
        return [output.outputs[0].text.strip() for output in outputs]


class _EmptyResponse(Exception):
    """An endpoint returned a completion without choices (e.g. an upstream error)."""


class OpenAIBackend:
    """Client for an OpenAI-compatible endpoint (``vllm serve`` or a hosted API).

    Batches are sent as concurrent requests. Thinking must be disabled in the
    server's chat template for Qwen3 models, or through ``extra_body`` for
    hosted models. With ``max_retries > 0``, rate limits, timeouts, and server
    errors are retried with backoff and then raised; authentication, billing,
    and key-limit errors (401/402/403) are raised at once; any other request
    the endpoint rejects (4xx) returns an empty response, which the policy
    records as a parse failure.
    """

    def __init__(
        self,
        model_name: str,
        base_url: str,
        max_new_tokens: int = 128,
        concurrency: int = 64,
        api_key: str = "EMPTY",
        extra_body: dict | None = None,
        max_retries: int = 0,
    ):
        import collections
        import threading

        from openai import OpenAI

        self.model_name = model_name
        self.client = OpenAI(base_url=base_url, api_key=api_key, timeout=600, max_retries=0)
        self.max_new_tokens = max_new_tokens
        self.concurrency = concurrency
        self.extra_body = extra_body or None
        self.max_retries = max_retries
        self.usage: collections.Counter = collections.Counter()
        self._lock = threading.Lock()

    def _create(self, prompt: str):
        return self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=self.max_new_tokens,
            seed=0,
            extra_body=self.extra_body,
        )

    def generate(self, prompt: str) -> str:
        import time

        import openai

        transient = (
            openai.RateLimitError,
            openai.APIConnectionError,
            openai.InternalServerError,
            _EmptyResponse,
        )
        for attempt in range(self.max_retries + 1):
            try:
                response = self._create(prompt)
                if not getattr(response, "choices", None):
                    raise _EmptyResponse(str(getattr(response, "model_extra", ""))[:200])
                break
            except transient:
                if attempt == self.max_retries:
                    raise
                with self._lock:
                    self.usage["retries"] += 1
                time.sleep(min(60, 2 ** attempt))
            except openai.APIStatusError as exc:
                if not self.max_retries or exc.status_code in (401, 402, 403):
                    raise
                with self._lock:
                    self.usage["rejected_requests"] += 1
                    self.usage[f"rejected:{exc.status_code}"] += 1
                return ""
        with self._lock:
            self.usage["requests"] += 1
            if response.usage is not None:
                self.usage["prompt_tokens"] += response.usage.prompt_tokens or 0
                self.usage["completion_tokens"] += response.usage.completion_tokens or 0
            provider = (getattr(response, "model_extra", None) or {}).get("provider")
            if provider:
                self.usage["provider:" + str(provider)] += 1
        return (response.choices[0].message.content or "").strip()

    def generate_batch(self, prompts: list[str]) -> list[str]:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(self.concurrency) as pool:
            return list(pool.map(self.generate, prompts))


class ChatGPTPlanBackend:
    """GPT models billed to a ChatGPT plan through a local ``chatgpt_plan_api`` client.

    The client module is imported from ``CHATGPT_PLAN_CLIENT_DIR`` (default
    ``~/.cursor/tools``); runs with an OpenAI API key should use ``OpenAIBackend``
    instead. The plan route samples at the model default (no temperature or token
    limit), so every response is cached by (model, effort, system, prompt):
    identical prompts get the same response, and a run interrupted by the
    plan quota replays from the cache when restarted. ``UsageLimit`` is
    propagated after the finished responses are saved; any other failed
    request returns an empty response, which the policy records as a parse
    failure.
    """

    def __init__(self, model_name: str, effort: str | None, cache_path: str, workers: int = 4):
        import collections
        import sys
        import threading
        from pathlib import Path

        client_dir = os.environ.get("CHATGPT_PLAN_CLIENT_DIR", Path.home() / ".cursor" / "tools")
        sys.path.insert(0, str(client_dir))
        import chatgpt_plan_api

        self.api = chatgpt_plan_api
        self.model_name = model_name
        self.effort = effort
        self.cache_path = Path(cache_path)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.workers = workers
        self.usage: collections.Counter = collections.Counter()
        self._lock = threading.Lock()
        self.cache: dict[str, str] = {}
        if self.cache_path.exists():
            for line in self.cache_path.read_text(encoding="utf-8").splitlines():
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                self.cache[row["id"]] = row["text"]

    def _key(self, prompt: str) -> str:
        import hashlib

        raw = json.dumps([self.model_name, self.effort, SYSTEM_PROMPT, prompt])
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _request(self, key: str, prompt: str, sink, stop) -> None:
        if stop.is_set():
            return
        try:
            row = self.api.chat_full(
                prompt, model=self.model_name, system=SYSTEM_PROMPT, effort=self.effort
            )
        except self.api.UsageLimit:
            stop.set()
            raise
        except self.api.PlanError:
            with self._lock:
                self.usage["failed_requests"] += 1
            return
        with self._lock:
            self.cache[key] = row["text"]
            sink.write(json.dumps({"id": key, **row}, ensure_ascii=False) + "\n")
            sink.flush()
            self.usage["requests"] += 1
            self.usage["input_tokens"] += row["input_tokens"]
            self.usage["output_tokens"] += row["output_tokens"]
            self.usage["reasoning_tokens"] += row["reasoning_tokens"]

    def generate(self, prompt: str) -> str:
        return self.generate_batch([prompt])[0]

    def generate_batch(self, prompts: list[str]) -> list[str]:
        import threading
        from concurrent.futures import ThreadPoolExecutor

        keys = [self._key(prompt) for prompt in prompts]
        todo = {key: prompt for key, prompt in zip(keys, prompts) if key not in self.cache}
        self.usage["cache_hits"] += len(prompts) - len(todo)
        if todo:
            stop = threading.Event()
            with self.cache_path.open("a", encoding="utf-8") as sink, ThreadPoolExecutor(
                self.workers
            ) as pool:
                futures = [
                    pool.submit(self._request, key, prompt, sink, stop)
                    for key, prompt in todo.items()
                ]
                errors = [f.exception() for f in futures if f.exception() is not None]
            if errors:
                raise errors[0]
        return [self.cache.get(key, "") for key in keys]


def make_backend(config: dict) -> GenerationBackend:
    max_new_tokens = int(config.get("max_new_tokens", 96))
    kind = str(config.get("backend", "transformers"))
    if kind == "chatgpt_plan":
        return ChatGPTPlanBackend(
            config["served_model_name"],
            effort=config.get("plan_effort"),
            cache_path=config["plan_cache"],
            workers=int(config.get("io_workers", 4)),
        )
    if kind == "openai":
        key_env = config.get("openai_api_key_env")
        return OpenAIBackend(
            config["served_model_name"],
            config["openai_base_url"],
            max_new_tokens=max_new_tokens,
            concurrency=int(config.get("io_workers", 64)),
            api_key=os.environ[key_env] if key_env else "EMPTY",
            extra_body=config.get("openai_extra_body"),
            max_retries=int(config.get("openai_max_retries", 0)),
        )
    if kind == "vllm":
        return VLLMBackend(
            config["model_name"],
            max_new_tokens=max_new_tokens,
            max_model_len=int(config.get("max_model_len", 6144)),
            gpu_memory_utilization=float(
                config.get("gpu_memory_utilization", 0.88)
            ),
        )
    return TransformersBackend(
        config["model_name"],
        max_new_tokens=max_new_tokens,
        batch_size=int(config.get("batch_size", 16)),
    )


def parse_selected_ids(
    response: str,
    candidate_ids: np.ndarray,
    top_k: int,
) -> np.ndarray:
    payload = json.loads(response)
    selected = payload.get("selected_ids")
    if not isinstance(selected, list) or len(selected) != top_k:
        raise ValueError("selected_ids must be a list with exactly top_k items")
    normalized = []
    for value in selected:
        if isinstance(value, bool):
            raise ValueError("selected_ids must contain integer-like values")
        if isinstance(value, int):
            normalized.append(value)
        elif isinstance(value, str) and value.isdigit():
            normalized.append(int(value))
        else:
            raise ValueError("selected_ids must contain integer-like values")
    if len(set(normalized)) != len(normalized):
        raise ValueError("selected_ids contains duplicates")
    candidate_set = set(map(int, candidate_ids))
    if not set(normalized).issubset(candidate_set):
        raise ValueError("selected_ids contains an item outside the candidate set")
    return np.asarray(normalized, dtype=np.int64)


def normalize_title(title: str) -> str:
    """Lower-cased title without a trailing year, with ', The' moved to front."""
    import re

    text = re.sub(r"\s*\(\d{4}\)\s*$", "", title).strip()
    match = re.match(r"^(.*), (The|A|An)$", text)
    if match:
        text = f"{match.group(2)} {match.group(1)}"
    return text.lower()


def parse_facts(response: str, max_facts: int, max_words: int) -> list[str]:
    payload = json.loads(response)
    facts = payload.get("facts")
    if not isinstance(facts, list):
        raise ValueError("facts must be a list")
    output = []
    for fact in facts:
        if not isinstance(fact, str) or not fact.strip():
            raise ValueError("facts must be non-empty strings")
        output.append(" ".join(fact.split()[:max_words]))
    return output[:max_facts]


class StructuredLLMPolicy:
    """Auditable LLM policy with strict JSON parsing and deterministic fallback.

    Each round is split into prompt construction and response application so a
    runner can batch prompts across users; ``recommend``/``observe`` remain the
    one-call-at-a-time entry points.
    """

    name = "structured_llm"

    def __init__(
        self,
        backend: GenerationBackend,
        metadata: dict[int, ItemMetadata],
        memory_mode: str = "raw",
        history_limit: int = 12,
        eam_decay: float = 0.8,
        eam_propensity_floor: float = 0.2,
        eam_reference_propensity: float = 0.5,
        ersm_top_genres: int = 6,
        ersm_commit_threshold: int = 2,
        recency_window: int = 3,
        candidate_observability: str = "full",
        ersm_user_tolerance: float = 2.0,
        ersm_user_prior: float = 3.0,
        fact_limit: int = 6,
        fact_words: int = 12,
        memory_store=None,
        retrieve_k: int = 5,
    ):
        if candidate_observability not in {"full", "title_only"}:
            raise ValueError(
                "candidate_observability must be 'full' or 'title_only'"
            )
        if memory_mode not in SUPPORTED_MODES:
            raise ValueError("Unsupported memory_mode")
        if memory_mode in MEM0_MODES and memory_store is None:
            raise ValueError("mem0 modes require a memory_store")
        self.memory_store = memory_store
        self.retrieve_k = retrieve_k
        self.last_retrieved: list[str] = []
        self.audit_titles: tuple[str, ...] = ()
        self.audit_genre: str | None = None
        self.backend = backend
        self.metadata = metadata
        self.memory_mode = memory_mode
        self.history_limit = history_limit
        self.eam_decay = eam_decay
        self.eam_propensity_floor = eam_propensity_floor
        self.eam_reference_propensity = eam_reference_propensity
        self.ersm_top_genres = ersm_top_genres
        self.ersm_commit_threshold = ersm_commit_threshold
        self.recency_window = recency_window
        self.candidate_observability = candidate_observability
        self.ersm_user_tolerance = ersm_user_tolerance
        self.ersm_user_prior = ersm_user_prior
        self.fact_limit = fact_limit
        self.fact_words = fact_words
        # ERSM write-path components: slate-level residualization and a
        # propensity weight that is either catalogue-level (IPS on item
        # popularity) or user-level (the user's own genre exposure so far).
        self.ersm_residualize = memory_mode in {
            "ersm",
            "ersm_slate_only",
            "ersm_user",
            "ersm_consolidated",
            "ersm_invariant",
            "factmem_ersm",
            "mem0_ersm",
            "raw_gate",
        }
        self.ersm_ips = memory_mode in {
            "ersm",
            "ersm_ips_only",
            "ersm_consolidated",
            "ersm_invariant",
            "factmem_ersm",
            "mem0_ersm",
            "raw_gate",
        }
        self.ersm_user_propensity = memory_mode == "ersm_user"
        self.ersm_consolidate = memory_mode == "ersm_consolidated"
        self.ersm_invariant = memory_mode == "ersm_invariant"
        self.genre_prevalence = self._catalogue_genre_prevalence(metadata)
        self.base_genres: set[str] = set()
        self.context: PolicyContext | None = None
        self.fallback = NoMemoryPolicy()
        self.memory_text = ""
        self.initial_memory_text = ""
        self.raw_events: list[str] = []
        self.eam_records: list[dict] = []
        self.facts: list[str] = []
        self.genre_scores: dict[str, float] = {}
        self.genre_confirmations: dict[str, int] = {}
        self.genre_exposure: dict[str, float] = {}
        self.exposed_items = 0
        self.trace: list[dict] = []
        self.parse_failures = 0

    @staticmethod
    def _catalogue_genre_prevalence(
        metadata: dict[int, ItemMetadata],
    ) -> dict[str, float]:
        counts: dict[str, float] = {}
        for item in metadata.values():
            for genre in set(item.attributes):
                counts[genre] = counts.get(genre, 0.0) + 1.0
        total = max(len(metadata), 1)
        return {genre: count / total for genre, count in counts.items()}

    def reset(self, context: PolicyContext) -> None:
        self.context = context
        self.fallback.reset(context)
        anchor_ids = self._history_anchor_ids(context)
        history = [
            self.metadata[int(item_id)].compact_text() for item_id in anchor_ids
        ]
        self.initial_memory_text = (
            "Initial representative liked items: " + "; ".join(history)
            if history
            else ""
        )
        self.memory_text = self.initial_memory_text
        self.raw_events = []
        self.eam_records = []
        self.facts = []
        self.genre_scores = self._initial_genre_scores(anchor_ids)
        # The base profile is built from pre-session history, identical across
        # treatment arms; capturing it lets ersm_invariant anchor membership to
        # an arm-invariant core.
        self.base_genres = set(self.genre_scores)
        # History genres are pre-confirmed so the arm-invariant base profile is
        # always eligible; only session-introduced genres must be confirmed.
        self.genre_confirmations = {
            genre: self.ersm_commit_threshold for genre in self.genre_scores
        }
        self.genre_exposure = {}
        self.exposed_items = 0
        self.trace = []
        self.parse_failures = 0

    def recommendation_prompt(self, observation: Observation, top_k: int) -> str:
        return self._recommendation_prompt(observation, top_k)

    def apply_recommendation(
        self,
        observation: Observation,
        top_k: int,
        prompt: str,
        response: str,
    ) -> np.ndarray:
        try:
            selected = parse_selected_ids(
                response, observation.candidate_ids, top_k
            )
            parse_error = None
        except (json.JSONDecodeError, ValueError, TypeError, AttributeError) as exc:
            selected = self.fallback.recommend(observation, top_k)
            parse_error = str(exc)
            self.parse_failures += 1
        self.trace.append(
            {
                "event": "recommend",
                "round_idx": int(observation.round_idx),
                "prompt": prompt,
                "response": response,
                "selected_ids": selected.astype(int).tolist(),
                "parse_error": parse_error,
            }
        )
        return selected

    def recommend(self, observation: Observation, top_k: int) -> np.ndarray:
        prompt = self.recommendation_prompt(observation, top_k)
        response = self.backend.generate(prompt)
        return self.apply_recommendation(observation, top_k, prompt, response)

    def memory_prompt(
        self, observation: Observation, result: StepResult
    ) -> str | None:
        """Prompt for an LLM-written memory update, or None if none is needed."""
        if result.clicked_item is None:
            return None
        clicked_id = int(result.clicked_item)
        clicked = self._candidate_text(clicked_id)
        shown = [
            self._candidate_text(int(item_id))
            for item_id in result.recommended_ids
        ]
        if self.memory_mode == "summary":
            return self._summary_prompt(clicked, shown)
        if self.memory_mode in FACT_MODES:
            if self.memory_mode == "factmem_ersm" and not self._ersm_admits(
                clicked_id, result
            ):
                return None
            return self._fact_prompt(clicked, shown)
        return None

    def observe(
        self,
        observation: Observation,
        result: StepResult,
        memory_response: str | None = None,
    ) -> None:
        if self.memory_mode == "none" or result.clicked_item is None:
            return
        clicked_id = int(result.clicked_item)
        clicked = self._candidate_text(clicked_id)
        shown = [
            self._candidate_text(int(item_id))
            for item_id in result.recommended_ids
        ]
        memory_record = None
        if self.memory_mode == "raw_gate":
            # Item-level log behind the ERSM write gate: identities are kept,
            # only exposure-driven clicks are withheld from memory.
            admitted = self._ersm_admits(clicked_id, result)
            memory_record = self._update_genre_scores(clicked_id, result)
            memory_record["admitted"] = admitted
            update = self.memory_text
            if admitted:
                event = f"Clicked {clicked} after seeing: {'; '.join(shown)}"
                self.raw_events.append(event)
                update = "\n".join(value for value in (update, event) if value)
        elif self.memory_mode in {"raw", "raw_recent"}:
            event = f"Clicked {clicked} after seeing: {'; '.join(shown)}"
            self.raw_events.append(event)
            if self.memory_mode == "raw_recent":
                # Recency-truncation defence: only the most recent interactions
                # are retained, so an early-exposure attack ages out of memory.
                kept = self.raw_events[-self.recency_window :]
                update = "\n".join(
                    value
                    for value in ([self.initial_memory_text] + kept)
                    if value
                )
            else:
                update = "\n".join(
                    value for value in (self.memory_text, event) if value
                )
        elif self.memory_mode in {"residual", "eam"}:
            alternatives = [
                text
                for item_id, text in zip(result.recommended_ids, shown)
                if int(item_id) != clicked_id
            ]
            if self.memory_mode == "residual":
                event = f"Preferred {clicked} over: {'; '.join(alternatives)}"
                update = "\n".join(
                    value for value in (self.memory_text, event) if value
                )
            else:
                memory_record = self._append_eam_record(
                    clicked_id,
                    clicked,
                    alternatives,
                    observation.round_idx,
                )
                update = self._render_eam_memory(observation.round_idx)
        elif self.memory_mode in ERSM_MODES:
            memory_record = self._update_genre_scores(clicked_id, result)
            update = self._render_ersm_memory()
        elif self.memory_mode in FACT_MODES:
            prompt = self.memory_prompt(observation, result)
            if self.memory_mode == "factmem_ersm":
                memory_record = self._update_genre_scores(clicked_id, result)
                memory_record["admitted"] = prompt is not None
            if prompt is not None:
                if memory_response is None:
                    memory_response = self.backend.generate(prompt)
                self._apply_fact_update(
                    clicked, prompt, memory_response, observation.round_idx
                )
            update = self._render_fact_memory()
        elif self.memory_mode in MEM0_MODES:
            admitted = True
            if self.memory_mode == "mem0_ersm":
                admitted = self._ersm_admits(clicked_id, result)
                memory_record = self._update_genre_scores(clicked_id, result)
                memory_record["admitted"] = admitted
            if admitted:
                self.memory_store.add(
                    f"I clicked on {clicked} from these recommendations: "
                    f"{'; '.join(shown)}."
                )
            update = self.memory_text
        else:
            prompt = self._summary_prompt(clicked, shown)
            if memory_response is None:
                memory_response = self.backend.generate(prompt)
            update = self._apply_summary(
                clicked, prompt, memory_response, observation.round_idx
            )
        self.memory_text = update
        self.trace.append(
            {
                "event": "memory_write",
                "round_idx": int(observation.round_idx),
                "memory_mode": self.memory_mode,
                "memory": update,
                "memory_record": memory_record,
            }
        )

    def _history_anchor_ids(self, context: PolicyContext) -> np.ndarray:
        available = np.asarray(
            [
                int(item_id)
                for item_id in context.history_items
                if int(item_id) in self.metadata
            ],
            dtype=np.int64,
        )
        if len(available) <= self.history_limit:
            return available
        scores = context.embeddings[available] @ context.base_profile
        order = np.argsort(-scores, kind="stable")[: self.history_limit]
        return available[order]

    def _catalogue_confidence(self, item_id: int) -> tuple[float, float]:
        if self.context is None:
            raise RuntimeError("reset must be called before observe")
        popularity = float(self.context.popularity[item_id])
        propensity = self.eam_propensity_floor + (
            1.0 - self.eam_propensity_floor
        ) * popularity
        confidence = min(
            1.0, self.eam_reference_propensity / max(propensity, 1e-8)
        )
        return confidence, popularity

    def _append_eam_record(
        self,
        item_id: int,
        clicked: str,
        alternatives: list[str],
        round_idx: int,
    ) -> dict:
        confidence, popularity = self._catalogue_confidence(item_id)
        record = {
            "round_idx": int(round_idx),
            "source": "implicit_click",
            "explicit": False,
            "clicked": clicked,
            "alternatives": alternatives,
            "exposure_prevalence": popularity,
            "confidence": confidence,
        }
        self.eam_records.append(record)
        self.eam_records = self.eam_records[-self.history_limit :]
        return record

    def _render_eam_memory(self, current_round: int) -> str:
        lines = [self.initial_memory_text] if self.initial_memory_text else []
        if self.eam_records:
            lines.append(
                "Exposure-aware implicit evidence; shown alternatives are "
                "context, not dislikes:"
            )
        for record in self.eam_records:
            age = max(int(current_round) - int(record["round_idx"]), 0)
            weight = float(record["confidence"]) * self.eam_decay**age
            alternatives = "; ".join(record["alternatives"])
            if not alternatives:
                alternatives = "no other selected items"
            lines.append(
                f"- round={record['round_idx']}; source=implicit_click; "
                f"weight={weight:.3f}; chose {record['clicked']} relative to "
                f"shown {alternatives}"
            )
        return "\n".join(lines)

    def _item_genres(self, item_id: int) -> tuple[str, ...]:
        metadata = self.metadata.get(int(item_id))
        return metadata.attributes if metadata else ()

    def _initial_genre_scores(self, anchor_ids: np.ndarray) -> dict[str, float]:
        scores: dict[str, float] = {}
        for item_id in anchor_ids:
            for genre in self._item_genres(int(item_id)):
                scores[genre] = scores.get(genre, 0.0) + 1.0
        return scores

    def _user_exposure_weight(self, genre: str) -> float:
        """Down-weight a genre the user has been shown far above its base rate.

        The propensity is the user's own smoothed exposure share of ``genre``
        across displayed slates; evidence is scaled by ``tolerance * base_rate /
        propensity`` once that share exceeds ``tolerance`` times the catalogue
        base rate, so a manipulated, user-specific exposure surge is discounted
        even when the promoted items are unpopular in the catalogue.
        """
        base_rate = self.genre_prevalence.get(genre, 0.0)
        propensity = (
            self.genre_exposure.get(genre, 0.0) + self.ersm_user_prior * base_rate
        ) / (self.exposed_items + self.ersm_user_prior)
        if propensity <= 0.0:
            return 1.0
        return min(1.0, self.ersm_user_tolerance * base_rate / propensity)

    def _ersm_evidence(
        self, clicked_id: int, result: StepResult
    ) -> tuple[dict[str, float], dict]:
        """Per-genre evidence increment for one click under this mode's write path."""
        clicked_genres = set(self._item_genres(clicked_id))
        slate_ids = [int(item_id) for item_id in result.recommended_ids]
        if self.ersm_ips:
            confidence, popularity = self._catalogue_confidence(clicked_id)
        else:
            confidence = 1.0
            popularity = float(self.context.popularity[clicked_id])
        if self.ersm_residualize:
            genre_universe = set(clicked_genres)
            for item_id in slate_ids:
                genre_universe.update(self._item_genres(item_id))
            raw = {}
            for genre in genre_universe:
                clicked_indicator = 1.0 if genre in clicked_genres else 0.0
                slate_rate = (
                    float(
                        np.mean(
                            [
                                1.0 if genre in self._item_genres(item_id) else 0.0
                                for item_id in slate_ids
                            ]
                        )
                    )
                    if slate_ids
                    else 0.0
                )
                raw[genre] = clicked_indicator - slate_rate
        else:
            raw = {genre: 1.0 for genre in clicked_genres}
        weights = {
            genre: confidence
            * (self._user_exposure_weight(genre) if self.ersm_user_propensity else 1.0)
            for genre in raw
        }
        evidence = {genre: weights[genre] * raw[genre] for genre in raw}
        clicked_weights = [weights[g] for g in clicked_genres if g in weights]
        record = {
            "round_idx": int(result.round_idx),
            "source": "implicit_click",
            "explicit": False,
            "residualized": self.ersm_residualize,
            "catalogue_ips": self.ersm_ips,
            "user_propensity": self.ersm_user_propensity,
            "clicked_genres": sorted(clicked_genres),
            "exposure_prevalence": popularity,
            "confidence": min(clicked_weights) if clicked_weights else confidence,
            "evidence": {g: round(v, 4) for g, v in sorted(evidence.items())},
        }
        return evidence, record

    def _record_exposure(self, result: StepResult) -> None:
        for item_id in result.recommended_ids:
            for genre in set(self._item_genres(int(item_id))):
                self.genre_exposure[genre] = self.genre_exposure.get(genre, 0.0) + 1.0
            self.exposed_items += 1

    def _update_genre_scores(self, clicked_id: int, result: StepResult) -> dict:
        """Deterministic exposure-residualized genre evidence (no item names)."""
        if self.context is None:
            raise RuntimeError("reset must be called before observe")
        for genre in list(self.genre_scores):
            self.genre_scores[genre] *= self.eam_decay
        self._record_exposure(result)
        evidence, record = self._ersm_evidence(clicked_id, result)
        for genre, value in evidence.items():
            if value != 0.0:
                self.genre_scores[genre] = self.genre_scores.get(genre, 0.0) + value
            if self.ersm_residualize and value > 0.0:
                self.genre_confirmations[genre] = min(
                    self.genre_confirmations.get(genre, 0) + 1,
                    self.ersm_commit_threshold,
                )
        return record

    def _ersm_admits(self, clicked_id: int, result: StepResult) -> bool:
        """ERSM write gate: admit a click only if it carries positive
        exposure-corrected evidence for at least one of its genres."""
        evidence, _ = self._ersm_evidence(clicked_id, result)
        clicked_genres = set(self._item_genres(clicked_id))
        return any(evidence.get(genre, 0.0) > 0.0 for genre in clicked_genres)

    def _render_ersm_memory(self) -> str:
        positive = {
            genre: score
            for genre, score in self.genre_scores.items()
            if score > 0.0
            and (
                not self.ersm_consolidate
                or self.genre_confirmations.get(genre, 0)
                >= self.ersm_commit_threshold
            )
        }
        if self.ersm_invariant:
            # Anchor membership to the arm-invariant base profile: history
            # genres are always eligible, and a session-introduced genre is
            # admitted only if its evidence exceeds the strongest base genre,
            # i.e. a shift strong enough to be arm-robust rather than noise.
            base_max = max(
                (
                    self.genre_scores.get(genre, 0.0)
                    for genre in self.base_genres
                ),
                default=0.0,
            )
            positive = {
                genre: score
                for genre, score in positive.items()
                if genre in self.base_genres or score > base_max
            }
        if not positive:
            return ""
        ranked = sorted(positive, key=lambda g: (-positive[g], g))[
            : self.ersm_top_genres
        ]
        if self.ersm_invariant:
            # Drop the score-implied priority channel: render an unordered set
            # in a canonical (alphabetical) order so arm-specific rank flips
            # among admitted genres cannot transmit early-exposure signal.
            return (
                "Exposure-corrected stable preference (genres, unordered set): "
                + ", ".join(sorted(ranked))
            )
        return (
            "Exposure-corrected stable preference (genres): "
            + ", ".join(ranked)
        )

    def _fact_prompt(self, clicked: str, shown: list[str]) -> str:
        existing = "\n".join(f"- {fact}" for fact in self.facts) or "none"
        return (
            "You maintain a recommender agent's long-term memory as short "
            "facts about the user's preferences. Given the existing facts and "
            "a new interaction, return the full updated fact list: add new "
            "facts, merge or update overlapping ones, and delete contradicted "
            f"ones. Keep at most {self.fact_limit} facts of at most "
            f"{self.fact_words} words each.\n"
            f"Existing facts:\n{existing}\n"
            f"New interaction: the user clicked {clicked} after being shown: "
            f"{'; '.join(shown)}\n"
            'Return JSON only: {"facts": ["fact", ...]}'
        )

    def _apply_fact_update(
        self, clicked: str, prompt: str, response: str, round_idx: int
    ) -> None:
        try:
            self.facts = parse_facts(response, self.fact_limit, self.fact_words)
            parse_error = None
        except (json.JSONDecodeError, ValueError, TypeError, AttributeError) as exc:
            fallback = " ".join(f"User clicked {clicked}.".split()[: self.fact_words])
            self.facts = (self.facts + [fallback])[-self.fact_limit :]
            parse_error = str(exc)
            self.parse_failures += 1
        self.trace.append(
            {
                "event": "memory_facts",
                "round_idx": int(round_idx),
                "prompt": prompt,
                "response": response,
                "parse_error": parse_error,
            }
        )

    def _render_fact_memory(self) -> str:
        lines = [self.initial_memory_text] if self.initial_memory_text else []
        if self.facts:
            lines.append("Stored preference facts:")
            lines.extend(f"- {fact}" for fact in self.facts)
        return "\n".join(lines)

    def set_audit(self, cluster_ids) -> None:
        """Register the promoted family so stored memories can be audited."""
        titles = []
        genre_counts: dict[str, int] = {}
        for item_id in cluster_ids:
            item = self.metadata.get(int(item_id))
            if item is None:
                continue
            title = normalize_title(item.title)
            if len(title) >= 4:
                titles.append(title)
            for genre in item.attributes:
                genre_counts[genre] = genre_counts.get(genre, 0) + 1
        self.audit_titles = tuple(titles)
        self.audit_genre = (
            max(sorted(genre_counts), key=genre_counts.get) if genre_counts else None
        )

    def memory_strings(self) -> list[str] | None:
        if self.memory_mode in MEM0_MODES:
            return self.memory_store.all()
        if self.memory_mode in FACT_MODES:
            return list(self.facts)
        if self.memory_mode in {"raw", "raw_recent", "raw_gate"}:
            return list(self.raw_events)
        return None

    def _mentions(self, text: str) -> tuple[bool, bool]:
        lowered = text.lower()
        title_hit = any(title in lowered for title in self.audit_titles)
        genre_hit = bool(self.audit_genre) and self.audit_genre.lower() in lowered
        return title_hit, genre_hit

    def audit(self) -> dict:
        """Share of stored (and last retrieved) memories naming the promoted
        family, by title or by its dominant attribute."""
        stored = self.memory_strings()
        if stored is None or not (self.audit_titles or self.audit_genre):
            return {}
        hits = [self._mentions(text) for text in stored]
        out = {
            "mem_items": len(stored),
            "mem_title_share": float(np.mean([h[0] for h in hits])) if hits else 0.0,
            "mem_genre_share": float(np.mean([h[1] for h in hits])) if hits else 0.0,
        }
        if self.memory_mode in MEM0_MODES:
            retrieved = [self._mentions(text) for text in self.last_retrieved]
            out["ret_items"] = len(retrieved)
            out["ret_title_share"] = (
                float(np.mean([h[0] for h in retrieved])) if retrieved else 0.0
            )
            out["ret_any_share"] = (
                float(np.mean([h[0] or h[1] for h in retrieved])) if retrieved else 0.0
            )
        return out

    def _item_text(self, item_id: int) -> str:
        metadata = self.metadata.get(item_id)
        return metadata.compact_text() if metadata else f"item {item_id}"

    def _candidate_text(self, item_id: int) -> str:
        """What the LLM sees of an item in candidates and the raw event log.

        Under ``title_only`` observability the genre tags are withheld, so the
        agent cannot read an item's category off the prompt and must instead
        infer the user's preference from accumulated feedback (memory). ERSM's
        structured genre profile is built from catalogue attributes and is
        unaffected; this isolates the value of preference memory.
        """
        metadata = self.metadata.get(item_id)
        if metadata is None:
            return f"item {item_id}"
        if self.candidate_observability == "title_only":
            return metadata.title
        return metadata.compact_text()

    def _recommendation_prompt(
        self, observation: Observation, top_k: int
    ) -> str:
        candidates = "\n".join(
            f"- {int(item_id)}: {self._candidate_text(int(item_id))}"
            for item_id in observation.candidate_ids
        )
        if self.memory_mode == "eam":
            memory = self._render_eam_memory(observation.round_idx)
            memory_instruction = (
                "Use evidence in proportion to its weight. Exposure alone and "
                "shown-only alternatives are not preference signals.\n"
            )
        elif self.memory_mode in ERSM_MODES:
            memory = self._render_ersm_memory()
            memory_instruction = (
                "Recommend candidates whose genres match the stable "
                "preference. Judge each candidate on its own content; item "
                "popularity is not a preference signal.\n"
            )
        elif self.memory_mode in FACT_MODES:
            memory = self._render_fact_memory()
            memory_instruction = ""
        elif self.memory_mode in MEM0_MODES:
            query = "User preferences relevant to: " + "; ".join(
                self._candidate_text(int(item_id))
                for item_id in observation.candidate_ids
            )
            self.last_retrieved = self.memory_store.search(query, self.retrieve_k)
            lines = [self.initial_memory_text] if self.initial_memory_text else []
            if self.last_retrieved:
                lines.append("Retrieved memories:")
                lines.extend(f"- {text}" for text in self.last_retrieved)
            memory = "\n".join(lines)
            memory_instruction = ""
        else:
            memory = self.memory_text
            memory_instruction = ""
        memory = memory or "No persistent preference memory."
        return (
            "You are a recommender. Select exactly "
            f"{top_k} candidate IDs for the user.\n"
            f"{memory_instruction}Persistent memory: {memory}\n"
            f"Candidates:\n{candidates}\n"
            'Return JSON only: {"selected_ids": [id, ...]}'
        )

    def _summary_prompt(self, clicked: str, shown: list[str]) -> str:
        return (
            "Update the persistent preference memory in at most 25 words using "
            "the new implicit feedback. Do not treat exposure alone as "
            "preference.\n"
            f"Old memory: {self.memory_text or 'none'}\n"
            f"Clicked: {clicked}\nShown: {'; '.join(shown)}\n"
            'Return JSON only: {"memory": "one concise sentence"}'
        )

    def _apply_summary(
        self, clicked: str, prompt: str, response: str, round_idx: int
    ) -> str:
        try:
            payload = json.loads(response)
            memory = payload["memory"]
            if not isinstance(memory, str) or not memory.strip():
                raise ValueError("memory must be a non-empty string")
            if len(memory.split()) > 25:
                raise ValueError("memory must contain at most 25 words")
            output = memory.strip()
            parse_error = None
        except (json.JSONDecodeError, KeyError, ValueError, TypeError) as exc:
            fallback = f"Clicked {clicked}."
            output = " ".join(fallback.split()[:25])
            parse_error = str(exc)
            self.parse_failures += 1
        self.trace.append(
            {
                "event": "memory_summary",
                "round_idx": int(round_idx),
                "prompt": prompt,
                "response": response,
                "parse_error": parse_error,
            }
        )
        return output
