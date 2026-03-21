"""
Dedicated Gemini Flash client for semantic scene segmentation only.

Supports per-key hourly throttling and API-key rotation without affecting
other LLM/VLM steps in the pipeline.
"""

from __future__ import annotations

import logging
import math
import re
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, List, Optional

try:
    from google import genai
    from google.genai import types
except Exception:  # pragma: no cover - optional dependency in local envs
    genai = None
    types = None


logger = logging.getLogger(__name__)


@dataclass
class _KeyState:
    api_key: str
    client: Any
    call_times: Deque[float] = field(default_factory=deque)
    cooldown_until: float = 0.0
    disabled: bool = False


class SceneGeminiClient:
    """Gemini client with scene-only rotation and conservative throttling."""

    WINDOW_SEC = 3600.0

    def __init__(
        self,
        api_keys: List[str],
        model: str,
        max_calls_per_hour: int = 15,
        timeout_ms: int = 45000,
    ):
        if not genai or not types:
            raise RuntimeError("google.genai is not installed for Gemini scene splitting.")

        self.model = model
        self.max_calls_per_hour = max(1, int(max_calls_per_hour))
        self.timeout_ms = max(1000, int(timeout_ms))
        self._states: List[_KeyState] = []

        for api_key in api_keys:
            try:
                client = genai.Client(
                    api_key=api_key,
                    http_options=types.HttpOptions(timeout=self.timeout_ms),
                )
                self._states.append(_KeyState(api_key=api_key, client=client))
            except Exception as exc:
                logger.warning(
                    "Skipping Gemini scene key %s during init: %s",
                    self._mask_key(api_key),
                    exc,
                )

        if not self._states:
            raise RuntimeError(
                "No Gemini scene API keys are configured. Set GEMINI_SCENE_API_KEYS "
                "or GEMINI_SCENE_API_KEY_*."
            )

    def generate_text(
        self,
        prompt: str,
        system_prompt: str = "",
        temperature: float = 0.4,
        thinking_level: str = "HIGH",
        max_output_tokens: Optional[int] = None,
    ) -> str:
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"System: {system_prompt}\nUser: {prompt}"

        max_attempts = max(len(self._states), 3)
        last_error: Exception | None = None

        for _attempt in range(max_attempts):
            state = self._acquire_state()
            now = time.time()
            state.call_times.append(now)

            try:
                config_kwargs = {
                    "temperature": temperature,
                }
                if max_output_tokens and int(max_output_tokens) > 0:
                    config_kwargs["max_output_tokens"] = int(max_output_tokens)

                response = state.client.models.generate_content(
                    model=self.model,
                    contents=full_prompt,
                    config=types.GenerateContentConfig(**config_kwargs),
                )
                response_text = (getattr(response, "text", "") or "").strip()
                if not response_text:
                    raise RuntimeError("Gemini scene splitter returned empty text.")

                state.cooldown_until = 0.0
                return response_text
            except Exception as exc:
                last_error = exc
                self._mark_failure(state, exc)
                if all(s.disabled for s in self._states):
                    raise RuntimeError("All Gemini scene keys are disabled.") from exc

        raise RuntimeError(
            f"Gemini scene splitter failed after {max_attempts} attempts."
        ) from last_error

    def _acquire_state(self) -> _KeyState:
        while True:
            now = time.time()
            candidates: List[tuple[int, int, _KeyState]] = []
            wait_until = math.inf

            for state in self._states:
                if state.disabled:
                    continue

                self._prune_calls(state, now)
                next_ready = self._next_ready_time(state, now)
                wait_until = min(wait_until, next_ready)

                if next_ready <= now:
                    candidates.append(
                        (len(state.call_times), int(state.cooldown_until > now), state)
                    )

            if candidates:
                candidates.sort(key=lambda item: (item[0], item[1]))
                return candidates[0][2]

            if not math.isfinite(wait_until):
                raise RuntimeError("No active Gemini scene keys are available.")

            sleep_for = max(1.0, wait_until - now)
            logger.info(
                "Gemini scene keys are throttled. Sleeping %.1fs until a key is reusable.",
                sleep_for,
            )
            time.sleep(sleep_for)

    def _next_ready_time(self, state: _KeyState, now: float) -> float:
        if state.cooldown_until > now:
            return state.cooldown_until
        if len(state.call_times) < self.max_calls_per_hour:
            return now
        return state.call_times[0] + self.WINDOW_SEC

    def _prune_calls(self, state: _KeyState, now: float) -> None:
        while state.call_times and now - state.call_times[0] >= self.WINDOW_SEC:
            state.call_times.popleft()

    def _mark_failure(self, state: _KeyState, exc: Exception) -> None:
        message = str(exc)
        retry_after = self._parse_retry_after_seconds(message)
        lower = message.lower()

        if "api key not valid" in lower or "permission" in lower or "forbidden" in lower:
            state.disabled = True
            logger.warning(
                "Disabling Gemini scene key %s after auth/permission failure: %s",
                self._mask_key(state.api_key),
                message,
            )
            return

        if "429" in lower or "resource_exhausted" in lower or "quota" in lower:
            cooldown = max(retry_after or 300, 60)
        elif "timed out" in lower or "timeout" in lower:
            cooldown = max(retry_after or 10, 5)
        else:
            cooldown = max(retry_after or 15, 5)

        state.cooldown_until = max(state.cooldown_until, time.time() + cooldown)
        logger.warning(
            "Gemini scene key %s failed; cooling down for %ss: %s",
            self._mask_key(state.api_key),
            int(cooldown),
            message,
        )

    @staticmethod
    def _parse_retry_after_seconds(message: str) -> int | None:
        patterns = [
            (r"retry[^0-9]*(\d+)\s*seconds?", 1),
            (r"wait[^0-9]*(\d+)\s*seconds?", 1),
            (r"after[^0-9]*(\d+)\s*seconds?", 1),
            (r"retry[^0-9]*(\d+)\s*minutes?", 60),
            (r"wait[^0-9]*(\d+)\s*minutes?", 60),
        ]
        for pattern, multiplier in patterns:
            match = re.search(pattern, message, re.IGNORECASE)
            if match:
                return int(match.group(1)) * multiplier
        return None

    @staticmethod
    def _mask_key(api_key: str) -> str:
        if len(api_key) <= 8:
            return "***"
        return f"{api_key[:4]}...{api_key[-4:]}"
