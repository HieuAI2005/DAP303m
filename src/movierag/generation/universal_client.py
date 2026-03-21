"""
Universal LLM Client wrapper for MovieRAG.

Priority chain:
1. Groq → moonshotai/kimi-k2-instruct (fastest, free-tier-friendly)
"""

import os
import logging
import time
import base64
import mimetypes
import json
import re
from typing import Any
from pathlib import Path
from dotenv import load_dotenv

# Robust .env loading
env_paths = [Path(".env"), Path("src/.env"), Path("../.env")]
for p in env_paths:
    if p.exists():
        load_dotenv(p)
        break

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None
    types = None

# ── Auto-load .env from anywhere in the project tree ──────────────────────────
try:
    import sys as _sys
    from pathlib import Path as _Path

    # Walk up to find src/ and import env_loader
    _src = _Path(__file__).resolve()
    for _ in range(5):  # walk up max 5 levels
        _src = _src.parent
        if (_src / "env_loader.py").exists():
            if str(_src) not in _sys.path:
                _sys.path.insert(0, str(_src))
            import env_loader as _env_loader  # noqa: F401 – side-effect import

            break
except Exception:
    pass  # If env_loader not found, fall back to OS env vars

logger = logging.getLogger(__name__)
_THINK_TAG_RE = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)

_KIMI_MODEL = os.getenv(
    "MOVIERAG_LLM_PRIMARY_MODEL",
    os.getenv("MOVIERAG_LLM_MODEL", "moonshotai/kimi-k2-instruct"),
)
_GEMINI_MODEL = "gemini-3.1-flash-lite-preview"
_GEMINI_MAX_OUTPUT_TOKENS = int(
    os.getenv("MOVIERAG_GEMINI_MAX_OUTPUT_TOKENS", "0") or 0
)
_VISION_MAX_COMPLETION_TOKENS = int(
    os.getenv("MOVIERAG_VISION_MAX_COMPLETION_TOKENS", "0") or 0
)
_MULTI_VISION_GROQ_SAFE_LIMIT = max(
    1, int(os.getenv("MOVIERAG_MULTI_VISION_SAFE_LIMIT", "5") or 5)
)
_ALLOW_GEMINI_VISION = os.getenv("MOVIERAG_ALLOW_GEMINI_VISION", "0").lower() in {
    "1",
    "true",
    "yes",
    "on",
}


class LLMRateLimitError(RuntimeError):
    """Raised when the active LLM provider rejects requests due to rate limits."""


def is_rate_limit_error(exc: Exception | str) -> bool:
    message = str(exc).lower()
    markers = (
        "429",
        "rate limit",
        "too many requests",
        "resource_exhausted",
        "quota",
        "rate_limited",
    )
    return any(marker in message for marker in markers)


def _positive_int(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _parse_model_chain() -> list[tuple[str, dict[str, Any]]]:
    fallback_csv = os.getenv(
        "MOVIERAG_LLM_FALLBACK_MODELS",
        "qwen/qwen3-32b,openai/gpt-oss-120b",
    )
    fallback_models = [
        model.strip()
        for model in fallback_csv.replace(";", ",").split(",")
        if model.strip()
    ]
    ordered = [_KIMI_MODEL, *fallback_models]

    model_chain: list[tuple[str, dict[str, Any]]] = []
    for model_name in ordered:
        if model_name == "openai/gpt-oss-120b":
            model_chain.append(
                (
                    model_name,
                    {
                        "temperature": 1.0,
                        "max_completion_tokens": 8192,
                        "top_p": 1.0,
                        "extra_body": {"reasoning_effort": "high"},
                    },
                )
            )
        elif model_name == "qwen/qwen3-32b":
            model_chain.append(
                (
                    model_name,
                    {
                        "temperature": 0.6,
                        "max_completion_tokens": 8000,
                        "top_p": 0.95,
                    },
                )
            )
        else:
            model_chain.append(
                (
                    model_name,
                    {
                        "temperature": 0.6,
                        "max_completion_tokens": 16384,
                    },
                )
            )
    return model_chain


class UniversalLLMClient:
    """
    Unified client that routes requests using: Groq(Kimi).
    """

    def __init__(self, model_id: str = _KIMI_MODEL):
        self.model_id = model_id
        self.max_retries = max(1, int(os.getenv("MOVIERAG_LLM_MAX_RETRIES", "5")))
        self.retry_base_seconds = max(
            0.25, float(os.getenv("MOVIERAG_LLM_RETRY_BASE_SEC", "1.0"))
        )
        self.models_to_try = _parse_model_chain()

        # Environment variables like GROQ_API_KEY and GEMINI_API_KEY expected via dotenv
        self._groq_client = self._init_groq()
        self._gemini_client = self._init_gemini()

        # Mock google.genai API namespace
        _self = self

        class ModelsMock:
            def generate_content(self, model: str, contents: Any, **kwargs) -> Any:
                return _self.generate_content(model, contents, **kwargs)

        self.models = ModelsMock()

    # ── Init helpers ──────────────────────────────────────────────────

    def _init_groq(self):
        try:
            from groq import Groq

            if "GROQ_API_KEY" not in os.environ:
                return None
            return Groq(api_key=os.environ["GROQ_API_KEY"])
        except Exception as e:
            logger.warning(f"Groq init failed: {e}")
            return None

    def _init_gemini(self):
        if not genai:
            return None
        try:
            api_key = os.environ.get("GEMINI_API_KEY")
            if not api_key:
                logger.warning("GEMINI_API_KEY not found in environment.")
                return None
            return genai.Client(api_key=api_key)
        except Exception as e:
            logger.warning(f"Gemini init failed: {e}")
            return None

    # ── Response wrapper ──────────────────────────────────────────────

    class GenerateContentResponse:
        def __init__(self, text: str):
            self.text = UniversalLLMClient._normalize_response_text(text)

    @staticmethod
    def _normalize_response_text(text: Any) -> str:
        raw_text = str(text or "").strip()
        if not raw_text:
            return ""
        cleaned = _THINK_TAG_RE.sub("", raw_text).strip()
        return cleaned or raw_text

    @staticmethod
    def _gemini_config_kwargs(
        temperature: float,
        max_output_tokens: Any = None,
    ) -> dict[str, Any]:
        config_kwargs: dict[str, Any] = {"temperature": temperature}
        token_cap = _positive_int(max_output_tokens)
        if token_cap is not None:
            config_kwargs["max_output_tokens"] = token_cap
        return config_kwargs

    # ── Core generate method ──────────────────────────────────────────

    def generate_content(
        self, model: str, contents: Any, **kwargs
    ) -> GenerateContentResponse:
        """Attempt Gemini (if requested or large context) or Groq."""

        # 1. Determine model strategy
        use_gemini = False
        target_model = model or self.model_id

        if "gemini" in target_model.lower() or target_model == _GEMINI_MODEL:
            use_gemini = True

        # Build text prompt
        prompt = ""
        if isinstance(contents, str):
            prompt = contents
        elif isinstance(contents, list):
            # Handle both string lists and google.genai.types.Part lists
            parts = []
            for c in contents:
                if isinstance(c, str):
                    parts.append(c)
                elif hasattr(c, "text"):
                    parts.append(c.text)
            prompt = "\n".join(parts)

        # ── Try Gemini if requested ──────────────────────────────
        if use_gemini and self._gemini_client:
            try:
                logger.info(f"🚀 Using Gemini: {target_model}")
                config = types.GenerateContentConfig(
                    **self._gemini_config_kwargs(
                        temperature=kwargs.get("temperature", 0.6),
                        max_output_tokens=kwargs.get(
                            "max_output_tokens",
                            kwargs.get("max_tokens", _GEMINI_MAX_OUTPUT_TOKENS),
                        ),
                    )
                )

                response = self._gemini_client.models.generate_content(
                    model=target_model, contents=prompt, config=config
                )
                return self.GenerateContentResponse(text=response.text)
            except Exception as e:
                logger.error(f"❌ Gemini call failed: {e}. Falling back to Groq...")
                # Fall through to Groq if Gemini fails

        messages = self._build_messages(prompt)

        # ── Try Kimi via Groq ───────────────────────────────────
        if self._groq_client:
            completion = None
            used_model = ""
            rate_limit_errors = []

            for attempt in range(self.max_retries):
                # Swap models per attempt if possible
                model_idx = attempt % len(self.models_to_try)
                model_name, model_kwargs = self.models_to_try[model_idx]

                try:
                    logger.info(
                        f"Trying Groq model: {model_name} (Attempt {attempt + 1}/{self.max_retries})..."
                    )
                    kwargs_copy = model_kwargs.copy()
                    kwargs_copy["temperature"] = kwargs.get(
                        "temperature", kwargs_copy.get("temperature", 0.6)
                    )
                    kwargs_copy["max_completion_tokens"] = kwargs.get(
                        "max_completion_tokens",
                        kwargs_copy.get("max_completion_tokens", 16384),
                    )
                    if "top_p" in kwargs_copy or "top_p" in kwargs:
                        kwargs_copy["top_p"] = kwargs.get(
                            "top_p", kwargs_copy.get("top_p", 1.0)
                        )
                    extra = kwargs_copy.pop("extra_body", None)
                    call_args = {
                        "model": model_name,
                        "messages": messages,
                        "stream": True,
                        "stop": None,
                        **kwargs_copy,
                    }
                    if extra:
                        call_args["extra_body"] = extra

                    completion = self._groq_client.chat.completions.create(**call_args)
                    used_model = model_name
                    break  # Success
                except Exception as e:
                    logger.error(f"❌ Groq call failed for {model_name}: {e}")
                    if is_rate_limit_error(e):
                        rate_limit_errors.append(e)
                    time.sleep(self.retry_base_seconds + attempt)

            if completion:
                try:
                    full_text = ""
                    for chunk in completion:
                        full_text += chunk.choices[0].delta.content or ""

                    logger.info(
                        f"✅ Groq ({used_model}) responded ({len(full_text)} chars)"
                    )
                    return self.GenerateContentResponse(text=full_text)
                except Exception as e:
                    logger.error(f"❌ Error reading Groq stream: {e}")
                    raise e

            # If we reach here, all Groq models failed
            if rate_limit_errors:
                raise LLMRateLimitError(
                    f"All Groq retry attempts exhausted due to rate limits: {rate_limit_errors[-1]}"
                ) from rate_limit_errors[-1]
            raise RuntimeError("All Groq retry attempts failed.")

        raise RuntimeError("No Groq LLM client available. Set GROQ_API_KEY.")

    def generate_vision_content(self, prompt: str, image_path: str, **kwargs) -> str:
        """Use VLM (Gemini or Groq fallback) to analyze an image."""
        def encode_image(img_path):
            with open(img_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode("utf-8")

        mime_type, _ = mimetypes.guess_type(image_path)
        if not mime_type:
            mime_type = "image/jpeg"
        base64_image = encode_image(image_path)

        last_error = None

        # ── Try Groq Vision first ──
        if self._groq_client:
            try:
                request_kwargs = {
                    "model": "meta-llama/llama-4-scout-17b-16e-instruct",
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:{mime_type};base64,{base64_image}"
                                    },
                                },
                            ],
                        }
                    ],
                    "temperature": kwargs.get("temperature", 0.6),
                    "top_p": 1,
                    "stream": False,
                }
                max_completion_tokens = kwargs.get(
                    "max_completion_tokens",
                    kwargs.get("max_tokens", _VISION_MAX_COMPLETION_TOKENS),
                )
                if max_completion_tokens and int(max_completion_tokens) > 0:
                    request_kwargs["max_completion_tokens"] = int(max_completion_tokens)
                completion = self._groq_client.chat.completions.create(
                    **request_kwargs
                )
                logger.info("✅ Groq/Vision responded.")
                return self._normalize_response_text(
                    completion.choices[0].message.content or ""
                )
            except Exception as e:
                last_error = e
                logger.error(f"❌ Groq/Vision analysis failed: {e}. Falling back to Gemini...")

        # ── Try Gemini Vision only when explicitly enabled ──
        if _ALLOW_GEMINI_VISION and self._gemini_client:
            try:
                logger.info("🚀 Falling back to Gemini for Vision analysis.")
                from google.genai import types
                content = [
                    types.Part.from_bytes(data=base64.b64decode(base64_image), mime_type=mime_type),
                    prompt
                ]
                response = self._gemini_client.models.generate_content(
                    model=_GEMINI_MODEL,
                    contents=content,
                    config=types.GenerateContentConfig(
                        **self._gemini_config_kwargs(
                            temperature=kwargs.get("temperature", 0.1),
                            max_output_tokens=kwargs.get(
                                "max_output_tokens",
                                kwargs.get("max_tokens", _GEMINI_MAX_OUTPUT_TOKENS),
                            ),
                        )
                    )
                )
                return self._normalize_response_text(response.text)
            except Exception as e:
                last_error = e
                logger.error(f"❌ Gemini Vision failed: {e}")

        if last_error and is_rate_limit_error(last_error):
            raise LLMRateLimitError(str(last_error)) from last_error
        return "[No VLM available]"

    def generate_text(
        self, prompt: str, system_prompt: str = "", temperature: float = 0.6, **kwargs
    ) -> str:
        """Convenience wrapper: send a text-only prompt and get the response string back."""
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"System: {system_prompt}\nUser: {prompt}"
        response = self.generate_content(
            self.model_id,
            full_prompt,
            temperature=temperature,
            **kwargs,
        )
        return (
            self._normalize_response_text(response.text)
            if hasattr(response, "text")
            else self._normalize_response_text(response)
        )

    def generate_multi_vision(self, prompt: str, images_base64: list, temperature: float = 0.2, **kwargs) -> str:
        """Send multiple images + text to VLM (Gemini primary, Groq fallback)."""
        
        last_error = None

        # ── Try Groq Multi-Vision first ──
        if self._groq_client:
            # Groq Llama 4 Scout has a provider-side image-count limit.
            SAFE_LIMIT = _MULTI_VISION_GROQ_SAFE_LIMIT
            if len(images_base64) > SAFE_LIMIT:
                logger.info(f"      Downsampling images from {len(images_base64)} to {SAFE_LIMIT} for Groq compatibility.")
                step = len(images_base64) / SAFE_LIMIT
                images_base64 = [images_base64[int(i * step)] for i in range(SAFE_LIMIT)]

            # Build content array for Groq
            content = []
            for img_b64 in images_base64:
                content.append({"type": "image_url", "image_url": {"url": img_b64}})
            content.append({"type": "text", "text": prompt})

            try:
                request_kwargs = {
                    "model": "meta-llama/llama-4-scout-17b-16e-instruct",
                    "messages": [{"role": "user", "content": content}],
                    "temperature": temperature,
                    "top_p": 1,
                    "stream": False,
                }
                max_completion_tokens = kwargs.get(
                    "max_completion_tokens",
                    kwargs.get("max_tokens", _VISION_MAX_COMPLETION_TOKENS),
                )
                if max_completion_tokens and int(max_completion_tokens) > 0:
                    request_kwargs["max_completion_tokens"] = int(max_completion_tokens)
                completion = self._groq_client.chat.completions.create(
                    **request_kwargs
                )
                logger.info(f"✅ Groq/Multi-Vision responded ({len(images_base64)} images).")
                return self._normalize_response_text(
                    completion.choices[0].message.content or ""
                )
            except Exception as e:
                last_error = e
                logger.error(f"❌ Multi-vision Groq call failed: {e}. Falling back to Gemini...")

        # ── Try Gemini Multi-Vision only when explicitly enabled ──
        if _ALLOW_GEMINI_VISION and self._gemini_client:
            try:
                logger.info(f"🚀 Falling back to Gemini for Multi-Vision ({len(images_base64)} images).")
                from google.genai import types
                content = []
                for img_b64 in images_base64:
                    if "," in img_b64:
                        prefix, data = img_b64.split(",", 1)
                        mime = prefix.split(";")[0].split(":")[1]
                    else:
                        data = img_b64
                        mime = "image/jpeg"
                    content.append(types.Part.from_bytes(data=base64.b64decode(data), mime_type=mime))
                content.append(prompt)
                
                response = self._gemini_client.models.generate_content(
                    model=_GEMINI_MODEL,
                    contents=content,
                    config=types.GenerateContentConfig(
                        **self._gemini_config_kwargs(
                            temperature=temperature,
                            max_output_tokens=kwargs.get(
                                "max_output_tokens",
                                kwargs.get("max_tokens", _GEMINI_MAX_OUTPUT_TOKENS),
                            ),
                        )
                    )
                )
                return self._normalize_response_text(response.text)
            except Exception as e:
                last_error = e
                logger.error(f"❌ Gemini Multi-Vision failed: {e}")

        if last_error and is_rate_limit_error(last_error):
            raise LLMRateLimitError(str(last_error)) from last_error
        return "[No VLM available]"

    @staticmethod
    def _build_messages(prompt: str) -> list:
        """Convert plain prompt string into OpenAI-style messages list."""
        if "System:" in prompt and "User:" in prompt:
            parts = prompt.split("User:", 1)
            sys_part = parts[0].replace("System:", "").strip()
            user_part = parts[1].strip()
            return [
                {"role": "system", "content": sys_part},
                {"role": "user", "content": user_part},
            ]
        return [{"role": "user", "content": prompt}]

    # ── Tool Calling ──────────────────────────────────────────────────

    # Standard MovieRAG tool schemas (OpenAI function-calling format)
    MOVIERAG_TOOLS = [
        {
            "type": "function",
            "function": {
                "name": "search_knowledge",
                "description": "Search the MovieRAG knowledge database (scripts, subtitles, metadata) for text information about movies, actors, plot, or dialogue.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query in natural language",
                        },
                        "k": {
                            "type": "integer",
                            "description": "Max number of results (default 5)",
                            "default": 5,
                        },
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "search_visual",
                "description": "Search the visual FAISS index to find relevant keyframes/scenes by text description. Use when the user asks about appearances, scenes, or visual details.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Visual scene description",
                        },
                        "k": {
                            "type": "integer",
                            "description": "Max frames (default 6)",
                            "default": 6,
                        },
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "search_script_scenes",
                "description": "Search the screenplay-derived sub-scene index for scene headings, locations, precise dialogue windows, and who appears in a specific segment.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Scene heading, location, dialogue fragment, or screenplay-oriented query",
                        },
                        "k": {
                            "type": "integer",
                            "description": "Max results (default 4)",
                            "default": 4,
                        },
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "search_dialogue",
                "description": "Search subtitles/dialogue index for specific quotes or spoken lines in movies.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Quote or dialogue fragment",
                        },
                        "k": {
                            "type": "integer",
                            "description": "Max results (default 4)",
                            "default": 4,
                        },
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "query_graph",
                "description": "Query the scene-entity graph for relationships, scene locations, character co-occurrence, and before/after scene links.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Relationship, location, or scene-graph question",
                        },
                        "k": {
                            "type": "integer",
                            "description": "Max results (default 5)",
                            "default": 5,
                        },
                    },
                    "required": ["query"],
                },
            },
        },
    ]

    def generate_with_tools(
        self,
        prompt: str,
        tool_executor,  # callable: (tool_name, tool_args) → str
        tools: list | None = None,
        max_tool_rounds: int = 5,
        **kwargs,
    ) -> GenerateContentResponse:
        """
        Agentic tool-calling loop with Kimi/Groq.

        - Sends the prompt + tool schemas to Kimi.
        - If Kimi returns a tool_call, executes the corresponding Python function via `tool_executor`.
        - Feeds results back to the model until Kimi gives a final text response.
        - Falls back to plain `generate_content` if Groq is unavailable.

        Args:
            prompt: User + context prompt string.
            tool_executor: Callable(tool_name: str, tool_args: dict) → str (tool result as text).
            tools: Optional override of tool schemas. Defaults to MOVIERAG_TOOLS.
            max_tool_rounds: Safety cap to avoid infinite loops.
        """
        if not self._groq_client:
            logger.warning(
                "Tool calling requires Groq client. Falling back to plain generate."
            )
            return self.generate_content("kimi", prompt, **kwargs)

        tools = tools or self.MOVIERAG_TOOLS
        messages = self._build_messages(prompt)

        import time

        for _round in range(max_tool_rounds):
            resp = None

            rate_limit_errors = []
            for attempt in range(self.max_retries):
                # Swap models per attempt if possible
                model_idx = attempt % len(self.models_to_try)
                model_name, model_kwargs = self.models_to_try[model_idx]

                try:
                    logger.info(
                        f"Tool-calling Groq request with {model_name} (Attempt {attempt + 1}/{self.max_retries})..."
                    )
                    kwargs_copy = model_kwargs.copy()
                    kwargs_copy["temperature"] = kwargs.get(
                        "temperature", kwargs_copy.get("temperature", 0.6)
                    )
                    kwargs_copy["max_completion_tokens"] = kwargs.get(
                        "max_completion_tokens",
                        kwargs_copy.get("max_completion_tokens", 16384),
                    )
                    if "top_p" in kwargs_copy or "top_p" in kwargs:
                        kwargs_copy["top_p"] = kwargs.get(
                            "top_p", kwargs_copy.get("top_p", 1.0)
                        )
                    extra = kwargs_copy.pop("extra_body", None)
                    call_args = {
                        "model": model_name,
                        "messages": messages,
                        "tools": tools,
                        "tool_choice": "auto",
                        **kwargs_copy,
                    }
                    if extra:
                        call_args["extra_body"] = extra

                    resp = self._groq_client.chat.completions.create(**call_args)
                    break  # Success
                except Exception as e:
                    if is_rate_limit_error(e):
                        rate_limit_errors.append(e)
                    logger.warning(
                        f"Tool-calling Groq request failed for {model_name}: {e}."
                    )
                    time.sleep(self.retry_base_seconds + attempt)

            if not resp:
                if rate_limit_errors:
                    raise LLMRateLimitError(
                        f"All tool-calling Groq attempts exhausted due to rate limits: {rate_limit_errors[-1]}"
                    ) from rate_limit_errors[-1]
                logger.warning("All Groq models failed. Reverting to base generation.")
                return self.generate_content("kimi", prompt, **kwargs)

            choice = resp.choices[0]
            finish = choice.finish_reason

            # ── Model returned a final text answer ──
            if finish == "stop" or not choice.message.tool_calls:
                text = choice.message.content or ""
                logger.info(
                    f"✅ Kimi responded after {_round} tool round(s) ({len(text)} chars)"
                )
                return self.GenerateContentResponse(text=text)

            # ── Model requested one or more tool calls ──
            # Safely append the assistant's message, stripping unsupported fields like 'reasoning'
            m_dict = choice.message.model_dump(exclude_unset=True)
            if "reasoning" in m_dict:
                del m_dict["reasoning"]
            messages.append(m_dict)

            for tc in choice.message.tool_calls:
                fn_name = tc.function.name
                try:
                    fn_args = json.loads(tc.function.arguments)
                except Exception:
                    fn_args = {}

                logger.info(f"🔧 Kimi calls tool: {fn_name}({fn_args})")
                try:
                    tool_result = tool_executor(fn_name, fn_args)
                except Exception as ex:
                    tool_result = f"[Tool error: {ex}]"

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": str(tool_result),
                    }
                )

        logger.warning(
            "Tool calling hit max_tool_rounds limit — returning last content."
        )
        last_content = (
            messages[-1].get("content", "") if isinstance(messages[-1], dict) else ""
        )
        return self.GenerateContentResponse(text=last_content)
