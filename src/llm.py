"""A thin client for any OpenAI-compatible chat completions endpoint.

Responses are cached on disk, keyed by the exact request, so re-running an
evaluation replays the same model outputs instead of paying for (and varying
on) new ones. Rate limits are waited out rather than treated as failures.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import openai
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT / ".cache" / "llm"


@dataclass
class Usage:
    calls: int = 0
    cached_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0


@dataclass
class LLMClient:
    model: str
    base_url: str
    api_key: str
    temperature: float = 0.7
    # Reasoning models spend part of this on hidden reasoning before the answer.
    max_tokens: int = 6000
    # Some free tiers cap prompt + max_tokens per request (Groq: 8000 tokens per
    # minute), so max_tokens shrinks to fit long prompts under this ceiling.
    request_token_cap: int = 7800
    max_retries: int = 8
    # Longer server-requested waits mean a daily quota, not a burst limit.
    max_wait: float = 600
    use_cache: bool = True
    usage: Usage = field(default_factory=Usage)

    def __post_init__(self):
        self._client = openai.OpenAI(base_url=self.base_url, api_key=self.api_key,
                                     max_retries=0, timeout=120)

    @classmethod
    def from_env(cls, model: str | None = None, **kwargs) -> "LLMClient":
        load_dotenv(ROOT / ".env")
        api_key = os.environ.get("LLM_API_KEY", "")
        if not api_key:
            sys.exit("LLM_API_KEY is not set. Copy .env.example to .env and fill it in.")
        return cls(
            model=model or os.environ.get("LLM_MODEL", "openai/gpt-oss-120b"),
            base_url=os.environ.get("LLM_BASE_URL", "https://api.groq.com/openai/v1"),
            api_key=api_key,
            request_token_cap=int(os.environ.get("LLM_REQUEST_TOKEN_CAP", 7800)),
            **kwargs,
        )

    def _cache_path(self, messages: list[dict], salt: str) -> Path:
        key = json.dumps({"model": self.model, "temperature": self.temperature,
                          "max_tokens": self.max_tokens, "messages": messages, "salt": salt},
                         sort_keys=True, ensure_ascii=True)
        return CACHE_DIR / f"{hashlib.sha256(key.encode()).hexdigest()}.json"

    def chat(self, messages: list[dict], salt: str = "") -> str:
        """Send a chat request in JSON mode and return the raw text of the reply.

        `salt` only affects the cache key, so asking the same prompt twice on
        purpose (say, as a retry) gets a fresh sample rather than a replay.
        """
        path = self._cache_path(messages, salt)
        if self.use_cache and path.is_file():
            cached = json.loads(path.read_text(encoding="utf-8"))
            content, usage = cached["content"], cached.get("usage") or {}
            self.usage.cached_calls += 1
        else:
            content, usage = self._request(messages)
        # Token counts include replayed calls, so they describe the whole run's cost.
        self.usage.calls += 1
        self.usage.prompt_tokens += usage.get("prompt_tokens") or 0
        self.usage.completion_tokens += usage.get("completion_tokens") or 0
        if self.use_cache and not path.is_file():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"model": self.model, "content": content, "usage": usage},
                                       ensure_ascii=False), encoding="utf-8")
        return content

    def _completion_budget(self, messages: list[dict]) -> int:
        # About 3 characters per token is a deliberately high estimate for this text.
        est_prompt = sum(len(m["content"]) for m in messages) // 3 + 50
        return max(1024, min(self.max_tokens, self.request_token_cap - est_prompt))

    def _request(self, messages: list[dict]) -> tuple[str, dict]:
        delay = 5.0
        for attempt in range(self.max_retries + 1):
            try:
                resp = self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self._completion_budget(messages),
                    response_format={"type": "json_object"},
                )
                usage = resp.usage.model_dump() if resp.usage else {}
                usage["finish_reason"] = resp.choices[0].finish_reason
                return resp.choices[0].message.content or "", usage
            except openai.RateLimitError as exc:
                wait = _retry_after(exc) or delay
                if wait > self.max_wait:
                    raise RuntimeError(
                        f"Rate limit needs a {wait:.0f}s wait, which looks like a daily quota. "
                        "Re-run the same command later; finished requests are replayed from the cache. "
                        f"{exc}") from exc
            except (openai.APITimeoutError, openai.APIConnectionError, openai.InternalServerError):
                wait = delay
            except openai.BadRequestError as exc:
                # JSON mode can refuse a reply that isn't valid JSON. Hand back the
                # failed generation, if the provider includes it, so the solver can
                # count it as an unparseable round rather than crash the run.
                failed = _failed_generation(exc)
                if failed is not None:
                    return failed, {"finish_reason": "json_validate_failed"}
                raise
            if attempt == self.max_retries:
                raise RuntimeError(f"LLM request still failing after {self.max_retries} retries")
            print(f"  [llm] waiting {wait:.0f}s before retrying", file=sys.stderr, flush=True)
            time.sleep(wait)
            delay = min(delay * 2, 120)
        raise AssertionError("unreachable")


def _retry_after(exc: openai.APIStatusError) -> float | None:
    value = exc.response.headers.get("retry-after") if exc.response is not None else None
    try:
        return float(value) + 1 if value else None
    except ValueError:
        return None


def _failed_generation(exc: openai.BadRequestError) -> str | None:
    body = exc.body if isinstance(exc.body, dict) else {}
    err = body.get("error", body)
    if isinstance(err, dict) and err.get("code") == "json_validate_failed":
        return err.get("failed_generation") or ""
    return None
