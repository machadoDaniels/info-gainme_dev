"""Lightweight OpenAI-compatible LLM adapter.

This module provides a minimal adapter around OpenAI (or OpenAI-compatible) chat
completions, maintaining an internal chat history. It is intentionally small
and dependency-light, following the project guidelines.

Public API (aligned with UML/IDEIA_ESCOPO):
  - class LLMAdapter
    - append_history(role: str, text: str) -> None
    - reset_history() -> None

No side effects occur on import. The OpenAI client is imported lazily at call time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Optional
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

Role = Literal["system", "user", "assistant"]

@dataclass(slots=True)
class LLMConfig:
    """Configuration for `LLMAdapter`.

    Args:
        model: Model name (e.g., "gpt-4o-mini" or a vLLM-exposed model).
        api_key: API key to authenticate. If None, the OpenAI client will
            fallback to environment configuration.
        base_url: Optional custom base URL (e.g., vLLM OpenAI-compatible endpoint).
        timeout: Request timeout in seconds.
        max_tokens: Optional cap for tokens in the response.
        temperature: Sampling temperature.
        user: Optional user identifier for tracking/auditing.
        extra: Extra provider-specific parameters to pass through.
    """

    model: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    timeout: float = 60.0
    max_tokens: Optional[int] = None
    temperature: float = 0.2
    user: Optional[str] = None
    extra: dict[str, Any] = field(default_factory=dict)


class LLMAdapterError(RuntimeError):
    """Represents errors raised by the `LLMAdapter`."""


class LLMAdapter:
    """Minimal adapter for OpenAI-compatible chat completions.

    This adapter keeps an internal chat `history` and provides methods to append
    messages and reset the history. The `generate` method performs a chat
    completion request using the configured provider.

    Example:
        config = LLMConfig(model="gpt-4o-mini")
        llm = LLMAdapter(config)
        llm.append_history("system", "You are a helpful assistant.")
        llm.append_history("user", "Hello!")
        reply = llm.generate()
    """

    def __init__(
        self, 
        config: Optional[LLMConfig] = None,
        save_history: bool = True
        ) -> None:
        self._config: LLMConfig = config
        self._history: list[dict[str, str]] = [] if save_history else None
        self._save_history = save_history

        assert self._config.api_key is not None if not self._config.base_url else True, "Both api_key and base_url cannot be None"

    # --- History management (Pythonic) ---
    def append_history(self, role: Role, text: str) -> None:
        """Append one message to the internal history.

        Args:
            role: One of {"system", "user", "assistant"}.
            text: Message content.

        Raises:
            ValueError: If `role` is invalid or `text` is empty.
        """
        assert self._save_history, "Trying to append history but save_history is False"
        if role not in ("system", "user", "assistant"):
            raise ValueError(f"Invalid role: {role}")
        if not text:
            raise ValueError("text must be non-empty")
        self._history.append({"role": role, "content": text})

    def reset_history(self) -> None:
        """Clear the internal chat history."""
        self._history.clear()


    # --- Accessors ---
    @property
    def history(self) -> list[dict[str, str]]:
        """Return a shallow copy of the internal history for read-only use."""
        return list(self._history)

    @property
    def config(self) -> LLMConfig:
        """Return the underlying configuration."""
        return self._config

    @staticmethod
    def see_messages(messages: list[dict[str, str]]) -> str:
        """Return the messages as a string pretty-printed."""
        lines = []
        for i, msg in enumerate(messages, 1):
            role = msg["role"].upper()
            content = msg["content"]
            lines.append(f"{i:2d}. {role}: {content}")
        
        return "\n".join(lines)
        
    def see_history(self) -> str:
        """Return the internal history as a string pretty-printed."""
        if not self._save_history or not self._history:
            return "No history available."
        
        return self.see_messages(self._history)


    # --- Core generation ---
    def generate(
        self,
        add_to_history: Optional[bool] = None,
        messages: Optional[list[dict[str, str]]] = None,
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        extra: Optional[dict[str, Any]] = None,
    ) -> str:
        """Call the provider to generate the next assistant message.

        Args:
            messages: If provided, use these messages instead of `self.history`.
            add_to_history: If True, append the assistant reply to `self.history`.
            temperature: Optional override for sampling temperature.
            max_tokens: Optional override for max tokens.
            extra: Optional dict of provider-specific parameters.

        Returns:
            Assistant reply text.

        Raises:
            LLMAdapterError: On client import or API errors.
            ValueError: If there are no messages to send.
        """
        if add_to_history is None:
            add_to_history = self._save_history

        if add_to_history and not self._save_history:
            raise ValueError("Trying to add to history but save_history is False")

        payload_messages = messages if messages is not None else self._history
        
        if not payload_messages:
            raise ValueError("No messages to send. Provide `messages` or add history.")

        client_kwargs: dict[str, Any] = {}
        if self._config.api_key is not None:
            client_kwargs["api_key"] = self._config.api_key
        if self._config.base_url is not None:
            client_kwargs["base_url"] = self._config.base_url

        client = OpenAI(**client_kwargs)

        request_kwargs: dict[str, Any] = {
            "model": self._config.model,
            "messages": payload_messages,
            "temperature": self._config.temperature if temperature is None else temperature,
            "timeout": self._config.timeout,
        }
        if self._config.user is not None:
            request_kwargs["user"] = self._config.user
        if self._config.max_tokens is not None or max_tokens is not None:
            request_kwargs["max_tokens"] = self._config.max_tokens if max_tokens is None else max_tokens
        if self._config.extra:
            request_kwargs.update(self._config.extra)
        if extra:
            request_kwargs.update(extra)

        try:
            completion = client.chat.completions.create(**request_kwargs)
        except Exception as exc:  # pragma: no cover - provider/network specific
            raise LLMAdapterError(f"Provider error: {exc}") from exc

        try:
            content = completion.choices[0].message.content or ""
        except Exception as exc:  # pragma: no cover - schema guard
            raise LLMAdapterError("Malformed response from provider.") from exc

        if add_to_history and content:
            self._history.append({"role": "assistant", "content": content})

        return content


__all__ = [
    "LLMConfig",
    "LLMAdapter",
    "LLMAdapterError",
]


if __name__ == "__main__":
    
    # Teste usando o modelo gpt-4o-mini através da API da OpenAI
    # adapter = LLMAdapter(LLMConfig(model="gpt-4o-mini"))

    # Teste através da API do vLLM
    adapter = LLMAdapter(
        LLMConfig(
            model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",  # deve bater com o served_model_name
            base_url="http://localhost:8000/v1",
            api_key="EMPTY",          # dummy
            temperature=0.2,
            max_tokens=128,
            timeout=60.0,
        )
    )

    adapter.append_history("system", "You are a helpful assistant.")
    adapter.append_history("user", "Explain how a geographic knowledge graph works.")
    resposta = adapter.generate()
    print(resposta)