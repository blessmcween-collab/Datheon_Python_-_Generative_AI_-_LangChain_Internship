"""Thin wrappers that give Gemini, OpenAI and a mock the same interface.

SDK imports are lazy, so you only need the package for the provider you use.
"""
from __future__ import annotations

import json
import os
from typing import Protocol


class LLMProvider(Protocol):
    def generate(self, system: str, messages: list[dict]) -> str:
        """messages: [{"role": "user"|"assistant", "content": str}, ...]; returns raw text."""


class OpenAIProvider:
    def __init__(self, model: str | None = None, temperature: float = 0.0):
        from openai import OpenAI  # pip install openai

        self.client = OpenAI()  # reads OPENAI_API_KEY
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.temperature = temperature

    def generate(self, system: str, messages: list[dict]) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            response_format={"type": "json_object"},  # JSON mode
            messages=[{"role": "system", "content": system}, *messages],
        )
        return resp.choices[0].message.content or ""


class GeminiProvider:
    """Uses the current `google-genai` SDK (the older `google-generativeai`
    package is deprecated). Set GEMINI_API_KEY or GOOGLE_API_KEY."""

    def __init__(self, model: str | None = None, temperature: float = 0.0):
        from google import genai  # pip install google-genai

        self.genai = genai
        self.client = genai.Client()
        self.model = model or os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        self.temperature = temperature

    def generate(self, system: str, messages: list[dict]) -> str:
        types = self.genai.types
        contents = [
            types.Content(role="model" if m["role"] == "assistant" else "user",
                          parts=[types.Part(text=m["content"])])
            for m in messages
        ]
        resp = self.client.models.generate_content(
            model=self.model,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=system,
                temperature=self.temperature,
                response_mime_type="application/json",  # JSON mode
            ),
        )
        return resp.text or ""


class MockProvider:
    """Returns canned responses in order - for tests and offline demos."""

    def __init__(self, responses: list[str | dict]):
        self.responses = [r if isinstance(r, str) else json.dumps(r) for r in responses]
        self.calls: list[list[dict]] = []

    def generate(self, system: str, messages: list[dict]) -> str:
        self.calls.append(list(messages))
        return self.responses.pop(0)


def get_provider(name: str, model: str | None = None) -> LLMProvider:
    name = name.lower()
    if name == "openai":
        return OpenAIProvider(model)
    if name == "gemini":
        return GeminiProvider(model)
    raise ValueError("provider must be 'openai' or 'gemini'")
