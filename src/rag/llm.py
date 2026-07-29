"""Заменяемые LLM-бэкенды: llama.cpp (локальная GGUF-модель) и mock для проверки связности."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


class LLMBackend(Protocol):
    name: str
    is_mock: bool

    def chat(self, messages: list[dict], max_tokens: int) -> str: ...

    def count_tokens(self, text: str) -> int: ...


@dataclass
class MockBackend:
    """Детерминированный бэкенд без модели.

    Отвечает первым предложением первого фрагмента контекста со ссылкой на него
    или отказом, если контекста нет. Нужен только для тестов связности:
    его ответы не попадают в таблицы качества.
    """

    name: str = "mock"
    is_mock: bool = True

    def chat(self, messages: list[dict], max_tokens: int) -> str:
        user = messages[-1]["content"]
        match = re.search(r'<source id="(\d+)"[^>]*>\n(.*?)\n</source>', user, flags=re.S)
        if match is None:
            return "В документах нет ответа на этот вопрос."
        source_id, text = match.group(1), match.group(2)
        sentence = re.split(r"(?<=[.!?])\s", text.strip(), maxsplit=1)[0]
        return f"{sentence} [{source_id}]"

    def count_tokens(self, text: str) -> int:
        # Грубая оценка: ~4 символа на токен.
        return max(1, len(text) // 4)


class LlamaCppBackend:
    """Локальная GGUF-модель через llama-cpp-python.

    Генерация жадная (temperature=0), поэтому ответ воспроизводим при тех же
    весах, промпте и версии llama.cpp. repeat_penalty=1.1: без него модель на
    1.5B параметров иногда зацикливается, повторяя заголовки фрагментов.
    """

    is_mock = False

    def __init__(self, model_path: str | Path, n_ctx: int = 4096, n_threads: int | None = None, seed: int = 42):
        from llama_cpp import Llama

        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"GGUF-модель не найдена: {model_path}. См. README, раздел «Модели».")
        self.name = f"llama.cpp:{model_path.name}"
        self.llm = Llama(
            model_path=str(model_path), n_ctx=n_ctx, n_threads=n_threads, seed=seed, verbose=False
        )

    def chat(self, messages: list[dict], max_tokens: int) -> str:
        out = self.llm.create_chat_completion(
            messages=messages, max_tokens=max_tokens, temperature=0.0, top_p=1.0,
            repeat_penalty=1.1, seed=0,
        )
        return out["choices"][0]["message"]["content"].strip()

    def count_tokens(self, text: str) -> int:
        return len(self.llm.tokenize(text.encode("utf-8"), add_bos=False))


def make_backend(cfg: dict) -> LLMBackend:
    kind = cfg["backend"]
    if kind == "mock":
        return MockBackend()
    if kind == "llama_cpp":
        return LlamaCppBackend(cfg["model_path"], n_ctx=cfg.get("n_ctx", 4096),
                               n_threads=cfg.get("n_threads"), seed=cfg.get("seed", 42))
    raise ValueError(f"Неизвестный LLM-бэкенд: {kind}")
