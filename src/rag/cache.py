"""Кеш ответов LLM на диске.

Генерация жадная, поэтому один и тот же запрос к той же модели с теми же
параметрами даёт тот же ответ (проверено на dev: два прогона совпали). Ключ —
sha256 от модели (файл весов, версия llama-cpp-python, параметры декодирования),
сообщений и max_tokens: изменение промпта, фрагментов, модели или декодирования
даёт новый ключ, и устаревший ответ не вернётся. Повторный вопрос и
повторная оценка не тратят 10–40 с CPU на генерацию.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path


def cache_key(model: str, messages: list[dict], max_tokens: int) -> str:
    payload = json.dumps({"model": model, "messages": messages, "max_tokens": max_tokens},
                         ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class CachedLLM:
    """Обёртка над LLM-бэкендом с тем же интерфейсом (`chat`, `count_tokens`)."""

    def __init__(self, llm, path: str | Path):
        self.llm = llm
        self.name = llm.name
        self.fingerprint = getattr(llm, "fingerprint", llm.name)
        self.is_mock = llm.is_mock
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.execute("CREATE TABLE IF NOT EXISTS answers (key TEXT PRIMARY KEY, model TEXT, reply TEXT)")
        self.hits = 0
        self.misses = 0

    def chat(self, messages: list[dict], max_tokens: int) -> str:
        key = cache_key(self.fingerprint, messages, max_tokens)
        row = self.db.execute("SELECT reply FROM answers WHERE key = ?", (key,)).fetchone()
        if row is not None:
            self.hits += 1
            return row[0]
        self.misses += 1
        reply = self.llm.chat(messages, max_tokens)
        self.db.execute("INSERT OR REPLACE INTO answers VALUES (?, ?, ?)", (key, self.fingerprint, reply))
        self.db.commit()
        return reply

    def count_tokens(self, text: str) -> int:
        return self.llm.count_tokens(text)
