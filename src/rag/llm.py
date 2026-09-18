"""Заменяемые LLM-бэкенды: llama.cpp (локальная GGUF-модель) и mock для проверки связности."""
from __future__ import annotations

import re
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Iterator
import json
from urllib.parse import urlparse


def weight_hashes(model_path):
    model_path = Path(model_path)
    split = re.search(r"-00001-of-(\d+)\.gguf$", model_path.name)
    paths = [model_path] if not split else [
        model_path.with_name(re.sub(r"-00001-of-", f"-{i:05}-of-", model_path.name))
        for i in range(1, int(split.group(1)) + 1)]
    result = {}
    for path in paths:
        with path.open("rb") as handle:
            result[path.name] = hashlib.file_digest(handle, "sha256").hexdigest()
    return result


class LLMBackend(Protocol):
    name: str
    is_mock: bool

    def chat(self, messages: list[dict], max_tokens: int) -> str: ...

    def count_tokens(self, text: str) -> int: ...

    def stream_chat(self, messages: list[dict], max_tokens: int) -> Iterator[str]: ...


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

    def stream_chat(self, messages: list[dict], max_tokens: int) -> Iterator[str]:
        yield self.chat(messages, max_tokens)


class LlamaCppBackend:
    """Локальная GGUF-модель через llama-cpp-python.

    Генерация жадная (temperature=0), поэтому ответ воспроизводим при тех же
    весах, промпте и версии llama.cpp. repeat_penalty=1.1: без него модель на
    1.5B параметров иногда зацикливается, повторяя заголовки фрагментов.
    """

    is_mock = False
    decoding = {"temperature": 0.0, "top_p": 1.0, "repeat_penalty": 1.1, "seed": 0}

    def __init__(self, model_path: str | Path, n_ctx: int = 4096, n_threads: int | None = None,
                 seed: int = 42, n_gpu_layers: int = 0):
        import llama_cpp
        from llama_cpp import Llama

        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"GGUF-модель не найдена: {model_path}. См. README, раздел «Модели».")
        self.name = f"llama.cpp:{model_path.name}"
        self.decoding = {**type(self).decoding, "seed": seed}
        self.weights_hashes = weight_hashes(model_path)
        self.weights_sha256 = hashlib.sha256(repr(sorted(self.weights_hashes.items())).encode()).hexdigest()
        # всё, от чего зависит ответ при тех же сообщениях: ключ кеша (rag/cache.py)
        self.fingerprint = (f"{self.name}|sha256:{self.weights_sha256}|llama-cpp-python {llama_cpp.__version__}"
                            f"|n_ctx:{n_ctx}|n_gpu_layers:{n_gpu_layers}|{sorted(self.decoding.items())}")
        self.llm = Llama(
            model_path=str(model_path), n_ctx=n_ctx, n_threads=n_threads, seed=seed,
            n_gpu_layers=n_gpu_layers, verbose=False
        )

    def chat(self, messages: list[dict], max_tokens: int) -> str:
        out = self.llm.create_chat_completion(
            messages=messages, max_tokens=max_tokens, **self.decoding,
        )
        return out["choices"][0]["message"]["content"].strip()

    def count_tokens(self, text: str) -> int:
        return len(self.llm.tokenize(text.encode("utf-8"), add_bos=False))

    def stream_chat(self, messages: list[dict], max_tokens: int) -> Iterator[str]:
        stream = self.llm.create_chat_completion(messages=messages, max_tokens=max_tokens,
                                                 stream=True, **self.decoding)
        try:
            for event in stream:
                content = event["choices"][0].get("delta", {}).get("content")
                if content:
                    yield content
        finally:
            stream.close()

    def json_chat(self, messages, max_tokens, schema):
        out = self.llm.create_chat_completion(messages=messages, max_tokens=max_tokens,
            response_format={"type": "json_object", "schema": schema}, **self.decoding)
        return out["choices"][0]["message"]["content"]


class LlamaServerBackend:
    """Официальный llama-server, привязанный к loopback; GPU-runtime отдельным процессом."""
    is_mock = False

    def __init__(self, base_url, model_path, seed=42, chat_template_kwargs=None, **kwargs):
        import requests
        parsed = urlparse(base_url)
        if parsed.scheme != "http" or parsed.hostname not in ("127.0.0.1", "localhost", "::1"):
            raise ValueError("llama_server разрешён только по локальному HTTP")
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.session.trust_env = False  # loopback не отправляется через внешний proxy
        path = Path(model_path)
        self.name = f"llama-server:{path.name}"
        self.seed = seed
        self.chat_template_kwargs = chat_template_kwargs or {}
        props = self.session.get(self.base_url + "/props", timeout=10)
        props.raise_for_status()
        self.props = props.json()
        loaded_path = self.props.get("model_path")
        if loaded_path and Path(loaded_path).resolve() != path.resolve():
            raise ValueError(f"llama-server загрузил другую модель: {loaded_path}")
        self.weights_hashes = weight_hashes(path)
        version = self.session.get(self.base_url + "/version", timeout=10)
        # Некоторые версии не имеют /version; полный /props всё равно сохранён.
        self.version = version.json() if version.ok else {"status": version.status_code}
        fingerprint_props = {**self.props, "model_path": path.name}
        if loaded_path and self.props.get("model_alias") == loaded_path:
            fingerprint_props["model_alias"] = path.name
        self.fingerprint = json.dumps({"name": self.name, "weights": self.weights_hashes,
                                       "props": fingerprint_props, "version": self.version,
                                       "seed": seed, "chat_template_kwargs":self.chat_template_kwargs}, sort_keys=True)

    def _payload(self, messages, max_tokens):
        return {"messages": messages, "max_tokens": max_tokens, "temperature": 0.0,
                "top_p": 1.0, "repeat_penalty": 1.1, "seed": self.seed,
                "chat_template_kwargs":self.chat_template_kwargs}

    def chat(self, messages, max_tokens):
        response = self.session.post(self.base_url + "/v1/chat/completions",
                                     json=self._payload(messages, max_tokens), timeout=600)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()

    def stream_chat(self, messages, max_tokens):
        payload = {**self._payload(messages, max_tokens), "stream": True}
        with self.session.post(self.base_url + "/v1/chat/completions", json=payload, stream=True, timeout=600) as response:
            response.raise_for_status()
            for line in response.iter_lines(chunk_size=1):
                if not line.startswith(b"data: "):
                    continue
                if line == b"data: [DONE]":
                    break
                event = json.loads(line[6:])
                content = event["choices"][0].get("delta", {}).get("content")
                if content:
                    yield content

    def count_tokens(self, text):
        response = self.session.post(self.base_url + "/tokenize", json={"content": text, "add_special": False}, timeout=30)
        response.raise_for_status()
        return len(response.json()["tokens"])

    def json_chat(self, messages, max_tokens, schema):
        payload = {**self._payload(messages, max_tokens), "response_format": {"type": "json_object", "schema": schema}}
        response = self.session.post(self.base_url + "/v1/chat/completions", json=payload, timeout=600)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]


def make_backend(cfg: dict) -> LLMBackend:
    kind = cfg["backend"]
    if kind == "mock":
        return MockBackend()
    if kind == "llama_server":
        return LlamaServerBackend(cfg["base_url"], cfg["model_path"], seed=cfg.get("seed", 42),
                                   chat_template_kwargs=cfg.get('chat_template_kwargs'))
    if kind == "llama_cpp":
        return LlamaCppBackend(cfg["model_path"], n_ctx=cfg.get("n_ctx", 4096),
                               n_threads=cfg.get("n_threads"), seed=cfg.get("seed", 42),
                               n_gpu_layers=cfg.get("n_gpu_layers", 0))
    raise ValueError(f"Неизвестный LLM-бэкенд: {kind}")
