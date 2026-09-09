import yaml

from rag.app import build_pipeline, paths_for
from rag.cache import CachedLLM, cache_key
from rag.chunking import write_chunks
from rag.cli import main
from rag.llm import MockBackend

from test_retrieval import make_chunks


class CountingMock(MockBackend):
    def __init__(self):
        super().__init__()
        self.name = "counting"
        self.is_mock = False
        self.calls = 0

    def chat(self, messages, max_tokens):
        self.calls += 1
        return super().chat(messages, max_tokens)


def test_llm_cache_returns_stored_reply_and_key_depends_on_everything(tmp_path):
    llm = CountingMock()
    cached = CachedLLM(llm, tmp_path / "llm.sqlite")
    messages = [{"role": "user", "content": '<source id="1" title="t">\nДлина массива.\n</source>'}]
    first = cached.chat(messages, 100)
    assert cached.chat(messages, 100) == first and llm.calls == 1 and cached.hits == 1
    reopened = CachedLLM(CountingMock(), tmp_path / "llm.sqlite")
    assert reopened.chat(messages, 100) == first and reopened.llm.calls == 0   # кеш на диске
    assert cache_key("m", messages, 100) != cache_key("m", messages, 101) != cache_key("other", messages, 100)


def config_for(tmp_path) -> dict:
    config = {"data_dir": str(tmp_path), "chunking": {"chunk_size": 1000, "overlap": 150},
              "retriever": {"model": "bm25"}, "reranker": None, "generation": {"top_k": 3},
              "llm": {"backend": "mock"}, "cache": {"llm": False}}
    write_chunks(paths_for(config).chunks, make_chunks())
    return config


def test_pipeline_from_config_with_bm25_and_mock(tmp_path):
    answer = build_pipeline(config_for(tmp_path)).answer("длина массива")
    assert answer.is_mock and answer.sources and {s.chunk_id for s in answer.sources} <= set(answer.context)


def test_cli_ask_json(tmp_path, capsys):
    path = tmp_path / "rag.yaml"
    path.write_text(yaml.safe_dump(config_for(tmp_path), allow_unicode=True), encoding="utf-8")
    assert main(["--config", str(path), "ask", "--json", "длина массива"]) == 0
    out = capsys.readouterr().out
    assert '"is_mock": true' in out and '"sources"' in out
