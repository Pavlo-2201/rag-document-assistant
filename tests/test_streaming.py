from rag.cache import CachedLLM
from rag.generation import GenerationConfig, RAGPipeline
from rag.llm import MockBackend
from rag.retrieval import build_bm25
from test_retrieval import make_chunks


class StreamingMock(MockBackend):
    def stream_chat(self, messages, max_tokens):
        yield "Текст "
        yield "[1] [99]"


def test_stream_is_visible_before_return_but_final_citations_are_checked():
    parts = []
    pipeline = RAGPipeline(build_bm25(make_chunks()), StreamingMock(), GenerationConfig())
    answer = pipeline.answer("длина массива", on_token=parts.append)
    assert parts == ["Текст ", "[1] [99]"]
    assert answer.answer == "Текст [1]" and answer.invalid_citations == [99]
    assert answer.sources and "first_token_s" in answer.timings


def test_cancelled_stream_does_not_poison_cache(tmp_path):
    cached = CachedLLM(StreamingMock(), tmp_path / "cache.sqlite")
    messages = [{"role": "user", "content": "question"}]
    stream = cached.stream_chat(messages, 10)
    assert next(stream) == "Текст "
    stream.close()
    assert cached.db.execute("SELECT count(*) FROM answers").fetchone()[0] == 0
    assert "".join(cached.stream_chat(messages, 10)) == "Текст [1] [99]"
    assert list(cached.stream_chat(messages, 10)) == ["Текст [1] [99]"]
    assert cached.hits == 1 and cached.misses == 2
