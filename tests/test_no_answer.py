import json
from pathlib import Path

import pytest

from rag.generation import GenerationConfig, RAGPipeline, has_content, is_refusal, read_reply
from rag.llm import MockBackend
from rag.prompt import NO_ANSWER_TEXT
from rag.retrieval import build_bm25

from test_generation import FixedAnswer
from test_retrieval import make_chunks

ROOT = Path(__file__).resolve().parents[1]


def pipeline(text: str, **config) -> RAGPipeline:
    return RAGPipeline(build_bm25(make_chunks()), FixedAnswer(text), GenerationConfig(top_k=3, **config))


@pytest.mark.parametrize("raw", [
    NO_ANSWER_TEXT,
    "[1] В документах нет ответа на этот вопрос.",
    "К сожалению, в документах НЕТ ОТВЕТА на этот вопрос [2].",
])
def test_refusal_never_has_sources(raw):
    answer = pipeline(raw).answer("длина массива")
    assert answer.refused and answer.refusal_reason == "model"
    assert answer.sources == [] and answer.answer == NO_ANSWER_TEXT
    assert answer.raw_answer == raw


@pytest.mark.parametrize("raw", ["[1]", "[1] [4]", " [2]. "])
def test_answer_with_only_citations_is_refusal(raw):
    answer = pipeline(raw).answer("длина массива")
    assert answer.refused and answer.refusal_reason == "empty" and answer.sources == []


def test_answer_without_citations_is_marked_unsupported():
    answer = pipeline("Длина массива хранится в свойстве length.").answer("длина массива")
    assert not answer.refused and answer.unsupported and answer.sources == []


def test_retrieval_gate_skips_model():
    backend = FixedAnswer("не должно вызываться [1]")
    rag = RAGPipeline(build_bm25(make_chunks()), backend, GenerationConfig(top_k=3, min_retrieval_score=1e6))
    answer = rag.answer("длина массива")
    assert answer.refused and answer.refusal_reason == "retrieval"
    assert backend.messages is None and answer.sources == [] and answer.context == []


@pytest.mark.parametrize("raw", [
    "Ответ [1].", "Ответ [3][1].", "Ответ [0] [99] [2,3].", "[5] да", "Текст [1, 2; 3].",
])
def test_sources_are_always_from_the_prompt(raw):
    answer = pipeline(raw).answer("массив")
    assert {s.chunk_id for s in answer.sources} <= set(answer.context)
    assert all(1 <= s.number <= len(answer.context) for s in answer.sources)
    assert not set(answer.invalid_citations) & {s.number for s in answer.sources}


@pytest.mark.parametrize("raw", [
    "Метод sort() сортирует массив на месте [1].\n\n[2] В документах нет ответа на этот вопрос.",
    "Метод sort() сортирует массив на месте [1]. В документах нет ответа на этот вопрос [2].",
])
def test_refusal_sentence_is_removed_from_real_answer(raw):
    answer = pipeline(raw).answer("массив")
    assert not answer.refused and answer.partial_refusal
    assert answer.answer == "Метод sort() сортирует массив на месте [1]."
    assert [s.number for s in answer.sources] == [1]


def test_helpers():
    assert is_refusal("в документах нет ответа")
    assert not is_refusal("Ответ есть [1].")
    assert read_reply("«В документах нет ответа на этот вопрос.»").refusal == "model"
    assert read_reply("[1] [2]").refusal == "empty"
    assert not has_content("[1] [2].") and has_content("x [1]")


def test_unanswerable_questions_file_is_well_formed():
    rows = [json.loads(line) for line in (ROOT / "data/eval/unanswerable.jsonl").read_text(encoding="utf-8").splitlines() if line]
    assert len({r["id"] for r in rows}) == len(rows)
    assert {r["split"] for r in rows} == {"dev", "test"}
    assert all(r["absent_terms"] and r["question"].endswith("?") for r in rows)


def test_mock_refuses_without_context():
    class Empty:
        def search(self, question, top_k):
            return []

    answer = RAGPipeline(Empty(), MockBackend()).answer("что угодно")
    assert answer.refused and answer.refusal_reason == "retrieval"


class Replies(FixedAnswer):
    """Возвращает ответы по очереди и запоминает все запросы."""

    def __init__(self, *texts):
        super().__init__(texts[0])
        self.texts = list(texts)
        self.calls = []

    def chat(self, messages, max_tokens):
        self.calls.append(messages)
        return self.texts[len(self.calls) - 1]


def test_citation_only_answer_is_retried_once_when_enabled():
    backend = Replies("[1] [2]", "Длина массива хранится в свойстве length [1].")
    rag = RAGPipeline(build_bm25(make_chunks()), backend, GenerationConfig(top_k=3, retry_empty=True))
    answer = rag.answer("длина массива")
    assert len(backend.calls) == 2 and backend.calls[1][-2] == {"role": "assistant", "content": "[1] [2]"}
    assert answer.retried and answer.first_raw_answer == "[1] [2]" and not answer.refused
    assert [s.number for s in answer.sources] == [1]


def test_retry_is_not_used_for_real_answers_or_when_disabled():
    backend = Replies("Ответ [1].")
    RAGPipeline(build_bm25(make_chunks()), backend, GenerationConfig(top_k=3, retry_empty=True)).answer("массив")
    assert len(backend.calls) == 1
    backend = Replies("[1]")
    answer = RAGPipeline(build_bm25(make_chunks()), backend, GenerationConfig(top_k=3, retry_empty=False)).answer("массив")
    assert len(backend.calls) == 1 and answer.refusal_reason == "empty" and not answer.retried
