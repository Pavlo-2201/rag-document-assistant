from rag.answer_eval import score_answer, span_from_chunk_id, summarize_scores
from rag.qa import Evidence, QAItem

ITEM = QAItem("q1", "dev", "сколько элементов в массиве", True, reference_answer="Свойство length.",
              answer_terms=["length"], evidence=[Evidence("s", "mdnru-abc", "q" * 40, start=100, end=140)])
NO_ANSWER = QAItem("u1", "dev", "Как настроить Kubernetes?", False)


def answer(text, sources=(), context=(), refused=False, reason=None):
    return {"answer": text, "refused": refused, "refusal_reason": reason, "context": list(context),
            "sources": [{"chunk_id": c} for c in sources], "timings": {"retrieval_s": 0.01, "generation_s": 10.0},
            "context_tokens": 900, "dropped_for_limit": []}


def test_span_from_chunk_id():
    span = span_from_chunk_id("mdnru-abc:0-950")
    assert (span.doc_id, span.start, span.end) == ("mdnru-abc", 0, 950)


def test_answer_citing_evidence_chunk():
    row = score_answer(answer("Используйте arr.length [1].", ["mdnru-abc:0-950"], ["mdnru-abc:0-950", "x:0-10"]), ITEM)
    assert row["evidence_in_context"] and row["cited_evidence"] and row["term_hit"] and row["answered_with_sources"]


def test_answer_citing_wrong_chunk_and_refusal_with_evidence():
    row = score_answer(answer("Используйте length [2].", ["x:0-10"], ["mdnru-abc:0-950", "x:0-10"]), ITEM)
    assert row["evidence_in_context"] and not row["cited_evidence"] and row["term_hit"]
    refused = score_answer(answer("В документах нет ответа на этот вопрос.", [], ["mdnru-abc:0-950"], True, "model"), ITEM)
    assert refused["refused"] and not refused["term_hit"] and not refused["cited_evidence"]
    summary = summarize_scores([row, refused, score_answer(answer("Ответ [1].", ["k:0-5"], ["k:0-5"]), NO_ANSWER)],
                               [answer("a")] * 3)
    assert summary["answerable"]["refused_with_evidence_in_context"] == 1
    assert summary["unanswerable"] == {"questions": 1, "refused": 0, "answered_with_sources": 1, "answered_without_sources": 0}
    assert summary["cost"]["llm_calls"] == 3
