import pytest

from rag.query_rewrite import HydeRetriever, rrf_merge
from rag.reranker import RerankingRetriever
from rag.retrieval import ChunkHit, build_bm25

from test_generation import FixedAnswer
from test_reranker import fake_reranker
from test_retrieval import make_chunks

QUESTION = "как сделать запрос на сервер"     # слов из ответа («fetch», «сетевой») в вопросе нет
EXPANSION = "Функция fetch выполняет сетевой запрос."


def test_concat_finds_page_named_by_api_and_keeps_original_question_for_reranker():
    llm = FixedAnswer(EXPANSION)
    first_stage = build_bm25(make_chunks())
    assert first_stage.search(QUESTION, top_k=1)[0].doc_id != "doc-b"
    hyde = HydeRetriever(first_stage, llm, mode="concat")
    assert hyde.search(QUESTION, top_k=1)[0].doc_id == "doc-b"
    assert llm.messages[-1]["content"] == QUESTION and hyde.last_expansion == EXPANSION
    assert set(hyde.last_timings) == {"hyde_s", "search_s"}

    seen = []
    reranker = fake_reranker()
    score = reranker.score
    reranker.score = lambda query, hits: seen.append(query) or score(query, hits)
    RerankingRetriever(hyde, reranker, candidates=4).search(QUESTION, top_k=2)
    assert seen == [QUESTION]


def test_rrf_keeps_hits_from_both_searches():
    hyde = HydeRetriever(build_bm25(make_chunks()), FixedAnswer(EXPANSION), mode="rrf")
    hits = hyde.search("код ответа 404", top_k=5)
    docs = [h.doc_id for h in hits]
    assert docs[0] == "doc-c" and "doc-b" in docs
    assert [h.rank for h in hits] == list(range(1, len(hits) + 1))
    assert len({h.chunk.chunk_id for h in hits}) == len(hits)


def test_empty_expansion_falls_back_to_question():
    first_stage = build_bm25(make_chunks())
    hyde = HydeRetriever(first_stage, FixedAnswer("  "), mode="rrf")
    assert [h.chunk.chunk_id for h in hyde.search("длина массива", top_k=3)] == \
        [h.chunk.chunk_id for h in first_stage.search("длина массива", top_k=3)]


def test_rrf_merge_scores_and_ties():
    a, b, c = make_chunks()[:3]
    merged = rrf_merge([[ChunkHit(a, 9, 1), ChunkHit(b, 8, 2)], [ChunkHit(b, 7, 1), ChunkHit(c, 6, 2)]], top_k=3, k=1)
    # b: 1/3 + 1/2; a: 1/2; c: 1/3
    assert [h.chunk.chunk_id for h in merged] == [b.chunk_id, a.chunk_id, c.chunk_id]
    assert merged[0].score == pytest.approx(1 / 3 + 1 / 2)


def test_unknown_mode_rejected():
    with pytest.raises(ValueError):
        HydeRetriever(build_bm25(make_chunks()), FixedAnswer(""), mode="nope")
