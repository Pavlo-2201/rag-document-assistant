from rag.reranker import CrossEncoderReranker, RerankingRetriever
from rag.retrieval import build_bm25

from test_retrieval import make_chunks


class FakeModel:
    """Оценка пары = число слов запроса, встречающихся во фрагменте."""

    def predict(self, pairs, batch_size, show_progress_bar):
        return [sum(word in text.lower() for word in query.lower().split()) for query, text in pairs]


def fake_reranker() -> CrossEncoderReranker:
    reranker = CrossEncoderReranker.__new__(CrossEncoderReranker)
    reranker.model, reranker.batch_size, reranker.name = FakeModel(), 16, "cross-encoder:fake"
    return reranker


def test_rerank_orders_by_score_and_keeps_first_stage_order_on_ties():
    hits = build_bm25(make_chunks()).search("массив длина", top_k=5)
    reranked = fake_reranker().rerank("массив длина", hits)
    scores = [h.score for h in reranked]
    assert scores == sorted(scores, reverse=True)
    assert [h.rank for h in reranked] == list(range(1, len(hits) + 1))
    assert {h.chunk.chunk_id for h in reranked} == {h.chunk.chunk_id for h in hits}
    for a, b in zip(reranked, reranked[1:]):
        if a.score == b.score:
            first = [h.chunk.chunk_id for h in hits]
            assert first.index(a.chunk.chunk_id) < first.index(b.chunk.chunk_id)


def test_reranking_retriever_only_reorders_first_stage_candidates():
    first_stage = build_bm25(make_chunks())
    retriever = RerankingRetriever(first_stage, fake_reranker(), candidates=4)
    hits = retriever.search("длина массива", top_k=2)
    candidates = {h.chunk.chunk_id for h in first_stage.search("длина массива", top_k=4)}
    assert len(hits) == 2 and {h.chunk.chunk_id for h in hits} <= candidates
    assert set(retriever.last_timings) == {"first_stage_s", "rerank_s"}
    assert fake_reranker().rerank("x", []) == []
