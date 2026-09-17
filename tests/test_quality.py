import pytest

from rag.retrieval import HybridRetriever, ChunkHit, rrf_merge
from rag.statistics import bootstrap_mean, cohen_kappa
from test_retrieval import make_chunks


def test_rrf_duplicate_votes_ties_and_batch_parity():
    chunks = make_chunks()
    a, b = [ChunkHit(c, 900, i + 1) for i, c in enumerate(chunks[:2])]
    assert [h.chunk.chunk_id for h in rrf_merge([[a, b], [b, a]], 2)] == [a.chunk.chunk_id, b.chunk.chunk_id]
    assert rrf_merge([[a, a]], 2)[0].score == pytest.approx(1 / 61)
    class Fixed:
        name = "fixed"
        def search(self, q, top_k=10):
            return [a, b][:top_k]
        def search_batch(self, qs, top_k=10):
            return [self.search(q, top_k) for q in qs]
    retriever = HybridRetriever(Fixed(), Fixed())
    assert retriever.search_batch(["q1", "q2"], 2) == [retriever.search("q1", 2), retriever.search("q2", 2)]


def test_bootstrap_paired_constant_delta_and_reproducibility():
    ci = bootstrap_mean([1, 1, 1], baseline=[0, 0, 0], samples=100)
    assert ci["low"] == ci["mean"] == ci["high"] == 1
    assert bootstrap_mean([0, 1, 1], samples=100) == bootstrap_mean([0, 1, 1], samples=100)
    with pytest.raises(ValueError):
        bootstrap_mean([1], baseline=[0, 1])


def test_kappa_chance_and_degenerate():
    assert cohen_kappa(["ДА", "ДА", "НЕТ", "НЕТ"], ["ДА", "НЕТ", "ДА", "НЕТ"])["kappa"] == 0
    assert cohen_kappa(["ДА", "НЕТ"], ["ДА", "НЕТ"])["kappa"] == 1
    assert cohen_kappa(["ДА"], ["ДА"])["kappa"] is None
    with pytest.raises(ValueError):
        cohen_kappa(["?"], ["ДА"])


def test_answer_evidence_requires_resolved_spans():
    from rag.answer_eval import score_answer
    from rag.qa import QAItem, Evidence
    item=QAItem('q','test','вопрос',True,evidence=[Evidence('slug','doc','цитата')])
    with pytest.raises(ValueError,match='смещения'):
        score_answer({'refused':True,'sources':[],'context':[],'refusal_reason':'model'},item)
