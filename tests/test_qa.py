import json
from pathlib import Path

import pytest

from rag.chunking import Chunk
from rag.evaluation import covers, evidence_metrics, evidence_rank
from rag.ingest import load_documents
from rag.qa import Evidence, QAItem, load_qa, mentions_answer_term
from rag.retrieval import ChunkHit

ROOT = Path(__file__).resolve().parents[1]
QA = ROOT / "data/eval/qa.jsonl"
DOCUMENTS = ROOT / "data/processed/documents.jsonl"


def test_qa_file_matches_sse_queries_and_unanswerable_set():
    items = load_qa(QA)
    assert len({i.id for i in items}) == len(items)
    answerable = [i for i in items if i.answerable]
    for split in ("dev", "test"):
        sse = [json.loads(line) for line in (ROOT / f"data/eval/sse/{split}.jsonl").read_text(encoding="utf-8").splitlines() if line]
        mine = {i.id: i for i in answerable if i.split == split}
        assert set(mine) == {q["query_id"] for q in sse}
        for q in sse:
            assert mine[q["query_id"]].question == q["query"] and mine[q["query_id"]].relevant == q["relevant"]
    unanswerable = [json.loads(line) for line in (ROOT / "data/eval/unanswerable.jsonl").read_text(encoding="utf-8").splitlines() if line]
    assert {i.id for i in items if not i.answerable} == {u["id"] for u in unanswerable}
    for item in answerable:
        assert item.reference_answer and item.evidence and item.answer_terms
        # термин проверяет ответ, а не повторяет вопрос; эталон его содержит
        assert not mentions_answer_term(item.question, item.answer_terms), item.id
        assert mentions_answer_term(item.reference_answer, item.answer_terms), item.id
        # доказательство — из размеченных страниц или страниц, ссылку на которые даёт ответ
        assert all(ev.quote.strip() == ev.quote and len(ev.quote) >= 20 for ev in item.evidence)


@pytest.mark.skipif(not DOCUMENTS.exists(), reason="нужен data/processed/documents.jsonl (scripts/ingest.py)")
def test_every_evidence_quote_is_in_its_document():
    items = load_qa(QA, load_documents(DOCUMENTS))   # падает, если цитаты нет
    for item in items:
        for ev in item.evidence:
            assert 0 <= ev.start < ev.end


def chunk(doc_id, start, end):
    return Chunk(f"{doc_id}:{start}-{end}", doc_id, start, end, "x" * (end - start), "t", [], "u")


def test_covers_needs_same_document_and_half_of_quote():
    ev = Evidence("s", "doc", "q" * 100, start=900, end=1000)
    assert covers(chunk("doc", 850, 1850), ev)
    assert covers(chunk("doc", 0, 950), ev)          # ровно половина цитаты
    assert not covers(chunk("doc", 0, 949), ev)
    assert not covers(chunk("other", 850, 1850), ev)


def test_evidence_rank_and_metrics():
    item = QAItem("q1", "dev", "?", True, evidence=[Evidence("s", "doc", "q" * 10, start=500, end=510)])
    hits = [ChunkHit(chunk("doc", 0, 400), 0.9, 1), ChunkHit(chunk("x", 0, 900), 0.8, 2), ChunkHit(chunk("doc", 300, 1300), 0.7, 3)]
    assert evidence_rank(hits, item) == 3
    m = evidence_metrics(hits, item)
    assert m["evidence@1"] == 0 and m["evidence@3"] == 1 and m["evidence_mrr@10"] == pytest.approx(1 / 3)
    assert evidence_rank(hits[:2], item) is None


def test_mentions_answer_term_matches_from_word_start():
    assert mentions_answer_term("Используйте arr.length.", ["length"])
    assert mentions_answer_term("комбинация функции и лексического окружения", ["лексическ"])
    assert mentions_answer_term("Метод SORT()", ["sort"])
    assert not mentions_answer_term("resorted", ["sort"])
    assert mentions_answer_term("x => x * 2", ["=>"])
