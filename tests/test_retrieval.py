import pytest
from semsearch.embedding import EncoderConfig

from rag.chunking import Chunk
from rag.retrieval import ChunkHit, build_bm25, build_dense_index, documents_from_chunks, load_dense

TEXTS = {
    "doc-a": ["Свойство length возвращает длину массива.", "Метод push добавляет элемент в конец массива."],
    "doc-b": ["Функция fetch выполняет сетевой запрос и возвращает промис."],
    "doc-c": ["Код ответа 404 означает, что страница не найдена.", "Код 500 — внутренняя ошибка сервера."],
}


def make_chunks() -> list[Chunk]:
    chunks = []
    for doc_id, texts in TEXTS.items():
        start = 0
        for text in texts:
            end = start + len(text)
            chunks.append(Chunk(f"{doc_id}:{start}-{end}", doc_id, start, end, text, f"Документ {doc_id}", [], f"https://e.org/{doc_id}"))
            start = end + 2
    return chunks


def test_bm25_returns_chunks_with_source():
    retriever = build_bm25(make_chunks())
    hits = retriever.search("длина массива", top_k=3)
    assert hits[0].chunk.chunk_id == "doc-a:0-41"
    assert hits[0].doc_id == "doc-a" and hits[0].rank == 1
    assert [h.rank for h in hits] == list(range(1, len(hits) + 1))


def test_faiss_index_roundtrip_gives_same_results(tmp_path):
    chunks = make_chunks()
    build_dense_index(chunks, EncoderConfig(name="lsa-4", backend="lsa", dim=4), tmp_path / "idx")
    first = load_dense(chunks, tmp_path / "idx").search("код ответа сервера", top_k=5)
    second = load_dense(chunks, tmp_path / "idx").search("код ответа сервера", top_k=5)
    assert [(h.chunk.chunk_id, round(h.score, 5)) for h in first] == [(h.chunk.chunk_id, round(h.score, 5)) for h in second]
    assert {h.chunk.chunk_id for h in first} == {c.chunk_id for c in chunks}


def test_index_built_on_other_chunks_is_rejected(tmp_path):
    chunks = make_chunks()
    build_dense_index(chunks, EncoderConfig(name="lsa-4", backend="lsa", dim=4), tmp_path / "idx")
    changed = chunks[:-1] + [Chunk(chunks[-1].chunk_id, "doc-c", 0, 5, "другой", "t", [], "u")]
    with pytest.raises(ValueError):
        load_dense(changed, tmp_path / "idx")


def test_documents_from_chunks_keeps_best_position_per_document():
    chunks = make_chunks()
    hits = [ChunkHit(chunks[i], 1.0 - i / 10, rank) for rank, i in enumerate([0, 1, 3, 2, 4], start=1)]
    docs = documents_from_chunks(hits, top_k=10)
    assert [d.doc_id for d in docs] == ["doc-a", "doc-c", "doc-b"]
    assert [d.rank for d in docs] == [1, 2, 3]
    assert documents_from_chunks(hits, top_k=2)[-1].doc_id == "doc-c"
