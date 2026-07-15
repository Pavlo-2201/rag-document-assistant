import re

import pytest

from rag.chunking import ChunkingConfig, chunk_document, chunk_documents, load_chunks, write_chunks
from rag.ingest import SourceDocument, sha256_text
from rag.parser import parse_markdown

LONG_SECTION = " ".join(f"Предложение номер {i} про массивы и строки." for i in range(120))
MARKDOWN = f"""---
title: Тест
---

Вступление.

## Короткий

Пара слов.

## Длинный

{LONG_SECTION}

## Спецификации

## Смотрите также

- ссылка
"""


def make_document(markdown: str = MARKDOWN, doc_id: str = "doc-1") -> SourceDocument:
    parsed = parse_markdown(markdown)
    return SourceDocument(
        doc_id=doc_id, title=parsed.title, url="https://example.org/page", source_path="p.md",
        format="markdown", text=parsed.text, sections=parsed.sections,
        raw_sha256=sha256_text(markdown), text_sha256=sha256_text(parsed.text),
    )


@pytest.mark.parametrize("size,overlap", [(200, 0), (300, 50), (1000, 150), (5000, 300)])
def test_chunk_is_exact_slice_of_document_and_within_size(size, overlap):
    document = make_document()
    for chunk in chunk_document(document, ChunkingConfig(size, overlap)):
        assert chunk.text == document.text[chunk.start:chunk.end]
        assert chunk.chunk_id == f"{document.doc_id}:{chunk.start}-{chunk.end}"
        assert len(chunk.text) <= size
        assert chunk.text == chunk.text.strip()


@pytest.mark.parametrize("size,overlap", [(200, 0), (300, 50), (1000, 150)])
def test_chunks_cover_every_word_of_document(size, overlap):
    document = make_document()
    covered = [False] * len(document.text)
    for chunk in chunk_document(document, ChunkingConfig(size, overlap)):
        for i in range(chunk.start, chunk.end):
            covered[i] = True
    missing = [m.group() for m in re.finditer(r"\S+", document.text) if not all(covered[m.start():m.end()])]
    assert missing == []


def test_overlap_between_windows_of_long_section():
    document = make_document()
    chunks = [c for c in chunk_document(document, ChunkingConfig(300, 60)) if "Длинный" in c.section_path]
    assert len(chunks) > 3
    for left, right in zip(chunks, chunks[1:]):
        shared = left.end - right.start
        assert 0 < shared <= 60
        # перекрытие начинается с начала слова
        assert document.text[right.start - 1].isspace()


def test_no_overlap_when_zero():
    chunks = chunk_document(make_document(), ChunkingConfig(300, 0))
    for left, right in zip(chunks, chunks[1:]):
        assert right.start >= left.end


def test_short_sections_are_merged_and_heading_only_chunks_do_not_exist():
    chunks = chunk_document(make_document(), ChunkingConfig(300, 0))
    assert chunks[0].text.startswith("Вступление.") and "Пара слов." in chunks[0].text
    for chunk in chunks:
        assert chunk.text not in {"Спецификации", "Смотрите также"}


def test_chunk_keeps_link_to_section_and_header():
    chunks = chunk_document(make_document(), ChunkingConfig(300, 50))
    long_chunk = next(c for c in chunks if "Длинный" in c.section_path)
    assert long_chunk.url == "https://example.org/page#длинный"
    assert long_chunk.header == "Тест — Длинный"
    assert long_chunk.embedding_text().startswith("Тест — Длинный\n")


def test_chunking_is_deterministic_and_roundtrips(tmp_path):
    docs = [make_document(doc_id="doc-b"), make_document(doc_id="doc-a")]
    first = chunk_documents(docs, ChunkingConfig(400, 80))
    second = chunk_documents(list(reversed(docs)), ChunkingConfig(400, 80))
    assert [c.to_dict() for c in first] == [c.to_dict() for c in second]
    write_chunks(tmp_path / "c.jsonl", first)
    assert [c.to_dict() for c in load_chunks(tmp_path / "c.jsonl")] == [c.to_dict() for c in first]


def test_invalid_config():
    with pytest.raises(ValueError):
        ChunkingConfig(100, 100)
