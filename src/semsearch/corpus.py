"""Документ корпуса, устойчивые идентификаторы, чтение и запись JSONL."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Iterator

DOC_ID_PREFIX = "mdnru"
DOC_ID_HASH_CHARS = 12


def make_doc_id(slug: str) -> str:
    """Устойчивый идентификатор документа.

    Идентификатор считается от slug'а страницы, а не от позиции в файле и не от
    порядка загрузки: при повторной загрузке корпуса или при добавлении новых
    страниц идентификаторы уже существующих документов не меняются. Сам slug
    тоже хранится в документе, он нужен человеку, а короткий хеш — индексам.
    """
    digest = hashlib.sha1(slug.encode("utf-8")).hexdigest()[:DOC_ID_HASH_CHARS]
    return f"{DOC_ID_PREFIX}-{digest}"


def text_checksum(text: str) -> str:
    """sha256 текста документа: позволяет заметить, что источник изменился."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass
class Document:
    doc_id: str
    slug: str
    title: str
    text: str
    url: str
    source_path: str
    source_repo: str
    source_ref: str
    license: str
    checksum: str
    char_len: int
    extra: dict = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        slug: str,
        title: str,
        text: str,
        url: str,
        source_path: str,
        source_repo: str,
        source_ref: str,
        license: str,
        **extra: object,
    ) -> "Document":
        return cls(
            doc_id=make_doc_id(slug),
            slug=slug,
            title=title,
            text=text,
            url=url,
            source_path=source_path,
            source_repo=source_repo,
            source_ref=source_ref,
            license=license,
            checksum=text_checksum(text),
            char_len=len(text),
            extra=dict(extra),
        )

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: dict) -> "Document":
        known = {f: data[f] for f in cls.__dataclass_fields__ if f in data}
        known.setdefault("extra", {})
        return cls(**known)


def write_corpus(path: str | Path, documents: Iterable[Document]) -> int:
    """Записать корпус в JSONL, отсортировав документы по doc_id.

    Порядок фиксирован, поэтому один и тот же набор страниц даёт один и тот же
    файл, а diff между двумя загрузками читаем.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    docs = sorted(documents, key=lambda d: d.doc_id)
    with path.open("w", encoding="utf-8") as handle:
        for doc in docs:
            handle.write(doc.to_json())
            handle.write("\n")
    return len(docs)


def iter_corpus(path: str | Path) -> Iterator[Document]:
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield Document.from_dict(json.loads(line))


def load_corpus(path: str | Path) -> list[Document]:
    documents = list(iter_corpus(path))
    seen: set[str] = set()
    for doc in documents:
        if doc.doc_id in seen:
            raise ValueError(f"повторяющийся doc_id в корпусе: {doc.doc_id}")
        seen.add(doc.doc_id)
    return documents


def corpus_fingerprint(documents: Iterable[Document]) -> str:
    """Отпечаток корпуса: sha256 от отсортированных пар (doc_id, checksum).

    Источник (ветка main репозитория MDN) со временем меняется, поэтому вместе с
    корпусом сохраняется отпечаток: по нему видно, что две загрузки дали
    одинаковый или разный корпус. Эксперименты сравнимы только при одинаковом
    отпечатке.
    """
    digest = hashlib.sha256()
    for doc_id, checksum in sorted((d.doc_id, d.checksum) for d in documents):
        digest.update(doc_id.encode("utf-8"))
        digest.update(b"\0")
        digest.update(checksum.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()
