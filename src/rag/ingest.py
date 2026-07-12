"""Ingest: исходные файлы → документы с метаданными и устойчивыми ссылками.

Поддерживаются два входа:

* корпус MDN, загруженный `scripts/download_corpus.py`: Markdown-файлы плюс
  `manifest.jsonl` со slug'ом, адресом страницы и sha256 исходника;
* произвольный каталог с `.md` и `.txt` (например, собственные документы).

Идентификатор документа устойчив: для MDN это `make_doc_id(slug)` из
semantic-search-engine (тот же doc_id, что в его индексах и разметке), для
каталога — хеш относительного пути. Ни порядок загрузки, ни правки текста
идентификатор не меняют; изменение текста видно по `text_sha256`.

PDF не поддерживается сознательно: в корпусе MDN PDF нет, а добавлять
разбор формата без реальной необходимости — лишняя зависимость.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

from semsearch.corpus import make_doc_id

from .parser import ParsedDocument, Section, parse_markdown, parse_plain_text

SUPPORTED_SUFFIXES = {".md": "markdown", ".markdown": "markdown", ".txt": "text"}


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass
class SourceDocument:
    doc_id: str
    title: str
    url: str                # адрес, который показывается пользователю как источник
    source_path: str        # путь исходного файла (относительно корня источника)
    format: str             # markdown | text
    text: str
    sections: list[Section]
    raw_sha256: str         # хеш исходного файла: изменился ли источник
    text_sha256: str        # хеш разобранного текста: изменился ли результат разбора
    license: str = ""
    source_ref: str = ""
    slug: str = ""
    meta: dict = field(default_factory=dict)

    @property
    def char_len(self) -> int:
        return len(self.text)

    def section_url(self, section: Section) -> str:
        """Устойчивая ссылка на раздел: адрес страницы и якорь заголовка."""
        return f"{self.url}#{section.anchor}" if section.anchor else self.url

    def to_dict(self) -> dict:
        data = asdict(self)
        data["sections"] = [s.to_dict() for s in self.sections]
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "SourceDocument":
        data = dict(data)
        data["sections"] = [Section(**s) for s in data["sections"]]
        return cls(**data)


def _from_parsed(parsed: ParsedDocument, **kwargs) -> SourceDocument:
    return SourceDocument(
        title=parsed.title,
        text=parsed.text,
        sections=parsed.sections,
        text_sha256=sha256_text(parsed.text),
        **kwargs,
    )


def ingest_mdn(raw_dir: str | Path, source: dict) -> list[SourceDocument]:
    """Документы корпуса MDN по manifest.jsonl, который пишет download_corpus.py."""
    raw_dir = Path(raw_dir)
    documents = []
    with (raw_dir / "manifest.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            markdown = (raw_dir / row["source_path"]).read_text(encoding="utf-8")
            if sha256_text(markdown) != row["raw_sha256"]:
                raise ValueError(f"файл изменился после загрузки: {row['source_path']}")
            default_title = row["slug"].rsplit("/", 1)[-1].replace("_", " ")
            documents.append(
                _from_parsed(
                    parse_markdown(markdown, default_title=default_title),
                    doc_id=make_doc_id(row["slug"]),
                    url=row["url"],
                    source_path=row["source_path"],
                    format="markdown",
                    raw_sha256=row["raw_sha256"],
                    license=source.get("license", ""),
                    source_ref=f"{source['repo']}@{source['ref']}",
                    slug=row["slug"],
                )
            )
    return documents


def make_path_doc_id(relative_path: str) -> str:
    return "doc-" + hashlib.sha1(relative_path.encode("utf-8")).hexdigest()[:12]


def ingest_directory(root: str | Path, base_url: str | None = None, license: str = "") -> list[SourceDocument]:
    """Все .md/.txt файлы каталога. Ссылка — base_url + путь или file:// путь."""
    root = Path(root)
    documents = []
    for path in sorted(root.rglob("*")):
        fmt = SUPPORTED_SUFFIXES.get(path.suffix.lower())
        if not path.is_file() or fmt is None:
            continue
        relative = path.relative_to(root).as_posix()
        raw = path.read_text(encoding="utf-8")
        default_title = path.stem.replace("_", " ").replace("-", " ")
        parsed = parse_markdown(raw, default_title) if fmt == "markdown" else parse_plain_text(raw, default_title)
        url = f"{base_url.rstrip('/')}/{relative}" if base_url else path.resolve().as_uri()
        documents.append(
            _from_parsed(
                parsed,
                doc_id=make_path_doc_id(relative),
                url=url,
                source_path=relative,
                format=fmt,
                raw_sha256=sha256_text(raw),
                license=license,
            )
        )
    return documents


def write_documents(path: str | Path, documents: Iterable[SourceDocument]) -> int:
    """JSONL, отсортированный по doc_id: одинаковый вход даёт одинаковый файл."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    documents = sorted(documents, key=lambda d: d.doc_id)
    ids = [d.doc_id for d in documents]
    if len(ids) != len(set(ids)):
        raise ValueError("повторяющиеся doc_id")
    with path.open("w", encoding="utf-8") as handle:
        for document in documents:
            handle.write(json.dumps(document.to_dict(), ensure_ascii=False) + "\n")
    return len(documents)


def load_documents(path: str | Path) -> list[SourceDocument]:
    with Path(path).open(encoding="utf-8") as handle:
        return [SourceDocument.from_dict(json.loads(line)) for line in handle if line.strip()]


def documents_fingerprint(documents: Iterable[SourceDocument]) -> str:
    """Отпечаток набора разобранных документов: sha256 пар (doc_id, text_sha256)."""
    digest = hashlib.sha256()
    for doc_id, checksum in sorted((d.doc_id, d.text_sha256) for d in documents):
        digest.update(f"{doc_id}\0{checksum}\n".encode("utf-8"))
    return digest.hexdigest()
