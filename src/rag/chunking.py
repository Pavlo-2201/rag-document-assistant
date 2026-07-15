"""Нарезка документов на фрагменты с перекрытием и привязкой к источнику.

Правила:

* фрагмент не пересекает границу раздела, если раздел помещается целиком;
  короткие соседние разделы склеиваются, пока помещаются в `chunk_size`
  (иначе получаются фрагменты из одного заголовка «Спецификации»);
* длинный раздел режется окнами по `chunk_size` символов с перекрытием
  `overlap`; граница окна сдвигается назад к концу абзаца, строки или слова,
  чтобы не резать слово пополам;
* фрагмент — это всегда точный отрезок текста документа: `document.text[start:end]`.
  По (doc_id, start, end) фрагмент однозначно возвращается к источнику, а
  evidence span из разметки проверяется простым сравнением отрезков.

Размер в символах, а не в токенах: так нарезка не зависит от модели, а лимит
контекста LLM проверяется отдельно, при сборке промпта, токенизатором самой
модели.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

from .ingest import SourceDocument

MIN_CONTENT_CHARS = 40  # фрагмент короче (без заголовка) приклеивается к соседу


@dataclass(frozen=True)
class ChunkingConfig:
    chunk_size: int = 1000
    overlap: int = 150

    def __post_init__(self) -> None:
        if self.chunk_size <= 0 or not 0 <= self.overlap < self.chunk_size:
            raise ValueError("нужно 0 <= overlap < chunk_size")

    @property
    def name(self) -> str:
        return f"c{self.chunk_size}-o{self.overlap}"


@dataclass
class Chunk:
    chunk_id: str        # "<doc_id>:<start>-<end>"
    doc_id: str
    start: int           # границы в SourceDocument.text
    end: int
    text: str
    title: str           # заголовок документа
    section_path: list[str]
    url: str             # ссылка на раздел, в котором начинается фрагмент

    @property
    def header(self) -> str:
        """Контекст фрагмента: документ и раздел. Добавляется при кодировании и в промпт."""
        path = " > ".join(p for p in self.section_path if p)
        return f"{self.title} — {path}" if path else self.title

    def embedding_text(self) -> str:
        return f"{self.header}\n{self.text}"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Chunk":
        return cls(**data)


def _break_position(text: str, start: int, limit: int) -> int:
    """Где закончить окно [start, limit): конец абзаца, строки, предложения или слова.

    Ищется во второй половине окна, чтобы фрагменты не становились слишком короткими.
    """
    floor = start + (limit - start) // 2
    for separator in ("\n\n", "\n", ". ", " "):
        position = text.rfind(separator, floor, limit)
        if position != -1:
            return position + len(separator)
    return limit


def _next_start(text: str, end: int, overlap: int, floor: int) -> int:
    """Начало следующего окна: на `overlap` символов раньше конца, по границе слова."""
    if overlap == 0:
        return end
    start = max(end - overlap, floor)
    if start > 0 and not text[start - 1].isspace():
        space = text.find(" ", start, end)
        newline = text.find("\n", start, end)
        candidates = [p for p in (space, newline) if p != -1]
        start = min(candidates) + 1 if candidates else start
    return start if start < end else end


def _split_span(text: str, start: int, end: int, config: ChunkingConfig) -> list[tuple[int, int]]:
    if end - start <= config.chunk_size:
        return [(start, end)]
    spans = []
    position = start
    while position < end:
        limit = min(position + config.chunk_size, end)
        stop = end if limit == end else _break_position(text, position, limit)
        spans.append((position, stop))
        if stop >= end:
            break
        following = _next_start(text, stop, config.overlap, position + 1)
        position = following if following > position else stop
    return spans


def chunk_document(document: SourceDocument, config: ChunkingConfig) -> list[Chunk]:
    text = document.text
    # 1. склеить короткие соседние разделы
    groups: list[tuple[int, int, int]] = []  # (start, end, индекс первого раздела)
    for index, section in enumerate(document.sections):
        if section.end <= section.start:
            continue
        if groups:
            g_start, g_end, g_first = groups[-1]
            fits = section.end - g_start <= config.chunk_size
            tiny = len(text[g_start:g_end].strip()) - len(document.sections[g_first].title) < MIN_CONTENT_CHARS
            if fits or tiny:
                groups[-1] = (g_start, section.end, g_first)
                continue
        groups.append((section.start, section.end, index))

    # 2. длинные группы нарезать окнами с перекрытием
    chunks = []
    for g_start, g_end, _ in groups:
        for start, end in _split_span(text, g_start, g_end, config):
            # пробельные символы по краям в границы не входят
            while start < end and text[start].isspace():
                start += 1
            while end > start and text[end - 1].isspace():
                end -= 1
            if end <= start:
                continue
            section = _section_at(document, start)
            chunks.append(
                Chunk(
                    chunk_id=f"{document.doc_id}:{start}-{end}",
                    doc_id=document.doc_id,
                    start=start,
                    end=end,
                    text=text[start:end],
                    title=document.title,
                    section_path=list(section.path),
                    url=document.section_url(section),
                )
            )
    return chunks


def _section_at(document: SourceDocument, offset: int):
    for section in document.sections:
        if section.start <= offset < section.end:
            return section
    return document.sections[-1]


def chunk_documents(documents: Iterable[SourceDocument], config: ChunkingConfig) -> list[Chunk]:
    chunks = []
    for document in sorted(documents, key=lambda d: d.doc_id):
        chunks.extend(chunk_document(document, config))
    return chunks


def chunks_fingerprint(chunks: Sequence[Chunk]) -> str:
    digest = hashlib.sha256()
    for chunk in chunks:
        digest.update(f"{chunk.chunk_id}\0{hashlib.sha256(chunk.text.encode()).hexdigest()}\n".encode())
    return digest.hexdigest()


def write_chunks(path: str | Path, chunks: Sequence[Chunk]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")


def load_chunks(path: str | Path) -> list[Chunk]:
    with Path(path).open(encoding="utf-8") as handle:
        return [Chunk.from_dict(json.loads(line)) for line in handle if line.strip()]
