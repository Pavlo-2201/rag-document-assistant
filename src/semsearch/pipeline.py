"""Сборка поиска по имени: одно место, где лексический и векторный поиск
создаются одинаково — для CLI, оценки и сравнения моделей.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from .corpus import Document, corpus_fingerprint
from .dense import ExactDenseRetriever, FaissDenseRetriever, encode_corpus
from .embedding import (
    TRAINABLE_BACKENDS,
    EncoderConfig,
    build_document_text,
    build_encoder,
    load_model_configs,
)
from .faiss_index import IndexSpec, VectorIndex
from .lexical import build_lexical_retriever

LEXICAL_KINDS = ("bm25", "tfidf")
DEFAULT_MODELS_CONFIG = "configs/models.yaml"
ENCODER_FILENAME = "encoder.pkl"


def resolve_model_config(
    name: str, models_config: str | Path = DEFAULT_MODELS_CONFIG
) -> EncoderConfig:
    configs = load_model_configs(models_config)
    if name not in configs:
        raise ValueError(f"модель '{name}' не найдена в {models_config}; есть {sorted(configs)}")
    return configs[name]


def corpus_texts_for(config: EncoderConfig, documents: Sequence[Document]) -> list[str]:
    return [build_document_text(d.title, d.text, config.document_max_chars) for d in documents]


def build_encoder_for_corpus(config: EncoderConfig, documents: Sequence[Document]):
    """Создать кодировщик; обучаемые на корпусе бэкенды обучаются на документах."""
    return build_encoder(config, corpus_texts_for(config, documents))


def save_encoder(encoder, directory: str | Path) -> None:
    """Сохранить кодировщик рядом с индексом, если он этого требует.

    Предобученная модель sentence-transformers восстанавливается по имени из
    конфигурации, а обученную на корпусе (lsa, word2vec) нужно сохранить: иначе
    запрос будет закодирован другой моделью, чем документы в индексе.
    """
    if encoder.config.backend in TRAINABLE_BACKENDS:
        encoder.save(Path(directory) / ENCODER_FILENAME)


def load_encoder(config: EncoderConfig, directory: str | Path):
    if config.backend in TRAINABLE_BACKENDS:
        path = Path(directory) / ENCODER_FILENAME
        if not path.exists():
            raise FileNotFoundError(
                f"рядом с индексом нет {ENCODER_FILENAME}: модель {config.backend} обучается "
                "на корпусе и должна храниться вместе с индексом"
            )
        return TRAINABLE_BACKENDS[config.backend].load(path)
    return build_encoder(config)


def build_index(
    documents: Sequence[Document],
    config: EncoderConfig,
    spec: IndexSpec | None = None,
    corpus_path: str = "",
    progress: bool = False,
) -> tuple[VectorIndex, object]:
    """Закодировать корпус и собрать индекс. Возвращает индекс и кодировщик."""
    encoder = build_encoder_for_corpus(config, documents)
    embeddings = encode_corpus(encoder, documents, progress=progress)
    index = VectorIndex.build(
        embeddings,
        [d.doc_id for d in documents],
        spec=spec,
        model=config.to_dict(),
        corpus_fingerprint=corpus_fingerprint(documents),
        corpus_path=corpus_path,
    )
    return index, encoder


def load_index_retriever(
    documents: Sequence[Document],
    directory: str | Path,
    models_config: str | Path = DEFAULT_MODELS_CONFIG,
    check_corpus: bool = True,
) -> FaissDenseRetriever:
    index = VectorIndex.load(directory)
    model_name = index.model.get("name")
    if not model_name:
        raise ValueError("в метаданных индекса не записана модель")
    try:
        config = resolve_model_config(model_name, models_config)
    except ValueError:
        # индекс самодостаточен: если модели уже нет в конфигурации, берём запись
        # из метаданных индекса
        config = EncoderConfig.from_dict(model_name, index.model)
    if check_corpus:
        index.check_matches_corpus(
            [d.doc_id for d in documents], fingerprint=corpus_fingerprint(documents)
        )
    encoder = load_encoder(config, directory)
    return FaissDenseRetriever(documents, encoder, index)


def build_retriever(
    kind: str,
    documents: Sequence[Document],
    model: str | None = None,
    models_config: str | Path = DEFAULT_MODELS_CONFIG,
    index_dir: str | Path | None = None,
    spec: IndexSpec | None = None,
    progress: bool = False,
):
    """`kind` — bm25, tfidf, dense (точный перебор) или faiss.

    Для dense нужна модель из configs/models.yaml, для faiss — каталог индекса.
    """
    if kind in LEXICAL_KINDS:
        return build_lexical_retriever(kind, documents)
    if kind == "dense":
        if not model:
            raise ValueError("для dense нужно указать модель из configs/models.yaml")
        config = resolve_model_config(model, models_config)
        encoder = build_encoder_for_corpus(config, documents)
        embeddings = encode_corpus(encoder, documents, progress=progress)
        return ExactDenseRetriever(documents, encoder, embeddings=embeddings)
    if kind == "faiss":
        if index_dir is None:
            raise ValueError("для faiss нужен каталог с индексом (--index)")
        return load_index_retriever(documents, index_dir, models_config=models_config)
    raise ValueError(f"неизвестный способ поиска: {kind}")


def retriever_label(kind: str, model: str | None) -> str:
    """Короткое имя прогона для отчётов и имён файлов."""
    return f"{kind}:{model}" if model and kind in ("dense", "faiss") else kind
