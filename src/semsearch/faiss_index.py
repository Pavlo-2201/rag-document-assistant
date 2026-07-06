"""Индекс FAISS: сборка, сохранение, загрузка и соответствие ID.

FAISS работает с позициями векторов (0..N-1), а корпус — с doc_id. Соответствие
хранится рядом с индексом в `meta.json`, вместе с конфигурацией модели и
отпечатком корпуса. Это важно: индекс, собранный на другом корпусе или другой
моделью, молча вернёт неверные документы, поэтому при загрузке проверяется и то,
и другое.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

INDEX_FILENAME = "index.faiss"
META_FILENAME = "meta.json"

#: Поддерживаемые типы индекса. flat-ip — точный перебор, остальные приближённые.
INDEX_TYPES = ("flat-ip", "ivf-flat", "hnsw")


@dataclass
class IndexSpec:
    """Описание индекса. Параметры приближённых индексов собраны здесь же."""

    index_type: str = "flat-ip"
    nlist: int = 128        # ivf-flat: число кластеров
    nprobe: int = 8         # ivf-flat: сколько кластеров просматривать при поиске
    hnsw_m: int = 32        # hnsw: число соседей
    ef_construction: int = 200
    ef_search: int = 64

    def to_dict(self) -> dict:
        return {
            "index_type": self.index_type,
            "nlist": self.nlist,
            "nprobe": self.nprobe,
            "hnsw_m": self.hnsw_m,
            "ef_construction": self.ef_construction,
            "ef_search": self.ef_search,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "IndexSpec":
        known = {f: data[f] for f in cls.__dataclass_fields__ if f in data}
        return cls(**known)

    @property
    def is_exact(self) -> bool:
        return self.index_type == "flat-ip"


def _build_faiss_index(spec: IndexSpec, dim: int, n_vectors: int):
    import faiss

    if spec.index_type == "flat-ip":
        # векторы нормированы, поэтому внутреннее произведение = косинус
        return faiss.IndexFlatIP(dim)
    if spec.index_type == "ivf-flat":
        # кластеров не может быть больше, чем векторов; FAISS просит хотя бы
        # несколько десятков точек на кластер, иначе обучение бессмысленно
        nlist = max(1, min(spec.nlist, max(1, n_vectors // 39)))
        quantizer = faiss.IndexFlatIP(dim)
        index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
        index.nprobe = min(spec.nprobe, nlist)
        return index
    if spec.index_type == "hnsw":
        index = faiss.IndexHNSWFlat(dim, spec.hnsw_m, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = spec.ef_construction
        index.hnsw.efSearch = spec.ef_search
        return index
    raise ValueError(f"неизвестный тип индекса: {spec.index_type}; есть {INDEX_TYPES}")


@dataclass
class VectorIndex:
    """Индекс FAISS вместе с соответствием позиция -> doc_id."""

    doc_ids: list[str]
    spec: IndexSpec
    dim: int
    model: dict = field(default_factory=dict)
    corpus_fingerprint: str = ""
    corpus_path: str = ""
    index: object = None

    def __post_init__(self) -> None:
        self._position_by_doc_id = {doc_id: i for i, doc_id in enumerate(self.doc_ids)}
        if len(self._position_by_doc_id) != len(self.doc_ids):
            raise ValueError("в индексе повторяются doc_id")

    @property
    def size(self) -> int:
        return len(self.doc_ids)

    def position_of(self, doc_id: str) -> int:
        return self._position_by_doc_id[doc_id]

    @classmethod
    def build(
        cls,
        embeddings: np.ndarray,
        doc_ids: Sequence[str],
        spec: IndexSpec | None = None,
        model: dict | None = None,
        corpus_fingerprint: str = "",
        corpus_path: str = "",
    ) -> "VectorIndex":
        spec = spec or IndexSpec()
        embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
        if embeddings.shape[0] != len(doc_ids):
            raise ValueError("число векторов не совпадает с числом doc_id")
        index = _build_faiss_index(spec, embeddings.shape[1], embeddings.shape[0])
        if not index.is_trained:
            index.train(embeddings)
        # векторы добавляются в том же порядке, что и doc_ids: позиция в индексе
        # равна позиции в списке
        index.add(embeddings)
        if index.ntotal != len(doc_ids):
            raise RuntimeError("в индекс попало не столько векторов, сколько документов")
        return cls(
            doc_ids=list(doc_ids),
            spec=spec,
            dim=int(embeddings.shape[1]),
            model=dict(model or {}),
            corpus_fingerprint=corpus_fingerprint,
            corpus_path=corpus_path,
            index=index,
        )

    def search(self, query_vectors: np.ndarray, top_k: int) -> tuple[np.ndarray, list[list[str]]]:
        """Вернуть оценки и doc_id для каждого запроса.

        FAISS возвращает -1 там, где результатов меньше, чем top_k; такие позиции
        отбрасываются, а не превращаются в случайный документ.
        """
        query_vectors = np.ascontiguousarray(query_vectors, dtype=np.float32)
        if query_vectors.shape[1] != self.dim:
            raise ValueError(
                f"размерность запроса {query_vectors.shape[1]} не совпадает с индексом {self.dim}"
            )
        scores, positions = self.index.search(query_vectors, min(top_k, self.size))
        doc_ids = [
            [self.doc_ids[position] for position in row if position != -1] for row in positions
        ]
        return scores, doc_ids

    def save(self, directory: str | Path) -> Path:
        import faiss

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(directory / INDEX_FILENAME))
        meta = {
            "dim": self.dim,
            "size": self.size,
            "spec": self.spec.to_dict(),
            "model": self.model,
            "corpus_fingerprint": self.corpus_fingerprint,
            "corpus_path": self.corpus_path,
            "doc_ids": self.doc_ids,
        }
        (directory / META_FILENAME).write_text(
            json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        return directory

    @classmethod
    def load(cls, directory: str | Path) -> "VectorIndex":
        import faiss

        directory = Path(directory)
        meta = json.loads((directory / META_FILENAME).read_text(encoding="utf-8"))
        index = faiss.read_index(str(directory / INDEX_FILENAME))
        loaded = cls(
            doc_ids=list(meta["doc_ids"]),
            spec=IndexSpec.from_dict(meta.get("spec", {})),
            dim=int(meta["dim"]),
            model=meta.get("model", {}),
            corpus_fingerprint=meta.get("corpus_fingerprint", ""),
            corpus_path=meta.get("corpus_path", ""),
            index=index,
        )
        if index.ntotal != loaded.size:
            raise ValueError(
                f"в индексе {index.ntotal} векторов, а в соответствии ID — {loaded.size}"
            )
        if loaded.spec.index_type == "ivf-flat":
            # nprobe не сохраняется внутри файла индекса во всех версиях FAISS
            index.nprobe = loaded.spec.nprobe
        elif loaded.spec.index_type == "hnsw":
            index.hnsw.efSearch = loaded.spec.ef_search
        return loaded

    def check_matches_corpus(self, doc_ids: Sequence[str], fingerprint: str = "") -> None:
        """Проверить, что индекс собран на этом корпусе."""
        if list(doc_ids) != self.doc_ids:
            missing = set(self.doc_ids) ^ set(doc_ids)
            raise ValueError(
                "индекс собран на другом наборе документов "
                f"(расхождение в {len(missing)} doc_id)"
            )
        if fingerprint and self.corpus_fingerprint and fingerprint != self.corpus_fingerprint:
            raise ValueError(
                "отпечаток корпуса не совпадает с отпечатком при сборке индекса: "
                "корпус изменился, индекс нужно пересобрать"
            )
