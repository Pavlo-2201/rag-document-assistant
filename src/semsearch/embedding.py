"""Кодирование текста в векторы.

Поддерживаются два бэкенда:

* `sentence-transformers` — предобученные многоязычные модели. Требования модели
  (префиксы запроса и документа, способ пулинга, нормализация) задаются в
  конфигурации и проверяются по загруженной модели, а не угадываются.
* `lsa` — латентно-семантический анализ: TF-IDF и усечённое SVD, обучается на
  самом корпусе. Бэкенд нужен там, где предобученную модель скачать нельзя, и
  как честная нижняя граница для векторного поиска.

Запросы и документы кодируются разными методами: у многих моделей (например,
E5) для них предусмотрены разные префиксы, и смешивать их нельзя.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Protocol, Sequence

import numpy as np

# Документ кодируется как заголовок и начало текста: трансформеры всё равно
# обрезают вход (у E5 это 512 токенов), а начало статьи MDN описывает её суть.
DEFAULT_DOCUMENT_MAX_CHARS = 2000


@dataclass(frozen=True)
class EncoderConfig:
    name: str
    backend: str
    model_name: str | None = None
    #: ревизия модели на Hugging Face; фиксирует веса так же, как commit sha данные
    revision: str | None = None
    #: каталог с уже скачанными весами; нужен там, где нет доступа к huggingface.co
    local_path: str | None = None
    query_prefix: str = ""
    document_prefix: str = ""
    pooling: str = "mean"          # mean | cls — проверяется по загруженной модели
    normalize: bool = True
    max_seq_length: int | None = None
    batch_size: int = 16
    dim: int | None = None         # число измерений для lsa
    document_max_chars: int = DEFAULT_DOCUMENT_MAX_CHARS
    notes: str = ""
    extra: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, name: str, data: dict) -> "EncoderConfig":
        known = {f for f in cls.__dataclass_fields__ if f != "extra"}
        kwargs = {k: v for k, v in data.items() if k in known}
        kwargs["name"] = data.get("name", name)
        kwargs["extra"] = {k: v for k, v in data.items() if k not in known}
        return cls(**kwargs)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "backend": self.backend,
            "model_name": self.model_name,
            "revision": self.revision,
            "local_path": self.local_path,
            "query_prefix": self.query_prefix,
            "document_prefix": self.document_prefix,
            "pooling": self.pooling,
            "normalize": self.normalize,
            "max_seq_length": self.max_seq_length,
            "batch_size": self.batch_size,
            "dim": self.dim,
            "document_max_chars": self.document_max_chars,
        }


def build_document_text(title: str, text: str, max_chars: int = DEFAULT_DOCUMENT_MAX_CHARS) -> str:
    body = " ".join(text.split())
    if max_chars and len(body) > max_chars:
        body = body[:max_chars].rsplit(" ", 1)[0]
    return f"{title}. {body}" if title else body


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return (matrix / norms).astype(np.float32)


class TextEncoder(Protocol):
    config: EncoderConfig

    @property
    def dim(self) -> int:
        ...

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray:
        ...

    def encode_documents(self, texts: Sequence[str]) -> np.ndarray:
        ...


def local_weights_path(config: EncoderConfig) -> Path | None:
    """Каталог с локальными весами, если он задан и существует."""
    if not config.local_path:
        return None
    path = Path(config.local_path)
    return path if path.exists() else None


def sentence_transformer_kwargs(config: EncoderConfig) -> dict:
    """Откуда грузить модель: локальный каталог или Hugging Face с фиксированной ревизией."""
    local = local_weights_path(config)
    if local is not None:
        # веса уже скачаны: сеть не нужна, ревизию задаёт сам каталог
        return {"model_name_or_path": str(local), "local_files_only": True}
    if not config.model_name:
        raise ValueError("для бэкенда sentence-transformers нужен model_name или local_path")
    kwargs = {"model_name_or_path": config.model_name}
    if config.revision:
        kwargs["revision"] = config.revision
    return kwargs


class SentenceTransformerEncoder:
    """Обёртка над sentence-transformers.

    Что делается по требованиям модели:

    * префиксы берутся из конфигурации и приписываются к запросам и документам
      раздельно (у E5 это «query: » и «passage: », пропуск префиксов заметно
      портит качество);
    * способ пулинга не задаётся руками, а читается из загруженной модели и
      сверяется с конфигурацией — расхождение это ошибка конфигурации;
    * нормализация включается флагом; для косинусной близости и для FAISS с
      внутренним произведением она обязательна.
    """

    def __init__(self, config: EncoderConfig, model=None):
        from sentence_transformers import SentenceTransformer

        self.config = config
        if model is None:
            model = SentenceTransformer(**sentence_transformer_kwargs(config))
        self.model = model
        if config.max_seq_length:
            self.model.max_seq_length = config.max_seq_length
        self._check_pooling()

    def _check_pooling(self) -> None:
        pooling_module = None
        for module in self.model._modules.values():
            if type(module).__name__ == "Pooling":
                pooling_module = module
        if pooling_module is None:
            return
        # в sentence-transformers 6 это строка pooling_mode, в более старых —
        # набор флагов pooling_mode_*_tokens
        actual = getattr(pooling_module, "pooling_mode", None)
        if actual is None:
            if getattr(pooling_module, "pooling_mode_mean_tokens", False):
                actual = "mean"
            elif getattr(pooling_module, "pooling_mode_cls_token", False):
                actual = "cls"
            else:
                actual = "other"
        if actual != self.config.pooling:
            raise ValueError(
                f"модель {self.config.model_name} использует pooling '{actual}', "
                f"а в конфигурации указан '{self.config.pooling}'"
            )

    @property
    def dim(self) -> int:
        return int(self.model.get_sentence_embedding_dimension())

    def _encode(self, texts: Sequence[str], prefix: str) -> np.ndarray:
        prepared = [prefix + text for text in texts]
        vectors = self.model.encode(
            prepared,
            batch_size=self.config.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=self.config.normalize,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode(texts, self.config.query_prefix)

    def encode_documents(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode(texts, self.config.document_prefix)


class LsaEncoder:
    """TF-IDF и усечённое SVD: векторная модель, обучаемая на корпусе.

    Обучается только на текстах документов. Запросы в обучении не участвуют,
    поэтому разметка запросов не протекает в модель.
    """

    def __init__(self, config: EncoderConfig):
        self.config = config
        self._vectorizer = None
        self._svd = None

    @property
    def dim(self) -> int:
        if self._svd is None:
            raise RuntimeError("модель lsa не обучена")
        return int(self._svd.n_components)

    def fit(self, document_texts: Sequence[str]) -> "LsaEncoder":
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer

        from .text import analyze

        n_components = int(self.config.dim or 300)
        # число измерений не может превышать число документов
        n_components = min(n_components, max(len(document_texts) - 1, 1))
        self._vectorizer = TfidfVectorizer(analyzer=analyze, min_df=1, sublinear_tf=True)
        matrix = self._vectorizer.fit_transform(document_texts)
        n_components = min(n_components, matrix.shape[1] - 1 if matrix.shape[1] > 1 else 1)
        self._svd = TruncatedSVD(n_components=n_components, random_state=42)
        self._svd.fit(matrix)
        return self

    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        if self._vectorizer is None or self._svd is None:
            raise RuntimeError("модель lsa не обучена: вызовите fit или load")
        vectors = self._svd.transform(self._vectorizer.transform(list(texts)))
        vectors = np.asarray(vectors, dtype=np.float32)
        return l2_normalize(vectors) if self.config.normalize else vectors

    # у LSA нет разных пространств для запроса и документа, но интерфейс общий
    def encode_queries(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode(texts)

    def encode_documents(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode(texts)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            pickle.dump(
                {"config": self.config, "vectorizer": self._vectorizer, "svd": self._svd},
                handle,
            )

    @classmethod
    def load(cls, path: str | Path) -> "LsaEncoder":
        with Path(path).open("rb") as handle:
            state = pickle.load(handle)
        encoder = cls(state["config"])
        encoder._vectorizer = state["vectorizer"]
        encoder._svd = state["svd"]
        return encoder


class Word2VecEncoder:
    """Word2Vec, обученный на корпусе; вектор текста — среднее по словам с весом IDF.

    Вторая обучаемая на корпусе модель, устроенная иначе, чем LSA: LSA работает с
    матрицей «документ — термин» целиком, Word2Vec учится на локальных
    контекстах слов. Это даёт содержательное сравнение двух векторных моделей
    даже там, где предобученные веса недоступны.

    Веса IDF нужны, чтобы частые слова не затирали редкие: без них усреднение
    делает все документы похожими.
    """

    def __init__(self, config: EncoderConfig):
        self.config = config
        self._vectors = None          # gensim KeyedVectors
        self._idf: dict[str, float] = {}
        self._default_idf = 1.0

    @property
    def dim(self) -> int:
        if self._vectors is None:
            raise RuntimeError("модель word2vec не обучена")
        return int(self._vectors.vector_size)

    def fit(self, document_texts: Sequence[str]) -> "Word2VecEncoder":
        import math
        from collections import Counter

        from gensim.models import Word2Vec

        from .text import analyze

        tokenized = [analyze(text) for text in document_texts]
        params = dict(self.config.extra.get("word2vec", {}))
        model = Word2Vec(
            sentences=tokenized,
            vector_size=int(self.config.dim or 200),
            window=int(params.get("window", 5)),
            min_count=int(params.get("min_count", 3)),
            sg=int(params.get("sg", 1)),          # skip-gram: лучше на небольших корпусах
            negative=int(params.get("negative", 10)),
            epochs=int(params.get("epochs", 10)),
            seed=int(params.get("seed", 42)),
            workers=1,                            # один поток: иначе обучение невоспроизводимо
        )
        self._vectors = model.wv

        document_frequency = Counter()
        for tokens in tokenized:
            document_frequency.update(set(tokens))
        n_docs = max(len(tokenized), 1)
        self._idf = {
            term: math.log((n_docs + 1) / (count + 1)) + 1.0
            for term, count in document_frequency.items()
        }
        self._default_idf = math.log(n_docs + 1) + 1.0
        return self

    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        if self._vectors is None:
            raise RuntimeError("модель word2vec не обучена: вызовите fit или load")
        from .text import analyze

        output = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            total_weight = 0.0
            for token in analyze(text):
                if token not in self._vectors:
                    continue
                weight = self._idf.get(token, self._default_idf)
                output[row] += weight * self._vectors[token]
                total_weight += weight
            if total_weight:
                output[row] /= total_weight
        return l2_normalize(output) if self.config.normalize else output

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode(texts)

    def encode_documents(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode(texts)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            pickle.dump(
                {
                    "config": self.config,
                    "vectors": self._vectors,
                    "idf": self._idf,
                    "default_idf": self._default_idf,
                },
                handle,
            )

    @classmethod
    def load(cls, path: str | Path) -> "Word2VecEncoder":
        with Path(path).open("rb") as handle:
            state = pickle.load(handle)
        encoder = cls(state["config"])
        encoder._vectors = state["vectors"]
        encoder._idf = state["idf"]
        encoder._default_idf = state["default_idf"]
        return encoder


#: бэкенды, которые обучаются на корпусе и поэтому хранятся вместе с индексом
TRAINABLE_BACKENDS = {"lsa": LsaEncoder, "word2vec": Word2VecEncoder}


def build_encoder(
    config: EncoderConfig,
    document_texts: Sequence[str] | None = None,
) -> TextEncoder:
    """Создать кодировщик по конфигурации.

    `document_texts` нужны только обучаемым на корпусе бэкендам (lsa, word2vec).
    """
    if config.backend == "sentence-transformers":
        return SentenceTransformerEncoder(config)
    if config.backend in TRAINABLE_BACKENDS:
        if document_texts is None:
            raise ValueError(f"бэкенду {config.backend} нужны тексты документов для обучения")
        return TRAINABLE_BACKENDS[config.backend](config).fit(document_texts)
    raise ValueError(f"неизвестный бэкенд: {config.backend}")


def load_model_configs(path: str | Path) -> dict[str, EncoderConfig]:
    import yaml

    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    models = data.get("models", data)
    return {name: EncoderConfig.from_dict(name, value) for name, value in models.items()}


def with_batch_size(config: EncoderConfig, batch_size: int) -> EncoderConfig:
    return replace(config, batch_size=batch_size)
