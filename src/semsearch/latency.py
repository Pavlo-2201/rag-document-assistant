"""Измерение задержек поиска.

Кодирование запроса и собственно поиск измеряются отдельно: это разные по
природе операции (прямой проход модели против обхода индекса), и оптимизируются
они по-разному. Перед измерением выполняется прогрев: первый вызов включает загрузку
ленивых структур и прогрев кешей процессора, и включать его в статистику нельзя.
"""

from __future__ import annotations

import math
import platform
import statistics
import time
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np


def hardware_info() -> dict:
    """Описание машины: без него числа задержек ничего не значат."""
    info = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "processor": platform.processor() or "неизвестно",
        "cpu_count": None,
        "cpu_model": "неизвестно",
        "memory_total_gb": None,
        "gpu": "нет (измерения на CPU)",
    }
    try:
        import os

        info["cpu_count"] = os.cpu_count()
    except Exception:  # pragma: no cover
        pass
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("model name"):
                    info["cpu_model"] = line.split(":", 1)[1].strip()
                    break
    except OSError:  # pragma: no cover - не Linux
        pass
    try:
        with open("/proc/meminfo", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("MemTotal"):
                    kb = int(line.split()[1])
                    info["memory_total_gb"] = round(kb / 1024 / 1024, 1)
                    break
    except OSError:  # pragma: no cover - не Linux
        pass
    try:
        import faiss

        info["faiss"] = faiss.__version__
        info["faiss_threads"] = faiss.omp_get_max_threads()
    except Exception:  # pragma: no cover
        pass
    info["numpy"] = np.__version__
    return info


@dataclass
class LatencyStats:
    name: str
    samples_ms: list[float]

    def summary(self) -> dict:
        values = sorted(self.samples_ms)
        return {
            "measurements": len(values),
            "mean_ms": round(statistics.fmean(values), 3),
            "p50_ms": round(percentile(values, 50), 3),
            "p95_ms": round(percentile(values, 95), 3),
            "min_ms": round(values[0], 3),
            "max_ms": round(values[-1], 3),
        }


def percentile(sorted_values: Sequence[float], percent: float) -> float:
    """Процентиль методом ближайшего ранга: значение с номером ceil(P/100 * N).

    Явная реализация вместо statistics.quantiles: при нескольких десятках
    измерений разные определения процентиля дают заметно разные числа, и важно
    знать, какое использовано.
    """
    if not sorted_values:
        raise ValueError("нет измерений")
    rank = math.ceil(percent / 100 * len(sorted_values))
    rank = max(1, min(len(sorted_values), rank))
    return sorted_values[rank - 1]


def measure(
    operation: Callable[[int], None],
    n_calls: int,
    repeats: int,
    warmup: int,
    name: str,
) -> LatencyStats:
    """Выполнить операцию `warmup` раз без учёта, затем `repeats` раз с замером.

    Операция получает номер вызова, чтобы перебирать запросы по кругу: один и тот
    же запрос подряд измеряет работу кешей, а не поиска.
    """
    for i in range(warmup):
        operation(i % n_calls)
    samples: list[float] = []
    for i in range(repeats):
        started = time.perf_counter()
        operation(i % n_calls)
        samples.append((time.perf_counter() - started) * 1000.0)
    return LatencyStats(name=name, samples_ms=samples)


def measure_dense_latency(
    encoder,
    search: Callable[[np.ndarray], object],
    queries: Sequence[str],
    repeats: int = 100,
    warmup: int = 10,
) -> dict:
    """Задержка векторного поиска: кодирование запроса и поиск по отдельности.

    Измеряется по одному запросу за раз — так, как работает поиск в ответ на
    запрос пользователя, а не батчами.
    """
    queries = list(queries)
    encode = measure(
        lambda i: encoder.encode_queries([queries[i]]),
        len(queries),
        repeats,
        warmup,
        "query_encoding",
    )
    vectors = [encoder.encode_queries([query]) for query in queries]
    lookup = measure(
        lambda i: search(vectors[i]),
        len(queries),
        repeats,
        warmup,
        "index_search",
    )
    total_p50 = encode.summary()["p50_ms"] + lookup.summary()["p50_ms"]
    total_p95 = encode.summary()["p95_ms"] + lookup.summary()["p95_ms"]
    return {
        "query_encoding": encode.summary(),
        "index_search": lookup.summary(),
        "total_p50_ms": round(total_p50, 3),
        "total_p95_ms": round(total_p95, 3),
    }


def measure_end_to_end_latency(
    retriever,
    queries: Sequence[str],
    top_k: int = 10,
    repeats: int = 100,
    warmup: int = 10,
) -> dict:
    """Задержка поиска целиком — для способов, где разделение не имеет смысла.

    У лексического поиска «кодирование запроса» это разбор строки на термины, он
    неотделим от обхода словаря, поэтому измеряется один общий этап.
    """
    queries = list(queries)
    stats = measure(
        lambda i: retriever.search(queries[i], top_k=top_k),
        len(queries),
        repeats,
        warmup,
        "search",
    )
    summary = stats.summary()
    return {
        "search": summary,
        "total_p50_ms": summary["p50_ms"],
        "total_p95_ms": summary["p95_ms"],
    }
