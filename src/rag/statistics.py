"""Парный percentile bootstrap по вопросам; одинаковые индексы у двух систем."""
from __future__ import annotations

import random
import statistics


def bootstrap_mean(values, *, baseline=None, samples=10000, seed=42, confidence=0.95):
    values = list(values)
    if not values or samples < 1 or not 0 < confidence < 1:
        raise ValueError("нужны непустая выборка, samples > 0 и 0 < confidence < 1")
    if baseline is not None:
        baseline = list(baseline)
        if len(values) != len(baseline):
            raise ValueError("парные выборки должны совпадать по длине")
        values = [a - b for a, b in zip(values, baseline)]
    rng = random.Random(seed)
    n = len(values)
    means = sorted(sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(samples))
    alpha = (1 - confidence) / 2
    return {"n": n, "mean": statistics.mean(values), "low": means[int(alpha * (samples - 1))],
            "high": means[int((1 - alpha) * (samples - 1))], "confidence": confidence,
            "samples": samples, "seed": seed, "method": "paired percentile" if baseline is not None else "percentile"}


def cohen_kappa(left, right):
    """Невзвешенная каппа для трёх категорий; '?' не является категорией."""
    left, right = list(left), list(right)
    if len(left) != len(right) or not left:
        raise ValueError("нужны непустые парные метки")
    labels = ("ДА", "ЧАСТИЧНО", "НЕТ")
    if any(v not in labels for v in left + right):
        raise ValueError("допустимы только ДА, ЧАСТИЧНО, НЕТ")
    n = len(left)
    observed = sum(a == b for a, b in zip(left, right)) / n
    expected = sum(left.count(c) * right.count(c) for c in labels) / n**2
    table = {a: {b: sum(x == a and y == b for x, y in zip(left, right)) for b in labels} for a in labels}
    return {"n": n, "agreement": observed, "kappa": (observed - expected) / (1 - expected) if expected < 1 else None,
            "confusion": table, "undefined_reason": "constant identical labels" if expected == 1 else None}
