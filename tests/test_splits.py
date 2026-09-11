"""Test не участвует в выборе: вопросы dev и test не пересекаются, ответы на
test посчитаны только для одного варианта — того, что выбран на dev и записан
в configs/rag.yaml."""

import json
from pathlib import Path

import yaml

from rag.qa import load_qa

ROOT = Path(__file__).resolve().parents[1]


def normalize(question: str) -> str:
    return " ".join(question.lower().replace("ё", "е").split())


def test_dev_and_test_questions_do_not_overlap():
    items = load_qa(ROOT / "data/eval/qa.jsonl")
    assert {i.split for i in items} == {"dev", "test"}
    dev = {normalize(i.question) for i in items if i.split == "dev"}
    test = {normalize(i.question) for i in items if i.split == "test"}
    assert dev and test and not dev & test


def test_test_split_answered_only_by_variant_chosen_on_dev():
    summary = json.loads((ROOT / "results/experiments/generation/summary.json").read_text(encoding="utf-8"))
    on_dev = {key.rsplit("__", 1)[0] for key in summary if key.endswith("__dev")}
    on_test = {key.rsplit("__", 1)[0] for key in summary if key.endswith("__test")}
    assert len(on_test) == 1 and on_test <= on_dev
    chosen = summary[f"{on_test.pop()}__test"]["variant"]
    config = yaml.safe_load((ROOT / "configs/rag.yaml").read_text(encoding="utf-8"))
    assert (chosen["chunk_size"], chosen["overlap"]) == (config["chunking"]["chunk_size"], config["chunking"]["overlap"])
    assert chosen["retriever"] == config["retriever"]["model"]
    assert chosen["rerank"] == (config.get("reranker") or {}).get("candidates", 0)
    assert chosen["top_k"] == config["generation"]["top_k"]
