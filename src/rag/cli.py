"""Командная строка ассистента: `python -m rag <команда>`.

    python -m rag build                      # фрагменты и индекс по configs/rag.yaml
    python -m rag ask "как узнать длину строки"
    python -m rag ask --backend mock "что значит код 404"   # без весов LLM
    python -m rag ask --hyde rrf "как рисовать графику прямо в браузере"
    python -m rag chat                       # вопросы по одному в терминале
    python -m rag check                      # веса, корпус, индекс, разметка, связность
"""

from __future__ import annotations

import argparse
import json
import sys

from .app import build_artifacts, build_pipeline, load_config
from .generation import format_answer


def _ask(pipeline, question: str, as_json: bool) -> None:
    answer = pipeline.answer(question)
    print(json.dumps(answer.to_dict(), ensure_ascii=False, indent=2) if as_json else format_answer(answer))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m rag", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/rag.yaml")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("build", help="собрать фрагменты и индекс")
    ask = commands.add_parser("ask", help="ответить на вопрос")
    ask.add_argument("question")
    chat = commands.add_parser("chat", help="отвечать на вопросы из терминала, пустая строка — выход")
    for sub in (ask, chat):
        sub.add_argument("--backend", choices=["llama_cpp", "mock"], help="LLM-бэкенд вместо указанного в конфиге")
        sub.add_argument("--no-rerank", action="store_true", help="без cross-encoder'а")
        sub.add_argument("--hyde", choices=["concat", "rrf", "off"], help="переформулировка вопроса перед поиском")
        sub.add_argument("--top-k", type=int)
        sub.add_argument("--json", action="store_true")
    check = commands.add_parser("check", help="проверить веса, данные, индекс и разметку")
    check.add_argument("--skip-models", action="store_true", help="не считать sha256 весов")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    if args.command == "build":
        paths = build_artifacts(config)
        print(f"готово: {paths.chunks}" + (f", {paths.index}" if paths.index else ""))
        return 0
    if args.command == "check":
        from .checks import run_checks

        results = run_checks(config, models=not args.skip_models)
        for name, status, detail in results:
            print(f"[{status:4s}] {name}: {detail}")
        return 1 if any(status == "fail" for _, status, _ in results) else 0

    if args.no_rerank:
        config["reranker"] = None
    if args.hyde:
        config["query_rewrite"] = {"hyde": None if args.hyde == "off" else args.hyde}
    if args.top_k:
        config.setdefault("generation", {})["top_k"] = args.top_k
    pipeline = build_pipeline(config, backend=args.backend)
    if args.command == "ask":
        _ask(pipeline, args.question, args.json)
        return 0
    print("Вопрос по документации MDN (пустая строка — выход):", file=sys.stderr)
    for line in sys.stdin:
        if not line.strip():
            break
        _ask(pipeline, line.strip(), args.json)
        print(file=sys.stderr)
    return 0
