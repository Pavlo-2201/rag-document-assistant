#!/usr/bin/env python3
"""Замороженный поиск, генераторы на одних hits, bootstrap и сырые ответы.

Сначала --phase retrieval, затем --phase generation; выбор только на dev.
Повтор старого test — диагностический: он уже использован при анализе ошибок.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rag.answer_eval import score_answer, summarize_scores
from rag.chunking import load_chunks
from rag.evaluation import evidence_rank
from rag.generation import GenerationConfig, RAGPipeline
from rag.ingest import load_documents
from rag.llm import make_backend
from rag.qa import load_qa
from rag.retrieval import ChunkHit, HybridRetriever, build_bm25, load_dense
from rag.reranker import CrossEncoderReranker
from rag.statistics import bootstrap_mean


def read_jsonl(path):
    return [json.loads(s) for s in Path(path).read_text(encoding="utf-8").splitlines() if s.strip()]


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def digest(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle,'sha256').hexdigest()


def compare(rows, baseline, candidate, metrics):
    a, b = rows[baseline], rows[candidate]
    if set(a) != set(b):
        raise ValueError("разные id у парных выборок")
    ids = sorted(a)
    return {m: bootstrap_mean([b[i][m] for i in ids], baseline=[a[i][m] for i in ids]) for m in metrics}


def retrieval(args, cfg, items, chunks, output):
    lexical = build_bm25(chunks)
    reranker = CrossEncoderReranker(device=args.reranker_device)
    cache = {}
    score_cache = {}
    all_rows = {}
    common_guard={'qa_sha256':digest(args.qa),'chunks_sha256':digest(cfg['chunks']),
                  'reranker_sha256':digest('models/reranker/model.safetensors'),
                  'reranker_device':args.reranker_device}
    for name, spec in cfg["retrievers"].items():
        path = output / f"retrieval__{name}__{args.split}.jsonl"
        index_meta=Path('data/indexes')/f"{Path(cfg['chunks']).stem}__{spec['model']}"/'meta.json'
        guard={**common_guard,'spec':spec,'index_meta_sha256':digest(index_meta),'question_ids':[i.id for i in items]}
        guard_path=path.with_suffix('.artifact.json')
        if guard_path.exists() and json.loads(guard_path.read_text(encoding='utf-8'))!=guard:
            raise ValueError(f'конфиг, вопросы, корпус или модель изменились после {path}; используйте новый output')
        if path.exists():
            rows = {r["id"]: r for r in read_jsonl(path)}
            if set(rows) != {i.id for i in items} or any(rows[i.id]['question'] != i.question for i in items):
                raise ValueError(f"неполный или устаревший прогон {path}; сохраните его отдельно перед перезапуском")
            all_rows[name] = {i: r for i, r in rows.items() if r["answerable"]}
            if not guard_path.exists(): write_json(guard_path,guard)
            continue
        write_json(guard_path,guard)
        model = spec["model"]
        if model not in cache:
            cache[model] = load_dense(chunks, Path("data/indexes") / f"{Path(cfg['chunks']).stem}__{model}")
        retriever = cache[model]
        if spec.get("hybrid"):
            retriever = HybridRetriever(retriever, lexical, depth=spec.get("depth", 100), k=spec.get("rrf_k", 60),
                                        fixed_depth=spec.get('fixed_depth',False))
        if spec.get("hyde"):
            from rag.query_rewrite import HydeRetriever
            rewrite_llm = make_backend(cfg["rewrite_llm"])
            retriever = HydeRetriever(retriever, rewrite_llm, mode=spec["hyde"], max_tokens=64, depth=100)
        rows = {}
        with path.open("x", encoding="utf-8") as handle:
            for item in items:
                started = time.perf_counter()
                candidates = retriever.search(item.question, top_k=spec.get("candidates", 20))
                candidate_s = time.perf_counter() - started
                missing = [h for h in candidates if (item.id, h.chunk.chunk_id) not in score_cache]
                scores = reranker.score(item.question, missing)
                score_cache.update({(item.id, h.chunk.chunk_id): s for h, s in zip(missing, scores)})
                order = sorted(candidates, key=lambda h: -score_cache[item.id, h.chunk.chunk_id])
                hits = [ChunkHit(h.chunk, score_cache[item.id, h.chunk.chunk_id], rank)
                        for rank, h in enumerate(order[:cfg["generation"]["top_k"]], 1)]
                row = {"id": item.id, "question": item.question, "answerable": item.answerable,
                       "candidate_evidence_rank": evidence_rank(candidates, item),
                       "evidence_rank": evidence_rank(hits, item),
                       "candidate_page_hit": any(h.doc_id in {e.doc_id for e in item.evidence} for h in candidates),
                       "candidate_s": candidate_s, "retrieval_s": time.perf_counter() - started,
                       "timing_mode": "shared_reranker_score_cache", "rerank_cache_hits": len(candidates) - len(missing),
                       "query_expansion": getattr(retriever, "last_expansion", ""),
                       "hits": [{"chunk_id": h.chunk.chunk_id, "score": h.score, "rank": h.rank} for h in hits]}
                row["evidence@5"] = row["evidence_rank"] is not None
                row["evidence@candidates"] = row["candidate_evidence_rank"] is not None
                row["candidate_count"] = len(candidates)
                rows[item.id] = row
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
        all_rows[name] = {i: r for i, r in rows.items() if r["answerable"]}
        print(name, sum(r["evidence@5"] for r in all_rows[name].values()), "/", len(all_rows[name]), flush=True)
    base = cfg["baseline"]
    summary = {"baseline": base, "split": args.split, "comparisons": {}, "systems": {}}
    for name, rows in all_rows.items():
        summary["systems"][name] = {m: bootstrap_mean([r[m] for r in rows.values()])
                                     for m in ("evidence@5", "evidence@candidates", "candidate_page_hit")}
        if name != base:
            summary["comparisons"][name] = compare(all_rows, base, name, ["evidence@5", "evidence@candidates", "candidate_page_hit"])
    write_json(output / f"retrieval_summary__{args.split}.json", summary)


def generation(args, cfg, items, chunks, output):
    chunks = {c.chunk_id: c for c in chunks}
    selected = args.retriever or cfg["selected_retriever"]
    hits_path = output / f"retrieval__{selected}__{args.split}.jsonl"
    frozen = {r["id"]: r for r in read_jsonl(hits_path)}
    all_scores = {}
    table = []
    for name in args.model or cfg["generators"]:
        model_cfg = cfg["generators"][name]
        llm = make_backend(model_cfg)  # без кеша: измеряется реальная задержка
        write_json(output / f"backend__{name}__{args.split}.json", {"name": llm.name, "fingerprint": getattr(llm, "fingerprint", "")})
        pipeline = RAGPipeline(None, llm, GenerationConfig(**cfg["generation"]))
        path = output / f"answers__{selected}__{name}__{args.split}.jsonl"
        scores, answers = {}, []
        with path.open("x", encoding="utf-8") as handle:
            for item in items:
                row = frozen[item.id]
                if row["question"] != item.question:
                    raise ValueError("вопрос изменился после заморозки поиска")
                hits = [ChunkHit(chunks[h["chunk_id"]], h["score"], h["rank"]) for h in row["hits"]]
                answer = pipeline.answer_from_hits(item.question, hits, {"retrieval_s": row["retrieval_s"]})
                record = {"id": item.id, **answer.to_dict()}
                answers.append(record)
                scores[item.id] = score_answer(record, item)
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
                print(name, item.id, answer.answer[:50], flush=True)
        all_scores[name] = {i: s for i, s in scores.items() if s["answerable"]}
        stats = summarize_scores(list(scores.values()), answers)
        for status, subset in [('answered',[a for a in answers if not a['refused']]),
                               ('refused',[a for a in answers if a['refused']])]:
            values=[a['timings'].get('generation_s',0)+a['timings'].get('retry_s',0) for a in subset]
            stats['cost'][f'generation_s_{status}_median']=statistics.median(values) if values else None
        stats["first_token_s_median"] = statistics.median(a["timings"]["first_token_s"] for a in answers if "first_token_s" in a["timings"]) if any("first_token_s" in a["timings"] for a in answers) else None
        table.append({"model": name, "retriever": selected, **stats})
        write_json(output / f"generation_summary__{selected}__{name}__{args.split}.json", table[-1])
        del pipeline, llm
    write_json(output / f"generation_summary__{selected}__{args.split}.json", table)
    if len(all_scores) > 1:
        base = next(iter(all_scores))
        write_json(output / f"generation_deltas__{selected}__{args.split}.json",
                   {n: compare(all_scores, base, n, ["term_hit", "cited_evidence"]) for n in all_scores if n != base})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/quality.yaml")
    p.add_argument("--qa", default="data/eval/qa.jsonl")
    p.add_argument("--split", default="dev")
    p.add_argument("--output", default="results/quality")
    p.add_argument("--phase", choices=["retrieval", "generation"], required=True)
    p.add_argument("--retriever")
    p.add_argument("--model", action="append")
    p.add_argument("--systems", nargs='+', help="только указанные retriever-системы; включите baseline")
    p.add_argument('--sample-stride',type=int,default=1,help='явная диагностическая подвыборка: каждый N-й вопрос')
    p.add_argument('--reranker-device',choices=['cpu','cuda'],default='cpu')
    args = p.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if args.systems:
        cfg['retrievers'] = {n:cfg['retrievers'][n] for n in args.systems}
        if cfg['baseline'] not in cfg['retrievers']:
            p.error('--systems должен включать baseline из конфига')
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    documents = load_documents("data/processed/documents.jsonl")
    items = load_qa(args.qa, documents, split=args.split)
    if args.sample_stride < 1: p.error('--sample-stride >= 1')
    items = items[::args.sample_stride]
    if not items or len({i.id for i in items}) != len(items):
        raise ValueError("пустой split или повтор id")
    chunks = load_chunks(cfg["chunks"])
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    manifest = output / f"manifest__{args.phase}__{args.retriever or cfg['selected_retriever']}__{'-'.join(args.model or [])}__{args.split}__{run_id}.json"
    write_json(manifest, {"started_at_utc": datetime.now(timezone.utc).isoformat(), "config": cfg,
                         "config_sha256": digest(args.config), "qa_sha256": digest(args.qa),
                         "question_ids":[i.id for i in items], "sample_stride":args.sample_stride,
                         "chunks_sha256": digest(cfg["chunks"]), "python": sys.version, "platform": platform.platform()})
    (retrieval if args.phase == "retrieval" else generation)(args, cfg, items, chunks, output)


if __name__ == "__main__":
    main()
