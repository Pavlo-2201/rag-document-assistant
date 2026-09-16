#!/usr/bin/env python3
"""GPU fine-tuning E5 на проверенных synthetic query/passage; eval-страницы запрещены."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import random
import time
import math
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rag.chunking import load_chunks
from rag.evaluation import covers
from rag.ingest import load_documents
from rag.qa import load_qa
from semsearch.embedding import build_document_text


def training_pairs(items, chunks, forbidden):
    pairs, seen = [], set()
    for item in items:
        if item.split != "train" or not item.answerable or not item.evidence:
            raise ValueError(f"{item.id}: ожидается train с evidence")
        if any(e.doc_id in forbidden for e in item.evidence):
            raise ValueError(f"{item.id}: eval-страница попала в обучение")
        hits = [c for c in chunks if any(covers(c, e, 1.0) for e in item.evidence)]
        if not hits:
            raise ValueError(f"{item.id}: ни один фрагмент не покрывает цитату целиком")
        chunk = min(hits, key=lambda c: len(c.text))
        if chunk.doc_id in seen:
            continue  # один позитив со страницы: меньше ложных in-batch negatives
        seen.add(chunk.doc_id)
        # Позитив — проверенная цитата, чтобы токенный лимит не отрезал сам ответ.
        quote = next(e.quote for e in item.evidence if covers(chunk, e, 1.0))
        pairs.append({"anchor": "query: " + item.question,
                      "positive": "passage: " + build_document_text(chunk.header, quote)})
    if len(pairs) < 2:
        raise ValueError("недостаточно независимых позитивов")
    return pairs


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train", required=True)
    p.add_argument("--review", required=True)
    p.add_argument("--exclude", action="append", required=True)
    p.add_argument("--reserved-docs", action="append", required=True)
    p.add_argument("--model", default="models/e5-small")
    p.add_argument("--output", default="models/e5-finetuned")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--prepare-only", action="store_true")
    args = p.parse_args()
    if args.batch_size<2 or args.epochs<1:
        p.error('batch-size >= 2, epochs >= 1')
    docs = load_documents("data/processed/documents.jsonl")
    items = load_qa(args.train, docs)
    reviews = list(csv.DictReader(open(args.review, encoding="utf-8-sig")))
    approved = {r["id"] for r in reviews if r.get("accept", "").lower() in ("yes", "да", "1") and r.get("reviewer")}
    items = [i for i in items if i.id in approved]
    forbidden = {e.doc_id for f in args.exclude for i in load_qa(f) for e in i.evidence}
    for f in args.reserved_docs:
        forbidden.update(json.loads(Path(f).read_text(encoding="utf-8")))
    pairs = training_pairs(items, load_chunks("data/processed/chunks-c1000-o150.jsonl"), forbidden)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    manifest = {"started_at_utc": datetime.now(timezone.utc).isoformat(), "pairs": len(pairs), "base": args.model,
                "seed": args.seed, "batch_size": args.batch_size, "epochs": args.epochs,
                "loss": "MultipleNegativesRankingLoss", "sampler": "NO_DUPLICATES", "trainable": "all parameters",
                "sha256": {f: hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in [args.train, args.review, *args.exclude, *args.reserved_docs]},
                "reviewer_types": sorted({r.get('reviewer_type', 'unspecified') for r in reviews if r['id'] in approved})}
    (output / "training_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.prepare_only:
        print(json.dumps(manifest, ensure_ascii=False))
        return
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA недоступна: запустите notebooks/quality_colab.ipynb на GPU")
    import numpy as np
    import sentence_transformers
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.sentence_transformer.losses import MultipleNegativesRankingLoss
    from sentence_transformers.util import batch_to_device
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    if len({p['anchor'] for p in pairs}) != len(pairs) or len({p['positive'] for p in pairs}) != len(pairs):
        raise ValueError('дубликаты anchor/positive создают ложные in-batch negatives')
    model = SentenceTransformer(args.model, device="cuda")
    model.max_seq_length = 256
    model[0].auto_model.gradient_checkpointing_enable()
    loss_function = MultipleNegativesRankingLoss(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    total_steps = math.ceil(len(pairs) / args.batch_size) * args.epochs
    warmup = max(1, int(total_steps * .1))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
        lambda step: min(step / warmup, max(0., (total_steps-step) / max(1,total_steps-warmup))))
    scaler = torch.amp.GradScaler('cuda', init_scale=1024)
    model.train()
    started = time.perf_counter()
    history = []
    rng = random.Random(args.seed)
    for epoch in range(args.epochs):
        indices = list(range(len(pairs)))
        rng.shuffle(indices)
        for start in range(0, len(indices), args.batch_size):
            batch = [pairs[i] for i in indices[start:start+args.batch_size]]
            if len(batch) < 2:
                continue
            features = [batch_to_device(model.tokenize([p[key] for p in batch]), 'cuda') for key in ('anchor','positive')]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.float16):
                loss = loss_function(features, labels=None)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            previous_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            skipped = scaler.get_scale() < previous_scale
            if not skipped:
                scheduler.step()
            row = {'step':len(history)+1,'epoch':epoch+1,'loss':loss.item(), 'batch_size':len(batch), 'optimizer_skipped':skipped}
            history.append(row)
            print(json.dumps(row), flush=True)
    model.eval()
    model.save_pretrained(str(output))
    manifest.update({"completed_at_utc": datetime.now(timezone.utc).isoformat(),
                     "gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
                     "sentence_transformers":sentence_transformers.__version__,
                     "trainer":"PyTorch AMP loop; unique anchor/positive strings in all batches",
                     "sampler":"seeded shuffle, no duplicate strings", "steps":len(history),
                     "max_seq_length":256, "training_seconds":time.perf_counter()-started})
    (output / "training_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / 'loss.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in history),encoding='utf-8')


if __name__ == "__main__":
    main()
