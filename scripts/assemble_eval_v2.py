#!/usr/bin/env python3
"""Собрать v2 из сохранённых кандидатов, явных решений и OOD-вопросов."""
import argparse
import csv
import json
import random
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--candidates',default='data/eval/candidates-v2-json/questions.jsonl')
    p.add_argument('--review',default='data/eval/candidates-v2-json/assistant_review.csv')
    p.add_argument('--ood',default='data/eval/unanswerable-v2.jsonl')
    p.add_argument('--answerable',type=int,default=160)
    p.add_argument('--output',required=True)
    args=p.parse_args()
    candidates=[json.loads(s) for s in Path(args.candidates).read_text(encoding='utf-8').splitlines()]
    reviews={r['id']:r for r in csv.DictReader(open(args.review,encoding='utf-8-sig'))}
    accepted=[]
    for row in candidates:
        review=reviews[row['id']]
        if review['accept']!='yes': continue
        if not review['reviewer'] or review['reviewer_type'] not in ('assistant','human'):
            raise ValueError('нет происхождения решения')
        row.update(question=review['question'],reference_answer=review['reference_answer'],answer_terms=json.loads(review['answer_terms']))
        accepted.append(row)
    random.Random(20260916).shuffle(accepted)
    if len(accepted)<args.answerable:
        raise ValueError('недостаточно принятых вопросов')
    for n,row in enumerate(accepted): row['split']='dev_v2' if n%2==0 else 'test_v2'
    rows=accepted[:args.answerable]
    rows.extend(json.loads(s) for s in Path(args.ood).read_text(encoding='utf-8').splitlines())
    random.Random(20260917).shuffle(rows)
    # The published artifact was created on Windows. Explicit CRLF reproduces
    # its recorded SHA256 on Linux/Colab too; JSONL readers accept both endings.
    with Path(args.output).open('xb') as handle:
        handle.write(''.join(json.dumps(r,ensure_ascii=False)+'\r\n' for r in rows).encode('utf-8'))


if __name__=='__main__': main()
