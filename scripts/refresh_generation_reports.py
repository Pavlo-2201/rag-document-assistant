#!/usr/bin/env python3
"""Пересчитать метрики и интервалы из сырых ответов без нового вызова моделей."""
import argparse
import json
from pathlib import Path
import statistics
from benchmark_quality import read_jsonl, write_json, compare
from rag.answer_eval import score_answer, summarize_scores
from rag.ingest import load_documents
from rag.qa import load_qa
from rag.statistics import bootstrap_mean


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--folder',required=True)
    p.add_argument('--qa',default='data/eval/qa.jsonl')
    p.add_argument('--split',default='dev')
    p.add_argument('--baseline',default='qwen1.5b')
    args=p.parse_args()
    folder=Path(args.folder)
    items={i.id:i for i in load_qa(args.qa,load_documents('data/processed/documents.jsonl'),split=args.split)}
    previous=json.loads((folder/'models.json').read_text(encoding='utf-8')) if (folder/'models.json').exists() else []
    previous={r['model']:r for r in previous}
    table=[]; scores={}
    for path in sorted(folder.glob(f'answers__*__{args.split}.jsonl')):
        _,retriever,model,_=path.stem.split('__')
        answers=read_jsonl(path)
        if len({a['id'] for a in answers})!=len(answers) or any(a['id'] not in items or a['question']!=items[a['id']].question for a in answers):
            raise ValueError(f'неверные id или изменившиеся вопросы в {path}')
        rows=[score_answer(a,items[a['id']]) for a in answers]
        stats={'model':model,'retriever':retriever,**summarize_scores(rows,answers)}
        for status,label in [(False,'answered'),(True,'refused')]:
            values=[a['timings'].get('generation_s',0)+a['timings'].get('retry_s',0) for a in answers if a['refused']==status]
            stats['cost'][f'generation_s_{label}_median']=statistics.median(values) if values else None
        tokens=[a['timings']['first_token_s'] for a in answers if 'first_token_s' in a['timings']]
        stats['first_token_s_median']=statistics.median(tokens) if tokens else None
        stats.update({k:v for k,v in previous.get(model,{}).items() if k in ('gpu_layers','sample_stride')})
        for metric in ('term_hit','cited_evidence'):
            stats['answerable'][metric+'_CI']=bootstrap_mean([r[metric] for r in rows if r['answerable']])
        write_json(folder/f'generation_summary__{retriever}__{model}__{args.split}.json',stats)
        write_jsonl=folder/f'scores__{retriever}__{model}__{args.split}.jsonl'
        write_jsonl.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')
        table.append(stats)
        scores[model]={r['id']:r for r in rows if r['answerable']}
    write_json(folder/'models.json',table)
    if args.baseline not in scores: raise ValueError('baseline отсутствует среди сырых ответов')
    write_json(folder/'paired_deltas.json',{m:compare(scores,args.baseline,m,['term_hit','cited_evidence']) for m in scores if m!=args.baseline})
    print([(r['model'],r['answerable']['term_hit'],r['answerable']['cited_evidence'],r['cost']['generation_s_answered_median']) for r in table])


if __name__=='__main__': main()
