#!/usr/bin/env python3
"""Последовательное сравнение GGUF через один и тот же llama-server, без кеша ответов."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import requests
import yaml

from benchmark_quality import read_jsonl, write_json, compare
from rag.qa import load_qa
from rag.answer_eval import score_answer
from rag.ingest import load_documents


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--server-exe',required=True)
    p.add_argument('--config',default='configs/quality.yaml')
    p.add_argument('--retriever',default='dense-small')
    p.add_argument('--split',default='dev')
    p.add_argument('--qa',default='data/eval/qa.jsonl')
    p.add_argument('--frozen-dir',default='results/quality')
    p.add_argument('--output',default='results/quality/generators-gpu')
    p.add_argument('--gpu-layers',type=int,default=99)
    p.add_argument('--sample-stride',type=int,default=1)
    p.add_argument('--model',action='append')
    args=p.parse_args()
    cfg=yaml.safe_load(Path(args.config).read_text(encoding='utf-8'))
    output=Path(args.output); output.mkdir(parents=True,exist_ok=True)
    frozen_name=f'retrieval__{args.retriever}__{args.split}.jsonl'
    source=Path(args.frozen_dir)/frozen_name
    destination=output/frozen_name
    if destination.exists() and destination.read_bytes()!=source.read_bytes():
        raise ValueError('в output уже другие frozen hits')
    if not destination.exists(): shutil.copy2(source,destination)
    session=requests.Session(); session.trust_env=False
    port=8093
    all_scores={}; table=[]
    items={i.id:i for i in load_qa(args.qa,load_documents('data/processed/documents.jsonl'),split=args.split)}
    if (output/'models.json').exists():
        table=json.loads((output/'models.json').read_text(encoding='utf-8'))
        for entry in table:
            name=entry['model']
            answers=read_jsonl(output/f'answers__{args.retriever}__{name}__{args.split}.jsonl')
            all_scores[name]={a['id']:score_answer(a,items[a['id']]) for a in answers if items[a['id']].answerable}
    for name in args.model or list(cfg['generators']):
        spec=cfg['generators'][name]
        try:
            response=session.get(f'http://127.0.0.1:{port}/health',timeout=1)
        except requests.ConnectionError:
            pass
        else:
            raise RuntimeError(f'порт {port} уже занят; остановите собственный сервер или выберите другую среду')
        logfile=output/f'server__{name}.log'
        with logfile.open('w',encoding='utf-8') as log:
            command=[str(Path(args.server_exe).resolve()),'-m',str(Path(spec['model_path']).resolve()),
                '--host','127.0.0.1','--port',str(port),'-c','4096','-ngl',str(args.gpu_layers),
                '--parallel','1','--threads','6','--no-ui']
            process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
            try:
                ready=False
                for _ in range(120):
                    if process.poll() is not None: raise RuntimeError(f'llama-server завершился; см. {logfile}')
                    try:
                        ready=session.get(f'http://127.0.0.1:{port}/health',timeout=1).ok
                    except requests.RequestException:
                        pass
                    if ready: break
                    time.sleep(.5)
                if not ready: raise RuntimeError('llama-server не готов за 60 секунд')
                run_cfg={**cfg,'generators':{name:{**spec,'backend':'llama_server','base_url':f'http://127.0.0.1:{port}'}}}
                config_path=output/f'config__{name}.yaml'
                config_path.write_text(yaml.safe_dump(run_cfg,allow_unicode=True),encoding='utf-8')
                subprocess.run([sys.executable,'scripts/benchmark_quality.py','--phase','generation',
                    '--config',str(config_path),'--retriever',args.retriever,'--split',args.split,'--qa',args.qa,
                    '--output',str(output),'--model',name,'--sample-stride',str(args.sample_stride)],check=True)
            finally:
                process.terminate()
                try: process.wait(timeout=10)
                except subprocess.TimeoutExpired: process.kill(); process.wait()
        summary=json.loads((output/f'generation_summary__{args.retriever}__{name}__{args.split}.json').read_text(encoding='utf-8'))
        summary.update(gpu_layers=args.gpu_layers,sample_stride=args.sample_stride)
        table=[r for r in table if r['model']!=name]+[summary]
        answers=read_jsonl(output/f'answers__{args.retriever}__{name}__{args.split}.jsonl')
        all_scores[name]={a['id']:score_answer(a,items[a['id']]) for a in answers if items[a['id']].answerable}
    write_json(output/'models.json',table)
    write_json(output/f'generation_summary__{args.retriever}__{args.split}.json',table)
    baseline=next(iter(all_scores))
    write_json(output/'paired_deltas.json',{name:compare(all_scores,baseline,name,['term_hit','cited_evidence'])
        for name in all_scores if name!=baseline})


if __name__=='__main__': main()
