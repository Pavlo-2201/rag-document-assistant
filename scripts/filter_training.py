#!/usr/bin/env python3
"""Автоматическая фильтрация synthetic train; это слабые метки, не ручная разметка."""
import argparse
import csv
import json
import re
from pathlib import Path


def rejection(row):
    question = row['question'].casefold()
    quote = row['evidence'][0]['quote']
    if any(s in quote.casefold() for s in ('статье', 'статьи', 'руководств', 'уроке', 'урока', 'курсе', 'задании', 'испытании', 'предыдущих статьях', 'посмотрите', 'проверьте свои', 'firefox bug', 'мы рассмотрим', 'мы покажем', 'вы узнаете')):
        return 'вводный/служебный фрагмент'
    if quote.rstrip().endswith(':'):
        return 'неполный фрагмент перед перечислением'
    for term in row['answer_terms']:
        if len(term) > 3 and re.search(r'[a-zA-Z]',term) and term.casefold() in question:
            return 'имя ответа уже названо в вопросе'
    source = quote.casefold() + ' ' + ' '.join(r['slug'] for r in row['relevant']).casefold()
    if not any(t.casefold() in source for t in row['answer_terms']):
        return 'ключевые понятия эталона отсутствуют в источнике'
    if any(s in question for s in ('этой статье', 'данной статье', 'этом примере', 'этом уроке')):
        return 'зависимость от неуказанного контекста'
    return ''


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',default='data/train/synthetic-v1/questions.jsonl')
    p.add_argument('--output',default='data/train/synthetic-v1/auto_review.csv')
    args=p.parse_args()
    rows=[json.loads(s) for s in Path(args.input).read_text(encoding='utf-8').splitlines()]
    accepted=0
    with Path(args.output).open('w',encoding='utf-8-sig',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=['id','question','reference_answer','quote','accept','reviewer','reviewer_type','comment'])
        writer.writeheader()
        for row in rows:
            reason=rejection(row)
            accepted += not bool(reason)
            writer.writerow({'id':row['id'],'question':row['question'],'reference_answer':row['reference_answer'],
                'quote':row['evidence'][0]['quote'],'accept':'no' if reason else 'yes',
                'reviewer':'synthetic-validator-v1','reviewer_type':'automated',
                'comment':reason or 'Структура JSON, точная цитата, бытовой вопрос без имени ответа; семантика автоматически не гарантируется.'})
    print(f'accepted {accepted}/{len(rows)}; weak synthetic labels')


if __name__=='__main__': main()
