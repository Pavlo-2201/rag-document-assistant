# rag-document-assistant

Помощник, который отвечает на вопросы по русскоязычной документации MDN Web Docs
и показывает, из каких фрагментов документов взят ответ.

Архитектура (собирается по этапам):
documents → parser → chunking → embeddings → vector index → retriever → reranker → LLM → answer + sources.

## Данные

Корпус тот же, что в предыдущем проекте semantic-search-engine: 2570 русских
страниц MDN Web Docs (репозиторий [mdn/translated-content](https://github.com/mdn/translated-content)),
лицензия [CC-BY-SA 2.5](https://creativecommons.org/licenses/by-sa/2.5/), авторство —
MDN Web Docs contributors.

- Источник зафиксирован commit'ом `f39fd206b38a94264315eb2bcacb705bd8b5d326`
  (`configs/corpus.yaml`).
- Отбор страниц тот же, что в semantic-search-engine. После загрузки отпечаток
  корпуса считается его способом и сравнивается с опубликованным
  (`0f5e613e…`): **совпал**, то есть документы и их текст одинаковые
  (`data/corpus.meta.json`).
- В Git не попадают ни исходники страниц, ни результат разбора: только скрипты и
  метаданные.

## Установка

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Команды

```bash
python scripts/download_corpus.py --config configs/corpus.yaml   # ~40 с, Markdown страниц
python scripts/ingest.py --config configs/corpus.yaml            # разбор → data/processed/documents.jsonl
python scripts/ingest.py --input-dir my_docs/ --output data/processed/my_docs.jsonl  # свои .md/.txt
python -m pytest
```

## Этапы

| этап | дата | решение и почему |
|---|---|---|
| Ingest | 2026-07-06 | Хранится исходный Markdown, а не очищенный текст из semantic-search-engine: для ответа со ссылками нужны заголовки разделов (ссылка `url#якорь`, нарезка по разделам) и блоки кода (в них часто сам ответ). `doc_id` взят из semantic-search-engine (`mdnru-` + sha1(slug)), чтобы его разметка запросов подходила без пересчёта. Пакет `semsearch` скопирован без изменений, версия зафиксирована sha256 файлов. PDF не добавлен: в корпусе его нет |
