# rag-document-assistant

Помощник, который отвечает на вопросы по русскоязычной документации MDN Web Docs
и показывает, из каких фрагментов документов взят ответ. Всё работает локально
на CPU, без платных API и фреймворков вроде LangChain:

```
MDN Markdown → разбор (разделы, якоря) → фрагменты 1000/150 → e5-small + FAISS
вопрос → 20 кандидатов → cross-encoder → top-5 → промпт → Qwen2.5-1.5B (llama.cpp)
       → разбор ответа: ссылки [n] → источники (заголовок, раздел, url#якорь)
```

## Итоги

Конфигурация выбрана только на dev: нарезка 1000 символов с перекрытием 150, `e5-small`
+ переранжирование 20 кандидатов, top-5 в промпт, строгое правило отказа
(`configs/rag.yaml`). Test использован один раз:

| вариант | набор | абзац с ответом в промпте (из 36) | ключевой термин в ответе (из 36) | отказ на вопросы без ответа (из 12) | генерация, с (медиана) |
|---|---|---|---|---|---|
| база: нарезка 1000/150, `e5-small`, top-5 | dev | 11 | 15 | 11 | 13.2 |
| **+ rerank@20 (выбран)** | dev | 15 | 14 | 11 | 12.2 |
| **+ rerank@20 (выбран)** | test | 13 | 15 | 11 | 11.7 |

- Ответ ссылается только на фрагменты, переданные модели; если ответа нет, помощник
  отказывается (11 из 12 вопросов без ответа).
- Узкое место — поиск по бытовым формулировкам: 18 из 21 ошибки на test — поиск, и в 14
  из 36 вопросов страницы с ответом нет даже среди 20 кандидатов.
- Лучший поиск не дал лучших ответов: с длинным контекстом 1.5B-модель чаще отказывается,
  хотя ответ перед ней.
- Оценочный набор маленький (48 вопросов на split), ручной проверки ответов нет, поэтому
  разница в 1–2 вопроса — шум.
- Переформулировка вопроса перед поиском (HyDE, флаг `--hyde`) находит абзац с ответом
  среди 20 кандидатов у 22 test-вопросов из 36 вместо 14, но вместе с переранжированием
  ответы на dev не улучшила, поэтому по умолчанию выключена
  ([docs/experiments.md](docs/experiments.md), «Сущности и HyDE»).

Подробности по поиску, генерации, экспериментам и оценке — в [docs/](docs/).

## Данные

Корпус тот же, что в `semantic-search-engine`: 2570 русских страниц MDN Web Docs из
[mdn/translated-content](https://github.com/mdn/translated-content), зафиксированных commit'ом
`f39fd206`; лицензия [CC-BY-SA 2.5](https://creativecommons.org/licenses/by-sa/2.5/), авторство —
MDN Web Docs contributors. Оценочный набор — вопросы с эталонными ответами и цитатами-доказательствами,
плюс вопросы, ответа на которые в корпусе нет. В Git нет ни корпуса, ни весов моделей.

## Быстрый старт (PowerShell)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .

python scripts/download_corpus.py --config configs/corpus.yaml
python -m rag build      # фрагменты и индекс
python -m rag check      # веса, корпус, индекс, разметка
python -m rag ask "как узнать длину строки"
python -m rag chat       # вопросы по одному
python -m pytest
```

### Модели

Веса скачиваются отдельно с фиксированной ревизией и проверяются по sha256
(`python -m rag check`):

```powershell
hf download Qwen/Qwen2.5-1.5B-Instruct-GGUF qwen2.5-1.5b-instruct-q4_k_m.gguf `
    --revision 91cad51170dc346986eccefdc2dd33a9da36ead9 --local-dir models/qwen
hf download intfloat/multilingual-e5-small config.json modules.json sentence_bert_config.json `
    1_Pooling/config.json tokenizer.json tokenizer_config.json special_tokens_map.json `
    sentencepiece.bpe.model model.safetensors `
    --revision 614241f622f53c4eeff9890bdc4f31cfecc418b3 --local-dir models/e5-small
hf download cross-encoder/mmarco-mMiniLMv2-L12-H384-v1 config.json tokenizer.json tokenizer_config.json `
    special_tokens_map.json sentencepiece.bpe.model model.safetensors `
    --revision 1427fd652930e4ba29e8149678df786c240d8825 --local-dir models/reranker
```

## Структура

```
src/rag/        разбор, нарезка, поиск, переранжирование, промпт, LLM, ссылки, кеш, CLI, оценка
src/semsearch/  поиск из semantic-search-engine (BM25, sentence-transformers, FAISS)
scripts/        загрузка корпуса, индексы, эксперименты, оценка
configs/        конфигурация конвейера и экспериментов
data/eval/      оценочный набор вопросов
results/        результаты экспериментов и оценки
docs/           подробности по этапам
tests/          pytest
```

## Стек

Python, multilingual-e5-small (sentence-transformers), FAISS, cross-encoder mmarco-mMiniLMv2,
Qwen2.5-1.5B-Instruct (GGUF, llama-cpp-python), PyTorch, Transformers, scikit-learn, SQLite, pytest.
