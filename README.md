# rag-document-assistant

Помощник, который отвечает на вопросы по русскоязычной документации MDN Web Docs
и показывает, из каких фрагментов документов взят ответ. Всё работает локально
на CPU, без платных API и фреймворков вроде LangChain:

```
MDN Markdown → разбор (разделы, якоря) → фрагменты 1000/150 → e5-small + FAISS
вопрос → 20 кандидатов → cross-encoder → top-5 → промпт → Qwen2.5-1.5B (llama.cpp)
       → разбор ответа: ссылки [n] → источники (заголовок, раздел, url#якорь)
```

## Продолжение: поиск, GPU и оценка v2

Добавлены BM25 + E5 через RRF, e5-base, GPU fine-tuning E5 с
MultipleNegativesRankingLoss, набор из 200 новых вопросов, парные bootstrap-интервалы,
проверка 7B-судьи и потоковый вывод `ask --stream` / `chat --stream`.

На одинаковом замороженном контексте нового test Qwen3-4B повысила число ответов
с ключевым термином с **42 до 55 из 80**: **+16,25 п.п., 95% парный CI
[+6,25; +26,25]**. Ссылок на размеченный абзац стало **56/80 вместо 36/80**,
отказов на вопросы вне корпуса — **20/20 вместо 17/20**. Медиана генерации
содержательного ответа на GPU выросла с **1,25 до 2,82 с**; первый токен 4B —
**0,91 с**. Это прокси-метрики, не доля ответов, подтверждённых человеком.

Поиск пока не дал устойчивого выигрыша: fine-tuned E5 на новом test нашла
доказательство в top-5 для **64/80** вопросов против **62/80**, разница
**+2,5 п.п., 95% CI [−3,75; +8,75]**. Гибрид и e5-base на старом dev
тоже не улучшили выбранную конфигурацию, поэтому базовый поиск сохранён.

48 старых тестовых ответов и новые вопросы проверены **ассистентом**;
независимая человеческая проверка ещё требуется. 7B-судья не прошёл калибровку:
κ корректности со структурированным промптом — **0,169** на 22 содержательных
ответах. Автоматические оценки судьи нельзя выдавать за подтверждённое качество.

Полный протокол, финальный тест генераторов, таблицы качества и задержки,
сырые результаты и команды воспроизведения — [docs/quality-v2.md](docs/quality-v2.md).
GPU workflow: [notebooks/quality_colab.ipynb](notebooks/quality_colab.ipynb).

## Исходный эксперимент v1

Конфигурация выбрана только на dev: нарезка 1000 символов с перекрытием 150, `e5-small`
+ переранжирование 20 кандидатов, top-5 в промпт, строгое правило отказа
(`configs/rag.yaml`). В исходном эксперименте test использован один раз;
в продолжении он служит диагностическим набором после прежнего разбора ошибок:

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
- Исходный оценочный набор маленький (48 вопросов на split), человеческой проверки ответов нет, поэтому
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
python scripts/ingest.py --config configs/corpus.yaml
python -m rag build      # фрагменты и индекс
python -m rag check      # веса, корпус, индекс, разметка
python -m rag ask "как узнать длину строки"
python -m rag ask --stream "как узнать длину строки"
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
