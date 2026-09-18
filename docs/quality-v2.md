# Поиск, дообучение и оценка v2

Продолжение исходного анализа ошибок: реализованы и измерены гибридный поиск,
более крупный эмбеддер, дообучение E5, новые вопросы, калибровка судьи и
генераторы 1.5B/3B/4B со стримингом. Подтверждённый выигрыш в поиске пока
не получен. Генератор 4B лучше использует уже найденный контекст.

## Протокол и ограничения

Корпус и нарезка сохранены: MDN `f39fd206b38a94264315eb2bcacb705bd8b5d326`,
2570 страниц, 16941 фрагмент, 1000 символов / перекрытие 150. Все варианты
используют один cross-encoder `mmarco-mMiniLMv2-L12-H384-v1`, top-5 в промпт,
3000 токенов контекста, до 300 токенов ответа, strict refusal и retry_empty.
`semsearch` переиспользован без изменения vendored-файлов.

Старый test уже использован для анализа ошибок и настройки рубрики судьи:
он диагностический. Выбор поиска и генератора записан на старом dev в
[`selection.json`](../results/quality/selection.json) и
[`generator_selection.json`](../results/quality/generator_selection.json).
Параметры после просмотра нового test не подбирались.

Новый набор v2 содержит 200 вопросов: по 80 с ответом и 20 без ответа в
`dev_v2` и `test_v2`. Из 240 кандидатов локальной Qwen2.5-3B ассистент прочитал
исходные абзацы, исправил вопросы/эталоны и принял 177. Из них seed-shuffle
выбрала 160; 40 вопросов вне корпуса составлены отдельно, отсутствие характерных
терминов проверено по всему корпусу. Каждая из 160 страниц используется один раз.
Разделение сделано по страницам, выбор не зависел от результатов поиска.
Это source-conditioned синтетический набор: он проще старых бытовых запросов,
и абсолютные результаты между v1 и v2 нельзя приписывать улучшению модели.

Проверка ассистентом — **не независимая человеческая разметка**.
[`qa-v2.provenance.json`](../data/eval/qa-v2.provenance.json) явно содержит
`reviewer_type=assistant`, `human_review_status=pending`. Решения по всем 240
кандидатам и исходные ответы LLM сохранены в `data/eval/candidates-v2-json/`.
Пустой файл для независимой проверки —
[`manual_review_v2.csv`](../data/eval/manual_review_v2.csv).

Доверительные интервалы — percentile bootstrap, 10000 повторов, seed=42,
95%; для разниц ресемплируются **пары ответов одного вопроса**. Знаменатель
метрик поиска/ответов — только answerable, отказы OOD считаются отдельно.
`term_hit` ищет ключевые понятия; `cited_evidence` проверяет ссылку на фрагмент,
перекрывающий размеченную цитату. Ни одна метрика не доказывает фактическую
корректность всего ответа. Проверка смещений цитат обязательна перед оценкой:
неразрешённая цитата вызывает ошибку вместо молчаливого нулевого результата.

## Поиск

BM25 и dense получают независимые выдачи; `rrf_merge` суммирует
`1 / (60 + rank)` по `chunk_id`, каждый фрагмент голосует один раз от каждой
выдачи. После объединения кандидаты проходят общий cross-encoder.
Оценки BM25 и косинусная близость напрямую не складываются.

На старом dev (36 вопросов с ответом):

| Поиск → число кандидатов в reranker | Доказательство в top-5 |
|---|---:|
| e5-small → 20, контроль | 15/36 |
| BM25 + small, RRF depth=100 → 20 | 9/36 |
| e5-small → 60 | 14/36 |
| BM25 + small, RRF depth=100 → 60 | 13/36 |
| e5-base → 20 / 60 | 15/36 / 15/36 |
| BM25 + base, RRF depth=100 → 60 | 13/36 |
| BM25 + small + HyDE (3B), RRF → 60 | 14/36 |
| fine-tuned small → 20 | 16/36 |
| BM25 + fine-tuned small, RRF → 60 | 13/36 |
| union top-20 small и top-20 BM25 → до 40 | 13/36 |

Union сохраняет все dense-кандидаты перед reranker, но после reranker также
хуже контроля: проблема не ограничивается отбрасыванием dense-кандидатов при RRF.
Добавление BM25 или увеличение размерности само по себе не исправило бытовые
формулировки. Default остался e5-small → 20. e5-large подключается конфигом,
но здесь не измерен; BGE-M3 не проверялся.

На v2 проверены фиксированные baseline и fine-tuned small:

| Набор | small | fine-tuned small | Разница, 95% парный CI |
|---|---:|---:|---:|
| dev_v2 evidence@5 | 70/80 | 67/80 | −3,75 п.п. [−8,75; 0] |
| test_v2 evidence@5 | 62/80 | 64/80 | +2,5 п.п. [−3,75; +8,75] |
| test_v2 evidence@20 | 65/80 | 69/80 | +5 п.п. [−2,5; +12,5] |
| test_v2 page@20 | 71/80 | 74/80 | +3,75 п.п. [−2,5; +10] |

Нулевая разница входит в интервалы: уверенного улучшения нет.
Полные интервалы и результаты каждого вопроса:
[`results/quality/`](../results/quality/retrieval_summary__dev.json),
[`results/quality/v2/`](../results/quality/v2/retrieval_summary__test_v2.json).
В этих прогонах оценки reranker повторно используются между системами:
`timing_mode=shared_reranker_score_cache`. Их `retrieval_s` не предназначена
для сравнения честной задержки полного поиска.

## GPU fine-tuning

Локальная Qwen2.5-3B сгенерировала 300 train-кандидатов. Автоматический фильтр
принял 164 пары; это **слабые синтетические метки**, а не семантическая
разметка человеком. Исключены все страницы старого eval, нового eval и
720 страниц резерва eval-кандидатов, включая отклонённые. Скрипт обучения
повторно проверяет отсутствие пересечения и дубликатов anchor/positive.

Обучена multilingual-e5-small целиком с
[MultipleNegativesRankingLoss](https://sbert.net/docs/sentence_transformer/losses.html):
`query: вопрос` / `passage: заголовок + точная цитата`. Другие позитивы
batch служат negatives; по одному позитиву на страницу снижает ложные negatives.
AdamW 2e-5, weight decay 0.01, batch=16, 2 эпохи, max_seq_length=256,
warmup 10%, AMP, gradient checkpointing, clipping=1, seed=42. Это небольшой
пилот: 22 шага без пропущенных AMP updates; цикл обучения и сохранение
заняли 6,74 с без загрузки исходной модели.
Уменьшение train loss не считается доказательством улучшения поиска.

RTX 3050 Ti Laptop 4 GB, torch 2.11.0+cu128, sentence-transformers 6.1.0.
Индекс дообученной модели собран на GPU/FP16 за 45,3 с; e5-base — за 134,0 с.
FAISS индекс и запросы работают на CPU. SHA весов записан в метаданных индекса:
после подмены весов старый индекс отклоняется. Train max_length=256,
при индексации/поиске используется 512, как у исходного E5.
[`training_manifest.json`](../results/quality/training/training_manifest.json)
содержит зависимости, GPU и SHA входных файлов;
[`loss.jsonl`](../results/quality/training/loss.jsonl) — каждый шаг.

## Разметка и судья

Все 48 исходных test-ответов размечены ассистентом в
[`manual_review.csv`](../results/evaluation/rerank20__test/manual_review.csv).
Исходные столбцы судьи сохранены; `reviewer_type=assistant` указан явно.
Отдельный [слепой CSV](../results/quality/review/manual_review.csv) содержит
вопросы, эталоны и полные процитированные фрагменты без оценок судьи — для человека.

Судья Qwen2.5-7B Q4_K_M использован только офлайн. Сравниваются три категории
ДА / ЧАСТИЧНО / НЕТ, unweighted Cohen κ. Детерминированные отказы и отсутствие
ссылок исключены из LLM-only κ: лёгкие правила не должны маскировать качество
самого судьи. Спорный `u-test-08` исключён из correctness до расчёта: сочетание
клавиш общего браузерного инструмента может подходить стороннему браузеру.
Покрытие, исключения и исходные вердикты сохранены в `results/quality/review/`.

| Судья | κ корректности (n=22) | Согласие | κ groundedness (n=18) | Согласие |
|---|---:|---:|---:|---:|
| исходная 1.5B, прежняя рубрика | 0,236 | 54,5% | 0 | 66,7% |
| 7B, та же рубрика | 0,052 | 31,8% | 0,214 | 50,0% |
| 7B, JSON: причина + вердикт, явная рубрика | 0,169 | 40,9% | 0,323 | 61,1% |

Более крупная модель не стала надёжным судьёй. Даже в причинах 7B встречаются
выдуманные детали ответа. Структурированная рубрика помогает разбору вердикта,
но не заменяет человеческую калибровку; она изменялась на диагностическом наборе.
Это κ **с ассистентом**, κ с независимым человеком пока не измерена.

## Генератор и задержка

Один официальный [llama.cpp b11669](https://github.com/ggml-org/llama.cpp/releases/tag/b11669),
commit `620d2225c`, Windows 11, i5-11400H (6 CPU threads), 16 GB RAM,
RTX 3050 Ti Laptop 4 GB, CUDA 12.8 runtime. GGUF Q4_K_M, context=4096,
temperature=0, seed=42, repeat_penalty=1.1, parallel=1, GPU layers=99.
У [Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B) отключён thinking
через `chat_template_kwargs.enable_thinking=false`. Весовые ревизии и SHA
проверены и сохранены в [`models.lock.json`](../results/quality/models.lock.json).
Результат зависит также от версии runtime, поэтому старые CPU-числа v1
не используются как контроль для оценки влияния размера модели.

Старый dev, одинаковые замороженные top-5, без кеша LLM-ответов:

| Модель | term_hit /36 | cited_evidence /36 | Отказы OOD /12 | Ответила /36 | Генерация: все, с | Только содержательные ответы, с | Первый токен, с |
|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen2.5-1.5B | 20 | 4 | 9 | 28 | 1,24 | 1,89 | 0,34 |
| Qwen2.5-3B | 12 | 8 | 12 | 20 | 0,93 | 1,85 | 0,63 |
| Qwen3-4B | 26 | 15 | 12 | 33 | 2,81 | 3,18 | 0,92 |

Разница 4B−1.5B: term_hit **+16,7 п.п. [0; +33,3]**,
cited_evidence **+30,6 п.п. [+16,7; +47,2]**. Это выбор на dev;
итоговый тест приводится ниже. Быстрые короткие отказы уменьшают общую
медиану 3B — это не доказательство более быстрой генерации текста.

CPU-контроль тем же runtime: каждый четвёртый вопрос dev, 12 вопросов
(9 answerable + 3 OOD), GPU layers=0. Это диагностическая подвыборка:

| Модель | term_hit /9 | cited_evidence /9 | Ответила /9 | Все ответы, с | Содержательные, с | Первый токен, с |
|---|---:|---:|---:|---:|---:|---:|
| 1.5B | 2 | 1 | 7 | 5,48 | 8,90 | 0,80 |
| 3B | 2 | 1 | 4 | 2,51 | 3,94 | 1,40 |
| 4B | 4 | 3 | 7 | 15,73 | 18,31 | 2,14 |

Медианы — время LLM-вызовов, включая retry, без загрузки модели, поиска
и подготовки промпта. Первый токен отсчитывается от начала streaming-запроса;
это **не** полное время от ввода вопроса. Результаты генерации используют
замороженный поиск; перенесённая `retrieval_s` не измеряет текущий поиск.
Один последовательный прогон на домашнем ноутбуке не измеряет throughput
сервера под нагрузкой. Сырые ответы, fingerprints runtime и поштучные timings
доступны в `results/quality/generators-*`.

### Финальный test_v2

Выбор 4B зафиксирован до генерации этих 100 ответов. Обе модели получили
те же замороженные top-5 от исходного dense-small; fine-tuned поиск не включён.

| Модель | term_hit /80 | cited_evidence /80 | Отказы OOD /20 | Ответила /80 | Генерация: все, с | Содержательные, с | Первый токен, с |
|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen2.5-1.5B | 42 | 36 | 17 | 60 | 0,71 | 1,25 | 0,33 |
| Qwen3-4B | 55 | 56 | 20 | 77 | 2,54 | 2,82 | 0,91 |

Разница 4B−1.5B: **term_hit +16,25 п.п., CI [+6,25; +26,25]**,
**cited_evidence +25 п.п., CI [+13,75; +36,25]**. Интервалы не включают ноль,
но сохраняются ограничения синтетического набора и автоматических прокси-метрик.
При наличии доказательства в контексте ложных отказов стало 2 вместо 12.
Одновременно 4B ответила без размеченного доказательства в контексте на 17
вопросов против 10: цитата на другой фрагмент сама по себе не гарантирует
правильность. Эти ответы особенно нуждаются в независимой проверке.

Сырые ответы, метрики каждого вопроса и интервалы:
[`generators-v2-test/models.json`](../results/quality/generators-v2-test/models.json),
[`paired_deltas.json`](../results/quality/generators-v2-test/paired_deltas.json).

## Проверки реализации

97 тестов прошли: RRF и повторные голоса, парный bootstrap, κ и вырожденные
метки, защита train от eval-страниц, byte-for-byte пересборка v2, hash всех
GGUF shards, отмена streaming-кеша, нормализация ссылок и прежние проверки
конвейера. `rag check --skip-models` прошёл на корпусе/индексе; streaming CLI
проверен с mock, реальные генераторные прогоны также читают SSE поток.
Контрольные суммы входов и отсутствие пересечения train/eval проверены отдельно:
[`validation.json`](../results/quality/validation.json).

## Запуск и воспроизведение

Базовый CPU-профиль `configs/rag.yaml` сохранён. Более крупная модель доступна
через отдельный `configs/rag-qwen4b.yaml`: сначала запустите локальный сервер
в одном терминале, затем CLI в другом. Установите официальный llama.cpp
сборки b11669 для своей платформы; CUDA runtime должен быть доступен загрузчику.

```powershell
python scripts/download_quality_models.py qwen4b
llama-server -m models/qwen4b/Qwen3-4B-Q4_K_M.gguf --host 127.0.0.1 --port 8093 -c 4096 -ngl 99 --parallel 1 --threads 6 --no-ui
python -m rag --config configs/rag-qwen4b.yaml ask --stream "как узнать длину строки"
```

Для CPU используйте `-ngl 0`. Стриминг поддерживается и старым локальным
`llama_cpp` бэкендом: `python -m rag ask --stream "как узнать длину строки"`.
Пока идёт генерация, текст предварительный. Ссылки и отказ нормализуются
в конце; при отличии показывается итоговый ответ. Прерванный поток не
сохраняется в кеше. `--stream` и `--json` взаимоисключающие.

[Colab notebook](../notebooks/quality_colab.ipynb) содержит сборку корпуса,
обучение и GPU-индексацию. Сам notebook в Colab не запускался; тот же
скриптовый путь выполнен локально на GPU. Веса/индексы не включены в Git.
После обычного quickstart из README:

```powershell
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements-training.txt
python scripts/download_quality_models.py e5-base
python scripts/build_index.py --model e5-base --device cuda --precision float16
python scripts/train_e5.py --train data/train/synthetic-v1/questions.jsonl --review data/train/synthetic-v1/auto_review.csv --exclude data/eval/qa.jsonl --exclude data/eval/qa-v2.jsonl --reserved-docs data/eval/candidates-v2-json/reserved_docs.json --batch-size 16 --epochs 2
python scripts/build_index.py --model e5-finetuned --device cuda --precision float16
python scripts/benchmark_quality.py --phase retrieval --systems dense-small dense-tuned --qa data/eval/qa-v2.jsonl --split dev_v2 --output work/repro-dev-v2 --reranker-device cuda
```

Для гибридного поиска в пользовательском YAML задайте
`retriever.hybrid: {enabled: true, depth: 100, rrf_k: 60}` и
`reranker.candidates: 60`. Для дообученного поиска —
`retriever.model: e5-finetuned` после сборки его индекса. Эти варианты
доступны для экспериментов; подтверждённого преимущества перед default нет.

Повторение результатов требует тех же весов и версий; GPU FP16 и разные
runtime могут изменить выдачу. Для нового прогона выбирайте новый `--output`:
скрипты не затирают сырые ответы и отвергают изменившиеся вопросы/артефакты.
Для полного старого dev перечислите системы из `configs/quality.yaml`;
HyDE требует отдельного запущенного 3B-сервера на 8091.

```powershell
python scripts/benchmark_local_models.py --server-exe path/to/llama-server.exe --model qwen1.5b --model qwen3b --model qwen4b --output work/repro-generators
python scripts/refresh_generation_reports.py --folder work/repro-generators
python scripts/assemble_eval_v2.py --output work/qa-v2-rebuilt.jsonl
```

Последняя команда воспроизводит финальный набор из сохранённых кандидатов
и явных решений, SHA256 `e48e76de69a971d53d67c20dd021a3b25f1d0ed9cc0c2339624eefac7d96f7a0`.
Для новой генерации кандидатов используйте `scripts/generate_synthetic.py`;
`--purpose train` требует `--exclude` нового eval и `--reserved-docs` его
резерва. `filter_training.py` проверяет форму/утечки имени ответа, но не смысл.

После независимой проверки CSV пересчитайте согласие:

```powershell
python scripts/review_answers.py agree --review path/to/human_review.csv --verdicts results/quality/review/judge7b_structured.jsonl --output work/human-agreement.json
```

Укажите `reviewer` и `reviewer_type=human`; не переименовывайте assistant-метки
в human. Для утверждения нового eval после человеческой проверки предусмотрен
`scripts/finalize_eval.py`, сохраняющий происхождение решений и точные цитаты.
