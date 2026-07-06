# Пакет `semsearch` взят из semantic-search-engine

Копия `src/semsearch` из предыдущего проекта портфолио (semantic-search-engine,
версия 1.0.0) без изменений. Пакет не опубликован на PyPI, поэтому он
скопирован, а не подключён зависимостью; правки в нём не делаются, всё новое
живёт в пакете `rag`.

Что отсюда используется:

| модуль | для чего |
|---|---|
| `corpus.py` | `make_doc_id` (doc_id = `mdnru-` + sha1(slug)[:12]), отпечаток корпуса |
| `mdn.py` | `slug_to_folder`, `markdown_to_text`: путь страницы в репозитории MDN и проверка, что корпус совпадает с корпусом semantic-search-engine |
| `text.py`, `lexical.py` | анализатор русского текста и BM25 (этап retrieval) |
| `embedding.py`, `faiss_index.py` | LSA, sentence-transformers и индекс FAISS с соответствием ID (этап retrieval) |
| `evaluation.py` | метрики Recall@K, MRR@K, nDCG@K |

Проверенная версия фиксируется не commit'ом (история исходного репозитория может
переписываться), а sha256 файлов:

```
dde3eeb3100127f450f2089f1e653787f25414636861a799decefaa2a07ddcf9  __init__.py
0503f448df1f8c68190f8967a50c2cb0b0b02edd645b06a2b514e3541efcff2f  corpus.py
1c2e43714e94738b57366cb5ea8c46b2bf926ff33ed7f64274223371c3b4f6e5  dense.py
eb116f42bded9f0bd884572b8513f6618f44247fa271c07951a18adc5a473ca2  embedding.py
dda2ce603abe19dbe8ecd2baf7782d38de67514646cc9abacd80f1773cb5b868  evaluation.py
3027d47c05acf17653b8744e8bdfcdf787b2fbaebb0e13b37545cbf4896f80dc  faiss_index.py
4e87063dc1378e9cdfc40bc7d936793d7780eab84f15f46d6dc6c13f9d88ec64  latency.py
d5e26c6be6ac89df8f87d0f22c25c0680177fff2187ab14abf6fb1595ed8e88a  lexical.py
6c229a16242f4bebee49e1330e04ef5942e2d6fea2493aa9e4177c0a1dcad29d  mdn.py
6df947fd293141fe9f9b48f30747b862c2790a18c354a4baf51c338f1d9aaa4f  pipeline.py
5d8554e03f270a6734fd3ca2db2eeec5479c5b378fec0fa02fd8e78ebdd937ca  retriever.py
e2dfac6c7349c21e80f35e70cde0015a764f5a63cbc5f6659b3c87fcaab750e3  text.py
```

Проверка: `sha256sum -c` по этому списку из каталога `src/semsearch`
(тест `tests/test_vendored.py` делает то же самое).
