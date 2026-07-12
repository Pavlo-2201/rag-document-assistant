import hashlib
import json

from semsearch.corpus import make_doc_id

from rag.ingest import (
    documents_fingerprint,
    ingest_directory,
    ingest_mdn,
    load_documents,
    write_documents,
)

SOURCE = {"repo": "mdn/translated-content", "ref": "abc123", "license": "CC-BY-SA-2.5"}


def make_raw_corpus(tmp_path, pages: dict[str, str]):
    rows = []
    for slug, markdown in pages.items():
        path = f"files/ru/{slug.lower()}/index.md"
        (tmp_path / path).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / path).write_text(markdown, encoding="utf-8")
        rows.append({
            "slug": slug,
            "source_path": path,
            "url": f"https://developer.mozilla.org/ru/docs/{slug}",
            "raw_sha256": hashlib.sha256(markdown.encode("utf-8")).hexdigest(),
        })
    with (tmp_path / "manifest.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


PAGES = {
    "Web/A": "---\ntitle: Страница A\n---\n\nВступление.\n\n## Раздел один\n\nТекст.\n",
    "Web/B": "---\ntitle: Страница B\n---\n\nТекст B.\n",
}


def test_mdn_doc_id_is_the_same_as_in_semantic_search_engine(tmp_path):
    make_raw_corpus(tmp_path, PAGES)
    documents = {d.slug: d for d in ingest_mdn(tmp_path, SOURCE)}
    assert documents["Web/A"].doc_id == make_doc_id("Web/A")
    assert documents["Web/A"].title == "Страница A"
    assert documents["Web/A"].source_ref == "mdn/translated-content@abc123"


def test_section_url_is_page_url_with_anchor(tmp_path):
    make_raw_corpus(tmp_path, PAGES)
    doc = next(d for d in ingest_mdn(tmp_path, SOURCE) if d.slug == "Web/A")
    urls = [doc.section_url(s) for s in doc.sections]
    assert urls == [
        "https://developer.mozilla.org/ru/docs/Web/A",
        "https://developer.mozilla.org/ru/docs/Web/A#раздел_один",
    ]


def test_changed_raw_file_is_detected(tmp_path):
    make_raw_corpus(tmp_path, PAGES)
    (tmp_path / "files/ru/web/b/index.md").write_text("подмена", encoding="utf-8")
    try:
        ingest_mdn(tmp_path, SOURCE)
    except ValueError as error:
        assert "изменился" in str(error)
    else:
        raise AssertionError("изменение исходника не обнаружено")


def test_ids_do_not_depend_on_order_and_text(tmp_path):
    make_raw_corpus(tmp_path / "one", PAGES)
    make_raw_corpus(tmp_path / "two", dict(reversed(list({**PAGES, "Web/B": "Другой текст B."}.items()))))
    one = {d.slug: d for d in ingest_mdn(tmp_path / "one", SOURCE)}
    two = {d.slug: d for d in ingest_mdn(tmp_path / "two", SOURCE)}
    assert {s: d.doc_id for s, d in one.items()} == {s: d.doc_id for s, d in two.items()}
    assert one["Web/B"].text_sha256 != two["Web/B"].text_sha256
    assert documents_fingerprint(one.values()) != documents_fingerprint(two.values())


def test_directory_ingest_markdown_and_text(docs_dir):
    documents = {d.source_path: d for d in ingest_directory(docs_dir, base_url="https://example.org/docs")}
    assert set(documents) == {"guide/arrays.md", "notes.txt"}
    md = documents["guide/arrays.md"]
    assert md.format == "markdown" and md.title == "Массивы"
    assert md.url == "https://example.org/docs/guide/arrays.md"
    txt = documents["notes.txt"]
    assert txt.format == "text" and txt.text == "Простой текстовый документ.\n\nВторой абзац про возврат товара."
    assert txt.doc_id.startswith("doc-")


def test_write_and_load_roundtrip_is_deterministic(tmp_path, docs_dir):
    documents = ingest_directory(docs_dir)
    write_documents(tmp_path / "a.jsonl", documents)
    write_documents(tmp_path / "b.jsonl", list(reversed(documents)))
    assert (tmp_path / "a.jsonl").read_bytes() == (tmp_path / "b.jsonl").read_bytes()
    loaded = load_documents(tmp_path / "a.jsonl")
    assert [d.to_dict() for d in loaded] == [d.to_dict() for d in sorted(documents, key=lambda d: d.doc_id)]
