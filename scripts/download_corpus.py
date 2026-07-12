#!/usr/bin/env python3
"""Загрузка исходного Markdown страниц MDN: configs/corpus.yaml -> data/raw/mdn-ru/.

Набор страниц и отбор те же, что в semantic-search-engine. В отличие от него
сохраняется исходный Markdown каждой страницы (разбор — отдельный шаг, см.
scripts/ingest.py), а ref источника зафиксирован commit'ом.

После загрузки считается отпечаток корпуса способом semantic-search-engine и
сравнивается с его опубликованным отпечатком: совпадение подтверждает, что оба
проекта работают с одними и теми же документами.

    python scripts/download_corpus.py --config configs/corpus.yaml
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from semsearch.corpus import Document, corpus_fingerprint  # noqa: E402
from semsearch.mdn import extract_title, markdown_to_text, slug_to_source_path  # noqa: E402


def select_slugs(slugs: list[str], selection: dict) -> list[str]:
    include = tuple(selection.get("include_prefixes") or ())
    exclude = tuple(selection.get("exclude_prefixes") or ())
    return [
        slug
        for slug in slugs
        if not (exclude and slug.startswith(exclude)) and (not include or slug.startswith(include))
    ]


def fetch(session: requests.Session, url: str, timeout: int, retries: int) -> str | None:
    last_error: Exception | None = None
    for _ in range(retries):
        try:
            response = session.get(url, timeout=timeout)
        except requests.RequestException as error:
            last_error = error
            continue
        if response.status_code == 404:
            return None
        if response.ok:
            response.encoding = "utf-8"
            return response.text
        last_error = requests.HTTPError(f"{response.status_code} {url}")
    raise RuntimeError(f"не удалось загрузить {url}: {last_error}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/corpus.yaml")
    parser.add_argument("--limit", type=int, help="только первые N страниц (для отладки)")
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    source = config["source"]
    download = config.get("download", {})
    raw_dir = Path(config["paths"]["raw_dir"])
    timeout = int(download.get("timeout_seconds", 30))
    retries = int(download.get("retries", 3))

    def raw_url(path: str) -> str:
        return f"{source['base_url']}/{source['repo']}/{source['ref']}/{path}"

    session = requests.Session()
    session.headers["User-Agent"] = "rag-document-assistant/0.1 (educational project)"
    listing = fetch(session, raw_url(source["listing_path"]), timeout, retries)
    if listing is None:
        raise RuntimeError("список страниц не найден")
    all_slugs = sorted(json.loads(listing).keys())
    slugs = select_slugs(all_slugs, config.get("selection", {}))
    if args.limit:
        slugs = slugs[: args.limit]
    print(f"страниц в источнике: {len(all_slugs)}, после отбора: {len(slugs)}")

    def work(slug: str) -> tuple[str, str, str | None]:
        path = slug_to_source_path(slug, source["docs_prefix"])
        return slug, path, fetch(session, raw_url(path), timeout, retries)

    min_chars = int(config.get("selection", {}).get("min_chars", 0))
    manifest, missing, too_short, reference_docs = [], [], [], []
    with ThreadPoolExecutor(max_workers=int(download.get("workers", 8))) as pool:
        for slug, path, markdown in tqdm(pool.map(work, slugs), total=len(slugs), desc="страницы"):
            if markdown is None:
                missing.append(slug)
                continue
            # тот же отбор, что в semantic-search-engine: длина очищенного текста
            reference_text = markdown_to_text(markdown)
            if len(reference_text) < min_chars:
                too_short.append(slug)
                continue
            target = raw_dir / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(markdown, encoding="utf-8")
            reference_docs.append(
                Document.create(
                    slug=slug,
                    title=extract_title(markdown, slug),
                    text=reference_text,
                    url=source["page_url_template"].format(slug=slug),
                    source_path=path,
                    source_repo=source["repo"],
                    source_ref=source["ref"],
                    license=source["license"],
                )
            )
            manifest.append(
                {
                    "slug": slug,
                    "source_path": path,
                    "url": source["page_url_template"].format(slug=slug),
                    "raw_sha256": hashlib.sha256(markdown.encode("utf-8")).hexdigest(),
                }
            )

    manifest.sort(key=lambda row: row["slug"])
    with (raw_dir / "manifest.jsonl").open("w", encoding="utf-8") as handle:
        for row in manifest:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    fingerprint = corpus_fingerprint(reference_docs)
    expected = config.get("reference_fingerprint_sha256")
    meta = {
        "corpus_name": config.get("name"),
        "downloaded_at": dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source": {k: source[k] for k in ("repo", "ref", "license", "license_url", "attribution")},
        "slugs_in_source": len(all_slugs),
        "slugs_selected": len(slugs),
        "documents": len(manifest),
        "missing_pages": len(missing),
        "dropped_too_short": len(too_short),
        "semsearch_fingerprint_sha256": fingerprint,
        "matches_semantic_search_engine": fingerprint == expected,
    }
    (raw_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"документов: {len(manifest)}, нет файла: {len(missing)}, короче {min_chars}: {len(too_short)}")
    print(f"отпечаток (как в semantic-search-engine): {fingerprint}")
    print("совпадает с semantic-search-engine" if fingerprint == expected else f"НЕ совпадает с {expected}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
