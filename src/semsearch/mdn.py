"""Загрузка русскоязычных страниц MDN Web Docs и приведение их к тексту.

Страницы берутся из публичного репозитория переводов mdn/translated-content через
raw.githubusercontent.com: это обычный HTTPS без токенов и без GitHub API, так что
загрузка повторяется в любом окружении с доступом в сеть.
"""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Iterable

import requests

from .corpus import Document

# Символы, которые MDN заменяет в именах каталогов (см. mdn/content, slugToFolder).
SLUG_CHAR_REPLACEMENTS = {
    "*": "_star_",
    "::": "_doublecolon_",
    ":": "_colon_",
    "?": "_doublequestion_",
    "<": "_lt_",
    ">": "_gt_",
    '"': "_doublequote_",
}

FRONTMATTER_RE = re.compile(r"\A---\n(.*?)\n---\n", re.DOTALL)
TITLE_RE = re.compile(r"^title:\s*(.+?)\s*$", re.MULTILINE)
FENCED_CODE_RE = re.compile(r"^```.*?^```", re.DOTALL | re.MULTILINE)
INDENTED_MACRO_RE = re.compile(r"\{\{[^{}]*\}\}")
HTML_TAG_RE = re.compile(r"<[^>\n]{1,200}>")
LINK_RE = re.compile(r"\[([^\]]*)\]\([^)]*\)")
EMPHASIS_RE = re.compile(r"(?<!\w)(?:\*\*|\*|__)(.+?)(?:\*\*|\*|__)(?!\w)", re.DOTALL)
BLANK_LINES_RE = re.compile(r"\n{3,}")
SPACES_RE = re.compile(r"[ \t]{2,}")


def slug_to_folder(slug: str) -> str:
    """Slug страницы -> путь каталога внутри files/ru."""
    folder = slug
    for char, replacement in SLUG_CHAR_REPLACEMENTS.items():
        folder = folder.replace(char, replacement)
    return folder.lower()


def slug_to_source_path(slug: str, docs_prefix: str) -> str:
    return f"{docs_prefix}/{slug_to_folder(slug)}/index.md"


def extract_title(markdown: str, slug: str) -> str:
    match = FRONTMATTER_RE.search(markdown)
    if match:
        title_match = TITLE_RE.search(match.group(1))
        if title_match:
            return title_match.group(1).strip().strip('"').strip("'")
    return slug.rsplit("/", 1)[-1].replace("_", " ")


def markdown_to_text(markdown: str) -> str:
    """Убрать из Markdown то, что мешает поиску по смыслу.

    Выбрасываются frontmatter, блоки кода, макросы KumaScript ({{jsxref(...)}} и
    подобные) и HTML-теги; из ссылок остаётся только видимый текст. Блоки кода
    удаляются сознательно: они состоят из англоязычных идентификаторов, сильно
    различаются по длине и для лексического baseline работают как шум. Заголовки
    и подписи сохраняются — это осмысленный текст.
    """
    text = FRONTMATTER_RE.sub("", markdown)
    text = FENCED_CODE_RE.sub(" ", text)
    text = INDENTED_MACRO_RE.sub(" ", text)
    text = HTML_TAG_RE.sub(" ", text)
    text = LINK_RE.sub(r"\1", text)
    text = text.replace("`", "")
    text = EMPHASIS_RE.sub(r"\1", text)
    text = re.sub(r"^\s{0,3}#{1,6}\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*[-*>]\s+", "", text, flags=re.MULTILINE)
    text = SPACES_RE.sub(" ", text)
    text = BLANK_LINES_RE.sub("\n\n", text)
    return text.strip()


@dataclass
class MdnSource:
    repo: str
    ref: str
    docs_prefix: str
    listing_path: str
    base_url: str
    page_url_template: str
    license: str
    timeout: int = 30
    retries: int = 3

    def raw_url(self, path: str) -> str:
        return f"{self.base_url}/{self.repo}/{self.ref}/{path}"

    def _get(self, session: requests.Session, path: str) -> str | None:
        url = self.raw_url(path)
        last_error: Exception | None = None
        for attempt in range(self.retries):
            try:
                response = session.get(url, timeout=self.timeout)
            except requests.RequestException as error:  # сеть бывает нестабильной
                last_error = error
                continue
            if response.status_code == 404:
                return None
            if response.ok:
                response.encoding = "utf-8"
                return response.text
            last_error = requests.HTTPError(f"{response.status_code} {url}")
        if last_error is not None and self.retries:
            raise RuntimeError(f"не удалось загрузить {url}: {last_error}")
        return None

    def list_slugs(self, session: requests.Session) -> list[str]:
        body = self._get(session, self.listing_path)
        if body is None:
            raise RuntimeError(f"список страниц не найден: {self.listing_path}")
        import json

        return sorted(json.loads(body).keys())

    def fetch_document(self, session: requests.Session, slug: str) -> Document | None:
        source_path = slug_to_source_path(slug, self.docs_prefix)
        markdown = self._get(session, source_path)
        if markdown is None:
            return None
        text = markdown_to_text(markdown)
        if not text:
            return None
        return Document.create(
            slug=slug,
            title=extract_title(markdown, slug),
            text=text,
            url=self.page_url_template.format(slug=slug),
            source_path=source_path,
            source_repo=self.repo,
            source_ref=self.ref,
            license=self.license,
        )

    def fetch_documents(
        self,
        slugs: Iterable[str],
        workers: int = 8,
        progress: bool = True,
    ) -> tuple[list[Document], list[str]]:
        slugs = list(slugs)
        documents: list[Document] = []
        missing: list[str] = []
        session = requests.Session()
        session.headers["User-Agent"] = "semantic-search-engine/0.1 (educational project)"

        def work(slug: str) -> tuple[str, Document | None]:
            return slug, self.fetch_document(session, slug)

        iterator = ThreadPoolExecutor(max_workers=workers).map(work, slugs)
        if progress:
            from tqdm import tqdm

            iterator = tqdm(iterator, total=len(slugs), desc="страницы MDN")
        for slug, document in iterator:
            if document is None:
                missing.append(slug)
            else:
                documents.append(document)
        return documents, missing
