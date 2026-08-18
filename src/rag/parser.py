"""Разбор Markdown-документа в текст с сохранённой структурой разделов.

В отличие от очистки текста для поиска в semantic-search-engine, здесь
сохраняется то, что нужно для ответа со ссылками:

* заголовки разделов — по ним строятся ссылки на раздел (`url#anchor`) и
  нарезка документа на фрагменты;
* блоки кода — ответ на вопрос «как сделать» часто находится именно в примере;
* inline-код в обратных кавычках.

Макросы KumaScript (`{{jsxref("Array")}}` и т. п.) заменяются видимым текстом,
служебные (`{{Compat}}`, `{{JSRef}}`) удаляются. Для каждого раздела хранятся
границы в итоговом тексте: `text[start:end]` — ровно текст раздела.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

FRONTMATTER_RE = re.compile(r"\A---\n(.*?)\n---\n", re.DOTALL)
HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
FENCE_RE = re.compile(r"^\s*(?:[-*+]\s+|\d+\.\s+)?(```|~~~)")
MACRO_RE = re.compile(r"\{\{\s*([A-Za-z_][\w-]*)\s*(?:\((.*?)\))?\s*\}\}")
MACRO_ARG_RE = re.compile(r'"([^"]*)"|\'([^\']*)\'')
IMAGE_RE = re.compile(r"!\[([^\]]*)\]\([^)]*\)")
LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]*\)")
HTML_TAG_RE = re.compile(r"</?[a-zA-Z][^>\n]{0,200}>")
INLINE_CODE_RE = re.compile(r"(`+)(.+?)\1")
HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
BOLD_RE = re.compile(r"\*\*(.+?)\*\*")
DEFINITION_RE = re.compile(r"^(\s*)- : ")
QUOTE_RE = re.compile(r"^\s{0,3}>\s?")

# Макросы-ссылки: показываются как текст ссылки (второй аргумент, если он есть)
XREF_MACROS = {
    "domxref", "jsxref", "cssxref", "glossary", "svgattr", "svgelement", "httpheader",
    "httpmethod", "httpstatus", "webextapiref", "mathmlelement", "htmlattrxref",
    "event", "rfc", "jsxref", "cssxref", "domxref", "xref", "httpstatusref",
}
# Пометки в строке: сохраняются по-русски, потому что отвечают на вопросы вида
# «этот параметр обязательный?»
INLINE_MARKERS = {
    "optional_inline": "(необязательный)",
    "readonlyinline": "(только для чтения)",
    "experimental_inline": "(экспериментальная возможность)",
    "deprecated_inline": "(устарело)",
    "non-standard_inline": "(нестандартное)",
    "non_standard_inline": "(нестандартное)",
}


def parse_frontmatter(markdown: str) -> tuple[dict, str]:
    match = FRONTMATTER_RE.match(markdown)
    if not match:
        return {}, markdown
    meta = {}
    for line in match.group(1).splitlines():
        if ":" in line and not line.startswith((" ", "-")):
            key, value = line.split(":", 1)
            meta[key.strip()] = value.strip().strip('"').strip("'")
    return meta, markdown[match.end():]


def _macro_args(raw: str | None) -> list[str]:
    """Аргументы макроса: в кавычках или, как {{HTTPStatus(410)}}, без них."""
    quoted = [a or b for a, b in MACRO_ARG_RE.findall(raw or "")]
    if quoted:
        return quoted
    return [part.strip() for part in (raw or "").split(",") if part.strip()]


def _macro_text(match: re.Match) -> str:
    name = match.group(1).lower()
    if name in INLINE_MARKERS:
        return f" {INLINE_MARKERS[name]}"
    if name == "htmlelement":
        args = _macro_args(match.group(2))
        return f"<{args[0]}>" if args else ""
    if name in XREF_MACROS:
        args = _macro_args(match.group(2))
        if not args:
            return ""
        if len(args) > 1 and args[1].strip():
            return args[1]
        target = args[0]
        if name == "rfc":
            return f"RFC {target}"
        # jsxref("Global_Objects/Array") показывается как Array
        return target.rsplit("/", 1)[-1] if name == "jsxref" else target
    return ""  # служебные макросы: сайдбары, таблицы совместимости, live-примеры


def _strip_tags_outside_code(line: str) -> str:
    """Убрать HTML-теги разметки, но не внутри `кода`: там `<canvas>` — содержание."""
    parts, last = [], 0
    for match in INLINE_CODE_RE.finditer(line):
        parts.append(HTML_TAG_RE.sub("", line[last:match.start()]))
        parts.append(match.group(0))
        last = match.end()
    parts.append(HTML_TAG_RE.sub("", line[last:]))
    return "".join(parts)


def clean_inline(line: str) -> str:
    # теги убираются до раскрытия макросов: {{HTMLElement("canvas")}} даёт
    # «<canvas>», и этот текст не должен удаляться как тег
    line = _strip_tags_outside_code(line)
    line = MACRO_RE.sub(_macro_text, line)
    line = IMAGE_RE.sub(r"\1", line)
    line = LINK_RE.sub(r"\1", line)
    line = BOLD_RE.sub(r"\1", line)
    line = DEFINITION_RE.sub(r"\1— ", line)
    line = QUOTE_RE.sub("", line)
    # двойные пробелы внутри строки остаются после удалённых макросов;
    # отступ в начале строки (вложенные списки) сохраняется
    line = re.sub(r"(?<=\S) {2,}", " ", line)
    return line.rstrip()


def make_anchor(heading: str) -> str:
    """Якорь раздела, как его строит MDN: нижний регистр, пробелы → «_».

    Это приближение к алгоритму сайта (для обычных заголовков совпадает); если
    якорь не сработает, ссылка всё равно ведёт на нужную страницу.
    """
    text = heading.replace("`", "").strip().lower()
    text = re.sub(r"[\"'.,:;!?()\[\]{}<>]", "", text)
    return re.sub(r"\s+", "_", text).strip("_")


@dataclass
class Section:
    title: str          # текст заголовка; у вступления до первого заголовка — ""
    level: int          # 1..6; у вступления 0
    anchor: str         # якорь для ссылки; у вступления ""
    path: list[str]     # цепочка заголовков: ["Синтаксис", "Параметры"]
    start: int          # границы раздела в ParsedDocument.text
    end: int

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ParsedDocument:
    title: str
    text: str
    sections: list[Section] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    def section_at(self, offset: int) -> Section:
        for section in self.sections:
            if section.start <= offset < section.end:
                return section
        return self.sections[-1]


def parse_markdown(markdown: str, default_title: str = "") -> ParsedDocument:
    """Markdown → текст и разделы с границами."""
    markdown = markdown.replace("\r\n", "\n")
    meta, body = parse_frontmatter(markdown)
    body = HTML_COMMENT_RE.sub("", body)
    title = meta.get("title") or default_title

    out_lines: list[str] = []
    offset = 0
    sections: list[Section] = []
    heading_stack: list[tuple[int, str]] = []
    used_anchors: dict[str, int] = {}
    in_code = False
    fence = ""

    def open_section(level: int, heading: str) -> None:
        if sections:
            sections[-1].end = offset
        anchor = make_anchor(heading) if heading else ""
        if anchor:
            count = used_anchors.get(anchor, 0) + 1
            used_anchors[anchor] = count
            if count > 1:
                anchor = f"{anchor}_{count}"
        sections.append(
            Section(heading, level, anchor, [h for _, h in heading_stack], offset, offset)
        )

    open_section(0, "")
    previous_blank = True
    quoted_code = False
    for raw in body.split("\n"):
        # блок кода может стоять внутри цитаты («> ```js»): маркер цитаты снимается
        unquoted = QUOTE_RE.sub("", raw) if (quoted_code or not in_code) else raw
        fence_match = FENCE_RE.match(unquoted)
        if in_code:
            line = unquoted
            if fence_match and fence_match.group(1) == fence:
                in_code = False
        elif fence_match:
            in_code, fence, line = True, fence_match.group(1), unquoted.strip()
            quoted_code = unquoted != raw
        else:
            heading = HEADING_RE.match(raw)
            if heading:
                level = len(heading.group(1))
                text = clean_inline(heading.group(2)).strip()
                while heading_stack and heading_stack[-1][0] >= level:
                    heading_stack.pop()
                heading_stack.append((level, text))
                if out_lines and not previous_blank:
                    out_lines.append("")
                    offset += 1
                open_section(level, text)
                line = text
            else:
                line = clean_inline(raw)
        blank = not line.strip()
        if blank and previous_blank:
            continue
        out_lines.append(line)
        offset += len(line) + 1
        previous_blank = blank

    text = "\n".join(out_lines)
    sections[-1].end = len(text)
    # пустое вступление (страница начинается с заголовка) не нужно
    sections = [s for s in sections if s.end > s.start or s.level > 0]
    for section in sections:
        section.end = min(section.end, len(text))
    return ParsedDocument(title=title, text=text, sections=sections, meta=meta)


def parse_plain_text(text: str, title: str) -> ParsedDocument:
    """Обычный текст: один раздел, абзацы нормализованы."""
    text = re.sub(r"\n{3,}", "\n\n", text.replace("\r\n", "\n")).strip()
    return ParsedDocument(title=title, text=text, sections=[Section("", 0, "", [], 0, len(text))])
