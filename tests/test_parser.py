from rag.parser import make_anchor, parse_markdown, parse_plain_text

MARKDOWN = (Path := __import__("pathlib").Path)(__file__).parent.joinpath("fixtures/docs/guide/arrays.md").read_text(encoding="utf-8")


def test_frontmatter_title_and_service_macros_removed():
    doc = parse_markdown(MARKDOWN)
    assert doc.title == "Массивы"
    assert "JSRef" not in doc.text
    assert "---" not in doc.text.split("\n")[0]


def test_xref_macro_and_inline_marker_become_text():
    doc = parse_markdown(MARKDOWN)
    assert "количество элементов массива." in doc.text
    assert "`index` (необязательный)" in doc.text


def test_code_block_kept_verbatim_and_not_split_into_sections():
    doc = parse_markdown(MARKDOWN)
    assert "console.log(a.length); // 3" in doc.text
    assert "// ## это не заголовок, а комментарий в коде" in doc.text
    assert all("это не заголовок" not in s.title for s in doc.sections)


def test_sections_cover_text_without_gaps_and_slice_starts_with_heading():
    doc = parse_markdown(MARKDOWN)
    assert doc.sections[0].start == 0
    for previous, current in zip(doc.sections, doc.sections[1:]):
        assert previous.end == current.start
    assert doc.sections[-1].end == len(doc.text)
    for section in doc.sections[1:]:
        assert doc.text[section.start:section.end].startswith(section.title)


def test_heading_path_and_unique_anchors():
    doc = parse_markdown(MARKDOWN)
    titles = [(s.title, s.level, s.anchor, s.path) for s in doc.sections]
    assert ("", 0, "", []) == titles[0]
    assert ("Параметры", 3, "параметры", ["Длина массива", "Параметры"]) in titles
    anchors = [s.anchor for s in doc.sections if s.title == "Длина массива"]
    assert anchors == ["длина_массива", "длина_массива_2"]


def test_links_quotes_and_bold_cleaned():
    doc = parse_markdown(MARKDOWN)
    assert "Примечание: заметка в цитате, см. документацию." in doc.text
    assert "](/ru/docs" not in doc.text


def test_make_anchor():
    assert make_anchor("Browser compatibility") == "browser_compatibility"
    assert make_anchor("Пример: обобщённое использование `map`") == "пример_обобщённое_использование_map"


def test_plain_text_single_section():
    doc = parse_plain_text("a\n\n\n\nb", "t")
    assert doc.text == "a\n\nb"
    assert len(doc.sections) == 1 and doc.sections[0].end == len(doc.text)
