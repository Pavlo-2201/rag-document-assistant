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


def test_html_element_macro_survives_tag_stripping():
    # раньше «<canvas>» из макроса удалялся следующим шагом как HTML-тег
    doc = parse_markdown('Элемент {{HTMLElement("canvas")}}, добавленный в HTML5, рисует <b>графику</b>.\n')
    assert doc.text.strip() == "Элемент <canvas>, добавленный в HTML5, рисует графику."


def test_tags_inside_inline_code_are_kept():
    doc = parse_markdown("Примените к подзаголовку `<h2>` курсив, <span>а тег</span> уберите.\n")
    assert doc.text.strip() == "Примените к подзаголовку `<h2>` курсив, а тег уберите."


def test_macro_with_unquoted_argument():
    doc = parse_markdown("Вместо 404 используйте {{HTTPStatus(410)}} (Gone).\n")
    assert doc.text.strip() == "Вместо 404 используйте 410 (Gone)."


def test_escaped_angle_brackets_are_text_not_tags():
    # «\<details>» в Markdown — текст; раньше тег удалялся и оставался «\»
    doc = parse_markdown("Элемент \\<details> или [\\<script>](/ru/docs/x), значения a \\| b, а `\\<x>` — код.\n")
    assert doc.text.strip() == "Элемент <details> или <script>, значения a | b, а `\\<x>` — код."


def test_html_entities_decoded_outside_code_only():
    # раньше «&lt;length&gt;» попадало во фрагменты как есть (1107 сущностей в 190 документах)
    doc = parse_markdown("Значение &lt;length&gt;&nbsp;или&#x20;<b>auto</b> &amp; `&lt;b&gt;`, адрес ?a=1&para=2.\n")
    assert doc.text.strip() == "Значение <length> или auto & `&lt;b&gt;`, адрес ?a=1&para=2."
