from rag.error_analysis import attribute, parser_artifacts, summarize_errors
from rag.judge import Judge, parse_verdict
from rag.llm import MockBackend


def score(answerable=True, refused=False, term_hit=False, evidence_in_context=False):
    return {"answerable": answerable, "refused": refused, "term_hit": term_hit, "evidence_in_context": evidence_in_context}


def test_attribute_stage():
    assert attribute(score(term_hit=True), 1, 20, 5) == "ok"
    assert attribute(score(evidence_in_context=True), 2, 20, 5) == "generation"
    assert attribute(score(refused=True, evidence_in_context=True), 2, 20, 5) == "generation"
    assert attribute(score(), 12, 20, 5) == "ranking"
    assert attribute(score(), None, 20, 5) == "retrieval"
    assert attribute(score(answerable=False, refused=True), None, 20, 5) == "ok"
    assert attribute(score(answerable=False), None, 20, 5) == "generation"


def kinds(texts):
    return [a["kind"] for a in parser_artifacts(texts)]


def test_parser_artifacts_by_kind_and_ignore_code():
    assert parser_artifacts(["Элемент \\ позволяет включить"]) == [
        {"kind": "backslash", "context": "Элемент \\ позволяет включить"}]
    assert kinds(["Используйте `<h2>` и ``"]) == ["empty_code"]
    assert kinds(["тег &lt;div&gt; и&#x20;пробел"]) == ["html_entity"] * 3
    # строка матрицы TeX: «\\» в ней — разделитель строк, отмечается одна формула
    assert kinds(["\\begin{pmatrix} a &#x26; c \\ b &#x26; d \\end{pmatrix}"]) == ["html_entity", "html_entity", "tex"]
    assert kinds(["результат:\\\nследующая строка"]) == ["backslash"]
    assert kinds(["```js\nconst s = 'a\\ b &lt;';\n```", "путь C:\\\\Users", "`a``b`", "`&lt;`"]) == []
    rows = [{"stage": "ok", "answerable": True, "parser_artifacts": []},
            {"stage": "retrieval", "answerable": True, "parser_artifacts": [{"kind": "tex", "context": "x"}]}]
    summary = summarize_errors(rows)
    assert summary["errors_with_parser_artifacts_in_context"] == 1
    assert summary["parser_artifacts_by_kind"] == {"tex": 1}


def test_parse_verdict_and_judge_without_sources():
    assert parse_verdict("ДА.").verdict == "ДА"
    assert parse_verdict("Частично, потому что").verdict == "ЧАСТИЧНО"
    assert parse_verdict("нет").verdict == "НЕТ"
    assert parse_verdict("не знаю").verdict == "?"
    judge = Judge(MockBackend())
    assert judge.groundedness("q", "ответ", []).verdict == "НЕТ"   # без ссылок модель не вызывается
    assert "критерии" in judge.config()["groundedness"]["system"].lower()
