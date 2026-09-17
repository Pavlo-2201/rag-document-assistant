from rag.judge_v2 import structured_verdict


def test_verdict_field_wins_over_label_mentioned_in_reason():
    verdict = structured_verdict('{"reason":"Слово НЕТ в примере не меняет оценку", "verdict":"ДА"}')
    assert verdict.verdict == 'ДА'
    assert structured_verdict('{"reason":"оборванный JSON"').verdict == '?'
