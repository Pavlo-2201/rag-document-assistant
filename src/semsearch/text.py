"""Нормализация русского текста для лексического поиска.

Для лексического поиска по-русски морфология важна: запрос «как получить длину
массива» и текст «возвращает длину массива» совпадут по словоформам, а «массивы» и
«массива» — нет. Поэтому слова приводятся к основе стеммером Snowball.
Векторный поиск в стемминге не нуждается и использует исходный текст.
"""

from __future__ import annotations

import functools
import re

TOKEN_RE = re.compile(r"[a-zа-яё0-9_]+", re.IGNORECASE)

# Короткий список русских служебных слов. Он сознательно небольшой: длинные списки
# стоп-слов удаляют слова, которые в документации значимы («все», «нет», «через»).
RUSSIAN_STOPWORDS = frozenset(
    """
    и в во не что он на я с со как а то все она так его но да ты к у же вы за бы
    по только ее мне было вот от меня еще нет о из ему теперь когда даже ну вдруг
    ли если уже или ни быть был него до вас нибудь опять уж вам ведь там потом себя
    ничего ей может они тут где есть надо ней для мы тебя их чем была сам чтоб без
    будто чего раз тоже себе под будет ж тогда кто этот того потому этого какой
    совсем ним здесь этом один почти мой тем чтобы нее сейчас были куда зачем всех
    никогда можно при наконец два об другой хоть после над больше тот через эти нас
    про всего них какая много разве три эту моя впрочем хорошо свою этой перед иногда
    лучше чуть том нельзя такой им более всегда конечно всю между это которые который
    также либо эта этих такие также
    """.split()
)

_WORD_CHAR_RE = re.compile(r"[а-яё]", re.IGNORECASE)


@functools.lru_cache(maxsize=1)
def _russian_stemmer():
    import snowballstemmer

    return snowballstemmer.stemmer("russian")


def normalize(text: str) -> str:
    """Приведение к нижнему регистру и склейка ё/е.

    В документации одно и то же слово встречается и с «ё», и с «е»
    («ещё»/«еще»), поэтому различие убирается до токенизации.
    """
    return text.lower().replace("ё", "е")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(normalize(text))


def stem_tokens(tokens: list[str]) -> list[str]:
    stemmer = _russian_stemmer()
    # Стеммер русского языка применяется только к словам с кириллицей: английские
    # идентификаторы («querySelector», «fetch») он портит без пользы.
    return [
        stemmer.stemWord(token) if _WORD_CHAR_RE.search(token) else token
        for token in tokens
    ]


def analyze(
    text: str,
    stem: bool = True,
    remove_stopwords: bool = True,
    min_token_len: int = 2,
) -> list[str]:
    """Полный разбор текста в список терминов для лексического индекса."""
    tokens = tokenize(text)
    if remove_stopwords:
        tokens = [t for t in tokens if t not in RUSSIAN_STOPWORDS]
    tokens = [t for t in tokens if len(t) >= min_token_len]
    if stem:
        tokens = stem_tokens(tokens)
    return tokens


def make_analyzer(
    stem: bool = True,
    remove_stopwords: bool = True,
    min_token_len: int = 2,
):
    """Фабрика анализаторов — в таком виде его принимает TfidfVectorizer."""

    def analyzer(text: str) -> list[str]:
        return analyze(
            text,
            stem=stem,
            remove_stopwords=remove_stopwords,
            min_token_len=min_token_len,
        )

    return analyzer
