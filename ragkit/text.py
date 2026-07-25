"""Tokenizing, stopwords, stemming.

Sitting under BM25, so it matters more than it looks. Stemming alone moves
nDCG on SciFact by a couple of points, which is the sort of thing that gets
attributed to "the embedding model" when nobody measures it.
"""

from __future__ import annotations

import re

WORD = re.compile(r"[a-z0-9]+")

# the usual english list. keeping it short on purpose, aggressive stopword
# removal hurts on question-shaped queries where "how" and "why" carry signal.
STOPWORDS = frozenset(
    """a about above after again against all am an and any are as at be because been
    before being below between both but by can cannot could did do does doing down
    during each few for from further had has have having he her here hers herself him
    himself his how i if in into is it its itself me more most my myself no nor not of
    off on once only or other ought our ours ourselves out over own same she should so
    some such than that the their theirs them themselves then there these they this
    those through to too under until up very was we were what when where which while
    who whom why with would you your yours yourself yourselves""".split()
)


def _measure(stem: str) -> int:
    """Porter's m: how many vowel-consonant sequences the stem contains."""
    vc = re.sub(r"[^aeiouy]", "c", re.sub(r"[aeiou]|(?<=[^aeiou])y", "v", stem))
    return vc.count("vc")


def _has_vowel(s: str) -> bool:
    return bool(re.search(r"[aeiou]|(?<=[^aeiou])y", s))


def _double_consonant(s: str) -> bool:
    return len(s) >= 2 and s[-1] == s[-2] and s[-1] not in "aeiou"


def _cvc(s: str) -> bool:
    if len(s) < 3:
        return False
    a, b, c = s[-3], s[-2], s[-1]
    return (
        a not in "aeiou"
        and b in "aeiou"
        and c not in "aeiouwxy"
    )


def stem(word: str) -> str:
    """Porter stemmer, the classic five steps.

    Not the fanciest option but it's deterministic, has no dependencies and
    behaves the same on every machine, which the eval harness cares about more
    than squeezing out the last half point.
    """
    if len(word) <= 2:
        return word
    w = word

    # step 1a, plurals
    if w.endswith("sses"):
        w = w[:-2]
    elif w.endswith("ies"):
        w = w[:-2]
    elif w.endswith("ss"):
        pass
    elif w.endswith("s"):
        w = w[:-1]

    # step 1b, past tense and gerunds
    if w.endswith("eed"):
        if _measure(w[:-3]) > 0:
            w = w[:-1]
    else:
        cut = None
        if w.endswith("ed") and _has_vowel(w[:-2]):
            cut = w[:-2]
        elif w.endswith("ing") and _has_vowel(w[:-3]):
            cut = w[:-3]
        if cut is not None:
            w = cut
            if w.endswith(("at", "bl", "iz")):
                w += "e"
            elif _double_consonant(w) and not w.endswith(("l", "s", "z")):
                w = w[:-1]
            elif _measure(w) == 1 and _cvc(w):
                w += "e"

    # step 1c
    if w.endswith("y") and _has_vowel(w[:-1]):
        w = w[:-1] + "i"

    for suf, rep in _STEP2:
        if w.endswith(suf) and _measure(w[: -len(suf)]) > 0:
            w = w[: -len(suf)] + rep
            break

    for suf, rep in _STEP3:
        if w.endswith(suf) and _measure(w[: -len(suf)]) > 0:
            w = w[: -len(suf)] + rep
            break

    for suf in _STEP4:
        if w.endswith(suf):
            base = w[: -len(suf)]
            if _measure(base) > 1 and (suf not in ("ion",) or base.endswith(("s", "t"))):
                w = base
            break

    # step 5
    if w.endswith("e"):
        m = _measure(w[:-1])
        if m > 1 or (m == 1 and not _cvc(w[:-1])):
            w = w[:-1]
    if _measure(w) > 1 and _double_consonant(w) and w.endswith("l"):
        w = w[:-1]
    return w


_STEP2 = [
    ("ational", "ate"), ("tional", "tion"), ("enci", "ence"), ("anci", "ance"),
    ("izer", "ize"), ("abli", "able"), ("alli", "al"), ("entli", "ent"),
    ("eli", "e"), ("ousli", "ous"), ("ization", "ize"), ("ation", "ate"),
    ("ator", "ate"), ("alism", "al"), ("iveness", "ive"), ("fulness", "ful"),
    ("ousness", "ous"), ("aliti", "al"), ("iviti", "ive"), ("biliti", "ble"),
]

_STEP3 = [
    ("icate", "ic"), ("ative", ""), ("alize", "al"), ("iciti", "ic"),
    ("ical", "ic"), ("ful", ""), ("ness", ""),
]

_STEP4 = [
    "al", "ance", "ence", "er", "ic", "able", "ible", "ant", "ement", "ment",
    "ent", "ion", "ou", "ism", "ate", "iti", "ous", "ive", "ize",
]

_stem_cache: dict[str, str] = {}


def stem_cached(word: str) -> str:
    s = _stem_cache.get(word)
    if s is None:
        s = stem(word)
        _stem_cache[word] = s
    return s


def tokenize(text: str, remove_stopwords: bool = True, stemming: bool = True) -> list[str]:
    toks = WORD.findall(text.lower())
    if remove_stopwords:
        toks = [t for t in toks if t not in STOPWORDS]
    if stemming:
        toks = [stem_cached(t) for t in toks]
    return toks


SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\[])")


def split_sentences(text: str) -> list[str]:
    parts = [p.strip() for p in SENT_SPLIT.split(text.strip())]
    return [p for p in parts if p]
