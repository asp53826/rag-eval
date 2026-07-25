"""Query rewriting.

Queries and documents are written by different people for different reasons.
A user asks "why did my deploy fail", the document says "deployment
terminated: exit code 137". Rewriting tries to close that gap before
retrieval, rather than asking the embedding model to do all of it.

Every rewriter here returns a list of queries whose result lists get fused, so
a bad expansion degrades the ranking instead of replacing it.
"""

from __future__ import annotations

from typing import Protocol

from .text import STOPWORDS, WORD


class Rewriter(Protocol):
    name: str

    def rewrite(self, query: str) -> list[str]: ...


class NoRewrite:
    name = "none"

    def rewrite(self, query: str) -> list[str]:
        return [query]


class StripQuestionWords:
    """Drop the interrogative scaffolding, keep the content terms.

    "What is the mechanism by which X causes Y" carries a lot of tokens that
    appear in every document and discriminate between none of them.
    """

    name = "strip-question"
    LEAD = frozenset(
        "what when where who whom whose which why how is are was were do does did "
        "can could should would will the a an of in on for to".split()
    )

    def rewrite(self, query: str) -> list[str]:
        toks = WORD.findall(query.lower())
        kept = [t for t in toks if t not in self.LEAD]
        stripped = " ".join(kept)
        return [query, stripped] if stripped and stripped != query.lower() else [query]


class KeywordExpansion:
    """Add a keywords-only variant alongside the original.

    Helps the lexical side, which gets diluted by function words, without
    taking anything away from the dense side that reads the full sentence.
    """

    name = "keyword-expansion"

    def rewrite(self, query: str) -> list[str]:
        toks = [t for t in WORD.findall(query.lower()) if t not in STOPWORDS and len(t) > 2]
        if not toks:
            return [query]
        return [query, " ".join(toks)]


class HyDE:
    """Hypothetical document embeddings.

    Ask a model to write the passage that *would* answer the question, then
    retrieve with that instead. It works because a fake answer looks far more
    like a real document than a question does. Needs a generator, so it stays
    off by default and out of the hermetic CI path.
    """

    name = "hyde"

    def __init__(self, generator, keep_query: bool = True):
        self.generator = generator
        self.keep_query = keep_query

    def rewrite(self, query: str) -> list[str]:
        doc = self.generator.complete(
            f"Write one short factual paragraph that would answer this question. "
            f"Do not hedge, do not say you are unsure.\n\nQuestion: {query}\n\nPassage:"
        )
        return [query, doc] if self.keep_query else [doc]


def get_rewriter(spec: str) -> Rewriter:
    return {
        "none": NoRewrite,
        "strip-question": StripQuestionWords,
        "keyword-expansion": KeywordExpansion,
    }[spec]()
