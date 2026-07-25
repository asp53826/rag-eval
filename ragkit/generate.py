"""Turning retrieved chunks into an answer, and deciding when not to.

The abstention threshold is the interesting part. A RAG system that always
answers scores well on any benchmark made only of answerable questions and
hallucinates confidently the moment a user asks something the corpus doesn't
cover. That failure is invisible unless the test set contains questions with
no answer and the metrics report them separately, which is what SQuAD v2's
unanswerable half is for here.

ExtractiveGenerator is the default because the harness has to run in CI with
no API key and produce the same answer every time. LLMGenerator is the same
interface for when there is a key.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Protocol

from .corpus import Chunk
from .text import split_sentences, tokenize


@dataclass
class AnswerOutput:
    text: str
    abstained: bool = False
    confidence: float = 0.0
    used_chunk_ids: list[str] = field(default_factory=list)


class Generator(Protocol):
    name: str

    def answer(self, query: str, chunks: list[Chunk]) -> AnswerOutput: ...


ABSTAIN_TEXT = "I don't have enough information in the provided context to answer that."


class ExtractiveGenerator:
    """Pick the sentence in the retrieved context that best covers the query.

    Scores by IDF-weighted term coverage, so matching a rare term counts for
    much more than matching "the". Deterministic, no model, no key. It will
    never paraphrase, which caps its exact match score, and that limitation is
    the honest cost of a hermetic baseline.
    """

    name = "extractive"

    def __init__(self, threshold: float = 0.35, max_sentences: int = 1):
        # tuned on a held out slice, not on the reported test set
        self.threshold = threshold
        self.max_sentences = max_sentences

    def answer(self, query: str, chunks: list[Chunk]) -> AnswerOutput:
        if not chunks:
            return AnswerOutput(ABSTAIN_TEXT, abstained=True, confidence=0.0)

        q_terms = set(tokenize(query))
        if not q_terms:
            return AnswerOutput(ABSTAIN_TEXT, abstained=True, confidence=0.0)

        sentences: list[tuple[str, str]] = []
        for c in chunks:
            for s in split_sentences(c.text):
                sentences.append((s, c.chunk_id))
        if not sentences:
            return AnswerOutput(ABSTAIN_TEXT, abstained=True, confidence=0.0)

        # document frequency over the retrieved set only, which is what makes
        # a term "rare" in the context that matters here
        df: dict[str, int] = {}
        toks = []
        for s, _ in sentences:
            t = set(tokenize(s))
            toks.append(t)
            for term in t & q_terms:
                df[term] = df.get(term, 0) + 1

        n = len(sentences)
        total_weight = sum(math.log(1 + n / df.get(t, n)) for t in q_terms) or 1.0

        best_i, best_score = 0, 0.0
        for i, (t, _) in enumerate(zip(toks, sentences)):
            hit = t & q_terms
            score = sum(math.log(1 + n / df[term]) for term in hit) / total_weight
            if score > best_score:
                best_i, best_score = i, score

        if best_score < self.threshold:
            return AnswerOutput(ABSTAIN_TEXT, abstained=True, confidence=best_score)

        text, chunk_id = sentences[best_i]
        return AnswerOutput(text, abstained=False, confidence=best_score,
                            used_chunk_ids=[chunk_id])


class AlwaysAnswerGenerator(ExtractiveGenerator):
    """Same extraction, abstention disabled.

    Exists to put a number on what refusing to abstain costs. It is the
    baseline every "our RAG scores 0.8 F1" claim is implicitly using.
    """

    name = "always-answer"

    def __init__(self):
        super().__init__(threshold=-1.0)


PROMPT = """Answer the question using only the context below.
If the context does not contain the answer, reply with exactly: {abstain}
Answer in as few words as possible.

Context:
{context}

Question: {question}
Answer:"""


class LLMGenerator:
    """OpenAI-compatible chat completions, which also covers local servers.

    Points at vllm-lite or any other OpenAI-shaped endpoint by setting
    base_url, so the eval harness can grade a real model when one is around.
    """

    def __init__(self, model: str = "gpt-4o-mini", base_url: str | None = None,
                 api_key: str | None = None, temperature: float = 0.0,
                 max_chunks: int = 5, timeout: float = 60.0):
        self.name = f"llm:{model}"
        self.model = model
        self.base_url = (base_url or os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")).rstrip("/")
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.temperature = temperature
        self.max_chunks = max_chunks
        self.timeout = timeout

    def complete(self, prompt: str) -> str:
        import json
        import urllib.request

        body = json.dumps({
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
        }).encode()
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=body,
            headers={"content-type": "application/json",
                     "authorization": f"Bearer {self.api_key}"},
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            data = json.load(r)
        return data["choices"][0]["message"]["content"].strip()

    def answer(self, query: str, chunks: list[Chunk]) -> AnswerOutput:
        used = chunks[: self.max_chunks]
        context = "\n\n".join(f"[{i + 1}] {c.full_text}" for i, c in enumerate(used))
        text = self.complete(
            PROMPT.format(abstain=ABSTAIN_TEXT, context=context, question=query)
        )
        abstained = ABSTAIN_TEXT.lower()[:30] in text.lower()
        return AnswerOutput(text, abstained=abstained, confidence=0.0,
                            used_chunk_ids=[c.chunk_id for c in used])


def get_generator(spec: str, **kw) -> Generator:
    if spec == "extractive":
        return ExtractiveGenerator(**kw)
    if spec == "always-answer":
        return AlwaysAnswerGenerator()
    if spec.startswith("llm:"):
        return LLMGenerator(model=spec[4:], **kw)
    raise ValueError(f"unknown generator {spec}")
