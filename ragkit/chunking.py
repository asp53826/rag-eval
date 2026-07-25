"""Splitting documents into retrievable units.

The most under-measured knob in RAG. People agonize over which embedding model
to use and then split on 512 characters with no overlap, which cuts sentences
in half and strands the answer across a boundary where no retriever can reach
it. The harness treats chunk size and overlap as first class variables.
"""

from __future__ import annotations

from typing import Protocol

from .corpus import Chunk, Document
from .text import split_sentences


class Chunker(Protocol):
    name: str

    def split(self, doc: Document) -> list[Chunk]: ...


class WholeDocument:
    """No splitting. The right answer more often than people expect, when
    documents are already short (abstracts, paragraphs, support tickets)."""

    name = "whole"

    def split(self, doc: Document) -> list[Chunk]:
        return [Chunk(doc.doc_id, doc.doc_id, doc.text, doc.title, 0)]


class FixedTokenChunker:
    """Fixed word windows with overlap.

    Overlap is the cheap insurance against the answer landing exactly on a
    boundary. It costs index size and it duplicates content across chunks,
    which is a real tradeoff the eval can quantify.
    """

    def __init__(self, size: int = 200, overlap: int = 50):
        if overlap >= size:
            raise ValueError("overlap must be smaller than size")
        self.size = size
        self.overlap = overlap
        self.name = f"fixed-{size}-{overlap}"

    def split(self, doc: Document) -> list[Chunk]:
        words = doc.text.split()
        if len(words) <= self.size:
            return [Chunk(doc.doc_id, doc.doc_id, doc.text, doc.title, 0)]
        step = self.size - self.overlap
        out = []
        for pos, start in enumerate(range(0, len(words), step)):
            piece = words[start : start + self.size]
            if not piece:
                break
            out.append(
                Chunk(f"{doc.doc_id}#{pos}", doc.doc_id, " ".join(piece), doc.title, pos)
            )
            if start + self.size >= len(words):
                break
        return out


class SentenceChunker:
    """Pack whole sentences up to a word budget, never mid-sentence.

    Overlap is measured in sentences so the carried-over context is always a
    complete thought.
    """

    def __init__(self, max_words: int = 200, overlap_sentences: int = 1):
        self.max_words = max_words
        self.overlap_sentences = overlap_sentences
        self.name = f"sentence-{max_words}-{overlap_sentences}"

    def split(self, doc: Document) -> list[Chunk]:
        sents = split_sentences(doc.text)
        if not sents:
            return [Chunk(doc.doc_id, doc.doc_id, doc.text, doc.title, 0)]

        out: list[Chunk] = []
        cur: list[str] = []
        count = 0
        pos = 0

        def flush():
            nonlocal cur, count, pos
            if cur:
                out.append(
                    Chunk(f"{doc.doc_id}#{pos}", doc.doc_id, " ".join(cur), doc.title, pos)
                )
                pos += 1

        for s in sents:
            n = len(s.split())
            if cur and count + n > self.max_words:
                flush()
                cur = cur[-self.overlap_sentences :] if self.overlap_sentences else []
                count = sum(len(x.split()) for x in cur)
            cur.append(s)
            count += n
        flush()
        return out or [Chunk(doc.doc_id, doc.doc_id, doc.text, doc.title, 0)]


def chunk_corpus(docs: list[Document], chunker: Chunker) -> list[Chunk]:
    out: list[Chunk] = []
    for d in docs:
        out.extend(chunker.split(d))
    return out


def get_chunker(spec: str) -> Chunker:
    if spec == "whole":
        return WholeDocument()
    kind, *rest = spec.split("-")
    if kind == "fixed":
        return FixedTokenChunker(int(rest[0]), int(rest[1]))
    if kind == "sentence":
        return SentenceChunker(int(rest[0]), int(rest[1]))
    raise ValueError(f"unknown chunker {spec}")
