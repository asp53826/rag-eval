"""Dataset loading.

Two real benchmarks rather than a synthetic corpus, because a corpus you
generated yourself will always flatter your retriever.

  SciFact (BEIR)  5183 abstracts, 300 test claims with human relevance
                  judgments. Published numbers exist, so the retrieval
                  implementation can be checked against the literature instead
                  of against itself.
  SQuAD v2        reference answers plus, crucially, unanswerable questions.
                  Those are what make a hallucination metric mean anything.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")


@dataclass(slots=True)
class Document:
    doc_id: str
    text: str
    title: str = ""

    @property
    def full_text(self) -> str:
        return f"{self.title}\n{self.text}" if self.title else self.text


@dataclass(slots=True)
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    title: str = ""
    position: int = 0

    @property
    def full_text(self) -> str:
        return f"{self.title}\n{self.text}" if self.title else self.text


@dataclass
class Query:
    query_id: str
    text: str
    # doc_id -> graded relevance, 0 means judged irrelevant
    relevant: dict[str, int] = field(default_factory=dict)
    # free text reference answers, empty list means the question is
    # unanswerable from the corpus and the right move is to abstain
    answers: list[str] = field(default_factory=list)
    answerable: bool = True
    gold_doc_id: str | None = None

    @property
    def relevant_ids(self) -> set[str]:
        return {d for d, s in self.relevant.items() if s > 0}


def _read_jsonl(path: str):
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_beir(name: str = "scifact", split: str = "test", root: str | None = None):
    """Standard BEIR layout: corpus.jsonl, queries.jsonl, qrels/<split>.tsv."""
    base = os.path.join(root or RAW_DIR, name)
    if not os.path.isdir(base):
        raise FileNotFoundError(f"{base} missing, run scripts/fetch_data.py")

    docs = [
        Document(d["_id"], d.get("text", ""), d.get("title", ""))
        for d in _read_jsonl(os.path.join(base, "corpus.jsonl"))
    ]

    qrels: dict[str, dict[str, int]] = {}
    with open(os.path.join(base, "qrels", f"{split}.tsv"), encoding="utf-8") as f:
        header = next(f)
        if "query" not in header.lower():  # some dumps have no header row
            f.seek(0)
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            qid, did, score = parts[0], parts[1], int(parts[2])
            qrels.setdefault(qid, {})[did] = score

    queries = [
        Query(q["_id"], q["text"], relevant=qrels[q["_id"]])
        for q in _read_jsonl(os.path.join(base, "queries.jsonl"))
        if q["_id"] in qrels
    ]
    return docs, queries


def load_squad(path: str | None = None, max_questions: int | None = None,
               unanswerable_ratio: float = 0.4, seed: int = 0):
    """Turn SQuAD v2 into a retrieval corpus plus a QA set.

    Each paragraph becomes a document, so retrieval has to find the right
    paragraph out of thousands before the answer step gets a chance. Keeping a
    controlled share of unanswerable questions, since those are the only way to
    measure whether the system knows when to shut up.
    """
    import random

    path = path or os.path.join(RAW_DIR, "squad-dev-v2.json")
    with open(path, encoding="utf-8") as f:
        blob = json.load(f)

    docs: list[Document] = []
    answerable: list[Query] = []
    unanswerable: list[Query] = []

    for article in blob["data"]:
        title = article["title"].replace("_", " ")
        for pi, para in enumerate(article["paragraphs"]):
            doc_id = f"{article['title']}::{pi}"
            docs.append(Document(doc_id, para["context"], title))
            for qa in para["qas"]:
                golds = sorted({a["text"] for a in qa.get("answers", [])})
                q = Query(
                    query_id=qa["id"],
                    text=qa["question"],
                    relevant={doc_id: 1},
                    answers=golds,
                    answerable=not qa.get("is_impossible", False) and bool(golds),
                    gold_doc_id=doc_id,
                )
                (answerable if q.answerable else unanswerable).append(q)

    rng = random.Random(seed)
    rng.shuffle(answerable)
    rng.shuffle(unanswerable)

    if max_questions:
        n_un = int(max_questions * unanswerable_ratio)
        queries = answerable[: max_questions - n_un] + unanswerable[:n_un]
        rng.shuffle(queries)
    else:
        queries = answerable + unanswerable
    return docs, queries


def dataset_stats(docs: list[Document], queries: list[Query]) -> dict:
    lens = [len(d.text.split()) for d in docs]
    return {
        "documents": len(docs),
        "queries": len(queries),
        "answerable": sum(1 for q in queries if q.answerable),
        "unanswerable": sum(1 for q in queries if not q.answerable),
        "mean_doc_words": round(sum(lens) / len(lens), 1) if lens else 0,
        "max_doc_words": max(lens) if lens else 0,
        "mean_relevant_per_query": round(
            sum(len(q.relevant_ids) for q in queries) / len(queries), 2
        ) if queries else 0,
    }
