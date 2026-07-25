"""Retrieval components on a tiny corpus where the right answer is obvious."""

import numpy as np
import pytest

from ragkit.chunking import FixedTokenChunker, SentenceChunker, WholeDocument, chunk_corpus
from ragkit.corpus import Chunk, Document
from ragkit.embed import HashingEmbedder
from ragkit.pipeline import RAGPipeline, RetrievalConfig
from ragkit.retrieval.bm25 import BM25
from ragkit.retrieval.dense import DenseIndex
from ragkit.retrieval.hybrid import reciprocal_rank_fusion, weighted_fusion
from ragkit.rerank import mmr
from ragkit.text import split_sentences, stem, tokenize

DOCS = [
    Document("d1", "The mitochondrion is the powerhouse of the cell and makes ATP.", "Cell biology"),
    Document("d2", "Kubernetes schedules containers across a cluster of worker nodes.", "K8s"),
    Document("d3", "Photosynthesis converts light energy into chemical energy in chloroplasts.", "Plants"),
    Document("d4", "A pod is the smallest deployable unit in Kubernetes.", "K8s pods"),
]
CHUNKS = [Chunk(d.doc_id, d.doc_id, d.text, d.title) for d in DOCS]


def test_bm25_finds_the_obvious_document():
    bm = BM25().index(CHUNKS)
    assert bm.search("what makes ATP in the cell", 1)[0][0] == "d1"
    assert bm.search("kubernetes pod deployable unit", 1)[0][0] == "d4"


def test_bm25_scores_are_positive_and_ranked():
    bm = BM25().index(CHUNKS)
    hits = bm.search("kubernetes containers cluster", 4)
    scores = [s for _, s in hits]
    assert all(s > 0 for s in scores)
    assert scores == sorted(scores, reverse=True)


def test_bm25_returns_nothing_for_out_of_vocabulary_queries():
    bm = BM25().index(CHUNKS)
    assert bm.search("zzzqqq nonexistentterm", 5) == []


def test_bm25_idf_prefers_the_rare_term():
    """'kubernetes' appears twice, 'photosynthesis' once, so the rare one
    should dominate a query containing both."""
    bm = BM25().index(CHUNKS)
    assert bm.search("kubernetes photosynthesis", 1)[0][0] == "d3"


def test_stemming_collapses_inflections():
    assert stem("running") == stem("run")
    assert stem("converts") == stem("convert")
    assert stem("schedules") == stem("schedule")


def test_tokenize_drops_stopwords():
    assert "the" not in tokenize("the cell membrane")
    assert tokenize("the cell", stemming=False, remove_stopwords=False) == ["the", "cell"]


def test_dense_index_ranks_by_cosine():
    idx = DenseIndex(HashingEmbedder()).index(CHUNKS)
    hits = idx.search("kubernetes cluster nodes", 4)
    assert hits[0][0] in ("d2", "d4")
    assert all(-1.001 <= s <= 1.001 for _, s in hits)


def test_dense_batch_search_matches_single_search():
    idx = DenseIndex(HashingEmbedder()).index(CHUNKS)
    qs = ["kubernetes pods", "cell ATP energy"]
    batch = idx.search_batch(qs, 3)
    for q, got in zip(qs, batch):
        one = idx.search(q, 3)
        assert [c for c, _ in got] == [c for c, _ in one]
        assert np.allclose([s for _, s in got], [s for _, s in one], atol=1e-5)


def test_hashing_embedder_is_stable_across_calls():
    e = HashingEmbedder()
    a = e.encode(["reproducibility matters"])
    b = e.encode(["reproducibility matters"])
    assert np.array_equal(a, b)


def test_rrf_rewards_agreement_between_rankers():
    a = ["x", "y", "z"]
    b = ["y", "x", "z"]
    fused = reciprocal_rank_fusion([a, b])
    # y is 2nd and 1st, x is 1st and 2nd, so they tie and both beat z
    assert {fused[0][0], fused[1][0]} == {"x", "y"}
    assert fused[-1][0] == "z"


def test_rrf_promotes_a_doc_both_rankers_rate_highly():
    fused = reciprocal_rank_fusion([["a", "b", "c"], ["a", "c", "b"]])
    assert fused[0][0] == "a"


def test_rrf_weights_shift_the_result():
    a = ["x", "y"]
    b = ["y", "x"]
    assert reciprocal_rank_fusion([a, b], weights=[10.0, 1.0])[0][0] == "x"
    assert reciprocal_rank_fusion([a, b], weights=[1.0, 10.0])[0][0] == "y"


def test_weighted_fusion_normalizes_incompatible_scales():
    """BM25 scores run to 20+, cosine sits in [-1, 1]. Without per query
    normalization the lexical side would always win by sheer magnitude."""
    ids = ["a", "b", "c"]
    lex = np.array([20.0, 0.0, 10.0])
    den = np.array([0.0, 1.0, 0.5])
    out = weighted_fusion([lex, den], ids, [0.0, 1.0], 3)
    assert out[0][0] == "b"  # dense-only weighting must follow the dense scores
    out = weighted_fusion([lex, den], ids, [1.0, 0.0], 3)
    assert out[0][0] == "a"


def test_mmr_trades_relevance_for_diversity():
    q = np.array([1.0, 0.0], dtype=np.float32)
    # two near-duplicates plus one different direction
    vecs = np.array([[1.0, 0.0], [0.99, 0.14], [0.0, 1.0]], dtype=np.float32)
    ids = ["dup1", "dup2", "other"]

    pure = [c for c, _ in mmr(q, vecs, ids, k=2, lam=1.0)]
    assert pure == ["dup1", "dup2"]

    diverse = [c for c, _ in mmr(q, vecs, ids, k=2, lam=0.4)]
    assert diverse == ["dup1", "other"]


def test_chunkers_cover_the_whole_document():
    doc = Document("d", " ".join(f"w{i}" for i in range(500)))
    for chunker in (WholeDocument(), FixedTokenChunker(100, 20), SentenceChunker(100, 1)):
        chunks = chunker.split(doc)
        joined = " ".join(c.text for c in chunks)
        assert "w0" in joined and "w499" in joined
        assert all(c.doc_id == "d" for c in chunks)


def test_fixed_chunker_overlaps():
    doc = Document("d", " ".join(f"w{i}" for i in range(30)))
    chunks = FixedTokenChunker(10, 5).split(doc)
    assert len(chunks) > 1
    first = set(chunks[0].text.split())
    second = set(chunks[1].text.split())
    assert first & second, "overlap should share words between neighbours"


def test_fixed_chunker_rejects_impossible_overlap():
    with pytest.raises(ValueError):
        FixedTokenChunker(50, 50)


def test_sentence_chunker_never_splits_mid_sentence():
    text = "First sentence here. Second sentence follows. Third one ends it."
    chunks = SentenceChunker(max_words=5, overlap_sentences=0).split(Document("d", text))
    for c in chunks:
        assert c.text.strip().endswith((".", "!", "?"))


def test_split_sentences_handles_abbreviation_free_text():
    assert len(split_sentences("One. Two! Three?")) == 3


def test_chunk_ids_are_unique():
    chunks = chunk_corpus(DOCS * 1, FixedTokenChunker(5, 1))
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))


def test_pipeline_maps_chunks_back_to_documents():
    cfg = RetrievalConfig(retriever="hybrid", embedder="hashing", chunker="fixed-5-1", top_k=3)
    pipe = RAGPipeline(cfg, embedder=HashingEmbedder()).build(DOCS)
    res = pipe.retrieve("kubernetes worker nodes cluster")
    assert res.doc_ids and all(d in {"d1", "d2", "d3", "d4"} for d in res.doc_ids)
    assert len(res.doc_ids) == len(set(res.doc_ids)), "doc ids must be deduped"


def test_pipeline_respects_top_k():
    cfg = RetrievalConfig(retriever="bm25", top_k=2)
    pipe = RAGPipeline(cfg).build(DOCS)
    assert len(pipe.retrieve("kubernetes cell energy").chunk_ids) <= 2


def test_config_fingerprint_changes_with_any_knob():
    a = RetrievalConfig()
    assert a.fingerprint() == RetrievalConfig().fingerprint()
    assert a.fingerprint() != RetrievalConfig(top_k=11).fingerprint()
    assert a.fingerprint() != RetrievalConfig(bm25_b=0.5).fingerprint()
    assert a.fingerprint() != RetrievalConfig(chunker="fixed-100-0").fingerprint()
