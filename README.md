# rag-eval

A RAG pipeline with hybrid retrieval, reranking and query rewriting — and, more
to the point, a reproducible evaluation harness that says whether any of it
actually helps.

Building a RAG pipeline is a weekend. Knowing whether your change made it
better is the hard part, and it's the part almost nobody does. This repo treats
retrieval quality as something you gate in CI, the same way you'd gate a unit
test: freeze a baseline, re-measure on every change, fail the build when a
number drops.

```bash
pip install -r requirements-full.txt
python scripts/fetch_data.py

python -m eval.cli sweep --dataset scifact --preset retrievers
python -m eval.cli calibrate --dataset squad2
python -m eval.cli check --dataset scifact --retriever bm25   # exits 1 on regression
pytest                                                        # 61 tests
```

## The implementation is validated against published numbers

Two real benchmarks, not a corpus I generated myself. A synthetic corpus will
always flatter your own retriever.

| | mine | published |
|---|---|---|
| BM25 on SciFact, nDCG@10 | **0.6643** | 0.665 (BEIR) |
| all-MiniLM-L6-v2 on SciFact, nDCG@10 | **0.6451** | 0.645 (BEIR) |
| BM25 + ms-marco cross-encoder, nDCG@10 | **0.6853** | ~0.688 (BEIR) |

Every component lands within a few thousandths of the literature, so the
comparisons below are measuring the ideas rather than my bugs. Adding a Porter
stemmer takes BM25 from 0.6643 to **0.6728**, which is the kind of thing that
gets credited to the embedding model when nobody ablates it.

## What the eval found

**SciFact** (5,183 abstracts, 300 claims with human relevance judgments):

| config | nDCG@10 | recall@10 | MRR@10 | hit@1 | p50 latency |
|---|---|---|---|---|---|
| BM25 | 0.6728 | 0.7888 | 0.6438 | 0.5600 | 0.17 ms |
| dense, hashing embedder | 0.3871 | 0.4844 | 0.3609 | 0.2933 | 0.25 ms |
| dense, MiniLM | 0.6451 | 0.7833 | 0.6047 | 0.5033 | 9.2 ms |
| hybrid, RRF | 0.7055 | 0.8406 | 0.6697 | 0.5800 | 0.48 ms |
| **hybrid, weighted** | **0.7267** | **0.8449** | **0.6964** | **0.6067** | 0.33 ms |

Hybrid beats the better of its two halves by **+8.0% nDCG@10** and +7.1%
recall@10. Lexical and dense retrieval fail on different queries — BM25 nails
exact terms, dosages, identifiers; dense handles the case where the user and
the document don't share vocabulary — so fusing them recovers documents neither
finds alone.

### Three findings that contradict the usual advice

**1. The cross-encoder reranker made things worse.**

| config | nDCG@10 | p50 latency |
|---|---|---|
| hybrid, weighted | **0.7267** | **0.33 ms** |
| hybrid + cross-encoder rerank | 0.6930 | 264 ms |
| BM25 | 0.6728 | 0.17 ms |
| BM25 + cross-encoder rerank | 0.6853 | 267 ms |

The reranker does improve its own first stage (+1.9% over BM25, matching the
published 0.688). It just loses to hybrid fusion, and stacking it *on top of*
hybrid destroys 3.4 points. The cause is domain mismatch: `ms-marco-MiniLM` is
trained on web search queries and SciFact queries are scientific claims. So the
standard "add a reranker" advice costs **800x the latency for a quality
regression** on this corpus. Verified at candidate depths of 20, 50 and 100 —
it isn't a depth artifact.

**2. Chunking hurt, every configuration of it.**

| chunker | nDCG@10 | chunks | per doc |
|---|---|---|---|
| **whole document** | **0.7267** | 5,183 | 1.0 |
| fixed 200 words, 50 overlap | 0.7146 | 7,758 | 1.5 |
| fixed 100 words, no overlap | 0.7166 | 13,016 | 2.5 |
| sentence, 150 words | 0.7056 | 10,063 | 1.9 |
| sentence, 100 words | 0.7018 | 16,560 | 3.2 |

SciFact documents are abstracts averaging 202 words — already the right
retrieval unit. Splitting them fragments the evidence across chunks and costs
1 to 2.5 points while tripling index size. Chunking is a fix for long
documents, not a default.

**3. Query rewriting did nothing measurable** (0.7267 → 0.7270, well inside
noise). SciFact queries are declarative claims, so there's no interrogative
scaffolding to strip. The rewriter earns its keep on SQuAD-style questions; on
this corpus it's dead weight and the harness says so.

### Retrieval on SQuAD v2

| config | nDCG@10 | recall@5 | hit@1 |
|---|---|---|---|
| BM25 | 0.8809 | 0.9250 | 0.7950 |
| dense, MiniLM | 0.7691 | 0.8725 | 0.6050 |
| **hybrid, weighted** | **0.9056** | **0.9525** | **0.8250** |

Note dense loses to BM25 here and wins nowhere near as much on SciFact. Which
retriever is "better" is a property of the corpus, not a fact about retrievers.

### Tuning the fusion weight

| BM25 / dense weight | nDCG@10 |
|---|---|
| 0.0 / 1.0 (pure dense) | 0.6451 |
| 0.25 / 0.75 | 0.7254 |
| **0.4 / 0.6** | **0.7306** |
| 0.5 / 0.5 | 0.7267 |
| 0.75 / 0.25 | 0.6980 |
| 1.0 / 0.0 (pure BM25) | 0.6728 |

A clean optimum slightly favouring dense. Worth 0.4 points over the 50/50
default, and worth 8.6 points over guessing wrong in either direction.

## Hallucination: the measurement that mattered most

A RAG system that always answers scores well on any benchmark made only of
answerable questions, then confidently invents things the moment a user asks
something the corpus doesn't cover. That failure is *invisible* unless the test
set contains unanswerable questions and the metrics report them separately.

SQuAD v2 provides them, and they're adversarial — written to look answerable
against the same paragraph. So this is a fair test, and the extractive
baseline fails it:

```
abstention signal on dev (160 questions):
  AUC 0.6352, separation 0.098   -> "usable" by the >0.6 rule
  chosen threshold (max accuracy): 0.5097

held out test (240 questions) @ threshold 0.5097:
  hallucination_rate   0.8021
  over_abstention_rate 0.1597
  answer_accuracy      0.5833
  signal_auc_test      0.5449   <- did not generalize
```

**Dev AUC 0.635, test AUC 0.545.** The signal looked usable on the split it was
chosen on and was near-random on held-out data. The calibrated threshold ends
up at 58.3% accuracy, *worse than the 60% you get by always answering*.

The conclusion is that lexical overlap cannot be an abstention signal here, and
no amount of threshold tuning will fix it — knowing a paragraph doesn't contain
the answer requires reading it, not counting shared terms. Retrieval finds the
right paragraph 82.5% of the time; the failure is entirely downstream.

Two things made this visible, and both are methodology rather than cleverness:
choosing the threshold on a dev split and reporting on a held-out one, and
checking the signal's AUC before bothering to tune a threshold on it. A single
split would have shipped AUC 0.635 as a result.

## The harness

```
ragkit/                       the pipeline
  text.py                     tokenizer, stopwords, Porter stemmer
  chunking.py                 whole / fixed / sentence, with overlap
  embed.py                    MiniLM or dependency-free hashing, disk cached
  retrieval/bm25.py           BM25 Okapi, k1 and b exposed for ablation
  retrieval/dense.py          exact cosine, batched
  retrieval/hybrid.py         RRF and weighted fusion
  rerank.py                   cross-encoder, lexical, MMR diversity
  rewrite.py                  question stripping, keyword expansion, HyDE
  generate.py                 extractive answerer with abstention, LLM interface
eval/                         the point of the repo
  metrics.py                  nDCG, recall, MRR, EM, token F1, faithfulness
  calibrate.py                dev/test split, ROC AUC, threshold sweep
  harness.py                  run a config, produce a hashed report
  regression.py               diff two runs, fail on regression
  cli.py
```

Rules the harness enforces:

- **Every number is tied to the config that produced it**, by hash. Change any
  knob — chunk size, `bm25_b`, top-k — and the fingerprint changes.
- **Reproducible by construction.** Same config plus same data gives byte
  identical metrics. Tested.
- **Timing is not a metric.** Latency lives in a separate `timing` block
  precisely because it's non-deterministic and would make every reproducibility
  check flaky.
- **Per-query records are kept**, so "recall dropped" can be traced to the
  specific queries that broke rather than re-run blind.
- **Floors are per-config.** BM25 scoring below the hybrid floor is correct
  behaviour, not a regression.

### Regression gates in CI

```bash
python -m eval.cli baseline --dataset scifact --retriever hybrid --fusion weighted
python -m eval.cli check    --dataset scifact --retriever hybrid --fusion weighted
```

```
baseline vs current: hybrid + minilm + weighted + w=0.5/0.5
  metric                  baseline   current     delta  status
  ------------------------------------------------------------
  hit@1                     0.6067    0.6067   +0.0000  ok
  ndcg@10                   0.7800    0.7267   -0.0533  REGRESSED
  recall@10                 0.9000    0.8449   -0.0551  REGRESSED
  newly failing (11): 108, 1149, 122, 13, 1330
  => FAIL
```

`check` exits 1 on regression, so CI fails on a retrieval quality drop the same
way it fails on a broken test. There are two independent guards, because the
easiest way to make a failing regression check go green is to re-freeze the
baseline:

- **relative** — per-metric tolerance against the frozen baseline
- **absolute floors** — a config may never drop below a stated number, whatever
  the baseline says

CI runs the BM25 gate on every push with no model downloads, so quality is
checked on a clean machine in seconds.

## Being straight about what this is

- **The default generator is extractive, not an LLM.** It picks the
  best-covering sentence from the retrieved context. That makes the harness
  hermetic, deterministic and free, and it caps exact match near zero because
  it returns whole sentences where SQuAD wants short spans. Token F1 is the
  meaningful answer number for it. `LLMGenerator` is the same interface against
  any OpenAI-shaped endpoint, including a local one.
- **Faithfulness is an n-gram grounding proxy, not entailment.** It catches the
  failure that matters most in practice — a model asserting specifics the
  retrieved passages never mentioned — without an API call. It will miss a
  claim that is unsupported but paraphrased in the corpus's own vocabulary.
- **The hashing embedder is genuinely bad** (0.3871 vs 0.6451). It exists so
  the pipeline stays testable in CI with no torch, and the eval reports the gap
  rather than hiding it.
- **Every finding above is one corpus.** "Reranking hurts" is true of SciFact
  with an ms-marco cross-encoder. The point isn't the conclusion, it's that the
  harness produced one instead of a vibe.
- **Not implemented:** multi-hop retrieval, learned sparse retrieval (SPLADE),
  approximate nearest neighbour indexing, LLM-as-judge faithfulness.

## Tests

61 tests, `pytest`. Metrics are checked against hand-computed values rather
than against themselves — nDCG against the explicit `1/log2(3)` sum, token F1
against the precision/recall arithmetic, ROC AUC against known-perfect,
known-inverted and known-random inputs.

The harness tests are the ones that matter:

- the same config twice produces identical metrics, for lexical and dense alike
- a report survives a JSON roundtrip and still hashes to its original config
- a real 5-point drop is flagged; 0.001 of noise is not
- improvements are never flagged as regressions
- `hallucination_rate` rising is a regression, not an improvement
- always-answering scores hallucination 1.0, always-abstaining scores 0.0
- absolute floors catch a baseline that was re-frozen to hide a drop
- the exact query ids that changed verdict are named
