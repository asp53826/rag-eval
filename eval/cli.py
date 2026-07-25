#!/usr/bin/env python3
"""Command line entry point.

  python -m eval.cli run --dataset scifact --retriever hybrid
  python -m eval.cli sweep --dataset scifact --preset retrievers
  python -m eval.cli calibrate --dataset squad2
  python -m eval.cli baseline --dataset scifact          # freeze current numbers
  python -m eval.cli check --dataset scifact             # fail if they got worse
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ragkit.corpus import dataset_stats, load_beir, load_squad  # noqa: E402
from ragkit.pipeline import RetrievalConfig  # noqa: E402

from .calibrate import (  # noqa: E402
    abstention_curve,
    pick_threshold,
    split_dev_test,
    summarize_signal,
)
from .harness import evaluate_retrieval, format_table, sweep  # noqa: E402
from .regression import (  # noqa: E402
    check_floors,
    compare,
    format_comparison,
    load_baseline,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASELINE_DIR = os.path.join(ROOT, "baselines")
RESULTS_DIR = os.path.join(ROOT, "results")


def load_dataset(name: str, limit: int | None = None):
    if name == "scifact":
        return load_beir("scifact", "test")
    if name == "squad2":
        return load_squad(max_questions=limit or 400)
    raise ValueError(f"unknown dataset {name}")


def build_shared(specs: set[str], rerankers: set[str], offline: bool):
    """Load each model once and share it across every config in a sweep."""
    from ragkit.embed import get_embedder
    from ragkit.rerank import LexicalOverlapReranker, NoopReranker

    emb = {}
    for s in specs:
        if s != "hashing" and offline:
            continue
        emb[s] = get_embedder(s)

    rr = {"none": NoopReranker(), "lexical": LexicalOverlapReranker()}
    if "cross-encoder" in rerankers and not offline:
        from ragkit.rerank import CrossEncoderReranker

        rr["cross-encoder"] = CrossEncoderReranker()
    return emb, rr


PRESETS = {
    "retrievers": lambda e: [
        RetrievalConfig(retriever="bm25"),
        RetrievalConfig(retriever="dense", embedder="hashing"),
        RetrievalConfig(retriever="dense", embedder=e),
        RetrievalConfig(retriever="hybrid", embedder=e, fusion="rrf"),
        RetrievalConfig(retriever="hybrid", embedder=e, fusion="weighted"),
    ],
    "chunking": lambda e: [
        RetrievalConfig(retriever="hybrid", embedder=e, fusion="weighted", chunker=c)
        for c in ("whole", "fixed-100-0", "fixed-100-25", "fixed-200-50",
                  "sentence-100-1", "sentence-150-1")
    ],
    "rerankers": lambda e: [
        RetrievalConfig(retriever="hybrid", embedder=e, fusion="weighted", reranker=r)
        for r in ("none", "lexical", "cross-encoder")
    ],
    "rewriters": lambda e: [
        RetrievalConfig(retriever="hybrid", embedder=e, fusion="weighted", rewriter=r)
        for r in ("none", "strip-question", "keyword-expansion")
    ],
    "fusion-weights": lambda e: [
        RetrievalConfig(retriever="hybrid", embedder=e, fusion="weighted",
                        weights=(w, round(1 - w, 2)))
        for w in (0.0, 0.25, 0.4, 0.5, 0.6, 0.75, 1.0)
    ],
}


def cfg_from_args(args) -> RetrievalConfig:
    return RetrievalConfig(
        chunker=args.chunker,
        retriever=args.retriever,
        embedder=args.embedder,
        fusion=args.fusion,
        rewriter=args.rewriter,
        reranker=args.reranker,
        top_k=args.top_k,
    )


def get_generator(args):
    if not args.generate:
        return None
    from ragkit.generate import get_generator as gg

    return gg(args.generator)


def cmd_run(args):
    docs, queries = load_dataset(args.dataset, args.limit)
    print(json.dumps(dataset_stats(docs, queries)))
    cfg = cfg_from_args(args)
    emb, rr = build_shared({cfg.embedder}, {cfg.reranker}, args.offline)
    rep = evaluate_retrieval(
        cfg, docs, queries,
        embedder=emb.get(cfg.embedder), reranker=rr.get(cfg.reranker),
        generator=get_generator(args), dataset=args.dataset,
    )
    print()
    print(format_table([rep]))
    print()
    print(json.dumps(rep.metrics, indent=2))
    if args.out:
        print("wrote", rep.save(args.out))


def cmd_sweep(args):
    docs, queries = load_dataset(args.dataset, args.limit)
    print(json.dumps(dataset_stats(docs, queries)))
    print()
    cfgs = PRESETS[args.preset](args.embedder)
    if args.offline:
        cfgs = [c for c in cfgs if c.embedder == "hashing" or c.retriever == "bm25"]
        cfgs = [c for c in cfgs if c.reranker != "cross-encoder"] or cfgs
    emb, rr = build_shared({c.embedder for c in cfgs}, {c.reranker for c in cfgs}, args.offline)
    out_dir = os.path.join(RESULTS_DIR, args.dataset, args.preset) if args.save else None
    reps = sweep(cfgs, docs, queries, embedders=emb, rerankers=rr,
                 generator=get_generator(args), dataset=args.dataset, out_dir=out_dir)
    print()
    cols = args.columns.split(",") if args.columns else None
    print(format_table(reps, cols))


def cmd_calibrate(args):
    """Sweep the abstention threshold on dev, report the choice on test.

    Reports the signal's AUC first. If that's near 0.5 the threshold is not
    the problem and no amount of tuning will fix it.
    """
    from ragkit.embed import get_embedder
    from ragkit.generate import ExtractiveGenerator
    from ragkit.pipeline import RAGPipeline

    docs, queries = load_dataset(args.dataset, args.limit)
    dev, test = split_dev_test(queries, args.dev_fraction, args.seed)
    print(f"dev={len(dev)} test={len(test)}")

    cfg = cfg_from_args(args)
    embedder = get_embedder(cfg.embedder) if cfg.retriever != "bm25" else None
    pipe = RAGPipeline(cfg, embedder=embedder).build(docs)
    gen = ExtractiveGenerator(threshold=-1.0)  # score everything, threshold later

    def confidences(qs):
        conf, ans = [], []
        for q in qs:
            res = pipe.retrieve(q.text, q.query_id)
            conf.append(gen.answer(q.text, res.chunks).confidence)
            ans.append(q.answerable)
        return conf, ans

    dev_conf, dev_ans = confidences(dev)
    print()
    print("abstention signal on dev:")
    print(json.dumps(summarize_signal(dev_conf, dev_ans), indent=2))

    curve = abstention_curve(dev_conf, dev_ans, args.steps)
    chosen = pick_threshold(curve, args.objective, args.max_hallucination)
    print(f"\nchosen threshold ({args.objective}): {chosen}")

    print(f"\n{'threshold':>10}{'halluc':>10}{'over-abst':>11}{'accuracy':>10}{'abstain%':>10}")
    for p in curve[:: max(1, len(curve) // 12)]:
        print(f"{p.threshold:>10.3f}{p.hallucination_rate:>10.3f}"
              f"{p.over_abstention_rate:>11.3f}{p.answer_accuracy:>10.3f}{p.abstain_share:>10.3f}")

    test_conf, test_ans = confidences(test)
    tp = fp = tn = fn = 0
    for c, a in zip(test_conf, test_ans):
        abst = c < chosen
        if a:
            fn += abst
            tp += not abst
        else:
            tn += abst
            fp += not abst
    print(f"\nheld out test @ threshold {chosen}:")
    print(json.dumps({
        "hallucination_rate": round(fp / max(fp + tn, 1), 4),
        "over_abstention_rate": round(fn / max(fn + tp, 1), 4),
        "answer_accuracy": round((tp + tn) / max(len(test_conf), 1), 4),
        "signal_auc_test": summarize_signal(test_conf, test_ans)["abstention_signal_auc"],
    }, indent=2))


def _baseline_path(dataset: str, cfg: RetrievalConfig) -> str:
    return os.path.join(BASELINE_DIR, f"{dataset}-{cfg.fingerprint()}.json")


def cmd_baseline(args):
    docs, queries = load_dataset(args.dataset, args.limit)
    cfg = cfg_from_args(args)
    emb, rr = build_shared({cfg.embedder}, {cfg.reranker}, args.offline)
    rep = evaluate_retrieval(cfg, docs, queries, embedder=emb.get(cfg.embedder),
                             reranker=rr.get(cfg.reranker), generator=get_generator(args),
                             dataset=args.dataset)
    path = _baseline_path(args.dataset, cfg)
    rep.save(path)
    print(f"froze baseline {cfg.label()} -> {path}")
    print(json.dumps(rep.metrics, indent=2))


def cmd_check(args):
    docs, queries = load_dataset(args.dataset, args.limit)
    cfg = cfg_from_args(args)
    path = _baseline_path(args.dataset, cfg)
    base = load_baseline(path)
    if base is None:
        print(f"no baseline at {path}, run `baseline` first", file=sys.stderr)
        return 2

    emb, rr = build_shared({cfg.embedder}, {cfg.reranker}, args.offline)
    cur = evaluate_retrieval(cfg, docs, queries, embedder=emb.get(cfg.embedder),
                             reranker=rr.get(cfg.reranker), generator=get_generator(args),
                             dataset=args.dataset)
    cmp = compare(base, cur)
    print(format_comparison(cmp))

    # floors are per config, not per dataset. bm25 scoring below the hybrid
    # floor is correct behaviour, not a regression.
    floors_path = os.path.join(BASELINE_DIR, f"{args.dataset}-{cfg.fingerprint()}-floors.json")
    failures = []
    if os.path.exists(floors_path):
        with open(floors_path) as f:
            failures = check_floors(cur, json.load(f))
        for msg in failures:
            print(f"  floor violated: {msg}")
    return 0 if (cmp.ok and not failures) else 1


def main(argv=None):
    p = argparse.ArgumentParser(prog="eval")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--dataset", default="scifact", choices=["scifact", "squad2"])
        sp.add_argument("--limit", type=int, default=None)
        sp.add_argument("--retriever", default="hybrid", choices=["bm25", "dense", "hybrid"])
        sp.add_argument("--embedder", default="minilm")
        sp.add_argument("--fusion", default="weighted", choices=["rrf", "weighted"])
        sp.add_argument("--chunker", default="whole")
        sp.add_argument("--rewriter", default="none")
        sp.add_argument("--reranker", default="none")
        sp.add_argument("--top-k", type=int, default=10)
        sp.add_argument("--generate", action="store_true", help="also evaluate answers")
        sp.add_argument("--generator", default="extractive")
        sp.add_argument("--offline", action="store_true",
                        help="skip anything that needs a downloaded model")
        return sp

    r = common(sub.add_parser("run"))
    r.add_argument("--out", default=None)
    r.set_defaults(fn=cmd_run)

    s = common(sub.add_parser("sweep"))
    s.add_argument("--preset", default="retrievers", choices=sorted(PRESETS))
    s.add_argument("--columns", default=None)
    s.add_argument("--save", action="store_true")
    s.set_defaults(fn=cmd_sweep)

    c = common(sub.add_parser("calibrate"))
    c.add_argument("--dev-fraction", type=float, default=0.4)
    c.add_argument("--seed", type=int, default=0)
    c.add_argument("--steps", type=int, default=41)
    c.add_argument("--objective", default="accuracy", choices=["accuracy", "safe"])
    c.add_argument("--max-hallucination", type=float, default=0.2)
    c.set_defaults(fn=cmd_calibrate)

    b = common(sub.add_parser("baseline"))
    b.set_defaults(fn=cmd_baseline)

    k = common(sub.add_parser("check"))
    k.set_defaults(fn=cmd_check)

    args = p.parse_args(argv)
    return args.fn(args) or 0


if __name__ == "__main__":
    sys.exit(main())
