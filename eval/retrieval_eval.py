"""Retrieval evaluation.

Measures the hybrid retriever against each arm alone, because "hybrid search"
is only worth its complexity if it actually beats both halves. Reports
precision@k, recall@k, MRR, and hit rate on a hand-built query set.

    uv run python ../eval/retrieval_eval.py
    uv run python ../eval/retrieval_eval.py --k 5
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from core.cache import Cache  # noqa: E402
from core.config import get_settings  # noqa: E402
from core.db import Database  # noqa: E402
from core.embeddings import EmbeddingService  # noqa: E402
from core.retrieval import HybridRetriever, reciprocal_rank_fusion  # noqa: E402

DATASET = Path(__file__).resolve().parent / "datasets" / "dpdp_retrieval.json"


@dataclass(slots=True)
class Metrics:
    """Aggregate retrieval quality over a query set."""

    precision_at_k: float
    recall_at_k: float
    mrr: float
    hit_rate: float
    p50_ms: float
    p95_ms: float
    queries: int

    def row(self, label: str) -> str:
        return (
            f"{label:<12} {self.precision_at_k:>9.3f} {self.recall_at_k:>8.3f} "
            f"{self.mrr:>7.3f} {self.hit_rate:>9.3f} {self.p50_ms:>8.0f} "
            f"{self.p95_ms:>8.0f}"
        )


def _section_of(label: str | None) -> str | None:
    """Normalise a chunk label to its section: 'Section 8(3)' -> 'Section 8'.

    Relevance is judged at section level: a sub-section of the right provision
    is a correct retrieval, since it cites the same law.
    """
    if not label or not label.startswith("Section "):
        return None
    head = label.removeprefix("Section ").split("(")[0].split(" ")[0].strip()
    return f"Section {head}" if head else None


def score(
    retrieved: list[str | None], relevant: set[str], k: int
) -> tuple[float, float, float, float]:
    """precision@k, recall@k, reciprocal rank, hit."""
    sections = [_section_of(label) for label in retrieved[:k]]
    matched = [s for s in sections if s in relevant]

    precision = len(matched) / k if k else 0.0
    recall = len({s for s in matched}) / len(relevant) if relevant else 0.0

    rr = 0.0
    for rank, section in enumerate(sections, start=1):
        if section in relevant:
            rr = 1.0 / rank
            break

    return precision, recall, rr, 1.0 if matched else 0.0


def aggregate(
    results: list[tuple[float, float, float, float]], latencies: list[float]
) -> Metrics:
    n = len(results) or 1
    ordered = sorted(latencies)
    return Metrics(
        precision_at_k=sum(r[0] for r in results) / n,
        recall_at_k=sum(r[1] for r in results) / n,
        mrr=sum(r[2] for r in results) / n,
        hit_rate=sum(r[3] for r in results) / n,
        p50_ms=ordered[len(ordered) // 2] if ordered else 0.0,
        p95_ms=ordered[min(int(len(ordered) * 0.95), len(ordered) - 1)]
        if ordered
        else 0.0,
        queries=len(results),
    )


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, default=5, help="cutoff for @k metrics")
    args = parser.parse_args()

    logging.disable(logging.INFO)

    dataset = json.loads(DATASET.read_text())
    queries = dataset["queries"]

    settings = get_settings()
    database = Database(settings)
    cache = Cache(settings)
    embeddings = EmbeddingService(settings, cache=cache)

    per_arm = max(args.k * 3, 20)
    scores: dict[str, list[tuple[float, float, float, float]]] = {
        "vector": [],
        "lexical": [],
        "hybrid": [],
    }
    latencies: dict[str, list[float]] = {"vector": [], "lexical": [], "hybrid": []}

    try:
        async with database.session() as session:
            retriever = HybridRetriever(session, embeddings, settings)

            for item in queries:
                relevant = set(item["relevant_sections"])

                # Each arm is measured on its own so the fusion's contribution
                # is visible rather than assumed.
                start = time.perf_counter()
                vector = await embeddings.embed_one(item["query"])
                vector_hits = await retriever.vector_search(vector, per_arm, None, None)
                latencies["vector"].append((time.perf_counter() - start) * 1000)

                start = time.perf_counter()
                lexical_hits = await retriever.lexical_search(
                    item["query"], settings.lexical_candidate_limit, None, None
                )
                latencies["lexical"].append((time.perf_counter() - start) * 1000)

                start = time.perf_counter()
                fused = reciprocal_rank_fusion(
                    [vector_hits, lexical_hits],
                    k=settings.rrf_k,
                    limit=args.k,
                    weights=[
                        settings.rrf_vector_weight,
                        settings.rrf_lexical_weight,
                    ],
                )
                latencies["hybrid"].append(
                    latencies["vector"][-1]
                    + latencies["lexical"][-1]
                    + (time.perf_counter() - start) * 1000
                )

                for name, hits in (
                    ("vector", vector_hits),
                    ("lexical", lexical_hits),
                    ("hybrid", fused),
                ):
                    scores[name].append(
                        score([h.label for h in hits], relevant, args.k)
                    )
    finally:
        await embeddings.close()
        await cache.close()
        await database.dispose()

    print(f"\nRetrieval evaluation — DPDP Act 2023, {len(queries)} queries, k={args.k}\n")
    header = (
        f"{'retriever':<12} {'P@k':>9} {'R@k':>8} {'MRR':>7} "
        f"{'hit-rate':>9} {'p50 ms':>8} {'p95 ms':>8}"
    )
    print(header)
    print("-" * len(header))

    computed = {
        name: aggregate(scores[name], latencies[name])
        for name in ("vector", "lexical", "hybrid")
    }
    for name in ("vector", "lexical", "hybrid"):
        print(computed[name].row(name))

    # Judged on recall as well as MRR: on a small single-statute corpus the
    # dense arm dominates the ranking, and the lexical arm earns its place by
    # surfacing relevant provisions the vector arm misses entirely.
    mrr_delta = computed["hybrid"].mrr - max(
        computed["vector"].mrr, computed["lexical"].mrr
    )
    recall_delta = computed["hybrid"].recall_at_k - max(
        computed["vector"].recall_at_k, computed["lexical"].recall_at_k
    )
    print(f"\nHybrid vs. best single arm:  MRR {mrr_delta:+.3f}   R@k {recall_delta:+.3f}")
    if recall_delta > 0:
        print("  Fusion adds recall: it finds provisions no single arm ranked.")
    elif mrr_delta >= 0:
        print("  Fusion holds ranking without loss.")
    else:
        print("  Fusion is not paying for itself — re-tune weights.")

    misses = [
        item["id"]
        for item, result in zip(queries, scores["hybrid"], strict=True)
        if result[3] == 0.0
    ]
    if misses:
        print(f"\nMissed entirely ({len(misses)}): {', '.join(misses)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
