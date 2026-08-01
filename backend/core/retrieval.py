"""Hybrid retrieval: dense vectors + lexical BM25, fused with RRF.

Neither half is sufficient for legal search on its own:

- Vector search finds paraphrases ("can they share my data?" → a disclosure
  provision) but is unreliable on exact identifiers — "Section 8(3)" and
  "Section 9(3)" sit close together in embedding space.
- Lexical search nails identifiers and defined terms but misses everything the
  user phrased differently from the statute.

Reciprocal Rank Fusion combines them without needing the two score scales to be
comparable, which they are not: cosine similarity and ts_rank have no shared
units, so normalising them would mean inventing a conversion. RRF only reads
rank position, sidestepping that entirely.
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from core.config import Settings
from core.db.models import DocumentKind
from core.db.session import Database
from core.embeddings import EmbeddingService
from core.logging import get_logger

log = get_logger(__name__)


@dataclass(slots=True)
class Hit:
    """One retrieved chunk with its provenance.

    `chunk_id` and `label` are what a citation must reference, so they travel
    with every hit rather than being looked up again later.
    """

    chunk_id: uuid.UUID
    document_id: uuid.UUID
    content: str
    label: str | None
    document_title: str
    source_ref: str
    score: float
    #: Which retrievers found this, for trace display and debugging.
    matched_by: tuple[str, ...] = ()
    meta: dict[str, Any] | None = None

    @property
    def citation(self) -> str:
        """Human-readable source, e.g. 'DPDP Act 2023, Section 8(3)'."""
        return f"{self.document_title}, {self.label}" if self.label else self.document_title


_VECTOR_SQL = text(
    """
    SELECT c.id, c.document_id, c.content, c.label, c.meta,
           d.title, d.source_ref,
           1 - (c.embedding <=> CAST(:query_vec AS vector)) AS score
    FROM chunks c
    JOIN documents d ON d.id = c.document_id
    WHERE c.embedding IS NOT NULL
      AND (CAST(:kind AS text) IS NULL OR d.kind = CAST(:kind AS text))
      AND (CAST(:document_id AS uuid) IS NULL
           OR c.document_id = CAST(:document_id AS uuid))
    ORDER BY c.embedding <=> CAST(:query_vec AS vector)
    LIMIT :limit
    """
)

#: OR-semantics, deliberately. websearch_to_tsquery and plainto_tsquery both
#: AND every term, so a single word absent from the statute — "company",
#: "failing" — drops the match to zero. Natural questions nearly always contain
#: such a word, which silently disabled the lexical arm exactly when it was
#: most needed. Ranking, not matching, is what should decide relevance here:
#: ts_rank_cd rewards chunks containing more of the query terms, and RRF only
#: reads the resulting order.
_LEXICAL_SQL = text(
    """
    WITH q AS (
        SELECT to_tsquery(
            'english',
            array_to_string(
                ARRAY(
                    SELECT lexeme FROM unnest(to_tsvector('english', :query))
                ),
                ' | '
            )
        ) AS tsq
    )
    SELECT c.id, c.document_id, c.content, c.label, c.meta,
           d.title, d.source_ref,
           ts_rank_cd(c.tsv, q.tsq) AS score
    FROM chunks c
    JOIN documents d ON d.id = c.document_id
    CROSS JOIN q
    WHERE q.tsq IS NOT NULL
      AND c.tsv @@ q.tsq
      AND (CAST(:kind AS text) IS NULL OR d.kind = CAST(:kind AS text))
      AND (CAST(:document_id AS uuid) IS NULL
           OR c.document_id = CAST(:document_id AS uuid))
    ORDER BY score DESC
    LIMIT :limit
    """
)


def _to_hit(row: Any, score: float, matched_by: str) -> Hit:
    return Hit(
        chunk_id=row.id,
        document_id=row.document_id,
        content=row.content,
        label=row.label,
        document_title=row.title,
        source_ref=row.source_ref,
        score=score,
        matched_by=(matched_by,),
        meta=row.meta,
    )


def reciprocal_rank_fusion(
    rankings: list[list[Hit]],
    *,
    k: int = 60,
    limit: int = 8,
    weights: list[float] | None = None,
) -> list[Hit]:
    """Fuse ranked lists by rank position rather than score.

    A chunk found by both retrievers accumulates contributions from each, so
    agreement between the two is what surfaces a result — which is precisely
    the signal we want when the score scales are incomparable.

    `weights` exists because the arms are not equally trustworthy here. With
    OR-semantics the lexical arm returns a long, weak tail — a chunk sharing
    only the word "data" still matches — and unweighted RRF lets that tail
    outvote a confident vector hit. Measured on the DPDP eval set, equal
    weighting scored *below* vector alone (MRR 0.633 vs 0.842); down-weighting
    lexical recovers the fusion's benefit without discarding the exact-term
    matching that vectors are bad at.
    """
    if weights is None:
        weights = [1.0] * len(rankings)

    scores: dict[uuid.UUID, float] = {}
    best: dict[uuid.UUID, Hit] = {}
    sources: dict[uuid.UUID, set[str]] = {}

    for ranking, weight in zip(rankings, weights, strict=True):
        for rank, hit in enumerate(ranking, start=1):
            scores[hit.chunk_id] = scores.get(hit.chunk_id, 0.0) + weight / (k + rank)
            sources.setdefault(hit.chunk_id, set()).update(hit.matched_by)
            best.setdefault(hit.chunk_id, hit)

    fused = []
    for chunk_id, score in sorted(scores.items(), key=lambda kv: kv[1], reverse=True):
        hit = best[chunk_id]
        hit.score = score
        hit.matched_by = tuple(sorted(sources[chunk_id]))
        fused.append(hit)
    return fused[:limit]


class HybridRetriever:
    def __init__(
        self,
        session: AsyncSession,
        embeddings: EmbeddingService,
        settings: Settings,
        database: Database | None = None,
    ) -> None:
        self._session = session
        self._embeddings = embeddings
        self._settings = settings
        # Optional: when supplied, the two arms each take their own session so
        # they can run concurrently. Without it they share `session` and run in
        # sequence — correct, but roughly twice the latency.
        self._database = database

    async def _vector_on_new_session(
        self,
        query: str,
        limit: int,
        kind: DocumentKind | None,
        document_id: uuid.UUID | None,
    ) -> list[Hit]:
        assert self._database is not None
        vector = await self._embeddings.embed_one(query)
        async with self._database.session() as session:
            retriever = HybridRetriever(session, self._embeddings, self._settings)
            return await retriever.vector_search(vector, limit, kind, document_id)

    async def _lexical_on_new_session(
        self,
        query: str,
        limit: int,
        kind: DocumentKind | None,
        document_id: uuid.UUID | None,
    ) -> list[Hit]:
        assert self._database is not None
        async with self._database.session() as session:
            retriever = HybridRetriever(session, self._embeddings, self._settings)
            return await retriever.lexical_search(query, limit, kind, document_id)

    async def vector_search(
        self,
        vector: list[float],
        limit: int,
        kind: DocumentKind | None,
        document_id: uuid.UUID | None,
    ) -> list[Hit]:
        result = await self._session.execute(
            _VECTOR_SQL,
            {
                # pgvector's text input format; asyncpg has no native binding.
                "query_vec": "[" + ",".join(str(v) for v in vector) + "]",
                "limit": limit,
                "kind": kind.value if kind else None,
                "document_id": document_id,
            },
        )
        return [_to_hit(r, float(r.score), "vector") for r in result]

    async def lexical_search(
        self,
        query: str,
        limit: int,
        kind: DocumentKind | None,
        document_id: uuid.UUID | None,
    ) -> list[Hit]:
        result = await self._session.execute(
            _LEXICAL_SQL,
            {
                "query": query,
                "limit": limit,
                "kind": kind.value if kind else None,
                "document_id": document_id,
            },
        )
        return [_to_hit(r, float(r.score), "lexical") for r in result]

    async def search(
        self,
        query: str,
        *,
        limit: int | None = None,
        kind: DocumentKind | None = None,
        document_id: uuid.UUID | None = None,
    ) -> list[Hit]:
        """Run both retrievers and fuse the results.

        Three round trips are involved — embed, vector query, lexical query —
        and against a hosted database each carries real network latency. They
        are overlapped as far as correctness allows:

        - The embedding runs concurrently with the lexical query, since the
          lexical arm does not need a vector.
        - The two database queries run on *separate sessions* when a Database
          was supplied. An AsyncSession is not concurrency-safe: sharing one
          across gathered tasks corrupts its transaction state, which is a bug
          this code shipped with once already. With one session they fall back
          to running in sequence, which is correct but slower.
        """
        limit = limit or self._settings.retrieval_top_k
        # Over-fetch per arm so fusion has room to promote agreed-upon hits.
        per_arm = max(limit * 3, 20)
        # Capped shallower than the vector arm: past the top few, OR-matched
        # lexical results share only common words and add noise to the fusion.
        lexical_depth = min(per_arm, self._settings.lexical_candidate_limit)

        if self._database is not None:
            vector_hits, lexical_hits = await asyncio.gather(
                self._vector_on_new_session(query, per_arm, kind, document_id),
                self._lexical_on_new_session(query, lexical_depth, kind, document_id),
            )
        else:
            vector_task = asyncio.create_task(self._embeddings.embed_one(query))
            lexical_hits = await self.lexical_search(query, lexical_depth, kind, document_id)
            query_vector = await vector_task
            vector_hits = await self.vector_search(query_vector, per_arm, kind, document_id)

        fused = reciprocal_rank_fusion(
            [vector_hits, lexical_hits],
            k=self._settings.rrf_k,
            limit=limit,
            weights=[
                self._settings.rrf_vector_weight,
                self._settings.rrf_lexical_weight,
            ],
        )
        log.info(
            "retrieval",
            query_chars=len(query),
            vector_hits=len(vector_hits),
            lexical_hits=len(lexical_hits),
            returned=len(fused),
        )
        return fused
