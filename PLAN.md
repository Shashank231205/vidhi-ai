# VidhiAI — Build Plan

Step-by-step execution plan. Derived from `VidhiAI_PRD.md` with four decisions locked in:

| Decision | Choice |
|---|---|
| Database | Supabase (hosted Postgres + pgvector) |
| Frontend | Next.js 15 + shadcn/ui + Tailwind |
| Classifiers | Trained on Colab, served via HuggingFace Inference API |
| Agent design | Self-correcting LangGraph loops (critic + retry edges) |
| Agent trace | Streamed live to the UI as structured events |
| Scope | Full build, phased — each phase ends in something runnable |

## Constraints (non-negotiable, from user)

1. **Company-grade UI/UX.** No default-AI-demo look. Real design system, real interaction states.
2. **No legacy/throwaway code.** Every module written once, properly.
3. **Free APIs / open-source models wherever possible.**
4. **Low latency.** Caching, streaming, parallel retrieval. Latency is a feature, not a v2 concern.
5. **Nothing local.** All state in hosted DB. No SQLite, no local model weights, no filesystem persistence.

---

## Corrections to the PRD

These are the places the PRD as written conflicts with the constraints above.

### Indian Kanoon API is paid
The PRD lists it as the CaseLens data source. It bills per document (~₹0.25–0.50/doc) with no bulk free tier. **Replacement:** open Indian judgment corpora on HuggingFace (pre-scraped, free, redistributable) plus the eCourts/SCI public portals for gap-filling. Ingestion code is written against a `JudgmentSource` interface so Kanoon can be dropped in later if you decide to pay for it.

### "No local" vs. fine-tuned models
Training needs a GPU. Resolution: train on free Colab/Kaggle GPU → push weights to HF Hub → serve over HF Inference API. Nothing local at any point. Cost: 1–3s cold start on the free tier, mitigated in Phase 5 by Redis-cached predictions and keep-warm pings.

### LLM provider
Gemini is out per user. **Primary:** Groq (Llama 3.3 70B — free tier, very fast). **Secondary:** Cerebras. **Fallback:** OpenRouter free models. All three are OpenAI-compatible, so one thin provider abstraction covers all of them with automatic failover on rate-limit.

### Redis
Upstash Redis (free tier, HTTP-based) rather than a container — keeps the "nothing local" rule intact.

### Deployment is split, and has to be
Vercel runs the frontend only. Its serverless functions cap at 250MB unzipped
and 10s on the free tier, while the backend needs ~3GB resident (BGE-M3 2.2GB,
DistilBERT 268MB, PyTorch ~800MB) and a contract audit takes 30-160s. The
backend therefore runs on HuggingFace Spaces — free Docker hosting with 16GB
RAM, the only free tier that fits local models. Spaces sleep after inactivity,
so a cold demo pays a ~30s wake-up.

---

## Architecture

```
Next.js 15 (Vercel)
        │  streaming SSE
FastAPI (Render/Fly)
        │
   ┌────┴────────────────────────────┐
   │   Shared Core                    │
   │   ingestion · embeddings ·       │
   │   hybrid retrieval · citation    │
   │   verifier · LLM router          │
   └────┬─────────────────┬───────────┘
        │                 │
  ComplianceGuard      CaseLens
   (LangGraph)         (LangGraph)
        │                 │
   ┌────┴─────────────────┴───────────┐
   │ Supabase (Postgres + pgvector)   │
   │ Upstash Redis (cache)            │
   │ HF Inference API (2 classifiers) │
   └──────────────────────────────────┘
```

---

## Agent design

The PRD describes linear pipelines. We build **self-correcting graphs** instead — the
path through the graph is decided at runtime by the state, not fixed in advance.

### ComplianceGuard graph

```
parse_contract
      │
      ▼
  ┌──────────────┐
  │ retrieve     │◄──────────┐
  └──────┬───────┘           │ reformulate query
         ▼                   │ (max 3 attempts)
  ┌──────────────┐           │
  │ critic       │───weak────┘
  └──────┬───────┘
    sufficient
         ▼
  ┌──────────────┐
  │ analyze      │──needs more context──► retrieve
  └──────┬───────┘
         ▼
  ┌──────────────┐
  │ classify     │  (risk model, Phase 6)
  └──────┬───────┘
         ▼
  ┌──────────────┐
  │ verify       │──rejected──► analyze (re-ground, max 2)
  └──────┬───────┘
       clean
         ▼
      emit flag
```

Key edges — these are what make it an agent rather than a chain:

- **retrieve → critic → retrieve.** A critic node scores whether retrieved statute text
  actually addresses the clause. If weak, it reformulates the query (legal synonyms,
  section-number expansion, broader statute scope) and retries. Bounded at 3 attempts.
- **analyze → retrieve.** Mid-reasoning, the analyzer can request additional context
  (e.g. a cross-referenced definitions section) instead of guessing.
- **verify → analyze.** The verifier rejects ungrounded claims and sends them back to be
  re-grounded rather than silently dropping them. Bounded at 2 retries, then discarded.

CaseLens uses the same skeleton: `retrieve → critic → stance → verify → synthesize`,
with the citation-graph expansion step able to pull in cases cited *by* strong hits.

### Streamed trace

Every node transition emits a typed event over SSE:

```
{ node, status, detail, elapsed_ms, attempt }
```

The frontend renders these live — "retrieving DPDP §8", "retrieved context weak,
reformulating", "verifying 3 citations". This makes multi-second latency legible
instead of dead, and makes the self-correction visible when it happens.

### Prompts as versioned assets

Prompts live in `backend/core/prompts/` as typed templates, never inline f-strings:

- One module per prompt, each exporting a template + a Pydantic output schema.
- Few-shot examples drawn from real Indian legal text, not invented ones.
- Golden-output tests, so a prompt change is measurable rather than vibes.
- Versioned (`v1`, `v2`) with the active version pinned in settings — lets us A/B a
  prompt revision against the eval set before switching.

Every prompt enforces the same discipline: structured output, explicit refusal to
answer beyond retrieved context, and mandatory citation of the chunk ID it relied on.
That last rule is what makes Phase 4 verification mechanically checkable.

## Phases

Each phase is independently verifiable. No phase depends on a later one.

### Phase 0 — Foundation
Repo skeleton, `uv`-managed Python deps, Next.js app, env config, typed settings, CI lint/test. Docker Compose for local dev only (services still point at hosted DB).
**Done when:** `make dev` brings up API + frontend, `/health` returns green, CI passes.

### Phase 1 — Data layer
Supabase project, pgvector extension, schema + migrations: `documents`, `chunks` (with `vector(1024)`), `statutes`, `judgments`, `citations`, `audit_log`. HNSW indexes. Row-level security. Typed repository layer — no raw SQL in business logic.
**Done when:** migrations apply clean, repository layer has passing integration tests against real Supabase.

### Phase 2 — Ingestion + retrieval
Legal-aware chunker (splits on sections/clauses, not fixed token windows — this materially affects retrieval quality). Embedding service (BGE-M3 via HF, 1024-dim, batched). Hybrid retriever: pgvector cosine + Postgres full-text BM25, fused with Reciprocal Rank Fusion. DPDP Act 2023 ingested end-to-end.
**Done when:** `scripts/ingest_statutes.py` loads DPDP into Supabase; retrieval eval harness reports precision@k/recall@k against a hand-built query set.

### Phase 3 — LLM router + prompt layer + ComplianceGuard v1
Provider abstraction (Groq → Cerebras → OpenRouter failover, streaming, token accounting). Versioned prompt package with typed output schemas and golden tests. LangGraph compliance agent with the self-correcting graph above: parse → retrieve ⇄ critic → analyze → flags. Typed trace events streamed over SSE.
**Done when:** POST a real contract, get streamed clause flags with statute citations, and observe the critic loop firing on a deliberately vague clause.

### Phase 4 — Citation verifier
Shared agent asserting every citation resolves to a real ingested chunk and the quoted text actually appears in it. Wired as a *retry edge*, not a filter: rejected claims go back to `analyze` to be re-grounded, and are discarded only after 2 failed attempts. This is the groundedness guarantee — the PRD's non-negotiable 100% metric.
**Done when:** eval set shows zero ungrounded citations; verifier rejects deliberately planted fabrications and the retry edge is observable in the trace.

### Phase 5 — CaseLens v1
Judgment corpus ingestion, citation-graph builder (which case cites which), fact-pattern → ranked precedents, synthesis memo. Reuses Phase 2 retrieval and Phase 4 verifier unchanged.
**Done when:** fact pattern in → ranked verified precedents out.

### Phase 6 — Custom ML
`risk_classifier` (clause → high/med/low) and `stance_classifier` (fact+case → supports/undermines/neutral). InLegalBERT base, trained on Colab, pushed to HF Hub, served via Inference API behind a Redis cache. Measured against the LLM-prompting baseline from Phases 3/5.
**Done when:** both models beat baseline on held-out data, before/after numbers documented.

### Phase 7 — Frontend
Next.js + shadcn/ui. Design system first (type scale, color tokens, spacing, motion), then: contract upload with live streaming flags, clause-level diff view, precedent explorer, citation hover-cards resolving to source text. Full light/dark, keyboard-navigable, loading/empty/error states everywhere.
**Done when:** both modules usable end-to-end by someone who hasn't read the docs.

### Phase 8 — Eval + polish
Unified eval harness, metrics dashboard, p50/p95 latency and cost-per-query logging, README, architecture diagram, deployed demo.
**Done when:** deployed, documented, reproducible.

---

## Stack

| Layer | Choice | Free tier |
|---|---|---|
| DB | Supabase Postgres + pgvector | 500MB |
| Cache | Upstash Redis | 10k cmd/day |
| LLM | Groq → Cerebras → OpenRouter | generous |
| Embeddings | BGE-M3 via HF Inference | free |
| Classifiers | InLegalBERT fine-tuned, HF Hub | free |
| Training | Colab / Kaggle GPU | free |
| API host | HuggingFace Spaces (Docker) | free, 16GB RAM |
| Frontend host | Vercel | free |

Total infra cost: ₹0.

---

## Latency plan

Cold-path work is pushed off the request path and cached aggressively:

- Embeddings cached by content hash (identical clause never re-embedded).
- Classifier predictions cached by input hash.
- Per-clause retrieval and analysis run concurrently, not sequentially.
- Responses stream — first token visible fast, rather than waiting on a complete answer.
- Keep-warm pings against HF endpoints to avoid cold starts.
- HNSW (not IVFFlat) indexes for low-latency ANN at this corpus size.

**Targets:** retrieval p95 < 300ms · first token < 1.5s · full clause analysis p95 < 8s.
