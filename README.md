# VidhiAI

A unified AI legal platform for Indian law. One backend, one retrieval core, two modules:

- **ComplianceGuard** — audits contracts against Indian statutes and internal policy, flagging violations with exact citations.
- **CaseLens** — retrieves relevant Indian case law for a fact pattern and assesses whether each precedent supports or undermines your position.

Every output is grounded: a shared citation-verifier agent asserts that each cited
passage resolves to a real ingested chunk and that the quoted text actually appears
in it. Claims that fail are sent back to be re-grounded, not silently dropped.

See [PLAN.md](PLAN.md) for the full build plan.

## Status

| Phase | Scope | State |
|---|---|---|
| 0 | Foundation — config, logging, trace events, API skeleton, CI | ✅ backend done |
| 1 | Data layer — Supabase schema, pgvector, repositories | next |
| 2 | Ingestion + hybrid retrieval | |
| 3 | LLM router + prompts + ComplianceGuard v1 | |
| 4 | Citation verifier | |
| 5 | CaseLens v1 | |
| 6 | Custom classifiers (risk, stance) | |
| 7 | Next.js frontend | |
| 8 | Eval + deploy | |

## Requirements

- Python 3.12+
- [uv](https://docs.astral.sh/uv/)
- Accounts (all free tier): Supabase, Upstash, HuggingFace, and at least one of
  Groq / Cerebras / OpenRouter

## Setup

```bash
cp .env.example backend/.env   # then fill in credentials
make install
make dev
```

`make dev` serves the API on `:8000`. Check it:

```bash
curl localhost:8000/health
```

`status` is `ok` only when every dependency is configured; anything missing shows
up as `degraded` with a per-component reason. Interactive docs are at `/docs`
(local environment only).

## Development

```bash
make check      # lint + typecheck + tests, same as CI
make format     # ruff format and autofix
make test       # pytest with coverage
```

The test suite is hermetic — settings are constructed explicitly in fixtures and
nothing touches the network, so it runs without credentials.

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

Both modules are self-correcting LangGraph graphs rather than linear chains: a
critic node can reformulate a weak retrieval and retry, the analyzer can request
more context mid-reasoning, and the verifier can reject and re-ground a claim.
Every node transition emits a typed trace event streamed to the UI, so the
self-correction is visible while it happens.

### Layout

```
backend/
  api/
    main.py     app factory — wiring only, no business logic
    routes.py   gateway endpoints (/health)
  core/         shared infrastructure (config, logging, agents/trace)
  tests/
```

One `routes.py` per package holds every endpoint that package exposes, so there
is a single place to look for a module's surface. Domain packages
(`compliance/routes.py`, `caselens/routes.py`), `ml/`, `frontend/`, and `eval/`
land in their respective phases.

## Design rules

1. **No local state.** Database, cache, and model endpoints are hosted services in
   every environment, including development.
2. **Grounded or discarded.** No citation ships unverified.
3. **Latency is a feature.** Aggressive caching, parallel retrieval, streamed
   responses. Targets: retrieval p95 < 300ms, first token < 1.5s.
4. **Free tier throughout.** Total infra cost: ₹0.
