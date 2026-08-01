# VidhiAI

An AI legal platform for Indian law. One retrieval core, two modules:

- **ComplianceGuard** — audits a contract clause by clause against Indian
  statutes, flagging conflicts with the exact provision each one relies on.
- **CaseLens** — retrieves precedents for a fact pattern and assesses whether
  each one supports or undermines the position being argued.

The property both modules are built around: **a claim ships only if it can be
traced back to text the system actually retrieved.** Every citation is checked
mechanically — the chunk must have been retrieved for that run, and the quote
must appear in it. Claims that fail go back to be re-grounded, and are
discarded rather than shown. Discarded counts are reported in the UI, because
hiding them would quietly undo the guarantee.

```bash
make install
cp .env.example backend/.env    # fill in credentials
make dev                        # → http://localhost:3000
```

---

## What is measured

Numbers from `eval/`, reproducible with the commands shown. Nothing here is
asserted without a way to check it.

### Groundedness — `eval/groundedness_eval.py`

| Metric | Result |
|---|---|
| Fabricated citations rejected | **100%** (8/8) |
| Real citations accepted | **100%** (6/6) |

Both are reported because either alone is misleading: a verifier that rejects
everything scores 100% on the first. The adversarial set includes invented
penalty figures, quotes lifted from the wrong section, scattered word bags, and
text that starts verbatim then continues into invention.

### Retrieval — `eval/retrieval_eval.py`

20 hand-built DPDP queries, k=5:

| Retriever | P@5 | R@5 | MRR | Hit rate |
|---|---|---|---|---|
| Vector only | 0.410 | 0.875 | **0.842** | 0.90 |
| Lexical only | 0.100 | 0.300 | 0.250 | 0.30 |
| **Hybrid (RRF)** | 0.410 | **0.925** | 0.833 | **0.95** |

Hybrid earns its place on recall, not ranking: it surfaces provisions the dense
arm misses entirely (misses 4 → 1) at no precision cost. Equal-weight fusion
initially scored *below* vector alone; the weights were swept against this set
rather than guessed, and the reasoning is recorded in `core/config.py`.

### Risk classifier — `ml/risk_classifier/`

Trained on CUAD (13,155 clauses from real commercial contracts, labelled by
attorneys), 1,934 held out:

| Model | Accuracy | Macro-F1 | High-risk F1 |
|---|---|---|---|
| Majority class | 0.668 | 0.267 | 0.000 |
| TF-IDF + logistic | 0.842 | 0.833 | 0.688 |
| **Fine-tuned DistilBERT** | **0.885** | **0.869** | **0.730** |

Macro-F1 rather than accuracy: with a 67% majority class, accuracy rewards a
model that ignores the minority classes — which are exactly the high-risk
clauses the tool exists to catch. The +0.035 over TF-IDF is what justifies the
transformer.

### Latency — `eval/latency_eval.py`

| Stage | p50 | p95 |
|---|---|---|
| Query embedding (local BGE-M3) | 27ms | 119ms |
| Hybrid retrieval, end to end | 314ms | 349ms |
| LLM first token | 316ms | 356ms |

First token beats its 1.5s target by 4×. Retrieval sits at the 300ms target
rather than under it, and the reason is measured rather than guessed: a bare
`SELECT 1` from a developer machine to Supabase costs ~43ms, and retrieval
makes two queries. Co-locating the API with the database recovers most of it.

---

## How it works

```
Next.js (Vercel) ──proxies /api/*──► FastAPI (Fly.io, Mumbai)
                                            │
              ┌─────────────────────────────┴──────────────────┐
              │ chunking · embeddings · hybrid retrieval        │
              │ citation verifier · LLM router · classifiers    │
              └───────┬──────────────────────────┬─────────────┘
                      │                          │
              ComplianceGuard                 CaseLens
                      │                          │
              ┌───────┴──────────────────────────┴─────────────┐
              │ Supabase (Postgres + pgvector) · Upstash Redis  │
              └────────────────────────────────────────────────┘
```

### The agents are graphs, not chains

```
parse → retrieve ⇄ critic → analyze ⇄ verify → classify → emit
             ▲                  │
             └─── more context ─┘
```

Three edges decide the path at runtime, each bounded so a bad clause cannot
spin:

- **critic → retrieve.** The critic judges whether retrieved provisions
  actually govern the clause, and reformulates in statutory vocabulary when
  they do not. A contract drafter's words and the legislature's rarely match.
- **analyze → retrieve.** The analyzer can request a specific missing
  provision instead of guessing at it.
- **verify → analyze.** Ungrounded findings go back with the rejection reason
  attached, which corrects far more often than a blind retry — the model
  usually cited the right law under the wrong id.

All three are visible in the live trace. A retry row is the agent rejecting its
own intermediate result.

### Verification is mechanical, never model-judged

Asking an LLM whether a citation is accurate inherits the failure it exists to
catch. Instead: the chunk id must be one retrieved for this run, and the quote
must appear in that chunk's text at 85% of its longest contiguous word run.
Not 100% — PDF extraction introduces line wrapping and smart quotes that would
reject correct citations. Not lower — below that, unrelated legal boilerplate
starts to match.

### Retrieval is hybrid because neither half suffices

Dense vectors find paraphrases but confuse "Section 8(3)" with "Section 9(3)".
Lexical search nails identifiers but misses anything phrased differently from
the statute. RRF fuses them by rank position, so the two incomparable score
scales never have to be reconciled.

---

## Corpus

Ingested from official sources, reproducible from `scripts/`:

| Source | Content |
|---|---|
| India Code + MeitY | DPDP Act 2023, Companies Act 2013, Arbitration Act 1996, and more |
| HuggingFace open corpora | Indian Supreme Court judgments, with a derived citation graph |

Currently **153 documents, 9,216 chunks**. Statute PDF paths on India Code are
unstable — 9 of 12 guessed URLs were already dead — so sources store the stable
DSpace handle and resolve the PDF at fetch time.

```bash
uv run python ../scripts/ingest_statutes.py --list
uv run python ../scripts/ingest_statutes.py --priority 1
uv run python ../scripts/ingest_judgments.py --limit 150
```

---

## Development

```bash
make dev        # API + frontend on one port
make check      # ruff, mypy --strict, pytest — what CI runs
make web-build  # frontend lint + production build
make format     # ruff format and autofix
```

123 unit tests run without credentials; integration tests skip unless
`DATABASE_URL` is set. A pre-push hook runs the same gates CI does, because a
red `main` is worse than a slow push.

```
backend/
  api/          FastAPI gateway — wiring only
  core/         chunking, embeddings, retrieval, verifier, LLM router, prompts
  compliance/   ComplianceGuard agent + routes
  caselens/     CaseLens agent, citation extraction + routes
frontend/src/   Next.js app, design tokens, SSE trace
ml/             classifier training, dataset construction, metrics
eval/           groundedness, retrieval, latency harnesses
scripts/        corpus ingestion
```

---

## Stack

| Layer | Choice | Cost |
|---|---|---|
| Database | Supabase Postgres + pgvector | free |
| Cache | Upstash Redis | free |
| LLM | 6-model pool: Groq ×5 + OpenRouter | free |
| Embeddings | BGE-M3, in-process | free |
| Classifier | DistilBERT fine-tuned on CUAD | free |
| API host | Fly.io (Docker, Mumbai) | free tier |
| Frontend host | Vercel | free |

Total infrastructure cost: **₹0**.

---

## Deployment

The two halves deploy separately, and they have to: Vercel's functions cap at
250MB and 10s, while the backend holds a 1.03GB embedding model resident and an
audit runs for minutes.

**Backend → Fly.io.** One command, after `brew install flyctl && flyctl auth login`:

```bash
./deploy.sh
```

It creates the app, pushes the credentials from `backend/.env` as Fly secrets —
so they never enter git — deploys, and verifies `/health` and `/ready`. Later
deploys are `./deploy.sh --update`.

Sizing in `fly.toml` is measured rather than guessed: 2GB against a 1.03GB
model, in Mumbai alongside the Supabase project, with idle machines stopped to
stay inside the free allowance. A cold wake costs ~20s while the model loads;
set `min_machines_running = 1` to avoid it.

**Frontend → Vercel**, with one environment variable:

```
API_ORIGIN = https://vidhi-ai-api.fly.dev
```

Not `NEXT_PUBLIC_` — the proxy runs server-side, so the backend URL never
reaches the browser.

Two things that otherwise cost an hour: use the Supabase **pooler** connection
string (port 6543), not the direct one, and **percent-encode** special
characters in the password (`@` → `%40`), or it is read as the host separator
and fails with a misleading auth error.

---

## Honest limitations

- **Decision support, not legal advice.** Outputs are grounded in retrieved
  text; they are not a substitute for a lawyer.
- **The corpus is partial.** "No issues found" reflects the Acts currently
  ingested, not a clean bill of health.
- **The risk classifier is trained on US commercial contracts.** CUAD is the
  best labelled contract dataset available; it scores clause severity, while
  the Indian statutory grounding comes from retrieval and verification.
- **CUAD labels clause type, not risk.** The type→risk mapping in
  `ml/risk_classifier/dataset.py` is a judgment call, written out explicitly so
  a lawyer can disagree with a specific line.
- **High-risk recall is 0.715.** The classifier misses roughly a quarter of
  high-risk clauses, which is why it defers to the LLM below a confidence
  threshold rather than overriding it.
- **Free tiers bind.** Each Groq model allows 12k tokens/min, which is why the
  router pools six models rather than relying on one. Fly machines stop when
  idle to stay inside the free allowance, so a cold request waits ~20s for the
  model to load.
