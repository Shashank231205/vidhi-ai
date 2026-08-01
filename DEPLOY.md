# Deployment

The frontend and backend deploy to different hosts, and they have to.

Vercel's serverless functions cap at 250MB unzipped and 10s of execution on the
free tier. The backend holds ~3GB resident — BGE-M3 at 2.2GB, the risk
classifier at 268MB, PyTorch around 800MB — and a contract audit runs for 30 to
160 seconds. Neither number fits. So:

| Component | Host | Why |
|---|---|---|
| Frontend | Vercel | Static Next.js output; free, fast, global |
| Backend | HuggingFace Spaces (Docker) | 16GB RAM free — the only free tier that fits the models |
| Postgres | Supabase | Already provisioned |
| Redis | Upstash | Already provisioned |

The browser only ever talks to the frontend origin. Next proxies `/api/*` to the
backend server-side, so there is one URL and no CORS to configure.

---

## 1. Backend → HuggingFace Spaces

**Create the Space**

1. https://huggingface.co/new-space
2. Name it `vidhi-ai-api`, SDK **Docker**, hardware **CPU basic (free)**
3. Visibility: public is fine — no secrets live in the image

**Set the secrets** (Settings → Variables and secrets → *New secret*)

| Secret | Value |
|---|---|
| `DATABASE_URL` | Supabase **transaction pooler** string, port 6543, password percent-encoded |
| `UPSTASH_REDIS_URL` | Upstash REST endpoint |
| `UPSTASH_REDIS_TOKEN` | Upstash REST token |
| `GROQ_API_KEY` | console.groq.com/keys |
| `OPENROUTER_API_KEY` | Optional fallback |
| `ENVIRONMENT` | `production` |
| `RISK_CLASSIFIER_MODEL` | Optional; an HF repo id once the model is pushed |

Two things that will otherwise cost an hour:

- Use the **pooler** connection string (port 6543), not the direct one. The
  direct endpoint refuses connections from most PaaS networks.
- **Percent-encode** special characters in the password. A literal `@` is read
  as the host separator and fails with a misleading authentication error:
  `@` → `%40`, `#` → `%23`, `/` → `%2F`.

**Push**

```bash
git remote add space https://huggingface.co/spaces/<user>/vidhi-ai-api
git push space main
```

The first build takes 10–15 minutes: it installs CPU-only Torch and bakes the
2.2GB embedding model into the image. That download is deliberate — a sleeping
Space that pulled the model on wake would stall the first request for minutes
rather than seconds.

**Verify**

```bash
curl https://<user>-vidhi-ai-api.hf.space/health   # configuration only
curl https://<user>-vidhi-ai-api.hf.space/ready    # actually probes Postgres
```

`/health` answers from configuration and never touches the network, so a slow
database cannot cause a restart loop. `/ready` is the one to gate traffic on.

---

## 2. Frontend → Vercel

```bash
cd frontend
npx vercel --prod
```

Set one environment variable in the Vercel project:

```
API_ORIGIN = https://<user>-vidhi-ai-api.hf.space
```

It is **not** `NEXT_PUBLIC_`: the proxy runs on Vercel's server, so the backend
URL never reaches the browser and the API is not directly addressable from it.

---

## 3. Seed the corpus

Ingestion runs from your machine against the shared Supabase instance — the
Space does not need to do it, and doing it there would repeat on every rebuild.

```bash
cd backend
uv run python ../scripts/ingest_statutes.py --priority 1   # core Acts
uv run python ../scripts/ingest_judgments.py --limit 150   # case law + graph
```

---

## Known limits

**Spaces sleep after ~48h idle.** The first request after that pays a ~30s
wake-up. A cron ping every few hours avoids it if the demo needs to be instant.

**Groq's free tier is 12,000 tokens per minute.** A long contract sends a lot of
statutory context per clause and will meet that ceiling; the router waits out
short `retry-after` windows and fails over to OpenRouter otherwise, so it
degrades rather than breaks.

**Supabase free tier is 500MB.** The current corpus — 3 Acts and 150 judgments,
9,216 chunks — uses a fraction of it, but the full 7,130-judgment dataset would
not fit.

**Retrieval latency depends on where the API runs.** From a developer machine a
bare `SELECT 1` to Supabase costs ~43ms, which puts a floor under the ~300ms
p50 measured locally. Co-located with the database it is materially lower.
