#!/usr/bin/env bash
# Print the environment variables to paste into Render, and sanity-check the
# image first.
#
#   ./deploy.sh
#
# Deployment itself happens in Render's dashboard from render.yaml — there is no
# CLI step and no billing account. This script exists so the credentials in
# backend/.env never have to be retyped or copied out of a file by hand.

set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

if [[ ! -f backend/.env ]]; then
  echo "backend/.env not found — it holds the credentials to copy." >&2
  exit 1
fi

# Only what the service needs at runtime. Deliberately excluded:
#
#   RISK_CLASSIFIER_MODEL — points at a local path, and .dockerignore keeps the
#   268MB weights out of the image. Setting it would make every audit attempt a
#   load that cannot succeed; unset, the agent uses the LLM path, which is also
#   the baseline the classifier is measured against.
WANTED=(
  DATABASE_URL UPSTASH_REDIS_URL UPSTASH_REDIS_TOKEN
  HF_API_TOKEN GROQ_API_KEY OPENROUTER_API_KEY CEREBRAS_API_KEY
)

cat <<'STEPS'

Deploying to Render
===================

1. Push this repo to GitHub, if it is not there already.

2. https://dashboard.render.com/blueprints -> New Blueprint Instance
   Pick this repo. Render reads render.yaml and creates the service.

3. It will prompt for the environment variables below. Paste them in.

STEPS

missing=()
while IFS='=' read -r key value; do
  [[ -z "${key// }" || "$key" == \#* ]] && continue
  value="${value%\"}"; value="${value#\"}"
  for wanted in "${WANTED[@]}"; do
    [[ "$key" == "$wanted" && -n "$value" ]] && printf '%s=%s\n' "$key" "$value"
  done
done < backend/.env

# HF_API_TOKEN is the one credential the deployed configuration cannot run
# without: EMBEDDING_BACKEND=remote authenticates every embedding call with it.
for required in DATABASE_URL HF_API_TOKEN; do
  grep -q "^${required}=" backend/.env || missing+=("$required")
done
if ((${#missing[@]})); then
  echo
  echo "WARNING: missing from backend/.env: ${missing[*]}" >&2
fi

cat <<'REST'

4. Deploy. The first build takes ~5 minutes.

5. Point the frontend at it on Vercel:
     API_ORIGIN = https://<your-service>.onrender.com

   Not NEXT_PUBLIC_ — the proxy runs server-side, so the backend URL never
   reaches the browser.

Note: the free instance sleeps after 15 minutes idle, and the next request
waits ~1 minute for it to wake.

REST
