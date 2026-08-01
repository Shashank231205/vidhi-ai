#!/usr/bin/env bash
# Deploy the backend to Fly.io.
#
#   ./deploy.sh            first run: creates the app, sets secrets, deploys
#   ./deploy.sh --update   subsequent runs: deploy only
#
# Secrets are read from backend/.env and pushed to Fly, so they never enter git
# and never have to be retyped.

set -euo pipefail

cd "$(git rev-parse --show-toplevel)"
APP="$(awk -F'"' '/^app =/{print $2}' fly.toml)"

if ! command -v flyctl >/dev/null 2>&1; then
  cat >&2 <<'INSTALL'
flyctl is not installed.

  brew install flyctl        # macOS
  flyctl auth signup         # or: flyctl auth login

INSTALL
  exit 1
fi

if [[ "${1:-}" != "--update" ]]; then
  if [[ ! -f backend/.env ]]; then
    echo "backend/.env not found — it holds the credentials to push." >&2
    exit 1
  fi

  echo "Creating $APP (no immediate deploy, so secrets land first)"
  flyctl apps create "$APP" 2>/dev/null || echo "  app already exists, continuing"

  # Only the keys the service actually reads. Anything else in .env — local
  # paths, the classifier location — would be wrong in production.
  WANTED=(
    DATABASE_URL UPSTASH_REDIS_URL UPSTASH_REDIS_TOKEN
    GROQ_API_KEY OPENROUTER_API_KEY CEREBRAS_API_KEY HF_API_TOKEN
  )

  args=()
  while IFS='=' read -r key value; do
    [[ -z "${key// }" || "$key" == \#* ]] && continue
    for wanted in "${WANTED[@]}"; do
      if [[ "$key" == "$wanted" && -n "$value" ]]; then
        args+=("$key=$value")
      fi
    done
  done < backend/.env

  if ((${#args[@]} == 0)); then
    echo "No credentials found in backend/.env" >&2
    exit 1
  fi

  echo "Setting ${#args[@]} secrets"
  flyctl secrets set --app "$APP" --stage "${args[@]}" >/dev/null
  echo "  done (values are not echoed)"
fi

echo
echo "Deploying — the first build takes 10-15 minutes"
flyctl deploy --app "$APP" --remote-only

URL="https://$APP.fly.dev"
echo
echo "Checking $URL"
curl -fsS --max-time 90 "$URL/health" >/dev/null && echo "  /health ok"
curl -fsS --max-time 90 "$URL/ready"  >/dev/null && echo "  /ready  ok"

cat <<DONE

Backend live at $URL

Point the frontend at it on Vercel:
  API_ORIGIN = $URL

Not NEXT_PUBLIC_ — the proxy runs server-side, so the backend URL never
reaches the browser.
DONE
