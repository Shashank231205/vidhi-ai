#!/usr/bin/env bash
# Deploy the backend to Google Cloud Run.
#
#   ./deploy.sh            first run: enables APIs, pushes secrets, deploys
#   ./deploy.sh --update   subsequent runs: build and deploy only
#
# Cloud Run rather than Fly.io: the image is ~9GB because BGE-M3's weights are
# baked in, and Fly's registry push failed on export after 158s. Cloud Run
# states no limit on image size and streams images block-by-block at boot, so a
# multi-GB image starts in about the same time as a small one.
#
# Credentials are read from backend/.env into Secret Manager, so they never
# enter git and never have to be retyped.

set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

SERVICE="${SERVICE:-vidhi-ai}"
REGION="${REGION:-asia-south1}"   # Mumbai — same region as Supabase
PROJECT="${PROJECT:-$(gcloud config get-value project 2>/dev/null)}"

if ! command -v gcloud >/dev/null 2>&1; then
  cat >&2 <<'INSTALL'
gcloud is not installed.

  brew install --cask google-cloud-sdk
  gcloud auth login
  gcloud projects create vidhi-ai-<something-unique>
  gcloud config set project vidhi-ai-<something-unique>

INSTALL
  exit 1
fi

if [[ -z "$PROJECT" || "$PROJECT" == "(unset)" ]]; then
  echo "No project set. Run: gcloud config set project <id>   (or PROJECT=<id> ./deploy.sh)" >&2
  exit 1
fi

# A personal project is almost certainly wanted here. Deploying a side project
# into an employer's org is easy to do by accident and awkward to undo, so the
# target is always shown and confirmed rather than assumed.
ACCOUNT="$(gcloud config get-value account 2>/dev/null)"
echo "Project: $PROJECT"
echo "Account: $ACCOUNT"
echo "Region:  $REGION"
read -r -p "Deploy to this project? [y/N] " reply
[[ "$reply" == [yY] ]] || { echo "Aborted."; exit 1; }

# Only credentials. Deliberately excluded:
#
#   RISK_CLASSIFIER_MODEL — points at a local path, and .dockerignore keeps the
#   268MB weights out of the image. Setting it would make every audit attempt a
#   load that cannot succeed; unset, the agent uses the LLM path, which is also
#   the baseline the classifier is measured against.
#
#   EMBEDDING_BACKEND — the image bakes the model in, so local is correct and is
#   already the default.
WANTED=(
  DATABASE_URL UPSTASH_REDIS_URL UPSTASH_REDIS_TOKEN
  GROQ_API_KEY OPENROUTER_API_KEY CEREBRAS_API_KEY HF_API_TOKEN
)

if [[ "${1:-}" != "--update" ]]; then
  if [[ ! -f backend/.env ]]; then
    echo "backend/.env not found — it holds the credentials to push." >&2
    exit 1
  fi

  echo
  echo "Enabling APIs (once per project, ~1 min)"
  gcloud services enable \
    run.googleapis.com \
    cloudbuild.googleapis.com \
    artifactregistry.googleapis.com \
    secretmanager.googleapis.com \
    --project "$PROJECT"

  # Secret Manager rather than plain env vars: --set-env-vars puts values in the
  # service's revision spec, where anyone with viewer access can read them.
  echo
  echo "Storing credentials in Secret Manager"
  while IFS='=' read -r key value; do
    [[ -z "${key// }" || "$key" == \#* ]] && continue
    value="${value%\"}"; value="${value#\"}"
    for wanted in "${WANTED[@]}"; do
      if [[ "$key" == "$wanted" && -n "$value" ]]; then
        if gcloud secrets describe "$key" --project "$PROJECT" >/dev/null 2>&1; then
          printf '%s' "$value" | gcloud secrets versions add "$key" \
            --data-file=- --project "$PROJECT" >/dev/null
        else
          printf '%s' "$value" | gcloud secrets create "$key" \
            --data-file=- --replication-policy=automatic --project "$PROJECT" >/dev/null
        fi
        echo "  $key"
      fi
    done
  done < backend/.env

  # The runtime service account reads those secrets at boot.
  NUMBER="$(gcloud projects describe "$PROJECT" --format='value(projectNumber)')"
  gcloud projects add-iam-policy-binding "$PROJECT" \
    --member="serviceAccount:${NUMBER}-compute@developer.gserviceaccount.com" \
    --role=roles/secretmanager.secretAccessor >/dev/null
fi

# Which secrets actually exist — a missing optional key (Cerebras, say) must not
# fail the deploy by being referenced.
present=()
for key in "${WANTED[@]}"; do
  if gcloud secrets describe "$key" --project "$PROJECT" >/dev/null 2>&1; then
    present+=("$key=$key:latest")
  fi
done
SECRETS="$(IFS=,; echo "${present[*]}")"

echo
echo "Building and deploying — the first build takes 10-15 minutes"

# --cpu-boost: the model load dominates startup, and boosted CPU during that
#   window cuts it materially.
# --min-instances=0: scale to zero so an idle service costs nothing. A cold
#   start pays the model load; image streaming keeps the pull itself cheap.
# --timeout=900: an audit runs for minutes, well past the 300s default.
# --memory=4Gi: BGE-M3 is ~2.2GB resident, plus the request working set.
gcloud run deploy "$SERVICE" \
  --source . \
  --project "$PROJECT" \
  --region "$REGION" \
  --platform managed \
  --allow-unauthenticated \
  --memory 4Gi \
  --cpu 2 \
  --cpu-boost \
  --min-instances 0 \
  --max-instances 3 \
  --concurrency 8 \
  --timeout 900 \
  --port 8080 \
  --set-secrets "$SECRETS" \
  --set-env-vars "ENVIRONMENT=production"

URL="$(gcloud run services describe "$SERVICE" --project "$PROJECT" \
        --region "$REGION" --format='value(status.url)')"

echo
echo "Checking $URL"
curl -fsS --max-time 180 "$URL/health" >/dev/null && echo "  /health ok"
curl -fsS --max-time 180 "$URL/ready"  >/dev/null && echo "  /ready  ok"

cat <<DONE

Backend live at $URL

Point the frontend at it on Vercel:
  API_ORIGIN = $URL

Not NEXT_PUBLIC_ — the proxy runs server-side, so the backend URL never
reaches the browser.
DONE
