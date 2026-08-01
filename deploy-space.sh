#!/usr/bin/env bash
# Push the backend to a HuggingFace Space.
#
# Spaces reads its configuration from YAML front-matter in README.md, which
# would be noise at the top of the GitHub README. So this pushes a branch whose
# README carries that front-matter, leaving main's untouched.
#
#   ./deploy-space.sh <hf-username> [space-name]

set -euo pipefail

USER="${1:?usage: ./deploy-space.sh <hf-username> [space-name]}"
SPACE="${2:-vidhi-ai-api}"
BRANCH="space-deploy"

cd "$(git rev-parse --show-toplevel)"

if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "Working tree is dirty. Commit or stash first." >&2
  exit 1
fi

echo "Preparing $BRANCH for huggingface.co/spaces/$USER/$SPACE"
git branch -D "$BRANCH" 2>/dev/null || true
git checkout -q -b "$BRANCH"

# Front-matter first, then the existing README beneath it.
{
  cat <<'FRONTMATTER'
---
title: VidhiAI API
emoji: ⚖️
colorFrom: yellow
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
---

FRONTMATTER
  cat README.md
} > README.space.md
mv README.space.md README.md

git add README.md
git commit -q -m "Space configuration"

git remote remove space 2>/dev/null || true
git remote add space "https://huggingface.co/spaces/$USER/$SPACE"

echo
echo "Pushing. HuggingFace asks for your username and an access token"
echo "(huggingface.co/settings/tokens — needs write scope), not your password."
git push --force space "$BRANCH:main"

git checkout -q main
git branch -D "$BRANCH" >/dev/null

cat <<DONE

Pushed. The first build takes 10-15 minutes.

Next, set the secrets under Settings -> Variables and secrets:
  DATABASE_URL, UPSTASH_REDIS_URL, UPSTASH_REDIS_TOKEN,
  GROQ_API_KEY, OPENROUTER_API_KEY, ENVIRONMENT=production

Then verify:
  curl https://$USER-$SPACE.hf.space/health
  curl https://$USER-$SPACE.hf.space/ready

Finally point the frontend at it on Vercel:
  API_ORIGIN = https://$USER-$SPACE.hf.space
DONE
