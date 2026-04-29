#!/usr/bin/env bash
# Bring up the full investigation platform locally.
#   ./start.sh
# Stops everything on Ctrl-C.

set -euo pipefail
cd "$(dirname "$0")"

# 1) Infra
docker compose up -d
until docker exec invplatform-postgres pg_isready -U invplatform >/dev/null 2>&1; do sleep 1; done

# 2) Backend deps + migrations
cd backend
[[ -f .env ]] || cp .env.example .env
uv sync --group ml --quiet
uv run alembic upgrade head

# 2b) Export CUDA + allocator settings into shell env so they apply BEFORE
# Python imports torch (pydantic-settings runs too late for these).
set -a
# shellcheck disable=SC1091
source <(grep -E '^(CUDA_VISIBLE_DEVICES|PYTORCH_CUDA_ALLOC_CONF)=' .env || true)
set +a

# 3) Kill any stale Celery workers from previous runs so they release GPU memory
# before we launch new ones. Harmless if nothing is running.
echo "Cleaning up stale workers…"
pkill -f "celery.*invplatform" 2>/dev/null || true
pkill -f "celery.*celery_app" 2>/dev/null || true
sleep 1   # give GPU a moment to release allocations

celery_pid=""
api_pid=""
front_pid=""
cleanup() {
  echo
  echo "stopping…"
  [[ -n "$celery_pid" ]] && kill $celery_pid 2>/dev/null || true
  [[ -n "$api_pid" ]] && kill $api_pid 2>/dev/null || true
  [[ -n "$front_pid" ]] && kill $front_pid 2>/dev/null || true
  wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# Concurrency 1 — every worker fork loads the full model stack (~15 GB)
# so concurrency=2 would request ~30 GB even before any other GPU tenant.
# Bump back to 2 once we have the GPU to ourselves (or split queues onto
# dedicated workers so models aren't double-loaded).
uv run celery -A app.worker.celery_app:celery_app worker --loglevel=info --concurrency=1 \
  -Q cpu,cpu_heavy,gpu_light,gpu_detect,gpu_embed,gpu_asr,gpu_vlm &
celery_pid=$!

uv run uvicorn app.main:app --host 0.0.0.0 --port 8000 --log-level info &
api_pid=$!

# 4) Frontend
cd ../frontend
[[ -d node_modules ]] || npm install --no-fund --no-audit
npm run dev -- --host 0.0.0.0 &
front_pid=$!

echo
echo "Frontend: http://localhost:5173"
echo "API:      http://localhost:8000  (docs /docs)"
echo "Postgres: localhost:5433  Redis: localhost:6380"
echo "Press Ctrl-C to stop everything."
wait
