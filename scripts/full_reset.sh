#!/usr/bin/env bash
# Full platform reset — wipes all DB data (except collections + alembic),
# extracted frames on disk, and the Redis queue.
# Source videos in data/videos/ are NOT touched.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"

echo "=== 1. Killing platform processes ==="
# Celery workers
pkill -SIGTERM -f "celery.*worker" 2>/dev/null || true
# Uvicorn (API server) — both uv-launched and direct launches
pkill -SIGTERM -f "uvicorn app.main" 2>/dev/null || true
pkill -SIGTERM -f "uvicorn.*8000"   2>/dev/null || true
# Vite dev server
pkill -SIGTERM -f "vite"            2>/dev/null || true
sleep 3
# Force-kill anything that didn't exit gracefully
pkill -SIGKILL -f "celery.*worker"  2>/dev/null || true
pkill -SIGKILL -f "uvicorn app.main" 2>/dev/null || true
pkill -SIGKILL -f "uvicorn.*8000"   2>/dev/null || true
pkill -SIGKILL -f "vite"            2>/dev/null || true

echo "=== 1b. Verifying ports are free ==="
for port in 8000 5173 5174; do
  pid=$(lsof -ti :"$port" 2>/dev/null || true)
  if [ -n "$pid" ]; then
    echo "  Port $port still held by pid $pid — killing"
    kill -SIGKILL $pid 2>/dev/null || true
  else
    echo "  Port $port: free"
  fi
done

echo "=== 2. Flushing Redis (queue + locks) ==="
docker exec invplatform-redis redis-cli FLUSHALL

echo "=== 3. Wiping DB tables ==="
docker exec invplatform-postgres psql -U invplatform -d invplatform -c "
TRUNCATE
  agent_actions,
  messages,
  investigations,
  results,
  prompt_cache,
  transcript_segments,
  ocr_texts,
  captions,
  detections,
  subject_instances,
  subject_references,
  subjects,
  frames,
  shots,
  videos,
  collections
CASCADE;
SELECT 'DB wiped' AS status;
"

echo "=== 4. Removing extracted frames from disk ==="
rm -rf "${REPO}/data/frames"/*
rm -rf "${REPO}/data/thumbnails"/*

echo ""
echo "Done. Source videos in data/videos/ are untouched."
echo "Start the platform with ./start.sh, then re-add your collections and videos."
