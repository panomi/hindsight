#!/usr/bin/env bash
# Kill all active Celery ingest jobs and reset their DB status to 'pending'.
# Usage:
#   ./scripts/kill_jobs.sh          — kill everything in the queue
#   ./scripts/kill_jobs.sh <video_id> — kill just one video

set -euo pipefail
cd "$(dirname "$0")/.."

VIDEO_ID="${1:-}"

echo "=== Active tasks ==="
.venv/bin/celery -A app.worker.celery_app:celery_app inspect active 2>/dev/null \
  | grep -E "id|args|worker_pid" || true

echo ""
echo "=== Purging queue ==="
.venv/bin/celery -A app.worker.celery_app:celery_app purge -f 2>/dev/null || true

echo ""
echo "=== Killing worker child processes ==="
# Kill child worker processes (prefork pool children) — main worker restarts them
PIDS=$(ps aux | grep "celery.*worker" | grep -v grep | awk '{print $2}' | tail -n +2 || true)
if [[ -n "$PIDS" ]]; then
    # Only kill the child forks (they have different ppid from the main worker)
    MAIN_PID=$(pgrep -f "celery -A app.worker" | head -1 || true)
    CHILDREN=$(ps --ppid "$MAIN_PID" -o pid= 2>/dev/null || true)
    if [[ -n "$CHILDREN" ]]; then
        echo "Killing worker children: $CHILDREN"
        kill -9 $CHILDREN 2>/dev/null || true
    fi
fi

echo ""
echo "=== Resetting video status in DB ==="
if [[ -n "$VIDEO_ID" ]]; then
    .venv/bin/python -c "
import asyncio
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy import text

async def reset():
    engine = create_async_engine('postgresql+asyncpg://invplatform:invplatform@localhost:5434/invplatform')
    async with AsyncSession(engine) as s:
        await s.execute(text(\"UPDATE videos SET status='pending', stage=NULL, progress_pct=0 WHERE id='$VIDEO_ID'\"))
        await s.commit()
        print('Reset video $VIDEO_ID to pending')
    await engine.dispose()
asyncio.run(reset())
"
else
    .venv/bin/python -c "
import asyncio
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy import text

async def reset():
    engine = create_async_engine('postgresql+asyncpg://invplatform:invplatform@localhost:5434/invplatform')
    async with AsyncSession(engine) as s:
        r = await s.execute(text(\"UPDATE videos SET status='pending', stage=NULL, progress_pct=0 WHERE status='ingesting' RETURNING id, filename\"))
        rows = r.fetchall()
        await s.commit()
        if rows:
            for row in rows:
                print(f'Reset {row[1]} ({row[0]}) → pending')
        else:
            print('No ingesting videos found')
    await engine.dispose()
asyncio.run(reset())
"
fi

echo ""
echo "Done. Workers are free. Use the UI or requeue to restart ingestion."
