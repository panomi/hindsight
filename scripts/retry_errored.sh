#!/usr/bin/env bash
# Re-queue every video currently in `error` status.
# Resets their state to `pending` and dispatches run_ingest.
set -euo pipefail

cd "$(dirname "$0")/../backend"

# Show what we're about to retry
echo "=== Errored videos ==="
docker exec invplatform-postgres psql -U invplatform -d invplatform -c \
  "SELECT filename, error FROM videos WHERE status='error';"

# Reset state
docker exec invplatform-postgres psql -U invplatform -d invplatform -c \
  "UPDATE videos SET status='pending', stage=NULL, progress_pct=0, error=NULL
   WHERE status='error' RETURNING id;" \
  | tail -n +3 | head -n -2 | tr -d ' ' \
  | while read -r vid; do
      [[ -z "$vid" ]] && continue
      echo "  re-queuing $vid"
      uv run python -c "
from app.worker.tasks.ingest import run_ingest
run_ingest.apply_async(args=['$vid'], queue='cpu')
" 2>&1 | tail -1
    done

echo ""
echo "Done."
