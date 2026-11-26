#!/bin/sh
# Run one pipeline job (nightly | retrain) in Docker next to the database.
# From cron on the server, e.g. (times in UTC; 08:30 UTC = 3:30 AM Ottawa):
#   30 8 * * *  /opt/ottawa-home-value/scripts/run_job.sh nightly
#   15 9 1 * *  /opt/ottawa-home-value/scripts/run_job.sh retrain
# flock keeps two jobs from overlapping. .env holds DATABASE_URL (never in git).
set -eu
DIR=$(cd "$(dirname "$0")/.." && pwd)
mkdir -p "$DIR/logs"
exec flock -w 3600 "$DIR/logs/.job.lock" docker run --rm --network listing-alerts_default \
  --env-file "$DIR/.env" -v "$DIR:/app" --cpus 1.5 --memory 2g \
  ottawa-home-value python -m src.pipeline "$1" >> "$DIR/logs/$1.log" 2>&1
