#!/usr/bin/env bash
# One-shot Supabase setup from the terminal.
#
#   scripts/supabase_setup.sh          create the project, then set it up
#   scripts/supabase_setup.sh <ref>    set up an existing project
#
# Needs: the Supabase CLI logged in (`supabase login`) and psql (`brew install libpq`).
# Writes .streamlit/secrets.toml with the PUBLISHABLE key only. The DB password is
# generated here and kept in ~/.supabase/<name>.dbpass (mode 600), never in the repo.
# Every step is idempotent: re-running on an existing project changes nothing.
set -euo pipefail
cd "$(dirname "$0")/.."

ORG_ID="${SUPABASE_ORG_ID:-REDACTED-ORG-ID}"   # Bonim Bina
REGION="${SUPABASE_REGION:-eu-central-1}"
NAME="${SUPABASE_PROJECT_NAME:?SUPABASE_PROJECT_NAME is required}"
CSV="${LADDER_CSV:?LADDER_CSV is required}"
TABLE="${LADDER_TABLE:-records}"
PSQL="${PSQL:-/opt/homebrew/opt/libpq/bin/psql}"
PWFILE="$HOME/.supabase/$NAME.dbpass"
py() { python3 -c "$1"; }

REF="${1:-}"
if [ -z "$REF" ]; then
  mkdir -p "$HOME/.supabase"
  PW="$(openssl rand -base64 36 | tr -d '/+=' | cut -c1-32)"
  (umask 077; printf '%s' "$PW" > "$PWFILE")
  OUT="$(supabase projects create "$NAME" --org-id "$ORG_ID" --region "$REGION" \
          --db-password "$PW" --yes --output-format json)"
  REF="$(echo "$OUT" | py '
import sys, json
d = json.load(sys.stdin)
d = d.get("project", d)
print(d.get("ref") or d.get("id"))')"
  echo "created project $NAME  ref=$REF  (db password saved to $PWFILE)"
else
  PW="$(cat "$PWFILE")"
fi

echo "waiting for the database to come up..."
STATUS=""
for i in $(seq 1 60); do
  STATUS="$(supabase projects list --output-format json 2>/dev/null | py "
import sys, json
ps = json.load(sys.stdin)['projects']
print(next((p['status'] for p in ps if p['ref'] == '$REF'), 'UNKNOWN'))")"
  [ "$STATUS" = "ACTIVE_HEALTHY" ] && break
  echo "  $STATUS ($i)"; sleep 10
done
[ "$STATUS" = "ACTIVE_HEALTHY" ] || { echo "project is $STATUS, giving up"; exit 1; }

# Session pooler, not the direct host: the direct host is IPv6-only and this Mac
# has no IPv6 route. Newer projects sit on aws-1, older on aws-0; try both, and
# give a freshly created database a few tries to start accepting connections.
export PGPASSWORD="$PW"
DB=""
for attempt in $(seq 1 8); do
  for H in "aws-1-$REGION.pooler.supabase.com" "aws-0-$REGION.pooler.supabase.com"; do
    CAND="host=$H port=5432 user=postgres.$REF dbname=postgres sslmode=require"
    if "$PSQL" "$CAND" -tA -c 'select 1' >/dev/null 2>&1; then DB="$CAND"; break 2; fi
  done
  echo "  pooler not answering yet ($attempt)"; sleep 10
done
[ -n "$DB" ] || { echo "could not reach the session pooler for $REGION"; exit 1; }
echo "connected via ${DB%% *}"

"$PSQL" "$DB" -v ON_ERROR_STOP=1 -q -f supabase/schema.sql
echo "schema applied"
if [ "$("$PSQL" "$DB" -tA -c "select count(*) from public.$TABLE")" = "0" ]; then
  "$PSQL" "$DB" -v ON_ERROR_STOP=1 -q \
    -c "\copy public.$TABLE from '$CSV' with (format csv, header true)"
fi
echo "project ref:  $REF"
echo "table rows:   $("$PSQL" "$DB" -tA -c "select count(*) from public.$TABLE")"

# The publishable key is public by design (it ships inside client apps). Find it
# wherever the CLI's JSON puts it, and never touch the sb_secret_ one.
KEY="$(supabase projects api-keys --project-ref "$REF" --output-format json | py '
import sys, json
def walk(x):
    if isinstance(x, dict):
        for v in x.values(): yield from walk(v)
    elif isinstance(x, list):
        for v in x: yield from walk(v)
    elif isinstance(x, str):
        yield x
print(next(s for s in walk(json.load(sys.stdin)) if s.startswith("sb_publishable_")))')"
mkdir -p .streamlit
cat > .streamlit/secrets.toml <<TOML
[connections.supabase]
SUPABASE_URL = "https://$REF.supabase.co"
SUPABASE_KEY = "$KEY"
TOML
echo "wrote .streamlit/secrets.toml (publishable key only)"
