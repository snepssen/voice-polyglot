#!/usr/bin/env bash
# Render and score a check-in for every new checkpoint of a run, into
# work/runs/checkin/<epoch>/ and work/runs/checkin/scores.txt. Prints an
# ALERT line when English falls under 95% or any language drops more than
# 10 points below its best so far -- the only lines worth interrupting for.
# The one-word "Breathe in." is scored but never alerts: Parakeet hears it
# as "Breathing." on and off, while the same words in the full sentence pass.
#
#   tools/checkin_watch.sh [RUN_DIR]
set -uo pipefail
cd "$(dirname "$0")/.."
RUN="${1:-work/runs/polyglot}"
OUT=work/runs/checkin
mkdir -p "$OUT"; touch "$OUT/scores.txt"
while pgrep -f "tools/train.py .* $RUN" >/dev/null; do
  for c in $(find "$RUN" -name 'epoch=*.ckpt' | sort -t= -k2 -n); do
    e=$(basename "$c" | sed -E 's/epoch=([0-9]+).*/\1/'); d="$OUT/epoch-$(printf %03d "$e")"
    [ -d "$d" ] && continue
    sleep 20  # let the checkpoint finish writing
    .venv/bin/python -I tools/checkin.py "$c" "$d" >/dev/null 2>&1 || { echo "ALERT check-in failed for $c"; continue; }
    line=$(.venv/bin/python -I tools/checkin_score.py "$d" 2>/dev/null | grep "^epoch-")
    echo "$line" >> "$OUT/scores.txt"
    python3 - "$OUT/scores.txt" <<'PY'
import re, sys
rows = [dict(re.findall(r"(\S+) (\d+)%", l)) for l in open(sys.argv[1]) if l.strip()]
*prev, last = rows
for k, v in last.items():
    if k == "en-short":
        continue
    v = int(v); best = max([int(r[k]) for r in prev if k in r] or [v])
    if (k == "en" and v < 95) or v < best - 10:
        print(f"ALERT {k} {v}% (best so far {best}%)", flush=True)
PY
  done
  sleep 60
done
echo "run ended"
