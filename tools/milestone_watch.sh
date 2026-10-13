#!/usr/bin/env bash
# Every EVERY epochs of a run: export the voice, render the check-in
# sentences from it, score them, flag regressions. Nothing in between --
# a long run is left to take its time.
#
#   tools/milestone_watch.sh [RUN_DIR]
#
# Exports land in RUN_DIR/exports/epoch-NNN/ (ONNX + config, verified against
# the checkpoint), renders in work/runs/checkin-v2/epoch-NNN/, scores in
# work/runs/checkin-v2/scores.txt. An ALERT line is printed when English
# falls under 95%, or a language sits more than 10 points under its best in
# two milestones running; the one-word English sentence never alerts.
set -uo pipefail
cd "$(dirname "$0")/.."
RUN="${1:-work/runs/polyglot}"
EVERY="${EVERY:-25}"
OUT="${OUT:-work/runs/checkin-v2}"
TMP="$RUN/.milestone.ckpt"
mkdir -p "$OUT" "$RUN/exports"; touch "$OUT/scores.txt"
seen=""
while pgrep -f "tools/train.py .* $RUN" >/dev/null; do
  last=$(ls -t "$RUN"/lightning_logs/version_*/checkpoints/last.ckpt 2>/dev/null | head -1)
  stamp=$( [ -n "$last" ] && stat -f %m "$last" )
  if [ -n "$last" ] && [ "$stamp" != "$seen" ]; then
    sleep 30; seen=$(stat -f %m "$last")      # let the save finish
    cp "$last" "$TMP"
    e=$(.venv/bin/python -I -c "import sys; sys.path.insert(0,'tools'); import piper_fixes, torch; print(torch.load('$TMP', map_location='cpu', weights_only=False)['epoch'])" 2>/dev/null)
    n=$(printf %03d "${e:-0}")
    if [ -n "$e" ] && [ $((e % EVERY)) -eq 0 ] && [ ! -d "$RUN/exports/epoch-$n" ]; then
      .venv/bin/python -I tools/export_voice.py "$TMP" "$RUN/exports/epoch-$n" 2>/dev/null | tail -1 || echo "ALERT export failed at epoch $e"
      .venv/bin/python -I tools/checkin.py "$TMP" "$OUT/epoch-$n" >/dev/null 2>&1 || echo "ALERT check-in failed at epoch $e"
      line=$(.venv/bin/python -I tools/checkin_score.py "$OUT/epoch-$n" 2>/dev/null | grep "^epoch-")
      echo "$line" >> "$OUT/scores.txt"; echo "milestone $line"
      python3 - "$OUT/scores.txt" <<'PY'
import re, sys
rows = [dict(re.findall(r"(\S+) (\d+)%", l)) for l in open(sys.argv[1]) if l.strip()]
*prev, last = rows
for k, v in last.items():
    if k == "en-short":
        continue
    v = int(v); best = max([int(r[k]) for r in prev if k in r] or [v])
    before = int(prev[-1][k]) if prev and k in prev[-1] else v
    best_before = max([int(r[k]) for r in prev[:-1] if k in r] or [before])
    if (k == "en" and v < 95) or (v < best - 10 and before < best_before - 10):
        print(f"ALERT {k} {v}% and {before}% (best so far {best}%)", flush=True)
PY
    fi
    rm -f "$TMP"
  fi
  sleep 60
done
echo "run ended"
