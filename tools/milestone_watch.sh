#!/usr/bin/env bash
# At every milestone of a run (train.py saves one each 25 epochs): export
# the voice, render the check-in
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
OUT="${OUT:-work/runs/checkin-v2}"
mkdir -p "$OUT" "$RUN/exports"; touch "$OUT/scores.txt"
# train.py's Milestones callback writes milestone-epoch=NNN.ckpt every 25
# epochs and never overwrites it; each one without an export gets one.
while pgrep -f "tools/train.py .* $RUN" >/dev/null; do
  for ckpt in $(ls "$RUN"/lightning_logs/version_*/checkpoints/milestone-epoch=*.ckpt 2>/dev/null | sort -t= -k2); do
    n=$(basename "$ckpt" .ckpt); n=${n#milestone-epoch=}; e=$((10#$n))
    [ -d "$RUN/exports/epoch-$n" ] && continue
    sleep 30                                  # let the save finish
    .venv/bin/python -I tools/export_voice.py "$ckpt" "$RUN/exports/epoch-$n" 2>/dev/null | tail -1 || echo "ALERT export failed at epoch $e"
    .venv/bin/python -I tools/checkin.py "$ckpt" "$OUT/epoch-$n" >/dev/null 2>&1 || echo "ALERT check-in failed at epoch $e"
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
  done
  sleep 60
done
echo "run ended"
