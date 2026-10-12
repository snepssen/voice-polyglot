#!/usr/bin/env bash
# Train snepssen on all eight languages, warm-started from the rebuilt
# epoch-702 voice.
#
#   tools/train-polyglot.sh new    [extra lightning args]   start the run
#   tools/train-polyglot.sh resume [extra lightning args]   continue it
#
# The data is work/train/all.csv: phoneme ids written by tools/build_ids.py
# through polyphon, so the trainer never phonemizes anything itself. Piper's
# trim_silence is off: it would cut every clip to 0.25 s of padding, undoing
# the 150 ms lead and up-to-700 ms tail TRAINING.md asks for and cut_clips
# leaves. A changed corpus needs a new cache dir -- the cache keys on row
# number and text, not audio.
set -euo pipefail
cd "$(dirname "$0")/.."
MODE="${1:?new or resume}"; shift
RUN="${RUN:-work/runs/polyglot}"
exec .venv/bin/python -I tools/train.py "$MODE" "$RUN" \
  --data.voice_name snepssen \
  --data.csv_path work/train/all.csv \
  --data.audio_dir work \
  --data.dataset_type phoneme_ids \
  --data.espeak_voice en-us \
  --data.cache_dir "${CACHE:-work/train/cache-v1}" \
  --data.config_path work/train/config.json \
  --data.batch_size "${BATCH:-8}" \
  --data.validation_split 0.02 \
  --data.num_test_examples 8 \
  --data.trim_silence false \
  --model.sample_rate 22050 \
  --model.warmstart_ckpt checkpoints/snepssen-epoch702-rebuilt.ckpt \
  --trainer.accelerator mps \
  "$@"
