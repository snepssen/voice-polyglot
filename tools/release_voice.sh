#!/usr/bin/env bash
# Publish an exported voice as a GitHub Release of this repository: the one
# place every app gets snepssen from.
#
#   tools/release_voice.sh EXPORT_DIR [--final]
#
# EXPORT_DIR is an export_voice.py folder (snepssen-polyglot-epoch-NNN.onnx and
# its .onnx.json, already verified against the checkpoint). The assets go up
# under fixed names, so
#
#   https://github.com/snepssen/voice-polyglot/releases/latest/download/snepssen-polyglot.onnx
#
# is always the newest voice, and an app that fetches it is updated by every
# release, as decided on 2026-10-14. The config's "polyglot" block is stamped
# with the epoch, the date and, until --final, "in training"; the release is
# titled the same way, with the asterisk.
set -euo pipefail
cd "$(dirname "$0")/.."
DIR="${1:?export folder}"
STATUS="in training"; [ "${2:-}" = "--final" ] && STATUS="final"
MODEL=$(ls "$DIR"/snepssen-polyglot-*.onnx | head -1)
EPOCH=$(basename "$MODEL" .onnx | sed -E 's/.*epoch-0*([0-9]+)$/\1/')
DATE=$(date +%Y-%m-%d)
TAG="voice-epoch-$(printf %03d "$EPOCH")"
OUT=$(mktemp -d)
cp "$MODEL" "$OUT/snepssen-polyglot.onnx"
python3 - "$MODEL.json" "$OUT/snepssen-polyglot.onnx.json" "$EPOCH" "$DATE" "$STATUS" "$TAG" <<'PY'
import json, sys
src, dst, epoch, date, status, tag = sys.argv[1:]
c = json.load(open(src))
c["dataset"] = "snepssen-polyglot"
c["polyglot"].update({"epoch": int(epoch), "released": date, "status": status, "release": tag})
open(dst, "w").write(json.dumps(c, ensure_ascii=False, indent=2))
PY
(cd "$OUT" && shasum -a 256 snepssen-polyglot.onnx snepssen-polyglot.onnx.json > SHA256SUMS)
MARK=$([ "$STATUS" = final ] && echo "" || echo " · *still in training")
NOTES="snepssen, the polyglot voice: English, Polish, Hungarian, German, Dutch, French, Spanish and Italian in one 64 MB Piper voice.

Epoch $EPOCH, exported $DATE$MARK.
$([ "$STATUS" = final ] || echo "
\\* Training is still running. A newer release replaces this one as the latest; apps that fetch \`releases/latest\` follow it.")

Phonemize with tools/polyphon.py (or its Swift port in translate-core), not plain espeak-ng, and speak each language at the length_scale in the config's \"polyglot\" block.

Verified against its training checkpoint (tools/verify_rebuild.py) before release."
if gh release view "$TAG" >/dev/null 2>&1; then
  gh release upload "$TAG" "$OUT"/snepssen-polyglot.onnx "$OUT"/snepssen-polyglot.onnx.json "$OUT"/SHA256SUMS --clobber
else
  gh release create "$TAG" "$OUT"/snepssen-polyglot.onnx "$OUT"/snepssen-polyglot.onnx.json "$OUT"/SHA256SUMS \
    --title "snepssen polyglot · epoch $EPOCH · $DATE$MARK" --notes "$NOTES" --latest
fi
rm -rf "$OUT"
echo "released $TAG"
