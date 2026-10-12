"""Transcribe Suno takes with Parakeet v3 and keep every word's timing.

The transcript is only ever used for *timing* and for checking what was
said against the script; the training text always comes from the script.

    python transcribe_takes.py TAKES_DIR OUT_DIR

Writes OUT_DIR/<take name>.json: {"words": [{"text", "start", "end"}], "text"}.
Takes already transcribed are skipped. Where a take has a Suno "(Lead Vocal)"
stem beside it, the stem is transcribed instead of the mix -- same length,
same timings, no music under the words.
"""

import json
import sys
from pathlib import Path

from parakeet_mlx import from_pretrained

MODEL = "mlx-community/parakeet-tdt-0.6b-v3"


def local_model() -> str:
    # The hub client hangs on this machine; use the snapshot already on disk.
    from huggingface_hub import snapshot_download
    return snapshot_download(MODEL, local_files_only=True)


def main(takes: str, out: str) -> None:
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    model = None
    for wav in sorted(Path(takes).glob("*.wav")):
        if wav.stem.endswith(" (Lead Vocal)"):
            continue
        stem = wav.with_name(wav.stem + " (Lead Vocal).wav")
        if stem.exists():
            wav = stem
        dest = out_dir / (wav.stem.removesuffix(" (Lead Vocal)") + ".json")
        if dest.exists():
            print("have", dest.name)
            continue
        if model is None:
            model = from_pretrained(local_model())
        result = model.transcribe(wav, chunk_duration=120.0, overlap_duration=15.0)
        words = [
            {"text": t.text, "start": round(t.start, 3), "end": round(t.end, 3)}
            for s in result.sentences for t in s.tokens
        ]
        dest.write_text(json.dumps({"text": result.text, "words": words}, ensure_ascii=False, indent=1))
        print(f"{wav.name}: {len(result.text.split())} words")


if __name__ == "__main__":
    main(*sys.argv[1:3])
