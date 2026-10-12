"""Score a check-in folder: Parakeet transcribes each sentence, and the share
of the sentence's characters it got right is printed per language.

    python checkin_score.py CHECKIN_DIR [CHECKIN_DIR ...]

A trend, not a verdict -- in the first test Parakeet scored foreign IPA far
more kindly than a native listener did. Use it to see which languages move
between epochs, then listen to those.
"""
import difflib, sys, tempfile, wave
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from checkin import SENTENCES  # noqa: E402
from line_coverage import fold, spell_digits  # noqa: E402


def norm(t, lang):
    return " ".join(fold(w) for w in spell_digits(t, lang if lang != "en" else "en").split() if fold(w))


def main(dirs):
    from parakeet_mlx import from_pretrained
    from huggingface_hub import snapshot_download
    model = from_pretrained(snapshot_download("mlx-community/parakeet-tdt-0.6b-v3", local_files_only=True))
    for d in dirs:
        row = []
        for key, text in SENTENCES.items():
            lang = key.split("-")[0]
            heard = model.transcribe(str(Path(d) / f"{key}.wav")).text
            a, b = norm(text, lang), norm(heard, lang)
            row.append(f"{key} {difflib.SequenceMatcher(None, a, b).ratio():.0%}")
        print(f"{Path(d).name:24} " + "  ".join(row), flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
