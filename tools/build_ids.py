"""Turn every language's train.csv into the phoneme ids the trainer reads.

    python build_ids.py WORK_DIR OUT_DIR [--clips SUBDIR] lang=folder [lang=folder ...]

--clips picks the clip folder inside each dataset (default clips; the
re-trimmed ones are clips-v2).

Piper's trainer accepts `utt_id|text|phoneme_ids` (its PHONEME_IDS dataset
type), which hands it ids we computed rather than letting it phonemize --
the only way to be sure training sees exactly what polyphon.phonemize, the
function the app will call, produces. Ids come from piper's own
phonemes_to_ids and the voice's symbol table, one sentence at a time and
chained, as its espeak path would.

utt_id is the clip's path under WORK_DIR without .wav, so the trainer's
audio_dir is WORK_DIR and no audio is copied. Writes OUT_DIR/<lang>.csv,
OUT_DIR/<lang>.ipa.tsv (the IPA, for reading) and OUT_DIR/all.csv.
"""

import itertools
import sys
from pathlib import Path

from piper.phoneme_ids import DEFAULT_PHONEME_ID_MAP, phonemes_to_ids

sys.path.insert(0, str(Path(__file__).parent))
import polyphon  # noqa: E402


def main(work: str, out: str, pairs: list[str], clips: str = "clips") -> None:
    work_dir, out_dir = Path(work), Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    every = []
    for pair in pairs:
        lang, folder = pair.split("=")
        rows, ipa_rows, unknown = [], [], set()
        for line in open(work_dir / folder / "dataset" / "train.csv", encoding="utf-8"):
            cid, text = line.rstrip("\n").split("|", 1)
            sentences = polyphon.phonemize(text, lang)
            unknown |= set(polyphon.check_symbols(sentences))
            ids = list(itertools.chain(*(phonemes_to_ids(s, DEFAULT_PHONEME_ID_MAP) for s in sentences)))
            utt = f"{folder}/dataset/{clips}/{cid}"
            assert (work_dir / f"{utt}.wav").exists(), utt
            rows.append(f"{utt}|{text}|{' '.join(map(str, ids))}")
            ipa_rows.append(f"{cid}\t{text}\t{' | '.join(''.join(s) for s in sentences)}")
        if unknown:
            sys.exit(f"{lang}: symbols outside the voice's table: {sorted(unknown)}")
        (out_dir / f"{lang}.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
        (out_dir / f"{lang}.ipa.tsv").write_text("\n".join(ipa_rows) + "\n", encoding="utf-8")
        every += rows
        print(f"{lang}: {len(rows)} utterances")
    (out_dir / "all.csv").write_text("\n".join(every) + "\n", encoding="utf-8")
    print(f"all: {len(every)} utterances -> {out_dir / 'all.csv'}")


if __name__ == "__main__":
    args = sys.argv[3:]
    clips = "clips"
    if "--clips" in args:
        i = args.index("--clips"); clips = args[i + 1]; del args[i:i + 2]
    main(sys.argv[1], sys.argv[2], args, clips)
