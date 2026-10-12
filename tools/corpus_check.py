"""Check a Suno corpus folder before it goes to Suno.

For every batch: character count against Suno's 5,000 limit, an estimated
track length, and tags that would end up in the transcript. Across the whole
language: how often each sound occurs once espeak-ng has spelled it in IPA,
which is what the voice will be trained on.

The pace estimate is measured, not guessed: Suno read the Polish batches at
11.6 characters per second of track and the Hungarian at 9.8 (the English
corpus, in shorter blocks, at 13.0). 10.7 is the Polish/Hungarian mean.

    python corpus_check.py corpus/pl pl
"""

import collections
import re
import sys
from pathlib import Path

from piper.phonemize_espeak import ESPEAK_DATA_DIR, EspeakPhonemizer

SUNO_LIMIT = 5000
CHARS_PER_SECOND = 10.7
TAG = re.compile(r"^\[[^\]]*\]\s*$")

# Multi-symbol sounds counted as one, longest first.
UNITS = ["t͡ʂ", "d͡ʐ", "t͡ɕ", "d͡ʑ", "t͡s", "d͡z", "tʂ", "dʐ", "tʃ", "dʒ", "tɕ", "dʑ", "ts", "dz",
         "ɛ̃", "ɔ̃", "ɛŋ", "ɔŋ"]


def transcript(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip() and not TAG.match(line)]


def units(ipa: str) -> list[str]:
    ipa = re.sub("[ˈˌ,.?!;: ]", "", ipa)
    out, i = [], 0
    while i < len(ipa):
        for u in UNITS:
            if ipa.startswith(u, i):
                out.append(u)
                i += len(u)
                break
        else:
            if out and ipa[i] in "̯̩̃͡ː":
                out[-1] += ipa[i]
            else:
                out.append(ipa[i])
            i += 1
    return out


def main(folder: str, lang: str) -> None:
    ph = EspeakPhonemizer(ESPEAK_DATA_DIR)
    counts: collections.Counter = collections.Counter()
    total_chars = total_lines = 0
    for f in sorted(Path(folder).glob("batch-*.suno.txt")):
        raw = f.read_text()
        lines = transcript(raw)
        stray = [l for l in lines if "[" in l or "]" in l]
        est = len(raw) / CHARS_PER_SECOND / 60
        flag = "OVER LIMIT" if len(raw) > SUNO_LIMIT else "ok"
        print(f"{f.name}: {len(raw):5} chars ({flag}), {len(lines):3} lines, ~{est:.1f} min"
              + (f", stray brackets: {stray}" if stray else ""))
        total_chars += len(raw)
        total_lines += len(lines)
        for line in lines:
            for sent in ph.phonemize(lang, line):
                counts.update(units("".join(sent)))
    print(f"\ntotal {total_chars} chars, {total_lines} lines, ~{total_chars / CHARS_PER_SECOND / 60:.0f} min")
    print("\nsound counts, rarest first:")
    for u, n in sorted(counts.items(), key=lambda kv: kv[1]):
        print(f"  {u:5} {n}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
