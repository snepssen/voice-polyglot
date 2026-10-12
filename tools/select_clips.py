"""Choose the clips that go into training, and say why the rest don't.

    python select_clips.py LANG DATASET_DIR [--trust-wording] [--drop ID,ID,...]

Writes DATASET_DIR/train.csv (id|text) and prints what was left out, by
reason. A clip is left out when:
  edge      it starts or ends in sound (a cut with nowhere quiet to go)
  length    its speech is far longer or shorter than its text AND Parakeet
            heard a different number of words -- a swallowed neighbour or a
            lost half (a line Suno merely read slowly keeps its word count)
  check     it failed the transcript check; with --trust-wording this is
            waived, for a language whose takes a native speaker has already
            heard through and whose recogniser is good enough that failures
            are misspellings (Polish)
  neighbour it sits beside a script line that could not be cut, which is
            where audio leaks between clips (Hungarian batch 1's garden)
  dropped   a native listener dropped it
"""

import csv
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from line_coverage import fold, spell_digits  # noqa: E402
from cut_clips import KEEP_SCORE  # noqa: E402


def main(lang: str, dataset: str, trust_wording: bool, dropped: set[str]) -> None:
    d = Path(dataset)
    rows = list(csv.DictReader(open(d / "manifest.tsv", encoding="utf-8"), delimiter="\t"))
    used = {(r["take"], int(n)) for r in rows for n in r["lines"].split("+")}
    reasons: dict[str, str] = {}
    for r in rows:
        tw = len([w for w in r["text"].split() if fold(w)])
        hw = len([w for w in spell_digits(r["heard"], lang).split() if fold(w)])
        lines = [int(n) for n in r["lines"].split("+")]
        beside_gap = any((r["take"], n) not in used and n > 0
                         for n in (lines[0] - 1, lines[-1] + 1)
                         if n <= max(k for t, k in used if t == r["take"]))
        if r["id"] in dropped:
            reasons[r["id"]] = "dropped"
        elif float(r["edge_db"]) >= -40:
            reasons[r["id"]] = "edge"
        elif not 0.6 <= float(r["pace"]) <= 1.5 and not (tw - 2 <= hw <= tw + 1):
            reasons[r["id"]] = "length"
        elif float(r["score"]) < KEEP_SCORE[lang] and not trust_wording:
            reasons[r["id"]] = "check"
        elif beside_gap and not trust_wording:
            reasons[r["id"]] = "neighbour"
    chosen = [r for r in rows if r["id"] not in reasons]
    with open(d / "train.csv", "w", encoding="utf-8") as f:
        for r in chosen:
            f.write(f"{r['id']}|{r['text']}\n")
    minutes = sum(float(r["seconds"]) for r in chosen) / 60
    print(f"{lang}: {len(chosen)} of {len(rows)} clips, {minutes:.1f} min -> {d / 'train.csv'}")
    for why, n in Counter(reasons.values()).most_common():
        print(f"  left out, {why}: {n}")


if __name__ == "__main__":
    args = sys.argv[1:]
    drop = set()
    if "--drop" in args:
        drop = set(args[args.index("--drop") + 1].split(","))
    main(args[0], args[1], "--trust-wording" in args, drop)
