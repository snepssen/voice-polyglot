"""How much of each script did its Suno take actually say?

Aligns the script's words to Parakeet's words (global alignment, fuzzy word
match so an ASR misspelling still counts) and reports, per take:
  - the share of script words heard
  - lines missing or only partly read
  - stretches of speech that match no script line (improvised, repeated,
    or sung filler) with their times, so they can be cut

    python line_coverage.py SCRIPT_DIR ASR_DIR [OUT.json] [LANG]

LANG defaults to SCRIPT_DIR's folder name (corpus/pl -> pl).

SCRIPT_DIR holds batch-NN.suno.txt; ASR_DIR holds <take>.json from
transcribe_takes.py, matched to scripts by the batch number in the name.
"""

import difflib
import json
import re
import sys
import unicodedata
from pathlib import Path

TAG = re.compile(r"^\[[^\]]*\]\s*$")


def fold(w: str) -> str:
    w = unicodedata.normalize("NFKD", w.lower())
    w = "".join(c for c in w if not unicodedata.combining(c))
    return re.sub(r"[^\w]", "", w.replace("ł", "l"))


def spell_digits(text: str, lang: str) -> str:
    """Parakeet writes numbers as digits; the scripts spell them out.

    Hungarian glues case endings onto numbers ("1938ban") and writes its
    numbers as one word, so there the spelling joins without spaces.
    """
    from num2words import num2words
    join = "" if lang == "hu" else " "
    return re.sub(r"\d+", lambda m: join + num2words(int(m.group()), lang=lang) + join, text)


def batch_of(asr: Path) -> int:
    """The script batch a take reads: the number in its name, unless
    ASR_DIR/../batch_map.json says otherwise (takes saved under the wrong
    name stay untouched on the Archive; the map records what they are)."""
    m = asr.parent.parent / "batch_map.json"
    if m.exists():
        mapping = json.loads(m.read_text())
        if asr.stem in mapping:
            return int(mapping[asr.stem])
    return int(re.search(r"(\d+)", asr.stem).group(1))


def script_lines(path: Path) -> list[str]:
    return [l.strip() for l in path.read_text().splitlines() if l.strip() and not TAG.match(l)]


def similar(a: str, b: str) -> bool:
    return a == b or (len(a) > 2 and difflib.SequenceMatcher(None, a, b).ratio() >= 0.75)


def align(a: list[str], b: list[str]) -> list[tuple[int | None, int | None]]:
    """Needleman-Wunsch: match +2, mismatch -1, gap -1."""
    n, m = len(a), len(b)
    score = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        score[i][0] = -i
    for j in range(1, m + 1):
        score[0][j] = -j
    sim = [[similar(a[i], b[j]) for j in range(m)] for i in range(n)]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            score[i][j] = max(score[i - 1][j - 1] + (2 if sim[i - 1][j - 1] else -1),
                              score[i - 1][j] - 1, score[i][j - 1] - 1)
    path, i, j = [], n, m
    while i or j:
        if i and j and score[i][j] == score[i - 1][j - 1] + (2 if sim[i - 1][j - 1] else -1):
            path.append((i - 1, j - 1) if sim[i - 1][j - 1] else (i - 1, None))
            if not sim[i - 1][j - 1]:
                path.append((None, j - 1))
            i, j = i - 1, j - 1
        elif i and score[i][j] == score[i - 1][j] - 1:
            path.append((i - 1, None)); i -= 1
        else:
            path.append((None, j - 1)); j -= 1
    return path[::-1]


def check(script: Path, asr: Path, lang: str | None = None) -> dict:
    lines = script_lines(script)
    owner, swords = [], []
    for k, line in enumerate(lines):
        for w in line.split():
            if fold(w):
                swords.append(fold(w)); owner.append(k)
    data = json.loads(asr.read_text())
    # Parakeet times subword pieces; a piece starting with a space starts a word.
    merged: list[dict] = []
    for t in data["words"]:
        if merged and not t["text"].startswith(" "):
            merged[-1]["text"] += t["text"]
            merged[-1]["end"] = t["end"]
        else:
            merged.append(dict(t))
    if lang:
        merged = [{"text": " " + part, "start": w["start"], "end": w["end"]}
                  for w in merged for part in spell_digits(w["text"], lang).split()]
    words = [w for w in merged if fold(w["text"])]
    awords = [fold(w["text"]) for w in words]

    heard = [False] * len(swords)
    asr_used = [False] * len(awords)
    line_times: dict[int, list[float]] = {}
    for i, j in align(swords, awords):
        if i is not None and j is not None:
            heard[i] = asr_used[j] = True
            t = line_times.setdefault(owner[i], [words[j]["start"], words[j]["end"]])
            t[0], t[1] = min(t[0], words[j]["start"]), max(t[1], words[j]["end"])

    per_line = []
    for k, line in enumerate(lines):
        idx = [i for i, o in enumerate(owner) if o == k]
        got = sum(heard[i] for i in idx)
        per_line.append({"line": k + 1, "text": line, "heard": got, "words": len(idx),
                         "time": line_times.get(k)})

    extras, run = [], []
    for j, used in enumerate(asr_used + [True]):
        if not used:
            run.append(j)
        elif run:
            if len(run) >= 3:
                extras.append({"start": words[run[0]]["start"], "end": words[run[-1]]["end"],
                               "text": " ".join(words[x]["text"].strip() for x in run)})
            run = []
    return {"take": asr.stem, "script_words": len(swords), "asr_words": len(awords),
            "heard": sum(heard), "lines": per_line, "extras": extras}


def main(script_dir: str, asr_dir: str, out: str | None = None, lang: str | None = None) -> None:
    lang = lang or Path(script_dir).name
    reports = []
    for asr in sorted(Path(asr_dir).glob("*.json")):
        r = check(Path(script_dir) / f"batch-{batch_of(asr):02d}.suno.txt", asr, lang)
        reports.append(r)
        missing = [l for l in r["lines"] if l["heard"] / l["words"] < 0.5]
        partial = [l for l in r["lines"] if 0.5 <= l["heard"] / l["words"] < 0.8]
        print(f"\n{r['take']}: {r['heard']}/{r['script_words']} script words heard "
              f"({r['heard'] / r['script_words']:.0%}); {len(r['lines'])} lines, "
              f"{len(missing)} missing, {len(partial)} partial; {len(r['extras'])} unscripted stretches")
        for l in missing:
            print(f"   missing  L{l['line']:<3} {l['text']}")
        for l in partial:
            print(f"   partial  L{l['line']:<3} {l['heard']}/{l['words']}  {l['text']}")
        for e in r["extras"]:
            print(f"   extra    {e['start']:6.1f}-{e['end']:6.1f}s  {e['text'][:110]}")
    if out:
        Path(out).write_text(json.dumps(reports, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--json"]
    main(*args)
