"""Count clips that end with the next utterance's first sound.

    python tail_check.py CLIPS_DIR [CLIPS_DIR ...]

The fault that taught the old voice to add a fragment after short sentences:
a clip's speech ends, a short gap follows, and then a brief burst of sound
sits at the very end -- the onset of whatever was said next. Found here as a
loud run shorter than MAX_BURST seconds, preceded by at least MIN_GAP seconds
of silence, ending within END_WINDOW of the clip's end.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from voicing import read  # noqa: E402

HOP, SILENT_DB = 0.01, -50.0
MIN_GAP, MAX_BURST, END_WINDOW = 0.12, 0.40, 0.05


def has_tail_fragment(path: Path) -> bool:
    x, rate = read(path)
    n = int(HOP * rate)
    env = 20 * np.log10(np.sqrt((x[: len(x) // n * n].reshape(-1, n) ** 2).mean(1)) + 1e-9)
    loud = env >= SILENT_DB
    idx = np.where(loud)[0]
    if not len(idx) or (len(env) - 1 - idx[-1]) * HOP > END_WINDOW:
        return False
    end = idx[-1]
    start = end
    while start > 0 and loud[start - 1]:
        start -= 1
    gap = start
    while gap > 0 and not loud[gap - 1]:
        gap -= 1
    return (end - start + 1) * HOP <= MAX_BURST and (start - gap) * HOP >= MIN_GAP and gap > 0


def main(dirs: list[str]) -> None:
    for d in dirs:
        clips = sorted(Path(d).glob("*.wav"))
        hits = [p.stem for p in clips if has_tail_fragment(p)]
        print(f"{d}: {len(hits)} of {len(clips)} clips end in a fragment" + (f", e.g. {', '.join(hits[:6])}" if hits else ""))


if __name__ == "__main__":
    main(sys.argv[1:])
