"""How much of each clip is voiced, rather than whispered or breathed.

    python voicing.py CLIPS_DIR [MANIFEST.tsv] [--baseline CLIPS_DIR]

For each 40 ms frame loud enough to be speech (above -45 dB), the normalised
autocorrelation peak between 60 and 400 Hz says whether the vocal folds are
vibrating. A whisper has the loudness of speech and none of the periodicity.
The share of speech frames that are voiced is the clip's voicing; the English
corpus the voice was trained on is the reference for what normal is.

With a manifest, writes a `voicing` column into it and prints the clips below
the bar, in take order, so a take that drifts into whispering shows as a run.
"""

import csv
import sys
import wave
from pathlib import Path

import numpy as np

FRAME, HOP, LOUD = 0.04, 0.02, -45.0


def read(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path)) as w:
        rate, ch, width = w.getframerate(), w.getnchannels(), w.getsampwidth()
        raw = w.readframes(w.getnframes())
    if width == 3:
        b = np.frombuffer(raw, np.uint8).reshape(-1, 3).astype(np.int32)
        x = b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)
        x = np.where(x >= 1 << 23, x - (1 << 24), x) / float(1 << 23)
    else:
        x = np.frombuffer(raw, np.int16) / 32768.0
    return x.reshape(-1, ch).mean(1), rate


def voicing(path: Path) -> float:
    x, rate = read(path)
    n, h = int(FRAME * rate), int(HOP * rate)
    lo, hi = int(rate / 400), int(rate / 60)
    speech = voiced = 0
    for i in range(0, len(x) - n, h):
        f = x[i:i + n]
        if 20 * np.log10(np.sqrt(np.mean(f ** 2)) + 1e-9) < LOUD:
            continue
        speech += 1
        f = f - f.mean()
        spec = np.fft.rfft(f, 2 * n)
        ac = np.fft.irfft(np.abs(spec) ** 2)[:n]
        ac = ac / (n - np.arange(n)) * n  # unbiased: long lags overlap fewer samples
        if ac[0] > 0 and ac[lo:hi].max() / ac[0] > 0.5:
            voiced += 1
    return voiced / speech if speech else 0.0


def main(clips: str, manifest: str | None = None, baseline: str | None = None) -> None:
    if baseline:
        ref = [voicing(p) for p in sorted(Path(baseline).glob("*.wav"))[::3]]
        print(f"baseline {baseline}: {len(ref)} clips, voicing median {np.median(ref):.2f}, "
              f"5th pct {np.percentile(ref, 5):.2f}, 1st pct {np.percentile(ref, 1):.2f}")
    if not manifest:
        v = [voicing(p) for p in sorted(Path(clips).glob("*.wav"))]
        print(f"{clips}: median {np.median(v):.2f}, 5th pct {np.percentile(v, 5):.2f}")
        return
    rows = list(csv.DictReader(open(manifest, encoding="utf-8"), delimiter="\t"))
    for r in rows:
        r["voicing"] = round(voicing(Path(clips) / f"{r['id']}.wav"), 2)
    with open(manifest, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter="\t")
        w.writeheader(); w.writerows(rows)
    v = np.array([r["voicing"] for r in rows])
    print(f"{manifest}: {len(rows)} clips, voicing median {np.median(v):.2f}, 5th pct {np.percentile(v, 5):.2f}")
    take = None
    for r in rows:
        if take != r["take"]:
            take = r["take"]
            tv = [x["voicing"] for x in rows if x["take"] == take]
            thirds = np.array_split(np.array(tv), 3)
            print(f"  {take}: voicing by third of the take {' / '.join(f'{np.median(t):.2f}' for t in thirds)}")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--baseline"]
    base = None
    if "--baseline" in sys.argv:
        base = sys.argv[sys.argv.index("--baseline") + 1]
        args.remove(base)
    main(*args, baseline=base)
