"""Re-trim clips to the padding the voice was first trained with.

    python retrim.py DATASET_DIR [--lead S] [--tail S] [--max-pause S]

Writes DATASET_DIR/clips-v2/ from DATASET_DIR/clips/. The first cut left
every clip with ~0.3 s before the first word and ~1.1 s after the last:
cut_clips measured "the last word" against -50 dB, and Suno's breathy reverb
decays between -50 and -40 dB, so each clip carried half a second of decay
plus the 0.7 s tail on top -- and the meditation reading leaves 0.55 s
pauses at commas. Trained on that, the voice learned half-second lead-ins,
more than a second of silence after every sentence and long comma gaps, with
the words packed tight in between: rushed speech separated by dead air, and
added latency on every line.

The targets are measured, from the old English clips that produced the voice
before this run (lead 0.07 s, tail 0.21 s, inner pauses median 0.25 s, 90th
percentile 0.57 s). A 0.45 s cap left most pauses sitting right at the cap,
still the meditation pace; 0.35 s is the default:

  lead       LEAD s of the original before the first frame above -40 dB
  tail       TAIL s after the last one, the last 30 ms faded to silence
  pauses     any inner stretch below -40 dB longer than MAX_PAUSE becomes
             MAX_PAUSE long, cut from its middle with 10 ms crossfades

Only silence is removed; nothing above -40 dB is touched.
"""

import sys
import wave
from pathlib import Path

import numpy as np

THRESH_DB, HOP = -40.0, 0.01


def write(path: Path, x: np.ndarray, rate: int) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(rate)
        w.writeframes((np.clip(x, -1, 1) * 32767).astype(np.int16).tobytes())


def retrim(x: np.ndarray, rate: int, lead: float, tail: float, max_pause: float) -> np.ndarray:
    n = int(HOP * rate)
    env = 20 * np.log10(np.sqrt((x[: len(x) // n * n].reshape(-1, n) ** 2).mean(1)) + 1e-9)
    loud = env > THRESH_DB
    idx = np.where(loud)[0]
    if not len(idx):
        return x
    start = max(0, idx[0] * n - int(lead * rate))
    end = min(len(x), (idx[-1] + 1) * n + int(tail * rate))
    # inner pauses, in samples relative to the original
    cuts, run_start = [], None
    for i in range(idx[0], idx[-1] + 1):
        if not loud[i] and run_start is None:
            run_start = i
        elif loud[i] and run_start is not None:
            length = (i - run_start) * HOP
            if length > max_pause:
                mid = (run_start + i) * n // 2
                remove = int((length - max_pause) * rate)
                cuts.append((mid - remove // 2, mid + remove - remove // 2))
            run_start = None
    pieces, pos = [], start
    for a, b in cuts:
        pieces.append(x[pos:a]); pos = b
    pieces.append(x[pos:end])
    fade = int(0.01 * rate)
    out = pieces[0]
    for p in pieces[1:]:
        if len(out) > fade and len(p) > fade:
            ramp = np.linspace(1, 0, fade)
            out = np.concatenate([out[:-fade], out[-fade:] * ramp + p[:fade] * (1 - ramp), p[fade:]])
        else:
            out = np.concatenate([out, p])
    f = min(int(0.03 * rate), len(out))
    out = out.copy()
    out[-f:] *= np.linspace(1, 0, f)
    return out


def main(dataset: str, lead: float = 0.08, tail: float = 0.25, max_pause: float = 0.35) -> None:
    d = Path(dataset)
    src, dst = d / "clips", d / "clips-v2"
    dst.mkdir(exist_ok=True)
    before = after = 0.0
    for p in sorted(src.glob("*.wav")):
        with wave.open(str(p)) as w:
            rate = w.getframerate()
            x = np.frombuffer(w.readframes(w.getnframes()), np.int16) / 32768.0
        y = retrim(x, rate, lead, tail, max_pause)
        write(dst / p.name, y, rate)
        before += len(x) / rate; after += len(y) / rate
    print(f"{d}: {before / 60:.1f} min -> {after / 60:.1f} min ({1 - after / before:.0%} silence removed)")


if __name__ == "__main__":
    args = sys.argv[1:]
    opts = {k: float(args[args.index(f"--{k.replace('_', '-')}") + 1]) for k in ("lead", "tail", "max_pause")
            if f"--{k.replace('_', '-')}" in args}
    main(args[0], **opts)
