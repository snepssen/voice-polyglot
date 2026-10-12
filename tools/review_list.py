"""Turn coverage.json into a listening list, and measure what sits in the pauses.

For each line heard less than 80%, the window to listen to runs from the end
of the last well-heard line before it to the start of the next one after it.
For each take, the level of the true pauses -- stretches longer than 0.8 s
with no recognised word in them -- says whether anything sounds between lines.
Clean Suno pauses sit near -60 dB; a hum or drone puts them near -20 dB.

    python review_list.py COVERAGE.json TAKES_DIR
"""
import json, sys, wave
from pathlib import Path
import numpy as np

def mmss(t): return f"{int(t // 60)}:{t % 60:04.1f}"

def level(a, s, e, rate):
    x = a[int(s * rate):int(e * rate)]
    return 20 * np.log10(np.sqrt(np.mean(x ** 2)) + 1e-9) if len(x) else None

reports = json.loads(Path(sys.argv[1]).read_text())
for r in reports:
    with wave.open(str(Path(sys.argv[2]) / (r["take"] + ".wav"))) as w:
        rate = w.getframerate()
        a = np.frombuffer(w.readframes(w.getnframes()), np.int16).reshape(-1, w.getnchannels()).mean(1) / 32768
    lines = r["lines"]
    good = [l for l in lines if l["time"] and l["heard"] / l["words"] >= 0.8]
    asr = json.loads((Path(sys.argv[1]).parent / "asr" / (r["take"] + ".json")).read_text())["words"]
    gaps = [(p["end"] + 0.2, n["start"] - 0.2) for p, n in zip(asr, asr[1:]) if n["start"] - p["end"] > 0.8]
    speech = [level(a, *l["time"], rate) for l in good]
    quiet = [level(a, s, e, rate) for s, e in gaps]
    loud = [(s, e) for (s, e), q in zip(gaps, quiet) if q > -45]
    last = max(l["time"][1] for l in good)
    print(f"\n{r['take']}  ({len(a) / rate / 60:.1f} min, last confident line ends {mmss(last)})"
          f"  speech {np.median(speech):.0f} dB, pauses {np.median(quiet):.0f} dB, "
          f"{len(loud)}/{len(gaps)} pauses not silent" + (f" (first at {mmss(loud[0][0])})" if loud else ""))
    for l in lines:
        if l["heard"] / l["words"] >= 0.8:
            continue
        before = [g["time"][1] for g in good if g["line"] < l["line"]]
        after = [g["time"][0] for g in good if g["line"] > l["line"]]
        s = before[-1] if before else 0.0
        e = after[0] if after else len(a) / rate
        print(f"   {mmss(s):>7} – {mmss(e):<7} L{l['line']:<3} {l['text']}")
