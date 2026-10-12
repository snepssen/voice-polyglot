"""Flag clips whose speech is far longer or shorter than their text.

    python length_check.py CLIPS_DIR MANIFEST.tsv

A clip that swallowed a neighbouring line, or lost half of its own, still
passes a lenient transcript check -- Parakeet hears most of the right words.
It cannot pass a length check: speech time (first to last sound) against the
label's characters, compared to that take's own median pace. Writes a `pace`
column (1.0 = the take's median) and marks clips outside LOW..HIGH `review`.
"""
import csv, sys, wave
from pathlib import Path
import numpy as np

LOW, HIGH = 0.6, 1.5

def speech_seconds(path):
    with wave.open(str(path)) as w:
        x = np.frombuffer(w.readframes(w.getnframes()), np.int16) / 32768; rate = w.getframerate()
    n = int(0.01 * rate); e = 20 * np.log10(np.sqrt((x[: len(x) // n * n].reshape(-1, n) ** 2).mean(1)) + 1e-9)
    loud = np.where(e > -50)[0]
    return (loud[-1] - loud[0] + 1) * 0.01 if len(loud) else 0.0

clips, manifest = Path(sys.argv[1]), sys.argv[2]
rows = list(csv.DictReader(open(manifest, encoding="utf-8"), delimiter="\t"))
for r in rows:
    r["_rate"] = len(r["text"]) / max(speech_seconds(clips / f"{r['id']}.wav"), 0.1)
flagged = []
for take in dict.fromkeys(r["take"] for r in rows):
    tr = [r for r in rows if r["take"] == take]
    med = np.median([r["_rate"] for r in tr if r["status"] == "kept"] or [r["_rate"] for r in tr])
    for r in tr:
        r["pace"] = round(r["_rate"] / med, 2)  # <1: more audio than its text explains
        if not LOW <= r["pace"] <= HIGH:
            if r["status"] == "kept":
                flagged.append(r)
            r["status"] = "review"
for r in rows:
    del r["_rate"]
with open(manifest, "w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter="\t"); w.writeheader(); w.writerows(rows)
print(f"{manifest}: {len(flagged)} kept clips moved to review on length")
for r in flagged:
    print(f"  {r['id']} pace {r['pace']}  {r['text'][:60]}")
