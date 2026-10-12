"""Cut Suno vocal stems into training clips, one script line (or a few short
ones) per clip.

    python cut_clips.py LANG SCRIPT_DIR ASR_DIR STEMS_DIR OUT_DIR

The script supplies the words; Parakeet's timings only say where they are.
Each boundary is pushed into the silence between two lines -- the longest
silent run between the last recognised word of one line and the first of the
next -- and the clip keeps 150 ms before its first sound and up to 700 ms
after its last (TRAINING.md: hard-trimmed ends teach the model to stop
abruptly).

Lines shorter than MIN_WORDS words are joined to the next line, as the Rode
corpus did ("Jeden, wdech." alone is a fragment with no prosody). The pause
inside a joined clip is shortened to JOIN_PAUSE so the voice doesn't learn
meditation-length silences at every full stop.

Every clip is then transcribed on its own and scored against its text: a cut
in the wrong place shows up as words missing or words belonging to a
neighbour. Clips under the bar are written but marked `review` in the
manifest rather than `kept`.

Stems are 48 kHz stereo; clips are written 44.1 kHz mono 16-bit, with one
gain per take bringing its median clip to the English corpus's -23.5 dB RMS.
"""

import csv
import json
import re
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from line_coverage import align, batch_of, fold, script_lines, similar, spell_digits  # noqa: E402

RATE = 44100
HOP = 0.01
SILENT_DB = -50.0
MIN_SILENCE = 0.15
LEAD = 0.15
TAIL = 0.70
MIN_WORDS = 4
MAX_CLIP = 12.0
JOIN_PAUSE = 0.45
SLACK = 0.4
TARGET_RMS = -23.5
KEEP_SCORE = {"en": 0.85, "pl": 0.85, "hu": 0.70, "de": 0.80, "nl": 0.80, "fr": 0.80, "es": 0.80, "it": 0.80}  # Parakeet's Hungarian is weaker


def load_mono(path: Path) -> np.ndarray:
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-ac", "1", "-ar", str(RATE),
                          "-f", "f32le", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.float32).copy()


def envelope(a: np.ndarray) -> np.ndarray:
    n = int(HOP * RATE)
    frames = a[: len(a) // n * n].reshape(-1, n)
    return 20 * np.log10(np.sqrt((frames ** 2).mean(1)) + 1e-9)


def silent_runs(env: np.ndarray, lo: float, hi: float) -> list[tuple[float, float]]:
    """Silent stretches (start, end) in seconds inside [lo, hi]."""
    i0, i1 = max(0, int(lo / HOP)), min(len(env), int(hi / HOP))
    runs, start = [], None
    for i in range(i0, i1):
        quiet = env[i] < SILENT_DB
        if quiet and start is None:
            start = i
        if (not quiet or i == i1 - 1) and start is not None:
            end = i if not quiet else i + 1
            if (end - start) * HOP >= MIN_SILENCE:
                runs.append((start * HOP, end * HOP))
            start = None
    return runs


def asr_words(path: Path, lang: str) -> list[dict]:
    merged: list[dict] = []
    for t in json.loads(path.read_text())["words"]:
        if merged and not t["text"].startswith(" "):
            merged[-1]["text"] += t["text"]
            merged[-1]["end"] = t["end"]
        else:
            merged.append(dict(t))
    out = []
    for w in merged:  # a spelled-out number shares its digits' timing
        for part in spell_digits(w["text"], lang).split():
            if fold(part):
                out.append({"text": " " + part, "start": w["start"], "end": w["end"]})
    return out


def line_spans(lines: list[str], words: list[dict]) -> list[tuple[float, float, float] | None]:
    """(first word start, last word end, share heard) per line, None if unheard."""
    owner, swords = [], []
    for k, line in enumerate(lines):
        for w in line.split():
            if fold(w):
                swords.append(fold(w)); owner.append(k)
    hits: dict[int, list[float]] = {}
    count = [0] * len(lines)
    for i, j in align(swords, [fold(w["text"]) for w in words]):
        if i is not None and j is not None:
            k = owner[i]
            count[k] += 1
            t = hits.setdefault(k, [words[j]["start"], words[j]["end"]])
            t[0], t[1] = min(t[0], words[j]["start"]), max(t[1], words[j]["end"])
    total = [owner.count(k) for k in range(len(lines))]
    return [(hits[k][0], hits[k][1], count[k] / total[k]) if k in hits else None
            for k in range(len(lines))]


def quietest(env, lo: float, hi: float, width: float = 0.03) -> float:
    """Centre of the quietest `width` window in [lo, hi]."""
    i0, i1 = int(lo / HOP), max(int(lo / HOP) + 1, int(hi / HOP))
    w = max(1, int(width / HOP))
    seg = env[i0:i1]
    if len(seg) <= w:
        return (lo + hi) / 2
    sums = np.convolve(seg, np.ones(w), "valid")
    return (i0 + int(np.argmin(sums)) + w / 2) * HOP


def split_between(env, end_prev: float, start_next: float) -> tuple[float, float]:
    """Where one clip ends and the next begins, given the last sound of one
    line and the first of the next. Into the longest silence if there is one;
    otherwise both meet at the quietest point, so neither cuts into a word."""
    runs = silent_runs(env, end_prev, start_next) if start_next > end_prev else []
    if runs:
        r0, r1 = max(runs, key=lambda r: r[1] - r[0])
        return min(r0 + TAIL, r1 - LEAD), max(r0, r1 - LEAD)
    # Lines read straight through: Parakeet's word edges are loose by a few
    # hundred ms in a fast list, so look for the dip a little either side.
    lo, hi = min(end_prev, start_next) - SLACK, max(end_prev, start_next) + SLACK
    c = quietest(env, lo, hi)
    return c, c


def bounds(spans, env, duration) -> list[tuple[float, float] | None]:
    """Clip start/end per heard line, pushed into the silences around it."""
    out: list[tuple[float, float] | None] = [None] * len(spans)
    heard = [k for k, s in enumerate(spans) if s and s[2] >= 0.5]
    starts, ends = {}, {}
    for n, k in enumerate(heard):
        s0, s1, _ = spans[k]
        if n == 0:
            runs = silent_runs(env, 0.0, s0)
            starts[k] = max(runs[-1][0], runs[-1][1] - LEAD) if runs else quietest(env, max(0, s0 - 1), s0)
        if n + 1 == len(heard):
            runs = silent_runs(env, s1, duration)
            ends[k] = min(runs[0][0] + TAIL, runs[0][1] - LEAD) if runs else quietest(env, s1, min(duration, s1 + 1))
            continue
        nk = heard[n + 1]
        if nk == k + 1:
            ends[k], starts[nk] = split_between(env, s1, spans[nk][0])
        else:
            # a dropped line sits between: stop at the first quiet after this
            # line, start at the last quiet before the next
            mid_lo, mid_hi = s1, spans[nk][0]
            after = silent_runs(env, s1, min(mid_hi, s1 + 2.0))
            ends[k] = min(after[0][0] + TAIL, after[0][1] - LEAD) if after else quietest(env, s1, min(mid_hi, s1 + 1.0))
            before = silent_runs(env, max(mid_lo, spans[nk][0] - 2.0), spans[nk][0])
            starts[nk] = max(before[-1][0], before[-1][1] - LEAD) if before else quietest(env, max(mid_lo, spans[nk][0] - 1.0), spans[nk][0])
    for k in heard:
        out[k] = (starts[k], max(ends[k], starts[k] + 0.2))
    return out


def plan_clips(lines, spans, cuts) -> list[list[int]]:
    """Group consecutive usable lines so no clip is a fragment."""
    groups: list[list[int]] = []
    for k, line in enumerate(lines):
        if cuts[k] is None:
            groups.append([])  # breaks adjacency
            continue
        if groups and groups[-1]:
            g = groups[-1]
            short = len(lines[g[-1]].split()) < MIN_WORDS or sum(len(lines[i].split()) for i in g) < MIN_WORDS
            span = cuts[k][1] - cuts[g[0]][0]
            if short and g[-1] == k - 1 and span <= MAX_CLIP:
                g.append(k)
                continue
        groups.append([k])
    return [g for g in groups if g]


def render(a, cuts, group) -> np.ndarray:
    keep = int(JOIN_PAUSE * RATE) // 2
    parts: list[np.ndarray] = []
    for n, k in enumerate(group):
        s, e = cuts[k]
        seg = a[int(s * RATE):int(e * RATE)]
        if n:  # joined lines: keep JOIN_PAUSE of silence between them
            parts[-1] = trim_tail(parts[-1], keep)
            seg = trim_head(seg, keep)
        parts.append(seg)
    return shorten_pauses(np.concatenate(parts))


def shorten_pauses(x: np.ndarray, longest: float = 1.0, keep: float = 0.6) -> np.ndarray:
    """Pauses longer than `longest` inside a clip become `keep` long.
    Suno's meditation pacing leaves 2-second gaps mid-sentence; a voice
    that learns them pauses like that in conversation too."""
    env = envelope(x)
    loud = np.where(env >= SILENT_DB)[0]
    if len(loud) < 2:
        return x
    n = int(HOP * RATE)
    first, last = loud[0] * n, (loud[-1] + 1) * n
    cuts, pos = [], first
    for r0, r1 in silent_runs(env, first / RATE, last / RATE):
        if r1 - r0 > longest:
            mid = (r0 + r1) / 2
            cuts.append((int((mid - keep / 2) * RATE), int((mid + keep / 2) * RATE), int(r0 * RATE), int(r1 * RATE)))
    if not cuts:
        return x
    out, pos = [], 0
    for k0, k1, r0, r1 in cuts:
        out.append(x[pos:r0 + (k1 - k0) // 2])
        pos = r1 - (k1 - k0) // 2
    out.append(x[pos:])
    return np.concatenate(out)


def trim_tail(x, keep):
    env = envelope(x)
    loud = np.where(env >= SILENT_DB)[0]
    last = (loud[-1] + 1) * int(HOP * RATE) if len(loud) else 0
    return x[: min(len(x), last + keep)]


def trim_head(x, keep):
    env = envelope(x)
    loud = np.where(env >= SILENT_DB)[0]
    first = loud[0] * int(HOP * RATE) if len(loud) else 0
    return x[max(0, first - keep):]


def score(text: str, heard: str, lang: str) -> float:
    heard = spell_digits(heard, lang)
    a = [fold(w) for w in text.split() if fold(w)]
    b = [fold(w) for w in heard.split() if fold(w)]
    path = align(a, b)
    hit = sum(1 for i, j in path if i is not None and j is not None)
    extra = sum(1 for i, j in path if i is None)
    return max(0.0, (hit - 0.5 * extra) / max(len(a), 1))


def repair(a, env, cuts, groups, lines, scored, check, keep) -> int:
    """Move boundaries that a failed clip check points at.

    Inside a fast run of short words or numbers Parakeet's timings are too
    loose to say which side of a pause a word fell on. For each boundary
    between two contiguous clips where either fails, try every silence in
    their combined span and keep the cut whose two clips score best together.
    """
    moved = 0
    for n in range(len(groups) - 1):
        g, h = groups[n], groups[n + 1]
        if g[-1] + 1 != h[0] or min(scored[n][0], scored[n + 1][0]) >= keep:
            continue
        lo, hi = cuts[g[0]][0] + 0.3, cuts[h[-1]][1] - 0.3
        best = (scored[n][0] + scored[n + 1][0], None)
        candidates = [(min(r[0] + TAIL, r[1] - LEAD), max(r[0], r[1] - LEAD)) for r in silent_runs(env, lo, hi)]
        if not candidates:  # no silence anywhere: try the quiet dips instead
            dips = np.where(env[int(lo / HOP):int(hi / HOP)] < np.percentile(env[int(lo / HOP):int(hi / HOP)], 10))[0]
            candidates = [((int(lo / HOP) + d) * HOP,) * 2 for d in dips[::5]]
        for end, start in candidates:
            if end - cuts[g[-1]][0] < 0.5 or cuts[h[0]][1] - start < 0.5:
                continue
            old = cuts[g[-1]], cuts[h[0]]
            cuts[g[-1]], cuts[h[0]] = (old[0][0], end), (start, old[1][1])
            sg, sh = check(render(a, cuts, g), g), check(render(a, cuts, h), h)
            cuts[g[-1]], cuts[h[0]] = old
            if sg[0] + sh[0] > best[0] + 0.05:
                best = (sg[0] + sh[0], (end, start, sg, sh))
        if best[1]:
            end, start, sg, sh = best[1]
            cuts[g[-1]] = (cuts[g[-1]][0], end)
            cuts[h[0]] = (start, cuts[h[0]][1])
            scored[n], scored[n + 1] = sg, sh
            moved += 1
    return moved


def main(lang, script_dir, asr_dir, stems_dir, out_dir) -> None:
    from parakeet_mlx import from_pretrained
    from huggingface_hub import snapshot_download
    model = from_pretrained(snapshot_download("mlx-community/parakeet-tdt-0.6b-v3", local_files_only=True))

    out = Path(out_dir)
    (out / "clips").mkdir(parents=True, exist_ok=True)
    meta, manifest = [], []
    for asr in sorted(Path(asr_dir).glob("*.json")):
        batch = batch_of(asr)
        lines = script_lines(Path(script_dir) / f"batch-{batch:02d}.suno.txt")
        stem = Path(stems_dir) / f"{asr.stem} (Lead Vocal).wav"
        a = load_mono(stem if stem.exists() else Path(stems_dir) / f"{asr.stem}.wav")
        env = envelope(a)
        spans = line_spans(lines, asr_words(asr, lang))
        cuts = bounds(spans, env, len(a) / RATE)
        groups = plan_clips(lines, spans, cuts)
        rms = [20 * np.log10(np.sqrt(np.mean(render(a, cuts, g) ** 2)) + 1e-9) for g in groups]
        gain = 10 ** ((TARGET_RMS - float(np.median(rms))) / 20)

        def check(x, g):
            text = " ".join(lines[k] for k in g)
            with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
                write_wav(Path(tmp.name), np.clip(x * gain, -1, 1))
                heard = model.transcribe(tmp.name).text.strip()
            return score(text, heard, lang), heard

        scored = [check(render(a, cuts, g), g) for g in groups]
        moved = repair(a, env, cuts, groups, lines, scored, check, KEEP_SCORE[lang])
        dropped = [k + 1 for k in range(len(lines)) if cuts[k] is None]
        for n, g in enumerate(groups):
            cid = f"{lang}-{batch:02d}-{n:03d}"
            x = np.clip(render(a, cuts, g) * gain, -1, 1)
            text = " ".join(lines[k] for k in g)
            sc, heard = scored[n]
            env_x = envelope(x)
            edge = max(env_x[:3].max(), env_x[-3:].max()) if len(env_x) > 6 else 0.0
            write_wav(out / "clips" / f"{cid}.wav", x)
            meta.append((cid, text))
            manifest.append({"id": cid, "take": asr.stem, "lines": "+".join(str(k + 1) for k in g),
                             "start": round(cuts[g[0]][0], 2), "end": round(cuts[g[-1]][1], 2),
                             "seconds": round(len(x) / RATE, 2), "score": round(sc, 2),
                             "edge_db": round(float(edge), 1),
                             "status": "kept" if sc >= KEEP_SCORE[lang] and edge < -40 else "review",
                             "heard": heard, "text": text})
        kept = sum(1 for m in manifest if m["take"] == asr.stem and m["status"] == "kept")
        print(f"{asr.stem}: {len(groups)} clips ({kept} kept), {moved} boundaries moved, "
              f"gain {20 * np.log10(gain):+.1f} dB, lines not cut: {dropped or 'none'}", flush=True)

    with open(out / "metadata.csv", "w", encoding="utf-8") as f:
        for cid, text in meta:
            f.write(f"{cid}|{text}\n")
    with open(out / "manifest.tsv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(manifest[0]), delimiter="\t")
        w.writeheader(); w.writerows(manifest)
    secs = [m["seconds"] for m in manifest]
    kept = [m["seconds"] for m in manifest if m["status"] == "kept"]
    print(f"\n{len(manifest)} clips, {sum(secs) / 60:.1f} min; kept {len(kept)} ({sum(kept) / 60:.1f} min), "
          f"review {len(secs) - len(kept)}; clip length min {min(secs):.1f} / median {np.median(secs):.1f} / max {max(secs):.1f} s")


def write_wav(path: Path, x: np.ndarray) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(RATE)
        w.writeframes((x * 32767).astype(np.int16).tobytes())


if __name__ == "__main__":
    main(*sys.argv[1:6])
