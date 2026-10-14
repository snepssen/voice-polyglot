# voice-polyglot

Teaching one small Piper voice to speak eight languages, so
[translate-core](https://github.com/snepssen/translate-core) can voice every
language it translates into with the same voice, rendered in milliseconds.

The voice is `snepssen` — the narrator from the Suno-generated English corpus,
the one [Voice Forge](https://github.com/snepssen/voice-forge) grew out of.
It has only ever heard English. Piper never reads letters: espeak-ng spells
every language in IPA first, and the voice's input table already holds 166
IPA symbols. So the alphabet was never the problem. What the voice lacks is
having *heard* the other languages, and that is what this repository gives it.

## Where it stands

| | clips | minutes |
|---|---|---|
| Polish | 386 | 23.6 |
| Hungarian | 287 | 21.8 |
| English | 223 | 21.4 |
| Spanish | 273 | 18.7 |
| German | 258 | 17.2 |
| Italian | 266 | 16.7 |
| French | 259 | 16.2 |
| Dutch | 244 | 15.4 |
| **all** | **2,196** | **2 h 31 min** |

The new languages were read by the same Suno vocal profile from scripts in
`corpus/`, cut into clips and checked here. Training started 2026-10-12 from the
epoch-702 voice with all eight languages in the mix.

## The English, re-cut

The English the voice already speaks was re-cut from its 70 source blocks by
the same pipeline, rather than patching the old clips' ends. That pass found
three things wrong with the corpus the voice was trained on:

- **The fragment.** 55 of the old 540 clips end with the next line's first
  sound after a short gap (`tools/tail_check.py`); none of the new ones in
  training do — the check is now part of selection for every language.
- **A drone.** 27 blocks were judged "original, nothing to remove" and used
  unseparated. Their pauses sit at −26 dB against −73 dB in the separated
  blocks: a steady 65 Hz tone (a low C, the old prompt's "soft low end") with
  89–98% of the pause energy under 150 Hz, running under the words too.
  Filtering cannot take it out without taking the voice's own low end with it,
  so those blocks are left out (`--bed`).
- **Labels from an old transcript.** The clip texts were a speech recogniser's,
  not a script: *Braden 2 34 hold 2 34 out* for a counted breath. Parakeet now
  has to agree with every label, and labels with figures are left out, since
  "2 34" doesn't say how it was spoken.

## What the first test showed

Fed other languages' IPA unchanged, the English voice dropped the rolled r
everywhere (*friss* → "fiss", *Brot* → "boogh"), lost Hungarian gy and ny, and
flapped t between vowels the American way (*vettem* → "vedem"). That was heard
by a native Hungarian ear; a speech recogniser scored it far more kindly. The
sounds a language lacks matter, and so does its rhythm, which only comes from
hearing it — which is why every language got its own corpus rather than being
inferred from its neighbours.

## The layout

    corpus/<lang>/batch-NN.suno.txt   reading scripts, ≤5,000 characters each (Suno's limit)
    corpus/README.md                  how to generate: style prompts, tags, accept-from lengths
    tools/                            everything below
    checkpoints/                      the rebuilt training checkpoint (not in git)
    work/                             transcripts, clips, training lists (not in git)

Takes live on the Archive drive, `polyglot-training-data/<language>/`, as the
mix and Suno's own `(Lead Vocal)` stem. Suno laid ambient music under every
Polish take however the prompt forbade it; the stems were clean, with pauses at
−62 to −78 dB.

## From takes to training data

    transcribe_takes.py   Parakeet v3 word timings, from the vocal stem
    line_coverage.py      which script lines each take actually said; a take
                          saved under the wrong batch shows up here (two Dutch
                          ones did — work/dutch/batch_map.json records it)
    cut_clips.py          one line (or a few short ones) per clip. The script
                          supplies the words, Parakeet only the timing; cuts go
                          into the silence between lines with 150 ms before and
                          up to 700 ms after; every clip is re-transcribed and
                          scored, and a failing boundary is moved to whichever
                          silence makes both neighbours pass
    length_check.py       a clip that swallowed a neighbouring line still passes
                          a lenient transcript check; it cannot pass this one
    voicing.py            how much of each clip is voiced rather than breathed
    select_clips.py       the training list, and why each clip left out was
    tail_check.py         clips ending in the next line's first sound
    retrim.py             clip padding brought to what the voice first learned
                          from (see Pace, below)
    review_page.py        a local listening page for what only a native ear
                          can judge
    export_voice.py       a checkpoint as a Piper voice (.onnx + .json), checked
                          to speak exactly like the checkpoint; carries the
                          tools-core export fixes (legacy tracer, scripted
                          generate_path -- the old "y-you" stutter)
    milestone_watch.sh    every 25 epochs of a run: export, check-in, score
    build_ids.py          every training line through polyphon, into the
                          phoneme ids Piper's trainer reads

`preflight-training-clips.json` is a [Media Preflight](https://github.com/snepssen/media-preflight)
profile for the finished clip set. Run against the English clips the voice was
trained on, it fails them more often than the new ones.

## polyphon — espeak-ng, corrected

`tools/polyphon.py` is the phonemizer every language goes through, in training
and when the voice speaks; a symbol learned under one spelling and asked for
under another is a sound the voice never made. Two kinds of rule, all plain
rewrites so the layer ports to Swift as a table:

- **fixes**, where espeak-ng says the wrong sound — Polish hard *dz* (it has
  none: *dzwon* came out *dźwon*), Hungarian silent final *h*, *n* before *b*,
  *sz + s*
- **separations**, where espeak-ng uses one symbol for sounds that differ by
  language. The voice has no language switch; a symbol is all that tells it
  which sound to make. German *r* and *-er* (`ʁ`, `ɐ`, not the Spanish tap and
  the English vowel), Dutch *g/ch* and *r* (`χ`, `ʀ`), one trilled Italian *r*,
  and every doubled consonant written one way (`Cː`)

English is never touched. Every symbol used was already in the voice's table.
`tools/polyphon_cases.tsv` holds the expected output any port must reproduce:

```sh
uv run python tools/polyphon.py --check tools/polyphon_cases.tsv
```

## Training

    tools/train-polyglot.sh new       start, warm-started from the rebuilt epoch-702 voice
    tools/train-polyglot.sh resume    continue from the newer of last.ckpt and latest.ckpt

`tools/train.py` is Piper's trainer with what 1.8.0 needs on this Mac — the
monotonic alignment core its wheel ships without (provided in numba), the
PosixPath allowance old checkpoints need, the val_mos callback that crashes
Lightning 2.6 dropped — and two guards: `new` refuses a run folder that
already holds checkpoints, and `resume` neutralises the warm start every
checkpoint carries, which would otherwise copy the base voice over the
restored weights. The warm start copies all 784 weights, none skipped.

It also saves what Piper doesn't. Piper's checkpoint callback writes
`last.ckpt` only when an epoch enters the val_mel top five; that is all
this Lightning does with `save_last`. Once val_mel flattens it hardly ever
happens. Epochs 75 and 100 of the long run were never on disk to export,
and a crash would have resumed from hours back. So `train.py` adds
`latest.ckpt`, replaced after every epoch, and
`milestone-epoch=NNN.ckpt` every 25 epochs (`MILESTONE_EVERY`), never
overwritten. `milestone_watch.sh` exports from those.

On the M1 at batch 8 an epoch of 2,152 clips took about 20 minutes on the
first cut and ~15 on the re-trimmed clips. The run is meant to go three to
four days and take its time; `milestone_watch.sh` exports the voice every 25
epochs to `work/runs/polyglot/exports/epoch-NNN/` and stays quiet otherwise. The run
starts under `caffeinate` and `nohup`, detached, logging to
`work/runs/polyglot.log`.

To listen while it trains:

```sh
tools/milestone_watch.sh                  # every 25 epochs: export, render, score, flag regressions
uv run python tools/checkin.py CHECKPOINT work/runs/checkin-v2/epoch-NNN
uv run python tools/checkin_score.py work/runs/checkin-v2/epoch-*
uv run python tools/listen_page.py        # every epoch side by side, blind if wanted
```

The same sentence in every language, noise off, from any checkpoint. The
score is how much Parakeet recovers — a trend to say which languages moved,
not a verdict. Before training:

    en 100%   es 75%   de 73%   it 68%   hu 51%   fr 40%   pl 35%   nl 21%

### Pace, and the padding that broke it

After 47 epochs the voice spoke every language intelligibly and sounded
rushed and robotic doing it. Two causes, measured:

- **The check-ins were rendered with the noise scales at zero.** The
  duration predictor's noise-free guess runs 20–25% faster than the speech
  it learned from, and the timbre goes flat. Rendered as the voice is
  played (0.667 / 0.8), German, French and Spanish came back to 24–26
  characters a second against 23–25 in their clips. Check-ins now render
  that way, with a fixed seed so epochs stay comparable.
- **The clips carried dead air.** cut_clips measured the last word against
  −50 dB, and Suno's breathy reverb decays between −50 and −40 dB, so each
  clip ended in half a second of decay plus the 0.7 s tail: ~0.3 s before
  the first word, ~1.1 s after the last, 0.55 s pauses at commas. The voice
  learned exactly that — half-second lead-ins, over a second of silence
  after every sentence — with the words packed between. The old English
  clips that produced the voice before this run had 0.07 s, 0.21 s and
  0.25 s. `tools/retrim.py` brings every clip to 0.08 s / 0.25 s with no
  inner pause over 0.35 s (25–38% of the new-language audio was silence),
  and training resumed from epoch 47 on the re-trimmed clips.

## The checkpoint

The epoch-702 training checkpoint was lost with the Steam Deck it trained on;
only its ONNX export survived. `tools/rebuild_from_onnx.py` takes every tensor
the voice speaks with from the export, and the training-only parts the export
drops — posterior encoder, the duration predictor's posterior flows,
discriminator — from the newest surviving checkpoint of the same voice, epoch
564. `tools/verify_rebuild.py` runs both with the noise scales at zero:

    25 ids    18688 samples each   max diff 1.75e-06 of peak
    125 ids   81408 samples each   max diff 3.31e-06 of peak
    288 ids  176128 samples each   max diff 2.49e-06 of peak

The first rebuild failed that check with durations 3–6% long: the exporter had
folded `exp(-logs)`'s minus sign into the stored constant.

```sh
E=/Volumes/Archive/voice-training-data/from-deck/gatewayforge-voice/voice-epoch702
B=/Volumes/Archive/piper-data/runs/snepssen/lightning_logs/version_1/checkpoints/last.ckpt
uv run python tools/rebuild_from_onnx.py $E/en_US-snepssen-medium.onnx $B checkpoints/snepssen-epoch702-rebuilt.ckpt
uv run python tools/verify_rebuild.py checkpoints/snepssen-epoch702-rebuilt.ckpt \
  $E/en_US-snepssen-medium.onnx $E/en_US-snepssen-medium.onnx.json
```

It is weights only, for `--model.warmstart_ckpt` on a **new** run; its own
`warmstart_ckpt` is cleared so it cannot copy a base model over itself if ever
resumed.

## Carried over, still open

- Piper's LR scheduler never steps (`automatic_optimization = False`); the old
  run sat at a constant 2e-4 / 1e-4.
- Piper's `trim_silence` would strip the padding these clips keep on purpose.
- The dataset cache keys on row number and text, not audio: a changed corpus
  needs a new `--data.cache_dir`.

## Setup

```sh
uv sync
```

Python 3.12, MLX for Parakeet (Apple Silicon), Piper 1.8.0. GPL-3.0, as Piper
and Voice Forge are.
