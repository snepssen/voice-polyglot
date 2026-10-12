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
| Spanish | 274 | 18.8 |
| German | 258 | 17.2 |
| Italian | 266 | 16.7 |
| French | 261 | 16.3 |
| Dutch | 244 | 15.4 |
| English | 540 | 42 — its old tail trim must be redone first |

The new languages were read by the same Suno vocal profile from scripts in
`corpus/`, cut into clips and checked here. Next: redo the English trim, then
train from the epoch-702 voice with all eight languages in the mix.

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
    review_page.py        a local listening page for what only a native ear
                          can judge
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

- The English corpus's corrected tail trim (`08-trimmed`) was lost with the
  Deck. Without it the voice relearns the fragment it used to add after short
  sentences.
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
