# Polish corpus

Five batches for Suno, one generation each, read by the same vocal profile as
the English corpus. About 24 minutes in all, at the 13.0 characters per second
the English Suno corpus measured.

| batch | theme | chars | est. track |
|---|---|---|---|
| `batch-01.suno.txt` | settling, the body, reassurance | 3,657 | ~4.7 min |
| `batch-02.suno.txt` | breath, counting, numbers, dates, prices | 3,983 | ~5.1 min |
| `batch-03.suno.txt` | places, seasons, a journey through Polish cities | 3,694 | ~4.7 min |
| `batch-04.suno.txt` | everyday conversation: shop, family, doctor, work, travel | 3,775 | ~4.8 min |
| `batch-05.suno.txt` | tongue twisters, sound sets, minimal pairs, a story, the return | 3,409 | ~4.4 min |

Every Polish spelling is covered, and the rarest — `dż` (7), `dź` (11) and
hard `dz` (36) — are rare in the language too. `tools/corpus_check.py corpus/pl pl`
re-measures sizes and sound counts after any edit.

Batch 4 is deliberately not meditation. translate-core will voice ordinary
conversation, so the voice needs to have read some, in the same calm delivery.

## Style prompt (580 characters)

```
Spoken word meditation read in Polish by a native Polish speaker. Male resonant baritone, centered mono voice, close mic intimacy, warm tape saturation, 70s tape color, introspective calm. A cappella from the first second to the last: one dry, unaccompanied voice and nothing else. Slow, even, unhurried narration with every syllable clearly articulated, natural Polish stress on the second-to-last syllable, rolled tongue-tip r, crisp sz, cz, ś and ć, full nasal ę and ą. One line at a time, each followed by a long silent pause. Plain reading, never sung, no melody, no humming.
```

## Exclude styles

```
music, instruments, beat, drums, percussion, piano, synth, pads, strings, guitar, bass, ambient drone, field recording, choir, backing vocals, singing, humming, harmony, sidechain, reverb, echo
```

The voice descriptors (baritone, centered mono, close mic, warm tape
saturation, 70s tape color, introspective calm) are word for word from the
English prompt. The voice's colour should not change with the language, or the
model learns that Polish sounds like a different microphone. What was dropped
is arrangement — the 60 BPM pulse, sidechain swell, sparse arrangement, soft
low end — and the breathy hums, which would be sound in a clip with no word in
its transcript.

## Generating

- One batch per generation, pasted whole. The `[bracketed]` lines are
  instructions to Suno, not text; the checker drops them from the transcript.
- If Suno rushes, skips lines or drifts into singing in a long batch, split it
  at any `[A Cappella]` / `[Voice Only]` line — every block stands alone.
- Two takes per batch, keep the cleaner one. Download WAV, not MP3.
- If a take has music under it anyway, keep it: the English corpus went
  through vocal separation for the same reason. A take that *sings* is lost.

## For the Polish reviewer

What only a native ear can judge, per take:

1. Words read wrongly, skipped or added — note the line.
2. Wrong stress (Polish stresses the second-to-last syllable; *uniwersytet*,
   *matematyka* and *zrobilibyśmy* in batch 5 are the exceptions).
3. Sounds that come out English: soft r instead of rolled, ś/ć/ź flattened into
   sz/cz/ż, nasal ę/ą missing.
4. Anything that sounds unnatural, even if correct.

## Before training: espeak-ng's Polish, checked

The model is taught each sound as espeak-ng spells it, so a wrong spelling is
learned as the right one. One real error, now corrected by `tools/polyphon.py`:

    dzwon      dʑvɔn      -> dzvɔn     espeak-ng has no hard dz; every dz came out soft
    bardzo     bardʑɔ     -> bardzɔ
    pieniądze  …ɔndʑɛ     -> …ɔndzɛ

An earlier version of this note also listed final-consonant errors (*bank*
`baŋɡ`, *dźwięk* `…ŋɡ`). Those were wrong: they came from phonemizing a list
of words as one sentence, where Polish voicing assimilation across words is
correct. Standing alone, espeak-ng devoices them properly (`baŋk`, `xlɛp`,
`buk`), and in running speech it voices across words as Polish does
(*jest bardzo* `jɛzd`).
