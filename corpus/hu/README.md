# Hungarian corpus

Five batches for Suno, same structure, tags and vocal profile as the Polish
(`../pl/README.md` has the generating notes). About 23 minutes at the English
corpus's measured 13.0 characters per second.

| batch | theme | chars | est. track |
|---|---|---|---|
| `batch-01.suno.txt` | settling, the body, reassurance, a summer garden | 3,801 | ~4.9 min |
| `batch-02.suno.txt` | breath, counting, numbers, dates, prices, national holidays | 3,745 | ~4.8 min |
| `batch-03.suno.txt` | places, seasons, a journey through Hungary, the Tisza at night | 3,642 | ~4.7 min |
| `batch-04.suno.txt` | everyday conversation: market, family, doctor, work, tickets, lecsó | 3,586 | ~4.6 min |
| `batch-05.suno.txt` | tongue twisters, sound sets, long–short pairs, assimilation, a story, the return | 2,891 | ~3.7 min |

Batch 5 drills what Hungarian keys meaning on: vowel length (*kor / kór*,
*tör / tőr*, *örül / őrül*), consonant length (*hal / hall*, *megy / meggy*),
the palatals gy, ty, ny and ly, and the spelling-to-sound shifts a reader
makes without noticing (*látja*, *barátság*, *egészség*, *anyja*, *különben*).

## Style prompt (641 characters)

```
Spoken word meditation read in Hungarian by a native Hungarian speaker. Male resonant baritone, centered mono voice, close mic intimacy, warm tape saturation, 70s tape color, introspective calm. A cappella from the first second to the last: one dry, unaccompanied voice and nothing else. Slow, even, unhurried narration with every syllable clearly articulated, natural Hungarian stress on the first syllable of every word, rolled tongue-tip r, long and short vowels clearly distinguished, doubled consonants held, soft gy, ty and ny. One line at a time, each followed by a long silent pause. Plain reading, never sung, no melody, no humming.
```

Exclude styles: the same list as Polish.

## Reviewing

You are the native ear here. Beyond misread or skipped words, listen for:
long vowels read short (*kór* as *kor*), doubled consonants not held, gy/ty
flattened to d/t, and stress drifting off the first syllable.

## Before training: what espeak-ng gets wrong

Much better than its Polish — it does the assimilations (*látja* → `laːcːɑ`,
*barátság* → `bɑraːtʃːaːɡ`) and keeps length apart. Wrong so far:

    méh, juh      meːh, juh       final h is silent: meː, juː
    különben      kylønbɛn        n before b is m: kylømbɛn
    egészség      ɛɡeːʃeːɡ        the ʃ is long: ɛɡeːʃːeːɡ
    hall, toll    hɑll, toll      written double rather than lː (harmless if consistent)
