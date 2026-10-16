# German, Dutch, French, Spanish and Italian corpora

Five Suno batches per language, same vocal profile and block tags as Polish and Hungarian (`pl/README.md` has the general notes). Built with what the first two languages taught:

- **Shorter batches.** Polish takes up to 5.4 min were clean; Hungarian takes over 6 min slurred in their last minute. These run about 3.5-5 min at the pace Suno actually read (10.7 characters per second, the Polish/Hungarian mean).
- **The last block of every batch is the expendable one** -- short imperatives or the closing -- so if a take degrades at the end, nothing rare is lost.
- **Lists in threes and fours, each line ending in a full stop.** Long comma-runs of numbers were where cuts went wrong.
- **Download the Lead Vocal stem** with every take. Suno added ambient music under every Polish take despite the prompt; the stems were clean.

Save takes as `/Volumes/Archive/Training data/polyglot-training-data/<language>/<XX>-Batch N.wav` plus `<XX>-Batch N (Lead Vocal).wav`, as before.

Accept a take once it is at least the *accept from* length below (85% of the estimate) -- not 5 minutes, which these batches are not meant to reach.

## German -- `de/`

| batch | theme | chars | est. | accept from |
|---|---|---|---|---|
| `DE-Batch 1` | settling, the body, a garden | 2,916 | 4:32 | 3:51 |
| `DE-Batch 2` | breath, counting, numbers, dates, prices, holidays | 2,943 | 4:35 | 3:53 |
| `DE-Batch 3` | places, seasons, a journey across the country, mountain and sea | 2,714 | 4:13 | 3:35 |
| `DE-Batch 4` | everyday conversation: market, family, doctor, weather, tickets, a local dish | 3,049 | 4:44 | 4:02 |
| `DE-Batch 5` | tongue twisters, sound drills, minimal pairs, a story, the return | 2,482 | 3:51 | 3:17 |

Style prompt (617 characters):

```
Spoken word meditation read in German by a native speaker from Germany. Male resonant baritone, centered mono voice, close mic intimacy, warm tape saturation, 70s tape color, introspective calm. A cappella from the first second to the last: one dry, unaccompanied voice and nothing else. Slow, even, unhurried narration in clear standard High German, every syllable articulated, natural German word stress, uvular r, crisp ich-sound and ach-sound, rounded ü and ö, long and short vowels clearly distinguished. One line at a time, each followed by a long silent pause. Plain reading, never sung, no melody, no humming.
```

## Dutch -- `nl/`

| batch | theme | chars | est. | accept from |
|---|---|---|---|---|
| `NL-Batch 1` | settling, the body, a garden | 2,764 | 4:18 | 3:39 |
| `NL-Batch 2` | breath, counting, numbers, dates, prices, holidays | 2,806 | 4:22 | 3:42 |
| `NL-Batch 3` | places, seasons, a journey across the country, mountain and sea | 2,661 | 4:08 | 3:31 |
| `NL-Batch 4` | everyday conversation: market, family, doctor, weather, tickets, a local dish | 2,874 | 4:28 | 3:48 |
| `NL-Batch 5` | tongue twisters, sound drills, minimal pairs, a story, the return | 2,306 | 3:35 | 3:03 |

Style prompt (602 characters):

```
Spoken word meditation read in Dutch by a native speaker from the Netherlands. Male resonant baritone, centered mono voice, close mic intimacy, warm tape saturation, 70s tape color, introspective calm. A cappella from the first second to the last: one dry, unaccompanied voice and nothing else. Slow, even, unhurried narration in standard Netherlands Dutch, every syllable articulated, natural Dutch word stress, soft throaty g, clear ui, ij and eu, long and short vowels clearly distinguished. One line at a time, each followed by a long silent pause. Plain reading, never sung, no melody, no humming.
```

## French -- `fr/`

| batch | theme | chars | est. | accept from |
|---|---|---|---|---|
| `FR-Batch 1` | settling, the body, a garden | 3,037 | 4:43 | 4:01 |
| `FR-Batch 2` | breath, counting, numbers, dates, prices, holidays | 2,981 | 4:38 | 3:56 |
| `FR-Batch 3` | places, seasons, a journey across the country, mountain and sea | 2,849 | 4:26 | 3:46 |
| `FR-Batch 4` | everyday conversation: market, family, doctor, weather, tickets, a local dish | 3,128 | 4:52 | 4:08 |
| `FR-Batch 5` | tongue twisters, sound drills, minimal pairs, a story, the return | 2,480 | 3:51 | 3:17 |

Style prompt (596 characters):

```
Spoken word meditation read in French by a native speaker from France. Male resonant baritone, centered mono voice, close mic intimacy, warm tape saturation, 70s tape color, introspective calm. A cappella from the first second to the last: one dry, unaccompanied voice and nothing else. Slow, even, unhurried narration in standard Parisian French, every syllable articulated, natural French stress at the end of each phrase, uvular r, full nasal vowels, rounded u and eu, smooth liaisons. One line at a time, each followed by a long silent pause. Plain reading, never sung, no melody, no humming.
```

## Spanish (Castilian) -- `es/`

| batch | theme | chars | est. | accept from |
|---|---|---|---|---|
| `ES-Batch 1` | settling, the body, a garden | 2,867 | 4:27 | 3:47 |
| `ES-Batch 2` | breath, counting, numbers, dates, prices, holidays | 2,800 | 4:21 | 3:42 |
| `ES-Batch 3` | places, seasons, a journey across the country, mountain and sea | 2,692 | 4:11 | 3:33 |
| `ES-Batch 4` | everyday conversation: market, family, doctor, weather, tickets, a local dish | 2,849 | 4:26 | 3:46 |
| `ES-Batch 5` | tongue twisters, sound drills, minimal pairs, a story, the return | 2,337 | 3:38 | 3:05 |

Style prompt (588 characters):

```
Spoken word meditation read in Spanish by a native speaker from Madrid, Spain. Male resonant baritone, centered mono voice, close mic intimacy, warm tape saturation, 70s tape color, introspective calm. A cappella from the first second to the last: one dry, unaccompanied voice and nothing else. Slow, even, unhurried narration in Castilian Spanish with distinción, z and soft c pronounced as th, rolled rr and tapped r, soft b, d and g between vowels, natural Spanish word stress. One line at a time, each followed by a long silent pause. Plain reading, never sung, no melody, no humming.
```

## Italian -- `it/`

| batch | theme | chars | est. | accept from |
|---|---|---|---|---|
| `IT-Batch 1` | settling, the body, a garden | 2,863 | 4:27 | 3:47 |
| `IT-Batch 2` | breath, counting, numbers, dates, prices, holidays | 2,736 | 4:15 | 3:37 |
| `IT-Batch 3` | places, seasons, a journey from Milan to Palermo, mountain and sea | 2,642 | 4:06 | 3:29 |
| `IT-Batch 4` | everyday conversation: market, family, doctor, weather, tickets, pasta al pomodoro | 2,834 | 4:24 | 3:45 |
| `IT-Batch 5` | tongue twisters, sound drills, double consonants, a story, the return | 2,200 | 3:25 | 2:54 |

Style prompt (582 characters):

```
Spoken word meditation read in Italian by a native speaker from Italy. Male resonant baritone, centered mono voice, close mic intimacy, warm tape saturation, 70s tape color, introspective calm. A cappella from the first second to the last: one dry, unaccompanied voice and nothing else. Slow, even, unhurried narration in standard Italian, every syllable articulated, natural Italian word stress, rolled r, double consonants clearly held, open and closed e and o kept apart. One line at a time, each followed by a long silent pause. Plain reading, never sung, no melody, no humming.
```

Exclude styles, all five: the same list as Polish.

## Before training

espeak-ng writes German *r* as `ɾ`, the symbol Spanish and Polish use for their tapped r, and German final *-er* as English `ɜ`. With no language switch in the model, one symbol meaning two sounds blurs them. The correction layer gives each language's own sound its own symbol (German `ʁ`, `ɐ`) -- in training and at inference alike.
Italian marks double consonants two ways (`tː` in *fatto*, but `nn` in *nonno*) and writes its rolled r as `ɾ`, the Spanish tap's symbol; both go to the same layer. French and Spanish espeak output looked right in spot checks (Castilian `θ`, soft `β ð ɣ`, French nasals and `ʁ`).
