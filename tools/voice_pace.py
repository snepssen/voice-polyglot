"""Per-language playback pace for the polyglot snepssen voice.

The voice speaks each language at the pace of its training clips. English
comes from the original snepssen corpus, slow and unhurried at ~4.9
syllables a second while sounding. Every other language came from Suno at
a brisk reading pace: 6.1-6.4 syllables a second, Spanish and Italian 8.0.
Next to the English they rush.

Piper's length_scale stretches every duration the model predicts, so the
fix is applied when speaking, not by retraining. Values were picked by ear
(2026-10-14) from the same joke rendered at several scales. Everything that
renders the voice reads them from here: export_voice.py writes them into
the voice's .onnx.json ("polyglot" -> "length_scale"), and checkin.py renders
milestones at them.
"""

LENGTH_SCALE = {
    "en": 1.0,
    "pl": 1.2,
    "hu": 1.2,
    "de": 1.4,
    # Dutch: E for most of the sentence, but "kan niet zwemmen" speeds up out
    # of nowhere and wanted F (1.85) -- a burst inside a sentence that one
    # scale per language cannot fix; per-token durations are the tool for it
    "nl": 1.7,
    "fr": 1.55,
    "es": 1.55,
    "it": 1.55,
}


def length_scale(lang: str) -> float:
    return LENGTH_SCALE.get(lang, 1.0)
