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
    # 1.4 was still too quick for these four; a slower round is being judged
    "nl": 1.4,
    "fr": 1.4,
    "es": 1.4,
    "it": 1.4,
}


def length_scale(lang: str) -> float:
    return LENGTH_SCALE.get(lang, 1.0)
