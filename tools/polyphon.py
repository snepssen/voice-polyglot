"""espeak-ng, corrected: the phonemizer every language of the voice goes through.

    phonemize(text, lang) -> list of sentences, each a list of symbols

The same function has to run when the training data is prepared and when the
voice speaks; a symbol the model learned under one spelling and is asked for
under another is a sound it has never made. Every rule is a plain rewrite --
on the spelling before espeak-ng reads it, or on the IPA it returns -- so the
whole layer ports to Swift as a table, and polyphon_cases.tsv holds the
expected output both implementations are checked against.

Two kinds of rule:

  fixes       espeak-ng says the wrong sound (Polish hard dz, Hungarian
              silent h). Found by checking its output word by word.
  separations espeak-ng uses one symbol for sounds that differ between the
              languages. The voice has no language switch -- a symbol is the
              only thing that tells it which sound to make -- so a sound that
              is only one language's gets a symbol only that language uses.
              Every symbol used is already in the voice's 166-symbol table.

English goes through untouched: the voice's English was trained on plain
espeak-ng en-us and must keep hearing exactly that.
"""

import re
import unicodedata
from functools import lru_cache

from piper.phoneme_ids import DEFAULT_PHONEME_ID_MAP
from piper.phonemize_espeak import ESPEAK_DATA_DIR, EspeakPhonemizer

ESPEAK_VOICE = {"en": "en-us", "pl": "pl", "hu": "hu", "de": "de", "nl": "nl",
                "fr": "fr", "es": "es", "it": "it"}

VOWELS = set("aeiouyæøœɐɑɒɔəɘɛɜɞɤɨɪɯɵɶʉʊʌʏ")
MARKS = set("ˈˌːˑ̯̩̪̺̻̝̃̊ʲʷˤ˞")

# --- spelling, before espeak-ng ---------------------------------------------

SPELLING: dict[str, list[tuple[str, str]]] = {
    "hu": [
        # Word-final h is silent in these; espeak-ng says it ("méh" meːh).
        (r"(?i)\b(mé|ju|cse|olá|plé)h\b", r"\1"),
        # n before b is m (különben); espeak-ng already does it before p.
        (r"nb", "mb"),
        # sz + s assimilates to a long s-sound (egészség); spelled "ss" in
        # Hungarian, which espeak-ng reads as ʃʃ -> ʃː.
        (r"szs", "ss"),
    ],
}

# --- IPA, after espeak-ng ------------------------------------------------------

IPA: dict[str, list[tuple[str, str]]] = {
    "de": [
        # ç arrives decomposed (c + combining cedilla). c alone is Hungarian ty;
        # the precomposed symbol keeps the two apart.
        ("ç", "ç"),
        # -er is ɐ, not the English NURSE vowel ɜ.
        ("ɜ", "ɐ"),
        # espeak-ng writes r after a vowel as ɾ (nur, Körper) -- the Spanish tap.
        # In German it is vocalised: ɐ.
        ("ɾ", "ɐ"),
        # Before a vowel German r is uvular, not the Polish/Spanish trill.
        ("r", "ʁ"),
    ],
    "nl": [
        # Netherlands g and ch are both a uvular fricative; espeak-ng's ɣ and x
        # are Spanish soft g and Polish/Spanish ch.
        ("ɣ", "χ"),
        ("x", "χ"),
        # Dutch r is none of the trill, the tap or the uvular fricative.
        ("ɾ", "ʀ"),
        ("r", "ʀ"),
    ],
    "it": [
        # espeak-ng writes word-initial r as ɾ and medial r as r -- the wrong
        # way round. Italian r is one trilled sound, long when doubled.
        ("ɾ", "r"),
    ],
}

# A doubled consonant is written two ways by espeak-ng (Italian fatto tː but
# nonno nn; Hungarian always ll, ss). One spelling for one sound: Cː.
DOUBLE = re.compile(r"([^\s" + "".join(VOWELS) + "".join(MARKS) + r"'\-,.?!;:])\1(?!ː)")


def _polish_hard_dz(text_words: list[str], ipa_words: list[str]) -> list[str]:
    """espeak-ng reads every Polish dz as soft dʑ (dzwon, bardzo, pieniądze).
    Hard dz is dz not followed by i, ź or ż. Each dʑ in a word's IPA comes, in
    order, from one of dzi / dź / hard dz in its spelling; the hard ones
    become dz. Words whose counts disagree -- a dʑ from voicing assimilation
    across words -- are left as they are."""
    out = []
    for word, ipa in zip(text_words, ipa_words):
        kinds = ["hard" if m.group() == "dz" else "soft"
                 for m in re.finditer(r"dzi|dź|dż|dz", word.lower())
                 if m.group() != "dż"]
        hits = [m.start() for m in re.finditer("dʑ", ipa)]
        if "hard" in kinds and len(hits) == len(kinds):
            chars = list(ipa)
            for kind, pos in reversed(list(zip(kinds, hits))):
                if kind == "hard":
                    chars[pos:pos + 2] = list("dz")
            ipa = "".join(chars)
        out.append(ipa)
    return out


@lru_cache(maxsize=1)
def _espeak() -> EspeakPhonemizer:
    return EspeakPhonemizer(ESPEAK_DATA_DIR)


def words(text: str) -> list[str]:
    return re.findall(r"[^\W\d_]+(?:['’-][^\W\d_]+)*", text)


def phonemize(text: str, lang: str) -> list[list[str]]:
    for pattern, repl in SPELLING.get(lang, []):
        text = re.sub(pattern, repl, text)
    sentences = _espeak().phonemize(ESPEAK_VOICE[lang], text)
    if lang == "en":
        return sentences
    out = []
    # Sentences and words are matched back to the text in order.
    text_sentences = [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s]
    for n, symbols in enumerate(sentences):
        ipa = "".join(symbols)
        if lang == "pl" and n < len(text_sentences):
            tw, iw = words(text_sentences[n]), ipa.split(" ")
            if len(tw) == len(iw):
                ipa = " ".join(_polish_hard_dz(tw, iw))
            else:
                # espeak-ng fuses clitics ("nie walcz" -> one word); match the
                # whole sentence as a single word instead.
                ipa = _polish_hard_dz([" ".join(tw)], [ipa])[0]
        for a, b in IPA.get(lang, []):
            ipa = ipa.replace(a, b)
        ipa = DOUBLE.sub(r"\1ː", ipa)
        out.append(split(ipa))
    return out


def split(ipa: str) -> list[str]:
    """Symbols as the voice's table knows them: one per code point, except a
    precomposed ç, which the table has and which must not decompose."""
    return [c for c in unicodedata.normalize("NFD", ipa).replace("ç", "ç")]


def check_symbols(sentences: list[list[str]]) -> list[str]:
    return sorted({s for sent in sentences for s in sent if s not in DEFAULT_PHONEME_ID_MAP})


def run_cases(path: str) -> int:
    """polyphon_cases.tsv: lang <tab> text <tab> expected IPA (sentences
    joined by " | "). Any port of this layer must pass the same file."""
    failed = 0
    for line in open(path, encoding="utf-8"):
        if not line.strip() or line.startswith("#"):
            continue
        lang, text, expected = line.rstrip("\n").split("\t")
        got = " | ".join("".join(s) for s in phonemize(text, lang))
        if got != expected:
            failed += 1
            print(f"FAIL [{lang}] {text}\n  expected {expected}\n  got      {got}")
    return failed


if __name__ == "__main__":
    import sys
    if sys.argv[1] == "--check":
        n = run_cases(sys.argv[2])
        print("all cases pass" if not n else f"{n} failed")
        sys.exit(1 if n else 0)
    lang, text = sys.argv[1], " ".join(sys.argv[2:])
    for s in phonemize(text, lang):
        print("".join(s))
