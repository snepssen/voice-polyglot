"""Render the same sentences in every language from a training checkpoint.

    python checkin.py CHECKPOINT OUT_DIR [--no-noise]

Straight from the checkpoint in torch -- no ONNX export -- so any epoch can
be heard while the run continues. Phonemes come from polyphon, as in
training. Rendered as the voice is played: Piper's 0.667 / 0.8 noise scales,
with the random seed fixed per sentence so two epochs still differ only in
what the model learned. --no-noise renders with both at zero -- which is how
the first check-ins were made, and it misled: the duration predictor's
noise-free guess runs 20-25% faster than the speech it learned from, and
the timbre goes flat and buzzy.

Also renders the English sentence that used to end in a stray fragment, so
the old fault can be listened for.
"""

import pathlib
import sys
import wave

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import polyphon  # noqa: E402
from piper.phoneme_ids import DEFAULT_PHONEME_ID_MAP, phonemes_to_ids  # noqa: E402
from piper.train.vits.models import SynthesizerTrn  # noqa: E402

torch.serialization.add_safe_globals([(pathlib.PosixPath, "pathlib.PosixPath")])

SENTENCES = {
    "en": "Breathe in. Let your shoulders settle, and notice the weight of your hands.",
    "pl": "Dzień dobry, czy mógłby pan powtórzyć trochę wolniej? Szczebrzeszyn, chrząszcz, dzwon.",
    "hu": "Szia, hogy vagy? Ma reggel elmentem a piacra, és vettem friss kenyeret.",
    "de": "Heute Morgen war ich auf dem Markt und habe frisches Brot gekauft. Ich, ach, über.",
    "nl": "Goedemorgen, hoe gaat het? Ik wil graag een kopje koffie en een broodje kaas.",
    "fr": "Bonjour, comment allez-vous ? Je voudrais un pain au chocolat et un café crème.",
    "es": "Buenos días, ¿cómo estás? El perro corre por la carretera hacia la ciudad.",
    "it": "Buongiorno, come sta? Vorrei un cappuccino e un cornetto, per favore.",
    "en-short": "Breathe in.",
}


def load(ckpt_path: str) -> tuple[SynthesizerTrn, int]:
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    hp = ck["hyper_parameters"]
    g = SynthesizerTrn(
        n_vocab=ck["state_dict"]["model_g.enc_p.emb.weight"].shape[0],
        spec_channels=hp["filter_length"] // 2 + 1,
        segment_size=hp["segment_size"] // hp["hop_length"],
        inter_channels=hp["inter_channels"], hidden_channels=hp["hidden_channels"],
        filter_channels=hp["filter_channels"], n_heads=hp["n_heads"], n_layers=hp["n_layers"],
        kernel_size=hp["kernel_size"], p_dropout=hp["p_dropout"], resblock=hp["resblock"],
        resblock_kernel_sizes=hp["resblock_kernel_sizes"],
        resblock_dilation_sizes=hp["resblock_dilation_sizes"],
        upsample_rates=hp["upsample_rates"], upsample_initial_channel=hp["upsample_initial_channel"],
        upsample_kernel_sizes=hp["upsample_kernel_sizes"], n_speakers=hp["num_speakers"],
        gin_channels=hp["gin_channels"], use_sdp=hp["use_sdp"],
    )
    g.load_state_dict({k[8:]: v for k, v in ck["state_dict"].items() if k.startswith("model_g.")})
    g.eval()
    return g, hp["sample_rate"]


def main(ckpt: str, out: str, noise: bool = True) -> None:
    g, rate = load(ckpt)
    out_dir = pathlib.Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    ns, nw = (0.667, 0.8) if noise else (0.0, 0.0)
    for key, text in SENTENCES.items():
        lang = key.split("-")[0]
        parts = []
        for n, sent in enumerate(polyphon.phonemize(text, lang)):
            ids = phonemes_to_ids(sent, DEFAULT_PHONEME_ID_MAP)
            torch.manual_seed(1000 * list(SENTENCES).index(key) + n)
            with torch.no_grad():
                audio = g.infer(torch.tensor([ids]), torch.tensor([len(ids)]),
                                noise_scale=ns, length_scale=1.0, noise_scale_w=nw)[0]
            parts += [audio.squeeze().numpy(), np.zeros(int(0.25 * rate), np.float32)]
        x = np.concatenate(parts)
        x = x / max(np.abs(x).max(), 1e-6) * 0.9
        with wave.open(str(out_dir / f"{key}.wav"), "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(rate)
            w.writeframes((x * 32767).astype(np.int16).tobytes())
    print(f"{len(SENTENCES)} sentences -> {out_dir}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], "--no-noise" not in sys.argv[3:])
