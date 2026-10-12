"""Check a rebuilt checkpoint speaks exactly like the ONNX export it came from.

Both are run with the noise scales at zero, so the comparison is
deterministic: same phoneme ids in, compare the audio out.

    python verify_rebuild.py REBUILT.ckpt EXPORT.onnx EXPORT.onnx.json
"""

import json
import sys

import numpy as np
import onnxruntime as ort
import torch
from piper import PiperVoice
from piper.train.vits.models import SynthesizerTrn

SENTENCES = [
    "Breathe in.",
    "Let your shoulders settle, and notice the weight of your hands.",
    "Hi, how are you? This morning I went to the market and bought fresh bread, "
    "then sat by the water for a long while, doing nothing at all.",
]


def main(ckpt_path: str, onnx_path: str, config_path: str) -> None:
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    hp = ck["hyper_parameters"]
    kept = sorted({".".join(k.split(".")[1:4]) for k in ck["state_dict"]
                   if k.startswith("model_g.dp.flows.")})
    print("dp.flows present:", kept)

    n_vocab = ck["state_dict"]["model_g.enc_p.emb.weight"].shape[0]
    g = SynthesizerTrn(
        n_vocab=n_vocab,
        spec_channels=hp["filter_length"] // 2 + 1,
        segment_size=hp["segment_size"] // hp["hop_length"],
        inter_channels=hp["inter_channels"],
        hidden_channels=hp["hidden_channels"],
        filter_channels=hp["filter_channels"],
        n_heads=hp["n_heads"],
        n_layers=hp["n_layers"],
        kernel_size=hp["kernel_size"],
        p_dropout=hp["p_dropout"],
        resblock=hp["resblock"],
        resblock_kernel_sizes=hp["resblock_kernel_sizes"],
        resblock_dilation_sizes=hp["resblock_dilation_sizes"],
        upsample_rates=hp["upsample_rates"],
        upsample_initial_channel=hp["upsample_initial_channel"],
        upsample_kernel_sizes=hp["upsample_kernel_sizes"],
        n_speakers=hp["num_speakers"],
        gin_channels=hp["gin_channels"],
        use_sdp=hp["use_sdp"],
    )
    sd = {k[len("model_g."):]: v for k, v in ck["state_dict"].items() if k.startswith("model_g.")}
    g.load_state_dict(sd, strict=True)
    g.eval()

    voice = PiperVoice.load(onnx_path, config_path=config_path)
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    worst = 0.0
    for text in SENTENCES:
        ids = []
        for ph in voice.phonemize(text):
            ids += voice.phonemes_to_ids(ph)
        x = np.array([ids], np.int64)
        ref = sess.run(None, {
            "input": x,
            "input_lengths": np.array([len(ids)], np.int64),
            "scales": np.array([0.0, 1.0, 0.0], np.float32),
        })[0].squeeze()
        with torch.no_grad():
            out = g.infer(torch.from_numpy(x), torch.tensor([len(ids)]),
                          noise_scale=0.0, length_scale=1.0, noise_scale_w=0.0)[0]
        out = out.squeeze().numpy()
        n = min(len(ref), len(out))
        diff = float(np.abs(ref[:n] - out[:n]).max()) if n else float("inf")
        peak = float(np.abs(ref).max())
        worst = max(worst, diff / peak)
        print(f"{len(ids):4} ids  onnx {len(ref):7} samples  rebuilt {len(out):7}  "
              f"max diff {diff:.2e} ({diff / peak:.2e} of peak)")
    print("PASS" if worst < 1e-3 else "FAIL", f"worst relative diff {worst:.2e}")


if __name__ == "__main__":
    main(*sys.argv[1:4])
