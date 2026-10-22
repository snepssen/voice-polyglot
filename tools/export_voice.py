"""Export a training checkpoint as a Piper voice: model.onnx + model.onnx.json.

    python export_voice.py CHECKPOINT OUT_DIR [--name NAME]

Writes OUT_DIR/NAME.onnx and NAME.onnx.json (default name
snepssen-polyglot-epoch-NNN), then checks the export speaks exactly like
the checkpoint (verify_rebuild.py, noise off, three lengths).

Three fixes, from tools-core/piper-workflows/export.py, where each was found:
  - torch.onnx.export defaults to the dynamo exporter, which refuses VITS's
    stochastic duration predictor; the legacy TorchScript tracer is what
    every Piper voice has been exported with.
  - Under that tracer generate_path reads its frame count as a Python int,
    frozen to the dummy trace's length -- the cause of the old "y-you"
    stutter. Scripting it reads the real length at runtime.
  - ONNX Runtime's telemetry uploader segfaults on this machine after the
    useful work is done; it is switched off before anything loads it.

The config is the shipped voice's, with this voice's name and a "polyglot"
block: these voices must be phonemized by polyphon (voice-polyglot), not by
plain espeak-ng, or the corrected and language-specific symbols they learned
are never asked for. The block also carries each language's playback
length_scale (voice_pace.py): pass it when speaking that language.
"""
# SPDX-License-Identifier: GPL-3.0-only
# The scripted sequence-mask/path implementation is adapted from
# https://github.com/OHF-Voice/piper1-gpl/tree/cb59d2dcc74602a24d02d235092c62f9d4a19c84

import json
import pathlib
import re
import subprocess
import sys

import onnxruntime as _ort

_ort.disable_telemetry_events()

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import piper_fixes  # noqa: E402,F401

import torch  # noqa: E402
import torch.onnx  # noqa: E402

_original_export = torch.onnx.export


def _export_with_legacy_tracer(*args, **kwargs):
    kwargs.setdefault("dynamo", False)
    return _original_export(*args, **kwargs)


torch.onnx.export = _export_with_legacy_tracer

import piper.train.vits.commons as _commons  # noqa: E402


@torch.jit.script
def _sequence_mask_scripted(length: torch.Tensor, max_length: int) -> torch.Tensor:
    x = torch.arange(max_length, dtype=length.dtype, device=length.device)
    return x.unsqueeze(0) < length.unsqueeze(1)


@torch.jit.script
def _generate_path_scripted(duration: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    b = mask.shape[0]
    t_y = mask.shape[2]
    t_x = mask.shape[3]
    cum_duration = torch.cumsum(duration, -1)
    cum_duration_flat = cum_duration.view(b * t_x)
    path = _sequence_mask_scripted(cum_duration_flat, t_y).to(mask.dtype)
    path = path.view(b, t_x, t_y)
    path = path - torch.nn.functional.pad(path, (0, 0, 1, 0, 0, 0))[:, :-1]
    path = path.unsqueeze(1).transpose(2, 3) * mask
    return path


_commons.generate_path = _generate_path_scripted

from piper.phoneme_ids import DEFAULT_PHONEME_ID_MAP  # noqa: E402
from voice_pace import LENGTH_SCALE  # noqa: E402
from piper.train.export_onnx import main as piper_export  # noqa: E402

TEMPLATE = {
    "audio": {"sample_rate": 22050, "quality": "medium"},
    "espeak": {"voice": "en-us"},
    "inference": {"noise_scale": 0.667, "length_scale": 1.0, "noise_w": 0.8},
    "phoneme_type": "espeak",
    "num_symbols": 256,
    "num_speakers": 1,
    "speaker_id_map": {},
    "piper_version": "1.8.0",
}
LANGUAGES = ["en", "pl", "hu", "de", "nl", "fr", "es", "it"]


def main(ckpt: str, out: str, name: str | None = None) -> None:
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    epoch = ck.get("epoch")
    name = name or f"snepssen-polyglot-epoch-{epoch:03d}"
    out_dir = pathlib.Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = out_dir / f"{name}.onnx"
    sys.argv = [sys.argv[0], "--checkpoint", ckpt, "--output-file", str(onnx_path)]
    piper_export()
    config = dict(TEMPLATE)
    config["dataset"] = name
    config["language"] = {"code": "mul", "family": "polyglot", "name_english": "English, Polish, Hungarian, German, Dutch, French, Spanish, Italian"}
    config["phoneme_id_map"] = {k: list(v) for k, v in DEFAULT_PHONEME_ID_MAP.items()}
    config["polyglot"] = {
        "phonemizer": "voice-polyglot tools/polyphon.py -- phonemize(text, lang); plain espeak-ng misses its fixes",
        "languages": LANGUAGES,
        "length_scale": {k: LENGTH_SCALE[k] for k in LANGUAGES},
        "epoch": epoch,
        "trained_from": "snepssen epoch 702 (rebuilt), all eight languages",
    }
    (out_dir / f"{name}.onnx.json").write_text(json.dumps(config, ensure_ascii=False, indent=2))
    check = subprocess.run([sys.executable, "-I", str(pathlib.Path(__file__).parent / "verify_rebuild.py"),
                            ckpt, str(onnx_path), str(out_dir / f"{name}.onnx.json")],
                           capture_output=True, text=True)
    verdict = [l for l in check.stdout.splitlines() if l.startswith(("PASS", "FAIL"))]
    print(f"{onnx_path} ({onnx_path.stat().st_size / 1e6:.0f} MB): {verdict[0] if verdict else 'verify failed: ' + check.stderr[-300:]}")
    if not verdict or not verdict[0].startswith("PASS"):
        sys.exit(1)


if __name__ == "__main__":
    args = sys.argv[1:]
    nm = args[args.index("--name") + 1] if "--name" in args else None
    main(args[0], args[1], nm)
