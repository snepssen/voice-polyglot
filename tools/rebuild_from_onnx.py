"""Rebuild a trainable snepssen checkpoint from the epoch-702 ONNX export.

The epoch-702 training checkpoint (Steam Deck, runs/snepssen version_5) is
gone; only its ONNX export survives. An export keeps every tensor the voice
needs to *speak*, so those are taken from it. What it drops is training-only:
the posterior encoder (enc_q), the duration predictor's posterior flows, and
the discriminator. Those come from the newest surviving checkpoint of the same
voice, epoch 564 (runs/snepssen version_1), and re-adapt within a few epochs.

Three export quirks are undone on the way in:
  - weight-normed convs are stored folded (w = g * v/|v|); split back with
    v = w, g = |w|, which reproduces w exactly
  - 33 of those carry tracer names (onnx::Conv_NNNN); the node that consumes
    them names the module
  - the phoneme embedding is stored as `sid`, and dp.flows.0.logs as the
    input of an Exp node, negated (the forward was exp(-logs))

The result is weights only, meant for --model.warmstart_ckpt on a *new* run.
Its warmstart_ckpt hyperparameter is cleared, so it cannot re-apply a base
model over itself if it is ever resumed.

    python rebuild_from_onnx.py EXPORT.onnx EPOCH564.ckpt OUT.ckpt
"""

import pathlib
import sys

import onnx
import torch
from onnx import numpy_helper

torch.serialization.add_safe_globals([(pathlib.PosixPath, "pathlib.PosixPath")])


def module_weight(node_name: str) -> str:
    # "/flow/flows.6/enc/in_layers.0/Conv" -> "flow.flows.6.enc.in_layers.0.weight"
    return ".".join(node_name.strip("/").split("/")[:-1]) + ".weight"


def onnx_tensors(path: str) -> dict[str, torch.Tensor]:
    m = onnx.load(path)
    raw = {i.name: torch.from_numpy(numpy_helper.to_array(i).copy()) for i in m.graph.initializer}
    out = {}
    for node in m.graph.node:
        for x in node.input:
            if x not in raw:
                continue
            if x == "sid" and node.name == "/enc_p/emb/Gather":
                out["enc_p.emb.weight"] = raw[x]
            elif x.startswith("onnx::Exp_"):
                # ElementwiseAffine's reverse is exp(-logs); the tracer folded
                # the minus sign into the constant, so this holds -logs.
                out[".".join(node.name.strip("/").split("/")[:-1]) + ".logs"] = -raw[x]
            elif x.startswith("onnx::Conv_"):
                out[module_weight(node.name)] = raw[x]
    for name, t in raw.items():
        if not name.startswith("onnx::") and name != "sid":
            out[name] = t
    return out


def split_weight_norm(w: torch.Tensor, g_shape: torch.Size) -> tuple[torch.Tensor, torch.Tensor]:
    keep = [d for d, n in enumerate(g_shape) if n != 1]
    dims = [d for d in range(w.dim()) if d not in keep]
    g = w.norm(dim=dims, keepdim=True)
    return g, w.clone()


def main(export: str, base: str, out: str) -> None:
    ck = torch.load(base, map_location="cpu", weights_only=False)
    sd = dict(ck["state_dict"])
    gen_keys = {k[len("model_g."):] for k in sd if k.startswith("model_g.")}
    taken: set[str] = set()

    for name, t in onnx_tensors(export).items():
        if name in gen_keys:
            want = sd["model_g." + name].shape
            if t.shape != want:
                sys.exit(f"shape mismatch {name}: export {tuple(t.shape)} vs ckpt {tuple(want)}")
            sd["model_g." + name] = t.to(sd["model_g." + name].dtype)
            taken.add(name)
        elif name.endswith(".weight") and name[:-6] + "weight_g" in gen_keys:
            b = "model_g." + name[:-6]
            if t.shape != sd[b + "weight_v"].shape:
                sys.exit(f"shape mismatch {name}")
            sd[b + "weight_g"], sd[b + "weight_v"] = split_weight_norm(t, sd[b + "weight_g"].shape)
            taken |= {name[:-6] + "weight_g", name[:-6] + "weight_v"}
        else:
            sys.exit(f"export tensor with no home in the checkpoint: {name} {tuple(t.shape)}")

    kept = sorted(gen_keys - taken)
    by_module: dict[str, int] = {}
    for k in kept:
        parts = k.split(".")
        mod = parts[0] if parts[0] != "dp" else ".".join(parts[:2])
        by_module[mod] = by_module.get(mod, 0) + 1
    print(f"generator: {len(taken)} tensors from epoch 702, {len(kept)} from epoch 564: {by_module}")
    print(f"discriminator: {sum(k.startswith('model_d.') for k in sd)} tensors from epoch 564")

    hp = dict(ck["hyper_parameters"])
    hp["warmstart_ckpt"] = None
    hp["vocoder_warmstart_ckpt"] = None
    torch.save(
        {
            "state_dict": sd,
            "hyper_parameters": hp,
            "rebuilt_from": {"export": export, "training_only_parts": base, "epoch": 702},
        },
        out,
    )
    print("wrote", out)


if __name__ == "__main__":
    main(*sys.argv[1:4])
