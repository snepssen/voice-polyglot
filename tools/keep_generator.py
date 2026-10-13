"""Save the speaking half of a training checkpoint: the generator's weights
and the hyperparameters, nothing the trainer needs to continue.

    python keep_generator.py CHECKPOINT OUT.pt

The trainer keeps only the five checkpoints with the lowest val_mel and
deletes the rest. Once val_mel flattens, which five survive is close to
arbitrary, and the epoch that sounds best can be the one that goes. These
copies are what lets an epoch be chosen by ear later; checkin.py and
export read them like a full checkpoint.
"""
import pathlib
import sys

import torch

torch.serialization.add_safe_globals([(pathlib.PosixPath, "pathlib.PosixPath")])
ck = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
torch.save({"state_dict": {k: v for k, v in ck["state_dict"].items() if k.startswith("model_g.")},
            "hyper_parameters": ck["hyper_parameters"], "epoch": ck.get("epoch")}, sys.argv[2])
