"""Piper's trainer, with what it needs on this Mac and the guards it lacks.

    python train.py new    RUN_DIR [lightning args...]   start a run, warm-started
    python train.py resume RUN_DIR [lightning args...]   continue one from last.ckpt

Fixes applied before piper.train is imported:

  - Piper 1.8.0's wheel ships monotonic_align without its compiled core
    (core.pyx is missing, so `from .monotonic_align.core import ...` fails).
    The routine -- VITS's monotonic alignment search -- is provided here,
    compiled with numba, as the module that import expects.
  - Checkpoints pickled under older torch hold a pathlib.PosixPath in their
    hyperparameters; torch's weights-only loader refuses it unless allowed by
    name (tools-python/piper-data/train.py found that the class alone is not
    enough on recent Python).
  - The val_mos checkpoint callback crashes Lightning 2.6 when no MOS
    predictor is loaded (also found in the old wrapper); it is dropped, and
    --model.mos_metric none is passed so none is loaded.

Guards, because both mistakes have cost a run before:

  - `new` refuses a RUN_DIR that already holds checkpoints: the warm start
    would copy the base voice over them.
  - `resume` neutralises the warm start for this process. VitsModel saves
    warmstart_ckpt into every checkpoint, so a plain resume rebuilds the model
    with it attached and on_fit_start copies the base voice over the restored
    weights while the epoch counter carries on (diagnosed on the Steam Deck).
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import piper_fixes  # noqa: E402,F401

import piper.train.__main__ as piper_main  # noqa: E402
from piper.train.vits.lightning import VitsModel  # noqa: E402

piper_main._DEFAULT_CALLBACKS = [
    cb for cb in piper_main._DEFAULT_CALLBACKS if getattr(cb, "monitor", None) != "val_mos"
]


def main() -> None:
    mode, run_dir, rest = sys.argv[1], pathlib.Path(sys.argv[2]), sys.argv[3:]
    existing = sorted(run_dir.rglob("*.ckpt")) if run_dir.exists() else []
    if mode == "new":
        if existing:
            sys.exit(f"refusing: {run_dir} already holds checkpoints; the warm start "
                     f"would copy the base voice over them. Use `resume`.")
        args = rest
    elif mode == "resume":
        last = sorted(run_dir.rglob("last.ckpt"), key=lambda p: p.stat().st_mtime)
        if not last:
            sys.exit(f"nothing to resume in {run_dir}")
        original = VitsModel.on_fit_start

        def on_fit_start_without_warmstart(self):
            if getattr(self, "_warmstart_ckpt", None) is not None:
                print(f"[resume] ignoring warmstart_ckpt from hyperparameters: {self._warmstart_ckpt}", flush=True)
            self._warmstart_ckpt = None
            self._vocoder_warmstart_ckpt = None
            original(self)

        VitsModel.on_fit_start = on_fit_start_without_warmstart
        args = rest + ["--ckpt_path", str(last[-1])]
        print(f"[resume] from {last[-1]}", flush=True)
    else:
        sys.exit("usage: train.py new|resume RUN_DIR [lightning args...]")
    sys.argv = [sys.argv[0], "fit", "--trainer.default_root_dir", str(run_dir),
                "--model.mos_metric", "none"] + args
    piper_main.main()


if __name__ == "__main__":
    main()
