"""What Piper 1.8.0 needs before its training code can be imported on this
Mac. Imported first by train.py and export_voice.py; importing it is the
whole effect.

  - Its wheel ships monotonic_align without the compiled core (core.pyx is
    missing, so `from .monotonic_align.core import ...` fails). VITS's
    monotonic alignment search is provided here, compiled with numba, as the
    module that import expects. (No numba disk cache: a cached function
    pickles the importing module's name, which differs between processes.)
  - Checkpoints pickled under older torch hold a pathlib.PosixPath in their
    hyperparameters; torch's weights-only loader refuses it unless allowed by
    name (https://github.com/snepssen/tools-core/blob/main/piper-workflows/train.py
    found that the class alone is not enough on recent Python).
"""

import pathlib
import sys
import types

import numba
import torch.serialization


@numba.njit
def _maximum_path_each(path, value, t_y, t_x):
    max_neg_val = -1e9
    for y in range(t_y):
        for x in range(max(0, t_x + y - t_y), min(t_x, y + 1)):
            v_cur = max_neg_val if x == y else value[y - 1, x]
            if x == 0:
                v_prev = 0.0 if y == 0 else max_neg_val
            else:
                v_prev = value[y - 1, x - 1]
            value[y, x] += max(v_prev, v_cur)
    index = t_x - 1
    for y in range(t_y - 1, -1, -1):
        path[y, index] = 1
        if index != 0 and (index == y or value[y - 1, index] < value[y - 1, index - 1]):
            index -= 1


@numba.njit
def maximum_path_c(paths, values, t_ys, t_xs):
    for i in range(paths.shape[0]):
        _maximum_path_each(paths[i], values[i], t_ys[i], t_xs[i])


_core = types.ModuleType("piper.train.vits.monotonic_align.monotonic_align.core")
_core.maximum_path_c = maximum_path_c
_pkg = types.ModuleType("piper.train.vits.monotonic_align.monotonic_align")
_pkg.core = _core
sys.modules[_pkg.__name__] = _pkg
sys.modules[_core.__name__] = _core

torch.serialization.add_safe_globals([(pathlib.PosixPath, "pathlib.PosixPath")])
