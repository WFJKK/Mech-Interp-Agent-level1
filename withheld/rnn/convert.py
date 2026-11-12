"""Make the networks of the RNN models rnn-a to rnn-f from ARC's AlgZoo. Withheld from the agent, like everything here.

    python withheld/rnn/convert.py              writes the networks that are missing, checks the others
    python withheld/rnn/convert.py --from DIR   the same from another copy of the files

AlgZoo (github.com/alignment-research-center/alg-zoo, commit 4433f50, licence MIT-0) trains tiny networks on
algorithmic tasks and publishes them uninterpreted. Its 2nd argmax family is one-layer ReLU RNNs without biases
(alg_zoo/architectures.py, DistRNN), trained to output the position of the second-largest of n numbers, each
drawn from the standard normal distribution (alg_zoo/tasks.py, task_2nd_argmax: x.argsort(-1)[..., -2]).
pth/ holds the state-dict files of its public folder gs://arc-ml-public/alg/zoo for three of its sizes, named
2nd_argmax_<hidden>_<n>_<training sequences>_<seed>.pth, as fetched on 12 November 2026 by

    for c in 2_2 4_3 16_10; do for n in 1073741824 0; do for s in 0 1 2 3 4; do
      curl -sfO "https://storage.googleapis.com/arc-ml-public/alg/zoo/2nd_argmax_${c}_${n}_${s}.pth"; done; done; done

The files are pickles. They are read without torch, by an unpickler that accepts an ordered dict and plain
tensors over storages and refuses anything else, so nothing in them is ever run. Each model gets the five
seeds, under the level-1 names input (rnn.weight_ih_l0), recurrent (rnn.weight_hh_l0) and readout
(linear.weight), float32 as published. The fully trained ones (2^30 training sequences) make rnn-a, rnn-b and
rnn-c; the same seeds before any training (0 sequences: AlgZoo saves the model as it starts) make their
baseline models rnn-d, rnn-e and rnn-f.

A network's file is written once. When it exists, its weights are compared with the conversion and left as they
are: models/<name>/config.yaml holds its sha256, and the tests convert again and compare.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import pickle
import zipfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
PTH = HERE / "pth"
TRAINED, UNTRAINED = 2**30, 0
MODELS = {  # level-1 name: (hidden units, numbers per input, training sequences)
    "rnn-a": (2, 2, TRAINED), "rnn-b": (4, 3, TRAINED), "rnn-c": (16, 10, TRAINED),
    "rnn-d": (2, 2, UNTRAINED), "rnn-e": (4, 3, UNTRAINED), "rnn-f": (16, 10, UNTRAINED),
}
SEEDS = range(5)
NAMES = {"rnn.weight_ih_l0": "input", "rnn.weight_hh_l0": "recurrent", "linear.weight": "readout"}
DTYPES = {"FloatStorage": np.float32, "DoubleStorage": np.float64, "HalfStorage": np.float16, "LongStorage": np.int64,
          "IntStorage": np.int32}


def load_state_dict(path) -> dict:
    """A PyTorch state-dict file (the zip format) as numpy arrays, without torch and without running its pickle."""
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        pkl = [n for n in names if n.endswith("/data.pkl")]
        if len(pkl) != 1:
            raise ValueError(f"{path}: not a single state-dict file")
        prefix = pkl[0][: -len("data.pkl")]
        order = z.read(prefix + "byteorder").decode().strip() if prefix + "byteorder" in names else "little"
        endian = "<" if order == "little" else ">"

        def rebuild(storage, offset, size, stride, requires_grad=False, hooks=None, metadata=None):
            if not isinstance(storage, np.ndarray):
                raise pickle.UnpicklingError("a tensor without a storage")
            offset, size, stride = int(offset), tuple(int(n) for n in size), tuple(int(s) for s in stride)
            if len(size) != len(stride) or offset < 0 or any(n < 0 for n in size) or any(s < 0 for s in stride):
                raise ValueError("a tensor's shape does not fit its strides")
            if 0 in size:
                return np.zeros(size, storage.dtype)
            if offset + sum((n - 1) * s for n, s in zip(size, stride)) >= storage.size:
                raise ValueError("a tensor reaches outside its storage")
            view = np.lib.stride_tricks.as_strided(storage[offset:], shape=size, strides=tuple(s * storage.itemsize for s in stride))
            return np.array(view, copy=True)

        class Restricted(pickle.Unpickler):
            def find_class(self, module, name):
                if (module, name) == ("collections", "OrderedDict"):
                    return collections.OrderedDict
                if (module, name) == ("torch._utils", "_rebuild_tensor_v2"):
                    return rebuild
                if module == "torch" and name in DTYPES:
                    return ("storage type", name)
                raise pickle.UnpicklingError(f"refused to load {module}.{name}")

            def persistent_load(self, pid):
                if not (isinstance(pid, tuple) and len(pid) == 5 and pid[0] == "storage" and isinstance(pid[1], tuple)):
                    raise pickle.UnpicklingError(f"unexpected persistent id {pid!r}")
                _, (_, kind), key, _where, numel = pid
                data = np.frombuffer(z.read(f"{prefix}data/{key}"), dtype=np.dtype(DTYPES[kind]).newbyteorder(endian))
                if data.size != numel:
                    raise ValueError(f"storage {key} holds {data.size} numbers, not {numel}")
                return data

        out = Restricted(z.open(pkl[0])).load()
    if not isinstance(out, dict) or not all(isinstance(v, np.ndarray) for v in out.values()):
        raise ValueError(f"{path}: not a state dict of tensors")
    return {k: v.astype(v.dtype.newbyteorder("=")) for k, v in out.items()}


def source(name: str, seed: int, folder: Path = PTH) -> Path:
    d, n, sequences = MODELS[name]
    return Path(folder) / f"2nd_argmax_{d}_{n}_{sequences}_{seed}.pth"


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def convert(name: str, seed: int, folder: Path = PTH) -> dict:
    """The network's weights under the level-1 names, as they are stored."""
    d, n, _ = MODELS[name]
    path = source(name, seed, folder)
    sd = load_state_dict(path)
    if set(sd) != set(NAMES):
        raise ValueError(f"{path.name}: unexpected tensors {sorted(sd)}")
    w = {NAMES[k]: v for k, v in sd.items()}
    shapes = {k: v.shape for k, v in w.items()}
    if shapes != {"input": (d, 1), "recurrent": (d, d), "readout": (n, d)} or any(v.dtype != np.float32 for v in w.values()):
        raise ValueError(f"{path.name}: unexpected shapes or types {shapes}")
    return w


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from", dest="folder", default=str(PTH), help="the folder holding the .pth files (default: pth/ here)")
    args = ap.parse_args(argv)
    for name in MODELS:
        zoo = REPO / "models" / name / "zoo"
        zoo.mkdir(parents=True, exist_ok=True)
        for seed in SEEDS:
            w = convert(name, seed, Path(args.folder))
            out = zoo / f"{name}-s{seed}.npz"
            if out.exists():
                with np.load(out, allow_pickle=False) as z:
                    same = set(z.files) == set(w) and all(z[k].dtype == w[k].dtype and np.array_equal(z[k], w[k]) for k in w)
                if not same:
                    raise SystemExit(f"{out.relative_to(REPO)} differs from its conversion; it is left as it is")
                what = "checked"
            else:
                np.savez(out, **w)
                what = "written"
            print(f"{out.relative_to(REPO)} {what}  sha256 {sha256(out)}  from {source(name, seed, Path(args.folder)).name} "
                  f"sha256 {sha256(source(name, seed, Path(args.folder)))}")


if __name__ == "__main__":
    main()
