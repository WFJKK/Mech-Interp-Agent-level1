"""Draw the networks of models/att-e. Withheld from the agent, like everything here.

    python withheld/att-e/draw.py

att-e's networks are never trained. Each is the state att-d's trainer starts from (withheld/att-d/train.py,
init: every matrix uniform in +-1/sqrt(shape[0]), float32, the proof pipeline's initialization for att-c's
networks too), drawn from numpy.random.default_rng(seed) with the seeds in SEEDS, and kept as drawn; embed then
gets one more row, all zeros, for an empty slot, as for att-c and att-d. The seeds are not att-d's (0 and 1).

Writes models/att-e/zoo/<network>.npz. models/att-e/config.yaml holds their sha256, and the tests draw them
again and compare.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
NAME, TRAINER = "att-e", "att-d"
SEEDS = {f"{NAME}-s{i}": 3000 + i for i in range(6)}


def _trainer():
    spec = importlib.util.spec_from_file_location(f"{TRAINER.replace('-', '_')}_train", REPO / "withheld" / TRAINER / "train.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def draw(network: str) -> dict:
    """The network's weights under the level-1 names, as they are stored."""
    tr = _trainer()
    P = tr.init(np.random.default_rng(SEEDS[network]))
    out = {tr.LEVEL1[k]: P[k] for k in tr.NAMES}
    out["embed"] = np.concatenate([P["E"], np.zeros((1, P["E"].shape[1]), P["E"].dtype)], axis=0)  # an empty slot
    return out


def main():
    for network in SEEDS:
        out = REPO / "models" / NAME / "zoo" / f"{network}.npz"
        np.savez(out, **draw(network))
        print(out.relative_to(REPO))


if __name__ == "__main__":
    main()
