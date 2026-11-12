"""Draw the networks of models/lin-f. Withheld from the agent, like everything here.

    python withheld/lin-f/draw.py

lin-f's networks are never trained. Each is the state lin-d's trainer starts from (withheld/lin-d/train.py,
init: uniform in +-1/sqrt(fan_in), float32), drawn from numpy.random.default_rng(seed) with the seeds in SEEDS,
and kept as drawn; embed then gets one more column, all zeros, for an empty slot, as for lin-d. The seeds are
not lin-d's (0 and 1), so no network here is lin-d's own starting point.

Writes models/lin-f/zoo/<network>.npz. models/lin-f/config.yaml holds their sha256, and the tests draw them
again and compare.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
NAME, TRAINER = "lin-f", "lin-d"
SEEDS = {f"{NAME}-s{i}": 2000 + i for i in range(6)}


def _trainer():
    spec = importlib.util.spec_from_file_location(f"{TRAINER.replace('-', '_')}_train", REPO / "withheld" / TRAINER / "train.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def draw(network: str) -> dict:
    """The network's weights under the level-1 names, as they are stored."""
    P = _trainer().init(np.random.default_rng(SEEDS[network]))
    embed = np.concatenate([P["embed"], np.zeros((P["embed"].shape[0], 1), np.float32)], axis=1)
    return {"embed": embed, "hidden": P["hidden"], "unembed": P["unembed"]}


def main():
    for network in SEEDS:
        out = REPO / "models" / NAME / "zoo" / f"{network}.npz"
        np.savez(out, **draw(network))
        print(out.relative_to(REPO))


if __name__ == "__main__":
    main()
