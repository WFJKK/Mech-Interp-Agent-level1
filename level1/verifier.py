"""The verifier: measures an entry on a network, on inputs of its own choosing.

    python -m level1.verifier --model models/lin-a --network lin-a-s0 --entry entry.json [--alpha 10]

It reads an entry as JSON data and interprets it (lang.py); nothing of an entry is ever executed as code.
A data set small enough is checked on every input; above verifier.enumerate_up_to a fresh uniform sample is
drawn for each scoring, from a seed nobody chose, and the fidelity's 95% lower bound (Wilson) is reported
next to the fidelity. The network's own outputs come from the network written as an entry (model.py), run
through the same interpreter, so a forward pass and an entry are priced by one list.

score() returns a receipt: status scored or refused (with the reason), the entry's hash and size, the inputs
used, the parts measured (rewards.py explains them) and the reward under the run's reward function.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np

from . import lang, rewards
from .networks import Model, load_model

DEFAULTS = {"enumerate_up_to": 200_000, "sample_size": 4000, "time_limit_s": 120.0}
ENTRY_DEFAULTS = {"max_nodes": 400, "max_const_scalars": 256}


def wilson_lower(k: int, n: int, z: float = 1.96) -> float:
    """The lower end of the Wilson interval for k successes in n draws."""
    if n == 0:
        return 0.0
    p = k / n
    centre = p + z * z / (2 * n)
    spread = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, (centre - spread) / (1 + z * z / n))


class Verifier:
    def __init__(self, model: Model, network_id: str, cfg: dict | None = None):
        cfg = cfg or {}
        self.model, self.network_id = model, network_id
        self.cfg = {**DEFAULTS, **(cfg.get("verifier") or {})}
        self.limits = {**ENTRY_DEFAULTS, **(cfg.get("entry") or {})}
        self.reward_cfg = dict(cfg.get("reward") or {"name": "r_alpha", "alpha": 10.0})
        self.weights = model.load_weights(network_id)
        self.reference = model.reference_entry(self.weights)
        self.enumerated = model.n_total <= int(self.cfg["enumerate_up_to"])
        self._all = None  # (inputs, the network's outputs, operations of one forward pass), when enumerated

    def inputs(self, seed: int | None = None) -> tuple:
        """(inputs, the network's outputs on them, operations of a forward pass, how they were chosen)"""
        if self.enumerated:
            if self._all is None:
                x = lang.all_inputs(self.model.input_space)
                answers, ops = lang.run_reference(self.reference, self.weights, self.model.n_ctx, x)
                self._all = (x, answers, ops)
            return (*self._all, {"mode": "all", "n": self.model.n_total, "seed": None})
        if seed is None:
            seed = int.from_bytes(os.urandom(8), "big")
        rng = np.random.default_rng(seed)
        n = int(self.cfg["sample_size"])
        x = np.stack([rng.integers(0, k, size=n) for k in self.model.input_space], axis=1).astype(np.int64)
        answers, ops = lang.run_reference(self.reference, self.weights, self.model.n_ctx, x)
        return x, answers, ops, {"mode": "sample", "n": n, "seed": seed}

    def score(self, entry, claimed: float | None = None, seed: int | None = None) -> dict:
        t0 = time.monotonic()
        receipt = {"status": "refused", "reason": None, "network": self.network_id, "entry_sha256": None, "size": None,
                   "inputs": None, "parts": None, "reward": None, "prices": lang.PRICES}
        try:
            receipt["size"] = lang.check_structure(entry, self.limits["max_nodes"], self.limits["max_const_scalars"])
            receipt["entry_sha256"] = lang.entry_hash(entry)
            x, answers, ops, how = self.inputs(seed)
            receipt["inputs"] = how
            parts = lang.measure(entry, self.weights, self.model.n_ctx, x, answers, ops, self.model.n_total,
                                 deadline=time.monotonic() + float(self.cfg["time_limit_s"]))
        except lang.EntryError as exc:
            receipt["reason"] = str(exc)[:1000]
            receipt["seconds"] = round(time.monotonic() - t0, 3)
            return receipt
        except Exception as exc:  # not an entry the interpreter could even fail on properly: refused, and marked
            # The reason goes back to the agent, so it names the kind of failure only; the detail stays in the receipt.
            receipt["reason"] = f"the verifier could not read or run this entry ({type(exc).__name__})"
            receipt["verifier_error"] = f"{type(exc).__name__}: {str(exc)[:500]}"
            receipt["seconds"] = round(time.monotonic() - t0, 3)
            return receipt
        if how["mode"] == "sample" and parts["n_in_region"]:
            parts["fidelity_low"] = wilson_lower(parts["n_agree"], parts["n_in_region"])
        else:
            parts["fidelity_low"] = parts["fidelity"]
        r = rewards.compute(parts, self.reward_cfg, claimed)
        receipt.update(status="scored", parts=parts, reward=r["reward"],
                       reward_detail={k: v for k, v in r.items() if k != "reward"})
        if claimed is not None:
            receipt["claimed_fidelity"] = float(claimed)
            if parts["fidelity"] is not None:
                receipt["overclaim"] = max(0.0, float(claimed) - parts["fidelity"])
        receipt["seconds"] = round(time.monotonic() - t0, 3)
        return receipt


def describe(receipt: dict) -> str:
    """One line for logs and reports."""
    if receipt["status"] != "scored":
        return f"refused: {receipt['reason']}"
    p = receipt["parts"]
    fid = "none" if p["fidelity"] is None else f"{p['fidelity']:.5f}"
    r = "none" if p["r"] is None else f"{p['r']:.4f}"
    return (f"reward {receipt['reward']:.4f}  coverage {p['coverage']:.4f}  fidelity {fid}  r {r}  "
            f"r_rule {p['r_rule']:.4f}  b {p['b']:.4f}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--network", required=True)
    ap.add_argument("--entry", required=True, help="an entry as a JSON file")
    ap.add_argument("--alpha", type=float, default=10.0)
    args = ap.parse_args(argv)
    v = Verifier(load_model(args.model), args.network, {"reward": {"name": "r_alpha", "alpha": args.alpha}})
    with open(args.entry) as f:
        receipt = v.score(json.load(f))
    print(json.dumps(receipt, indent=1))
    return 0 if receipt["status"] == "scored" else 1


if __name__ == "__main__":
    sys.exit(main())
