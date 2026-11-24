"""The verifier: measures an entry on a network, on inputs of its own choosing.

    python -m level1.verifier --model models/lin-a --network lin-a-s0 --entry entry.json [--alpha 10]

It reads an entry as JSON data and interprets it (lang.py); nothing of an entry is ever executed as code.

Which inputs (config: verifier.check):

    all      every input of the data set, each once. Only for a data set that can be listed.
    sample   a fresh sample of verifier.sample_size inputs, drawn from the data set for each scoring from a
             seed nobody chose. The fidelity's 95% lower bound (Wilson) is reported next to the fidelity, and
             the reward uses it.
    auto     all when the data set can be listed and has at most verifier.enumerate_up_to inputs, else sample.

The network's own outputs come from the network written as an entry (model.py), run through the same
interpreter, so a forward pass and an entry are priced by one list.

score() returns a receipt: status scored or refused (with the reason), the entry's hash and size, the inputs
used, the parts measured (rewards.py explains them) and the reward under the run's reward function.
"""

from __future__ import annotations

import argparse
import json
import sys
import time

from . import lang, regions, rewards
from .lang import wilson_lower  # noqa: F401  (kept under this name for callers)
from .networks import Model, load_model

DEFAULTS = {"check": "auto", "enumerate_up_to": 200_000, "sample_size": 4000, "reference_samples": 8, "time_limit_s": 120.0}
ENTRY_DEFAULTS = {"max_nodes": 400, "max_const_scalars": 256}
CHECKS = ("auto", "all", "sample")


def _check_deadline(deadline: float | None) -> None:
    if deadline is not None and time.monotonic() >= deadline:
        raise lang.EntryError("the verifier ran out of time")


def resolve_check(model: Model, verifier_cfg: dict | None) -> str:
    """all or sample: how this run's verifier checks this model's data set."""
    cfg = {**DEFAULTS, **(verifier_cfg or {})}
    check = cfg.get("check", "auto")
    if check not in CHECKS:
        raise RuntimeError(f"verifier.check must be one of {CHECKS}, not {check!r}")
    if check == "all" and not model.can_enumerate():
        raise RuntimeError(f"the data set of {model.name} cannot be checked on every input: its inputs are drawn, not listed; "
                           "use verifier.check: sample")
    if check == "auto":
        return "all" if model.can_enumerate() and model.n_total <= int(cfg["enumerate_up_to"]) else "sample"
    return check


class Verifier:
    def __init__(self, model: Model, network_id: str, cfg: dict | None = None, *, deadline: float | None = None):
        cfg = cfg or {}
        self.model, self.network_id = model, network_id
        self.cfg = {**DEFAULTS, **(cfg.get("verifier") or {})}
        self.limits = {**ENTRY_DEFAULTS, **(cfg.get("entry") or {})}
        self.reward_cfg = dict(cfg.get("reward") or {"name": "r_alpha", "alpha": 10.0})
        _check_deadline(deadline)
        self.weights = model.load_weights(network_id)
        self.reference = model.reference_entry(self.weights)
        _check_deadline(deadline)
        self.scope = None
        self._scope_meta = None
        if self.cfg.get("scope") is not None:
            self.scope = regions.Scope(model.input_space, model.info()["data_set"], self.weights, model.n_ctx,
                                       self.cfg["scope"], limits=self.limits,
                                       enumerate_up_to=int(self.cfg["enumerate_up_to"]),
                                       deadline=self._capped_deadline(deadline))
            self.check = self.scope.resolve_check(self.cfg)
        else:
            self.check = resolve_check(model, self.cfg)
        self.enumerated = self.check == "all"
        self._all = None  # (inputs, the network's outputs, operations of one forward pass), when every input is checked
        _check_deadline(deadline)

    def _capped_deadline(self, deadline: float | None) -> float:
        configured = time.monotonic() + float(self.cfg["time_limit_s"])
        return configured if deadline is None else min(configured, deadline)

    def n_inputs(self) -> int:
        """How many inputs one scoring uses."""
        if self.enumerated:
            return self.scope.cardinality if self.scope is not None else self.model.n_total
        return int(self.cfg["sample_size"])

    def inputs(self, seed: int | None = None, *, deadline: float | None = None) -> tuple:
        """(inputs, the network's outputs on them, operations of a forward pass, how they were chosen)"""
        # An attempt deadline is local to this call, never retained as a verifier-wide policy.
        deadline = self._capped_deadline(deadline) if deadline is not None else None
        _check_deadline(deadline)
        if self.enumerated and self._all is not None:
            return self._all
        x, how, self._scope_meta = regions.prepare_inputs(self.model.info(), self.weights, self.cfg,
                                                         seed=seed, limits=self.limits, scope=self.scope, deadline=deadline)
        _check_deadline(deadline)
        answers, ops = lang.run_reference(self.reference, self.weights, self.model.n_ctx, x, deadline=deadline)
        _check_deadline(deadline)
        result = (x, answers, ops, how)
        if self.enumerated:
            self._all = result
        return result

    def score(self, entry, claimed: float | None = None, seed: int | None = None, *, deadline: float | None = None) -> dict:
        t0 = time.monotonic()
        deadline = self._capped_deadline(deadline) if deadline is not None else None
        receipt = {"status": "refused", "reason": None, "network": self.network_id, "entry_sha256": None, "size": None,
                   "inputs": None, "parts": None, "reward": None, "prices": lang.PRICES}
        try:
            _check_deadline(deadline)
            receipt["size"] = lang.check_structure(entry, self.limits["max_nodes"], self.limits["max_const_scalars"])
            receipt["entry_sha256"] = lang.entry_hash(entry)
            x, answers, ops, how = self.inputs(seed, deadline=deadline) if deadline is not None else self.inputs(seed)
            receipt["inputs"] = how
            if self._scope_meta is not None:
                receipt["scope"] = self._scope_meta
            parts = regions.measure(entry, self.weights, self.model.n_ctx, x, answers, ops, how=how,
                                    scope=self._scope_meta, verifier_cfg=self.cfg, limits=self.limits,
                                    deadline=self._capped_deadline(deadline))
            _check_deadline(deadline)
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
        if rewards.needs_numbers(self.reward_cfg):  # a reward that prices what an entry stores
            parts["numbers"], parts["numbers_network"] = lang.stored_numbers(entry), lang.stored_numbers(self.reference)
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
    cost = rewards.cost_of(p)
    cost = "none" if cost is None else f"{cost:.4f}"
    pooled = receipt.get("pooled")
    tail = f"  ({pooled['samples']} samples pooled, {pooled['n_inputs']:,} inputs)" if pooled else ""
    return f"reward {receipt['reward']:.4f}  coverage {p['coverage']:.4f}  fidelity {fid}  cost {cost}{tail}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--network", required=True)
    ap.add_argument("--entry", required=True, help="an entry as a JSON file")
    ap.add_argument("--alpha", type=float, default=10.0)
    ap.add_argument("--check", choices=CHECKS, default="auto")
    ap.add_argument("--sample-size", type=int, default=DEFAULTS["sample_size"])
    args = ap.parse_args(argv)
    v = Verifier(load_model(args.model), args.network, {"reward": {"name": "r_alpha", "alpha": args.alpha},
                                                         "verifier": {"check": args.check, "sample_size": args.sample_size}})
    with open(args.entry) as f:
        receipt = v.score(json.load(f))
    print(json.dumps(receipt, indent=1))
    return 0 if receipt["status"] == "scored" else 1


if __name__ == "__main__":
    sys.exit(main())
