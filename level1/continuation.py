"""A run that goes on from a finished single-agent run (run config: continue_from; docs/CONTINUATION.md).

At the earlier price a wrong answer may cost too little for a fix to pay, so the last errors of the earlier rounds
stay unexplained. The new run asks the same network again at a higher price, set from the entry the earlier run
locked, as measured again on fresh inputs (or on every input, when every input is checked):

    alpha = margin * (c_max - C) / e,  rounded up, and never below the earlier price

C is that entry's cost over all inputs (an input outside its region costs a forward pass) and e the share of all
inputs it answers wrongly. Removing every one of its errors then pays, by the factor margin, even if it raises the
cost to c_max (1: a forward pass on every input). When the verifier samples, the sample grows with the price to
inputs_per_alpha * alpha inputs (at most max_sample_size), so that an entry right on every input drawn loses
little to the confidence bound.

The earlier run is read and never written. With memory, this run's rounds are numbered after the earlier ones, and
its prompt carries their table (each reward recomputed at the new price), their notes, their scored entries and a
note on what changed. Without memory the run gets the same price and sample and nothing else: the control that
shows what the memory is worth. Nothing here depends on the model, so any single-agent run can be continued.
"""

from __future__ import annotations

import copy
import math
from pathlib import Path

from . import lang, rewards
from .util import read_json


def options(cfg: dict) -> dict:
    return cfg.get("continuation") or {}


def derived(cfg: dict) -> dict | None:
    """What the run's start worked out from the earlier run; None for a run that goes on from nothing."""
    return options(cfg).get("derived")


def first_round(cfg: dict) -> int:
    """The number of the run's first round: 1, or one past the earlier rounds it carries."""
    return int((derived(cfg) or {}).get("first_round", 1))


def _runs_dir(cfg: dict) -> Path:
    return Path(str(cfg["runs_dir"])).expanduser().resolve()


def _rounds_of(path: Path) -> list:
    from .loop import completed  # the loop writes the records read here

    return completed(path)


def prepare(cfg: dict) -> dict:
    """The run config of a run that goes on from cfg['continue_from'], its price, sample and first round set from
    the earlier run's records. A config without continue_from comes back unchanged."""
    earlier_id = cfg.get("continue_from")
    if not earlier_id:
        return cfg
    opts = options(cfg)
    if (cfg.get("orchestration") or {}).get("enabled"):
        raise RuntimeError("continue_from goes on from a single agent's run in a single agent's run")
    if cfg.get("exact_on_any_weights"):
        raise RuntimeError("continue_from is for runs on trained networks, not for a baseline's exact entries")
    if str(earlier_id) == str(cfg["run_id"]):
        raise RuntimeError("a run cannot go on from itself")
    earlier = _runs_dir(cfg) / str(earlier_id)
    if not (earlier / "run.json").is_file():
        raise RuntimeError(f"there is no run {earlier_id} in {_runs_dir(cfg)} to go on from")
    run = read_json(earlier / "run.json")
    ecfg = run["config"]
    if (ecfg.get("orchestration") or {}).get("enabled"):
        raise RuntimeError(f"{earlier_id} is a team run; continue_from goes on from a single agent's run")
    for key in ("model", "network"):
        if ecfg[key] != cfg[key]:
            raise RuntimeError(f"{earlier_id} ran on {key} {ecfg[key]}, and this run is on {cfg[key]}")
    if run.get("prices") != lang.PRICES:
        raise RuntimeError(f"{earlier_id} was measured under {run.get('prices')}, and this code prices under {lang.PRICES}")
    if ecfg["reward"].get("name") != cfg["reward"].get("name") or "alpha" not in ecfg["reward"]:
        raise RuntimeError(f"{earlier_id} was scored with another reward, or one without a price for a wrong answer")
    if not (earlier / "lock.json").is_file():
        raise RuntimeError(f"{earlier_id} is not finished: it has no lock.json")
    lock = read_json(earlier / "lock.json")
    if not lock.get("locked"):
        raise RuntimeError(f"{earlier_id} locked no entry, so there is nothing to set a price from")
    measured = lock.get("measured_again") or lock
    parts = measured["parts"]
    t = rewards.terms(parts, parts["fidelity"])
    saved, errors = t["saved"], t["errors"]
    cost = 1.0 - saved
    c_max = float(opts["c_max"])
    if errors <= 0:
        raise RuntimeError(f"the entry {earlier_id} locked is right on every input it was measured on: there is no "
                           "error to price higher")
    if cost >= c_max:
        raise RuntimeError(f"the entry {earlier_id} locked costs {cost:.4f} over all inputs, not less than c_max {c_max:g}")
    alpha_before = float(ecfg["reward"]["alpha"])
    by_rule = math.ceil(float(opts["margin"]) * (c_max - cost) / errors)
    alpha = float(opts["alpha"]) if opts.get("alpha") is not None else float(max(alpha_before, by_rule))
    out = copy.deepcopy(cfg)
    out["reward"] = {**cfg["reward"], "alpha": alpha}
    sampled = not run["enumerated"]
    sample_before = int(ecfg["verifier"]["sample_size"])
    sample = sample_before
    if sampled:
        sample = max(sample_before, min(math.ceil(float(opts["inputs_per_alpha"]) * alpha), int(opts["max_sample_size"])))
        out["verifier"] = {**cfg["verifier"], "sample_size": sample}
    memory = bool(opts["memory"])
    before = derived(ecfg)
    chain = list(before["runs"]) if before and options(ecfg).get("memory", True) else []
    runs = chain + [str(earlier_id)] if memory else []
    prices = []
    for rid in runs:
        rcfg = read_json(_runs_dir(cfg) / rid / "run.json")["config"]
        first = first_round(rcfg)
        own = len(_rounds_of(_runs_dir(cfg) / rid))
        if own:
            prices.append({"run": rid, "rounds": [first, first + own - 1], "alpha": float(rcfg["reward"]["alpha"])})
    last = prices[-1]["rounds"][1] if prices else 0
    out["continuation"] = {**opts, "derived": {
        "from": str(earlier_id), "runs": runs, "first_round": last + 1 if memory else 1, "prices": prices,
        "locked": {"carried_as": lock.get("carried_as"), "name": lock.get("name"), "entry_sha256": lock["entry_sha256"],
                   "measured_again": bool(lock.get("measured_again")), "coverage": parts["coverage"],
                   "fidelity": parts["fidelity"], "cost": rewards.cost_of(parts), "saved": saved, "errors": errors,
                   "cost_over_all_inputs": cost, "reward_then": measured["reward"],
                   "reward_now": rewards.compute(parts, out["reward"], lock.get("claimed_fidelity"))["reward"]},
        "alpha_before": alpha_before, "alpha_by_rule": by_rule, "alpha": alpha,
        "sampled": sampled, "sample_before": sample_before, "sample": sample,
        "exact_charge": alpha * (1.0 - lang.wilson_lower(sample, sample)) if sampled else 0.0,
    }}
    return out


def earlier_rounds(cfg: dict) -> list:
    """The rounds of the runs this one goes on from, as its agent sees them: every scored submission's reward
    recomputed under this run's reward (the recorded one kept as reward_recorded), and its run noted, so that its
    entry is read from there. Empty for a run that goes on from nothing, and for a control."""
    d = derived(cfg)
    if not d or not d["runs"]:
        return []
    out = []
    for rid in d["runs"]:
        where = _runs_dir(cfg) / rid
        for rd in _rounds_of(where):
            rd = copy.deepcopy(rd)
            for s in rd["submissions"]:
                s["from_run"] = str(where)
                if s["status"] == "scored":
                    s["reward_recorded"] = s["reward"]
                    s["reward"] = rewards.compute(s["parts"], cfg["reward"], s.get("claimed_fidelity"))["reward"]
            out.append(rd)
    return out


def understood_share(cfg: dict, parts: dict | None) -> float | None:
    """(1 - C_ex) / (1 - C): the share of the earlier locked entry's saving that an entry right on every input
    measured keeps. None unless the entry was right on every input measured."""
    d = derived(cfg)
    if not d or not parts or parts.get("fidelity") != 1.0:
        return None
    return rewards.terms(parts, 1.0)["saved"] / d["locked"]["saved"]


def _rounds(a: int, b: int) -> str:
    return f"round {a}" if a == b else f"rounds {a} to {b}"


def price_history(d: dict) -> str:
    return " and ".join(f"alpha = {p['alpha']:g} ({_rounds(*p['rounds'])})" for p in d["prices"])
