"""Reward functions: one number from what the verifier measured. A run config picks one by name.

The verifier's parts (verifier.py) are kept with every receipt, so a run can be scored again under another
reward without running anything (python -m level1.loop rescore), and the reference entries of a model can
be scored under any reward with no run at all (python -m level1.report --model ...).

    coverage   mu     the share of the inputs in the entry's region S
    fidelity   q      the share of S on which the interpretation gives the network's output
    cost       c      everything the entry computes (its interpretation on S, its region test on every input,
                      its build once), per input of S, over one forward pass

The cost has three parts, kept in every receipt:

    r                 operations of the interpretation per input of S, over one forward pass
    r_rule            operations of the region test per input, over one forward pass
    b                 operations of the build, over one forward pass on every input scored

so that c = r + (r_rule + b) / mu. Use the entry on S and the network elsewhere, and compare with running
the network everywhere: mu * (1 - c) = mu * (1 - r) - r_rule - b is the share of the work saved, and
mu * (1 - q) is the share of inputs now answered wrongly. The network itself, as an entry, saves nothing and
errs nowhere: every reward here gives it 0.

A new reward is a function here, taking the parts (and the claimed fidelity) and its own parameters by
name, with an entry in REWARDS and its two texts for the agent's prompt in FORMULAS and EXPLANATIONS.
This file is also placed in the agent's sandbox, so nothing in it may say what a model does.
"""

from __future__ import annotations

import inspect


def cost_of(parts: dict | None) -> float | None:
    """The entry's one cost: its whole work per input of its region, over one forward pass. None for an
    empty region."""
    if not parts or not parts.get("coverage"):
        return None
    if parts.get("cost") is not None:
        return parts["cost"]
    return parts["r"] + (parts["r_rule"] + parts["b"]) / parts["coverage"]


def terms(parts: dict, fidelity: float | None) -> dict:
    mu = parts["coverage"]
    r = parts["r"] if parts["r"] is not None else 0.0
    p = fidelity if fidelity is not None else 1.0  # an empty region answers nothing wrongly
    return {"saved": mu * (1.0 - r) - parts["r_rule"] - parts["b"], "errors": mu * (1.0 - p)}


def r_alpha(parts: dict, alpha: float = 10.0, lower_bound: bool = True, **_) -> dict:
    """saved - alpha * errors: one wrong answer is priced at alpha forward passes.

    On a sample, the fidelity's lower confidence bound is used unless lower_bound is false, so that luck in
    the sample does not pay."""
    p = parts.get("fidelity_low") if lower_bound and parts.get("fidelity_low") is not None else parts["fidelity"]
    t = terms(parts, p)
    return {"reward": t["saved"] - float(alpha) * t["errors"], "saved": t["saved"], "errors": t["errors"], "fidelity_used": p}


def r_alpha_claimed(parts: dict, alpha: float = 10.0, penalty: float = 2.0, claimed: float | None = None, **_) -> dict:
    """r_alpha with the claimed fidelity in place of the measured one, as the levels were first described:
    credit for the claim when the measurement confirms it; otherwise the measured fidelity, less penalty
    times the overclaim. Without a claim it is r_alpha."""
    p = parts["fidelity"]
    if claimed is None or p is None:
        return r_alpha(parts, alpha)
    credited = float(claimed) if p >= float(claimed) else p - float(penalty) * (float(claimed) - p)
    t = terms(parts, credited)
    return {"reward": t["saved"] - float(alpha) * t["errors"], "saved": t["saved"], "errors": t["errors"], "fidelity_used": credited}


NUMBER_DIGITS = 4  # as lang.NUMBER_DIGITS, which does the counting (this file stands alone in the sandbox)


def r_alpha_numbers(parts: dict, alpha: float = 10.0, price: float = 0.002, lower_bound: bool = True, **_) -> dict:
    """r_alpha less a price for every number the entry stores beyond those of the network's own definition:
    saved - alpha * errors - price * numbers.

    Reading a number from a table costs one operation however long the table, so r_alpha cannot tell an entry
    that stores what was observed of the network's outputs from one that computes its answers. Here every stored
    number has to earn its price. The verifier counts them (lang.stored_numbers: whatever is the same on every
    input and for every weight, however it is written) for the entry and for the network's own definition, and
    keeps both in the parts. The network itself stores nothing beyond its definition, so it still scores 0."""
    if parts.get("numbers") is None or parts.get("numbers_network") is None:
        raise ValueError("r_alpha_numbers needs the entry's stored numbers in the parts (numbers, numbers_network)")
    out = r_alpha(parts, alpha, lower_bound)
    extra = max(0, int(parts["numbers"]) - int(parts["numbers_network"]))
    return {**out, "reward": out["reward"] - float(price) * extra, "numbers": extra}


REWARDS = {"r_alpha": r_alpha, "r_alpha_claimed": r_alpha_claimed, "r_alpha_numbers": r_alpha_numbers}
NEEDS_NUMBERS = {"r_alpha_numbers"}  # rewards for which the verifier counts an entry's stored numbers into the parts


def needs_numbers(cfg: dict | None) -> bool:
    return (cfg or {}).get("name", "r_alpha") in NEEDS_NUMBERS


# The reward as the agent's prompt states it, and one or two sentences on how to read it. Parameters in
# braces are filled from the run config (or the function's defaults).
FORMULAS = {
    "r_alpha": "reward = coverage * [ (1 - cost) - alpha * (1 - fidelity) ],   alpha = {alpha:g}",
    "r_alpha_claimed": (
        "reward = coverage * [ (1 - cost) - alpha * (1 - credited) ],   alpha = {alpha:g}\n"
        "    credited = the fidelity you claimed, if the measured fidelity is at least that;\n"
        "               otherwise measured - {penalty:g} * (claimed - measured)"),
    "r_alpha_numbers": ("reward = coverage * [ (1 - cost) - alpha * (1 - fidelity) ] - price * numbers,"
                        "   alpha = {alpha:g}, price = {price:g}"),
}
EXPLANATIONS = {
    "r_alpha": (
        "Read it for one input of the region: using the entry there in place of the network saves\n"
        "(1 - cost) of a forward pass, and with probability (1 - fidelity) its answer is wrong, which is\n"
        "charged at {alpha:g} forward passes. Coverage turns that into a share of the data set; the inputs\n"
        "outside the region are left to the network and earn nothing."),
    "r_alpha_claimed": (
        "Read it for one input of the region: using the entry there in place of the network saves\n"
        "(1 - cost) of a forward pass, and a wrong answer is charged at {alpha:g} forward passes. The share\n"
        "of wrong answers is taken from the fidelity you claimed when the measurement confirms it. A claim\n"
        "above what is measured costs more than it could have gained, and a claim below it gives credit\n"
        "away: claim what you expect to be measured. Coverage turns the result into a share of the data set."),
    "r_alpha_numbers": (
        "Read it for one input of the region: using the entry there in place of the network saves\n"
        "(1 - cost) of a forward pass, and with probability (1 - fidelity) its answer is wrong, which is\n"
        "charged at {alpha:g} forward passes. Coverage turns that into a share of the data set; the inputs\n"
        "outside the region are left to the network and earn nothing.\n"
        "numbers is what the entry stores beyond the network's own definition: every number that is the\n"
        "same on every input and for every weight, however it is written. That is every number in its\n"
        "constants, the length of every arange, every place a slice reads out of a list made of those, and\n"
        "every value it makes from its input or the weights that comes out the same whatever they are.\n"
        "Each counts once per {digits} significant digits it is written with (0.25 once, 0.123456 twice)\n"
        "and is charged {price:g}, so a number is worth storing only if it earns more than that. What the\n"
        "entry computes from the weights is not counted here; it is charged as computation, in the cost."),
}


def parameters(name: str) -> dict:
    """The parameters a reward takes, with their defaults: {name: default}."""
    sig = inspect.signature(REWARDS[name])
    return {k: p.default for k, p in sig.parameters.items() if k not in ("parts", "claimed") and p.kind is not p.VAR_KEYWORD}


def check(cfg: dict) -> None:
    """Refuse a reward block that names no reward or a parameter the reward does not take (a typo would
    otherwise be ignored in silence)."""
    name = cfg.get("name", "r_alpha")
    if name not in REWARDS:
        raise ValueError(f"no reward named {name!r}; there are {sorted(REWARDS)}")
    unknown = sorted(set(cfg) - {"name"} - set(parameters(name)))
    if unknown:
        raise ValueError(f"the reward {name} does not take {unknown}; it takes {sorted(parameters(name))}")


def _filled(cfg: dict) -> tuple:
    check(cfg)
    name = cfg.get("name", "r_alpha")
    return name, {**parameters(name), **{k: v for k, v in cfg.items() if k != "name"}}


def warnings(cfg: dict) -> list:
    """What a reward block invites, in words, for the run's log: settings under which the reward can be
    raised without predicting better or computing less."""
    name, params = _filled(cfg)
    out = []
    if "alpha" in params and float(params["alpha"]) <= 1:
        out.append(f"alpha = {float(params['alpha']):g}: a wrong answer is priced at no more than the forward pass it saves, so "
                   "padding a region with inputs the entry gets wrong does not lower the reward (below 1 it raises it), "
                   "and a free guess that is sometimes right scores above 0")
    if name == "r_alpha" and params.get("lower_bound") is False:
        out.append("lower_bound is off: on a sampled data set, a lucky sample pays, and more submissions mean more luck")
    if name == "r_alpha_claimed" and float(params.get("penalty", 0)) <= 0:
        out.append("penalty is 0 or less: a claim above the measured fidelity costs nothing")
    return out


def formula(cfg: dict) -> str:
    """The reward as a formula, for the agent's prompt."""
    name, params = _filled(cfg)
    return FORMULAS[name].format(**params)


def explanation(cfg: dict) -> str:
    """How to read the formula, for the agent's prompt."""
    name, params = _filled(cfg)
    return EXPLANATIONS[name].format(**params, digits=NUMBER_DIGITS)


def compute(parts: dict, cfg: dict, claimed: float | None = None) -> dict:
    """cfg is the run config's reward block: {name, ...parameters}."""
    check(cfg)
    name = cfg.get("name", "r_alpha")
    params = {k: v for k, v in cfg.items() if k != "name"}
    out = REWARDS[name](parts, claimed=claimed, **params)
    return {"name": name, "params": params, **out}
