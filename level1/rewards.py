"""Reward functions: one number from what the verifier measured. A run config picks one by name.

The verifier's parts (verifier.py) are kept with every receipt, so a run can be scored again under another
reward without running anything (python -m level1.loop rescore), and the reference entries of a model can
be scored under any reward with no run at all (python -m level1.report --model ...).

    coverage   mu     the share of the data set in the entry's region S
    fidelity   p      the share of S on which the entry gives the network's output
    r                 operations of the prediction per input of S, over one forward pass
    r_rule            operations of the rule per input, over one forward pass
    b                 operations of the build, over one forward pass on every input of the data set

Use the entry on S and the network elsewhere, and compare with running the network everywhere:
mu * (1 - r) - r_rule - b is the share of the work saved, and mu * (1 - p) is the share of inputs now
answered wrongly. The network itself, as an entry, saves nothing and errs nowhere: every reward here gives it 0.

A new reward is a function here, taking the parts (and the claimed fidelity) and its own parameters by
name, with an entry in REWARDS and its two texts for the agent's prompt in FORMULAS and EXPLANATIONS.
This file is also placed in the agent's sandbox, so nothing in it may say what a model does.
"""

from __future__ import annotations

import inspect


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


REWARDS = {"r_alpha": r_alpha, "r_alpha_claimed": r_alpha_claimed}

# The reward as the agent's prompt states it, and one or two sentences on how to read it. Parameters in
# braces are filled from the run config (or the function's defaults).
FORMULAS = {
    "r_alpha": "reward = coverage * (1 - r) - r_rule - b  -  alpha * coverage * (1 - fidelity),   alpha = {alpha:g}",
    "r_alpha_claimed": (
        "reward = coverage * (1 - r) - r_rule - b  -  alpha * coverage * (1 - credited),   alpha = {alpha:g}\n"
        "    credited = the fidelity you claimed, if the measured fidelity is at least that;\n"
        "               otherwise measured - {penalty:g} * (claimed - measured)"),
}
EXPLANATIONS = {
    "r_alpha": (
        "The first part is the share of the work saved if the entry is used on its region and the network\n"
        "elsewhere. The second part prices a wrong answer at {alpha:g} forward passes."),
    "r_alpha_claimed": (
        "The first part is the share of the work saved if the entry is used on its region and the network\n"
        "elsewhere. The second part prices a wrong answer at {alpha:g} forward passes, with the fidelity you\n"
        "claimed in place of the measured one when the measurement confirms it. A claim above what is measured\n"
        "costs more than it could have gained, and a claim below it gives credit away: claim what you expect\n"
        "to be measured."),
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
    return EXPLANATIONS[name].format(**params)


def compute(parts: dict, cfg: dict, claimed: float | None = None) -> dict:
    """cfg is the run config's reward block: {name, ...parameters}."""
    check(cfg)
    name = cfg.get("name", "r_alpha")
    params = {k: v for k, v in cfg.items() if k != "name"}
    out = REWARDS[name](parts, claimed=claimed, **params)
    return {"name": name, "params": params, **out}
