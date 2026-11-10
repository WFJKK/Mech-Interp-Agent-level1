"""Reward functions: one number from what the verifier measured. A run config picks one by name.

The verifier's parts (verifier.py) are kept with every receipt, so a run can be scored again under another
reward without running anything (python -m level1.loop rescore).

    coverage   mu     the share of the data set in the entry's region S
    fidelity   p      the share of S on which the entry gives the network's output
    r                 operations of the prediction per input of S, over one forward pass
    r_rule            operations of the rule per input, over one forward pass
    b                 operations of the build, over one forward pass on every input of the data set

Use the entry on S and the network elsewhere, and compare with running the network everywhere:
mu * (1 - r) - r_rule - b is the share of the work saved, and mu * (1 - p) is the share of inputs now
answered wrongly. The network itself, as an entry, saves nothing and errs nowhere: every reward here gives it 0.
"""

from __future__ import annotations


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

FORMULAS = {
    "r_alpha": "reward = coverage * (1 - r) - r_rule - b  -  alpha * coverage * (1 - fidelity),   alpha = {alpha}",
    "r_alpha_claimed": (
        "reward = coverage * (1 - r) - r_rule - b  -  alpha * coverage * (1 - credited),   alpha = {alpha}\n"
        "    credited = the fidelity you claimed, if the measured fidelity is at least that;\n"
        "               otherwise measured - {penalty} * (claimed - measured)"),
}


def formula(cfg: dict) -> str:
    """The reward in words, for the agent's prompt."""
    name = cfg.get("name", "r_alpha")
    return FORMULAS[name].format(alpha=cfg.get("alpha", 10.0), penalty=cfg.get("penalty", 2.0))


def compute(parts: dict, cfg: dict, claimed: float | None = None) -> dict:
    """cfg is the run config's reward block: {name, ...parameters}."""
    name = cfg.get("name", "r_alpha")
    if name not in REWARDS:
        raise ValueError(f"no reward named {name!r}; there are {sorted(REWARDS)}")
    params = {k: v for k, v in cfg.items() if k != "name"}
    out = REWARDS[name](parts, claimed=claimed, **params)
    return {"name": name, "params": params, **out}
