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


def wilson_lower(k: int, n: int, z: float = 1.96) -> float:
    """The lower end of the Wilson interval for k successes in n draws, as lang.wilson_lower (this file stands
    alone in the sandbox)."""
    if n == 0:
        return 0.0
    p = k / n
    centre = p + z * z / (2 * n)
    spread = z * (p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5
    return max(0.0, (centre - spread) / (1 + z * z / n))


def within_budget(parts: dict, budget: float) -> int:
    """How many inputs of the region the entry answered as the network does while spending at most budget of a
    forward pass on each (its region test and its prediction; the build is spread over all inputs, in the cost)."""
    limit = float(budget) * float(parts["model_ops"])
    return sum(int(agreed) for ops, _n, agreed in parts["input_ops"] if ops <= limit)


def r_budget(parts: dict, alpha: float = 10.0, budget: float = 0.1, price: float = 0.01, lower_bound: bool = True, **_) -> dict:
    """[saved + alpha * predicted - price * numbers] / (1 + alpha), where predicted is the share of the data set
    on which the entry gives the network's answer with at most budget of a forward pass spent on that input.

    An input on which the entry spends more than the budget counts as not predicted, whatever it answered: running
    the network there, or work close to it, earns nothing for that input. What scores is then what the entry
    predicts without running the network. The network itself spends a whole forward pass on every input and
    saves nothing, so it scores 0; an entry that predicts every input at no cost scores 1. Every number stored
    beyond those of the network's own definition is charged price, in forward passes, as in r_alpha_numbers.
    On a sample the share predicted is taken at its lower confidence bound unless lower_bound is false."""
    return _budgeted(parts, alpha, budget, price, 0.0, lower_bound, "r_budget")


def r_budget_nodes(parts: dict, alpha: float = 10.0, budget: float = 0.1, price: float = 0.01, node_price: float = 0.0005,
                   lower_bound: bool = True, **_) -> dict:
    """r_budget, with the entry's length charged too: node_price for every node beyond those of the network's own
    definition, in forward passes like the cost, before the division by 1 + alpha.

    A budget per input cannot tell a short explanation from a long map of the network's answers, a table looked
    up in a few operations: both are cheap on every input. What tells them apart is how long they are to write. A
    node counts whether or not it runs on a given input, so a long table or a deep tree of cases pays for every
    step of it, and a step is worth writing only if it earns more than node_price. The network itself is as long
    as its own definition and still scores 0."""
    return _budgeted(parts, alpha, budget, price, node_price, lower_bound, "r_budget_nodes")


def _budgeted(parts: dict, alpha, budget, price, node_price, lower_bound, name: str) -> dict:
    if parts.get("input_ops") is None:
        raise ValueError(f"{name} needs what single inputs cost in the parts (input_ops)")
    if parts.get("numbers") is None or parts.get("numbers_network") is None:
        raise ValueError(f"{name} needs the entry's stored numbers in the parts (numbers, numbers_network)")
    if node_price and (parts.get("nodes") is None or parts.get("nodes_network") is None):
        raise ValueError(f"{name} needs the entry's length in the parts (nodes, nodes_network)")
    mu, m = parts["coverage"], int(parts["n_in_region"])
    k = within_budget(parts, budget)
    q = (wilson_lower(k, m) if lower_bound and parts.get("sampled") else k / m) if m else 0.0
    r = parts["r"] if parts["r"] is not None else 0.0
    saved = mu * (1.0 - r) - parts["r_rule"] - parts["b"]
    extra = max(0, int(parts["numbers"]) - int(parts["numbers_network"]))
    longer = max(0, int(parts["nodes"]) - int(parts["nodes_network"])) if node_price else 0
    reward = (saved + float(alpha) * mu * q - float(price) * extra - float(node_price) * longer) / (1.0 + float(alpha))
    out = {"reward": reward, "saved": saved, "predicted": mu * q, "fidelity_used": q, "numbers": extra,
           "budget_ops": float(budget) * float(parts["model_ops"])}
    if node_price:
        out["nodes"] = longer
    return out


REWARDS = {"r_alpha": r_alpha, "r_alpha_claimed": r_alpha_claimed, "r_alpha_numbers": r_alpha_numbers, "r_budget": r_budget,
           "r_budget_nodes": r_budget_nodes}
NEEDS_NUMBERS = {"r_alpha_numbers", "r_budget", "r_budget_nodes"}  # rewards for which the verifier counts an entry's stored numbers
NEEDS_INPUT_OPS = {"r_budget", "r_budget_nodes"}  # rewards that read what single inputs cost (input_ops), which measure() then keeps


def needs_numbers(cfg: dict | None) -> bool:
    return (cfg or {}).get("name", "r_alpha") in NEEDS_NUMBERS


def needs_input_ops(cfg: dict | None) -> bool:
    return (cfg or {}).get("name", "r_alpha") in NEEDS_INPUT_OPS


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
    "r_budget_nodes": ("reward = [ coverage * ( (1 - cost) + alpha * within ) - price * numbers - node_price * nodes ]\n"
                       "    / (1 + alpha),   alpha = {alpha:g}, budget = {budget:g} of a forward pass per input,\n"
                       "    price = {price:g}, node_price = {node_price:g}"),
    "r_budget": ("reward = [ coverage * ( (1 - cost) + alpha * within ) - price * numbers ] / (1 + alpha),\n"
                 "    alpha = {alpha:g}, budget = {budget:g} of a forward pass per input, price = {price:g}"),
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
    "r_budget": (
        "Read it for one input of the region: using the entry there in place of the network saves\n"
        "(1 - cost) of a forward pass, and the entry earns {alpha:g} more when it gives the network's answer\n"
        "while spending at most {budget:g} of a forward pass on that input (its region test and its\n"
        "interpretation; the build is done once and paid in the cost). within is the share of the region's\n"
        "inputs where both hold: an input on which the entry spends more than the budget counts as not\n"
        "predicted, whatever it answers. Coverage turns that into a share of the data set; the inputs outside\n"
        "the region earn nothing. Dividing by (1 + alpha) puts the reward between 0, for the network itself,\n"
        "and 1, for an entry that predicts every input at no cost.\n"
        "numbers is what the entry stores beyond the network's own definition: every number that is the\n"
        "same on every input and for every weight, however it is written. That is every number in its\n"
        "constants, the length of every arange, every place a slice reads out of a list made of those, and\n"
        "every value it makes from its input or the weights that comes out the same whatever they are.\n"
        "Each counts once per {digits} significant digits it is written with (0.25 once, 0.123456 twice)\n"
        "and is charged {price:g}, in forward passes like the cost. What the entry computes from the weights\n"
        "is not counted here; it is charged as computation, in the cost."),
}
EXPLANATIONS["r_budget_nodes"] = EXPLANATIONS["r_budget"] + (
    "\nnodes is how long the entry is beyond the network's own definition: its number of nodes, each\n"
    "counted whether or not it runs on a given input, less the network's. Each is charged {node_price:g}, in\n"
    "forward passes like the cost, so a step is worth writing only if it earns more than that.")


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
    if name in ("r_budget", "r_budget_nodes") and float(params.get("budget", 0)) >= 1:
        out.append(f"budget = {float(params['budget']):g}: a whole forward pass fits in it, so running the network on an input "
                   "earns as much as predicting it")
    if name in ("r_budget", "r_budget_nodes") and params.get("lower_bound") is False:
        out.append("lower_bound is off: on a sampled data set, a lucky sample pays, and more submissions mean more luck")
    if name == "r_budget_nodes" and float(params.get("node_price", 0)) <= 0:
        out.append("node_price is 0 or less: a long map of the network's answers costs no more than a short explanation")
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
