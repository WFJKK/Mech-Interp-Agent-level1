"""What a run billed to an API key may spend, and what it has spent.

The run folder keeps an append-only log, spend.jsonl. Before the Claude Code CLI starts a session, a line
gives the round, the try (the number of its stream_NN.jsonl) and the cap the CLI is given; after it, a line
gives the cost the CLI reported:

    {"event": "start", "round": 2, "try": 1, "cap_usd": 20.0, "at": ...}
    {"event": "end", "round": 2, "try": 1, "cost_usd": 4.31, "ended": "success", "at": ...}
    {"event": "budget", "run_usd": 80.0, "at": ...}        the run's budget changed (resume --budget-usd)

A try counts at the cost the CLI reported. A try with no reported cost (the CLI was stopped at the session's
time limit, or the loop itself was stopped or killed) counts at its cap in the run's total: the CLI checks its
spending after every answer and stops at the cap, so the cap is what it can have spent, give or take its last
answer. The run's count therefore never falls short of the bill by more than that, and a resumed run goes on
counting where it was.

A round's own count holds the tries that ended while the loop ran: their reported costs, and the caps of those
stopped at the time limit. A try the loop never saw end (it was stopped with Ctrl-C, or killed) stays out of
it, so a round resumed after a stop starts with a whole round's budget instead of ending at once; what that try
may have spent is still in the run's total, which bounds everything.
"""

from __future__ import annotations

import math
from pathlib import Path

from .util import append_jsonl, now, read_jsonl

LOG = "spend.jsonl"
MIN_CAP_USD = 0.01  # the CLI takes the cap in cents; less than a cent left is nothing left


def state(run_path: Path, budget: dict) -> dict:
    """The budgets and what the log counts as spent: the run's total, and each round's own count (which leaves
    out the tries the loop never saw end; see above). "unreported" is the number of tries counted at their caps,
    "stopped" those of them that the loop never saw end."""
    run_usd, round_usd = float(budget["run_usd"]), float(budget["round_usd"])
    # In the order written. A try stopped with the loop left no stream file, so the try that resumes its round can
    # carry the same number: each start is a try of its own, and an end closes the latest open start of its number.
    tries, open_ = [], {}  # [round, cap, ended, cost]; (round, try) -> index of its start that has no end yet
    for e in read_jsonl(Path(run_path) / LOG):
        event = e.get("event")
        if event == "budget":
            run_usd = float(e["run_usd"])
        elif event == "start":
            key = (int(e["round"]), int(e["try"]))
            open_[key] = len(tries)
            tries.append([key[0], float(e["cap_usd"]), False, None])
        elif event == "end":
            i = open_.pop((int(e["round"]), int(e["try"])), None)
            if i is not None:
                tries[i][2], tries[i][3] = True, e.get("cost_usd")
    rounds, spent, unreported, stopped = {}, 0.0, 0, 0
    for k, cap, ended, cost in tries:
        if not ended:  # stopped with the loop: in the run's total at its cap, not in its round's count
            spent, unreported, stopped = spent + cap, unreported + 1, stopped + 1
            continue
        if not isinstance(cost, (int, float)) or isinstance(cost, bool):
            cost, unreported = cap, unreported + 1
        spent += float(cost)
        rounds[k] = rounds.get(k, 0.0) + float(cost)
    return {"run_usd": run_usd, "round_usd": round_usd, "spent": spent, "rounds": rounds, "unreported": unreported,
            "stopped": stopped}


def left(st: dict, k: int | None = None) -> float:
    """What is left of the run's budget, and of round k's when k is given: the smaller."""
    out = st["run_usd"] - st["spent"]
    if k is not None:
        out = min(out, st["round_usd"] - st["rounds"].get(k, 0.0))
    return out


def cap(st: dict, k: int) -> float:
    """The cap for the next try of round k, in whole cents, rounded down; 0 when less than a cent is left."""
    usd = left(st, k)
    return 0.0 if usd < MIN_CAP_USD else math.floor(usd * 100 + 1e-9) / 100


def spent_why(st: dict, k: int | None = None) -> str:
    """Which budget is spent, in words."""
    if st["run_usd"] - st["spent"] < MIN_CAP_USD:
        return f"the run's API budget is spent: ${st['spent']:.2f} counted of ${st['run_usd']:.2f}"
    return f"round {k}'s API budget is spent: ${st['rounds'].get(k, 0.0):.2f} counted of ${st['round_usd']:.2f}"


def start(run_path: Path, k: int, n: int, cap_usd: float) -> None:
    append_jsonl(Path(run_path) / LOG, {"event": "start", "round": k, "try": n, "cap_usd": cap_usd, "at": now()})


def end(run_path: Path, k: int, n: int, cost_usd, ended: str | None) -> None:
    cost = float(cost_usd) if isinstance(cost_usd, (int, float)) and not isinstance(cost_usd, bool) else None
    append_jsonl(Path(run_path) / LOG, {"event": "end", "round": k, "try": n, "cost_usd": cost, "ended": ended, "at": now()})


def set_run_budget(run_path: Path, run_usd: float) -> None:
    append_jsonl(Path(run_path) / LOG, {"event": "budget", "run_usd": float(run_usd), "at": now()})


def check(budget) -> list[str]:
    """What is wrong with a budget block, if anything."""
    if not isinstance(budget, dict):
        return ["budget must be a mapping with run_usd and round_usd"]
    errors = []
    for key in ("run_usd", "round_usd"):
        v = budget.get(key)
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not v > 0 or not math.isfinite(v):
            errors.append(f"budget.{key} must be a positive number of dollars, not {v!r}")
    return errors
