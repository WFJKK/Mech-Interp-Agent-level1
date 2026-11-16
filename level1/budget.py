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

Grouped sessions opt in with cycle_id and cycle_usd. They reserve a cap under an OS lock before starting,
against the run, their own round, and their shared cycle. An interrupted grouped try keeps its reservation
in all three counts. Closing the try replaces its reservation with the reported cost, or its cap when no
valid cost was reported. The CLI can still exceed its cap by the cost of its last answer.
"""

from __future__ import annotations

import math
from contextlib import contextmanager
from pathlib import Path

try:
    import fcntl
except ImportError:  # the legacy single-session path does not need a Unix file lock
    fcntl = None

from .util import append_jsonl, now, read_jsonl

LOG = "spend.jsonl"
LOCK = "spend.lock"
MIN_CAP_USD = 0.01  # the CLI takes the cap in cents; less than a cent left is nothing left


@contextmanager
def _locked(run_path: Path):
    if fcntl is None:
        raise RuntimeError("grouped API budgets require OS file locking (fcntl on Unix or macOS)")
    path = Path(run_path)
    path.mkdir(parents=True, exist_ok=True)
    with (path / LOCK).open("a") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _append_locked(run_path: Path, event: dict) -> None:
    """Append under the caller's lock, separating any line a prior crash left incomplete."""
    path = Path(run_path) / LOG
    if path.exists() and path.stat().st_size:
        with path.open("rb+") as handle:
            handle.seek(-1, 2)
            if handle.read(1) != b"\n":
                handle.write(b"\n")
    append_jsonl(path, event)


def state(run_path: Path, budget: dict) -> dict:
    """The budgets and what the log counts as spent: the run's total, and each round's own count (which leaves
    out legacy tries the loop never saw end; grouped tries keep their reservations). "unreported" is the
    number of tries counted at their caps, "stopped" those of them that the loop never saw end."""
    if "cycle_id" in budget:
        with _locked(run_path):
            return _state(run_path, budget)
    return _state(run_path, budget)


def _state(run_path: Path, budget: dict) -> dict:
    run_usd, round_usd = float(budget["run_usd"]), float(budget["round_usd"])
    # In the order written. A try stopped with the loop left no stream file, so the try that resumes its round can
    # carry the same number: each start is a try of its own, and an end closes the latest open start of its number.
    tries, open_, cycle_limits = [], {}, {}
    for e in read_jsonl(Path(run_path) / LOG):
        event = e.get("event")
        if event == "budget":
            run_usd = float(e["run_usd"])
        elif event == "start":
            key = (int(e["round"]), int(e["try"]))
            open_[key] = len(tries)
            cycle_id = e.get("cycle_id")
            tries.append([key[0], float(e["cap_usd"]), False, None, cycle_id])
            if cycle_id is not None and "cycle_usd" in e:
                limit = float(e["cycle_usd"])
                cycle_limits[cycle_id] = min(cycle_limits.get(cycle_id, limit), limit)
        elif event == "end":
            i = open_.pop((int(e["round"]), int(e["try"])), None)
            if i is not None:
                tries[i][2], tries[i][3] = True, e.get("cost_usd")
    rounds, cycles, spent, unreported, stopped = {}, {}, 0.0, 0, 0
    for k, cap, ended, cost, cycle_id in tries:
        if not ended:  # no closing record yet: reserve its cap
            spent, unreported, stopped = spent + cap, unreported + 1, stopped + 1
            if cycle_id is not None:  # grouped sessions retain all their reservations on resume
                rounds[k] = rounds.get(k, 0.0) + cap
                cycles[cycle_id] = cycles.get(cycle_id, 0.0) + cap
            continue
        if (not isinstance(cost, (int, float)) or isinstance(cost, bool)
                or (cycle_id is not None and (not math.isfinite(cost) or cost < 0))):
            cost, unreported = cap, unreported + 1
        spent += float(cost)
        rounds[k] = rounds.get(k, 0.0) + float(cost)
        if cycle_id is not None:
            cycles[cycle_id] = cycles.get(cycle_id, 0.0) + float(cost)
    out = {"run_usd": run_usd, "round_usd": round_usd, "spent": spent, "rounds": rounds,
           "cycles": cycles, "unreported": unreported, "stopped": stopped}
    if "cycle_id" in budget:
        cycle_id, limit = budget["cycle_id"], float(budget["cycle_usd"])
        # Once a cycle has reserved money, another worker or a resume cannot silently enlarge its cap.
        out.update(cycle_id=cycle_id, cycle_usd=min(limit, cycle_limits.get(cycle_id, limit)))
    return out


def left(st: dict, k: int | None = None) -> float:
    """Run balance alone, or the minimum of run, round and active cycle balances when k is given."""
    out = st["run_usd"] - st["spent"]
    if k is not None:
        out = min(out, st["round_usd"] - st["rounds"].get(k, 0.0))
        if "cycle_id" in st:
            out = min(out, st["cycle_usd"] - st["cycles"].get(st["cycle_id"], 0.0))
    return out


def cap(st: dict, k: int) -> float:
    """The cap for the next try of round k, in whole cents, rounded down; 0 when less than a cent is left."""
    usd = left(st, k)
    return 0.0 if usd < MIN_CAP_USD else math.floor(usd * 100 + 1e-9) / 100


def spent_why(st: dict, k: int | None = None) -> str:
    """Which budget is spent, in words."""
    if st["run_usd"] - st["spent"] < MIN_CAP_USD:
        return f"the run's API budget is spent: ${st['spent']:.2f} counted of ${st['run_usd']:.2f}"
    if k is not None and "cycle_id" in st and st["cycle_usd"] - st["cycles"].get(st["cycle_id"], 0.0) < MIN_CAP_USD:
        return (f"cycle {st['cycle_id']}'s API budget is spent: "
                f"${st['cycles'].get(st['cycle_id'], 0.0):.2f} counted of ${st['cycle_usd']:.2f}")
    return f"round {k}'s API budget is spent: ${st['rounds'].get(k, 0.0):.2f} counted of ${st['round_usd']:.2f}"


def start(run_path: Path, k: int, n: int, cap_usd: float) -> None:
    append_jsonl(Path(run_path) / LOG, {"event": "start", "round": k, "try": n, "cap_usd": cap_usd, "at": now()})


def claim(run_path: Path, k: int, n: int, budget: dict) -> float:
    """Atomically reserve and log a grouped try's cap; return zero without starting when a budget is spent.

    Callers give each session a distinct round number in the parent run, including sessions in parallel.
    All workers in one cycle share its ID and dollar cap. Use end(..., atomic=True) to settle a claim.
    """
    errors = check(budget)
    if not isinstance(budget, dict) or "cycle_id" not in budget:
        errors.append("an atomic budget claim requires cycle_id and cycle_usd")
    if errors:
        raise ValueError("; ".join(errors))
    with _locked(run_path):
        st = _state(run_path, budget)
        usd = cap(st, k)
        if usd > 0:
            _append_locked(run_path, {"event": "start", "round": k, "try": n, "cap_usd": usd,
                                      "cycle_id": st["cycle_id"], "cycle_usd": st["cycle_usd"], "at": now()})
        return usd


def end(run_path: Path, k: int, n: int, cost_usd, ended: str | None, *, atomic: bool = False) -> None:
    cost = float(cost_usd) if isinstance(cost_usd, (int, float)) and not isinstance(cost_usd, bool) else None
    event = {"event": "end", "round": k, "try": n, "cost_usd": cost, "ended": ended, "at": now()}
    if atomic:
        with _locked(run_path):
            _append_locked(run_path, event)
    else:
        append_jsonl(Path(run_path) / LOG, event)


def set_run_budget(run_path: Path, run_usd: float, *, atomic: bool = False) -> None:
    event = {"event": "budget", "run_usd": float(run_usd), "at": now()}
    if atomic:
        with _locked(run_path):
            _append_locked(run_path, event)
    else:
        append_jsonl(Path(run_path) / LOG, event)


def check(budget) -> list[str]:
    """What is wrong with a budget block, if anything."""
    if not isinstance(budget, dict):
        return ["budget must be a mapping with run_usd and round_usd"]
    errors = []
    keys = ["run_usd", "round_usd"]
    if "cycle_id" in budget or "cycle_usd" in budget:
        if not isinstance(budget.get("cycle_id"), str) or not budget["cycle_id"].strip():
            errors.append("budget.cycle_id must be a nonempty string")
        keys.append("cycle_usd")
    for key in keys:
        v = budget.get(key)
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not v > 0 or not math.isfinite(v):
            errors.append(f"budget.{key} must be a positive number of dollars, not {v!r}")
    return errors
