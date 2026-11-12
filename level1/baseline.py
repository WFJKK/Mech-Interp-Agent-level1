"""The baseline: what a run of the same shape achieves where there is nothing learned to find.

A baseline model (withheld/<name>/task.yaml, `baseline_of`) has the design, sizes and data set of the models
it is the baseline of, its twins, and networks with nothing in them for an agent to find. A baseline run is
a usual run on it under the same conditions: the agent is given exactly what a run on a twin gives it, and is
not told the difference. What it achieves there is generic, and a run on a twin is read against it.

    python -m level1.loop baseline --config config/pilot/att-c.yaml    the baseline for that config's conditions
    python -m level1.loop baseline --run <run folder>                   the baseline for a run's own conditions
    python -m level1.loop run --config ... --with-baseline             a run and its baseline, side by side

Three steps, each written once and read from the records afterwards:

1. The baseline run. Its rounds use the baseline model's first network, its seed check the second.
2. Freezing, in the baseline run's folder (baseline.json, baseline_entry.json). The scored submissions are
   taken in the order of their reward on the second network (seed_scores.jsonl: every submission on one
   fixed sample), one per different entry, and each is scored again on every one of the model's other
   networks, all on one fresh sample. It passes when its rewards there agree within the tolerance (config:
   baseline.tolerance), counting each reward's sampling interval; it fails when two of them are further
   apart than that even with their intervals; otherwise it is inconclusive. That says what was established
   and no more. The first that passes is frozen; when none of the first baseline.candidates does, the first
   of them is frozen with its verdict, and the report says it is not established.
3. Reading a run against it, in that run's folder (baseline_scores.jsonl, and the report). The frozen entry is
   scored on the run's network and on its seed check's network, on the inputs of the run's own measurements
   of its locked entry: the second scoring on a fresh sample, and the seed check. With R0 and R1 the locked
   entry's rewards there and B0 and B1 the baseline's on the same inputs:

       the baseline on the run's network                  B0
       the excess kept on the held-out network            R1 - B1
       the gap between the run's network and the other    (R0 - B0) - (R1 - B1)

   which add up to R0. Negative values are kept. This is accounting, not a causal split: the baseline is a
   known achievable generic score, not the best possible one, so the excess can hold generic tricks the
   baseline run did not find; and the gap can be sampling noise or a difference between the networks as
   well as fitting the run's own network.

A run and a baseline run match when the run's model is one of the baseline model's twins and the conditions
that shape what the agent can achieve (CONDITIONS, and the price list) are the same. The report also says
whether the two first prompts were the same, byte for byte.

A baseline run that goes alongside a run (--with-baseline) is a second process; a lock file in its folder
keeps a second one from starting on the same baseline, and whoever needs it waits for it.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import os
import time
from pathlib import Path

from . import lang, rewards
from .networks import load_model
from .util import REPO_ROOT, append_jsonl, now, read_json, read_jsonl, read_yaml, write_json
from .verifier import Verifier, describe

FILE = "baseline.json"            # in a baseline run's folder: what was frozen, and how it fared on the other networks
ENTRY = "baseline_entry.json"     # in a baseline run's folder: the frozen entry, byte for byte as submitted
SCORES = "baseline_scores.jsonl"  # in a run's folder: the frozen entry on the run's own inputs, appended
LOCK = "baseline.lock"            # in a baseline run's folder while a process runs it
REQUEST = "baseline_request.json"  # in a baseline run's folder: the config a run with --with-baseline asked for

# What shapes what an agent can achieve in a run, besides the model: two runs are compared only when these
# are the same, and the price list too. Who pays, the sandbox, the budget and the names are not among them.
CONDITIONS = ("backend", "agent_model", "effort", "prompt_template", "rounds_max", "patience", "max_consecutive_refusals",
              "session", "python", "entry", "verifier", "reward")
STATES = ("frozen", "finished", "unfinished")  # a baseline run's states, the most useful first


# which models -------------------------------------------------------------------------------------
def twins(model) -> list:
    """The models a baseline model is the baseline of; empty for any other model."""
    return [str(x) for x in (model.task().get("baseline_of") or [])]


def is_baseline(model) -> bool:
    return bool(twins(model))


def baseline_models(model) -> list:
    """The baseline models whose twins include this model, by name."""
    out = []
    for task in sorted((REPO_ROOT / "withheld").glob("*/task.yaml")):
        if model.name in [str(x) for x in ((read_yaml(task) or {}).get("baseline_of") or [])]:
            out.append(task.parent.name)
    return out


# which runs ---------------------------------------------------------------------------------------
def conditions(cfg: dict, prices: str | None) -> dict:
    return {**{k: cfg.get(k) for k in CONDITIONS}, "prices": prices}


def differences(a: dict, b: dict) -> list:
    return sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))


def config_for(cfg: dict) -> dict:
    """The config of the baseline run for a run's config: the same in everything but the model and its networks.
    The baseline model's first network is for the rounds, its second for the seed check, the rest for freezing."""
    model = load_model(cfg["model"])
    names = baseline_models(model)
    if not names:
        raise RuntimeError(f"no baseline model has the shape of {model.name} (a withheld/<name>/task.yaml whose baseline_of names it)")
    name = names[0]
    nets = [n["id"] for n in load_model(f"models/{name}").config["networks"]]
    if len(nets) < 4:
        raise RuntimeError(f"the baseline model {name} needs at least four networks: the rounds', the seed check's and two to freeze on")
    key = hashlib.sha256(json.dumps(conditions(cfg, lang.PRICES), sort_keys=True).encode()).hexdigest()[:8]
    # Named like any pilot: nothing of the run's folder reaches the agent, and nothing in its name would say more.
    return {**cfg, "run_id": f"pilot-{name}-{key}", "model": f"models/{name}", "network": nets[0],
            "seed_check": {"network": nets[1]}}


def runs_dir(cfg: dict) -> Path:
    return Path(str(cfg["runs_dir"])).expanduser().resolve()


def state_of(path: Path) -> str:
    path = Path(path)
    if (path / FILE).is_file():
        return "frozen"
    return "finished" if (path / "lock.json").is_file() else "unfinished"


def find(cfg: dict, prices: str | None = None, where: Path | None = None) -> dict | None:
    """The baseline run that matches a run's config, in its runs folder: frozen first, then finished, then the
    earliest unfinished one. {"path", "state", "config"}, or None."""
    model = load_model(cfg["model"])
    names = set(baseline_models(model))
    want = conditions(cfg, prices or lang.PRICES)
    found = []
    root = Path(where) if where else runs_dir(cfg)
    for run_json in sorted(root.glob("*/run.json")) if root.is_dir() else []:
        try:
            run = read_json(run_json)
        except (OSError, ValueError):
            continue
        rc = run.get("config") or {}
        if Path(str(rc.get("model", ""))).name not in names or differences(conditions(rc, run.get("prices")), want):
            continue
        state = state_of(run_json.parent)
        found.append((STATES.index(state), run.get("created", ""), str(run_json.parent), state, rc))
    if not found:
        return None
    _, _, path, state, rc = min(found)
    return {"path": Path(path), "state": state, "config": rc}


def plan(cfg: dict, prices: str | None = None) -> dict:
    """Where the baseline for a run's config is, or is to be: {"path", "state" (or "new"), "config"}."""
    hit = find(cfg, prices)
    if hit:
        return hit
    bcfg = config_for(cfg)
    return {"path": runs_dir(cfg) / bcfg["run_id"], "state": "new", "config": bcfg}


# the lock -----------------------------------------------------------------------------------------
def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def running(path: Path) -> int | None:
    """The process that runs this baseline now, if one does."""
    f = Path(path) / LOCK
    try:
        pid = int(f.read_text().split()[0])
    except (OSError, ValueError, IndexError):
        return None
    return pid if _alive(pid) else None


def take_lock(path: Path) -> bool:
    """Take the baseline run's lock for this process; False when another living process holds it."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    f = path / LOCK
    for _ in range(2):
        try:
            fd = os.open(f, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            if running(path) not in (None, os.getpid()):
                return False
            f.unlink(missing_ok=True)  # a lock left by a process that is gone
            continue
        with os.fdopen(fd, "w") as fh:
            fh.write(f"{os.getpid()} {now()}\n")
        return True
    return False


def release_lock(path: Path) -> None:
    f = Path(path) / LOCK
    if running(path) in (None, os.getpid()):
        f.unlink(missing_ok=True)


def wait_for(path: Path, out=print, every_s: float = 30.0) -> None:
    """Wait while another process runs this baseline."""
    pid = running(path)
    if pid is not None:
        out(f"waiting for the baseline run in {path} (process {pid}) to end")
    while running(path) is not None:
        time.sleep(every_s)


# measuring ---------------------------------------------------------------------------------------
def reward_range(receipt: dict | None, reward_cfg: dict) -> tuple:
    """(low, high): the reward with the fidelity's 95% interval at either end. On a sample the reward already
    uses the lower end; the upper end is the reward the same entry would get were the true fidelity at the top
    of that interval. With every input checked the two are the same."""
    if not receipt or receipt.get("status") != "scored":
        return (None, None)
    low, p = receipt["reward"], receipt["parts"]
    if (receipt.get("inputs") or {}).get("mode") != "sample" or not p.get("n_in_region"):
        return (low, low)
    k, n = int(p["n_agree"]), int(p["n_in_region"])
    upper = 1.0 - lang.wilson_lower(n - k, n)
    high = rewards.compute({**p, "fidelity_low": upper}, reward_cfg)["reward"]
    return (low, max(low, high))


def judge(scores: dict, tolerance: float) -> tuple:
    """passes, fails or inconclusive, and why, for one entry's rewards on several networks: {network: (low, high)
    or None when it was refused there}."""
    refused = [n for n, r in scores.items() if r is None or r[0] is None]
    if refused:
        return "fails", f"it was refused on {', '.join(refused)}"
    if len(scores) < 2:
        return "inconclusive", "it was scored on fewer than two networks"
    inside, apart = True, []
    for (a, (alo, ahi)), (b, (blo, bhi)) in itertools.combinations(scores.items(), 2):
        dlo, dhi = alo - bhi, ahi - blo  # the difference of the two rewards, with both intervals
        if dlo > tolerance or dhi < -tolerance:
            apart.append((a, b))
        if dlo < -tolerance or dhi > tolerance:
            inside = False
    lows = [r[0] for r in scores.values()]
    span = f"{min(lows):.4f} to {max(lows):.4f}"
    if apart:
        a, b = apart[0]
        return "fails", (f"on {a} and {b} its rewards are more than {tolerance:g} apart even with their sampling intervals "
                         f"({scores[a][0]:.4f} and {scores[b][0]:.4f}); on all {len(scores)}: {span}")
    if inside:
        return "passes", f"its rewards on all {len(scores)} networks agree within {tolerance:g}, with their sampling intervals: {span}"
    return "inconclusive", (f"its rewards ({span}) neither agree within {tolerance:g} nor differ by more, once their sampling "
                            "intervals are counted")


def _check_prices(run: dict, path: Path) -> None:
    if run.get("prices") != lang.PRICES:
        raise RuntimeError(f"{path.name} was measured under {run.get('prices')}, and this code prices under {lang.PRICES}: "
                           "their rewards cannot be compared")


def _receipt(r: dict) -> dict:
    return {k: r.get(k) for k in ("status", "reason", "network", "reward", "parts", "inputs")}


# step 2: freezing ---------------------------------------------------------------------------------
def freeze(path: Path, log=print) -> dict:
    """Freeze a finished baseline run's entry (once: an existing baseline.json is the result and is returned)."""
    from .loop import entry_of, scored, seed_scores, completed

    path = Path(path)
    if (path / FILE).is_file():
        return read_json(path / FILE)
    run = read_json(path / "run.json")
    cfg = run["config"]
    model = load_model(cfg["model"])
    if not is_baseline(model):
        raise RuntimeError(f"{path.name} is a run on {model.name}, which is not a baseline model")
    if not (path / "lock.json").is_file():
        raise RuntimeError(f"{path.name} has not finished: nothing to freeze yet")
    _check_prices(run, path)
    rounds = completed(path)
    seed_scores(path, cfg, model, rounds, log)  # any submission not yet scored on the second network
    second = cfg["seed_check"]["network"]
    on_second = {(e.get("carried_as"), e.get("entry_sha256")): e for e in read_jsonl(path / "seed_scores.jsonl")}
    candidates = []
    for s in scored(rounds):
        e = on_second.get((s["carried_as"], s["entry_sha256"]))
        if e and e.get("status") == "scored":
            candidates.append((-e["reward"], s["round"], s["submission"], s, e))
    settings = cfg.get("baseline") or {}
    tolerance, tries = float(settings.get("tolerance", 0.01)), int(settings.get("candidates", 3))
    record = {"baseline_run": path.name, "model": model.name, "twins": twins(model), "conditions": conditions(cfg, run.get("prices")),
              "tolerance": tolerance, "frozen_at": now()}
    if not candidates:
        record.update(entry_sha256=None, label="none", why="no submission was scored on the second network")
        write_json(path / FILE, record)
        log(f"the baseline: nothing to freeze ({record['why']})")
        return record
    others = [n["id"] for n in model.config["networks"] if n["id"] not in (cfg["network"], second)]
    seed = int.from_bytes(os.urandom(8), "big")  # one fresh sample for every candidate and every network
    tried, seen = [], set()
    for _, _, _, sub, e in sorted(candidates, key=lambda c: c[:3]):
        if sub["entry_sha256"] in seen:
            continue  # the same entry submitted again
        seen.add(sub["entry_sha256"])
        entry = entry_of(path, sub)
        if lang.entry_hash(entry) != sub["entry_sha256"]:
            raise RuntimeError(f"{sub['carried_as']}'s entry on disk is not the one the verifier scored")
        confirm, ranges = {}, {}
        for nid in others:
            r = Verifier(model, nid, cfg).score(entry, seed=seed)
            confirm[nid] = {**_receipt(r), "reward_range": list(reward_range(r, cfg["reward"]))}
            ranges[nid] = None if r["status"] != "scored" else reward_range(r, cfg["reward"])
            log(f"  the baseline candidate {sub['carried_as']} on {nid}: {describe(r)}")
        label, why = judge(ranges, tolerance)
        tried.append({"sub": sub, "on_second": e, "entry": entry, "confirmation": confirm, "label": label, "why": why})
        log(f"  {sub['carried_as']} ({sub['entry_sha256'][:12]}), {e['reward']:.4f} on the second network: {label} ({why})")
        if label == "passes" or len(tried) >= tries:
            break
    pick = next((x for x in tried if x["label"] == "passes"), tried[0])
    sub, e = pick["sub"], pick["on_second"]
    record.update(entry_sha256=sub["entry_sha256"], chosen=sub["carried_as"], name=sub["name"],
                  chosen_by=(f"the highest reward on the second network that passes the check, trying up to {tries} different "
                             "entries in that order (every submission is scored on the second network on one fixed sample)"),
                  on_first={k: sub.get(k) for k in ("reward", "parts", "inputs")},
                  on_second={k: e.get(k) for k in ("network", "reward", "parts", "inputs")},
                  confirmation={"inputs_seed": seed, "networks": pick["confirmation"]}, label=pick["label"], why=pick["why"],
                  established=pick["label"] == "passes",
                  tried=[{"chosen": x["sub"]["carried_as"], "entry_sha256": x["sub"]["entry_sha256"], "name": x["sub"]["name"],
                          "on_second_reward": x["on_second"]["reward"], "label": x["label"], "why": x["why"]} for x in tried])
    write_json(path / ENTRY, pick["entry"])
    write_json(path / FILE, record)
    log(f"the baseline: {sub['carried_as']} ({sub['entry_sha256'][:12]}) frozen, {pick['label']}"
        + ("" if pick["label"] == "passes" else f": none of the {len(tried)} tried passed, so it is not established"))
    return record


# step 3: reading a run against it -------------------------------------------------------------------
def _measurements(path: Path, cfg: dict) -> list:
    """The run's own measurements of its locked entry that the baseline is compared on: (role, network, receipt)."""
    lock = read_json(path / "lock.json")
    first = lock.get("measured_again")
    if first is None and lock.get("locked"):
        from .loop import completed  # every input checked: the scoring the lock picked is not a lucky sample
        sub = next((s for rd in completed(path) for s in rd["submissions"] if s["carried_as"] == lock["carried_as"]), None)
        first = {"status": "scored", "reward": lock["reward"], "parts": lock["parts"], "inputs": (sub or {}).get("inputs")}
    seed = read_json(path / "seed_check.json") if (path / "seed_check.json").is_file() else None
    out = []
    if first and first.get("status") == "scored" and first.get("inputs"):
        out.append(("first", cfg["network"], first))
    second = (cfg.get("seed_check") or {}).get("network")
    if second and seed and seed.get("status") == "scored" and seed.get("inputs"):
        out.append(("held out", second, seed))
    return out


def compare(path: Path, log=print) -> dict:
    """Score the matching frozen baseline on a finished run's own inputs, unless that is on record already, and say
    where things stand: {"state": ...}. Freezes a baseline run that has finished but is not frozen yet."""
    path = Path(path)
    run = read_json(path / "run.json")
    cfg = run["config"]
    model = load_model(cfg["model"])
    if is_baseline(model):
        return {"state": "is a baseline"}
    lock = read_json(path / "lock.json") if (path / "lock.json").is_file() else None
    if not lock or not lock.get("locked"):
        return {"state": "nothing locked"}
    if not baseline_models(model):
        return {"state": "no baseline model", "why": f"no baseline model has the shape of {model.name}"}
    hit = find(cfg, run.get("prices"), path.parent)  # a baseline run is looked for next to the run
    if hit is None:
        return {"state": "none", "command": f"python -m level1.loop baseline --run {path}"}
    if hit["state"] == "unfinished":
        return {"state": "pending", "baseline_run": hit["path"].name}
    _check_prices(run, path)
    frozen = freeze(hit["path"], log)
    if not frozen.get("entry_sha256"):
        return {"state": "empty", "baseline_run": hit["path"].name, "why": frozen.get("why")}
    entry = read_json(hit["path"] / ENTRY)
    if lang.entry_hash(entry) != frozen["entry_sha256"]:
        raise RuntimeError(f"{hit['path'].name}'s frozen entry on disk is not the one it records")
    have = {(e.get("baseline_entry_sha256"), e.get("network"), (e.get("inputs") or {}).get("seed"), (e.get("inputs") or {}).get("mode"))
            for e in read_jsonl(path / SCORES)}
    for role, nid, mine in _measurements(path, cfg):
        how = mine["inputs"]
        if (frozen["entry_sha256"], nid, how.get("seed"), how.get("mode")) in have:
            continue
        r = Verifier(model, nid, cfg).score(entry, seed=how.get("seed"))
        if r.get("inputs") and mine["inputs"].get("mode") == "sample" and r["inputs"].get("seed") != how.get("seed"):
            raise RuntimeError("the baseline was not scored on the run's own inputs")
        append_jsonl(path / SCORES, {"baseline_run": hit["path"].name, "baseline_entry_sha256": frozen["entry_sha256"], "role": role,
                                     **_receipt(r), "network": nid, "at": now()})
        log(f"  the baseline ({hit['path'].name}) on {nid}, on the inputs of the locked entry's {role} measurement: {describe(r)}")
    return {"state": "read", "baseline_run": hit["path"].name}


def same_first_prompt(a: Path, b: Path) -> bool | None:
    pa, pb = Path(a) / "round_001" / "prompt.md", Path(b) / "round_001" / "prompt.md"
    if not (pa.is_file() and pb.is_file()):
        return None
    return pa.read_bytes() == pb.read_bytes()


def reading(path: Path) -> dict:
    """What the records say about a run and its baseline, for the report; nothing is scored here."""
    path = Path(path)
    run = read_json(path / "run.json")
    cfg = run["config"]
    model = load_model(cfg["model"])
    if is_baseline(model):
        frozen = read_json(path / FILE) if (path / FILE).is_file() else None
        return {"kind": "baseline run", "twins": twins(model), "frozen": frozen}
    if not baseline_models(model):
        return {"kind": "run", "state": "no baseline model"}
    hit = find(cfg, run.get("prices"), path.parent)
    if hit is None:
        return {"kind": "run", "state": "none", "command": f"python -m level1.loop baseline --run {path}"}
    out = {"kind": "run", "state": hit["state"], "baseline_run": hit["path"].name,
           "same_first_prompt": same_first_prompt(path, hit["path"])}
    if hit["state"] != "frozen":
        return out
    frozen = read_json(hit["path"] / FILE)
    out.update(label=frozen.get("label"), why=frozen.get("why"), chosen=frozen.get("chosen"), entry_sha256=frozen.get("entry_sha256"),
               tolerance=frozen.get("tolerance"))
    if not frozen.get("entry_sha256"):
        out["state"] = "empty"
        return out
    mine = {role: (nid, r) for role, nid, r in _measurements(path, cfg)}
    theirs = {}
    for e in read_jsonl(path / SCORES):
        if e.get("baseline_entry_sha256") == frozen["entry_sha256"]:
            theirs[e["role"]] = e
    if set(mine) != {"first", "held out"} or set(theirs) < {"first", "held out"}:
        out["state"] = "frozen, not yet read against this run"
        return out
    rc = cfg["reward"]
    R0, R1 = reward_range(mine["first"][1], rc), reward_range(mine["held out"][1], rc)
    B0, B1 = reward_range(theirs["first"], rc), reward_range(theirs["held out"], rc)
    if None in (R0[0], R1[0], B0[0], B1[0]):
        out["state"] = "refused"
        return out
    keep = (R1[0] - B1[0], (R1[0] - B1[1], R1[1] - B1[0]))
    gap = ((R0[0] - B0[0]) - (R1[0] - B1[0]), ((R0[0] - B0[1]) - (R1[1] - B1[0]), (R0[1] - B0[0]) - (R1[0] - B1[1])))
    out.update(state="read", networks={"first": mine["first"][0], "held out": mine["held out"][0]},
               locked={"first": mine["first"][1], "held out": mine["held out"][1]},
               baseline={"first": theirs["first"], "held out": theirs["held out"]},
               ranges={"R0": R0, "R1": R1, "B0": B0, "B1": B1},
               split={"baseline": (B0[0], B0), "kept": keep, "gap": gap})
    return out


def runs_it_serves(bpath: Path) -> list:
    """The finished runs in the same runs folder that this baseline run matches."""
    bpath = Path(bpath)
    brun = read_json(bpath / "run.json")
    bmodel = load_model(brun["config"]["model"])
    out = []
    for run_json in sorted(bpath.parent.glob("*/run.json")):
        if run_json.parent == bpath:
            continue
        try:
            run = read_json(run_json)
        except (OSError, ValueError):
            continue
        rc = run.get("config") or {}
        if Path(str(rc.get("model", ""))).name not in twins(bmodel) or not (run_json.parent / "lock.json").is_file():
            continue
        if differences(conditions(rc, run.get("prices")), conditions(brun["config"], brun.get("prices"))):
            continue
        out.append(run_json.parent)
    return out
