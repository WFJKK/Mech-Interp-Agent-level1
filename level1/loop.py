"""The loop: the rounds on one network, then the lock, the seed check and the report.

    python -m level1.loop run --config config/pilot/lin-a.yaml [--run-id X] [--sandbox hardened] [--fake DIR]
                              [--auth api_key] [--budget-usd 60]
    python -m level1.loop resume --run <run folder> [--budget-usd 80]
    python -m level1.loop finish --run <run folder>     the lock, the seed check and the report again, from the records
    python -m level1.loop baseline --config <config> | --run <run folder>    the baseline for that shape and those
                                  conditions: found, or run, then frozen (level1/baseline.py)
    run and resume also take --with-baseline: the run's baseline goes alongside, in a second process
    python -m level1.loop rescore --run <run folder> --name r_alpha --alpha 1    the records under another reward
                                  (--param KEY=VALUE for any other parameter of a reward)

A run folder (outside the repository, <runs_dir>/<run_id>):

    run.json            the config, the commit, the hash of the trusted files, the sandbox and its probe
    log.jsonl           what happened, appended
    spend.jsonl         a run billed to an API key: every try's cap and cost, appended (budget.py)
    round_001/          prompt.md, session/ (session.py), response.md, meta.json, round.json
    locked_entry.json   the best entry of the rounds, as submitted
    lock.json           which submission it is, its hash and its receipt
    seed_check.json     the verifier's receipt for the locked entry on the second network
    seed_scores.jsonl   every scored submission also scored on the second network, on one fixed sample,
                        appended after each round; nothing the agent sees reads it, and nothing is chosen by it
    report.md, report.json
    baseline_scores.jsonl   the frozen baseline of this run's shape, scored on this run's own inputs (baseline.py)
    baseline.json, baseline_entry.json   in a baseline run: what was frozen, and how it fared on the other networks

A round is done when its round.json is written. `resume` starts at the first round that is not, and a
session that was cut off keeps its budgets and submissions. The rounds end at rounds_max, after `patience`
rounds in a row that did not raise the best reward, after max_consecutive_refusals refusals in a row, or,
billed to an API key, when the run's budget is spent (a round that was under way is ended and kept first).

When they end, the submission with the highest reward is locked: the entry as it was submitted, byte for
byte (its hash is checked). The seed check hands that entry to the verifier on the second network. No agent
takes part and nothing of the entry can change, so what comes out is how an entry written on one network
does on one it was not written on.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from . import budget, hygiene, lang, prompt, rewards, run_config
from .backends import StopRun, make
from .networks import load_model
from .session import Session, surface
from .util import REPO_ROOT, append_jsonl, atomic_write, git_state, now, read_json, read_jsonl, sha256_file, write_json
from .verifier import Verifier, describe

TRUSTED_GLOBS = ("level1/**/*.py", "prompts/*.md", "agent_docs/*.md")
NOTES_HEADING = "## What the network does"
MAX_NOTES = 6000


def trusted_hash(model) -> str:
    """One hash over the code, the prompt, the agent's guide, the model folder and the withheld folder."""
    import hashlib

    files = [p for g in TRUSTED_GLOBS for p in REPO_ROOT.glob(g)]
    for root in (model.path, model.withheld()):
        if root.is_dir():
            files += list(root.rglob("*"))
    h = hashlib.sha256()
    for p in sorted({q.resolve() for q in files if q.is_file() and "__pycache__" not in q.parts}):
        h.update(os.path.relpath(p, REPO_ROOT).encode())
        h.update(b"\0")
        h.update(sha256_file(p).encode())
        h.update(b"\0")
    return h.hexdigest()


class Log:
    def __init__(self, path: Path, out=print):
        self.path, self.out = Path(path), out

    def __call__(self, message: str, **fields) -> None:
        self.out(message)
        append_jsonl(self.path / "log.jsonl", {"at": now(), "message": message, **fields})


# what the records say ---------------------------------------------------------------------------
def completed(path: Path) -> list:
    """The rounds that are done, in order: round.json of round 1, 2, ... until one is missing."""
    out = []
    while True:
        f = Path(path) / f"round_{len(out) + 1:03d}" / "round.json"
        if not f.is_file():
            return out
        out.append(read_json(f))


def scored(rounds: list) -> list:
    return [s for rd in rounds for s in rd["submissions"] if s["status"] == "scored"]


def best_submission(rounds: list) -> dict | None:
    """The scored submission with the highest reward; the earliest one on a tie."""
    best = None
    for s in scored(rounds):
        if best is None or s["reward"] > best["reward"]:
            best = s
    return best


def stop_reason(rounds: list, cfg: dict) -> str | None:
    if len(rounds) >= int(cfg["rounds_max"]):
        return f"rounds_max ({cfg['rounds_max']}) reached"
    tail = 0
    for rd in reversed(rounds):
        if rd["improved"]:
            break
        tail += 1
    if tail >= int(cfg["patience"]):
        return f"no higher reward in {tail} rounds in a row (patience {cfg['patience']})"
    refusals = 0
    for rd in reversed(rounds):
        if rd.get("stop_reason") != "refusal":
            break
        refusals += 1
    if refusals >= int(cfg["max_consecutive_refusals"]):
        return f"the model declined {refusals} rounds in a row"
    return None


def entry_of(path: Path, sub: dict) -> dict:
    return read_json(Path(path) / f"round_{sub['round']:03d}" / "session" / "submissions" / f"{sub['submission']:03d}" / "entry.json")


def carried_entries(path: Path, rounds: list) -> dict:
    """The scored entries of earlier rounds, by the names the prompt's history gives them, and the best as `best`."""
    out = {s["carried_as"]: entry_of(path, s) for s in scored(rounds)}
    best = best_submission(rounds)
    if best is not None:
        out["best"] = entry_of(path, best)
    return out


SEED_SCORES = "seed_scores.jsonl"


def seed_score_sample(run_id: str) -> int:
    """The one sample on which every submission of a run is scored on the second network. It is the same for all
    of them, so that they can be compared with each other without the noise of separate samples."""
    return int.from_bytes(hashlib.sha256(f"{run_id}: every submission on the second network".encode()).digest()[:8], "big")


def seed_scores(path: Path, cfg: dict, model, rounds: list, log=None) -> list:
    """Score every scored submission of these rounds on the second network, unless seed_scores.jsonl has it
    already, and append it there. The agent never sees these scores (the prompt reads round.json, not this file),
    and nothing is chosen by them: the lock stays the best submission on the first network, and the seed check of
    the locked entry stays the run's one fresh measurement on the second."""
    path = Path(path)
    second = (cfg.get("seed_check") or {}).get("network")
    if not second:
        return []
    have = {(e.get("carried_as"), e.get("entry_sha256")) for e in read_jsonl(path / SEED_SCORES)}
    todo = [s for s in scored(rounds) if (s["carried_as"], s["entry_sha256"]) not in have]
    if not todo:
        return []
    verifier, sample = Verifier(model, second, cfg), seed_score_sample(cfg["run_id"])
    out = []
    for s in todo:
        entry = entry_of(path, s)
        if lang.entry_hash(entry) != s["entry_sha256"]:
            raise RuntimeError(f"{s['carried_as']}'s entry on disk is not the one the verifier scored")
        r = verifier.score(entry, claimed=s.get("expected_fidelity_other"), seed=sample)
        line = {"round": s["round"], "submission": s["submission"], "carried_as": s["carried_as"], "name": s["name"],
                "entry_sha256": s["entry_sha256"], "network": second, "status": r["status"], "reason": r["reason"],
                "reward": r["reward"], "parts": r["parts"], "inputs": r["inputs"], "at": now()}
        append_jsonl(path / SEED_SCORES, line)
        out.append(line)
        if log:
            log(f"  {s['carried_as']} on the second network: {describe(r)}")
    return out


def notes_of(text: str) -> str:
    """The agent's closing notes: from the first asked-for heading on, or the end of what it wrote."""
    i = text.rfind(NOTES_HEADING)
    return (text[i:] if i >= 0 else text[-MAX_NOTES:])[:MAX_NOTES].strip()


# a run ---------------------------------------------------------------------------------------------
def forbidden_strings(cfg: dict, model) -> list:
    return sorted({str(s) for s in list(cfg.get("hygiene") or []) + model.hygiene()})


def agent_surface(cfg: dict, model, sandbox_mode: str | None = None) -> dict:
    """Every text the pipeline itself puts within the agent's reach, besides a round's prompt: the session's
    surface (session.py), the note a restarted session gets, and the paths that can show from inside: the
    temporary folder the CLI is started in and, unless the agent's code runs in a container (which has its
    own paths), the sandbox folder's location and this Python's (a traceback prints it)."""
    out = {**surface(model, cfg["verifier"]), "restart note": prompt.RESTART_NOTE,
           "temporary folder's location": tempfile.gettempdir()}
    if sandbox_mode != "container":
        out["sandbox folder's location"] = str(cfg.get("sandbox_dir") or tempfile.gettempdir())
        out["python's location"] = " ".join(sorted({sys.prefix, sys.base_prefix, sys.executable,
                                                     str(cfg.get("python_executable") or "")}))
    return out


def start(cfg: dict, fake_scripts=None, out=print) -> Path:
    """Make the run folder, check everything that can be checked before a model is asked, then run the rounds."""
    from .sandbox import check_probe, choose, probe, protected_paths

    path = run_config.run_path(cfg)
    if (path / "run.json").exists():
        raise RuntimeError(f"{path} is a run already: resume it, or give the config another run_id")
    model = load_model(cfg["model"])
    model.network(cfg["network"])
    second = (cfg.get("seed_check") or {}).get("network")
    if second:
        model.network(second)
        if second == cfg["network"]:
            raise RuntimeError("the seed check needs another network than the rounds'")
    make(cfg, log=out, fake_scripts=fake_scripts)  # a missing CLI or login fails here, before a run folder exists
    path.mkdir(parents=True, exist_ok=True)
    log = Log(path, out)
    verifier = Verifier(model, cfg["network"], cfg)
    itself = verifier.score(verifier.reference)
    # The network as an entry: right on every input, at exactly one forward pass each. (Its reward is then 0
    # when every input is checked, and a little below on a sample, where the fidelity's lower bound is used.)
    if itself["status"] != "scored" or itself["parts"]["fidelity"] != 1.0 or itself["parts"]["cost"] != 1.0:
        raise RuntimeError(f"the network written as an entry does not come out at fidelity 1 and cost 1: {describe(itself)}")
    sb = choose(cfg, protected_paths(cfg, model, path))
    first_prompt = prompt.build(cfg, model, 1, [], itself["parts"]["model_ops"], verifier.enumerated)
    hygiene.check({"prompt": first_prompt, **agent_surface(cfg, model, sb.mode)}, forbidden_strings(cfg, model))
    report = probe(sb, model, cfg["network"], path / "probe")                      # the system's sandbox on its own
    guarded = probe(sb, model, cfg["network"], path / "probe", guard=True)         # with the guard inside Python
    try:
        (path / "probe").rmdir()  # the probe's target folder; it leaves nothing in it
    except OSError:
        pass
    check_probe(sb, report)
    check_probe(sb, guarded)
    for w in rewards.warnings(cfg["reward"]):
        log(f"WARNING: the reward: {w}")
    if not sb.hardened:
        log("WARNING: the sandbox is in process mode: only a guard inside Python keeps the agent's code from your files, "
            "the network and other programs. It blocked every escape the probe tried, and records every attempt, but it "
            "shares its process with the code it watches: a fence, not a wall. A run that counts needs container, "
            "landlock or seatbelt mode.")
    write_json(path / "run.json", {
        "run_id": cfg["run_id"], "config": cfg, "created": now(), "git": git_state(), "trusted_hash": trusted_hash(model),
        "sandbox": {**sb.describe(), "probe": report, "guard_probe": guarded}, "prices": lang.PRICES, "model_ops": itself["parts"]["model_ops"],
        "n_total": model.n_total, "enumerated": verifier.enumerated, "check": verifier.check, "n_inputs": verifier.n_inputs(),
    })
    log(f"run {cfg['run_id']} at {path}: {cfg['model']} / {cfg['network']}, sandbox {sb.mode}, backend {cfg['backend']}, "
        + (f"every one of {model.n_total:,} inputs checked" if verifier.enumerated else f"{verifier.n_inputs():,} inputs sampled per scoring"))
    if cfg["backend"] == "claude_code" and cfg["auth"] == "api_key":
        b = cfg["budget"]
        log(f"billed to the API key in {cfg['api_key_env']}: at most ${float(b['run_usd']):.2f} for the run and "
            f"${float(b['round_usd']):.2f} a round, by Claude Code's count (spend.jsonl)")
    elif cfg["backend"] == "claude_code":
        log("under your Claude Code login: no API key is used")
    drive(path, fake_scripts=fake_scripts, out=out)
    return path


def drive(path: Path, fake_scripts=None, out=print, accept_changes: bool = False, budget_usd: float | None = None) -> str:
    """Run the rounds that are left, then finish. Returns how it ended: finished, or stopped (resume later).
    budget_usd: a new budget for a run billed to an API key, kept in its spend log."""
    path = Path(path)
    run = read_json(path / "run.json")
    cfg = run["config"]
    log = Log(path, out)
    if budget_usd is not None:
        if cfg.get("auth") != "api_key":
            raise RuntimeError("this run is not billed to an API key, so it has no budget to change")
        errors = budget.check({**cfg["budget"], "run_usd": budget_usd})
        if errors:
            raise RuntimeError("; ".join(errors))
        before = budget.state(path, cfg["budget"])
        budget.set_run_budget(path, budget_usd)
        log(f"the run's API budget is now ${budget_usd:.2f} (was ${before['run_usd']:.2f}; ${before['spent']:.2f} counted so far)",
            budget_usd=budget_usd)
    model = load_model(cfg["model"])
    if trusted_hash(model) != run["trusted_hash"]:
        if not accept_changes:
            raise RuntimeError("the code, the prompt, the guide or the model folder changed since this run began; "
                               "resume with --accept-changes to go on all the same (the log will say so)")
        log("the trusted files changed since the run began; going on as asked", trusted_hash=trusted_hash(model))
    backend = make(cfg, log=log, fake_scripts=fake_scripts)
    forbidden = forbidden_strings(cfg, model)
    while True:
        done = completed(path)
        why = stop_reason(done, cfg)
        if why is None and not backend.available(len(done)):
            why = "the backend has no answer for another round"
        # A spent budget ends the rounds; a round that was under way runs first, and ends at once with what it
        # submitted, so that its submissions are kept.
        if why is None and hasattr(backend, "exhausted") and not (path / f"round_{len(done) + 1:03d}").exists():
            why = backend.exhausted(path)
        if why is not None:
            log(f"the rounds end: {why}", ended=why)
            break
        try:
            run_round(path, cfg, model, len(done) + 1, done, backend, forbidden, run, log)
        except StopRun as exc:
            log(f"stopped: {exc}. Continue with: python -m level1.loop resume --run {path}", stopped=str(exc))
            return "stopped"
    finish(path, out=out)
    return "finished"


def run_round(path: Path, cfg: dict, model, k: int, done: list, backend, forbidden: list, run: dict, log) -> dict:
    rdir = path / f"round_{k:03d}"
    session = Session.create(rdir / "session", cfg, cfg["network"], path, carried_entries(path, done))
    text = prompt.build(cfg, model, k, done, run["model_ops"], run["enumerated"])
    # What the pipeline itself says is checked; the history is the agent's own words about what it found.
    ours = prompt.build(cfg, model, k, [], run["model_ops"], run["enumerated"])
    hygiene.check({"prompt": ours, **agent_surface(cfg, model, run["sandbox"]["mode"])}, forbidden)
    atomic_write(rdir / "prompt.md", text)
    log(f"round {k}: asking {backend.name}")
    response = backend.respond(text, k - 1, rdir, session)
    atomic_write(rdir / "response.md", response.text)
    write_json(rdir / "meta.json", response.meta)
    session.reload()
    subs = []
    for r in session.receipts():
        subs.append({"round": k, "submission": r["submission"], "name": r.get("name"), "carried_as": f"r{k}s{r['submission']}",
                     "status": r["status"], "reason": r["reason"], "reward": r["reward"], "parts": r["parts"], "size": r["size"],
                     "entry_sha256": r["entry_sha256"], "inputs": r["inputs"], "claimed_fidelity": r.get("claimed_fidelity"),
                     "expected_fidelity_other": r.get("expected_fidelity_other"), "overclaim": r.get("overclaim")})
    before = best_submission(done)
    got = [s["reward"] for s in subs if s["status"] == "scored"]
    improved = bool(got) and (before is None or max(got) > before["reward"] + 1e-12)
    record = {"round": k, "submissions": subs, "improved": improved, "best_reward_before": None if before is None else before["reward"],
              "best_reward_after": max(got + ([before["reward"]] if before else [])) if (got or before) else None,
              "stop_reason": response.meta.get("stop_reason"), "notes": notes_of(response.text), "tools_used": dict(session.state["used"]),
              "at": now()}
    write_json(rdir / "round.json", record)
    for s in subs:
        log(f"  {s['carried_as']} ({s['name']}): " + (describe(s) if s["status"] == "scored" else f"refused: {s['reason']}"))
    best_text = "none" if record["best_reward_after"] is None else f"{record['best_reward_after']:.4f}"
    log(f"round {k} done: {len(subs)} submissions, best reward so far {best_text}", round=k, improved=improved)
    try:  # for the record only; a failure here must not cost the run, and finish tries again
        seed_scores(path, cfg, model, [record], log)
    except Exception as exc:
        log(f"  could not score round {k}'s submissions on the second network ({type(exc).__name__}: {exc}); finish tries again")
    return record


def finish(path: Path, out=print) -> dict:
    """Lock the best entry, run the seed check, write the report. Works from the records alone, so it can be
    run again."""
    from . import report

    path = Path(path)
    cfg = read_json(path / "run.json")["config"]
    log = Log(path, out)
    model = load_model(cfg["model"])
    done = completed(path)
    best = best_submission(done)
    second = (cfg.get("seed_check") or {}).get("network")
    # What an earlier finish measured on a sample is a result and is kept: finishing again never redraws it.
    old_lock = read_json(path / "lock.json") if (path / "lock.json").is_file() else {}
    old_seed = read_json(path / "seed_check.json") if (path / "seed_check.json").is_file() else {}
    seed_scores(path, cfg, model, done, log)  # any a round could not do, or a run from before they were kept
    if best is None:
        lock = {"locked": False, "reason": "no submission was scored"}
        write_json(path / "lock.json", lock)
        log("nothing to lock: no submission was scored")
    else:
        entry = entry_of(path, best)
        if lang.entry_hash(entry) != best["entry_sha256"]:
            raise RuntimeError("the best submission's entry on disk is not the one the verifier scored")
        write_json(path / "locked_entry.json", entry)
        lock = {"locked": True, "entry_sha256": best["entry_sha256"], "network": cfg["network"],
                **{k: best[k] for k in ("round", "submission", "name", "carried_as", "reward", "parts", "size", "claimed_fidelity",
                                        "expected_fidelity_other")}}
        first = Verifier(model, cfg["network"], cfg)
        if old_lock.get("entry_sha256") == lock["entry_sha256"] and old_lock.get("measured_again"):
            lock["measured_again"] = old_lock["measured_again"]
        elif not first.enumerated:
            # The locked entry was picked as the best of several scorings, each on its own sample, so its recorded
            # reward leans high. One more scoring on a fresh sample is a fair measurement of it.
            again = first.score(entry, claimed=best.get("claimed_fidelity"))
            if again["entry_sha256"] != lock["entry_sha256"]:
                raise RuntimeError("the second scoring did not score the locked entry")
            lock["measured_again"] = {k: again[k] for k in ("status", "reason", "reward", "parts", "inputs")}
        write_json(path / "lock.json", lock)
        log(f"locked {best['carried_as']} ({best['entry_sha256'][:12]}): " + describe(best))
        if lock.get("measured_again"):
            log(f"the locked entry on a fresh sample: {describe(lock['measured_again'])}")
        if second and old_seed.get("entry_sha256") == lock["entry_sha256"] and old_seed.get("status") == "scored":
            log(f"seed check on the second network (as measured before): {describe(old_seed)}")
        elif second:
            receipt = Verifier(model, second, cfg).score(read_json(path / "locked_entry.json"), claimed=best.get("expected_fidelity_other"))
            if receipt["entry_sha256"] != lock["entry_sha256"]:
                raise RuntimeError("the seed check did not score the locked entry")
            write_json(path / "seed_check.json", {**receipt, "at": now()})
            log(f"seed check on the second network: {describe(receipt)}")
    _baseline_step(path, model, log)
    report.write(path)
    log(f"report: {path / 'report.md'}")
    return lock


def _baseline_step(path: Path, model, log) -> None:
    """A baseline run is frozen, and the finished runs it serves are read against it; any other run is read
    against its baseline when one is frozen. For the record only: a failure here does not cost the run."""
    from . import baseline, report

    try:
        if baseline.is_baseline(model):
            baseline.freeze(path, log)
            for other in baseline.runs_it_serves(path):  # finished runs of its shape not yet read against it
                if baseline.reading(other).get("state") != "read" and baseline.compare(other, log).get("state") == "read":
                    report.write(other)
                    log(f"  {other.name} is read against this baseline: {other / 'report.md'}")
            return
        r = baseline.compare(path, log)
        if r["state"] == "pending":
            log(f"the baseline run {r['baseline_run']} has not finished; finish this run again once it has")
        elif r["state"] == "none":
            log(f"no baseline run for this shape and these conditions yet; to make one: {r['command']}")
    except Exception as exc:
        log(f"the baseline could not be read ({type(exc).__name__}: {exc})")


def run_baseline(cfg: dict, fake_scripts=None, out=print, prices: str | None = None, accept_changes: bool = False,
                 budget_usd: float | None = None) -> Path:
    """The baseline for a run's config: found and frozen if it is on record, resumed if it was stopped, made if it is
    not. One process at a time runs a baseline; another one waits for it."""
    from . import baseline

    if (prices or lang.PRICES) != lang.PRICES:
        raise RuntimeError(f"the run was measured under {prices}, and this code prices under {lang.PRICES}: no baseline "
                           "made now could be compared with it")
    from . import report

    p = baseline.plan(cfg, prices)
    path = p["path"]
    if p["state"] != "frozen":
        if not baseline.take_lock(path):
            baseline.wait_for(path, out)  # another process runs it: what it leaves is the baseline
        else:
            try:
                out(f"the baseline: {path} ({'a new run' if p['state'] == 'new' else p['state']})")
                if p["state"] == "new":
                    start(p["config"], fake_scripts=fake_scripts, out=out)
                elif p["state"] == "unfinished":
                    drive(path, fake_scripts=fake_scripts, out=out, accept_changes=accept_changes, budget_usd=budget_usd)
                else:
                    finish(path, out=out)
            finally:
                baseline.release_lock(path)
        p = baseline.plan(cfg, prices)
        if p["state"] != "frozen":
            out(f"the baseline run {p['path']} has not finished ({p['state']}); go on with it the same way")
            return p["path"]
    rec = baseline.freeze(p["path"], out)
    out(f"the baseline for these conditions is {p['path']}: {rec.get('chosen') or 'nothing'} frozen, {rec.get('label')} "
        f"({rec.get('why')})")
    for other in baseline.runs_it_serves(p["path"]):  # finished runs of this shape not yet read against it
        if baseline.reading(other).get("state") != "read" and baseline.compare(other, out).get("state") == "read":
            report.write(other)
            out(f"  {other.name} is read against it: {other / 'report.md'}")
    return p["path"]


def run_with_baseline(cfg: dict, fake_scripts=None, out=print, resume: Path | None = None, accept_changes: bool = False,
                      budget_usd: float | None = None) -> str:
    """A run and its baseline side by side: the baseline in a second process (what it prints goes to console.txt in
    its folder), the run in this one; at the end this one waits for the baseline and reads the run against it."""
    from . import baseline

    prices = read_json(Path(resume) / "run.json").get("prices") if resume else lang.PRICES
    if prices != lang.PRICES:
        raise RuntimeError(f"{resume} was measured under {prices}, and this code prices under {lang.PRICES}")
    p = baseline.plan(cfg, prices)
    child = None
    if p["state"] == "frozen":
        out(f"its baseline is on record: {p['path']}")
    elif baseline.running(p["path"]):
        out(f"its baseline is being run already, by process {baseline.running(p['path'])}: {p['path']}")
    else:
        p["path"].mkdir(parents=True, exist_ok=True)
        write_json(p["path"] / baseline.REQUEST, {"config": cfg, "prices": prices, "at": now()})
        cmd = [sys.executable, "-m", "level1.loop", "baseline", "--request", str(p["path"] / baseline.REQUEST)]
        cmd += (["--fake", str(fake_scripts)] if fake_scripts else []) + (["--accept-changes"] if accept_changes else [])
        console = open(p["path"] / "console.txt", "a")
        child = subprocess.Popen(cmd, cwd=str(REPO_ROOT), stdout=console, stderr=subprocess.STDOUT)
        out(f"its baseline goes alongside, in process {child.pid}: {p['path']} (what it prints: {p['path'] / 'console.txt'})")
    try:
        if resume:
            ended, path = drive(Path(resume), fake_scripts=fake_scripts, out=out, accept_changes=accept_changes, budget_usd=budget_usd), Path(resume)
        else:
            path = start(cfg, fake_scripts=fake_scripts, out=out)
            ended = "finished" if (path / "lock.json").is_file() else "stopped"
    finally:
        if child is not None:
            out(f"waiting for the baseline run to end (process {child.pid})")
            code = child.wait()
            console.close()
            out("the baseline run ended" + ("" if code == 0 else f" with exit code {code}; see {p['path'] / 'console.txt'}"))
        else:
            baseline.wait_for(p["path"], out)
    if ended == "finished" and baseline.reading(path).get("state") != "read":
        finish(path, out=out)
    return ended


def rescore(path: Path, reward_cfg: dict, out=print) -> list:
    """Every scored submission (and the seed check) under another reward, from the parts on record."""
    path = Path(path)
    rows = []
    for s in scored(completed(path)):
        r = rewards.compute(s["parts"], reward_cfg, s.get("claimed_fidelity"))
        rows.append({"entry": s["carried_as"], "name": s["name"], "recorded": s["reward"], "rescored": r["reward"]})
    seed = path / "seed_check.json"
    if seed.is_file():
        rec = read_json(seed)
        if rec["status"] == "scored":
            r = rewards.compute(rec["parts"], reward_cfg, rec.get("claimed_fidelity"))
            rows.append({"entry": "locked entry on the second network", "name": "", "recorded": rec["reward"], "rescored": r["reward"]})
    for row in rows:
        out(f"{row['entry']:38s} {row['name'] or '':24s} recorded {row['recorded']:.4f}   {reward_cfg} -> {row['rescored']:.4f}")
    if rows:
        top = max((r for r in rows if r["name"]), key=lambda r: r["rescored"], default=None)
        if top:
            out(f"highest under this reward: {top['entry']} ({top['name']})")
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--config", required=True)
    r.add_argument("--run-id")
    r.add_argument("--runs-dir")
    r.add_argument("--fake", help="a folder of scripted rounds: sets the backend to fake")
    r.add_argument("--sandbox", help="auto, hardened, container, landlock, seatbelt or process (default: the config's)")
    r.add_argument("--auth", choices=("login", "api_key"), help="who pays for the rounds (default: the config's)")
    r.add_argument("--budget-usd", type=float, help="api_key: what the whole run may spend, in dollars (default: the config's)")
    r.add_argument("--with-baseline", action="store_true", help="run the baseline of this shape and these conditions alongside, "
                   "unless it is on record (level1/baseline.py)")
    s = sub.add_parser("resume")
    s.add_argument("--run", required=True)
    s.add_argument("--fake")
    s.add_argument("--accept-changes", action="store_true")
    s.add_argument("--budget-usd", type=float, help="api_key: a new budget for the whole run, in dollars")
    s.add_argument("--with-baseline", action="store_true", help="run, or go on with, the run's baseline alongside")
    f = sub.add_parser("finish")
    f.add_argument("--run", required=True)
    b = sub.add_parser("baseline", help="the baseline for a config's or a run's shape and conditions: found, run or resumed, frozen")
    which = b.add_mutually_exclusive_group(required=True)
    which.add_argument("--config", help="a run config: the baseline for its conditions")
    which.add_argument("--run", help="a run folder: the baseline for its conditions")
    which.add_argument("--request", help=argparse.SUPPRESS)  # written by run --with-baseline for its second process
    b.add_argument("--runs-dir")
    b.add_argument("--sandbox", help="as for run (default: the config's)")
    b.add_argument("--auth", choices=("login", "api_key"), help="as for run (default: the config's)")
    b.add_argument("--budget-usd", type=float, help="api_key: what the baseline run may spend, in dollars")
    b.add_argument("--fake", help="a folder of scripted rounds: sets the backend to fake")
    b.add_argument("--accept-changes", action="store_true", help="go on with a stopped baseline run although the code changed")
    c = sub.add_parser("rescore")
    c.add_argument("--run", required=True)
    c.add_argument("--name", default="r_alpha", help="a reward of level1/rewards.py")
    c.add_argument("--alpha", type=float)
    c.add_argument("--penalty", type=float)
    c.add_argument("--param", action="append", default=[], metavar="KEY=VALUE", help="any other parameter of the reward")
    args = ap.parse_args(argv)
    from .backends import BackendError
    from .networks import ModelError
    from .sandbox import SandboxError

    try:
        if args.cmd == "run":
            over = {k: v for k, v in (("run_id", args.run_id), ("runs_dir", args.runs_dir), ("sandbox", args.sandbox),
                                      ("auth", args.auth)) if v}
            if args.budget_usd is not None:
                over["budget"] = {"run_usd": args.budget_usd}
            if args.fake:
                over["backend"] = "fake"
            cfg = run_config.load(args.config, over)
            if args.with_baseline:
                return 0 if run_with_baseline(cfg, fake_scripts=args.fake) == "finished" else 2
            start(cfg, fake_scripts=args.fake)
            return 0
        if args.cmd == "resume":
            if args.with_baseline:
                cfg = read_json(Path(args.run) / "run.json")["config"]
                return 0 if run_with_baseline(cfg, fake_scripts=args.fake, resume=Path(args.run), accept_changes=args.accept_changes,
                                              budget_usd=args.budget_usd) == "finished" else 2
            return 0 if drive(Path(args.run), fake_scripts=args.fake, accept_changes=args.accept_changes,
                              budget_usd=args.budget_usd) == "finished" else 2
        if args.cmd == "finish":
            finish(Path(args.run))
            return 0
        if args.cmd == "baseline":
            prices = None
            if args.request:
                req = read_json(Path(args.request))
                cfg, prices = req["config"], req.get("prices")
            elif args.run:
                run = read_json(Path(args.run) / "run.json")
                cfg, prices = run["config"], run.get("prices")
            else:
                over = {k: v for k, v in (("runs_dir", args.runs_dir), ("sandbox", args.sandbox), ("auth", args.auth)) if v}
                cfg = run_config.load(args.config, over)
            if args.run or args.request:
                cfg = run_config.merge(cfg, {k: v for k, v in (("runs_dir", args.runs_dir), ("sandbox", args.sandbox), ("auth", args.auth)) if v})
            if args.budget_usd is not None:
                cfg = run_config.merge(cfg, {"budget": {"run_usd": args.budget_usd}})
            if args.fake:
                cfg = {**cfg, "backend": "fake"}
            run_baseline(cfg, fake_scripts=args.fake, prices=prices, accept_changes=args.accept_changes, budget_usd=args.budget_usd)
            return 0
        rescore(Path(args.run), run_config.reward_from_args(args.name, args.param, args.alpha, args.penalty))
        return 0
    except (run_config.ConfigError, hygiene.HygieneError, BackendError, ModelError, SandboxError, RuntimeError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)  # what the user needs to read, without a traceback around it
        return 1


if __name__ == "__main__":
    sys.exit(main())
