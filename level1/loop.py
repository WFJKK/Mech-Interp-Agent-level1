"""The loop: the rounds on one network, then the lock, the seed check and the report.

    python -m level1.loop run --config config/pilot/lin-a.yaml [--run-id X] [--sandbox hardened] [--fake DIR]
    python -m level1.loop resume --run <run folder>
    python -m level1.loop finish --run <run folder>     the lock, the seed check and the report again, from the records
    python -m level1.loop rescore --run <run folder> --name r_alpha --alpha 1    the records under another reward
                                  (--param KEY=VALUE for any other parameter of a reward)

A run folder (outside the repository, <runs_dir>/<run_id>):

    run.json            the config, the commit, the hash of the trusted files, the sandbox and its probe
    log.jsonl           what happened, appended
    round_001/          prompt.md, session/ (session.py), response.md, meta.json, round.json
    locked_entry.json   the best entry of the rounds, as submitted
    lock.json           which submission it is, its hash and its receipt
    seed_check.json     the verifier's receipt for the locked entry on the second network
    report.md, report.json

A round is done when its round.json is written. `resume` starts at the first round that is not, and a
session that was cut off keeps its budgets and submissions. The rounds end at rounds_max, after `patience`
rounds in a row that did not raise the best reward, or after max_consecutive_refusals refusals in a row.

When they end, the submission with the highest reward is locked: the entry as it was submitted, byte for
byte (its hash is checked). The seed check hands that entry to the verifier on the second network. No agent
takes part and nothing of the entry can change, so what comes out is how an entry written on one network
does on one it was not written on.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

from . import hygiene, lang, prompt, rewards, run_config
from .backends import StopRun, make
from .networks import load_model
from .session import Session, surface
from .util import REPO_ROOT, append_jsonl, atomic_write, git_state, now, read_json, sha256_file, write_json
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
    path.mkdir(parents=True, exist_ok=True)
    log = Log(path, out)
    verifier = Verifier(model, cfg["network"], cfg)
    itself = verifier.score(verifier.reference)
    if itself["status"] != "scored" or abs(itself["reward"]) > 1e-12 or itself["parts"]["fidelity"] != 1.0:
        raise RuntimeError(f"the network written as an entry does not score 0: {describe(itself)}")
    sb = choose(cfg, protected_paths(cfg, model, path))
    first_prompt = prompt.build(cfg, model, 1, [], itself["parts"]["model_ops"], verifier.enumerated)
    hygiene.check({"prompt": first_prompt, **agent_surface(cfg, model, sb.mode)}, forbidden_strings(cfg, model))
    report = probe(sb, model, cfg["network"], path / "probe")
    try:
        (path / "probe").rmdir()  # the probe's target folder; it leaves nothing in it
    except OSError:
        pass
    check_probe(sb, report)
    for w in rewards.warnings(cfg["reward"]):
        log(f"WARNING: the reward: {w}")
    if not sb.hardened:
        log("WARNING: the sandbox is in process mode: the agent's code can read whatever you can, the withheld "
            "folder and the second network's weights included. A run that counts needs container or landlock mode.")
    write_json(path / "run.json", {
        "run_id": cfg["run_id"], "config": cfg, "created": now(), "git": git_state(), "trusted_hash": trusted_hash(model),
        "sandbox": {**sb.describe(), "probe": report}, "prices": lang.PRICES, "model_ops": itself["parts"]["model_ops"],
        "n_total": model.n_total, "enumerated": verifier.enumerated,
    })
    log(f"run {cfg['run_id']} at {path}: {cfg['model']} / {cfg['network']}, sandbox {sb.mode}, backend {cfg['backend']}")
    drive(path, fake_scripts=fake_scripts, out=out)
    return path


def drive(path: Path, fake_scripts=None, out=print, accept_changes: bool = False) -> str:
    """Run the rounds that are left, then finish. Returns how it ended: finished, or stopped (resume later)."""
    path = Path(path)
    run = read_json(path / "run.json")
    cfg = run["config"]
    log = Log(path, out)
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
        write_json(path / "lock.json", lock)
        log(f"locked {best['carried_as']} ({best['entry_sha256'][:12]}): " + describe(best))
        if second:
            receipt = Verifier(model, second, cfg).score(read_json(path / "locked_entry.json"), claimed=best.get("expected_fidelity_other"))
            if receipt["entry_sha256"] != lock["entry_sha256"]:
                raise RuntimeError("the seed check did not score the locked entry")
            write_json(path / "seed_check.json", {**receipt, "at": now()})
            log(f"seed check on the second network: {describe(receipt)}")
    report.write(path)
    log(f"report: {path / 'report.md'}")
    return lock


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
    r.add_argument("--sandbox", help="auto, hardened, container, landlock or process (default: the config's)")
    s = sub.add_parser("resume")
    s.add_argument("--run", required=True)
    s.add_argument("--fake")
    s.add_argument("--accept-changes", action="store_true")
    f = sub.add_parser("finish")
    f.add_argument("--run", required=True)
    c = sub.add_parser("rescore")
    c.add_argument("--run", required=True)
    c.add_argument("--name", default="r_alpha", help="a reward of level1/rewards.py")
    c.add_argument("--alpha", type=float)
    c.add_argument("--penalty", type=float)
    c.add_argument("--param", action="append", default=[], metavar="KEY=VALUE", help="any other parameter of the reward")
    args = ap.parse_args(argv)
    if args.cmd == "run":
        over = {k: v for k, v in (("run_id", args.run_id), ("runs_dir", args.runs_dir), ("sandbox", args.sandbox)) if v}
        if args.fake:
            over["backend"] = "fake"
        cfg = run_config.load(args.config, over)
        start(cfg, fake_scripts=args.fake)
        return 0
    if args.cmd == "resume":
        return 0 if drive(Path(args.run), fake_scripts=args.fake, accept_changes=args.accept_changes) == "finished" else 2
    if args.cmd == "finish":
        finish(Path(args.run))
        return 0
    rescore(Path(args.run), run_config.reward_from_args(args.name, args.param, args.alpha, args.penalty))
    return 0


if __name__ == "__main__":
    sys.exit(main())
