"""The report of a run: report.json for programs, report.md for people.

    python -m level1.report --model models/lin-a [--name r_alpha] [--alpha 1] [--param KEY=VALUE]

prints the withheld reference entries of a model under a reward, on each of its networks, with no run at all:
the quick way to see what a change to the reward does to entries that are understood.

Written from the run folder's records alone (the rounds, the lock, the seed check), plus the withheld
reference entries, which are scored here on both networks so that the agent's reward can be read against
what is known. Those references are never shown to the agent; this file runs on the verifier's side.

Next to the reward, the report says which submissions are on the frontier of the three things measured:
coverage, fidelity and cost, where the cost is the entry's whole work (prediction, rule and build) per
input of its region, in forward passes. A submission is on the frontier when no other one is at least as
good in all three and better in one. No reward function enters into that.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import lang, rewards, run_config
from .networks import load_model
from .util import atomic_write, read_json, read_jsonl, write_json
from .verifier import Verifier

REFERENCE_SEED = 0  # references are scored on one fixed sample when a data set is sampled


def _f(v, digits=4) -> str:
    return "" if v is None else f"{v:.{digits}f}"


def _parts_row(receipt: dict | None) -> list:
    if not receipt or receipt.get("status") != "scored":
        return ["refused" if receipt else "", "", "", "", "", ""]
    p = receipt["parts"]
    return [_f(receipt["reward"]), _f(p["coverage"]), _f(p["fidelity"], 5), _f(p["r"]), _f(p["r_rule"]), _f(p["b"])]


def _session_meta(path: Path, k: int) -> dict:
    """What a round's session took, from its meta.json: seconds, output tokens, and what the same tokens
    would have cost through the API when the backend says (a subscription is not billed per call)."""
    f = Path(path) / f"round_{k:03d}" / "meta.json"
    meta = read_json(f) if f.is_file() else {}
    return {"seconds": meta.get("seconds"), "output_tokens": meta.get("output_tokens"),
            "api_equivalent_usd": meta.get("api_equivalent_usd"), "failed_tries": len(meta.get("failed_tries") or [])}


def whole_cost(parts: dict) -> float | None:
    """The entry's whole work per input of its region, in forward passes: r + (r_rule + b) / coverage."""
    if not parts or not parts.get("coverage"):
        return None
    return parts["r"] + (parts["r_rule"] + parts["b"]) / parts["coverage"]


def frontier(subs: list) -> set:
    """The submissions (by their carried_as names) that no other scored submission dominates in coverage,
    fidelity and whole cost. Of entries with the same three numbers, the first stays."""
    pts = []
    for s in subs:
        p = s.get("parts")
        if s.get("status") == "scored" and p and p.get("coverage") and p.get("fidelity") is not None:
            pts.append((s["carried_as"], p["coverage"], p["fidelity"], whole_cost(p)))
    out = set()
    for i, (name, cov, fid, cost) in enumerate(pts):
        dominated = False
        for j, (_, c2, f2, k2) in enumerate(pts):
            if j == i or c2 < cov or f2 < fid or k2 > cost:
                continue
            if c2 > cov or f2 > fid or k2 < cost or j < i:
                dominated = True
                break
        if not dominated:
            out.add(name)
    return out


def references(model, cfg: dict, networks: list) -> dict:
    """{reference name: {network id: receipt}}, with the network itself first."""
    out: dict = {}
    for nid in networks:
        v = Verifier(model, nid, cfg)
        entries = {"the network itself": v.reference, **model.reference_entries(v.weights)}
        for name, entry in entries.items():
            r = v.score(entry, seed=REFERENCE_SEED)
            out.setdefault(name, {})[nid] = {k: r[k] for k in ("status", "reason", "reward", "parts", "size")}
    return out


def build(path: Path) -> dict:
    from .loop import completed  # the loop writes the records this reads

    path = Path(path)
    run = read_json(path / "run.json")
    cfg = run["config"]
    model = load_model(cfg["model"])
    rounds = completed(path)
    lock = read_json(path / "lock.json") if (path / "lock.json").is_file() else None
    seed = read_json(path / "seed_check.json") if (path / "seed_check.json").is_file() else None
    second = (cfg.get("seed_check") or {}).get("network")
    networks = [cfg["network"]] + ([second] if second else [])
    ended = next((e.get("ended") or e.get("stopped") for e in reversed(read_jsonl(path / "log.jsonl")) if e.get("ended") or e.get("stopped")), None)
    notes = next((rd["notes"] for rd in reversed(rounds) if rd.get("notes")), "")
    entry = read_json(path / "locked_entry.json") if (path / "locked_entry.json").is_file() else None
    subs = [dict(s) for rd in rounds for s in rd["submissions"]]
    on_frontier = frontier(subs)
    for s in subs:
        s["whole_cost"] = whole_cost(s.get("parts")) if s["status"] == "scored" else None
        s["on_frontier"] = s["carried_as"] in on_frontier
    return {
        "run_id": run["run_id"], "created": run["created"], "git": run["git"], "model": cfg["model"], "networks": networks,
        "backend": cfg["backend"], "agent_model": cfg["agent_model"], "effort": cfg["effort"], "reward": cfg["reward"],
        "prices": run["prices"], "model_ops": run["model_ops"], "n_total": run["n_total"], "enumerated": run["enumerated"],
        "sandbox": {k: v for k, v in run["sandbox"].items() if k != "probe"},
        "sandbox_escapes": run["sandbox"]["probe"]["critical_escapes"], "ended": ended,
        "rounds": [{"round": rd["round"], "submissions": len(rd["submissions"]), "improved": rd["improved"],
                    "best_reward_after": rd["best_reward_after"], "stop_reason": rd.get("stop_reason"),
                    "tools_used": rd.get("tools_used"), **_session_meta(path, rd["round"])} for rd in rounds],
        "submissions": subs,
        "lock": lock, "seed_check": seed, "locked_entry": entry,
        "locked_entry_shown": lang.show(entry) if entry else None,
        "references": references(model, cfg, networks), "notes": notes, "task": model.task().get("task"),
    }


def markdown(rep: dict) -> str:
    first = rep["networks"][0]
    second = rep["networks"][1] if len(rep["networks"]) > 1 else None
    out = [f"# Run {rep['run_id']}", ""]
    out += [f"- Model {rep['model']}, rounds on {first}" + (f", seed check on {second}" if second else "") + ".",
            f"- Agent: {rep['agent_model']} at effort {rep['effort']}, through the {rep['backend']} backend.",
            f"- Data set: {rep['n_total']:,} inputs, " + ("every one checked" if rep["enumerated"] else "sampled per scoring")
            + f". One forward pass: {rep['model_ops']:,} operations ({rep['prices']}).",
            "- Reward: " + rep["reward"].get("name", "r_alpha") + "".join(f", {k} = {v}" for k, v in rep["reward"].items() if k != "name") + ".",
            f"- Sandbox: {rep['sandbox']['mode']}" + ("" if rep["sandbox"]["hardened"] else
                                                     " (weak: the agent's code could read any file, the withheld folder included)") + ".",
            f"- The rounds ended: {rep['ended'] or 'not yet'}.", ""]

    out += ["## The locked entry", ""]
    lock, seed = rep["lock"], rep["seed_check"]
    if not lock or not lock.get("locked"):
        out += ["No submission was scored, so nothing was locked.", ""]
    else:
        out += [f"Submission {lock['carried_as']} (saved as `{lock['name']}`), {lock['size']['nodes']} nodes, "
                f"{lock['size']['const_scalars']} numbers in constants, sha256 `{lock['entry_sha256'][:16]}`.", "",
                "| Network | Reward | Coverage | Fidelity | r | r_rule | b | Fidelity claimed |",
                "| --- | --- | --- | --- | --- | --- | --- | --- |",
                "| " + " | ".join([f"{first} (the rounds)"] + _parts_row({"status": "scored", **lock}) + [_f(lock.get("claimed_fidelity"), 5)]) + " |"]
        if seed:
            out += ["| " + " | ".join([f"{second} (the seed check)"] + _parts_row(seed) + [_f(lock.get("expected_fidelity_other"), 5)]) + " |"]
            if seed.get("status") != "scored":
                out += ["", f"The seed check was refused: {seed.get('reason')}"]
        out += ["", "```", rep["locked_entry_shown"] or "", "```", ""]

    out += ["## Reference entries", "",
            "Scored here for comparison. They are withheld from the agent." + (f" The task: {rep['task']}" if rep.get("task") else ""), "",
            "| Entry | Network | Reward | Coverage | Fidelity | r | r_rule | b |", "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for name, by_net in rep["references"].items():
        for nid, r in by_net.items():
            out.append("| " + " | ".join([name, nid] + _parts_row(r)) + " |")
    out.append("")

    out += ["## Rounds", "", "| Round | Submissions | Best reward so far | Raised it | Ended with | Tool calls | Minutes | Output tokens |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for rd in rep["rounds"]:
        minutes = "" if rd.get("seconds") is None else f"{rd['seconds'] / 60:.1f}"
        tokens = "" if not rd.get("output_tokens") else f"{rd['output_tokens']:,}"
        out.append(f"| {rd['round']} | {rd['submissions']} | {_f(rd['best_reward_after'])} | {'yes' if rd['improved'] else 'no'} | "
                   f"{rd['stop_reason'] or ''} | " + ", ".join(f"{v} {k}" for k, v in (rd["tools_used"] or {}).items())
                   + f" | {minutes} | {tokens} |")
    usd = [rd["api_equivalent_usd"] for rd in rep["rounds"] if isinstance(rd.get("api_equivalent_usd"), (int, float))]
    if usd:
        out += ["", f"The same tokens through the API would have cost about ${sum(usd):.2f} (a subscription is not billed per call)."]
    out += ["", "## Submissions", "",
            "Whole cost: the entry's prediction, rule and build per input of its region, in forward passes. On the "
            "frontier: no other submission is at least as good in coverage, fidelity and whole cost, and better in one.", "",
            "| Entry | Saved as | Reward | Coverage | Fidelity | r | r_rule | b | Whole cost | On the frontier | Claimed | Overclaim |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for s in rep["submissions"]:
        if s["status"] == "scored":
            out.append("| " + " | ".join([s["carried_as"], f"`{s['name']}`"] + _parts_row(s) + [
                _f(s.get("whole_cost")), "yes" if s.get("on_frontier") else "no",
                _f(s.get("claimed_fidelity"), 5), _f(s.get("overclaim"), 5)]) + " |")
        else:
            out.append(f"| {s['carried_as']} | `{s['name']}` | refused: {str(s['reason'])[:100]} | | | | | | | | | |")
    out += ["", "## The agent's closing notes", "", "\n".join("> " + line for line in (rep["notes"] or "(none)").splitlines()), ""]
    return "\n".join(out)


def write(path: Path) -> dict:
    path = Path(path)
    rep = build(path)
    write_json(path / "report.json", rep)
    atomic_write(path / "report.md", markdown(rep))
    return rep


def reference_table(model, reward_cfg: dict) -> str:
    """The withheld reference entries under a reward, on every network of the model, as a table."""
    rewards.check(reward_cfg)
    networks = [n["id"] for n in model.config["networks"]]
    refs = references(model, {"reward": reward_cfg}, networks)
    out = ["Reward: " + reward_cfg.get("name", "r_alpha") + "".join(f", {k} = {v}" for k, v in reward_cfg.items() if k != "name")]
    out += [f"Warning: {w}." for w in rewards.warnings(reward_cfg)]
    out += ["",
           "| Entry | Network | Reward | Coverage | Fidelity | r | r_rule | b |", "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for name, by_net in refs.items():
        for nid, r in by_net.items():
            out.append("| " + " | ".join([name, nid] + _parts_row(r)) + " |")
    return "\n".join(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="The withheld reference entries of a model under a reward, on each of its networks.")
    ap.add_argument("--model", required=True, help="a model folder, for example models/lin-a")
    ap.add_argument("--name", default="r_alpha", help="a reward of level1/rewards.py")
    ap.add_argument("--alpha", type=float)
    ap.add_argument("--penalty", type=float)
    ap.add_argument("--param", action="append", default=[], metavar="KEY=VALUE", help="any other parameter of the reward")
    args = ap.parse_args(argv)
    print(reference_table(load_model(args.model), run_config.reward_from_args(args.name, args.param, args.alpha, args.penalty)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
