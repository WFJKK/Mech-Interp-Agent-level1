"""The report of a run: report.json for programs, report.md for people.

    python -m level1.report --model models/lin-a [--name r_alpha] [--alpha 1] [--param KEY=VALUE]

prints the withheld reference entries of a model under a reward, on each of its networks, with no run at all:
the quick way to see what a change to the reward does to entries that are understood.

Written from the run folder's records alone (the rounds, the lock, the seed check), plus the withheld
reference entries, which are scored here on both networks so that the agent's reward can be read against
what is known. Those references are never shown to the agent; this file runs on the verifier's side.

Next to the reward, the report says which submissions are on the frontier. Each input of the data set is
answered rightly by an entry, answered wrongly, or left to the network; the frontier is over the share
answered rightly (more is better), the share answered wrongly (less is better) and the cost (less is
better). A submission is on it when no other one is at least as good in all three and better in one. No
reward function enters into that. (Coverage is the two shares together and fidelity their ratio; on those
two axes an entry that answers more inputs wrongly would count as covering more.)

It also says what the agent did to get there: how many programs it ran in the sandbox and for how long, and
which of them reach for files or the system, for a person to read (in a weak sandbox nothing stops them).
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from . import budget, lang, rewards, run_config
from .networks import load_model
from .util import atomic_write, read_json, read_jsonl, write_json
from .verifier import DEFAULTS, Verifier

REFERENCE_SEED = 0  # references are scored on fixed samples when a data set is sampled: seeds 0, 1, ...
MEANED = ("coverage", "fidelity", "fidelity_low", "cost", "r", "r_rule", "b")

# Words in a program that reach beyond the tools the sandbox hands it. A match is a reason to read the program,
# not a verdict: in a hardened sandbox such a reach fails, and some of these have innocent uses.
REACHES = [(re.compile(rx), label) for rx, label in (
    (r"\bopen\s*\(", "open("), (r"\bos\.", "os."), (r"\bpathlib\b|\bPath\s*\(", "pathlib"), (r"\bglob\b", "glob"),
    (r"\bsubprocess\b", "subprocess"), (r"\bsocket\b", "socket"), (r"\burllib\b|\brequests\b|\bhttp\b", "network"),
    (r"\bimportlib\b|__import__", "import machinery"), (r"\bshutil\b", "shutil"), (r"\bsys\.", "sys."),
    (r"np\.load\b|np\.loadtxt|np\.fromfile|np\.genfromtxt", "numpy file reading"), (r"withheld", "the word withheld"),
    (r"\.\./", "a parent folder"), (r"\beval\s*\(|\bexec\s*\(", "eval or exec"))]


def _f(v, digits=4) -> str:
    return "" if v is None else f"{v:.{digits}f}"


def shares(parts: dict | None) -> tuple:
    """(share of the inputs answered rightly, share answered wrongly), or (None, None)."""
    if not parts or parts.get("fidelity") is None:
        return (0.0, 0.0) if parts and parts.get("coverage") == 0 else (None, None)
    return parts["coverage"] * parts["fidelity"], parts["coverage"] * (1 - parts["fidelity"])


def whole_cost(parts: dict | None) -> float | None:
    """The entry's one cost: its whole work per input of its region, over one forward pass."""
    return rewards.cost_of(parts)


def _parts_row(receipt: dict | None) -> list:
    """reward, coverage, fidelity, cost"""
    if not receipt or receipt.get("status") != "scored":
        return ["refused" if receipt else "", "", "", ""]
    p = receipt["parts"]
    return [_f(receipt["reward"]), _f(p["coverage"]), _f(p["fidelity"], 5), _f(whole_cost(p))]


def _session_meta(path: Path, k: int) -> dict:
    """What a round's session took, from its meta.json: seconds, output tokens, what it was billed (an API key)
    or what the same tokens would have cost through the API (a subscription is not billed per call)."""
    f = Path(path) / f"round_{k:03d}" / "meta.json"
    meta = read_json(f) if f.is_file() else {}
    return {"seconds": meta.get("seconds"), "output_tokens": meta.get("output_tokens"),
            "api_equivalent_usd": meta.get("api_equivalent_usd"), "failed_tries": len(meta.get("failed_tries") or []),
            "cost_usd": meta.get("cost_usd"), "billed_to": meta.get("billed_to"), "auth": meta.get("auth")}


def exploration(path: Path, k: int) -> dict:
    """What the agent ran in the sandbox in round k: how many programs, for how long, which of them reach for
    files or the system (file name and what matched), what the guard inside Python refused, and which programs
    left the guard's record incomplete (a time limit, or code that stopped its process its own way)."""
    session = Path(path) / f"round_{k:03d}" / "session"
    calls = [e for e in read_jsonl(session / "tools.jsonl") if e.get("tool") == "run_python" and "error" not in (e.get("outcome") or {})]
    flagged = []
    folder = session / "python"
    programs = sorted(folder.glob("*.py")) if folder.is_dir() else []
    for prog in programs:
        try:
            text = prog.read_text()
        except (OSError, UnicodeDecodeError):
            continue
        found = sorted({label for rx, label in REACHES if rx.search(text)})
        if found:
            flagged.append({"program": f"round_{k:03d}/session/python/{prog.name}", "reaches": found})
    blocked, incomplete = [], []
    for f in sorted(folder.glob("*.audit.json")) if folder.is_dir() else []:
        try:
            a = read_json(f)
        except (OSError, ValueError):
            continue
        name = f"round_{k:03d}/session/python/{f.name.replace('.audit.json', '.py')}"
        blocked += [{"program": name, "event": b.get("event"), "target": b.get("target")} for b in a.get("blocked") or []]
        if a.get("record") != "complete":
            incomplete.append({"program": name, "record": a.get("record"), "status": a.get("status")})
    return {"programs": len(programs), "sandbox_seconds": round(sum(float(e.get("seconds") or 0) for e in calls), 1), "flagged": flagged,
            "blocked": blocked, "incomplete": incomplete}


def frontier(subs: list) -> set:
    """The submissions (by their carried_as names) that no other scored submission dominates in the share of
    inputs answered rightly, the share answered wrongly and the cost. Of entries with the same three numbers,
    the first stays."""
    pts = []
    for s in subs:
        p = s.get("parts")
        if s.get("status") == "scored" and p and p.get("coverage") and p.get("fidelity") is not None:
            right, wrong = shares(p)
            pts.append((s["carried_as"], right, wrong, whole_cost(p)))
    out = set()
    for i, (name, right, wrong, cost) in enumerate(pts):
        dominated = False
        for j, (_, r2, w2, k2) in enumerate(pts):
            if j == i or r2 < right or w2 > wrong or k2 > cost:
                continue
            if r2 > right or w2 < wrong or k2 < cost or j < i:
                dominated = True
                break
        if not dominated:
            out.add(name)
    return out


def _mean_receipt(receipts: list) -> dict:
    """One receipt for several scorings of one entry: the mean of the reward and of the measured shares.
    A refusal among them stands as it is."""
    for r in receipts:
        if r["status"] != "scored":
            return {k: r[k] for k in ("status", "reason", "reward", "parts", "size")}
    first, n = receipts[0], len(receipts)
    parts = dict(first["parts"])
    for k in MEANED:
        values = [r["parts"].get(k) for r in receipts]
        parts[k] = None if any(x is None for x in values) else sum(values) / n
    spread = max(r["reward"] for r in receipts) - min(r["reward"] for r in receipts)
    return {"status": "scored", "reason": None, "reward": sum(r["reward"] for r in receipts) / n, "parts": parts,
            "size": first["size"], "scorings": n, "reward_spread": spread}


def references(model, cfg: dict, networks: list) -> dict:
    """{reference name: {network id: receipt}}, with the network itself first. On a data set that is checked
    on every input a reference is scored once. On a sampled one it is scored on verifier.reference_samples
    fixed samples and the mean is given: one sample can sit a few thousandths off, the same way for every
    reference, which would tilt the comparison with an entry scored on a sample of its own."""
    out: dict = {}
    for nid in networks:
        v = Verifier(model, nid, cfg)
        k = 1 if v.enumerated else max(1, int(v.cfg["reference_samples"]))
        entries = {"the network itself": v.reference, **model.reference_entries(v.weights)}
        for name, entry in entries.items():
            out.setdefault(name, {})[nid] = _mean_receipt([v.score(entry, seed=REFERENCE_SEED + i) for i in range(k)])
    return out


def _references_scorings(refs: dict) -> int:
    """On how many samples each reference was scored; 0 when every input was checked."""
    return max([r.get("scorings") or 0 for by_net in refs.values() for r in by_net.values()] or [0])


def _references_rows(refs: dict) -> list:
    """The reference entries as a table. Scored on samples, a last column gives how far one scoring's reward
    lay from another's (highest less lowest), which is the noise to keep in mind when comparing."""
    several = _references_scorings(refs) > 1
    out = ["| Entry | Network | Reward | Coverage | Fidelity | Cost |" + (" Reward, highest less lowest |" if several else ""),
           "| --- | --- | --- | --- | --- | --- |" + (" --- |" if several else "")]
    for name, by_net in refs.items():
        for nid, r in by_net.items():
            spread = [_f(r.get("reward_spread"))] if several else []
            out.append("| " + " | ".join([name, nid] + _parts_row(r) + spread) + " |")
    return out


def _references_note(refs: dict) -> str:
    """How the references were scored, when that was on samples."""
    k = _references_scorings(refs)
    return f" Each number is the mean of {k} scorings on fixed samples." if k > 1 else ""


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
    on_second = {(e.get("carried_as"), e.get("entry_sha256")): e for e in read_jsonl(path / "seed_scores.jsonl")}
    for s in subs:
        scored = s["status"] == "scored"
        e = on_second.get((s["carried_as"], s.get("entry_sha256")))
        s["second_network"] = {k: e.get(k) for k in ("network", "status", "reward", "parts", "inputs")} if e else None
        s["whole_cost"] = whole_cost(s.get("parts")) if scored else None
        s["right_share"], s["wrong_share"] = shares(s.get("parts")) if scored else (None, None)
        s["on_frontier"] = s["carried_as"] in on_frontier
    return {
        "run_id": run["run_id"], "created": run["created"], "git": run["git"], "model": cfg["model"], "networks": networks,
        "backend": cfg["backend"], "agent_model": cfg["agent_model"], "effort": cfg["effort"], "reward": cfg["reward"],
        "prices": run["prices"], "model_ops": run["model_ops"], "n_total": run["n_total"], "enumerated": run["enumerated"],
        "n_inputs": run.get("n_inputs"),
        "sandbox": {k: v for k, v in run["sandbox"].items() if k not in ("probe", "guard_probe")},
        "sandbox_escapes": run["sandbox"]["probe"]["critical_escapes"],
        "guard_escapes": (run["sandbox"].get("guard_probe") or {}).get("critical_escapes"), "ended": ended,
        "rounds": [{"round": rd["round"], "submissions": len(rd["submissions"]), "improved": rd["improved"],
                    "best_reward_after": rd["best_reward_after"], "stop_reason": rd.get("stop_reason"),
                    "tools_used": rd.get("tools_used"), **_session_meta(path, rd["round"]),
                    "exploration": exploration(path, rd["round"])} for rd in rounds],
        "submissions": subs,
        "lock": lock, "seed_check": seed, "locked_entry": entry,
        "locked_entry_shown": lang.show(entry) if entry else None,
        "references": references(model, cfg, networks), "notes": notes, "task": model.task().get("task"),
        "auth": cfg.get("auth", "login"),
        "spend": budget.state(path, cfg["budget"]) if cfg.get("auth") == "api_key" else None,
    }


def _sandbox_note(rep: dict) -> str:
    if rep["sandbox"]["hardened"]:
        return ""
    if rep.get("guard_escapes") is not None:
        return (" (the system does not keep the agent's code in; a guard inside Python blocked every escape the probe "
                "tried and records every attempt, but it shares its process with that code: a fence, not a wall)")
    return " (weak: the agent's code could read any file, the withheld folder included)"


def markdown(rep: dict) -> str:
    first = rep["networks"][0]
    second = rep["networks"][1] if len(rep["networks"]) > 1 else None
    checked = ("every one checked" if rep["enumerated"] else
               f"{rep['n_inputs']:,} drawn afresh for each scoring" if rep.get("n_inputs") else "sampled for each scoring")
    out = [f"# Run {rep['run_id']}", ""]
    out += [f"- Model {rep['model']}, rounds on {first}" + (f", seed check on {second}" if second else "") + ".",
            f"- Agent: {rep['agent_model']} at effort {rep['effort']}, through the {rep['backend']} backend.",
            f"- Data set: {rep['n_total']:,} inputs, {checked}. One forward pass: {rep['model_ops']:,} operations ({rep['prices']}).",
            "- Reward: " + rep["reward"].get("name", "r_alpha") + "".join(f", {k} = {v}" for k, v in rep["reward"].items() if k != "name") + ".",
            f"- Sandbox: {rep['sandbox']['mode']}" + _sandbox_note(rep) + ".",
            f"- The rounds ended: {rep['ended'] or 'not yet'}.", ""]

    out += ["## The locked entry", ""]
    lock, seed = rep["lock"], rep["seed_check"]
    if not lock or not lock.get("locked"):
        out += ["No submission was scored, so nothing was locked.", ""]
    else:
        out += [f"Submission {lock['carried_as']} (saved as `{lock['name']}`), {lock['size']['nodes']} nodes, "
                f"{lock['size']['const_scalars']} numbers in constants, sha256 `{lock['entry_sha256'][:16]}`.", "",
                "| Network | Reward | Coverage | Fidelity | Cost | Fidelity claimed |",
                "| --- | --- | --- | --- | --- | --- |",
                "| " + " | ".join([f"{first} (the rounds)"] + _parts_row({"status": "scored", **lock}) + [_f(lock.get("claimed_fidelity"), 5)]) + " |"]
        if lock.get("measured_again"):
            out += ["| " + " | ".join([f"{first} (measured again, on a fresh sample)"] + _parts_row(lock["measured_again"])
                                      + [_f(lock.get("claimed_fidelity"), 5)]) + " |"]
        if seed:
            out += ["| " + " | ".join([f"{second} (the seed check)"] + _parts_row(seed) + [_f(lock.get("expected_fidelity_other"), 5)]) + " |"]
            if seed.get("status") != "scored":
                out += ["", f"The seed check was refused: {seed.get('reason')}"]
        if lock.get("measured_again"):
            out += ["", "The first row is the best of several scorings, each on its own sample, so it leans high; the second "
                    "row is one more scoring of the same entry."]
        out += ["", "```", rep["locked_entry_shown"] or "", "```", ""]

    out += ["## Reference entries", "",
            "Scored here for comparison. They are withheld from the agent." + _references_note(rep["references"])
            + (f" The task: {rep['task']}" if rep.get("task") else ""), ""]
    out += _references_rows(rep["references"]) + [""]

    out += ["## Rounds", "",
            "| Round | Submissions | Best reward so far | Raised it | Ended with | Tool calls | Minutes | Output tokens | Programs run | Sandbox minutes |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for rd in rep["rounds"]:
        minutes = "" if rd.get("seconds") is None else f"{rd['seconds'] / 60:.1f}"
        tokens = "" if not rd.get("output_tokens") else f"{rd['output_tokens']:,}"
        ex = rd.get("exploration") or {}
        out.append(f"| {rd['round']} | {rd['submissions']} | {_f(rd['best_reward_after'])} | {'yes' if rd['improved'] else 'no'} | "
                   f"{rd['stop_reason'] or ''} | " + ", ".join(f"{v} {k}" for k, v in (rd["tools_used"] or {}).items())
                   + f" | {minutes} | {tokens} | {ex.get('programs', '')} | {ex.get('sandbox_seconds', 0) / 60:.1f} |")
    billed = [rd for rd in rep["rounds"] if rd.get("auth") == "api_key" or rd.get("billed_to") in ("apiKeyHelper", "/login managed key")]
    usd = [rd["api_equivalent_usd"] for rd in rep["rounds"] if isinstance(rd.get("api_equivalent_usd"), (int, float))]
    if billed:
        spent = sum(rd["cost_usd"] for rd in billed if isinstance(rd.get("cost_usd"), (int, float)))
        out += ["", f"The rounds were billed to an API key: ${spent:.2f} by Claude Code's count"
                + (f" (spend.jsonl counts ${rep['spend']['spent']:.2f}, with every try that reported no cost at its cap)."
                   if rep.get("spend") and abs(rep["spend"]["spent"] - spent) > 0.005 else ".")]
    elif usd:
        out += ["", f"The same tokens through the API would have cost about ${sum(usd):.2f} (a subscription is not billed per call)."]

    flagged = [f for rd in rep["rounds"] for f in (rd.get("exploration") or {}).get("flagged", [])]
    blocked = [b for rd in rep["rounds"] for b in (rd.get("exploration") or {}).get("blocked", [])]
    incomplete = [b for rd in rep["rounds"] for b in (rd.get("exploration") or {}).get("incomplete", [])]
    guarded = rep.get("guard_escapes") is not None
    out += ["", "## What the agent ran", "",
            "Every program is kept in full in its round's `session/python/` folder, with what it got back; the round's "
            "`response.md` has what the agent wrote, and `stream_NN.jsonl` everything the CLI emitted."]
    if guarded:
        if blocked:
            out += ["", f"The guard inside Python refused {len(blocked)} attempt{'s' if len(blocked) != 1 else ''} to reach files, "
                    "the network or other programs:", ""]
            out += [f"- `{b['program']}`: {b['event']} {b['target']}" for b in blocked[:40]]
            if len(blocked) > 40:
                out += [f"- and {len(blocked) - 40} more, in the programs' `.audit.json` files"]
        else:
            out += ["", "The guard inside Python refused nothing: no program tried to reach files outside its folder, the "
                    "network or other programs."]
        if incomplete:
            out += ["", "Programs whose guard record is incomplete (a time limit, or code that stopped its process its own "
                    "way); read these:", ""]
            out += [f"- `{b['program']}`: record {b['record']}, run {b['status']}" for b in incomplete]
    if flagged:
        out += ["", "Programs whose text reaches for files or the system. A match is a reason to read the program, not a verdict"
                + ("" if rep["sandbox"]["hardened"] or guarded else "; in this run's weak sandbox nothing stopped such a reach") + ":", ""]
        out += [f"- `{f['program']}`: {', '.join(f['reaches'])}" for f in flagged]
    else:
        out += ["", "No program's text reaches for files or the system."]

    out += ["", "## Submissions", "",
            "Cost: everything the entry computes, per input of its region, over one forward pass. Right and wrong: the "
            "share of the inputs the entry answers rightly and wrongly. On the frontier: no other submission is at least "
            "as good in right share, wrong share and cost, and better in one. On the second network: the reward and the "
            "fidelity of the same entry on the seed check's network, every submission on one fixed sample, scored after "
            "its round; the agent never saw these, and they chose nothing.", "",
            "| Entry | Saved as | Reward | Coverage | Fidelity | Cost | Right | Wrong | On the frontier | Claimed | Overclaim "
            "| On the second network | Fidelity there |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for s in rep["submissions"]:
        two = s.get("second_network") or {}
        there = [_f(two.get("reward")), _f((two.get("parts") or {}).get("fidelity"), 5)] if two.get("status") == "scored" else \
            ([f"refused: {str(two.get('reason'))[:60]}", ""] if two else ["", ""])
        if s["status"] == "scored":
            out.append("| " + " | ".join([s["carried_as"], f"`{s['name']}`"] + _parts_row(s) + [
                _f(s.get("right_share")), _f(s.get("wrong_share"), 5), "yes" if s.get("on_frontier") else "no",
                _f(s.get("claimed_fidelity"), 5), _f(s.get("overclaim"), 5)] + there) + " |")
        else:
            out.append(f"| {s['carried_as']} | `{s['name']}` | refused: {str(s['reason'])[:100]} | | | | | | | | | | |")
    best_there = max((s for s in rep["submissions"] if (s.get("second_network") or {}).get("status") == "scored"),
                     key=lambda s: s["second_network"]["reward"], default=None)
    if best_there is not None and rep.get("lock") and rep["lock"].get("locked"):
        locked = next((s for s in rep["submissions"] if s["carried_as"] == rep["lock"]["carried_as"]), None)
        mine = (locked or {}).get("second_network") or {}
        out += ["", f"On the second network the best submission was {best_there['carried_as']} (`{best_there['name']}`), "
                f"{best_there['second_network']['reward']:.4f}; the locked entry, {rep['lock']['carried_as']}, scored "
                + (f"{mine['reward']:.4f} there on the same sample." if mine.get("status") == "scored" else "nothing there.")]
    out += ["", "## The agent's closing notes", "", "\n".join("> " + line for line in (rep["notes"] or "(none)").splitlines()), ""]
    return "\n".join(out)


def write(path: Path) -> dict:
    path = Path(path)
    rep = build(path)
    write_json(path / "report.json", rep)
    atomic_write(path / "report.md", markdown(rep))
    return rep


def reference_table(model, reward_cfg: dict, verifier_cfg: dict | None = None) -> str:
    """The withheld reference entries under a reward, on every network of the model, as a table."""
    rewards.check(reward_cfg)
    networks = [n["id"] for n in model.config["networks"]]
    refs = references(model, {"reward": reward_cfg, "verifier": verifier_cfg or {}}, networks)
    out = ["Reward: " + reward_cfg.get("name", "r_alpha") + "".join(f", {k} = {v}" for k, v in reward_cfg.items() if k != "name")]
    out += [f"Warning: {w}." for w in rewards.warnings(reward_cfg)]
    out += [_references_note(refs).strip()] if _references_note(refs) else []
    out += [""] + _references_rows(refs)
    return "\n".join(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="The withheld reference entries of a model under a reward, on each of its networks.")
    ap.add_argument("--model", required=True, help="a model folder, for example models/lin-a")
    ap.add_argument("--name", default="r_alpha", help="a reward of level1/rewards.py")
    ap.add_argument("--alpha", type=float)
    ap.add_argument("--penalty", type=float)
    ap.add_argument("--param", action="append", default=[], metavar="KEY=VALUE", help="any other parameter of the reward")
    ap.add_argument("--check", choices=("auto", "all", "sample"), default="auto", help="every input, or a sample (verifier.check)")
    ap.add_argument("--sample-size", type=int, default=DEFAULTS["sample_size"])
    ap.add_argument("--reference-samples", type=int, default=DEFAULTS["reference_samples"],
                    help="on a sampled data set: how many fixed samples each entry is scored on")
    args = ap.parse_args(argv)
    print(reference_table(load_model(args.model), run_config.reward_from_args(args.name, args.param, args.alpha, args.penalty),
                          {"check": args.check, "sample_size": args.sample_size, "reference_samples": args.reference_samples}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
