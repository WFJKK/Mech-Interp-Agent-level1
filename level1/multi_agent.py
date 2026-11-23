"""Team research and global integration, with a shared measured archive.

This driver is opt-in. Its sessions use the ordinary backend, sandbox and verifier. A
lead assigns bounded regional investigations, workers submit on those conditional
input distributions, and the lead submits complete programs on the original one.
Only submissions measured on the full domain may become the global winner, whoever wrote them. The coordinator owns
the archive and the final lock; neither worker notes nor held-out results select it.

Session plans, indices and input snapshots are written before a backend is called.
Each session has its own directory but spends against the parent run. Worker calls
may overlap; archive updates and state transitions have one writer.
"""

from __future__ import annotations

import copy
import fcntl
import hashlib
import json
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from pathlib import Path

from . import artifacts, budget, global_baseline, hygiene, lang, prompt, regions, report, research_modes, rewards, synthesis
from .backends import StopRun, make
from .networks import load_model
from .session import Session
from .util import REPO_ROOT, append_jsonl, atomic_write, now, read_json, write_json
from .verifier import Verifier


ROOT = "multi_agent"
ALL = {"kind": "all"}


@contextmanager
def _controller_lock(path: Path):
    directory = Path(path) / ROOT
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "controller.lock").open("a") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("another controller is already using this multi-agent run") from exc
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _options(cfg: dict) -> dict:
    return cfg["orchestration"]


def _state_path(path: Path) -> Path:
    return Path(path) / ROOT / "state.json"


def _check_saved_workflow(path: Path, cfg: dict) -> None:
    if _state_path(path).exists():
        saved = read_json(_state_path(path))
        if saved.get("mode", research_modes.mode(cfg)) != research_modes.mode(cfg):
            raise RuntimeError("the saved workflow mode changed; resume with its original configuration or start a new run")
        version = research_modes.validation_version(cfg)
        if saved.get("baseline_validation_version", version) != version:
            raise RuntimeError("the saved baseline validation policy changed; use its original configuration or start a new run")


def _archive_path(path: Path) -> Path:
    return Path(path) / ROOT / "archive.json"


def _empty_archive() -> dict:
    return {"version": 1, "candidates": [], "global": {"best": None, "frontier": []},
            "regions": {}, "baseline_scores": {}, "drafts": []}


def read_archive(path: Path) -> dict:
    target = _archive_path(path)
    return read_json(target) if target.exists() else _empty_archive()


def _candidate_map(archive: dict) -> dict:
    return {candidate["id"]: candidate for candidate in archive["candidates"]}


def _usable(candidate: dict) -> bool:
    receipt = candidate["selection"]
    return receipt.get("status") == "scored" and receipt.get("reward") is not None


def _frontier(candidates: list[dict]) -> list[str]:
    """Use the existing right-share/wrong-share/cost frontier within one scope."""
    distinct, seen = [], set()
    for candidate in candidates:
        key = candidate["entry_sha256"]
        if key not in seen and _usable(candidate):
            seen.add(key)
            distinct.append(candidate)
    rows = [{"carried_as": c["id"], **c["selection"]} for c in distinct]
    selected = report.frontier(rows)
    return [c["id"] for c in distinct if c["id"] in selected]


def update_frontiers(archive: dict) -> dict:
    """Rebuild selections from development receipts, never from final audits."""
    archive["candidates"].sort(key=lambda c: (c["session_id"], c["submitted"]["submission"]))
    global_candidates = [c for c in archive["candidates"] if c["scope_id"] == "all"]
    scored_global = [c for c in global_candidates if _usable(c)]
    best = max(scored_global, key=lambda c: c["selection"]["reward"], default=None)
    archive["global"] = {"best": best["id"] if best else None, "frontier": _frontier(global_candidates)}
    groups = {}
    for candidate in archive["candidates"]:
        if candidate["role"] == "regional" or candidate["scope_id"] != "all":
            groups.setdefault(candidate["scope_id"], []).append(candidate)
    archive["regions"] = {
        key: {"scope": candidates[0]["scope"], "frontier": _frontier(candidates),
              "best": max((c for c in candidates if _usable(c)),
                          key=lambda c: c["selection"]["reward"], default={}).get("id")}
        for key, candidates in groups.items()
    }
    return archive


def _signature(archive: dict) -> dict:
    candidates = _candidate_map(archive)

    def value(candidate_id):
        if candidate_id is None:
            return None
        candidate = candidates[candidate_id]
        return {"hash": candidate["entry_sha256"], "reward": candidate["selection"]["reward"]}

    return {"global": value(archive["global"]["best"]),
            "regional": {key: [value(cid) for cid in group["frontier"]]
                         for key, group in archive["regions"].items()}}


def progress(before: dict, after: dict) -> dict:
    """Protected regional progress is not contingent on a global reward increase."""
    old, new = before.get("global"), after.get("global")
    global_improved = new is not None and (old is None or new["reward"] > old["reward"] + 1e-12)
    regional_improved = any(values and values != before.get("regional", {}).get(key)
                            for key, values in after.get("regional", {}).items())
    return {"global_improved": global_improved, "regional_improved": regional_improved,
            "improved": global_improved or regional_improved}


def _scope_cfg(cfg: dict, scope: dict) -> dict:
    out = copy.deepcopy(cfg)
    out["verifier"]["scope"] = regions.normalize_scope(scope)
    out["verifier"]["sample_seed"] = int(_options(cfg)["development_seed"])
    return out


def _baselines(path: Path, cfg: dict) -> dict | None:
    """Read the frozen global comparison, or snapshot an operator-supplied one."""
    if research_modes.mode(cfg) == "trained":
        return None
    target = Path(path) / ROOT / "baseline_entry.json"
    if target.exists():
        entry = read_json(target)
        metadata = Path(path) / ROOT / "baseline.json"
        if metadata.exists():
            frozen = read_json(metadata)
            if frozen.get("entry_sha256") != lang.entry_hash(entry):
                raise RuntimeError("the frozen global baseline changed after selection")
        return entry
    source = _options(cfg).get("baseline_entry")
    if not source:
        return None
    if research_modes.validation_version(cfg) >= 2:
        # New supplied entries take the same validation route as generated ones.
        # Until it succeeds there is no comparison to carry into research.
        return None
    source = Path(source).expanduser()
    if not source.is_absolute():
        source = REPO_ROOT / source
    entry = read_json(source)
    lang.check_structure(entry, cfg["entry"]["max_nodes"], cfg["entry"]["max_const_scalars"])
    write_json(target, entry)
    write_json(Path(path) / ROOT / "baseline.json", {
        "source": "provided", "status": "provided", "entry_sha256": lang.entry_hash(entry),
        "reason": "A supplied frozen program; no independent baseline-agent search was run.", "at": now()})
    return entry


def _comparison(path: Path, cfg: dict, model, archive: dict, scope: dict) -> None:
    key = regions.scope_id(scope)
    if key in archive["baseline_scores"]:
        return
    entry = _baselines(path, cfg)
    if entry is None:
        return
    verifier = Verifier(model, cfg["network"], _scope_cfg(cfg, scope))
    archive["baseline_scores"][key] = verifier.score(entry, seed=int(_options(cfg)["development_seed"]))


def _entry(path: Path, candidate: dict) -> dict:
    entry = read_json(Path(path) / candidate["entry_path"])
    if lang.entry_hash(entry) != candidate["entry_sha256"]:
        raise RuntimeError(f"the archived entry {candidate['id']} changed after it was scored")
    return entry


def archive_summary(archive: dict, limit: int) -> str:
    """Only development results enter agent context, never audit/holdout files."""
    candidates = _candidate_map(archive)
    sections = []

    def row(cid):
        candidate = candidates[cid]
        receipt = candidate["selection"]
        p = receipt["parts"]
        scope_meta = receipt.get("scope") or {}
        return (f"{cid}: scope {candidate['scope_id']}; reward {receipt['reward']:.6f}; "
                f"coverage {p['coverage']:.6f}; fidelity {p['fidelity']}; cost {rewards.cost_of(p)}; "
                f"fidelity lower bound {p.get('fidelity_low')}; "
                f"outer scope mass {scope_meta.get('mass')} (exact: {scope_meta.get('mass_exact')}); "
                f"scope cardinality {scope_meta.get('cardinality')}; workload {scope_meta.get('workload_size')}; "
                f"build operations {p.get('build_ops')}; prediction operations {p.get('predict_ops')}")

    if archive["global"]["best"]:
        sections.append("Global incumbent:\n" + row(archive["global"]["best"]))
    for key, group in archive["regions"].items():
        ids = sorted(group["frontier"], key=lambda cid: candidates[cid]["selection"]["reward"], reverse=True)
        sections.append(f"Regional scope {key}: {json.dumps(group['scope'], sort_keys=True)}\n" +
                        "\n".join(row(cid) for cid in ids[:limit]))
    for key, receipt in archive["baseline_scores"].items():
        if receipt["status"] == "scored":
            parts = receipt["parts"]
            exact = (parts.get("coverage") == 1.0 and parts.get("n_inputs", 0) > 0
                     and parts.get("n_agree") == parts.get("n_in_region") == parts.get("n_inputs"))
            sections.append(f"Frozen comparison on {key}: reward {receipt['reward']:.6f}, "
                            f"fidelity {parts['fidelity']}, cost {rewards.cost_of(parts)}. "
                            + ("Full coverage and agreement on these checked inputs." if exact else
                               "It failed full coverage or agreement here; do not treat its reward gap as a validated generic improvement."))
    drafts = artifacts.describe(archive)
    if drafts:
        sections.append(drafts)
    return "\n\n".join(sections) or "No measured candidates yet."


def _snapshot(archive: dict) -> dict:
    return copy.deepcopy(archive)


def _carried(path: Path, cfg: dict, archive: dict) -> dict:
    out = {c["id"]: _entry(path, c) for c in archive["candidates"] if _usable(c)}
    best = archive["global"]["best"]
    if best:
        out["best_global"] = out[best]
    entry = _baselines(path, cfg)
    if entry is not None:
        out["comparison_entry"] = entry
    out.update(artifacts.carried(path, archive, out))
    return out


def _allocation(cfg: dict, role: str, cycle: int, workers: int) -> tuple[int, int, float, float]:
    """Tool quotas sum to at most the old run's quota, rather than multiplying it."""
    opt, session = _options(cfg), cfg["session"]
    total_api, per_session = float(cfg["budget"]["run_usd"]), float(cfg["budget"]["round_usd"])
    summary = synthesis.fraction(cfg)
    baseline_share = global_baseline.fraction(cfg)
    if role == "baseline":
        cap = min(per_session, total_api * baseline_share)
        return (math.floor(session["python_calls"] * cfg["rounds_max"] * baseline_share),
                int(opt["baseline_submissions"]), cap, cap)
    if role == "summary":
        cap = min(per_session, total_api * summary)
        return 0, 0, cap, cap
    planning, regional = float(opt["planning_fraction"]), float(opt["regional_fraction"])
    global_fraction = 1.0 - planning - regional
    total_api *= 1.0 - summary - baseline_share
    python_calls = session["python_calls"] * (1.0 - baseline_share)
    if cycle == 0:
        cap = min(per_session, total_api * planning)
        return math.floor(python_calls * cfg["rounds_max"] * planning), 0, cap, cap
    cycle_cap = min(per_session, total_api * (1.0 - planning) / cfg["rounds_max"])
    if role == "regional":
        calls = math.floor(python_calls * regional) // workers
        cap = cycle_cap * regional / (1.0 - planning) / workers
        return calls, int(opt.get("regional_submissions", 1)), cap, cycle_cap
    calls = math.floor(python_calls * global_fraction)
    cap = cycle_cap * global_fraction / (1.0 - planning)
    return calls, session["submissions"] - workers * int(opt.get("regional_submissions", 1)), cap, cycle_cap


def _quota(cfg: dict, base: float, role: str, cycle: int, workers: int, integer: bool):
    opt = _options(cfg)
    summary = synthesis.fraction(cfg)
    baseline_share = global_baseline.fraction(cfg)
    if role == "baseline":
        value = base * cfg["rounds_max"] * baseline_share
        return math.floor(value) if integer else value
    if role == "summary":
        value = base * cfg["rounds_max"] * summary
        return math.floor(value) if integer else value
    base *= 1.0 - summary - baseline_share
    if cycle == 0:
        value = base * cfg["rounds_max"] * opt["planning_fraction"]
    elif role == "regional":
        value = base * opt["regional_fraction"]
        value = math.floor(value) // workers if integer else value / workers
    else:
        value = base * (1.0 - opt["planning_fraction"] - opt["regional_fraction"])
    return math.floor(value) if integer else value


def role_config(cfg: dict, plan: dict) -> dict:
    out = (global_baseline.config_for(cfg) if plan["role"] == "baseline"
           else _scope_cfg(cfg, plan["scope"]))
    calls, submissions, cap, cycle_cap = _allocation(cfg, plan["role"], plan["cycle"], plan["worker_count"])
    out["session"]["python_calls"] = calls
    out["session"]["submissions"] = submissions
    out["session"]["max_turns"] = _quota(cfg, cfg["session"]["max_turns"], plan["role"], plan["cycle"], plan["worker_count"], True)
    out["session"]["seconds"] = _quota(cfg, cfg["session"]["seconds"], plan["role"], plan["cycle"], plan["worker_count"], False)
    if out["session"]["max_turns"] < 1 and cfg["backend"] != "fake":
        raise RuntimeError("the turn budget is too small for the configured research-role allocation")
    out["budget"].update(round_usd=cap, cycle_usd=cycle_cap,
                         cycle_id=plan["role"] if plan["role"] in ("baseline", "summary") else
                         "planning" if plan["cycle"] == 0 else f"cycle-{plan['cycle']:03d}")
    max_assignments = (_options(cfg)["workers"] if plan["role"] == "lead" and plan["cycle"] < cfg["rounds_max"] else 0)
    out["research_task"] = {"role": plan["role"], "task": plan["task"], "cycle": plan["cycle"],
                            "session_id": plan["id"], "max_assignments": max_assignments}
    checkpoint = _options(cfg).get("checkpoint_python_calls")
    if (plan["role"] == "baseline" or (plan["role"] == "lead" and plan["cycle"] > 0)) and checkpoint is not None:
        out["research_task"]["checkpoint_python_calls"] = min(int(checkpoint), calls)
    return out


def _new_plan(state: dict, role: str, cycle: int, scope: dict, task: str, workers: int, snapshot=None, history=None) -> dict:
    index = state["next_index"]
    state["next_index"] += 1
    sid = f"a{index + 1:04d}"
    plan = {"id": sid, "index": index, "role": role, "cycle": cycle,
            "scope": regions.normalize_scope(scope), "task": str(task), "worker_count": workers,
            "directory": f"{ROOT}/sessions/{sid}-{role}", "snapshot": snapshot, "history": history,
            "status": "planned"}
    state["sessions"][sid] = plan
    return plan


def _metadata(record: dict) -> dict:
    if not isinstance(record, dict):
        return {}
    meta = record.get("meta")
    meta = dict(meta) if isinstance(meta, dict) else {}
    if "stop_reason" not in meta and record.get("stop_reason") is not None:
        meta["stop_reason"] = record["stop_reason"]
    return meta


def _is_refusal(meta: dict) -> bool:
    if not isinstance(meta, dict):
        return False
    if meta.get("stop_reason") == "refusal":
        return True
    events = meta.get("model_events")
    return isinstance(events, list) and any(isinstance(event, dict) and event.get("subtype") in
                                            ("model_refusal_no_fallback", "refusal") for event in events)


def _research_notes(record: dict) -> str:
    meta = _metadata(record)
    if _is_refusal(meta) or meta.get("stop_reason") in ("error", "api_error", "backend_error"):
        return ("This session ended without a research explanation. Any saved programs and checked "
                "submissions remain on record. Provider diagnostics are retained separately for the operator.")
    notes = record.get("notes", "")
    reason = "timeout" if meta.get("timed_out") else meta.get("stop_reason")
    if reason in ("timeout", "turn_limit", "budget"):
        return (f"Partial research notes: this session ended with {reason}. Saved programs and checked "
                f"submissions remain separately recorded.\n\n{notes}")
    return notes


def _safe_history(path: Path, state: dict, history: list[dict]) -> list[dict]:
    """Also sanitize frozen histories made by older code, without editing their raw records."""
    rows = copy.deepcopy(history)
    for row in rows:
        plan = state["sessions"].get(row.get("session_id"))
        source = Path(path) / plan["directory"] / "record.json" if plan else None
        record = read_json(source) if source and source.exists() else row
        row["notes"] = _research_notes(record)
    return rows


def _history(path: Path, state: dict, before_index: int) -> list[dict]:
    rows = []
    for plan in sorted(state["sessions"].values(), key=lambda item: item["index"]):
        record_path = Path(path) / plan["directory"] / "record.json"
        if plan["role"] not in ("baseline", "summary") and plan["index"] < before_index and record_path.exists():
            record = read_json(record_path)
            rows.append({"session_id": plan["id"], "role": plan["role"], "scope": plan["scope"],
                         "notes": _research_notes(record), "submissions": record["submissions"]})
    return rows


def _refusal_stop(path: Path, cfg: dict, state: dict) -> str | None:
    """Latch the first threshold reached in session-index order, including legacy records."""
    if state.get("refusal_stop"):
        return state["refusal_stop"]["reason"]
    limit = int(cfg["max_consecutive_refusals"])
    streak = []
    for plan in sorted(state["sessions"].values(), key=lambda item: item["index"]):
        if plan["role"] == "summary":
            continue
        directory = Path(path) / plan["directory"]
        if (directory / "record.json").exists():
            meta = _metadata(read_json(directory / "record.json"))
        elif (directory / "meta.json").exists():
            meta = read_json(directory / "meta.json")
        else:
            streak = []
            continue
        streak = [*streak, plan["id"]] if _is_refusal(meta) else []
        if len(streak) >= limit:
            reason = f"the model declined {len(streak)} consecutive research sessions; no new agent sessions will start"
            state["refusal_stop"] = {"reason": reason, "session_ids": streak, "threshold": limit}
            return reason
    return None


def _run_session(path: Path, cfg: dict, model, state: dict, plan: dict, fake_scripts, out) -> dict:
    """A worker writes only its own directory; the caller ingests its record."""
    directory = Path(path) / plan["directory"]
    completed = directory / "record.json"
    if completed.exists():
        return read_json(completed)
    role_cfg = role_config(cfg, plan)
    if plan["role"] == "baseline":
        model = load_model(role_cfg["model"])
    meta_path = directory / "meta.json"
    session_existed = (directory / "session" / "session.json").exists()
    exhausted = None
    if not meta_path.exists() and cfg["auth"] == "api_key":
        spent = budget.state(path, role_cfg["budget"])
        if budget.left(spent, plan["index"] + 1) < budget.MIN_CAP_USD:
            exhausted = budget.spent_why(spent, plan["index"] + 1)
            if not session_existed:
                raise StopRun(exhausted)
    archive = plan["snapshot"]
    carried = {} if plan["role"] == "baseline" else _carried(path, cfg, archive)
    session = Session.create(directory / "session", role_cfg, role_cfg["network"], path, carried)
    history = _safe_history(path, state, plan["history"] or [])
    prompt_path = directory / ("prompt.history_safe.md" if history != (plan["history"] or []) else "prompt.md")
    if not prompt_path.exists():
        text = prompt.build_multi(cfg=role_cfg, model=model, session_index=plan["index"], role=plan["role"],
                                  task=plan["task"], archive_summary=archive_summary(archive, _options(cfg)["archive_per_scope"]),
                                  history=history)
        # Generated notes and tasks are discoveries, as in the original loop. Check
        # the coordinator-authored prompt separately from those discoveries.
        ours = prompt.build_multi(cfg=role_cfg, model=model, session_index=plan["index"], role=plan["role"],
                                  task="Investigate the assigned input distribution.", archive_summary="", history=[])
        from . import loop
        hygiene.check({"prompt": ours, **loop.agent_surface(role_cfg, model, role_cfg["sandbox"])},
                      loop.forbidden_strings(role_cfg, model))
        atomic_write(prompt_path, text)
    if meta_path.exists():
        meta = read_json(meta_path)
        response_text = (directory / "response.md").read_text()
    else:
        if exhausted:
            # A finished or interrupted CLI can consume the remaining budget before
            # the coordinator writes its record. Its persisted submissions still
            # belong in the archive; recovering them requires no new model call.
            response_text = f"The session ended with its saved work: {exhausted}."
            meta = {"backend": cfg["backend"], "stop_reason": "budget", "recovered": True,
                    "stop_details": {"budget": exhausted}}
        else:
            backend = make(role_cfg, log=out, fake_scripts=fake_scripts)
            if not backend.available(plan["index"]):
                raise StopRun(f"no backend response for session {plan['id']}")
            response = backend.respond(prompt_path.read_text(), plan["index"], directory, session)
            response_text, meta = response.text, response.meta
        atomic_write(directory / "response.md", response_text)
        write_json(meta_path, meta)
    session.reload()
    from .loop import notes_of
    record = {"id": plan["id"], "index": plan["index"], "role": plan["role"], "cycle": plan["cycle"],
              "scope": plan["scope"], "notes": notes_of(response_text), "meta": meta,
              "submissions": session.receipts(), "assignments": session.state.get("assignments", []),
              "finish": session.state.get("finish"),
              "tools_used": session.state["used"], "prompt_file": prompt_path.name, "at": now()}
    write_json(completed, record)
    return record


def _ingest(path: Path, cfg: dict, model, archive: dict, plan: dict, record: dict) -> None:
    existing = {c["id"] for c in archive["candidates"]}
    _comparison(path, cfg, model, archive, plan["scope"])
    verifiers = {}
    for receipt in record["submissions"]:
        cid = f"{plan['id']}s{receipt['submission']}"
        if cid in existing:
            continue
        scope = regions.normalize_scope(receipt.get("claim_scope", plan["scope"]))
        sid = regions.scope_id(scope)
        if regions.scope_id(regions.intersect_scopes(plan["scope"], scope)) != sid:
            raise RuntimeError(f"submission {cid} claims a domain outside its assignment")
        if receipt.get("assigned_scope_id", regions.scope_id(plan["scope"])) != regions.scope_id(plan["scope"]):
            raise RuntimeError(f"submission {cid} changed its assigned domain")
        if receipt.get("scope") and receipt["scope"]["id"] != sid:
            raise RuntimeError(f"submission {cid} has inconsistent claim and measured domains")
        _comparison(path, cfg, model, archive, scope)
        entry_path = f"{plan['directory']}/session/submissions/{receipt['submission']:03d}/entry.json"
        entry = read_json(Path(path) / entry_path)
        digest = lang.entry_hash(entry)
        if receipt.get("entry_sha256") and receipt["entry_sha256"] != digest:
            raise RuntimeError(f"submission {cid} changed before it entered the archive")
        selection = receipt
        if receipt["status"] == "scored":
            verifier_key = lang.canonical(scope)
            if verifier_key not in verifiers:
                verifiers[verifier_key] = Verifier(model, cfg["network"], _scope_cfg(cfg, scope))
            selection = verifiers[verifier_key].score(entry, claimed=receipt.get("claimed_fidelity"),
                                                     seed=int(_options(cfg)["development_seed"]))
        archive["candidates"].append({"id": cid, "role": plan["role"], "session_id": plan["id"],
                                      "cycle": plan["cycle"], "scope": scope, "scope_id": regions.scope_id(scope),
                                      "entry_path": entry_path, "entry_sha256": digest, "selection": selection,
                                      "submitted": receipt,
                                      "expected_fidelity_other": receipt.get("expected_fidelity_other")})
    artifacts.collect(path, cfg, archive, plan)
    update_frontiers(archive)
    write_json(_archive_path(path), archive)


def _assignments(cfg: dict, record: dict) -> list[dict]:
    options = _options(cfg)
    limit = int(options["workers"])
    proposed = (record.get("assignments") or options.get("initial_tasks")
                or options.get("initial_regions") or [])
    out = []
    for assignment in proposed[:limit]:
        if not isinstance(assignment, dict) or not isinstance(assignment.get("task"), str):
            raise RuntimeError("an assignment needs a task")
        out.append({"scope": regions.normalize_scope(assignment.get("scope") or ALL), "task": assignment["task"]})
    return out


def _stop_cycles(state: dict, cfg: dict) -> str | None:
    completed = [cycle for cycle in state["cycles"] if cycle["status"] == "completed"]
    if len(completed) >= cfg["rounds_max"]:
        return "the configured research cycles are complete"
    tail = 0
    for cycle in reversed(completed):
        if cycle["progress"]["improved"]:
            break
        tail += 1
    if tail >= cfg["patience"]:
        return "neither the global nor a regional frontier improved within patience"
    return None


def drive(path, fake_scripts=None, out=print, budget_usd=None) -> str:
    with _controller_lock(Path(path)):
        if budget_usd is not None:
            cfg = read_json(Path(path) / "run.json")["config"]
            if cfg["auth"] != "api_key":
                raise RuntimeError("this run is not billed to an API key, so it has no budget to change")
            errors = budget.check({**cfg["budget"], "run_usd": budget_usd})
            if errors:
                raise RuntimeError("; ".join(errors))
            previous = budget.state(Path(path), cfg["budget"])
            budget.set_run_budget(Path(path), budget_usd, atomic=True)
            event = {"at": now(), "budget_usd_before": previous["run_usd"], "budget_usd": float(budget_usd),
                     "message": "the experiment-wide API budget was updated; planned role and cycle caps are unchanged"}
            append_jsonl(Path(path) / "log.jsonl", event)
            out(f"the experiment's API budget is now ${float(budget_usd):.2f} (was ${previous['run_usd']:.2f}); "
                "planned role and cycle caps are unchanged")
        return _drive(Path(path), fake_scripts=fake_scripts, out=out)


def _drive(path, fake_scripts=None, out=print) -> str:
    path = Path(path)
    run = read_json(path / "run.json")
    cfg = run["config"]
    _check_saved_workflow(path, cfg)
    if (path / ROOT / "lock.json").exists():
        _finish(path, out=out, fake_scripts=fake_scripts)
        return "finished"
    model = load_model(cfg["model"])
    directory = path / ROOT
    directory.mkdir(parents=True, exist_ok=True)
    _baselines(path, cfg)
    archive = read_archive(path)
    if _state_path(path).exists():
        state = read_json(_state_path(path))
    else:
        state = {"version": 1, "mode": research_modes.mode(cfg), "next_index": 0,
                 "baseline_validation_version": research_modes.validation_version(cfg),
                 "sessions": {}, "cycles": [], "stop_reason": None}
        if global_baseline.enabled(cfg):
            baseline_plan = _new_plan(state, "baseline", 0, ALL,
                                      "Develop a full-domain predictor exact for any parameter values, with low measured cost.",
                                      1, _empty_archive(), [])
            allocated = role_config(cfg, baseline_plan)
            baseline_plan["allocation"] = {"session": allocated["session"], "budget": allocated["budget"]}
            state["baseline"] = baseline_plan["id"]
        write_json(_state_path(path), state)

    def run_one(plan):
        out(f"{plan['id']}: {plan['role']} on {regions.scope_id(plan['scope'])}")
        record = _run_session(path, cfg, model, state, plan, fake_scripts, out)
        if plan["role"] != "baseline":
            _ingest(path, cfg, model, archive, plan, record)
        plan["status"] = "completed"
        write_json(_state_path(path), state)
        return record

    def refusal_barrier():
        reason = _refusal_stop(path, cfg, state)
        if reason:
            raise StopRun(reason)

    try:
        # Recovery and draft backfill require no new backend call. In particular,
        # an old planning session may have useful saved entries but no submissions.
        for plan in sorted(state["sessions"].values(), key=lambda item: item["index"]):
            if plan["role"] == "summary":
                continue
            folder = path / plan["directory"]
            if (folder / "record.json").exists() or ((folder / "meta.json").exists() and (folder / "response.md").exists()):
                run_one(plan)
        refusal_barrier()
        if global_baseline.needs_validation(cfg) and _options(cfg).get("baseline_entry"):
            result = global_baseline.validate_provided(path, cfg, out=out)
            if result.get("status") != "established":
                raise StopRun("the supplied global baseline was not established: " + str(result.get("reason", "validation failed")))
        if global_baseline.enabled(cfg):
            if "baseline" not in state:
                raise StopRun("this persisted research plan has no baseline phase; start a new run with the updated configuration")
            baseline_plan = state["sessions"][state["baseline"]]
            record = run_one(baseline_plan)
            control_cfg = role_config(cfg, baseline_plan)
            result = global_baseline.freeze(path, cfg, load_model(control_cfg["model"]), baseline_plan, record, out)
            if result.get("status") != "established":
                raise StopRun("the global baseline was not established: " + str(result.get("reason", "no validated global program")))
            refusal_barrier()
        if research_modes.mode(cfg) == "baseline":
            state["stop_reason"] = "the requested global baseline is established; no trained research was requested"
            write_json(_archive_path(path), archive)
            write_json(_state_path(path), state)
            _finish(path, out=out, fake_scripts=fake_scripts)
            return "finished"
        _comparison(path, cfg, model, archive, ALL)
        write_json(_archive_path(path), archive)
        if "planning" not in state:
            planning = _new_plan(state, "lead", 0, ALL,
                                 "Decide what each researcher works on in the first research phase, and give out the tasks.",
                                 int(_options(cfg)["workers"]), _snapshot(archive), [])
            state["planning"] = planning["id"]
            write_json(_state_path(path), state)
        planning = state["sessions"][state["planning"]]
        last_lead = run_one(planning)
        while True:
            refusal_barrier()
            stopped = _stop_cycles(state, cfg)
            if stopped:
                state["stop_reason"] = stopped
                break
            pending = next((cycle for cycle in state["cycles"] if cycle["status"] != "completed"), None)
            if pending is None:
                if state["cycles"]:
                    last = state["sessions"][state["cycles"][-1]["lead"]]
                    last_lead = read_json(path / last["directory"] / "record.json")
                finish = last_lead.get("finish") if isinstance(last_lead, dict) else None
                if finish:
                    if not archive["global"]["best"]:  # the tool refuses this; a guard for edited records
                        state["stop_reason"] = "the lead finished the research before a program for the whole network was scored"
                        write_json(_state_path(path), state)
                        out(state["stop_reason"])
                        write_report(path)
                        return "stopped"
                    state["stop_reason"] = f"the lead finished the research: {finish['reason']}"
                    write_json(_state_path(path), state)
                    out(state["stop_reason"])
                    break
                assignments = _assignments(cfg, last_lead)
                if not assignments:
                    state["stop_reason"] = "the lead gave out no tasks and no initial tasks are configured"
                    write_json(_state_path(path), state)
                    out(state["stop_reason"])
                    write_report(path)
                    return "stopped"
                number = len(state["cycles"]) + 1
                for assignment in assignments:
                    _comparison(path, cfg, model, archive, assignment["scope"])
                write_json(_archive_path(path), archive)
                snapshot = _snapshot(archive)
                history = _history(path, state, state["next_index"])
                workers = [_new_plan(state, "regional", number, task["scope"], task["task"], len(assignments), snapshot, history)
                           for task in assignments]
                lead = _new_plan(state, "lead", number, ALL,
                                 "Use the researchers' findings to improve the program for the whole network; give out the next tasks.",
                                 len(workers))
                pending = {"cycle": number, "workers": [worker["id"] for worker in workers], "lead": lead["id"],
                           "before": _signature(archive), "status": "planned"}
                state["cycles"].append(pending)
                write_json(_state_path(path), state)
            workers = [state["sessions"][sid] for sid in pending["workers"]]
            errors = []
            if _options(cfg)["parallel"] and len(workers) > 1:
                out(f"cycle {pending['cycle']}: running {len(workers)} researcher sessions")
                with ThreadPoolExecutor(max_workers=len(workers)) as pool:
                    futures = {pool.submit(_run_session, path, cfg, model, state, plan, fake_scripts, out): plan for plan in workers}
                    for future in as_completed(futures):
                        plan = futures[future]
                        try:
                            record = future.result()
                            _ingest(path, cfg, model, archive, plan, record)
                            plan["status"] = "completed"
                            write_json(_state_path(path), state)
                        except Exception as exc:
                            errors.append(exc)
                refusal_barrier()
                if errors:
                    raise errors[0]
            else:
                for plan in workers:
                    refusal_barrier()
                    run_one(plan)
            refusal_barrier()
            lead = state["sessions"][pending["lead"]]
            if lead["snapshot"] is None:
                lead["snapshot"] = _snapshot(archive)
                lead["history"] = _history(path, state, lead["index"])
                write_json(_state_path(path), state)
            last_lead = run_one(lead)
            pending.update(status="completed", progress=progress(pending["before"], _signature(archive)), at=now())
            write_json(_state_path(path), state)
            append_jsonl(directory / "cycles.jsonl", {k: v for k, v in pending.items() if k != "before"})
            refusal_barrier()
            write_report(path)
    except StopRun as exc:
        # A budget interruption can occur after save_entry but before a response
        # record exists. Preserve that work without promoting it to a submission.
        for plan in sorted(state["sessions"].values(), key=lambda item: item["index"]):
            if plan["role"] not in ("baseline", "summary"):
                artifacts.collect(path, cfg, archive, plan)
        write_json(_archive_path(path), archive)
        state["stop_reason"] = str(exc)
        write_json(_state_path(path), state)
        out(f"multi-agent research stopped: {exc}; its session plans and submissions are saved")
        write_report(path)
        return "stopped"
    write_json(_state_path(path), state)
    _finish(path, out=out, fake_scripts=fake_scripts)
    return "finished"


def _audit_seed(run_id: str, scope_id: str, purpose: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{run_id}:{scope_id}:{purpose}".encode()).digest()[:8], "big")


def _audit_candidate(path: Path, cfg: dict, model, candidate: dict) -> dict:
    target = path / ROOT / "audits" / f"{candidate['id']}.json"
    audit = read_json(target) if target.exists() else {"candidate_id": candidate["id"], "entry_sha256": candidate["entry_sha256"],
                                                     "scope": candidate["scope"]}
    if audit["entry_sha256"] != candidate["entry_sha256"] or audit["scope"] != candidate["scope"]:
        raise RuntimeError("the frozen candidate differs from its saved audit")
    entry, scoped = _entry(path, candidate), _scope_cfg(cfg, candidate["scope"])
    checks = [("fresh", cfg["network"])]
    if cfg["seed_check"].get("network"):
        checks.append(("held_out", cfg["seed_check"]["network"]))
    for purpose, network in checks:
        if purpose in audit:
            continue
        seed = _audit_seed(cfg["run_id"], candidate["scope_id"], purpose)
        claimed = (candidate["selection"].get("claimed_fidelity") if purpose == "fresh"
                   else candidate.get("expected_fidelity_other"))
        audit[purpose] = Verifier(model, network, scoped).score(entry, claimed=claimed, seed=seed)
        comparison = _baselines(path, cfg)
        if comparison is not None:
            audit[f"{purpose}_baseline"] = Verifier(model, network, scoped).score(comparison, seed=seed)
        write_json(target, audit)
    return audit


def finish(path, out=print, fake_scripts=None) -> dict:
    with _controller_lock(Path(path)):
        return _finish(Path(path), out=out, fake_scripts=fake_scripts)


def _finish(path, out=print, fake_scripts=None) -> dict:
    """Freeze the whole selection before evaluating either final input sample."""
    path = Path(path)
    run = read_json(path / "run.json")
    cfg, model = run["config"], load_model(run["config"]["model"])
    _check_saved_workflow(path, cfg)
    if global_baseline.needs_validation(cfg):
        status = _baseline_status(path, cfg)
        if status.get("status") != "established" or _baselines(path, cfg) is None:
            write_report(path)
            raise RuntimeError("the required global baseline is not established; resume the saved baseline phase before finishing research")
        if research_modes.validation_version(cfg) >= 2:
            global_baseline.verify_frozen(path, cfg)
    if research_modes.mode(cfg) == "baseline":
        return _finish_baseline(path, cfg, out)
    archive = read_archive(path)
    frozen_path = path / ROOT / "lock.json"
    if frozen_path.exists():
        frozen = read_json(frozen_path)
    else:
        ids = []
        if archive["global"]["best"]:
            ids.append(archive["global"]["best"])
        for region in archive["regions"].values():
            ids.extend(region["frontier"])
        candidates = _candidate_map(archive)
        frozen = {"version": 1, "global": archive["global"]["best"],
                  "regional": {key: value["frontier"] for key, value in archive["regions"].items()},
                  "candidates": [copy.deepcopy(candidates[cid]) for cid in dict.fromkeys(ids)], "at": now()}
        write_json(frozen_path, frozen)
    candidates = {candidate["id"]: candidate for candidate in frozen["candidates"]}
    global_id = frozen["global"]
    if global_id:
        candidate = candidates[global_id]
        if candidate["scope_id"] != "all":
            raise RuntimeError("a regional score cannot select the global program")
        write_json(path / "locked_entry.json", _entry(path, candidate))
        lock = {"locked": True, "candidate_id": global_id, "entry_sha256": candidate["entry_sha256"],
                "network": cfg["network"], "reward": candidate["selection"]["reward"],
                "parts": candidate["selection"]["parts"], "scope": candidate["scope"], "orchestration": "regional_global"}
    else:
        lock = {"locked": False, "reason": "no global submission was scored", "orchestration": "regional_global"}
    write_json(path / "lock.json", lock)
    for candidate in frozen["candidates"]:
        audit = _audit_candidate(path, cfg, model, candidate)
        if candidate["id"] == global_id:
            lock["measured_again"] = audit["fresh"]
            write_json(path / "lock.json", lock)
            if "held_out" in audit:
                write_json(path / "seed_check.json", audit["held_out"])
    _synthesize(path, cfg, model, frozen, archive, fake_scripts, out)
    write_report(path)
    out(f"locked {len(frozen['candidates'])} candidates; global choice: {global_id or 'none'}")
    return lock


def _finish_baseline(path: Path, cfg: dict, out=print) -> dict:
    """Complete the standalone verifier result without inventing a research winner."""
    status = _baseline_status(path, cfg)
    entry = _baselines(path, cfg)
    if status.get("status") != "established" or entry is None:
        raise RuntimeError("the required global baseline is not established; resume its saved phase")
    digest = lang.entry_hash(entry)
    frozen_path = path / ROOT / "lock.json"
    if frozen_path.exists():
        frozen = read_json(frozen_path)
        if frozen.get("mode") != "baseline" or frozen.get("baseline_entry_sha256") != digest:
            raise RuntimeError("the baseline-only completion differs from its frozen program")
    else:
        frozen = {"version": 1, "mode": "baseline", "global": None, "regional": {}, "candidates": [],
                  "baseline_entry_sha256": digest, "at": now()}
        write_json(frozen_path, frozen)
    lock = {"locked": False, "completed": True, "mode": "baseline", "orchestration": "regional_global",
            "baseline_established": True, "baseline_entry_sha256": digest,
            "reason": "Baseline-only workflow completed; no trained research candidate was requested."}
    write_json(path / "lock.json", lock)
    write_report(path)
    out(f"global baseline completed: {path / ROOT / 'baseline_entry.json'}; no trained research sessions ran")
    return lock


def _synthesize(path: Path, cfg: dict, model, frozen: dict, archive: dict, fake_scripts, out) -> dict:
    """The sole post-lock agent has no research tools and cannot enter the archive."""
    if not synthesis.enabled(cfg):
        return {"status": "disabled"}
    state = read_json(_state_path(path))
    halted = _refusal_stop(path, cfg, state)
    if halted:
        state["stop_reason"] = halted
        write_json(_state_path(path), state)
        previous = synthesis.report(path, cfg)
        if previous["status"] == "completed":
            return previous
        if not (path / "multi_agent/summary/response.json").exists():
            result = {"status": "stopped", "reason": halted, "blocked_by_refusals": True}
            write_json(path / "multi_agent/summary/status.json", result)
            out(f"final lead synthesis stopped: {halted}")
            return result
    if "summary" not in state:
        plan = _new_plan(state, "summary", cfg["rounds_max"] + 1, ALL,
                         "Explain the frozen global result and regional findings from the recorded evidence.", 1)
        plan["directory"] = f"{ROOT}/summary"
        state["summary"] = plan["id"]
        write_json(_state_path(path), state)
    plan = state["sessions"][state["summary"]]
    result = synthesis.run(path, cfg, model, plan, role_config(cfg, plan), frozen, archive,
                           _history(path, state, plan["index"]), fake_scripts=fake_scripts, out=out)
    plan["status"] = result["status"]
    write_json(_state_path(path), state)
    return result


def _baseline_status(path: Path, cfg: dict) -> dict:
    if research_modes.mode(cfg) == "trained":
        return {"source": None, "status": "not_configured",
                "reason": "Trained-only mode requested no baseline search or comparison."}
    saved = path / ROOT / "baseline.json"
    if saved.exists():
        return read_json(saved)
    if (path / ROOT / "baseline_entry.json").exists():
        return {"source": "provided", "status": "provided",
                "reason": "A frozen comparison supplied under the saved configuration."}
    if global_baseline.required(cfg):
        return {"source": "provided" if _options(cfg).get("baseline_entry") else "single_agent_search",
                "status": "pending", "reason": "The global baseline must pass the saved validation policy before acceptance."}
    return {"source": None, "status": "not_configured",
            "reason": "This saved configuration has no automatic baseline search or supplied program."}


def write_report(path) -> None:
    path = Path(path)
    run, archive = read_json(path / "run.json"), read_archive(path)
    state = read_json(_state_path(path)) if _state_path(path).exists() else {}
    records = []
    for plan in state.get("sessions", {}).values():
        target = path / plan["directory"] / "record.json"
        if target.exists():
            records.append(read_json(target))
    audits = {}
    audit_dir = path / ROOT / "audits"
    if audit_dir.exists():
        audits = {target.stem: read_json(target) for target in sorted(audit_dir.glob("*.json"))}
    api_budget = budget.state(path, run["config"]["budget"]) if run["config"]["auth"] == "api_key" else None
    mode = research_modes.mode(run["config"])
    result = {"orchestration": "regional_global", "mode": mode,
              "completed": (path / ROOT / "lock.json").is_file(), "network": run["config"]["network"],
              "archive": archive, "state": state, "sessions": records, "audits": audits}
    baseline_status = _baseline_status(path, run["config"])
    result["baseline"] = baseline_status
    summary = synthesis.report(path, run["config"])
    result["lead_summary"] = summary
    if api_budget is not None:
        result["api_budget"] = api_budget
    result["comparisons"] = {
        "development": {candidate["id"]: synthesis.comparison_result(
            candidate["selection"], archive["baseline_scores"].get(candidate["scope_id"]))
            for candidate in archive["candidates"]},
        "final": {cid: {label: synthesis.comparison_result(audit.get(label), audit.get(f"{label}_baseline"))
                        for label in ("fresh", "held_out") if label in audit} for cid, audit in audits.items()}}
    write_json(path / "report.json", result)
    if mode == "baseline":
        _write_baseline_report(path, result)
        return
    policy = ("Full coverage and agreement were required on the declared control and trained checkpoints."
              if baseline_status.get("trained_checkpoints_used_for_acceptance") else
              "This record does not establish acceptance on the declared trained checkpoints.")
    lines = ["# Team research and global integration", "", f"Workflow mode: `{mode}`.", "",
             f"Network: `{run['config']['network']}`.", "",
             "The same reward is measured separately within each declared scope. Regional rewards do not select the global program.",
             "All findings here are measured claims unless an independent certificate is supplied.", "",
             "## Global baseline", "", f"Status: `{baseline_status['status']}`.", "",
             str(baseline_status.get("reason") or "The frozen program passed the recorded parameter checks."), "",
             policy, "",
             "One frozen global program is scored on the target network within each claim's scope, using the same inputs and workload as the candidate. Regional comparisons do not launch additional baseline agents.", "",
             "A single-agent comparison search has different resources from the research team. A reward gap over this known program can include generic optimizations; it does not by itself measure an effect of training.", "",
             "## Global selection", "", f"Selected candidate: `{archive['global']['best'] or 'none'}`.", "",
             "## Development archive", "", "| Candidate | Role | Scope | Coverage | Fidelity | Cost | Reward | Comparison gap |",
             "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for candidate in archive["candidates"]:
        receipt = candidate["selection"]
        if receipt["status"] != "scored":
            continue
        p, key = receipt["parts"], candidate["scope_id"]
        comparison = archive["baseline_scores"].get(key)
        matched = result["comparisons"]["development"][candidate["id"]]
        gap = f"{matched['reward_gap']:+.6f}" if matched["valid"] else "unavailable"
        fidelity = "none" if p["fidelity"] is None else f"{p['fidelity']:.6f}"
        cost = rewards.cost_of(p)
        cost = "none" if cost is None else f"{cost:.6f}"
        lines.append(f"| {candidate['id']} | {candidate['role']} | {key} | {p['coverage']:.6f} | {fidelity} | {cost} | {receipt['reward']:.6f} | {gap} |")
    lines += ["", "## Regional frontiers", ""]
    for key, group in archive["regions"].items():
        lines.append(f"- `{key}`: {', '.join(group['frontier']) or 'no covered candidate'}; scope `{json.dumps(group['scope'], sort_keys=True)}`.")
        if group["best"]:
            metadata = _candidate_map(archive)[group["best"]]["selection"].get("scope") or {}
            mass_basis = "exact" if metadata.get("mass_exact") else "estimated"
            lines.append(f"  Outer probability ({mass_basis}): {metadata.get('mass')}; support size: {metadata.get('cardinality')}; "
                         f"declared workload: {metadata.get('workload_size')}.")
    if archive.get("drafts"):
        lines += ["", "## Saved draft artifacts", "", artifacts.describe(archive), ""]
    lines += ["", "## Final locked checks", "", "| Candidate | Check | Network | Fidelity | Reward | Baseline fidelity | Baseline reward | Reward gap |",
              "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for cid, audit in audits.items():
        for label in ("fresh", "held_out"):
            receipt = audit.get(label)
            if receipt and receipt["status"] == "scored":
                comparison = audit.get(f"{label}_baseline")
                if comparison and comparison.get("status") == "scored":
                    matched = result["comparisons"]["final"][cid][label]
                    gap = f"{matched['reward_gap']:+.6f}" if matched["valid"] else "unavailable (comparison invalid)"
                    details = f"{comparison['parts']['fidelity']} | {comparison['reward']:.6f} | {gap}"
                else:
                    details = "unavailable | unavailable | unavailable"
                lines.append(f"| {cid} | {label} | {receipt['network']} | {receipt['parts']['fidelity']} | {receipt['reward']:.6f} | {details} |")
    lines += ["", "A comparison gap is reported only when the baseline has full coverage and exact agreement on the matching target check. "
              "Other raw receipts are retained, but an invalid comparison has no validated improvement figure."]
    lines += ["", "## Execution", "", f"Completed cycles: {sum(c['status'] == 'completed' for c in state.get('cycles', []))}.",
              f"Stop reason: {state.get('stop_reason') or 'research is in progress'}.", "",
              "The raw session receipts, fixed-development comparisons and frozen final checks are kept separately. Final checks never feed research sessions or archive selection. A separate tool-less lead synthesis may describe them after locking.", ""]
    lines += ["## Final lead synthesis", "", f"Status: {summary['status']}.", ""]
    if summary.get("markdown"):
        lines += ["[Read the final lead synthesis](lead_summary.md).", ""]
    elif summary.get("reason"):
        lines += [summary["reason"], ""]
    if api_budget is not None:
        lines += [f"Counted API spending: ${api_budget['spent']:.2f} of the experiment's ${api_budget['run_usd']:.2f} budget, "
                  f"from the shared spending ledger. {budget.accounting_note(api_budget)}", ""]
    atomic_write(path / "report.md", "\n".join(lines))


def _write_baseline_report(path: Path, result: dict) -> None:
    """A standalone result is complete without a research winner or a lead synthesis."""
    status = result["baseline"]
    lines = ["# Global baseline", "", "Workflow mode: `baseline`.", "",
             f"Status: `{status['status']}`. Completed: `{result['completed']}`.", "",
             str(status.get("reason") or "Validation has not completed."), "",
             "This workflow runs one global investigator, or verifies a supplied program. "
             "It does not run trained research, regional investigators or a lead synthesis.", "",
             "Validation checks on trained checkpoints are verifier computations, not additional agent sessions.", "",
             "Agreement on the checked inputs is empirical evidence, not a proof for arbitrary inputs or weights.", ""]
    if status.get("entry_sha256"):
        lines += [f"Frozen program hash: `{status['entry_sha256']}`.", "",
                  "Program: `multi_agent/baseline_entry.json`. Validation record: `multi_agent/baseline.json`.", ""]
    if status.get("trained_checkpoints_used_for_acceptance"):
        lines += ["Acceptance required full coverage and agreement on all declared control and trained checkpoints. "
                  "Candidate rewards on trained checkpoints did not determine their ranking.", ""]
    else:
        lines += ["This record does not establish trained-checkpoint acceptance.", ""]
    for title, key in (("Control checks", "control_receipts"), ("Trained checkpoint checks", "trained_receipts")):
        lines += [f"## {title}", "", "| Checkpoint | Status | Coverage | Fidelity | Reward |",
                  "| --- | --- | ---: | ---: | ---: |"]
        rows = status.get(key) or {}
        for name, receipt in rows.items():
            parts = receipt.get("parts") or {}
            reward = receipt.get("reward")
            reward = f"{reward:.6f}" if isinstance(reward, (int, float)) else "unavailable"
            lines.append(f"| {name} | {receipt.get('status')} | {parts.get('coverage')} | {parts.get('fidelity')} | {reward} |")
        if not rows:
            lines.append("| none recorded | pending or unsuccessful | | | |")
        lines.append("")
    api_budget = result.get("api_budget")
    if api_budget is not None:
        lines += [f"Counted API spending: ${api_budget['spent']:.2f} of ${api_budget['run_usd']:.2f}. "
                  + budget.accounting_note(api_budget), ""]
    atomic_write(path / "report.md", "\n".join(lines))


def rescore(path, reward_cfg: dict, out=print) -> list[dict]:
    """Read-only development rescoring, with independent selections per scope."""
    archive = read_archive(Path(path))
    rows = []
    for candidate in archive["candidates"]:
        receipt = candidate["selection"]
        if receipt["status"] != "scored":
            continue
        claimed = receipt.get("claimed_fidelity", candidate["submitted"].get("claimed_fidelity"))
        value = rewards.compute(receipt["parts"], reward_cfg, claimed)["reward"]
        rows.append({"entry": candidate["id"], "name": candidate["submitted"].get("name"),
                     "role": candidate["role"], "scope_id": candidate["scope_id"],
                     "recorded": receipt["reward"], "rescored": value})
    out("Development submissions rescored within their original scopes. Final locked checks and selections are unchanged.")
    for row in rows:
        out(f"{row['entry']}  {row['role']}  {row['scope_id']}  recorded {row['recorded']:.6f}  rescored {row['rescored']:.6f}")
    groups = {}
    for row in rows:
        groups.setdefault(row["scope_id"], []).append(row)
    for scope_id, candidates in groups.items():
        best = max(candidates, key=lambda row: row["rescored"])
        label = "global" if scope_id == "all" else f"regional scope {scope_id}"
        out(f"Best {label} under this reward: {best['entry']} ({best['rescored']:.6f}).")
    return rows
