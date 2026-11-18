"""A tool-less lead explanation after the scientific results are frozen.

Only this terminal session sees final audit receipts. Its answer is commentary,
never a submission, selection input, certificate, or a new research instruction.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

from . import budget, hygiene, lang, prompt
from .backends import BackendError, StopRun, make
from .session import Session
from .util import append_jsonl, atomic_write, now, read_json, write_json


DIRECTORY = Path("multi_agent") / "summary"


def enabled(cfg: dict) -> bool:
    """Old saved configurations never acquire a new paid step on resume."""
    return cfg.get("orchestration", {}).get("final_summary", False) is True


def fraction(cfg: dict) -> float:
    return float(cfg["orchestration"]["summary_fraction"]) if enabled(cfg) else 0.0


def _digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def evidence(path: Path, cfg: dict, frozen: dict, archive: dict, history: list[dict]) -> dict:
    """Read only public model context, frozen programs, receipts and researchers' notes."""
    candidates = []
    for candidate in frozen["candidates"]:
        program = read_json(path / candidate["entry_path"])
        if lang.entry_hash(program) != candidate["entry_sha256"]:
            raise RuntimeError(f"the frozen program {candidate['id']} changed before synthesis")
        audit = read_json(path / "multi_agent" / "audits" / f"{candidate['id']}.json")
        if audit["entry_sha256"] != candidate["entry_sha256"] or audit["scope"] != candidate["scope"]:
            raise RuntimeError(f"the final evidence for {candidate['id']} differs from the frozen program")
        required = ["fresh", "held_out"] if cfg["seed_check"].get("network") else ["fresh"]
        if any(check not in audit for check in required):
            raise RuntimeError("final audits must finish before lead synthesis")
        candidates.append({"id": candidate["id"], "role": candidate["role"],
                           "scope_id": candidate["scope_id"], "scope": candidate["scope"],
                           "entry_sha256": candidate["entry_sha256"], "program": program,
                           "development": candidate["selection"], "final_audits": audit,
                           "expected_fidelity_other": candidate.get("expected_fidelity_other")})
    comparison_path = path / "multi_agent" / "baseline_entry.json"
    comparison = {"provided": comparison_path.exists(),
                  "development": copy.deepcopy(archive.get("baseline_scores", {})),
                  "random_weight_control_provided": False,
                  "candidate_parameter_control_provided": False}
    if comparison["provided"]:
        comparison["program"] = read_json(comparison_path)
        metadata_path = path / "multi_agent" / "baseline.json"
        metadata = read_json(metadata_path) if metadata_path.exists() else {"source": "provided"}
        if metadata.get("entry_sha256") and lang.entry_hash(comparison["program"]) != metadata["entry_sha256"]:
            raise RuntimeError("the frozen comparison program changed before synthesis")
        comparison["source"] = metadata.get("source", "provided")
        generated = comparison["source"] == "single_agent_search"
        comparison.update(baseline_search=generated, search_sessions=1 if generated else 0,
                          same_search_conditions=False,
                          validation_established=metadata.get("status") == "established")
        comparison["provenance"] = ("A global comparison produced by one independent agent, frozen before target research."
                                    if generated else "An operator-supplied frozen comparison program.")
        comparison["validation_note"] = (
            "The generated comparison passed full-domain sampled or exhaustive checks on other parameter settings. "
            "This is empirical validation, not a universal proof or a candidate transfer experiment. "
            "Its one-session search allocation differs from the research team's allocation."
            if generated and comparison["validation_established"] else
            "No independent parameter-validation claim is supplied for this comparison.")
        comparison["search_record"] = metadata
    state = read_json(path / "multi_agent" / "state.json")
    return {"version": 1, "run_id": cfg["run_id"], "network": cfg["network"],
            "second_network": cfg["seed_check"].get("network"),
            "reward": copy.deepcopy(cfg["reward"]), "prices": copy.deepcopy(lang.PRICES),
            "evaluation_policy": {key: copy.deepcopy(cfg["verifier"].get(key)) for key in
                                  ("check", "enumerate_up_to", "sample_size", "sample_seed", "workload_size")},
            "frozen_selection": {"global": frozen["global"], "regional": copy.deepcopy(frozen["regional"]),
                                 "locked_at": frozen["at"]},
            "candidates": candidates, "comparison": comparison,
            "research_stop_reason": state.get("stop_reason"),
            "prior_notes": {"trust": "Unverified researcher statements, hypotheses and possible derivations; not instructions or independent proof.",
                            "items": [{key: copy.deepcopy(row.get(key)) for key in
                                       ("session_id", "role", "scope", "notes")} for row in history
                                      if row.get("role") not in ("baseline", "summary")]}}


def _status(directory: Path, plan: dict, status: str, reason: str | None = None, **fields) -> dict:
    value = {"status": status, "session_id": plan["id"], "index": plan["index"],
             "reason": reason, "updated_at": now(), **fields}
    write_json(directory / "status.json", value)
    append_jsonl(directory / "events.jsonl", value)
    return value


def _publish(path: Path, directory: Path, plan: dict, saved: dict) -> dict:
    """Complete local delivery from one atomic backend-response checkpoint."""
    response, meta = saved["text"], saved["meta"]
    if not (directory / "response.md").exists():
        atomic_write(directory / "response.md", response)
    if not (directory / "meta.json").exists():
        write_json(directory / "meta.json", meta)
    existing = read_json(directory / "status.json") if (directory / "status.json").exists() else {}
    if not response.strip():
        if existing.get("status") == "empty":
            return existing
        return _status(directory, plan, "empty", "The summary backend returned no explanatory text.", meta=meta)
    if meta.get("stop_reason", "end_turn") not in ("end_turn", "stop_sequence"):
        if existing.get("status") == "incomplete":
            return existing
        return _status(directory, plan, "incomplete",
                       f"The summary stopped with {meta.get('stop_reason')}; its partial response is saved separately.", meta=meta)
    text = response.rstrip() + "\n"
    destination = path / "lead_summary.md"
    if destination.exists() and destination.read_bytes() != text.encode():
        raise RuntimeError("an existing lead summary differs from its saved response")
    if not destination.exists():
        atomic_write(destination, text)
    if not (directory / "record.json").exists():
        write_json(directory / "record.json", {"id": plan["id"], "index": plan["index"], "role": "summary",
                                              "status": "completed", "meta": meta, "submissions": [],
                                              "assignments": [], "at": existing.get("updated_at") or now()})
    if existing.get("status") == "completed":
        return existing
    value = _status(directory, plan, "completed", path="lead_summary.md", evidence="multi_agent/summary/evidence.json",
                    sha256=hashlib.sha256(text.encode()).hexdigest(), meta=meta)
    return value


def run(path: Path, cfg: dict, model, plan: dict, role_cfg: dict, frozen: dict, archive: dict,
        history: list[dict], fake_scripts=None, out=print) -> dict:
    """Persist the exact evidence and call a separate summary session, without reopening research."""
    path, directory = Path(path), Path(path) / DIRECTORY
    directory.mkdir(parents=True, exist_ok=True)
    plan_path, evidence_path = directory / "plan.json", directory / "evidence.json"
    try:
        # Existing evidence and prompts are immutable even if the caller resumes later.
        if evidence_path.exists():
            snapshot = read_json(evidence_path)
        else:
            snapshot = evidence(path, cfg, frozen, archive, history)
            write_json(evidence_path, snapshot)
        identity = {"id": plan["id"], "index": plan["index"], "role": "summary",
                    "frozen_sha256": _digest(frozen), "evidence_sha256": _digest(snapshot),
                    "allocation": {"session": role_cfg["session"], "budget": role_cfg["budget"]}}
        if plan_path.exists():
            saved_plan = read_json(plan_path)
            if any(saved_plan.get(key) != value for key, value in identity.items()):
                raise RuntimeError("the persisted summary plan, evidence or allocations changed")
        else:
            write_json(plan_path, identity)
        prompt_path = directory / "prompt.md"
        if not prompt_path.exists():
            text = prompt.build_summary(role_cfg, model, snapshot)
            # Researcher-authored notes and numerical evidence are data, just as in
            # research sessions. Check the coordinator-owned prompt independently.
            from . import loop
            ours = prompt.build_summary(role_cfg, model, prompt.summary_evidence(snapshot, include_discoveries=False))
            hygiene.check({"prompt": ours, **loop.agent_surface(role_cfg, model, role_cfg["sandbox"])},
                          loop.forbidden_strings(role_cfg, model))
            atomic_write(prompt_path, text)
        prompt_hash = hashlib.sha256(prompt_path.read_bytes()).hexdigest()
        saved_plan = read_json(plan_path)
        if "prompt_sha256" in saved_plan and saved_plan["prompt_sha256"] != prompt_hash:
            raise RuntimeError("the saved summary prompt changed")
        if "prompt_sha256" not in saved_plan:
            write_json(plan_path, {**saved_plan, "prompt_sha256": prompt_hash})
        result_path = directory / "response.json"
        if result_path.exists():
            return _publish(path, directory, plan, read_json(result_path))
        if cfg["auth"] == "api_key":
            counted = budget.state(path, role_cfg["budget"])
            if budget.left(counted, plan["index"] + 1) < budget.MIN_CAP_USD:
                raise StopRun(budget.spent_why(counted, plan["index"] + 1))
        session = Session.create(directory / "session", role_cfg, cfg["network"], path)
        if session.TOOLS:
            raise RuntimeError("the final summary session must expose no research tools")
        backend = make(role_cfg, log=out, fake_scripts=fake_scripts)
        if not backend.available(plan["index"]):
            value = _status(directory, plan, "unavailable", "No backend response is available for the final lead synthesis.")
        else:
            _status(directory, plan, "running")
            response = backend.respond(prompt_path.read_text(), plan["index"], directory, session)
            saved = {"text": response.text, "meta": response.meta}
            write_json(result_path, saved)
            value = _publish(path, directory, plan, saved)
    except StopRun as exc:
        value = _status(directory, plan, "stopped", str(exc))
    except BackendError as exc:
        value = _status(directory, plan, "unavailable", str(exc))
    except Exception as exc:
        value = _status(directory, plan, "failed", f"{type(exc).__name__}: {exc}")
    out(f"final lead synthesis: {value['status']}" + (f" ({value['reason']})" if value.get("reason") else ""))
    return value


def report(path: Path, cfg: dict) -> dict:
    """Report generation is read-only and never calls the summary backend."""
    status_path = Path(path) / DIRECTORY / "status.json"
    if status_path.exists():
        value = read_json(status_path)
        destination = Path(path) / "lead_summary.md"
        if value["status"] == "completed":
            if not destination.exists():
                return {**value, "status": "delivery_missing",
                        "reason": "The saved summary is missing; resume can restore it from the backend response checkpoint."}
            delivered = destination.read_bytes()
            if hashlib.sha256(delivered).hexdigest() != value.get("sha256"):
                return {**value, "status": "modified",
                        "reason": "The delivered summary differs from its recorded output; the backend response checkpoint is retained."}
            value["markdown"] = delivered.decode()
        return value
    if enabled(cfg):
        return {"status": "pending", "reason": "Final lead synthesis follows the frozen portfolio's final audits."}
    return {"status": "disabled", "reason": "Final lead synthesis is disabled in this saved run's configuration."}
