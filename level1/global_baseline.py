"""One global comparison search, frozen before regional research begins.

The controller owns session execution and the shared spending ledger. This module
chooses a declared parameter-control model and validates only formal submissions
from that session. It never uses target-network scores to choose the comparison.
Finite checks establish an empirical comparator, not a proof for arbitrary weights.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import json
import math
import os
from pathlib import Path

from . import baseline, lang
from .networks import load_model
from .util import now, read_json, write_json
from .verifier import Verifier


DIRECTORY = Path("multi_agent")
FILE = DIRECTORY / "baseline.json"
ENTRY = DIRECTORY / "baseline_entry.json"
CHECKS = DIRECTORY / "baseline_checks.json"
ALL = {"kind": "all"}


def enabled(cfg: dict) -> bool:
    """Old saved configurations never acquire another paid session on resume."""
    options = cfg.get("orchestration") or {}
    return (options.get("enabled") is True and options.get("baseline_agent") is True
            and not options.get("baseline_entry"))


def fraction(cfg: dict) -> float:
    return float(cfg["orchestration"].get("baseline_fraction", 0.0)) if enabled(cfg) else 0.0


def _digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _architecture(model) -> str:
    """Ignore presentation-only changes without claiming general program equivalence."""
    tree = ast.parse(model.source())
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if (node.body and isinstance(node.body[0], ast.Expr)
                    and isinstance(node.body[0].value, ast.Constant)
                    and isinstance(node.body[0].value.value, str)):
                node.body = node.body[1:]
    return ast.dump(tree, include_attributes=False)


def config_for(cfg: dict) -> dict:
    """Preflight the declared control and return an isolated single-session config.

    This deliberately has no fallback to target weights or guessed initialization.
    The existing control-model declaration supplies at least four checkpoints.
    """
    source = copy.deepcopy(cfg)
    source.setdefault("orchestration", {})["enabled"] = False
    control_cfg = baseline.config_for(source)
    target, control = load_model(cfg["model"]), load_model(control_cfg["model"])
    differences = []
    if _architecture(target) != _architecture(control):
        differences.append("network definition")
    target_info, control_info = target.info(), control.info()
    for field in ("sizes", "input_space", "n_outputs", "weights", "data_set"):
        if target_info.get(field) != control_info.get(field):
            differences.append(field)
    if target.data_set_text() != control.data_set_text():
        differences.append("data-set description")
    if differences:
        raise RuntimeError("the declared baseline control does not match the target: " + ", ".join(differences))
    control_cfg["verifier"]["scope"] = dict(ALL)
    control_cfg["verifier"]["sample_seed"] = int(cfg["orchestration"]["development_seed"])
    control_cfg.pop("research_task", None)
    return control_cfg


def _full_domain_exact(receipt: dict | None) -> bool:
    if not isinstance(receipt, dict) or receipt.get("status") != "scored":
        return False
    parts = receipt.get("parts") or {}
    n = parts.get("n_inputs")
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        return False
    scope = receipt.get("claim_scope") or (receipt.get("scope") or {}).get("spec")
    if scope is not None and scope != ALL:
        return False
    return (parts.get("coverage") == 1.0 and parts.get("n_in_region") == n
            and parts.get("n_agree") == n)


def _fresh_seed(excluded: set[int]) -> int:
    value = int.from_bytes(os.urandom(8), "big")
    while value in excluded:
        value = (value + 1) % (1 << 64)
    return value


def _model_identity(model) -> dict:
    return {"name": model.name, "definition_sha256": _digest(_architecture(model)),
            "info": model.info(), "networks": copy.deepcopy(model.config["networks"])}


def _entry(path: Path, plan: dict, submission: dict) -> dict:
    number = submission.get("submission")
    if isinstance(number, bool) or not isinstance(number, int) or number < 1:
        raise RuntimeError("a baseline submission has no valid submission number")
    source = path / plan["directory"] / "session" / "submissions" / f"{number:03d}" / "entry.json"
    entry = read_json(source)
    if lang.entry_hash(entry) != submission.get("entry_sha256"):
        raise RuntimeError(f"baseline submission {number}'s entry changed after scoring")
    return entry


def _allocation(path: Path, cfg: dict, plan: dict, record: dict) -> dict:
    session_path = path / plan["directory"] / "session" / "session.json"
    session = read_json(session_path) if session_path.exists() else {}
    return {"fraction_of_shared_api_allowance": fraction(cfg),
            "parent_budget": copy.deepcopy(cfg.get("budget")),
            "planned": copy.deepcopy(plan.get("allocation")),
            "tool_allowances": copy.deepcopy(session.get("budgets")),
            "reported_api_budget": copy.deepcopy((record.get("meta") or {}).get("budget"))}


def freeze(path: Path, cfg: dict, control_model, plan: dict, record: dict, out=print) -> dict:
    """Freeze one full-domain entry using independent control checkpoints only.

    Local validation checkpoints persist their input seeds and completed receipts.
    A resume finishes missing checks without reselecting inputs or rerunning an
    already completed check. Both established and failed final records are immutable.
    """
    path = Path(path)
    control_cfg = config_for(cfg)
    if control_model.name != load_model(control_cfg["model"]).name:
        raise RuntimeError("the baseline session used a different control model than its declared configuration")
    networks = [network["id"] for network in control_model.config["networks"]]
    first, second = control_cfg["network"], control_cfg["seed_check"]["network"]
    others = [network for network in networks if network not in (first, second)]
    if len(set(networks)) != len(networks) or first == second or len(others) < 2:
        raise RuntimeError("the global baseline needs distinct search, selection and confirmation checkpoints")
    limit = (cfg.get("baseline") or {}).get("candidates", 3)
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise RuntimeError("baseline.candidates must be a positive whole number")
    identity = {"session_id": plan["id"], "control": _model_identity(control_model),
                "verifier": control_cfg["verifier"], "reward": control_cfg["reward"],
                "entry_limits": control_cfg["entry"], "prices": lang.PRICES,
                "candidates_limit": limit, "formal_submissions_sha256": _digest(record.get("submissions", []))}
    identity_hash = _digest(identity)
    if (path / FILE).exists():
        saved = read_json(path / FILE)
        if saved.get("source") != "single_agent_search" or saved.get("identity_sha256") != identity_hash:
            raise RuntimeError("the frozen baseline belongs to different session evidence or validation conditions")
        if saved.get("status") == "established":
            if not (path / ENTRY).is_file() or lang.entry_hash(read_json(path / ENTRY)) != saved["entry_sha256"]:
                raise RuntimeError("the frozen baseline entry differs from its immutable record")
        elif saved.get("status") != "failed":
            raise RuntimeError("the saved global baseline has an unknown status")
        return saved

    if (path / CHECKS).exists():
        checks = read_json(path / CHECKS)
        if checks.get("identity_sha256") != identity_hash:
            raise RuntimeError("the pending baseline checks belong to different session evidence or validation conditions")
    else:
        dev_seed = int(control_cfg["verifier"]["sample_seed"])
        selection_seed = _fresh_seed({dev_seed})
        checks = {"identity_sha256": identity_hash,
                  "selection_seed": selection_seed,
                  "confirmation_seed": _fresh_seed({dev_seed, selection_seed}), "candidates": {}}
        write_json(path / CHECKS, checks)

    eligible, seen = [], set()
    for submission in record.get("submissions", []):
        if submission.get("status") != "scored":
            continue
        entry = _entry(path, plan, submission)
        lang.check_structure(entry, **control_cfg["entry"])
        digest = submission["entry_sha256"]
        if digest in seen:
            continue
        seen.add(digest)
        key = str(submission["submission"])
        detail = checks["candidates"].setdefault(key, {
            "submission": submission["submission"], "name": submission.get("name"),
            "entry_sha256": digest, "search": copy.deepcopy(submission), "confirmation": {}})
        if submission.get("network") != first or not _full_domain_exact(submission):
            detail["reason"] = "the submitted program was not exact with full coverage on its search checkpoint"
            write_json(path / CHECKS, checks)
            continue
        if "selection" not in detail:
            detail["selection"] = Verifier(control_model, second, control_cfg).score(entry, seed=checks["selection_seed"])
            write_json(path / CHECKS, checks)
        selection = detail["selection"]
        reward = selection.get("reward")
        if (not _full_domain_exact(selection) or isinstance(reward, bool)
                or not isinstance(reward, (int, float)) or not math.isfinite(reward)):
            detail["reason"] = "the program did not pass the independent full-domain selection check"
            write_json(path / CHECKS, checks)
            continue
        eligible.append((-reward, submission["submission"], entry, detail))

    chosen, tried = None, []
    for _, _, entry, detail in sorted(eligible, key=lambda item: item[:2])[:limit]:
        for network in others:
            if network not in detail["confirmation"]:
                detail["confirmation"][network] = Verifier(control_model, network, control_cfg).score(
                    entry, seed=checks["confirmation_seed"])
                write_json(path / CHECKS, checks)
        passes = all(_full_domain_exact(receipt) for receipt in detail["confirmation"].values())
        detail["passed"] = passes
        detail["reason"] = ("all independent confirmation checks had full coverage and exact agreement"
                            if passes else "an independent confirmation check failed full coverage or exact agreement")
        write_json(path / CHECKS, checks)
        tried.append(detail["submission"])
        if passes:
            chosen = (entry, detail)
            break

    saved = {"source": "single_agent_search", "status": "established" if chosen else "failed",
             "established": chosen is not None, "session_id": plan["id"],
             "chosen": chosen[1]["submission"] if chosen else None,
             "entry_sha256": chosen[1]["entry_sha256"] if chosen else None,
             "reason": ("One global program passed full-coverage checks on the independent control checkpoints."
                        if chosen else "No formal baseline submission passed the independent full-domain checks."),
             "scope": dict(ALL), "frozen_at": now(), "identity_sha256": identity_hash,
             "identity": identity, "allocation": _allocation(path, cfg, plan, record),
             "selection": {"network": second, "input_seed": checks["selection_seed"],
                           "order": "highest reward among entries with full coverage and exact agreement on the independent selection checkpoint",
                           "candidates_limit": limit, "tried": tried},
             "confirmation": {"networks": others, "input_seed": checks["confirmation_seed"]},
             "candidates": copy.deepcopy(checks["candidates"]),
             "control_receipts": ({first: chosen[1]["search"], second: chosen[1]["selection"],
                                  **chosen[1]["confirmation"]} if chosen else {}),
             "provenance": "A single global agent searched the declared parameter-control model. Its unchanged program is a known achievable generic comparison, not a matched multi-agent search.",
             "validation_limit": "Agreement on finite control checkpoints and inputs is empirical evidence, not a proof for arbitrary weights.",
             "target_weights_used_in_selection": False,
             "research_candidate_parameter_transfer_performed": False}
    if chosen:
        if (path / ENTRY).exists():
            if lang.entry_hash(read_json(path / ENTRY)) != saved["entry_sha256"]:
                raise RuntimeError("an existing comparison entry differs from the selected global baseline")
        else:
            write_json(path / ENTRY, chosen[0])
    elif (path / ENTRY).exists():
        raise RuntimeError("a comparison entry exists although no global baseline passed validation")
    write_json(path / FILE, saved)
    out(f"global baseline: {saved['status']}; {saved['reason']}")
    return saved
