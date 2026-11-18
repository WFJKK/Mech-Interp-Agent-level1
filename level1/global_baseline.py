"""One global comparison search, frozen before regional research begins.

The controller owns session execution and the shared spending ledger. This module
chooses a declared parameter-control model and validates formal submissions from
that session, or a supplied program. Current validation also requires exact full
coverage on every checkpoint of every declared trained twin. Only the independent
control checkpoint's reward ranks programs; trained results are pass/fail checks.
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

import numpy as np

from . import baseline, lang, research_modes
from .networks import load_model
from .util import REPO_ROOT, now, read_json, write_json
from .verifier import Verifier


DIRECTORY = Path("multi_agent")
FILE = DIRECTORY / "baseline.json"
ENTRY = DIRECTORY / "baseline_entry.json"
CHECKS = DIRECTORY / "baseline_checks.json"
PROVIDED = DIRECTORY / "baseline_provided_entry.json"
ALL = {"kind": "all"}


def enabled(cfg: dict) -> bool:
    """Old saved configurations never acquire another paid session on resume."""
    return research_modes.generates_baseline(cfg)


def required(cfg: dict) -> bool:
    return research_modes.requires_baseline(cfg)


def needs_validation(cfg: dict) -> bool:
    """Legacy supplied programs keep their recorded, unvalidated provenance."""
    return required(cfg) and (enabled(cfg) or research_modes.validation_version(cfg) >= 2)


def fraction(cfg: dict) -> float:
    return research_modes.baseline_fraction(cfg)


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


def _candidate_limit(cfg: dict) -> int:
    options = cfg.get("baseline") or {}
    limit = options.get("candidates", 3) if isinstance(options, dict) else None
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise RuntimeError("baseline.candidates must be a positive whole number")
    return limit


def _control_networks(model) -> list[str]:
    manifests = model.config.get("networks")
    if not isinstance(manifests, list) or len(manifests) < 4:
        raise RuntimeError("the global baseline control needs at least four distinct checkpoints")
    names = [row.get("id") if isinstance(row, dict) else None for row in manifests]
    if any(not isinstance(name, str) or not name.strip() for name in names):
        raise RuntimeError("every global baseline control checkpoint needs a nonempty network ID")
    if len(set(names)) != len(names):
        raise RuntimeError("global baseline control network IDs must be distinct")
    return names


def _trained_networks(model) -> list[str]:
    manifests = model.config.get("networks")
    if not isinstance(manifests, list) or not manifests:
        raise RuntimeError(f"the declared trained twin {model.name} has no checkpoints")
    names = [row.get("id") if isinstance(row, dict) else None for row in manifests]
    if any(not isinstance(name, str) or not name.strip() for name in names) or len(set(names)) != len(names):
        raise RuntimeError(f"the declared trained twin {model.name} needs distinct nonempty checkpoint IDs")
    return names


def _matching(reference, other, label: str) -> None:
    differences = []
    if _architecture(reference) != _architecture(other):
        differences.append("network definition")
    reference_info, other_info = reference.info(), other.info()
    for field in ("sizes", "input_space", "n_outputs", "weights", "data_set"):
        if reference_info.get(field) != other_info.get(field):
            differences.append(field)
    if reference.data_set_text() != other.data_set_text():
        differences.append("data-set description")
    if differences:
        raise RuntimeError(f"{label} does not match: " + ", ".join(differences))


def trained_models(control_model) -> list:
    """The whole declared trained cohort, for operator-side checks only."""
    names = baseline.twins(control_model)
    if not names or any(not name.strip() for name in names) or len(set(names)) != len(names):
        raise RuntimeError("the global baseline needs a nonempty distinct list of declared trained twins")
    models = []
    for name in names:
        model = load_model(f"models/{name}")
        if model.name == control_model.name:
            raise RuntimeError("a baseline control cannot be its own declared trained twin")
        _matching(control_model, model, f"the declared trained twin {name}")
        _trained_networks(model)
        models.append(model)
    return models


def config_for(cfg: dict) -> dict:
    """Preflight the declared control and return an isolated single-session config.

    This uses the declared control checkpoints without substituting target weights.
    The existing control-model declaration supplies at least four checkpoints.
    """
    if research_modes.validation_version(cfg) < 2:
        _candidate_limit(cfg)
    source = copy.deepcopy(cfg)
    source.setdefault("orchestration", {})["enabled"] = False
    try:
        control_cfg = baseline.config_for(source)
    except KeyError as exc:
        if exc.args == ("id",):
            raise RuntimeError("every global baseline control checkpoint needs a nonempty network ID") from exc
        raise
    target, control = load_model(cfg["model"]), load_model(control_cfg["model"])
    _control_networks(control)
    _matching(target, control, "the declared baseline control")
    if research_modes.validation_version(cfg) >= 2:
        trained_models(control)
    control_cfg["verifier"]["scope"] = dict(ALL)
    control_cfg["verifier"]["sample_seed"] = int(cfg["orchestration"]["development_seed"])
    control_cfg.pop("research_task", None)
    return control_cfg


def validation_models(cfg: dict) -> list:
    """All model definitions whose identities must remain pinned for this policy."""
    control = load_model(config_for(cfg)["model"])
    return [control, *trained_models(control)] if research_modes.validation_version(cfg) >= 2 else [control]


def preflight(cfg: dict) -> dict:
    """Validate all policy-required definitions and weights before any paid session."""
    control_cfg = config_for(cfg)
    control = load_model(control_cfg["model"])
    seen = {}
    for network in _control_networks(control):
        weights = control.load_weights(network)
        digest = hashlib.sha256()
        for name in sorted(weights):
            value = np.ascontiguousarray(weights[name])
            digest.update(json.dumps([name, list(value.shape), value.dtype.str]).encode())
            digest.update(b"\0")
            if value.size:
                digest.update(memoryview(value).cast("B"))
        fingerprint = digest.hexdigest()
        if fingerprint in seen:
            raise RuntimeError("the global baseline controls repeat the same parameter values: "
                               f"{seen[fingerprint]} and {network}; distinct control settings are required")
        seen[fingerprint] = network
    if research_modes.validation_version(cfg) >= 2:
        for model in trained_models(control):
            for network in _trained_networks(model):
                model.load_weights(network)
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


def _freeze_v1(path: Path, cfg: dict, control_model, plan: dict, record: dict, out=print) -> dict:
    """Freeze one full-domain entry using independent control checkpoints only.

    Local validation checkpoints persist their input seeds and completed receipts.
    A resume finishes missing checks without reselecting inputs or rerunning an
    already completed check. Both established and failed final records are immutable.
    """
    path = Path(path)
    control_cfg = config_for(cfg)
    if control_model.name != load_model(control_cfg["model"]).name:
        raise RuntimeError("the baseline session used a different control model than its declared configuration")
    networks = _control_networks(control_model)
    first, second = control_cfg["network"], control_cfg["seed_check"]["network"]
    others = [network for network in networks if network not in (first, second)]
    if len(set(networks)) != len(networks) or first == second or len(others) < 2:
        raise RuntimeError("the global baseline needs distinct search, selection and confirmation checkpoints")
    limit = _candidate_limit(cfg)
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


def _write_sealed(path: Path, value: dict) -> None:
    """Detect accidental or agent-side changes to persisted validation evidence."""
    value["record_sha256"] = _digest({key: item for key, item in value.items() if key != "record_sha256"})
    write_json(path, value)


def _read_sealed(path: Path, identity_hash: str) -> dict:
    value = read_json(path)
    actual = _digest({key: item for key, item in value.items() if key != "record_sha256"})
    if value.get("record_sha256") != actual:
        raise RuntimeError("the persisted global baseline validation evidence changed or uses a different policy")
    if value.get("identity_sha256") != identity_hash:
        raise RuntimeError("the global baseline belongs to different session evidence, validation conditions or checkpoint cohort")
    return value


def _bound_receipt(receipt: dict, *, entry_sha256: str, network: str, seed: int,
                   model: str | None = None) -> None:
    """A cached result must belong to this program, checkpoint and input draw."""
    if not isinstance(receipt, dict) or receipt.get("network") != network or receipt.get("prices") != lang.PRICES:
        raise RuntimeError("a baseline validation receipt belongs to a different checkpoint or price list")
    digest = receipt.get("entry_sha256")
    if digest != entry_sha256 and not (receipt.get("status") == "refused" and digest is None):
        raise RuntimeError("a baseline validation receipt belongs to a different program")
    if model is not None and receipt.get("model") != model:
        raise RuntimeError("a baseline validation receipt belongs to a different declared model")
    inputs = receipt.get("inputs") or {}
    if inputs.get("mode") == "sample" and inputs.get("seed") != seed:
        raise RuntimeError("a baseline validation receipt belongs to a different input sample")


def _v2_context(cfg: dict, control_model) -> tuple:
    control_cfg = config_for(cfg)
    expected = load_model(control_cfg["model"])
    if _model_identity(control_model) != _model_identity(expected):
        raise RuntimeError("the baseline session used a different control model than its declared configuration")
    networks = _control_networks(control_model)
    first, second = control_cfg["network"], control_cfg["seed_check"]["network"]
    others = [network for network in networks if network not in (first, second)]
    if first == second or first not in networks or second not in networks or len(others) < 2:
        raise RuntimeError("the global baseline needs distinct search, selection and confirmation checkpoints")
    trained = trained_models(control_model)
    cohort = [(model, network) for model in trained for network in _trained_networks(model)]
    return control_cfg, networks, first, second, others, trained, cohort


def _v2_frozen(path: Path, identity: dict, networks: list[str], cohort: list) -> dict | None:
    if not (path / FILE).exists():
        return None
    saved = _read_sealed(path / FILE, _digest(identity))
    if saved.get("identity") != identity or saved.get("source") != identity["source"]:
        raise RuntimeError("the frozen global baseline has inconsistent validation provenance")
    if saved.get("validation_policy_version") != 2:
        raise RuntimeError("the frozen global baseline has a different validation policy")
    if saved.get("status") == "established":
        digest = saved.get("entry_sha256")
        if not (path / ENTRY).is_file() or lang.entry_hash(read_json(path / ENTRY)) != digest:
            raise RuntimeError("the frozen baseline entry differs from its immutable record")
        controls, trained = saved.get("control_receipts", {}), saved.get("trained_receipts", {})
        expected = {f"{model.name}/{network}" for model, network in cohort}
        if set(controls) != set(networks) or set(trained) != expected:
            raise RuntimeError("the frozen baseline is missing a required checkpoint receipt")
        for network, row in controls.items():
            seed = (identity["verifier"]["sample_seed"] if network == saved["search"]["network"]
                    else saved["selection"]["input_seed"] if network == saved["selection"]["network"]
                    else saved["confirmation"]["input_seed"])
            _bound_receipt(row, entry_sha256=digest, network=network, seed=seed)
            if not _full_domain_exact(row):
                raise RuntimeError("the frozen baseline has a control receipt without exact full coverage")
        for model, network in cohort:
            row = trained[f"{model.name}/{network}"]
            _bound_receipt(row, entry_sha256=digest, network=network,
                           seed=saved["confirmation"]["input_seed"], model=model.name)
            if not _full_domain_exact(row):
                raise RuntimeError("the frozen baseline has a trained receipt without exact full coverage")
    elif saved.get("status") != "failed":
        raise RuntimeError("the saved global baseline has an unknown status")
    elif (path / ENTRY).exists():
        raise RuntimeError("a comparison entry exists although no global baseline passed validation")
    return saved


def _v2_checks(path: Path, identity: dict) -> dict:
    identity_hash = _digest(identity)
    if (path / CHECKS).exists():
        checks = _read_sealed(path / CHECKS, identity_hash)
    else:
        development = int(identity["verifier"]["sample_seed"])
        selection = _fresh_seed({development})
        checks = {"validation_policy_version": 2, "identity_sha256": identity_hash,
                  "selection_seed": selection, "confirmation_seed": _fresh_seed({development, selection}),
                  "candidates": {}}
        _write_sealed(path / CHECKS, checks)
    seeds = [identity["verifier"]["sample_seed"], checks.get("selection_seed"), checks.get("confirmation_seed")]
    if any(isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 for seed in seeds) or len(set(seeds)) != 3:
        raise RuntimeError("the global baseline validation seeds are missing or inconsistent")
    return checks


def _validation_score(path: Path, checks: dict, destination: dict, key: str, program: dict,
                      model, network: str, cfg: dict, seed: int, *, trained: bool = False) -> dict:
    if key not in destination:
        score_cfg = copy.deepcopy(cfg)
        score_cfg.update(model=str(model.path), network=network)
        scored = Verifier(model, network, score_cfg).score(program, seed=seed)
        if trained:
            scored["model"] = model.name
        destination[key] = scored
        _write_sealed(path / CHECKS, checks)
    row = destination[key]
    _bound_receipt(row, entry_sha256=lang.entry_hash(program), network=network, seed=seed,
                   model=model.name if trained else None)
    return row


def _v2_identity(control_cfg: dict, control_model, trained: list, plan: dict | None,
                  record: dict | None, provided: dict | None, provided_source: str | None) -> dict:
    identity = {"validation_policy_version": 2, "source": "provided" if provided is not None else "single_agent_search",
                "session_id": plan["id"] if plan else None, "control": _model_identity(control_model),
                "trained_models": [_model_identity(model) for model in trained],
                "verifier": control_cfg["verifier"], "reward": control_cfg["reward"],
                "entry_limits": control_cfg["entry"], "prices": lang.PRICES}
    if provided is None:
        identity["formal_submissions_sha256"] = _digest(record.get("submissions", []))
    else:
        identity.update(provided_entry_sha256=lang.entry_hash(provided), provided_source=provided_source)
    return identity


def _freeze_v2(path: Path, cfg: dict, control_model, plan: dict | None, record: dict | None,
               out, *, provided: dict | None = None, provided_source: str | None = None) -> dict:
    path = Path(path)
    control_cfg, networks, first, second, others, trained, cohort = _v2_context(cfg, control_model)
    source = "provided" if provided is not None else "single_agent_search"
    identity = _v2_identity(control_cfg, control_model, trained, plan, record, provided, provided_source)
    frozen = _v2_frozen(path, identity, networks, cohort)
    if frozen is not None:
        return frozen
    checks = _v2_checks(path, identity)
    submissions = record.get("submissions", []) if record is not None else [{
        "submission": 1, "name": "provided", "status": "scored", "entry_sha256": lang.entry_hash(provided)}]
    eligible, seen = [], set()
    for submission in submissions:
        if submission.get("status") != "scored":
            continue
        program = provided if provided is not None else _entry(path, plan, submission)
        lang.check_structure(program, **control_cfg["entry"])
        digest = lang.entry_hash(program)
        if digest in seen:
            continue
        seen.add(digest)
        number = submission["submission"]
        detail = checks["candidates"].setdefault(str(number), {
            "submission": number, "name": submission.get("name"), "entry_sha256": digest,
            "confirmation": {}, "trained_confirmation": {}})
        if detail.get("entry_sha256") != digest:
            raise RuntimeError("a pending baseline candidate differs from its submitted program")
        if provided is None:
            if "search" in detail and detail["search"] != submission:
                raise RuntimeError("a pending baseline search receipt differs from its formal submission")
            detail["search"] = copy.deepcopy(submission)
        else:
            _validation_score(path, checks, detail, "search", program, control_model, first,
                              control_cfg, int(control_cfg["verifier"]["sample_seed"]))
        search = detail["search"]
        if search.get("network") != first or not _full_domain_exact(search):
            detail["reason"] = "the program was not exact with full coverage on its search checkpoint"
            _write_sealed(path / CHECKS, checks)
            continue
        _bound_receipt(search, entry_sha256=digest, network=first,
                       seed=int(control_cfg["verifier"]["sample_seed"]))
        selection = _validation_score(path, checks, detail, "selection", program, control_model, second,
                                      control_cfg, checks["selection_seed"])
        reward = selection.get("reward")
        if (not _full_domain_exact(selection) or isinstance(reward, bool)
                or not isinstance(reward, (int, float)) or not math.isfinite(reward)):
            detail["reason"] = "the program did not pass the independent full-domain selection check"
            _write_sealed(path / CHECKS, checks)
            continue
        eligible.append((-reward, number, program, detail))

    chosen, tried = None, []
    # There is no old three-candidate cutoff. A lower-ranked valid program must
    # still be considered after every higher-ranked invalid generic claim.
    for _, _, program, detail in sorted(eligible, key=lambda item: item[:2]):
        for network in others:
            _validation_score(path, checks, detail["confirmation"], network, program,
                              control_model, network, control_cfg, checks["confirmation_seed"])
        for model, network in cohort:
            _validation_score(path, checks, detail["trained_confirmation"], f"{model.name}/{network}",
                              program, model, network, control_cfg, checks["confirmation_seed"], trained=True)
        controls_pass = all(_full_domain_exact(detail["confirmation"][network]) for network in others)
        trained_pass = all(_full_domain_exact(detail["trained_confirmation"][f"{model.name}/{network}"])
                           for model, network in cohort)
        detail["passed"] = controls_pass and trained_pass
        detail["reason"] = ("every control and declared trained checkpoint had exact full coverage"
                            if detail["passed"] else
                            "a control or declared trained checkpoint was refused, partial or inexact")
        _write_sealed(path / CHECKS, checks)
        tried.append(detail["submission"])
        if detail["passed"]:
            chosen = (program, detail)
            break

    saved = {"source": source, "validation_policy_version": 2,
             "status": "established" if chosen else "failed", "established": chosen is not None,
             "session_id": plan["id"] if plan else None,
             "chosen": chosen[1]["submission"] if chosen else None,
             "entry_sha256": chosen[1]["entry_sha256"] if chosen else None,
             "reason": ("One global program passed exact full-coverage checks on all declared control and trained checkpoints."
                        if chosen else "No eligible global program passed exact full-coverage checks on all declared control and trained checkpoints."),
             "scope": dict(ALL), "frozen_at": now(), "identity": identity, "identity_sha256": _digest(identity),
             "allocation": (_allocation(path, cfg, plan, record) if plan else
                            {"fraction_of_shared_api_allowance": 0.0, "paid_agent_session": False}),
             "search": {"network": first, "input_seed": control_cfg["verifier"]["sample_seed"]},
             "selection": {"network": second, "input_seed": checks["selection_seed"],
                           "ranking_population": "independent_parameter_control",
                           "order": "highest independent control reward among exact full-domain entries; trained checkpoints only determine pass or fail",
                           "candidates_limit": None, "tried": tried},
             "confirmation": {"networks": others, "input_seed": checks["confirmation_seed"],
                              "trained_checkpoints": [{"model": model.name, "network": network} for model, network in cohort]},
             "candidates": copy.deepcopy(checks["candidates"]),
             "control_receipts": ({first: chosen[1]["search"], second: chosen[1]["selection"],
                                  **chosen[1]["confirmation"]} if chosen else {}),
             "trained_receipts": copy.deepcopy(chosen[1]["trained_confirmation"]) if chosen else {},
             "trained_checkpoints_used_for_acceptance": True,
             "trained_rewards_used_for_selection": False,
             "research_candidate_parameter_transfer_performed": False,
             "provenance": ("A single global agent searched the declared parameter-control model. Independent control reward ranked its unchanged programs; every declared trained checkpoint was used for pass/fail acceptance. This is a known achievable generic comparison, not a matched multi-agent search."
                            if provided is None else
                            "The operator supplied one unchanged global program. It passed the same control and trained-checkpoint acceptance requirements; no baseline-agent search was run."),
             "validation_limit": "Agreement on the finite declared checkpoints and evaluated inputs is empirical evidence, not a proof for arbitrary weights or unchecked inputs."}
    if provided is not None:
        saved["provided_source"] = provided_source
    if chosen:
        if (path / ENTRY).exists():
            if lang.entry_hash(read_json(path / ENTRY)) != saved["entry_sha256"]:
                raise RuntimeError("an existing comparison entry differs from the selected global baseline")
        else:
            write_json(path / ENTRY, chosen[0])
    elif (path / ENTRY).exists():
        raise RuntimeError("a comparison entry exists although no global baseline passed validation")
    _write_sealed(path / FILE, saved)
    out(f"global baseline: {saved['status']}; {saved['reason']}")
    return saved


def freeze(path: Path, cfg: dict, control_model, plan: dict, record: dict, out=print) -> dict:
    """Freeze under this run's saved policy; never upgrade older evidence silently."""
    if research_modes.validation_version(cfg) < 2:
        return _freeze_v1(path, cfg, control_model, plan, record, out)
    return _freeze_v2(path, cfg, control_model, plan, record, out)


def validate_provided(path: Path, cfg: dict, out=print) -> dict:
    """Snapshot a supplied program and apply the current acceptance gate without an agent."""
    source = (cfg.get("orchestration") or {}).get("baseline_entry")
    if not source or not required(cfg) or research_modes.validation_version(cfg) < 2:
        raise RuntimeError("supplied-program validation requires a current baseline research mode")
    path = Path(path)
    source_path = Path(source).expanduser()
    if not source_path.is_absolute():
        source_path = REPO_ROOT / source_path
    control_cfg = config_for(cfg)
    if (path / PROVIDED).exists():
        program = read_json(path / PROVIDED)
    else:
        program = read_json(source_path)
        lang.check_structure(program, **control_cfg["entry"])
        write_json(path / PROVIDED, program)
    return _freeze_v2(path, cfg, load_model(control_cfg["model"]), None, None, out,
                      provided=program, provided_source=str(source_path.resolve()))


def verify_frozen(path: Path, cfg: dict) -> dict:
    """Read existing frozen evidence without scoring, drawing seeds or writing.

    Completion and resume use this route even when the research controller has
    already finished. Legacy evidence remains subject to its original contract.
    """
    path = Path(path)
    if not (path / FILE).is_file():
        raise RuntimeError("the required frozen global baseline record is missing")
    if research_modes.validation_version(cfg) < 2:
        return read_json(path / FILE)
    control_model = load_model(config_for(cfg)["model"])
    control_cfg, networks, _, _, _, trained, cohort = _v2_context(cfg, control_model)
    source = (cfg.get("orchestration") or {}).get("baseline_entry")
    plan = record = provided = provided_source = None
    if source:
        if not (path / PROVIDED).is_file():
            raise RuntimeError("the frozen baseline's supplied-program snapshot is missing")
        provided = read_json(path / PROVIDED)
        source_path = Path(source).expanduser()
        if not source_path.is_absolute():
            source_path = REPO_ROOT / source_path
        provided_source = str(source_path.resolve())
    else:
        state_path = path / DIRECTORY / "state.json"
        if not state_path.is_file():
            raise RuntimeError("the frozen baseline's saved session plan is missing")
        state = read_json(state_path)
        session_id = state.get("baseline")
        plan = (state.get("sessions") or {}).get(session_id)
        if not isinstance(plan, dict) or plan.get("id") != session_id or plan.get("role") != "baseline":
            raise RuntimeError("the frozen baseline has no matching saved baseline session plan")
        record_path = path / plan["directory"] / "record.json"
        if not record_path.is_file():
            raise RuntimeError("the frozen baseline's formal session record is missing")
        record = read_json(record_path)
    identity = _v2_identity(control_cfg, control_model, trained, plan, record, provided, provided_source)
    frozen = _v2_frozen(path, identity, networks, cohort)
    if frozen is None:
        raise RuntimeError("the required frozen global baseline record is missing")
    return frozen
