"""Control reward ranks the comparator; the versioned checkpoint gate accepts it."""

import copy

import numpy as np
import pytest

from level1 import global_baseline, lang, run_config
from level1.networks import Model, ModelError, load_model
from level1.util import read_json, write_json


def config(tmp_path, **changes):
    overrides = {"run_id": "global-control-test", "runs_dir": str(tmp_path),
                 "backend": "fake", "sandbox": "process", "rounds_max": 1,
                 "session": {"python_calls": 40, "submissions": 4},
                 "verifier": {"sample_size": 16, "reference_samples": 1},
                 "orchestration": {"enabled": True, "baseline_agent": True,
                                   "baseline_fraction": 0.2, "development_seed": 73,
                                   "final_summary": False}}
    cfg = run_config.load("config/pilot/lin-c-smoke.yaml", run_config.merge(overrides, changes))
    if "baseline_validation_version" not in (changes.get("orchestration") or {}):
        # These original cases exercise already-saved runs, before policy v2.
        cfg["orchestration"].pop("baseline_validation_version", None)
    return cfg


def entry(value):
    return {"format": "level1-entry-v1", "nodes": [{"op": "const", "dtype": "int", "value": value}],
            "predict": 0, "rule": None}


def receipt(network, program, *, reward=0.7, coverage=1.0, agree=4, size=4, seed=73):
    return {"status": "scored", "reason": None, "network": network,
            "entry_sha256": lang.entry_hash(program), "reward": reward, "prices": lang.PRICES,
            "inputs": {"mode": "sample", "n": size, "seed": seed},
            "scope": {"spec": {"kind": "all"}},
            "parts": {"coverage": coverage, "n_inputs": size, "n_in_region": int(size * coverage),
                      "n_agree": agree, "fidelity": 1.0 if coverage else None}}


def prepare(tmp_path, values=(1, 2), **cfg_changes):
    cfg = config(tmp_path, **cfg_changes)
    path = tmp_path / "run"
    control_cfg = global_baseline.config_for(cfg)
    model = load_model(control_cfg["model"])
    plan = {"id": "a0001", "role": "baseline", "directory": "multi_agent/sessions/a0001-baseline",
            "allocation": {"session": {"python_calls": 8, "submissions": 2}, "budget": {"round_usd": 5}}}
    record = {"submissions": [], "meta": {"stop_reason": "end_turn", "budget": {"round_usd": 5}}}
    for number, value in enumerate(values, 1):
        program = entry(value)
        sub = receipt(control_cfg["network"], program, reward=1.0 - value / 10)
        sub.update(submission=number, name=f"candidate-{number}", claim_scope={"kind": "all"})
        write_json(path / plan["directory"] / "session/submissions" / f"{number:03d}" / "entry.json", program)
        record["submissions"].append(sub)
    write_json(path / plan["directory"] / "session/session.json", {"budgets": {"run_python": 8, "submit_entry": 2}})
    return path, cfg, model, plan, record


def fake_verifier(monkeypatch, overrides=None, interrupt=None):
    calls = []
    overrides = overrides or {}

    class FakeVerifier:
        def __init__(self, model, network, cfg):
            self.model, self.network, self.cfg = model, network, copy.deepcopy(cfg)

        def score(self, program, seed=None):
            value = program["nodes"][0]["value"]
            calls.append({"network": self.network, "value": value, "seed": seed, "config": self.cfg})
            if interrupt and interrupt(self.network, value):
                raise RuntimeError("interrupted local validation")
            fields = dict(overrides.get((self.network, value), {}))
            status = fields.pop("status", "scored")
            row = receipt(self.network, program, seed=seed, **fields)
            row["status"] = status
            if status == "refused":
                row.update(reason="the test program cannot run on this checkpoint", parts=None, reward=None)
            return row

    monkeypatch.setattr(global_baseline, "Verifier", FakeVerifier)
    return calls


def test_only_new_enabled_research_configs_allocate_a_baseline():
    assert not global_baseline.enabled({})
    assert global_baseline.fraction({}) == 0
    old = {"orchestration": {"enabled": True}}
    assert not global_baseline.enabled(old)
    automatic = {"orchestration": {"enabled": True, "baseline_agent": True, "baseline_fraction": 0.2}}
    assert global_baseline.enabled(automatic) and global_baseline.fraction(automatic) == 0.2
    for overrides in ({"enabled": False}, {"baseline_agent": False}, {"baseline_entry": "existing.json"}):
        cfg = {"orchestration": {**automatic["orchestration"], **overrides}}
        assert not global_baseline.enabled(cfg) and global_baseline.fraction(cfg) == 0


def test_control_config_has_matching_design_and_no_parent_mutation(tmp_path):
    cfg = config(tmp_path)
    original = copy.deepcopy(cfg)
    control = global_baseline.config_for(cfg)
    assert cfg == original
    assert control["model"] == "models/lin-e" and control["network"] == "lin-e-s0"
    assert control["seed_check"] == {"network": "lin-e-s1"}
    assert not control["orchestration"]["enabled"] and control["exact_on_any_weights"]
    assert control["verifier"]["scope"] == {"kind": "all"}
    assert control["verifier"]["sample_seed"] == 73
    assert control["reward"] == cfg["reward"] and control["entry"] == cfg["entry"]


def test_absent_control_fails_preflight_instead_of_using_target_weights(tmp_path):
    cfg = config(tmp_path, model="models/lin-a", network="lin-a-s0")
    with pytest.raises(RuntimeError, match="no baseline model"):
        global_baseline.config_for(cfg)


@pytest.mark.parametrize("network_ids", [
    ["duplicate", "duplicate", "third", "fourth"],
    ["first", "", "third", "fourth"],
    ["first", "  ", "third", "fourth"],
    ["first", None, "third", "fourth"],
    ["first", 2, "third", "fourth"],
])
def test_invalid_control_ids_fail_configuration_before_a_session(tmp_path, monkeypatch, network_ids):
    cfg = config(tmp_path)
    original_load = global_baseline.load_model

    def invalid_ids(path):
        model = original_load(path)
        if model.name == "lin-e":
            model.config["networks"] = [{"id": value} for value in network_ids]
        return model

    monkeypatch.setattr(global_baseline, "load_model", invalid_ids)
    with pytest.raises(RuntimeError, match="(nonempty network ID|IDs must be distinct)"):
        global_baseline.config_for(cfg)


@pytest.mark.parametrize("value", [0, -1, True, 1.5, "3", None])
def test_invalid_candidate_limit_fails_before_control_selection(tmp_path, monkeypatch, value):
    cfg = config(tmp_path)
    cfg.setdefault("baseline", {})["candidates"] = value  # main's defaults no longer carry a baseline block

    def no_model_lookup(*args, **kwargs):
        pytest.fail("the invalid candidate limit should fail before selecting a control model")

    monkeypatch.setattr(global_baseline.baseline, "config_for", no_model_lookup)
    with pytest.raises(RuntimeError, match="baseline.candidates must be a positive whole number"):
        global_baseline.config_for(cfg)


def test_preflight_loads_every_declared_control_checkpoint_sequentially(tmp_path, monkeypatch):
    cfg = config(tmp_path)
    visited = []

    def load_weights(model, network):
        visited.append((model.name, network))
        return {"parameter": np.array([len(visited)], dtype=np.float64)}

    monkeypatch.setattr(Model, "load_weights", load_weights)
    control_cfg = global_baseline.preflight(cfg)
    assert control_cfg == global_baseline.config_for(cfg)
    assert visited == [("lin-e", f"lin-e-s{i}") for i in range(6)]


def test_unusable_confirmation_weights_fail_preflight_before_the_search(tmp_path, monkeypatch):
    cfg = config(tmp_path)
    visited = []

    def load_weights(model, network):
        visited.append(network)
        if network == "lin-e-s3":
            raise ModelError("the confirmation weights do not match their manifest")
        return {"parameter": np.array([len(visited)], dtype=np.float64)}

    monkeypatch.setattr(Model, "load_weights", load_weights)
    with pytest.raises(ModelError, match="confirmation weights"):
        global_baseline.preflight(cfg)
    assert visited == ["lin-e-s0", "lin-e-s1", "lin-e-s2", "lin-e-s3"]


def test_different_checkpoint_manifests_cannot_repeat_identical_parameter_values(tmp_path, monkeypatch):
    cfg = config(tmp_path)
    visited = []

    def load_weights(model, network):
        visited.append(network)
        value = 1 if network == "lin-e-s5" else len(visited)
        # Each declared checkpoint has its own filename and file hash. Compare
        # the actual arrays, including when dictionary insertion order changes.
        values = {"a": np.array([value], dtype=np.float64), "b": np.array([[value]], dtype=np.float64)}
        return dict(reversed(list(values.items()))) if network == "lin-e-s5" else values

    monkeypatch.setattr(Model, "load_weights", load_weights)
    with pytest.raises(RuntimeError, match="repeat the same parameter values"):
        global_baseline.preflight(cfg)
    assert visited == [f"lin-e-s{i}" for i in range(6)]


@pytest.mark.parametrize("changed", ["sizes", "input_space", "weights", "data_set", "definition"])
def test_mismatched_declared_control_is_rejected(tmp_path, monkeypatch, changed):
    cfg = config(tmp_path)
    original_load = global_baseline.load_model

    def mismatched(path):
        model = original_load(path)
        if model.name == "lin-e":
            if changed == "definition":
                model.source = lambda: "def reference(g, lv1):\n    return g.token(0)\n"
            else:
                info = model.info()
                info[changed] = {"different": True}
                model.info = lambda: info
        return model

    monkeypatch.setattr(global_baseline, "load_model", mismatched)
    with pytest.raises(RuntimeError, match="does not match"):
        global_baseline.config_for(cfg)


def test_freeze_selects_by_independent_checkpoint_and_never_by_target(tmp_path, monkeypatch):
    path, cfg, model, plan, record = prepare(tmp_path)
    calls = fake_verifier(monkeypatch, {("lin-e-s1", 1): {"reward": 0.6},
                                       ("lin-e-s1", 2): {"reward": 0.8}})
    result = global_baseline.freeze(path, cfg, model, plan, record, out=lambda _: None)
    assert result["status"] == "established" and result["chosen"] == 2
    assert read_json(path / global_baseline.ENTRY) == entry(2)
    assert result["target_weights_used_in_selection"] is False
    assert result["research_candidate_parameter_transfer_performed"] is False
    assert result["allocation"]["tool_allowances"] == {"run_python": 8, "submit_entry": 2}
    assert result["allocation"]["planned"] == plan["allocation"]
    assert set(result["control_receipts"]) == {f"lin-e-s{i}" for i in range(6)}
    assert all(call["network"].startswith("lin-e-") for call in calls)
    assert all(call["config"]["verifier"]["scope"] == {"kind": "all"} for call in calls)
    selection = [call for call in calls if call["network"] == "lin-e-s1"]
    confirmation = [call for call in calls if call["network"] != "lin-e-s1"]
    assert len(selection) == 2 and len(confirmation) == 4
    assert len({call["seed"] for call in selection}) == len({call["seed"] for call in confirmation}) == 1
    assert selection[0]["seed"] != confirmation[0]["seed"]
    assert all(call["seed"] != cfg["orchestration"]["development_seed"] for call in calls)


@pytest.mark.parametrize("stage", ["search", "selection", "confirmation"])
def test_exact_only_on_a_small_region_cannot_establish_global_baseline(tmp_path, monkeypatch, stage):
    path, cfg, model, plan, record = prepare(tmp_path, values=(1,))
    small = {"coverage": 0.5, "agree": 2}
    if stage == "search":
        record["submissions"][0]["parts"].update(coverage=0.5, n_in_region=2, n_agree=2)
    changes = {("lin-e-s1" if stage == "selection" else "lin-e-s2", 1): small} if stage != "search" else {}
    calls = fake_verifier(monkeypatch, changes)
    result = global_baseline.freeze(path, cfg, model, plan, record, out=lambda _: None)
    assert result["status"] == "failed" and result["entry_sha256"] is None
    assert not (path / global_baseline.ENTRY).exists()
    if stage == "search":
        assert not calls


def test_a_narrow_claim_is_not_global_even_with_full_conditional_coverage(tmp_path, monkeypatch):
    path, cfg, model, plan, record = prepare(tmp_path, values=(1,))
    record["submissions"][0]["claim_scope"] = {"kind": "lengths", "lengths": [2]}
    calls = fake_verifier(monkeypatch)
    assert global_baseline.freeze(path, cfg, model, plan, record, out=lambda _: None)["status"] == "failed"
    assert not calls


@pytest.mark.parametrize("limit, expected", [(1, "failed"), (2, "established")])
def test_confirmation_rejection_and_configured_candidate_limit(tmp_path, monkeypatch, limit, expected):
    path, cfg, model, plan, record = prepare(tmp_path, baseline={"candidates": limit})
    calls = fake_verifier(monkeypatch, {("lin-e-s1", 1): {"reward": 0.9},
                                       ("lin-e-s1", 2): {"reward": 0.7},
                                       ("lin-e-s2", 1): {"agree": 3}})
    result = global_baseline.freeze(path, cfg, model, plan, record, out=lambda _: None)
    assert result["status"] == expected
    assert result["selection"]["tried"] == ([1] if limit == 1 else [1, 2])
    if limit == 1:
        assert not any(call["network"] == "lin-e-s2" and call["value"] == 2 for call in calls)
    else:
        assert result["chosen"] == 2


@pytest.mark.parametrize("has_candidates", [False, True])
def test_final_success_or_failure_is_immutable_and_does_not_repeat_checks(tmp_path, monkeypatch, has_candidates):
    path, cfg, model, plan, record = prepare(tmp_path, values=(1,) if has_candidates else ())
    fake_verifier(monkeypatch)
    saved = global_baseline.freeze(path, cfg, model, plan, record, out=lambda _: None)
    before = (path / global_baseline.FILE).read_bytes()

    def no_more_checks(*args, **kwargs):
        raise AssertionError("a final baseline was checked again")

    monkeypatch.setattr(global_baseline, "Verifier", no_more_checks)
    assert global_baseline.freeze(path, cfg, model, plan, record, out=lambda _: None) == saved
    assert (path / global_baseline.FILE).read_bytes() == before
    if has_candidates:
        write_json(path / global_baseline.ENTRY, entry(9))
        with pytest.raises(RuntimeError, match="differs from its immutable record"):
            global_baseline.freeze(path, cfg, model, plan, record, out=lambda _: None)


def test_resume_keeps_seeds_and_completed_local_control_receipts(tmp_path, monkeypatch):
    path, cfg, model, plan, record = prepare(tmp_path, values=(1,))
    first_calls = fake_verifier(monkeypatch, interrupt=lambda network, value: network == "lin-e-s3")
    with pytest.raises(RuntimeError, match="interrupted local validation"):
        global_baseline.freeze(path, cfg, model, plan, record, out=lambda _: None)
    pending = read_json(path / global_baseline.CHECKS)
    assert not (path / global_baseline.FILE).exists()
    second_calls = fake_verifier(monkeypatch)
    result = global_baseline.freeze(path, cfg, model, plan, record, out=lambda _: None)
    assert result["status"] == "established"
    assert [call["network"] for call in second_calls] == ["lin-e-s3", "lin-e-s4", "lin-e-s5"]
    assert all(call["seed"] == pending["confirmation_seed"] for call in second_calls)
    assert result["selection"]["input_seed"] == pending["selection_seed"]
    assert [call["network"] for call in first_calls] == ["lin-e-s1", "lin-e-s2", "lin-e-s3"]


def test_only_formal_submissions_count_and_program_hashes_are_checked(tmp_path, monkeypatch):
    path, cfg, model, plan, record = prepare(tmp_path, values=(1,))
    fake_verifier(monkeypatch)
    write_json(path / plan["directory"] / "session/submissions/001/entry.json", entry(9))
    with pytest.raises(RuntimeError, match="changed after scoring"):
        global_baseline.freeze(path, cfg, model, plan, record, out=lambda _: None)


def test_duplicate_entries_are_not_an_extra_confirmation_candidate(tmp_path, monkeypatch):
    path, cfg, model, plan, record = prepare(tmp_path, values=(1, 1))
    calls = fake_verifier(monkeypatch)
    result = global_baseline.freeze(path, cfg, model, plan, record, out=lambda _: None)
    assert result["chosen"] == 1 and len(result["candidates"]) == 1
    assert len(calls) == 5


def test_real_reference_program_passes_independent_control_verifier(tmp_path):
    path, cfg, model, plan, record = prepare(tmp_path, values=())
    control_cfg = global_baseline.config_for(cfg)
    program = model.reference_entry(model.load_weights(control_cfg["network"]))
    first = global_baseline.Verifier(model, control_cfg["network"], control_cfg).score(program)
    first.update(submission=1, name="reference", claim_scope={"kind": "all"})
    record["submissions"] = [first]
    write_json(path / plan["directory"] / "session/submissions/001/entry.json", program)
    result = global_baseline.freeze(path, cfg, model, plan, record, out=lambda _: None)
    assert result["status"] == "established"
    assert all(global_baseline._full_domain_exact(row) for row in result["control_receipts"].values())
    assert all(row["inputs"]["n"] == 16 for row in result["control_receipts"].values())


V2 = {"baseline_validation_version": 2}
TRAINED = {"lin-c/lin-c-s0", "lin-c/lin-c-s1", "lin-a-1to4/lin-a-1to4-s0", "lin-a-1to4/lin-a-1to4-s1"}


def test_research_modes_control_baseline_generation_and_validation():
    options = {"enabled": True, "baseline_agent": False, "baseline_fraction": 0.2,
               "baseline_validation_version": 2}
    cfg = {"orchestration": {**options, "mode": "baseline"}}
    assert global_baseline.required(cfg) and global_baseline.enabled(cfg)
    assert global_baseline.needs_validation(cfg) and global_baseline.fraction(cfg) == 1.0
    cfg["orchestration"].update(mode="both", baseline_entry="provided.json")
    assert global_baseline.required(cfg) and global_baseline.needs_validation(cfg)
    assert not global_baseline.enabled(cfg) and global_baseline.fraction(cfg) == 0
    cfg["orchestration"].pop("baseline_validation_version")
    assert not global_baseline.needs_validation(cfg)  # saved legacy supplied entry
    cfg["orchestration"].update(mode="trained", baseline_agent=True)
    assert not global_baseline.required(cfg) and not global_baseline.needs_validation(cfg)
    assert not global_baseline.enabled(cfg) and global_baseline.fraction(cfg) == 0


def test_current_preflight_loads_all_declared_trained_twins_before_research(tmp_path, monkeypatch):
    cfg = config(tmp_path, orchestration=V2)
    visited = []

    def load_weights(model, network):
        visited.append(f"{model.name}/{network}")
        return {"parameter": np.array([len(visited)], dtype=np.float64)}

    monkeypatch.setattr(Model, "load_weights", load_weights)
    global_baseline.preflight(cfg)
    assert visited[:6] == [f"lin-e/lin-e-s{i}" for i in range(6)]
    assert set(visited[6:]) == TRAINED and len(visited) == 10
    assert {model.name for model in global_baseline.validation_models(cfg)} == {"lin-e", "lin-c", "lin-a-1to4"}


def test_an_unusable_other_trained_twin_fails_preflight(tmp_path, monkeypatch):
    cfg = config(tmp_path, orchestration=V2)
    real = Model.load_weights

    def unusable(model, network):
        if network == "lin-a-1to4-s1":
            raise ModelError("the other trained checkpoint is corrupt")
        return real(model, network)

    monkeypatch.setattr(Model, "load_weights", unusable)
    with pytest.raises(ModelError, match="other trained checkpoint"):
        global_baseline.preflight(cfg)


@pytest.mark.parametrize("changed", ["definition", "sizes", "data_set", "ids", "empty"])
def test_current_config_preflights_every_trained_twin_definition(tmp_path, monkeypatch, changed):
    cfg = config(tmp_path, orchestration=V2)
    real = global_baseline.load_model

    def altered(path):
        model = real(path)
        if model.name == "lin-a-1to4":
            if changed == "definition":
                model.source = lambda: "def reference(g, lv1):\n    return g.token(0)\n"
            elif changed in ("ids", "empty"):
                model.config["networks"] = [{"id": "same"}, {"id": "same"}] if changed == "ids" else []
            else:
                info = model.info()
                info[changed] = {"different": True}
                model.info = lambda: info
        return model

    monkeypatch.setattr(global_baseline, "load_model", altered)
    with pytest.raises(RuntimeError, match="declared trained twin"):
        global_baseline.config_for(cfg)


def test_current_freeze_checks_every_trained_checkpoint_but_ranks_only_control_reward(tmp_path, monkeypatch):
    path, cfg, model, plan, record = prepare(tmp_path, orchestration=V2)
    changes = {("lin-e-s1", 1): {"reward": 0.5}, ("lin-e-s1", 2): {"reward": 0.8}}
    for name in ("lin-c-s0", "lin-c-s1", "lin-a-1to4-s0", "lin-a-1to4-s1"):
        changes[(name, 1)] = {"reward": 100}
        changes[(name, 2)] = {"reward": -100}
    calls = fake_verifier(monkeypatch, changes)
    result = global_baseline.freeze(path, cfg, model, plan, record, out=lambda _: None)
    assert result["chosen"] == 2 and result["validation_policy_version"] == 2
    assert result["trained_checkpoints_used_for_acceptance"] is True
    assert result["trained_rewards_used_for_selection"] is False
    assert result["selection"]["ranking_population"] == "independent_parameter_control"
    assert "target_weights_used_in_selection" not in result
    assert set(result["trained_receipts"]) == TRAINED
    assert {f"{item['model']}/{item['network']}" for item in result["confirmation"]["trained_checkpoints"]} == TRAINED
    assert all(row["model"] == key.split("/")[0] for key, row in result["trained_receipts"].items())
    assert all(row["reward"] == -100 for row in result["trained_receipts"].values())
    assert len(calls) == 10  # two rankings, four remaining controls, all four trained checkpoints
    confirm = [call for call in calls if call["network"] != "lin-e-s1"]
    assert len({call["seed"] for call in confirm}) == 1
    assert confirm[0]["seed"] != result["selection"]["input_seed"]
    assert all(call["config"]["verifier"] == global_baseline.config_for(cfg)["verifier"] for call in calls)


@pytest.mark.parametrize("bad", [{"agree": 3}, {"coverage": 0.5, "agree": 2}, {"status": "refused"}])
def test_control_exactness_does_not_hide_failed_trained_checks(tmp_path, monkeypatch, bad):
    path, cfg, model, plan, record = prepare(tmp_path, values=(1,), orchestration=V2)
    calls = fake_verifier(monkeypatch, {("lin-a-1to4-s1", 1): bad})
    result = global_baseline.freeze(path, cfg, model, plan, record, out=lambda _: None)
    assert result["status"] == "failed" and not (path / global_baseline.ENTRY).exists()
    detail = result["candidates"]["1"]
    assert all(global_baseline._full_domain_exact(row) for row in detail["confirmation"].values())
    assert not detail["passed"] and set(detail["trained_confirmation"]) == TRAINED
    assert len(calls) == 9


def test_later_eligible_program_is_tried_after_trained_failure_despite_legacy_limit(tmp_path, monkeypatch):
    path, cfg, model, plan, record = prepare(tmp_path, orchestration=V2, baseline={"candidates": 1})
    fake_verifier(monkeypatch, {("lin-e-s1", 1): {"reward": 0.9}, ("lin-e-s1", 2): {"reward": 0.7},
                                ("lin-c-s1", 1): {"agree": 3}})
    result = global_baseline.freeze(path, cfg, model, plan, record, out=lambda _: None)
    assert result["chosen"] == 2 and result["selection"]["tried"] == [1, 2]
    assert result["selection"]["candidates_limit"] is None


def test_current_policy_does_not_read_obsolete_candidate_cap(tmp_path, monkeypatch):
    cfg = config(tmp_path, orchestration=V2)
    cfg["baseline"]["candidates"] = None
    assert global_baseline.config_for(cfg)["model"] == "models/lin-e"


def test_current_validation_resume_preserves_completed_checks_and_seeds(tmp_path, monkeypatch):
    path, cfg, model, plan, record = prepare(tmp_path, values=(1,), orchestration=V2)
    fake_verifier(monkeypatch, interrupt=lambda network, value: network == "lin-a-1to4-s0")
    with pytest.raises(RuntimeError, match="interrupted local validation"):
        global_baseline.freeze(path, cfg, model, plan, record, out=lambda _: None)
    pending = read_json(path / global_baseline.CHECKS)
    completed = set(pending["candidates"]["1"]["trained_confirmation"])
    calls = fake_verifier(monkeypatch)
    result = global_baseline.freeze(path, cfg, model, plan, record, out=lambda _: None)
    assert result["status"] == "established"
    assert {call["network"] for call in calls} == {key.split("/")[1] for key in TRAINED - completed}
    assert all(call["seed"] == pending["confirmation_seed"] for call in calls)
    assert result["selection"]["input_seed"] == pending["selection_seed"]
    before = (path / global_baseline.FILE).read_bytes()
    calls.clear()
    assert global_baseline.freeze(path, cfg, model, plan, record, out=lambda _: None) == result
    assert not calls and (path / global_baseline.FILE).read_bytes() == before


@pytest.mark.parametrize("kind", ["receipt", "cohort", "policy", "entry"])
def test_current_frozen_evidence_cannot_silently_change(tmp_path, monkeypatch, kind):
    path, cfg, model, plan, record = prepare(tmp_path, values=(1,), orchestration=V2)
    fake_verifier(monkeypatch)
    result = global_baseline.freeze(path, cfg, model, plan, record, out=lambda _: None)
    if kind == "receipt":
        result["trained_receipts"]["lin-c/lin-c-s0"]["parts"]["n_agree"] -= 1
        write_json(path / global_baseline.FILE, result)
    elif kind == "policy":
        result["validation_policy_version"] = 1
        write_json(path / global_baseline.FILE, result)
    elif kind == "entry":
        write_json(path / global_baseline.ENTRY, entry(9))
    else:
        real = global_baseline.load_model

        def other_cohort(source):
            value = real(source)
            if value.name == "lin-a-1to4":
                value.config["networks"][1]["sha256"] = "different checkpoint"
            return value

        monkeypatch.setattr(global_baseline, "load_model", other_cohort)
    with pytest.raises(RuntimeError, match="(changed|different|differs)"):
        global_baseline.freeze(path, cfg, model, plan, record, out=lambda _: None)


def test_pending_trained_receipt_tampering_is_detected_before_resume(tmp_path, monkeypatch):
    path, cfg, model, plan, record = prepare(tmp_path, values=(1,), orchestration=V2)
    fake_verifier(monkeypatch, interrupt=lambda network, value: network == "lin-a-1to4-s0")
    with pytest.raises(RuntimeError, match="interrupted local validation"):
        global_baseline.freeze(path, cfg, model, plan, record, out=lambda _: None)
    pending = read_json(path / global_baseline.CHECKS)
    pending["confirmation_seed"] += 1
    write_json(path / global_baseline.CHECKS, pending)
    calls = fake_verifier(monkeypatch)
    with pytest.raises(RuntimeError, match="validation evidence changed"):
        global_baseline.freeze(path, cfg, model, plan, record, out=lambda _: None)
    assert not calls


def test_supplied_program_gets_identical_acceptance_without_an_agent(tmp_path, monkeypatch):
    source = tmp_path / "provided.json"
    write_json(source, entry(1))
    cfg = config(tmp_path, orchestration={**V2, "baseline_entry": str(source)})
    calls = fake_verifier(monkeypatch)
    path = tmp_path / "run"
    result = global_baseline.validate_provided(path, cfg, out=lambda _: None)
    assert result["status"] == "established" and result["source"] == "provided"
    assert result["session_id"] is None and len(calls) == 10
    assert result["allocation"] == {"fraction_of_shared_api_allowance": 0.0, "paid_agent_session": False}
    assert set(result["trained_receipts"]) == TRAINED
    assert set(result["control_receipts"]) == {f"lin-e-s{i}" for i in range(6)}
    assert read_json(path / global_baseline.ENTRY) == entry(1)
    write_json(source, entry(9))
    calls.clear()
    assert global_baseline.validate_provided(path, cfg, out=lambda _: None) == result
    assert not calls  # resume retains its original immutable supplied program


@pytest.mark.parametrize("bad", [{"agree": 3}, {"coverage": 0.5, "agree": 2}, {"status": "refused"}])
def test_supplied_program_cannot_skip_trained_acceptance(tmp_path, monkeypatch, bad):
    source = tmp_path / "provided.json"
    write_json(source, entry(1))
    cfg = config(tmp_path, orchestration={**V2, "baseline_entry": str(source)})
    calls = fake_verifier(monkeypatch, {("lin-c-s1", 1): bad})
    path = tmp_path / "run"
    result = global_baseline.validate_provided(path, cfg, out=lambda _: None)
    assert result["status"] == "failed" and result["source"] == "provided"
    assert not (path / global_baseline.ENTRY).exists()
    calls.clear()
    assert global_baseline.validate_provided(path, cfg, out=lambda _: None) == result
    assert not calls


def test_legacy_frozen_baseline_stays_controls_only_and_is_never_upgraded(tmp_path, monkeypatch):
    path, cfg, model, plan, record = prepare(tmp_path, values=(1,))
    fake_verifier(monkeypatch)
    result = global_baseline.freeze(path, cfg, model, plan, record, out=lambda _: None)
    before = (path / global_baseline.FILE).read_bytes()
    assert "validation_policy_version" not in result and "trained_receipts" not in result
    calls = fake_verifier(monkeypatch, {("lin-c-s0", 1): {"status": "refused"}})
    assert global_baseline.freeze(path, cfg, model, plan, record, out=lambda _: None) == result
    assert not calls and (path / global_baseline.FILE).read_bytes() == before
    cfg["orchestration"].update(V2)
    with pytest.raises(RuntimeError, match="different policy"):
        global_baseline.freeze(path, cfg, model, plan, record, out=lambda _: None)
    assert not calls and (path / global_baseline.FILE).read_bytes() == before


def _persist_baseline_session(path, plan, record):
    write_json(path / global_baseline.DIRECTORY / "state.json",
               {"baseline": plan["id"], "sessions": {plan["id"]: plan}})
    write_json(path / plan["directory"] / "record.json", record)


@pytest.mark.parametrize("supplied", [False, True])
def test_verify_frozen_is_read_only_without_any_new_validation_work(tmp_path, monkeypatch, supplied):
    fake_verifier(monkeypatch)
    if supplied:
        source = tmp_path / "provided.json"
        write_json(source, entry(1))
        cfg = config(tmp_path, orchestration={**V2, "baseline_entry": str(source)})
        path = tmp_path / "run"
        result = global_baseline.validate_provided(path, cfg, out=lambda _: None)
    else:
        path, cfg, model, plan, record = prepare(tmp_path, values=(1,), orchestration=V2)
        _persist_baseline_session(path, plan, record)
        result = global_baseline.freeze(path, cfg, model, plan, record, out=lambda _: None)
    before = {str(file.relative_to(path)): file.read_bytes() for file in path.rglob("*") if file.is_file()}

    def forbidden(*args, **kwargs):
        pytest.fail("frozen verification must not score, draw seeds, load weights or write")

    monkeypatch.setattr(global_baseline, "Verifier", forbidden)
    monkeypatch.setattr(global_baseline, "_fresh_seed", forbidden)
    monkeypatch.setattr(global_baseline, "write_json", forbidden)
    monkeypatch.setattr(Model, "load_weights", forbidden)
    assert global_baseline.verify_frozen(path, cfg) == result
    assert {str(file.relative_to(path)): file.read_bytes() for file in path.rglob("*") if file.is_file()} == before


@pytest.mark.parametrize("change", ["receipts", "rehashed_receipts", "formal_record", "missing_record", "missing_frozen"])
def test_verify_frozen_rejects_inconsistent_or_missing_completed_evidence(tmp_path, monkeypatch, change):
    path, cfg, model, plan, record = prepare(tmp_path, values=(1,), orchestration=V2)
    _persist_baseline_session(path, plan, record)
    fake_verifier(monkeypatch)
    result = global_baseline.freeze(path, cfg, model, plan, record, out=lambda _: None)
    if change in ("receipts", "rehashed_receipts"):
        result["trained_receipts"].pop("lin-c/lin-c-s1")
        if change == "rehashed_receipts":
            global_baseline._write_sealed(path / global_baseline.FILE, result)
        else:
            write_json(path / global_baseline.FILE, result)
    elif change == "formal_record":
        record["submissions"][0]["parts"]["n_agree"] -= 1
        write_json(path / plan["directory"] / "record.json", record)
    elif change == "missing_record":
        (path / plan["directory"] / "record.json").rename(path / "moved-record.json")
    else:
        (path / global_baseline.FILE).rename(path / "moved-baseline.json")
    with pytest.raises(RuntimeError, match="(changed|different|missing)"):
        global_baseline.verify_frozen(path, cfg)


def test_verify_frozen_rejects_changed_supplied_snapshot(tmp_path, monkeypatch):
    source = tmp_path / "provided.json"
    write_json(source, entry(1))
    cfg = config(tmp_path, orchestration={**V2, "baseline_entry": str(source)})
    fake_verifier(monkeypatch)
    path = tmp_path / "run"
    global_baseline.validate_provided(path, cfg, out=lambda _: None)
    write_json(path / global_baseline.PROVIDED, entry(9))
    with pytest.raises(RuntimeError, match="different"):
        global_baseline.verify_frozen(path, cfg)


def test_verify_frozen_preserves_legacy_raw_record_without_new_checks(tmp_path, monkeypatch):
    path, cfg, model, plan, record = prepare(tmp_path, values=(1,))
    fake_verifier(monkeypatch)
    result = global_baseline.freeze(path, cfg, model, plan, record, out=lambda _: None)

    def forbidden(*args, **kwargs):
        pytest.fail("reading a legacy record must not reconstruct v2 validation conditions")

    monkeypatch.setattr(global_baseline, "config_for", forbidden)
    assert global_baseline.verify_frozen(path, cfg) == result
