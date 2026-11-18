"""The one-agent comparator is selected on independent full-domain controls."""

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
    return run_config.load("config/pilot/lin-c-smoke.yaml", run_config.merge(overrides, changes))


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
            fields = overrides.get((self.network, value), {})
            return receipt(self.network, program, seed=seed, **fields)

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
    cfg["baseline"]["candidates"] = value

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
