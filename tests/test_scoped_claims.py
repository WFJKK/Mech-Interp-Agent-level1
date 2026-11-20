"""Narrow claims retain their own evaluation domain through every research layer."""

import copy
import json

import numpy as np
import pytest

from level1 import lang, multi_agent, regions, run_config
from level1.networks import load_model
from level1.session import Session, ToolError
from level1.util import read_json, write_json


def research_session(tmp_path, assigned=None, *, research=True, **overrides):
    cfg = run_config.load("config/pilot/lin-a-1to4.yaml", run_config.merge({
        "runs_dir": str(tmp_path), "backend": "fake", "sandbox": "process",
        "session": {"python_calls": 4, "submissions": 4},
        "verifier": {"scope": assigned or {"kind": "all"}, "check": "sample",
                     "sample_size": 32, "sample_seed": 17},
    }, overrides))
    if research:
        cfg["research_task"] = {"role": "regional", "cycle": 1, "max_assignments": 0}
    model = load_model(cfg["model"])
    entry = model.reference_entry(model.load_weights(cfg["network"]))
    session = Session.create(tmp_path / "session", cfg, cfg["network"], tmp_path / "run", {"program": entry})
    return session, cfg


def gate(threshold):
    g = lang.Graph({}, 4)
    return g.entry(predict=g.const(0), rule=g.token(0) < threshold)


def test_local_child_measurement_matches_formal_scope_sample_and_workload(tmp_path):
    parent = {"kind": "lengths", "lengths": [1, 2]}
    child = {"kind": "lengths", "lengths": [1]}
    session, _ = research_session(tmp_path, parent)
    result = session.run_python("""
import json
child = {"kind": "lengths", "lengths": [1]}
g = graph()
mapping = (g.weight("unembed") @ g.weight("hidden")) @ g.weight("embed")
entry = g.entry(predict=lv1.take_sum(mapping, g.tokens(), axis=1).argmax())
save_entry("shared", entry)
x_child = network_outputs(child)[0].copy()
child_measurement = measure(entry, child)
parent_measurement = measure(entry)
assert np.array_equal(network_outputs(child)[0], x_child)
assert np.array_equal(network_outputs({"kind": "all"})[0], network_outputs()[0])
print(json.dumps({"child": child_measurement, "parent": parent_measurement, "x": x_child.tolist()}))
""")
    assert result["status"] == "ok", result
    measured = json.loads(result["output"])
    scored = session.call("submit_entry", {"name": "shared", "claimed_fidelity": 1.0, "scope": child})
    assert scored["status"] == "scored", scored
    local = dict(measured["child"])
    assert local.pop("scope") == scored["scope"]
    assert local.pop("reward") == scored["reward"]
    assert local == scored["parts"]
    assert np.array_equal(np.asarray(measured["x"]), session.verifier(child).inputs()[0])
    assert scored["inputs"]["seed"] == 17 and scored["scope"]["mass"] == 0.25
    assert scored["scope"]["workload_size"] == 64
    assert measured["parent"]["scope"]["workload_size"] == 64 + 64 ** 2
    assert local["build_ops"] == measured["parent"]["build_ops"] > 0
    assert local["b"] == pytest.approx(measured["parent"]["b"] * 65)
    frozen = session.receipts()[0]
    assert frozen["claim_scope"] == child
    assert frozen["assigned_scope_id"] == regions.scope_id(parent)
    assert read_json(session.path / "submissions/001/claim.json") == {
        "claim_scope": child, "assigned_scope_id": regions.scope_id(parent),
    }


def test_requesting_all_preserves_assigned_domain_and_ordinary_sessions_reject_scopes(tmp_path):
    assigned = {"kind": "lengths", "lengths": [1, 2]}
    session, _ = research_session(tmp_path / "research", assigned)
    scored = session.submit_entry("program", 1.0, scope={"kind": "all"})
    assert scored["status"] == "scored"
    assert scored["scope"]["spec"] == assigned
    assert scored["scope"]["mass"] == 0.5
    assert session.receipts()[0]["claim_scope"] == assigned
    ordinary, _ = research_session(tmp_path / "ordinary", assigned, research=False)
    with pytest.raises(ToolError, match="only in research"):
        ordinary.submit_entry("program", 1.0, scope={"kind": "lengths", "lengths": [1]})
    assert "scope" not in ordinary.TOOLS["submit_entry"]["schema"]["properties"]
    assert ordinary.state["used"]["submit_entry"] == 0


def test_predicate_claim_captures_saved_rule_before_the_entry_is_changed(tmp_path):
    assigned = {"kind": "lengths", "lengths": [1, 2]}
    session, _ = research_session(tmp_path, assigned)
    write_json(session.entries_dir / "gate.json", gate(2))
    first = session.call("submit_entry", {"name": "program", "claimed_fidelity": 1.0,
                                          "scope": {"kind": "predicate"}, "scope_entry_name": "gate"})
    assert first["status"] == "scored", first
    files = [session.path / "submissions/001" / name for name in ("claim.json", "receipt.json", "entry.json")]
    frozen = {file: file.read_bytes() for file in files}
    expected = regions.intersect_scopes(assigned, {"kind": "predicate", "entry": gate(2)})
    assert session.receipts()[0]["claim_scope"] == expected
    assert first["scope"]["cardinality"] == 2 + 2 * 64
    changed = session.run_python("g=graph()\nsave_entry('gate', g.entry(predict=g.const(0), rule=g.token(0)<3))")
    assert changed["status"] == "ok", changed
    second = session.submit_entry("program", 1.0, scope={"kind": "predicate"}, scope_entry_name="gate")
    assert second["status"] == "scored", second
    assert second["scope"]["cardinality"] == 3 + 3 * 64
    assert first["scope"]["id"] != second["scope"]["id"]
    assert all(file.read_bytes() == content for file, content in frozen.items())
    assert Session(session.path).receipts()[0]["claim_scope"] == expected


@pytest.mark.parametrize("claim, pattern", [
    ({"kind": "all", "workload_size": 10 ** 9}, "does not take"),
    ({"kind": "lengths", "lengths": [5]}, "subset"),
    ({"kind": "lengths", "lengths": []}, "nonempty"),
    ({"kind": "predicate", "entry": gate(2)}, "workload_size"),
])
def test_invalid_scope_or_unknown_workload_does_not_consume_a_submission(tmp_path, claim, pattern):
    session, _ = research_session(tmp_path)
    with pytest.raises(ToolError, match=pattern):
        session.submit_entry("program", 1.0, scope=claim)
    assert session.state["used"]["submit_entry"] == 0
    assert read_json(session.state_path)["used"]["submit_entry"] == 0
    assert not (session.path / "submissions").exists()


def test_sampled_predicate_claim_uses_the_configured_workload_and_same_local_sample(tmp_path):
    session, _ = research_session(tmp_path, verifier={"workload_size": 1000})
    write_json(session.entries_dir / "gate.json", gate(2))
    result = session.run_python("""
import json
domain = {"kind": "predicate", "entry": entries["gate"]}
print(json.dumps(measure(entries["program"], scope=domain)))
""")
    assert result["status"] == "ok", result
    local = json.loads(result["output"])
    receipt = session.submit_entry("program", 1.0, scope={"kind": "predicate"}, scope_entry_name="gate")
    assert receipt["status"] == "scored", receipt
    assert local.pop("scope") == receipt["scope"]
    assert local.pop("reward") == receipt["reward"]
    assert local == receipt["parts"]
    assert receipt["scope"]["cardinality"] is None
    assert receipt["scope"]["workload_size"] == 1000
    assert receipt["scope"]["workload_basis"] == "declared"
    assert receipt["inputs"]["seed"] == 17


@pytest.fixture(scope="module")
def scoped_run(tmp_path_factory):
    where = tmp_path_factory.mktemp("scoped-claims")
    model = load_model("models/lin-a-1to4")
    comparison = model.reference_entry(model.load_weights("lin-a-1to4-s0"))
    baseline = where / "comparison.json"
    write_json(baseline, comparison)
    cfg = run_config.load("config/pilot/lin-a-1to4.yaml", {
        "run_id": "scoped-claims", "runs_dir": str(where), "backend": "fake", "sandbox": "process",
        "rounds_max": 1, "session": {"python_calls": 20, "submissions": 4},
        "verifier": {"check": "sample", "sample_size": 32, "reference_samples": 1},
        "orchestration": {"enabled": True, "workers": 1, "regional_submissions": 2,
                          "checkpoint_python_calls": 1, "development_seed": 73,
                          "final_summary": True, "baseline_entry": str(baseline)},
    })
    path = run_config.run_path(cfg)
    path.mkdir(parents=True)
    write_json(path / "run.json", {"run_id": cfg["run_id"], "config": cfg,
                                   "model_ops": 671, "sandbox": {"mode": "process"}})
    fake = where / "scripts"
    fake.mkdir()
    scripts = [
        {"calls": [{"tool": "assign_task", "args": {
            "scope": {"kind": "lengths", "lengths": [1, 2]}, "task": "Investigate this declared region.",
        }}], "response": "## What the network does\nOne regional task is assigned."},
        {"calls": [
            {"tool": "run_python", "args": {"code": "save_entry('same', reference_entry())"}},
            {"tool": "submit_entry", "args": {"name": "same", "claimed_fidelity": 1.0}},
            {"tool": "submit_entry", "args": {"name": "same", "claimed_fidelity": 1.0,
                                                "scope": {"kind": "lengths", "lengths": [1]}}},
        ], "response": "## What the network does\nThe parent and a narrower claim were checked."},
        {"calls": [
            {"tool": "run_python", "args": {"code": "save_entry('same', reference_entry())"}},
            {"tool": "submit_entry", "args": {"name": "same", "claimed_fidelity": 1.0}},
            {"tool": "submit_entry", "args": {"name": "same", "claimed_fidelity": 1.0,
                                                "scope": {"kind": "lengths", "lengths": [2]}}},
        ], "response": "## What the network does\nThe complete domain and a distinct region were checked."},
        {"calls": [], "response": "## Research summary\nThe frozen evidence retains distinct regional claims."},
    ]
    for i, script in enumerate(scripts, 1):
        write_json(fake / f"{i:02d}.json", script)
    assert multi_agent.drive(path, fake_scripts=fake, out=lambda *_: None) == "finished"
    return path, cfg, fake


def test_lead_child_claims_and_identical_programs_keep_independent_frontiers(scoped_run):
    path, cfg, _ = scoped_run
    archive = multi_agent.read_archive(path)
    candidates = {candidate["id"]: candidate for candidate in archive["candidates"]}
    assert len(candidates) == 4 and len({c["entry_sha256"] for c in candidates.values()}) == 1
    assert archive["global"] == {"best": "a0003s1", "frontier": ["a0003s1"]}
    expected = {"a0002s1": [1, 2], "a0002s2": [1], "a0003s2": [2]}
    for cid, lengths in expected.items():
        spec = {"kind": "lengths", "lengths": lengths}
        sid = regions.scope_id(spec)
        candidate = candidates[cid]
        assert candidate["scope"] == candidate["submitted"]["claim_scope"] == spec
        assert candidate["selection"]["scope"]["id"] == sid
        assert archive["regions"][sid]["frontier"] == [cid]
    assert candidates["a0003s2"]["role"] == "lead"
    for candidate in candidates.values():
        selected, submitted = candidate["selection"], candidate["submitted"]
        assert selected["inputs"]["seed"] == submitted["inputs"]["seed"] == cfg["orchestration"]["development_seed"]
        assert selected["parts"] == submitted["parts"]
        assert selected["scope"] == submitted["scope"]
    state = read_json(path / "multi_agent/state.json")
    lead_snapshot = state["sessions"]["a0003"]["snapshot"]
    assert {c["scope_id"] for c in lead_snapshot["candidates"]} == {
        candidates["a0002s1"]["scope_id"], candidates["a0002s2"]["scope_id"],
    }


def test_comparison_audits_and_summary_follow_actual_claims_and_remain_frozen(scoped_run, monkeypatch):
    path, cfg, fake = scoped_run
    archive = multi_agent.read_archive(path)
    frozen = read_json(path / "multi_agent/lock.json")
    evidence = read_json(path / "multi_agent/summary/evidence.json")
    expected_ids = {candidate["scope_id"] for candidate in archive["candidates"]}
    assert set(archive["baseline_scores"]) == expected_ids
    assert len(frozen["candidates"]) == len(evidence["candidates"]) == 4
    assert read_json(path / "multi_agent/summary/status.json")["status"] == "completed"
    assert evidence["frozen_selection"]["global"] == "a0003s1"
    assert evidence["frozen_selection"]["regional"] == frozen["regional"]
    for candidate in evidence["candidates"]:
        sid, spec = candidate["scope_id"], candidate["scope"]
        assert candidate["development"]["scope"]["spec"] == spec
        comparison = evidence["comparison"]["development"][sid]
        assert comparison["scope"]["id"] == sid
        assert comparison["scope"]["workload_size"] == candidate["development"]["scope"]["workload_size"]
        audit = candidate["final_audits"]
        assert audit["scope"] == spec
        for purpose in ("fresh", "held_out"):
            assert audit[purpose]["scope"]["id"] == audit[f"{purpose}_baseline"]["scope"]["id"] == sid
            assert audit[purpose]["scope"]["spec"] == spec
            assert audit[purpose]["inputs"] == audit[f"{purpose}_baseline"]["inputs"]
            assert audit[purpose]["inputs"]["seed"] != cfg["orchestration"]["development_seed"]
    files = [path / "multi_agent/lock.json", path / "multi_agent/summary/evidence.json",
             path / "multi_agent/summary/response.json", path / "lead_summary.md"]
    files.extend((path / "multi_agent/audits").glob("*.json"))
    contents = {file: file.read_bytes() for file in files}
    monkeypatch.setattr(multi_agent, "make", lambda *args, **kwargs: pytest.fail("frozen claims must not reopen research"))
    assert multi_agent.drive(path, fake_scripts=fake, out=lambda *_: None) == "finished"
    assert all(file.read_bytes() == before for file, before in contents.items())


@pytest.mark.parametrize("change, pattern", [
    ({"claim_scope": {"kind": "all"}, "scope": {"id": "all"}}, "outside its assignment"),
    ({"assigned_scope_id": "all"}, "changed its assigned domain"),
    ({"scope": {"id": "all"}}, "inconsistent claim and measured"),
])
def test_ingest_rejects_broadened_or_inconsistent_claims_before_selection(tmp_path, change, pattern):
    assigned = {"kind": "lengths", "lengths": [1]}
    session, cfg = research_session(tmp_path, assigned)
    session.submit_entry("program", 1.0)
    receipt = copy.deepcopy(session.receipts()[0])
    receipt.update(change)
    plan = {"id": "a0001", "scope": assigned}
    archive = multi_agent._empty_archive()
    with pytest.raises(RuntimeError, match=pattern):
        multi_agent._ingest(tmp_path / "run", cfg, session.model(), archive, plan, {"submissions": [receipt]})
    assert archive["candidates"] == [] and archive["global"]["best"] is None


def test_ingest_keeps_frozen_sampler_options_when_semantic_scope_ids_match(tmp_path):
    assigned = {"kind": "lengths", "lengths": [1]}
    session, cfg = research_session(tmp_path, assigned)
    cfg["orchestration"]["development_seed"] = cfg["verifier"]["sample_seed"]
    for cap, samples in ((100, 10), (900, 50)):
        session.submit_entry("program", 1.0, scope={"kind": "predicate", "entry": gate(32),
                                                    "max_draws": cap, "mass_samples": samples})
    receipts = session.receipts()
    assert receipts[0]["scope"]["id"] == receipts[1]["scope"]["id"]
    assert receipts[0]["claim_scope"] != receipts[1]["claim_scope"]
    plan = {"id": "a0001", "scope": assigned, "directory": ".", "role": "regional", "cycle": 1}
    archive = multi_agent._empty_archive()
    multi_agent._ingest(tmp_path, cfg, session.model(), archive, plan, {"submissions": receipts})
    for candidate in archive["candidates"]:
        assert candidate["scope"] == candidate["submitted"]["claim_scope"]
        assert candidate["selection"]["scope"]["spec"] == candidate["scope"]
        assert candidate["selection"]["parts"] == candidate["submitted"]["parts"]
    assert len(archive["candidates"]) == 2 and len(archive["regions"]) == 1
    assert archive["regions"][receipts[0]["scope"]["id"]]["frontier"] == ["a0001s1"]
