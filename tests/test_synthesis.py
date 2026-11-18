"""Post-lock explanations use the real coordinator and a scripted, unpaid backend."""

import copy
import hashlib

import pytest

from level1 import budget, lang, multi_agent, prompt, run_config, synthesis
from level1.util import read_json, write_json


ANSWER = "# Lead synthesis\n\nThe global selection and the regional interpretation answer different questions.\n"


def prepare(tmp_path, summary_script=None, **overrides):
    base = {"run_id": "synthesis-test", "runs_dir": str(tmp_path), "backend": "fake", "sandbox": "process",
            "rounds_max": 1, "session": {"python_calls": 20, "submissions": 4},
            "verifier": {"check": "sample", "sample_size": 64, "reference_samples": 1},
            "orchestration": {"enabled": True, "workers": 1, "final_summary": True, "baseline_agent": False}}
    cfg = run_config.load("config/pilot/lin-a-1to4.yaml", run_config.merge(base, overrides))
    path = run_config.run_path(cfg)
    path.mkdir(parents=True)
    write_json(path / "run.json", {"config": cfg, "run_id": cfg["run_id"], "sandbox": {"mode": "process"}})
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    items = [
        {"calls": [{"tool": "assign_region", "args": {"scope": {"kind": "lengths", "lengths": [2]},
                                                           "task": "Investigate this input class."}}],
         "response": "## What the network does\nInitial planning notes."},
        {"calls": [{"tool": "run_python", "args": {"code": "g=graph()\nsave_entry('local',g.entry(predict=lv1.maximum(g.token(0),g.token(1))))"}},
                   {"tool": "submit_entry", "args": {"name": "local", "claimed_fidelity": 1.0,
                                                         "expected_fidelity_other": 1.0}}],
         "response": "## What the network does\nA local hypothesis, not an independently checked derivation.\n## Next\nUNTRUSTED_NOTE_MARKER"},
        {"calls": [{"tool": "run_python", "args": {"code": "g=graph()\nsave_entry('global',g.entry(predict=model.reference(g,lv1)))"}},
                   {"tool": "submit_entry", "args": {"name": "global", "claimed_fidelity": 1.0}}],
         "response": "## What the network does\nA global reference interpretation was submitted."},
    ]
    if summary_script is not None:
        items.append(summary_script)
    for index, item in enumerate(items, 1):
        write_json(scripts / f"{index:02d}.json", item)
    return path, cfg, scripts


def scientific_files(path):
    names = [path / "multi_agent/archive.json", path / "multi_agent/lock.json", path / "locked_entry.json",
             path / "lock.json", path / "seed_check.json", *(path / "multi_agent/audits").glob("*.json")]
    return {p: p.read_bytes() for p in names}


def assert_unchanged(snapshot):
    assert all(path.read_bytes() == content for path, content in snapshot.items())


@pytest.fixture(scope="module")
def completed(tmp_path_factory):
    path, cfg, scripts = prepare(tmp_path_factory.mktemp("summary"), {"response": ANSWER, "calls": [
        {"tool": "run_python", "args": {"code": "raise RuntimeError('must never execute')"}},
        {"tool": "submit_entry", "args": {"name": "global", "claimed_fidelity": 1}},
        {"tool": "assign_region", "args": {"scope": {"kind": "all"}, "task": "Change the research plan."}},
    ]})
    assert multi_agent.drive(path, fake_scripts=scripts, out=lambda *_: None) == "finished"
    return path, cfg, scripts


def test_summary_runs_after_frozen_audits_with_no_research_tools(completed):
    path, cfg, _ = completed
    state = read_json(path / "multi_agent/state.json")
    plan = state["sessions"][state["summary"]]
    assert plan["index"] == 3 and plan["role"] == "summary" and plan["status"] == "completed"
    session = read_json(path / "multi_agent/summary/session/session.json")
    assert session["run_path"] == str(path.resolve())
    assert session["budgets"] == {"run_python": 0, "submit_entry": 0}
    assert session["used"] == {"run_python": 0, "submit_entry": 0}
    assert session["assignments"] == []
    meta = read_json(path / "multi_agent/summary/meta.json")
    assert len(meta["tool_calls"]) == 3 and all(not call["ok"] for call in meta["tool_calls"])
    frozen = read_json(path / "multi_agent/lock.json")
    snapshot = read_json(path / "multi_agent/summary/evidence.json")
    assert snapshot["frozen_selection"]["global"] == frozen["global"]
    assert snapshot["frozen_selection"]["regional"] == frozen["regional"]
    assert {c["id"] for c in snapshot["candidates"]} == {c["id"] for c in frozen["candidates"]}
    for candidate in snapshot["candidates"]:
        assert candidate["program"]["nodes"]
        assert candidate["development"]["scope"]["workload_size"] > 0
        assert candidate["development"]["scope"]["mass"] > 0
        assert candidate["final_audits"]["fresh"]["network"] == cfg["network"]
        assert candidate["final_audits"]["held_out"]["network"] == cfg["seed_check"]["network"]
    assert snapshot["comparison"]["provided"] is False
    assert snapshot["comparison"]["baseline_search"] is False
    assert snapshot["comparison"]["search_sessions"] == 0
    assert snapshot["comparison"]["candidate_parameter_control_provided"] is False
    assert "random_weight_control_provided" not in snapshot["comparison"]
    assert "UNTRUSTED_NOTE_MARKER" in str(snapshot["prior_notes"])
    assert len(multi_agent.read_archive(path)["candidates"]) == 2


def test_completed_summary_is_reported_and_never_reopens_research(completed, monkeypatch):
    path, _, scripts = completed
    before = scientific_files(path)
    assert (path / "lead_summary.md").read_text() == ANSWER
    report = read_json(path / "report.json")
    assert report["lead_summary"]["status"] == "completed"
    assert report["lead_summary"]["markdown"] == ANSWER
    assert "[Read the final lead synthesis](lead_summary.md)" in (path / "report.md").read_text()
    monkeypatch.setattr(synthesis, "make", lambda *a, **k: pytest.fail("a saved summary needs no new backend call"))
    monkeypatch.setattr(multi_agent, "make", lambda *a, **k: pytest.fail("a locked experiment cannot resume research"))
    assert multi_agent.drive(path, fake_scripts=scripts, out=lambda *_: None) == "finished"
    assert_unchanged(before)


def test_summary_reserves_shared_api_turn_and_time_allowances(tmp_path):
    _, cfg, _ = prepare(tmp_path, rounds_max=5, session={"python_calls": 40, "submissions": 4},
                        orchestration={"workers": 2})
    roles = [("lead", 0)] + [(role, cycle) for cycle in range(1, 6) for role in ("regional", "regional", "lead")]
    original = copy.deepcopy(cfg)
    allocations = [multi_agent._allocation(cfg, role, cycle, 2) for role, cycle in roles]
    summary = multi_agent._allocation(cfg, "summary", 6, 1)
    assert summary[:2] == (0, 0) and summary[2] == pytest.approx(6)
    assert allocations[0][2] == pytest.approx(5.4)
    assert sum(a[0] for a in allocations) == 200
    assert sum(a[1] for a in allocations) == 20
    assert sum(a[2] for a in allocations) + summary[2] == pytest.approx(60)
    for key, integer in (("max_turns", True), ("seconds", False)):
        total = sum(multi_agent._quota(cfg, cfg["session"][key], role, cycle, 2, integer) for role, cycle in roles)
        total += multi_agent._quota(cfg, cfg["session"][key], "summary", 6, 1, integer)
        assert total <= cfg["session"][key] * cfg["rounds_max"]
    assert cfg == original


def test_missing_response_preserves_lock_and_can_resume_only_summary(tmp_path, monkeypatch):
    path, _, scripts = prepare(tmp_path)
    assert multi_agent.drive(path, fake_scripts=scripts, out=lambda *_: None) == "finished"
    assert read_json(path / "multi_agent/summary/status.json")["status"] == "unavailable"
    assert read_json(path / "report.json")["lead_summary"]["status"] == "unavailable"
    assert not (path / "lead_summary.md").exists()
    before = scientific_files(path)
    evidence = (path / "multi_agent/summary/evidence.json").read_bytes()
    write_json(scripts / "04.json", {"response": ANSWER})
    monkeypatch.setattr(multi_agent, "make", lambda *a, **k: pytest.fail("resume must not reopen research"))
    assert multi_agent.drive(path, fake_scripts=scripts, out=lambda *_: None) == "finished"
    assert read_json(path / "multi_agent/summary/status.json")["status"] == "completed"
    assert read_json(path / "multi_agent/state.json")["next_index"] == 4
    assert (path / "multi_agent/summary/evidence.json").read_bytes() == evidence
    assert_unchanged(before)


def test_interrupted_summary_resumes_with_same_plan(tmp_path):
    path, _, scripts = prepare(tmp_path, {"stop_run": "scripted summary interruption"})
    assert multi_agent.drive(path, fake_scripts=scripts, out=lambda *_: None) == "finished"
    before = scientific_files(path)
    plan = read_json(path / "multi_agent/summary/plan.json")
    assert read_json(path / "multi_agent/summary/status.json")["status"] == "stopped"
    assert not (path / "lead_summary.md").exists()
    write_json(scripts / "04.json", {"response": ANSWER})
    assert multi_agent.drive(path, fake_scripts=scripts, out=lambda *_: None) == "finished"
    assert read_json(path / "multi_agent/summary/plan.json") == plan
    assert (path / "lead_summary.md").read_text() == ANSWER
    assert_unchanged(before)


def test_saved_summary_response_recovers_before_spent_budget_check(tmp_path, monkeypatch):
    path, cfg, scripts = prepare(tmp_path, auth="api_key")
    assert multi_agent.drive(path, fake_scripts=scripts, out=lambda *_: None) == "finished"
    before = scientific_files(path)
    write_json(path / "multi_agent/summary/response.json", {"text": ANSWER, "meta": {"stop_reason": "end_turn", "backend": "fake"}})
    budget.start(path, 99, 1, cfg["budget"]["run_usd"])
    monkeypatch.setattr(synthesis, "make", lambda *a, **k: pytest.fail("recovery must not make a paid call"))
    assert multi_agent.drive(path, fake_scripts=scripts, out=lambda *_: None) == "finished"
    assert (path / "lead_summary.md").read_text() == ANSWER
    assert read_json(path / "multi_agent/summary/status.json")["status"] == "completed"
    assert_unchanged(before)


@pytest.mark.parametrize("script,status", [({"response": ""}, "empty"),
                                          ({"response": "Partial analysis", "stop_reason": "budget"}, "incomplete"),
                                          ({"response": ANSWER, "stop_reason": "timeout"}, "incomplete"),
                                          ({"response": ANSWER, "stop_reason": "turn_limit"}, "incomplete")])
def test_noncompletion_is_never_presented_as_a_generated_summary(tmp_path, monkeypatch, script, status):
    path, _, scripts = prepare(tmp_path, script)
    assert multi_agent.drive(path, fake_scripts=scripts, out=lambda *_: None) == "finished"
    assert read_json(path / "multi_agent/summary/status.json")["status"] == status
    assert read_json(path / "report.json")["lead_summary"]["status"] == status
    assert not (path / "lead_summary.md").exists()
    assert read_json(path / "lock.json")["locked"]
    assert (path / "multi_agent/summary/response.md").read_text() == script["response"]
    before = scientific_files(path)
    monkeypatch.setattr(synthesis, "make", lambda *a, **k: pytest.fail("a stopped response must not trigger another summary call"))
    assert multi_agent.drive(path, fake_scripts=scripts, out=lambda *_: None) == "finished"
    assert read_json(path / "report.json")["lead_summary"]["status"] == status
    assert not (path / "lead_summary.md").exists()
    assert_unchanged(before)


def test_old_saved_configuration_does_not_gain_a_summary_call(tmp_path, monkeypatch):
    path, cfg, scripts = prepare(tmp_path)
    cfg["orchestration"].pop("final_summary")
    cfg["orchestration"].pop("summary_fraction")
    run = read_json(path / "run.json")
    run["config"] = cfg
    write_json(path / "run.json", run)
    monkeypatch.setattr(synthesis, "make", lambda *a, **k: pytest.fail("old saved runs cannot acquire new paid work"))
    assert multi_agent.drive(path, fake_scripts=scripts, out=lambda *_: None) == "finished"
    assert not (path / "multi_agent/summary").exists()
    assert read_json(path / "report.json")["lead_summary"]["status"] == "disabled"
    assert multi_agent._allocation(cfg, "lead", 0, 1)[2] == pytest.approx(6)


def test_completed_checkpoint_repairs_missing_record_without_another_call(tmp_path):
    directory = tmp_path / "multi_agent/summary"
    plan = {"id": "summary", "index": 7}
    saved = {"text": ANSWER, "meta": {"stop_reason": "end_turn"}}
    write_json(directory / "status.json", {"status": "completed", "updated_at": "recorded-time",
                                           "sha256": hashlib.sha256(ANSWER.encode()).hexdigest()})
    result = synthesis._publish(tmp_path, directory, plan, saved)
    assert result["status"] == "completed"
    assert read_json(directory / "record.json")["at"] == "recorded-time"
    assert (tmp_path / "lead_summary.md").read_text() == ANSWER


def test_report_distinguishes_modified_text_from_recorded_agent_output(tmp_path):
    write_json(tmp_path / "multi_agent/summary/status.json", {"status": "completed",
               "sha256": hashlib.sha256(ANSWER.encode()).hexdigest()})
    (tmp_path / "lead_summary.md").write_text("Operator edit\n")
    result = synthesis.report(tmp_path, {})
    assert result["status"] == "modified" and "markdown" not in result


def test_summary_hygiene_neutralizes_operator_labels_and_preserves_discoveries(tmp_path):
    path, _, scripts = prepare(tmp_path, {"response": ANSWER}, run_id="OPERATOR_ONLY_LABEL",
                               hygiene=["OPERATOR_ONLY_LABEL", "RESEARCH_DISCOVERY_MARKER"])
    worker = read_json(scripts / "02.json")
    worker["response"] += "\nRESEARCH_DISCOVERY_MARKER"
    write_json(scripts / "02.json", worker)
    assert multi_agent.drive(path, fake_scripts=scripts, out=lambda *_: None) == "finished"
    assert read_json(path / "multi_agent/summary/status.json")["status"] == "completed"
    assert read_json(path / "multi_agent/summary/evidence.json")["run_id"] == "OPERATOR_ONLY_LABEL"
    text = (path / "multi_agent/summary/prompt.md").read_text()
    assert "OPERATOR_ONLY_LABEL" not in text and "RESEARCH_DISCOVERY_MARKER" in text


def test_baseline_only_has_no_paid_summary_or_summary_artifacts(tmp_path, monkeypatch):
    cfg = {"orchestration": {"enabled": True, "mode": "baseline", "final_summary": True,
                             "summary_fraction": 0.1}}
    assert synthesis.enabled(cfg) is False
    assert synthesis.fraction(cfg) == 0.0
    monkeypatch.setattr(synthesis, "make", lambda *a, **k: pytest.fail("baseline-only mode cannot start a lead session"))
    assert synthesis.run(tmp_path, cfg, None, {}, {}, {}, {}, [], out=lambda *_: None)["status"] == "disabled"
    assert synthesis.report(tmp_path, cfg)["status"] == "disabled"
    assert not (tmp_path / "multi_agent/summary").exists()
    for mode in ("trained", "both"):
        other = copy.deepcopy(cfg)
        other["orchestration"]["mode"] = mode
        assert synthesis.enabled(other)
        assert synthesis.fraction(other) == 0.1


def _receipt(*, reward=0.9, network="target", seed=7):
    return {"status": "scored", "network": network, "prices": lang.PRICES, "reward": reward,
            "reward_detail": {"name": "r_alpha", "params": {"alpha": 10.0}},
            "inputs": {"mode": "sample", "n": 4, "seed": seed, "scope_id": "all"},
            "scope": {"id": "all", "spec": {"kind": "all"}, "workload_size": 16},
            "parts": {"coverage": 1.0, "fidelity": 1.0, "n_inputs": 4, "n_in_region": 4,
                      "n_agree": 4, "workload_size": 16}}


def test_comparison_gap_requires_exact_full_scope_comparator_but_candidate_can_be_partial():
    candidate, comparison = _receipt(), _receipt(reward=0.7)
    candidate["parts"].update(coverage=0.5, fidelity=0.5, n_in_region=2, n_agree=1)
    result = synthesis.comparison_result(candidate, comparison)
    assert result["valid"] is True and result["reward_gap"] == pytest.approx(0.2)


@pytest.mark.parametrize("changes", [
    {"coverage": 0.5, "n_in_region": 2, "n_agree": 2},
    {"fidelity": 0.75, "n_agree": 3},
    {"n_inputs": 0, "n_in_region": 0, "n_agree": 0},
])
def test_incorrect_or_partial_comparator_withholds_gap_and_keeps_raw_arithmetic(changes):
    candidate, comparison = _receipt(), _receipt(reward=0.7)
    comparison["parts"].update(changes)
    result = synthesis.comparison_result(candidate, comparison)
    assert result["valid"] is False and result["reward_gap"] is None
    assert result["reason"]
    assert result["raw_reward_difference"] == pytest.approx(0.2)


@pytest.mark.parametrize("field,value", [("network", "different"), ("prices", None),
                                         ("reward_detail", {"name": "r_alpha", "params": {"alpha": 1.0}}),
                                         ("reward_detail", None),
                                         ("inputs", {"mode": "sample", "n": 4, "seed": 9}),
                                         ("scope", {"id": "different", "spec": {"kind": "all"}})])
def test_unmatched_or_missing_conditions_withhold_comparison_gap(field, value):
    candidate, comparison = _receipt(), _receipt(reward=0.7)
    comparison[field] = value
    assert synthesis.comparison_result(candidate, comparison)["reward_gap"] is None
    assert synthesis.comparison_result(candidate, comparison)["valid"] is False


def test_comparison_rejects_changed_workload_or_old_incomplete_receipts():
    candidate, comparison = _receipt(), _receipt(reward=0.7)
    comparison["parts"]["workload_size"] = 4096
    assert "workload" in synthesis.comparison_result(candidate, comparison)["reason"]
    old = {"status": "scored", "reward": 0.7, "parts": {"coverage": 1.0, "fidelity": 1.0}}
    assert synthesis.comparison_result(candidate, old)["valid"] is False
    assert synthesis.comparison_result(candidate, None)["reward_gap"] is None


def _evidence_fixture(tmp_path, metadata):
    cfg = run_config.load("config/pilot/lin-c.yaml", {"backend": "fake", "sandbox": "process"})
    program = {"format": lang.FORMAT, "nodes": [{"op": "const", "dtype": "int", "value": 0}],
               "predict": 0, "rule": None}
    digest = lang.entry_hash(program)
    candidate = {"id": "candidate", "entry_sha256": digest, "entry_path": "candidate.json",
                 "role": "lead", "scope_id": "all", "scope": {"kind": "all"},
                 "selection": _receipt(network=cfg["network"])}
    frozen = {"global": "candidate", "regional": {}, "candidates": [candidate], "at": "frozen-time"}
    archive = {"baseline_scores": {"all": _receipt(network=cfg["network"], reward=0.7)}}
    audit = {"candidate_id": "candidate", "entry_sha256": digest, "scope": candidate["scope"],
             "fresh": _receipt(network=cfg["network"], seed=11),
             "fresh_baseline": _receipt(network=cfg["network"], reward=0.7, seed=11),
             "held_out": _receipt(network=cfg["seed_check"]["network"], seed=13),
             "held_out_baseline": _receipt(network=cfg["seed_check"]["network"], reward=0.7, seed=13)}
    audit["fresh_baseline"]["parts"].update(fidelity=0.75, n_agree=3)
    write_json(tmp_path / "candidate.json", program)
    write_json(tmp_path / "multi_agent/baseline_entry.json", program)
    write_json(tmp_path / "multi_agent/baseline.json", {"entry_sha256": digest, **metadata})
    write_json(tmp_path / "multi_agent/state.json", {})
    write_json(tmp_path / "multi_agent/audits/candidate.json", audit)
    return cfg, synthesis.evidence(tmp_path, cfg, frozen, archive, [])


@pytest.mark.parametrize("source,sessions", [("single_agent_search", 1), ("provided", 0)])
def test_v2_acceptance_provenance_and_actual_scope_validity_are_separate(tmp_path, source, sessions):
    _, evidence = _evidence_fixture(tmp_path, {
        "source": source, "status": "established", "validation_policy_version": 2,
        "trained_checkpoints_used_for_acceptance": True, "trained_rewards_used_for_selection": False,
        "trained_receipts": {"PRIVATE_TRAINED_MODEL/network": {"model": "PRIVATE_TRAINED_MODEL"}},
        "control_receipts": {"PRIVATE_CONTROL_NETWORK": {"network": "PRIVATE_CONTROL_NETWORK"}},
    })
    comparison = evidence["comparison"]
    assert comparison["validation_established"] is True
    assert comparison["validation_policy_version"] == 2
    assert comparison["trained_checkpoints_used_for_acceptance"] is True
    assert comparison["trained_rewards_used_for_selection"] is False
    assert comparison["candidate_parameter_control_provided"] is False
    assert "random_weight_control_provided" not in comparison
    assert comparison["search_sessions"] == sessions
    assert comparison["baseline_search"] == (sessions == 1)
    checks = evidence["candidates"][0]["comparisons"]
    assert checks["development"]["valid"] is True
    assert checks["fresh"]["valid"] is False and checks["fresh"]["reward_gap"] is None
    assert checks["fresh"]["raw_reward_difference"] == pytest.approx(0.2)
    assert checks["held_out"]["valid"] is True
    view = prompt.summary_evidence(evidence)
    assert view["comparison"]["independent_comparison_search"] == (sessions == 1)
    assert view["comparison"]["candidate_parameter_control_provided"] is False
    assert "random_weight_control_provided" not in view["comparison"]
    assert view["comparison"]["additional_checkpoints_used_for_acceptance"] is True
    assert view["comparison"]["additional_checkpoint_rewards_used_for_selection"] is False
    assert "trained_checkpoints_used_for_acceptance" not in view["comparison"]
    assert "PRIVATE_TRAINED_MODEL" not in str(view) and "PRIVATE_CONTROL_NETWORK" not in str(view)
    assert "search_record" not in view["comparison"]
    assert view["candidates"][0]["comparisons"] == checks
    assert prompt.summary_evidence(view) == view


def test_old_control_only_record_is_not_upgraded_by_current_defaults(tmp_path):
    cfg, evidence = _evidence_fixture(tmp_path, {"source": "single_agent_search", "status": "established"})
    assert cfg["orchestration"]["baseline_validation_version"] == 2
    comparison = evidence["comparison"]
    assert comparison["validation_policy_version"] == 1 and comparison["legacy_validation"] is True
    assert comparison["trained_checkpoints_used_for_acceptance"] is False
    assert "only search-family" in comparison["validation_note"]
    assert "not been upgraded" in comparison["validation_note"]


def test_legacy_import_is_not_claimed_to_have_parameter_validation(tmp_path):
    _, evidence = _evidence_fixture(tmp_path, {"source": "provided", "status": "provided"})
    assert evidence["comparison"]["validation_established"] is False
    assert evidence["comparison"]["search_sessions"] == 0


@pytest.mark.parametrize("source,search", [("single_agent_search", True), ("provided", False)])
@pytest.mark.parametrize("legacy_control_flag", [False, True])
def test_legacy_control_flag_cannot_override_comparison_search_provenance(source, search, legacy_control_flag):
    raw = {"comparison": {"provided": True, "source": source, "baseline_search": search,
                          "search_sessions": int(search), "validation_established": False,
                          "same_search_conditions": False,
                          "candidate_parameter_control_provided": False,
                          "random_weight_control_provided": legacy_control_flag}}
    before = copy.deepcopy(raw)
    view = prompt.summary_evidence(raw)
    comparison = view["comparison"]
    assert comparison["independent_comparison_search"] is search
    assert comparison["search_sessions"] == int(search)
    assert comparison["validation_established"] is False
    assert comparison["same_search_conditions"] is False
    assert comparison["candidate_parameter_control_provided"] is False
    assert "random_weight_control_provided" not in comparison
    assert "checked on other parameter settings" not in comparison["provenance"]
    assert prompt.summary_evidence(view) == view
    assert raw == before
