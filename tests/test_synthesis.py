"""Post-lock explanations use the real coordinator and a scripted, unpaid backend."""

import copy
import hashlib

import pytest

from level1 import budget, multi_agent, run_config, synthesis
from level1.util import read_json, write_json


ANSWER = "# Lead synthesis\n\nThe global selection and the regional interpretation answer different questions.\n"


def prepare(tmp_path, summary_script=None, **overrides):
    base = {"run_id": "synthesis-test", "runs_dir": str(tmp_path), "backend": "fake", "sandbox": "process",
            "rounds_max": 1, "session": {"python_calls": 20, "submissions": 4},
            "verifier": {"check": "sample", "sample_size": 64, "reference_samples": 1},
            "orchestration": {"enabled": True, "workers": 1, "final_summary": True}}
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
    assert snapshot["comparison"]["random_weight_control_provided"] is False
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
