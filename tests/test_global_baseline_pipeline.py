"""The automatic global comparator runs once inside the real unpaid pipeline."""

import pytest

from level1 import global_baseline, lang, loop, multi_agent, regions, run_config
from level1.backends import StopRun
from level1.backends.fake import FakeBackend
from level1.networks import load_model
from level1.util import read_json, write_json


CONTROL_NOTE = "CONTROL_SESSION_PRIVATE_NOTE"
SUMMARY = "# Research summary\n\nGlobal and regional comparisons are available.\n"


def config(tmp_path, **overrides):
    base = {"run_id": "comparison-pipeline-test", "runs_dir": str(tmp_path),
            "backend": "fake", "sandbox": "process", "rounds_max": 1,
            "session": {"python_calls": 40, "submissions": 4},
            "budget": {"run_usd": 25, "round_usd": 20},
            "verifier": {"check": "sample", "sample_size": 32, "reference_samples": 1},
            "orchestration": {"enabled": True, "workers": 2, "parallel": True,
                              "baseline_agent": True, "baseline_fraction": 0.2,
                              "baseline_submissions": 2, "development_seed": 73,
                              "final_summary": True}}
    return run_config.load("config/pilot/lin-c-smoke.yaml", run_config.merge(base, overrides))


def prepare(tmp_path, *, supplied=False, baseline_kind="reference", old_config=False, **overrides):
    cfg = config(tmp_path, **overrides)
    model = load_model(cfg["model"])
    reference = model.reference_entry(model.load_weights(cfg["network"]))
    if supplied:
        source = tmp_path / "supplied-entry.json"
        write_json(source, reference)
        cfg["orchestration"]["baseline_entry"] = str(source)
    if old_config:
        for key in ("baseline_agent", "baseline_fraction", "baseline_submissions"):
            cfg["orchestration"].pop(key, None)
    path = run_config.run_path(cfg)
    path.mkdir(parents=True)
    write_json(path / "run.json", {"run_id": cfg["run_id"], "config": cfg, "prices": lang.PRICES,
                                   "sandbox": {"mode": "process"}})
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    controls = []
    if global_baseline.enabled(cfg):
        code = "assert not entries\ng=graph()\nsave_entry('control_only_program',g.entry(predict=model.reference(g,lv1)))"
        if baseline_kind == "partial":
            code = ("assert not entries\ng=graph()\n"
                    "save_entry('control_only_program',g.entry(predict=model.reference(g,lv1),"
                    "rule=g.token(0)<(info['input_space'][0]//2)))")
        calls = [{"tool": "run_python", "args": {"code": code}}]
        if baseline_kind != "draft_only":
            calls.append({"tool": "submit_entry", "args": {"name": "control_only_program", "claimed_fidelity": 1.0}})
        controls.append({"calls": calls, "response": "## What the network does\n" + CONTROL_NOTE})
    planning = {"calls": [
        {"tool": "assign_region", "args": {"scope": {"kind": "lengths", "lengths": [1, 2]},
                                             "task": "Investigate a compact predictor in this input class."}},
        {"tool": "assign_region", "args": {"scope": {"kind": "lengths", "lengths": [3, 4]},
                                             "task": "Investigate this separate input class."}},
    ], "response": "## What the network does\nTwo regional investigations are ready."}
    reference_code = "g=graph()\nsave_entry('candidate',g.entry(predict=model.reference(g,lv1)))"
    workers = [{"calls": [
        {"tool": "run_python", "args": {"code": reference_code}},
        {"tool": "submit_entry", "args": {"name": "candidate", "claimed_fidelity": 1.0,
                                             "expected_fidelity_other": 1.0,
                                             "scope": {"kind": "lengths", "lengths": [length]}}},
    ], "response": "## What the network does\nA narrower regional claim was submitted."} for length in (2, 3)]
    integration = {"calls": [
        {"tool": "run_python", "args": {"code": reference_code}},
        {"tool": "submit_entry", "args": {"name": "candidate", "claimed_fidelity": 1.0,
                                             "expected_fidelity_other": 1.0}},
    ], "response": "## What the network does\nA full-domain program was submitted."}
    items = [*controls, planning, *workers, integration]
    if cfg["orchestration"].get("final_summary"):
        items.append({"response": SUMMARY})
    for number, script in enumerate(items, 1):
        write_json(scripts / f"{number:02d}.json", script)
    return path, cfg, scripts


@pytest.fixture(scope="module")
def completed(tmp_path_factory):
    path, cfg, scripts = prepare(tmp_path_factory.mktemp("global-comparison"))
    messages = []
    assert multi_agent.drive(path, fake_scripts=scripts, out=messages.append) == "finished"
    return path, cfg, scripts, messages


def test_one_global_baseline_precedes_research_and_is_not_a_research_candidate(completed):
    path, cfg, _, _ = completed
    state = read_json(path / "multi_agent/state.json")
    plans = sorted(state["sessions"].values(), key=lambda item: item["index"])
    assert [plan["role"] for plan in plans] == ["baseline", "lead", "regional", "regional", "lead", "summary"]
    assert [plan["index"] for plan in plans] == list(range(6))
    assert state["baseline"] == plans[0]["id"] and state["planning"] == plans[1]["id"]
    assert all(plan["status"] == "completed" for plan in plans)
    baseline = read_json(path / "multi_agent/baseline.json")
    assert baseline["status"] == "established" and baseline["source"] == "single_agent_search"
    archive = multi_agent.read_archive(path)
    assert len(archive["candidates"]) == 3 and archive["global"]["best"] is not None
    assert all(candidate["role"] != "baseline" for candidate in archive["candidates"])
    assert all(item.get("session_id") != plans[0]["id"] for item in archive["drafts"])
    report = read_json(path / "report.json")
    assert report["baseline"]["status"] == "established"
    assert report["lead_summary"]["status"] == "completed"


def test_frozen_global_program_is_measured_on_new_regional_claims_under_same_conditions(completed):
    path, cfg, _, _ = completed
    archive = multi_agent.read_archive(path)
    comparisons = archive["baseline_scores"]
    expected = [{"kind": "all"}, *({"kind": "lengths", "lengths": lengths}
                                     for lengths in ([1, 2], [3, 4], [2], [3]))]
    assert set(comparisons) == {regions.scope_id(scope) for scope in expected}
    digest = lang.entry_hash(read_json(path / "multi_agent/baseline_entry.json"))
    assert all(receipt["entry_sha256"] == digest for receipt in comparisons.values())
    for candidate in archive["candidates"]:
        result, comparator = candidate["selection"], comparisons[candidate["scope_id"]]
        assert comparator["network"] == cfg["network"]
        assert comparator["inputs"] == result["inputs"]
        assert comparator["prices"] == result["prices"] == lang.PRICES
        assert comparator["scope"] == result["scope"]
        assert comparator["parts"]["workload_size"] == result["parts"]["workload_size"]
        assert result["reward"] - comparator["reward"] == pytest.approx(0)


def test_final_checks_and_synthesis_include_frozen_comparisons_with_honest_provenance(completed):
    path, cfg, _, _ = completed
    frozen = read_json(path / "multi_agent/lock.json")
    for candidate in frozen["candidates"]:
        audit = read_json(path / "multi_agent/audits" / (candidate["id"] + ".json"))
        for purpose, network in (("fresh", cfg["network"]), ("held_out", cfg["seed_check"]["network"])):
            result, comparator = audit[purpose], audit[purpose + "_baseline"]
            assert result["network"] == comparator["network"] == network
            assert result["inputs"] == comparator["inputs"]
            assert result["prices"] == comparator["prices"]
            assert result["parts"]["workload_size"] == comparator["parts"]["workload_size"]
            assert result["reward"] - comparator["reward"] == pytest.approx(0)
    evidence = read_json(path / "multi_agent/summary/evidence.json")
    comparison = evidence["comparison"]
    assert comparison["provided"] and comparison["validation_established"]
    assert comparison["source"] == "single_agent_search" and comparison["search_sessions"] == 1
    assert comparison["baseline_search"] and not comparison["same_search_conditions"]
    assert comparison["candidate_parameter_control_provided"] is False
    assert comparison["random_weight_control_provided"] is False
    assert "not a universal proof" in comparison["validation_note"]
    assert CONTROL_NOTE not in str(evidence["prior_notes"])


def test_baseline_context_is_independent_and_only_its_frozen_program_is_carried(completed):
    path, cfg, _, _ = completed
    state = read_json(path / "multi_agent/state.json")
    plan = state["sessions"][state["baseline"]]
    session = read_json(path / plan["directory"] / "session/session.json")
    control_cfg = global_baseline.config_for(cfg)
    assert session["model"] == control_cfg["model"] != cfg["model"]
    assert session["network"] == control_cfg["network"] != cfg["network"]
    assert plan["history"] == [] and plan["snapshot"]["candidates"] == []
    metadata = read_json(path / plan["directory"] / "meta.json")
    assert all(call["ok"] for call in metadata["tool_calls"])
    assert session["used"] == {"run_python": 1, "submit_entry": 1}
    prompt = (path / plan["directory"] / "prompt.md").read_text()
    assert "single investigator" in prompt and "any weights" in prompt
    for later in state["sessions"].values():
        if later["role"] in ("baseline", "summary"):
            continue
        text = (path / later["directory"] / "prompt.md").read_text()
        assert CONTROL_NOTE not in text
        assert all(item["role"] != "baseline" for item in (later.get("history") or []))
        carried = path / later["directory"] / "session/workspace/entries/comparison_entry.json"
        assert carried.is_file()
        assert lang.entry_hash(read_json(carried)) == lang.entry_hash(read_json(path / "multi_agent/baseline_entry.json"))
        assert not (carried.parent / "control_only_program.json").exists()


def test_baseline_and_research_share_parent_budget_and_bounded_allocations(completed):
    path, cfg, _, _ = completed
    state = read_json(path / "multi_agent/state.json")
    cap, calls, turns, seconds = 0, 0, 0, 0
    for plan in state["sessions"].values():
        role_cfg = multi_agent.role_config(cfg, plan)
        location = path / ("multi_agent/summary" if plan["role"] == "summary" else plan["directory"]) / "session/session.json"
        session = read_json(location)
        assert session["run_path"] == str(path.resolve())
        assert role_cfg["budget"]["run_usd"] == cfg["budget"]["run_usd"]
        if plan["role"] == "baseline":
            assert role_cfg["budget"]["cycle_id"] == "baseline"
            assert role_cfg["budget"]["round_usd"] == pytest.approx(5)
        cap += role_cfg["budget"]["round_usd"]
        calls += role_cfg["session"]["python_calls"]
        turns += role_cfg["session"]["max_turns"]
        seconds += role_cfg["session"]["seconds"]
    assert cap <= cfg["budget"]["run_usd"] + 1e-10
    assert calls <= cfg["session"]["python_calls"] * cfg["rounds_max"]
    assert turns <= cfg["session"]["max_turns"] * cfg["rounds_max"]
    assert seconds <= cfg["session"]["seconds"] * cfg["rounds_max"] + 1e-10


def test_resume_after_frozen_baseline_does_not_repeat_its_completed_backend_call(tmp_path, monkeypatch):
    path, cfg, scripts = prepare(tmp_path)
    original_respond, original_freeze = FakeBackend.respond, global_baseline.freeze
    roles = []

    def respond(self, prompt, index, where, session):
        roles.append(session.state["research_task"]["role"])
        return original_respond(self, prompt, index, where, session)

    def pause_after_freeze(*args, **kwargs):
        original_freeze(*args, **kwargs)
        raise StopRun("simulated stop after the baseline checkpoint")

    monkeypatch.setattr(FakeBackend, "respond", respond)
    monkeypatch.setattr(global_baseline, "freeze", pause_after_freeze)
    assert multi_agent.drive(path, fake_scripts=scripts, out=lambda *_: None) == "stopped"
    before = {name: (path / name).read_bytes() for name in (global_baseline.FILE, global_baseline.ENTRY)}
    assert roles == ["baseline"]
    monkeypatch.setattr(global_baseline, "freeze", original_freeze)
    assert multi_agent.drive(path, fake_scripts=scripts, out=lambda *_: None) == "finished"
    assert roles.count("baseline") == 1 and len(roles) == 6
    assert all((path / name).read_bytes() == content for name, content in before.items())


def test_supplied_program_suppresses_the_automatic_agent(tmp_path):
    path, cfg, scripts = prepare(tmp_path, supplied=True)
    assert not global_baseline.enabled(cfg)
    assert multi_agent.drive(path, fake_scripts=scripts, out=lambda *_: None) == "finished"
    state = read_json(path / "multi_agent/state.json")
    assert "baseline" not in state and all(plan["role"] != "baseline" for plan in state["sessions"].values())
    assert read_json(path / "multi_agent/baseline.json")["source"] == "provided"
    assert multi_agent.read_archive(path)["baseline_scores"]["all"]["status"] == "scored"


@pytest.mark.parametrize("kind", ["draft_only", "partial"])
def test_failed_global_baseline_stops_before_any_research_agent(tmp_path, kind):
    path, cfg, scripts = prepare(tmp_path, baseline_kind=kind)
    messages = []
    assert multi_agent.drive(path, fake_scripts=scripts, out=messages.append) == "stopped"
    state = read_json(path / "multi_agent/state.json")
    assert len(state["sessions"]) == 1 and "planning" not in state
    assert read_json(path / "multi_agent/baseline.json")["status"] == "failed"
    assert read_json(path / "report.json")["baseline"]["status"] == "failed"
    assert not (path / "multi_agent/baseline_entry.json").exists()
    assert not (path / "multi_agent/lock.json").exists()
    assert not multi_agent.read_archive(path)["candidates"]
    assert any("global baseline was not established" in message for message in messages)


def test_missing_control_fails_start_before_backend_or_run_creation(tmp_path, monkeypatch):
    cfg = config(tmp_path, model="models/lin-a", network="lin-a-s0", seed_check={"network": None})

    def no_backend(*args, **kwargs):
        pytest.fail("a missing control must fail preflight before backend construction")

    monkeypatch.setattr(loop, "make", no_backend)
    with pytest.raises(RuntimeError, match="no baseline model"):
        loop.start(cfg, fake_scripts=[], out=lambda *_: None)
    assert not (run_config.run_path(cfg) / "run.json").exists()


def test_older_saved_config_does_not_acquire_a_paid_baseline_phase(tmp_path):
    path, cfg, scripts = prepare(tmp_path, old_config=True)
    assert not global_baseline.enabled(cfg)
    assert multi_agent.drive(path, fake_scripts=scripts, out=lambda *_: None) == "finished"
    state = read_json(path / "multi_agent/state.json")
    assert "baseline" not in state
    assert all(plan["role"] != "baseline" for plan in state["sessions"].values())
    assert not (path / "multi_agent/baseline.json").exists()
    assert read_json(path / "report.json")["baseline"]["status"] == "not_configured"


@pytest.mark.parametrize("status", ["pending", "failed"])
def test_manual_finish_cannot_bypass_a_required_unestablished_baseline(tmp_path, monkeypatch, status):
    path, cfg, scripts = prepare(tmp_path)
    if status == "failed":
        write_json(path / "multi_agent/baseline.json", {"source": "single_agent_search", "status": "failed",
                                                       "entry_sha256": None, "reason": "No global program passed."})

    def no_backend(*args, **kwargs):
        pytest.fail("manual finish must not start an agent when the baseline is absent")

    monkeypatch.setattr(multi_agent, "make", no_backend)
    from level1 import synthesis
    monkeypatch.setattr(synthesis, "make", no_backend)
    with pytest.raises(RuntimeError, match="required global baseline is not established"):
        multi_agent.finish(path, fake_scripts=scripts, out=lambda *_: None)
    assert not (path / "lock.json").exists()
    assert not (path / "multi_agent/lock.json").exists()
    assert not (path / "multi_agent/summary").exists()
