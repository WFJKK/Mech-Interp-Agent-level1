"""The automatic global comparator runs once inside the real unpaid pipeline."""

import pytest

from level1 import global_baseline, lang, loop, multi_agent, regions, run_config
from level1.backends import StopRun
from level1.backends.fake import FakeBackend
from level1.networks import Model, ModelError, load_model
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
        for key in ("mode", "baseline_agent", "baseline_fraction", "baseline_submissions", "baseline_validation_version"):
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
    assert "random_weight_control_provided" not in comparison
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


@pytest.mark.parametrize("broken", ["missing", "corrupt", "duplicate"])
def test_unusable_later_control_weights_fail_before_backend_construction(tmp_path, monkeypatch, broken):
    cfg = config(tmp_path)
    original_load = Model.load_weights
    visited = []
    first_weights = {}

    def load_weights(model, network):
        visited.append((model.name, network))
        if model.name == "lin-e" and network == "lin-e-s5":
            if broken == "missing":
                raise FileNotFoundError("the confirmation checkpoint is absent")
            if broken == "duplicate":
                return {name: value.copy() for name, value in first_weights.items()}
            model.network(network)["sha256"] = "0" * 64
        weights = original_load(model, network)
        if model.name == "lin-e" and network == "lin-e-s0":
            first_weights.update(weights)
        return weights

    def no_backend(*args, **kwargs):
        pytest.fail("all control checkpoints must validate before backend construction")

    monkeypatch.setattr(Model, "load_weights", load_weights)
    monkeypatch.setattr(loop, "make", no_backend)
    error = {"missing": FileNotFoundError, "corrupt": ModelError, "duplicate": RuntimeError}[broken]
    with pytest.raises(error):
        loop.start(cfg, fake_scripts=[], out=lambda *_: None)
    assert visited == [("lin-e", f"lin-e-s{i}") for i in range(6)]
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


@pytest.mark.parametrize("mode,roles", [
    ("trained", ["lead", "regional", "regional", "lead", "summary"]),
    ("baseline", ["baseline"]),
    ("both", ["baseline", "lead", "regional", "regional", "lead", "summary"]),
])
def test_explicit_workflow_modes_run_exactly_the_requested_stages(tmp_path, mode, roles):
    path, cfg, scripts = prepare(tmp_path, orchestration={"mode": mode})
    messages = []
    assert multi_agent.drive(path, fake_scripts=scripts, out=messages.append) == "finished"
    state = read_json(path / "multi_agent/state.json")
    assert state["mode"] == mode
    plans = sorted(state["sessions"].values(), key=lambda row: row["index"])
    assert [plan["role"] for plan in plans] == roles
    report = read_json(path / "report.json")
    assert report["mode"] == mode and report["completed"] is True
    if mode == "trained":
        assert report["baseline"]["status"] == "not_configured"
        assert not (path / global_baseline.ENTRY).exists()
        assert not multi_agent.read_archive(path)["baseline_scores"]
    else:
        record = read_json(path / global_baseline.FILE)
        assert record["status"] == "established" and record["validation_policy_version"] == 2
        assert record["trained_checkpoints_used_for_acceptance"] is True
        assert record["trained_rewards_used_for_selection"] is False
        assert len(record["trained_receipts"]) == 4
    if mode == "baseline":
        assert not multi_agent.read_archive(path)["candidates"]
        assert not (path / "lead_summary.md").exists()
        assert not (path / "multi_agent/summary").exists()
        assert not (path / "locked_entry.json").exists()
        assert report["lead_summary"]["status"] == "disabled"
        assert read_json(path / "lock.json")["completed"] is True
        assert read_json(path / "lock.json")["locked"] is False
        assert "no trained research sessions ran" in messages[-1]
        assert "## Trained checkpoint checks" in (path / "report.md").read_text()


def test_baseline_only_uses_its_full_allocation_without_research_requirements(tmp_path):
    path, cfg, scripts = prepare(tmp_path, session={"submissions": 0},
        orchestration={"mode": "baseline", "workers": None, "planning_fraction": None,
                       "regional_fraction": None, "summary_fraction": None, "final_summary": None})
    assert multi_agent.drive(path, fake_scripts=scripts, out=lambda *_: None) == "finished"
    state = read_json(path / "multi_agent/state.json")
    plan = state["sessions"][state["baseline"]]
    allocation = multi_agent.role_config(cfg, plan)
    assert allocation["budget"]["round_usd"] == min(cfg["budget"]["run_usd"], cfg["budget"]["round_usd"])
    assert allocation["session"]["python_calls"] == cfg["session"]["python_calls"] * cfg["rounds_max"]
    assert allocation["session"]["max_turns"] == cfg["session"]["max_turns"] * cfg["rounds_max"]
    assert allocation["session"]["seconds"] == cfg["session"]["seconds"] * cfg["rounds_max"]
    assert allocation["session"]["submissions"] == cfg["orchestration"]["baseline_submissions"]


def test_baseline_only_start_pins_every_validation_model_and_ignores_unused_research_settings(tmp_path):
    _, cfg, scripts = prepare(tmp_path, seed_check={"network": "lin-c-s0"}, session={"submissions": 0},
        orchestration={"mode": "baseline", "workers": None, "initial_regions": None,
                       "planning_fraction": None, "regional_fraction": None, "summary_fraction": None})
    cfg["run_id"] = "standalone-start"
    path = loop.start(cfg, fake_scripts=scripts, out=lambda *_: None)
    run = read_json(path / "run.json")
    assert {item["model"] for item in run["global_baseline_validation_models"]} == {
        "models/lin-e", "models/lin-c", "models/lin-a-1to4"}
    assert all(item["trusted_hash"] for item in run["global_baseline_validation_models"])
    state = read_json(path / "multi_agent/state.json")
    assert [p["role"] for p in state["sessions"].values()] == ["baseline"]
    assert read_json(path / "report.json")["completed"] is True


def test_baseline_only_resume_reuses_completed_program_and_checks(tmp_path, monkeypatch):
    path, cfg, scripts = prepare(tmp_path, orchestration={"mode": "baseline"})
    assert multi_agent.drive(path, fake_scripts=scripts, out=lambda *_: None) == "finished"
    names = [global_baseline.FILE, global_baseline.ENTRY, global_baseline.CHECKS, "multi_agent/lock.json"]
    before = {name: (path / name).read_bytes() for name in names}

    def no_backend(*args, **kwargs):
        pytest.fail("a completed standalone baseline must not ask an agent again")

    monkeypatch.setattr(FakeBackend, "respond", no_backend)
    assert multi_agent.drive(path, fake_scripts=scripts, out=lambda *_: None) == "finished"
    assert all((path / name).read_bytes() == content for name, content in before.items())


def test_baseline_only_can_validate_a_supplied_program_without_an_agent(tmp_path, monkeypatch):
    path, cfg, scripts = prepare(tmp_path, supplied=True, orchestration={"mode": "baseline"})

    def no_backend(*args, **kwargs):
        pytest.fail("checking a supplied standalone program requires no agent")

    monkeypatch.setattr(FakeBackend, "respond", no_backend)
    assert multi_agent.drive(path, fake_scripts=scripts, out=lambda *_: None) == "finished"
    state = read_json(path / "multi_agent/state.json")
    assert state["sessions"] == {}
    record = read_json(path / global_baseline.FILE)
    assert record["status"] == "established" and record["source"] == "provided"
    assert record["trained_checkpoints_used_for_acceptance"] is True


def test_invalid_supplied_program_stops_before_research(tmp_path, monkeypatch):
    path, cfg, scripts = prepare(tmp_path, supplied=True)
    model = load_model(cfg["model"])
    graph = lang.Graph(model.load_weights(cfg["network"]), model.n_ctx)
    write_json(cfg["orchestration"]["baseline_entry"], graph.entry(predict=graph.token(0)))

    def no_backend(*args, **kwargs):
        pytest.fail("failed supplied comparison must stop before research")

    monkeypatch.setattr(FakeBackend, "respond", no_backend)
    assert multi_agent.drive(path, fake_scripts=scripts, out=lambda *_: None) == "stopped"
    assert read_json(path / global_baseline.FILE)["status"] == "failed"
    assert not (path / global_baseline.ENTRY).exists()
    assert not (path / "multi_agent/lock.json").exists()


@pytest.mark.parametrize("action", [multi_agent.drive, multi_agent.finish])
def test_completed_baseline_mode_cannot_change_into_paid_research(tmp_path, monkeypatch, action):
    path, cfg, scripts = prepare(tmp_path, orchestration={"mode": "baseline"})
    assert multi_agent.drive(path, fake_scripts=scripts, out=lambda *_: None) == "finished"
    run = read_json(path / "run.json")
    run["config"]["orchestration"]["mode"] = "both"
    write_json(path / "run.json", run)

    def no_backend(*args, **kwargs):
        pytest.fail("a changed saved mode must fail before any new paid stage")

    monkeypatch.setattr(FakeBackend, "respond", no_backend)
    with pytest.raises(RuntimeError, match="saved workflow mode changed"):
        action(path, fake_scripts=scripts, out=lambda *_: None)


@pytest.mark.parametrize("action", [multi_agent.drive, multi_agent.finish])
def test_completed_baseline_rechecks_its_frozen_acceptance_evidence(tmp_path, monkeypatch, action):
    path, cfg, scripts = prepare(tmp_path, orchestration={"mode": "baseline"})
    assert multi_agent.drive(path, fake_scripts=scripts, out=lambda *_: None) == "finished"
    baseline = read_json(path / global_baseline.FILE)
    baseline["trained_receipts"].pop(next(iter(baseline["trained_receipts"])))
    write_json(path / global_baseline.FILE, baseline)
    names = [global_baseline.FILE, global_baseline.ENTRY, "multi_agent/lock.json", "lock.json"]
    before = {name: (path / name).read_bytes() for name in names}

    def no_backend(*args, **kwargs):
        pytest.fail("verifying a completed baseline must not ask an agent")

    monkeypatch.setattr(FakeBackend, "respond", no_backend)
    with pytest.raises(RuntimeError, match="validation evidence changed"):
        action(path, fake_scripts=scripts, out=lambda *_: None)
    assert all((path / name).read_bytes() == value for name, value in before.items())


def test_failed_supplied_validation_cannot_be_downgraded_on_resume(tmp_path, monkeypatch):
    path, cfg, scripts = prepare(tmp_path, supplied=True)
    model = load_model(cfg["model"])
    graph = lang.Graph(model.load_weights(cfg["network"]), model.n_ctx)
    write_json(cfg["orchestration"]["baseline_entry"], graph.entry(predict=graph.token(0)))
    assert multi_agent.drive(path, fake_scripts=scripts, out=lambda *_: None) == "stopped"
    failed = (path / global_baseline.FILE).read_bytes()
    run = read_json(path / "run.json")
    run["config"]["orchestration"]["baseline_validation_version"] = 1
    write_json(path / "run.json", run)

    def no_backend(*args, **kwargs):
        pytest.fail("a changed validation policy must not introduce a new paid stage")

    monkeypatch.setattr(FakeBackend, "respond", no_backend)
    with pytest.raises(RuntimeError, match="saved baseline validation policy changed"):
        multi_agent.drive(path, fake_scripts=scripts, out=lambda *_: None)
    assert (path / global_baseline.FILE).read_bytes() == failed


def test_a_later_inexact_comparison_has_no_reported_validated_gap(completed, tmp_path):
    import copy
    import shutil

    original, cfg, _, _ = completed
    path = tmp_path / "report-check"
    shutil.copytree(original, path)
    archive = multi_agent.read_archive(path)
    candidate = archive["candidates"][0]
    comparator = archive["baseline_scores"][candidate["scope_id"]]
    comparator["parts"]["n_agree"] -= 1
    comparator["parts"]["fidelity"] = comparator["parts"]["n_agree"] / comparator["parts"]["n_in_region"]
    write_json(path / "multi_agent/archive.json", archive)
    audit_path = path / "multi_agent/audits" / (candidate["id"] + ".json")
    audit = read_json(audit_path)
    changed = copy.deepcopy(audit["fresh_baseline"])
    changed["parts"]["n_agree"] -= 1
    changed["parts"]["fidelity"] = changed["parts"]["n_agree"] / changed["parts"]["n_in_region"]
    audit["fresh_baseline"] = changed
    write_json(audit_path, audit)
    multi_agent.write_report(path)
    result = read_json(path / "report.json")["comparisons"]
    for match in (result["development"][candidate["id"]], result["final"][candidate["id"]]["fresh"]):
        assert match["valid"] is False and match["reward_gap"] is None
        assert match["raw_reward_difference"] is not None
    row = next(row for row in (path / "report.md").read_text().splitlines()
               if row.startswith(f"| {candidate['id']} | fresh |"))
    assert "unavailable (comparison invalid)" in row
