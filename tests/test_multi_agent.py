"""The research coordinator, using only the existing scripted fake backend."""

import pytest

from level1 import budget, lang, multi_agent, rewards, run_config
from level1.networks import load_model
from level1.session import Session
from level1.util import read_json, write_json


def config(tmp_path, **overrides):
    base = {"run_id": "research-test", "runs_dir": str(tmp_path), "backend": "fake", "sandbox": "process",
            "rounds_max": 1, "patience": 2, "session": {"python_calls": 20, "submissions": 4},
            "verifier": {"check": "sample", "sample_size": 128, "reference_samples": 1},
            "orchestration": {"enabled": True, "parallel": True, "development_seed": 73, "final_summary": False,
                              "baseline_agent": False}}
    return run_config.load("config/pilot/lin-a-1to4.yaml", run_config.merge(base, overrides))


def prepare(tmp_path, **overrides):
    cfg = config(tmp_path, **overrides)
    path = run_config.run_path(cfg)
    path.mkdir(parents=True)
    write_json(path / "run.json", {"run_id": cfg["run_id"], "config": cfg,
                                   "model_ops": 671, "sandbox": {"mode": "process"}})
    return path, cfg


def scripts(tmp_path, stopped=False):
    where = tmp_path / "fake-research"
    where.mkdir()
    planning = {"calls": [
        {"tool": "assign_task", "args": {"scope": {"kind": "lengths", "lengths": [2]},
                                             "task": "Find a compact predictor on this input class."}},
        {"tool": "assign_task", "args": {"scope": {"kind": "lengths", "lengths": [1]},
                                             "task": "Investigate a different input class."}},
    ], "response": "## What the network does\nTwo investigations are ready."}
    pair_worker = {"calls": [
        {"tool": "run_python", "args": {"code": "g=graph()\nsave_entry('local', g.entry(predict=lv1.maximum(g.token(0),g.token(1))))"}},
        {"tool": "submit_entry", "args": {"name": "local", "claimed_fidelity": 0.99, "expected_fidelity_other": 0.99}},
    ], "response": "## What the network does\nWorker alpha private completion."}
    if stopped:
        pair_worker["stop_run"] = "scripted worker interruption"
    one_worker = {"calls": [
        {"tool": "run_python", "args": {"code": "g=graph()\nsave_entry('local',g.entry(predict=g.token(0)))"}},
        {"tool": "submit_entry", "args": {"name": "local", "claimed_fidelity": 0.98, "expected_fidelity_other": 0.98}},
    ], "response": "## What the network does\nWorker beta private completion."}
    integration = {"calls": [
        {"tool": "run_python", "args": {"code": """g=graph()
save_entry('full',g.entry(predict=model.reference(g,lv1)))
g,local,rule=lv1.Graph.load(entries['a0002s1'],weights,info['sizes']['n_ctx'])
inside=(g.token(1)<64)&(g.token(2)==64)
combined=lv1.cond(inside,local,model.reference(g,lv1))
save_entry('combined',g.entry(predict=combined))
"""}},
        {"tool": "submit_entry", "args": {"name": "full", "claimed_fidelity": 1.0}},
        {"tool": "submit_entry", "args": {"name": "combined", "claimed_fidelity": 0.99}},
    ], "response": "## What the network does\nThe complete composition has been measured."}
    for index, item in enumerate((planning, pair_worker, one_worker, integration), 1):
        write_json(where / f"{index:02d}.json", item)
    return where


def candidate(cid, role, scope_id, reward, fidelity=1.0, cost=0.1, coverage=1.0, digest=None):
    parts = {"coverage": coverage, "fidelity": fidelity, "fidelity_low": fidelity,
             "cost": cost, "r": cost, "r_rule": 0.0, "b": 0.0}
    return {"id": cid, "session_id": cid, "role": role, "scope_id": scope_id,
            "scope": {"kind": "all"}, "entry_sha256": digest or cid,
            "selection": {"status": "scored", "reward": reward, "parts": parts},
            "submitted": {"submission": 1}}


def test_only_full_domain_scores_choose_global_winner():
    archive = multi_agent._empty_archive()
    archive["candidates"] = [candidate("global", "lead", "all", 0.2, cost=0.8),
                             candidate("local", "regional", "scope-x", 0.999, cost=0.001),
                             candidate("local_all", "regional", "all", 0.99, cost=0.01)]
    multi_agent.update_frontiers(archive)
    assert archive["global"]["best"] == "local_all"
    assert archive["global"]["frontier"] == ["local_all"]
    assert archive["regions"]["scope-x"]["frontier"] == ["local"]
    assert archive["regions"]["all"]["frontier"] == ["local_all"]


def test_regional_frontiers_are_separate_and_reset_progress():
    archive = multi_agent._empty_archive()
    archive["candidates"] = [candidate("global", "lead", "all", 0.8, cost=0.2),
                             candidate("old", "regional", "scope-x", 0.1, cost=0.9)]
    multi_agent.update_frontiers(archive)
    before = multi_agent._signature(archive)
    archive["candidates"].extend([candidate("new", "regional", "scope-x", 0.7, cost=0.3),
                                  candidate("other", "regional", "scope-y", 0.05, cost=0.95)])
    multi_agent.update_frontiers(archive)
    changed = multi_agent.progress(before, multi_agent._signature(archive))
    assert changed == {"global_improved": False, "regional_improved": True, "improved": True}
    assert archive["regions"]["scope-x"]["frontier"] == ["new"]
    assert archive["regions"]["scope-y"]["frontier"] == ["other"]
    assert len(archive["candidates"]) == 4  # dominated submissions remain on record
    before = multi_agent._signature(archive)
    archive["candidates"].append(candidate("repeat", "regional", "scope-x", 0.7, cost=0.3, digest="new"))
    multi_agent.update_frontiers(archive)
    assert not multi_agent.progress(before, multi_agent._signature(archive))["improved"]


def test_role_budgets_do_not_multiply_original_run_allowances(tmp_path):
    cfg = config(tmp_path, rounds_max=5, session={"python_calls": 40, "submissions": 4})
    calls, submissions, api_cap, planning_cap = multi_agent._allocation(cfg, "lead", 0, 2)
    assert calls == 20 and submissions == 0
    total_calls, total_submissions, total_api = calls, submissions, api_cap
    for cycle in range(1, cfg["rounds_max"] + 1):
        allocations = [multi_agent._allocation(cfg, "regional", cycle, 2) for _ in range(2)]
        allocations.append(multi_agent._allocation(cfg, "lead", cycle, 2))
        assert [a[:2] for a in allocations] == [(12, 1), (12, 1), (12, 2)]
        assert sum(a[2] for a in allocations) <= allocations[0][3] + 1e-12
        total_calls += sum(a[0] for a in allocations)
        total_submissions += sum(a[1] for a in allocations)
        total_api += sum(a[2] for a in allocations)
    assert total_calls == 200 and total_submissions == 20
    assert total_api <= cfg["budget"]["run_usd"] + 1e-12
    for key, integer in (("max_turns", True), ("seconds", False)):
        total = multi_agent._quota(cfg, cfg["session"][key], "lead", 0, 2, integer)
        for cycle in range(1, cfg["rounds_max"] + 1):
            total += 2 * multi_agent._quota(cfg, cfg["session"][key], "regional", cycle, 2, integer)
            total += multi_agent._quota(cfg, cfg["session"][key], "lead", cycle, 2, integer)
        assert total <= cfg["session"][key] * cfg["rounds_max"]


@pytest.fixture(scope="module")
def completed(tmp_path_factory):
    where = tmp_path_factory.mktemp("regional-research")
    path, cfg = prepare(where)
    fake = scripts(where)
    lines = []
    assert multi_agent.drive(path, fake_scripts=fake, out=lines.append) == "finished"
    return path, cfg, fake, lines


def test_real_fake_sessions_plan_research_and_measure_composition(completed):
    path, cfg, _, _ = completed
    state = read_json(path / "multi_agent/state.json")
    archive = multi_agent.read_archive(path)
    assert [p["index"] for p in state["sessions"].values()] == [0, 1, 2, 3]
    assert [p["role"] for p in state["sessions"].values()] == ["lead", "regional", "regional", "lead"]
    assert len(archive["candidates"]) == 4
    assert len(archive["regions"]) == 2
    assert archive["global"]["best"] == "a0004s2"
    winner = next(c for c in archive["candidates"] if c["id"] == archive["global"]["best"])
    assert winner["selection"]["scope"]["id"] == "all"
    assert winner["selection"]["parts"]["n_inputs"] == 128
    assert winner["selection"]["parts"]["coverage"] == 1.0
    assert winner["selection"]["parts"]["fidelity"] == 1.0
    assert winner["selection"]["parts"]["cost"] < 1
    regional = [c for c in archive["candidates"] if c["role"] == "regional"]
    assert max(c["selection"]["reward"] for c in regional) > winner["selection"]["reward"]
    for plan in state["sessions"].values():
        session = read_json(path / plan["directory"] / "session/session.json")
        assert session["run_path"] == str(path.resolve())
        assert session["research_task"]["session_id"] == plan["id"]
        assert session["verifier"]["sample_seed"] == cfg["orchestration"]["development_seed"]


def test_workers_share_frozen_context_and_lead_gets_both_findings(completed):
    path, _, _, _ = completed
    state = read_json(path / "multi_agent/state.json")
    first, second = (state["sessions"][sid] for sid in ("a0002", "a0003"))
    assert first["snapshot"] == second["snapshot"] and first["history"] == second["history"]
    for worker in (first, second):
        text = (path / worker["directory"] / "prompt.md").read_text()
        assert "Worker alpha private completion" not in text
        assert "Worker beta private completion" not in text
    lead = state["sessions"]["a0004"]
    text = (path / lead["directory"] / "prompt.md").read_text()
    assert "Worker alpha private completion" in text and "Worker beta private completion" in text
    assert "a0002s1" in text and "a0003s1" in text
    assert "held_out" not in text and "lin-a-1to4-s1" not in text


def test_final_scope_locks_and_audits_are_immutable(completed, monkeypatch):
    path, cfg, fake, _ = completed
    lock = read_json(path / "multi_agent/lock.json")
    assert lock["global"] == "a0004s2" and len(lock["regional"]) == 2
    assert len(lock["candidates"]) == 3
    snapshots = {p: p.read_bytes() for p in (path / "multi_agent/audits").glob("*.json")}
    for target in snapshots:
        audit = read_json(target)
        assert audit["fresh"]["network"] == cfg["network"]
        assert audit["held_out"]["network"] == cfg["seed_check"]["network"]
        assert audit["fresh"]["scope"]["id"] == audit["held_out"]["scope"]["id"]
        assert audit["fresh"]["inputs"]["seed"] != cfg["orchestration"]["development_seed"]
    monkeypatch.setattr(multi_agent, "make", lambda *a, **k: pytest.fail("a locked run must not call an agent again"))
    assert multi_agent.drive(path, fake_scripts=fake, out=lambda *_: None) == "finished"
    assert read_json(path / "multi_agent/lock.json") == lock
    assert all(target.read_bytes() == content for target, content in snapshots.items())
    assert "Outer probability (exact):" in (path / "report.md").read_text()


def test_final_audits_preserve_claimed_reward_semantics(tmp_path):
    path, cfg = prepare(tmp_path, reward={"name": "r_alpha_claimed"})
    model = load_model(cfg["model"])
    entry = model.reference_entry(model.load_weights(cfg["network"]))
    write_json(path / "entry.json", entry)
    frozen = {"id": "claimed", "entry_path": "entry.json", "entry_sha256": lang.entry_hash(entry),
              "scope": {"kind": "all"}, "scope_id": "all",
              "selection": {"claimed_fidelity": 0.6}, "expected_fidelity_other": 0.8}
    audit = multi_agent._audit_candidate(path, cfg, model, frozen)
    for purpose, claim in (("fresh", 0.6), ("held_out", 0.8)):
        receipt = audit[purpose]
        assert receipt["status"] == "scored" and receipt["claimed_fidelity"] == claim
        assert receipt["reward"] == pytest.approx(rewards.compute(receipt["parts"], cfg["reward"], claim)["reward"])
        assert receipt["reward"] != pytest.approx(rewards.compute(receipt["parts"], cfg["reward"])["reward"])


def test_rescoring_keeps_global_and_regional_comparisons_separate(completed):
    path, _, _, _ = completed
    archive_path = path / "multi_agent/archive.json"
    original = archive_path.read_bytes()
    lines = []
    rows = multi_agent.rescore(path, {"name": "r_alpha", "alpha": 1}, out=lines.append)
    assert len(rows) == 4
    assert all({"role", "scope_id", "recorded", "rescored"} <= row.keys() for row in rows)
    assert len([line for line in lines if line.startswith("Best global")]) == 1
    assert len([line for line in lines if line.startswith("Best regional scope")]) == 2
    assert "a0004s2" in next(line for line in lines if line.startswith("Best global"))
    assert archive_path.read_bytes() == original


def test_interrupted_worker_resumes_same_session_without_losing_other_worker(tmp_path):
    path, _ = prepare(tmp_path)
    fake = scripts(tmp_path, stopped=True)
    assert multi_agent.drive(path, fake_scripts=fake, out=lambda *_: None) == "stopped"
    state = read_json(path / "multi_agent/state.json")
    assert state["next_index"] == 4
    assert state["sessions"]["a0003"]["status"] == "completed"
    assert not (path / "multi_agent/lock.json").exists()
    other_record = path / state["sessions"]["a0003"]["directory"] / "record.json"
    original = other_record.read_bytes()
    item = read_json(fake / "02.json")
    item.pop("stop_run")
    write_json(fake / "02.json", item)
    assert multi_agent.drive(path, fake_scripts=fake, out=lambda *_: None) == "finished"
    assert read_json(path / "multi_agent/state.json")["next_index"] == 4
    assert other_record.read_bytes() == original
    assert len(multi_agent.read_archive(path)["candidates"]) == 4


def test_no_investigations_is_not_reported_as_complete(tmp_path):
    path, _ = prepare(tmp_path)
    fake = tmp_path / "no-tasks"
    fake.mkdir()
    write_json(fake / "01.json", {"calls": [], "response": "No assignments."})
    assert multi_agent.drive(path, fake_scripts=fake, out=lambda *_: None) == "stopped"
    assert "gave out no tasks" in read_json(path / "multi_agent/state.json")["stop_reason"]
    assert not (path / "multi_agent/lock.json").exists()


def test_only_one_controller_can_mutate_a_run(tmp_path):
    path, _ = prepare(tmp_path)
    with multi_agent._controller_lock(path):
        with pytest.raises(RuntimeError, match="another controller"):
            multi_agent.drive(path, out=lambda *_: None)


def test_comparison_entry_is_snapshotted_and_scored_in_each_scope(tmp_path):
    model = load_model("models/lin-a-1to4")
    comparison = model.reference_entry(model.load_weights("lin-a-1to4-s0"))
    source = tmp_path / "frozen.json"
    write_json(source, comparison)
    path, _ = prepare(tmp_path, orchestration={"baseline_entry": str(source)})
    fake = scripts(tmp_path)
    assert multi_agent.drive(path, fake_scripts=fake, out=lambda *_: None) == "finished"
    archive = multi_agent.read_archive(path)
    assert len(archive["baseline_scores"]) == 3 and "all" in archive["baseline_scores"]
    for key, receipt in archive["baseline_scores"].items():
        assert receipt["status"] == "scored" and receipt["scope"]["id"] == key
        assert receipt["parts"]["fidelity"] == 1.0
    assert read_json(path / "multi_agent/baseline_entry.json") == comparison
    state = read_json(path / "multi_agent/state.json")
    for plan in state["sessions"].values():
        assert (path / plan["directory"] / "session/workspace/entries/comparison_entry.json").exists()


def test_saved_cli_result_is_recovered_before_backend_or_budget_checks(tmp_path, monkeypatch):
    path, cfg = prepare(tmp_path, auth="api_key")
    model = load_model(cfg["model"])
    state = {"next_index": 0, "sessions": {}}
    plan = multi_agent._new_plan(state, "lead", 0, {"kind": "all"}, "Plan the investigations.", 2,
                                 multi_agent._empty_archive(), [])
    directory = path / plan["directory"]
    Session.create(directory / "session", multi_agent.role_config(cfg, plan), cfg["network"], path)
    (directory / "prompt.md").write_text("Saved prompt.")
    (directory / "response.md").write_text("## What the network does\nSaved completion.")
    write_json(directory / "meta.json", {"backend": "fake", "stop_reason": "end_turn"})
    budget.start(path, 1, 1, cfg["budget"]["run_usd"])
    monkeypatch.setattr(multi_agent, "make", lambda *a, **k: pytest.fail("a saved CLI result needs no new backend"))
    record = multi_agent._run_session(path, cfg, model, state, plan, None, lambda *_: None)
    assert "Saved completion" in record["notes"]
    assert record["meta"]["stop_reason"] == "end_turn"


def test_spent_interrupted_session_keeps_its_submissions_without_another_call(tmp_path, monkeypatch):
    path, cfg = prepare(tmp_path, auth="api_key")
    model = load_model(cfg["model"])
    state = {"next_index": 0, "sessions": {}}
    plan = multi_agent._new_plan(state, "regional", 1, {"kind": "lengths", "lengths": [2]},
                                 "Inspect this input class.", 2, multi_agent._empty_archive(), [])
    directory = path / plan["directory"]
    session = Session.create(directory / "session", multi_agent.role_config(cfg, plan), cfg["network"], path)
    session.run_python("g=graph()\nsave_entry('saved',g.entry(predict=g.token(0)))")
    session.submit_entry("saved", 0.5)
    (directory / "prompt.md").write_text("Saved prompt.")
    budget.start(path, 1, 1, cfg["budget"]["run_usd"])
    monkeypatch.setattr(multi_agent, "make", lambda *a, **k: pytest.fail("exhausted recovery must not make a new call"))
    record = multi_agent._run_session(path, cfg, model, state, plan, None, lambda *_: None)
    assert len(record["submissions"]) == 1
    assert record["meta"]["stop_reason"] == "budget" and record["meta"]["recovered"]


def test_experiment_budget_update_is_locked_and_keeps_planned_caps(tmp_path, monkeypatch):
    path, cfg = prepare(tmp_path, auth="api_key")
    monkeypatch.setattr(multi_agent, "_drive", lambda *a, **k: "stopped")
    with multi_agent._controller_lock(path):
        with pytest.raises(RuntimeError, match="another controller"):
            multi_agent.drive(path, budget_usd=75, out=lambda *_: None)
    assert budget.state(path, cfg["budget"])["run_usd"] == cfg["budget"]["run_usd"]
    assert multi_agent.drive(path, budget_usd=75, out=lambda *_: None) == "stopped"
    assert budget.state(path, cfg["budget"])["run_usd"] == 75
    assert read_json(path / "run.json")["config"]["budget"] == cfg["budget"]
