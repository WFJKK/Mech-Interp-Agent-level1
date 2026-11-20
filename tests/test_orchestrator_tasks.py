"""The orchestrator-tasks setup: the lead decides what each researcher works on, with any task, and a team of
any size runs through several cycles. Only the scripted fake backend is used."""

import pytest

from level1 import hygiene, loop, multi_agent, prompt, regions, run_config
from level1.networks import load_model
from level1.session import RESEARCH_TOOLS, Session, ToolError
from level1.util import read_json, write_json

ALL = {"kind": "all"}
TEAM_CONFIGS = ("config/experiments/rnn-a-team.yaml", "config/experiments/rnn-b-team.yaml",
                "config/experiments/rnn-b-team-smoke.yaml")


def session_cfg(tmp_path, **changes):
    cfg = run_config.load("config/experiments/lin-c-regional.yaml", {
        "runs_dir": str(tmp_path), "sandbox": "process", "backend": "fake",
        "session": {"python_calls": 10, "submissions": 6},
    })
    cfg.update(changes)
    return cfg


def test_the_lead_has_one_assignment_tool_and_only_the_task_is_required():
    assert set(RESEARCH_TOOLS) == {"assign_task"}
    schema = RESEARCH_TOOLS["assign_task"]["schema"]
    assert schema["required"] == ["task"] and set(schema["properties"]) == {"task", "scope", "entry_name"}


def test_a_task_without_a_scope_is_scored_on_every_input(tmp_path):
    cfg = session_cfg(tmp_path, research_task={"role": "lead", "max_assignments": 3, "cycle": 0})
    lead = Session.create(tmp_path / "lead", cfg, cfg["network"], tmp_path / "run")
    out = lead.call("assign_task", {"task": "Describe units 0 and 1 at every step from the inputs."})
    assert out["status"] == "assigned" and out["scope_id"] == regions.scope_id(ALL)
    assert out["remaining_assignments"] == 2
    lead.call("assign_task", {"task": "Make the saved program cheaper.", "scope": {"kind": "lengths", "lengths": [2]}})
    recorded = Session(lead.path).state["assignments"]
    assert [a["scope"] for a in recorded] == [ALL, {"kind": "lengths", "lengths": [2]}]
    assert recorded[0]["task"] == "Describe units 0 and 1 at every step from the inputs."
    with pytest.raises(ToolError, match="nonempty"):
        lead.call("assign_task", {"task": "   "})
    lead.call("assign_task", {"task": "Test the claim on the board."})
    with pytest.raises(ToolError, match="already assigned its 3 tasks"):
        lead.call("assign_task", {"task": "One more."})


def test_researchers_cannot_give_out_tasks(tmp_path):
    cfg = session_cfg(tmp_path, research_task={"role": "regional", "max_assignments": 0, "cycle": 1})
    worker = Session.create(tmp_path / "worker", cfg, cfg["network"], tmp_path / "run")
    assert "assign_task" not in worker.TOOLS
    with pytest.raises(ToolError, match="no tool"):
        worker.call("assign_task", {"task": "Change the plan."})
    with pytest.raises(ToolError, match="cannot assign tasks"):
        worker.assign_task("Change the plan.")


def test_assignments_default_to_every_input_and_follow_the_lead_first():
    cfg = {"orchestration": {"workers": 3, "initial_tasks": [{"task": "Fallback."}],
                             "initial_regions": [{"scope": {"kind": "lengths", "lengths": [1]}, "task": "Old fallback."}]}}
    given = {"assignments": [{"task": "a"}, {"task": "b", "scope": {"kind": "lengths", "lengths": [2]}},
                             {"task": "c", "scope": None}, {"task": "d"}]}
    out = multi_agent._assignments(cfg, given)
    assert [a["task"] for a in out] == ["a", "b", "c"]  # no more than the team's size
    assert [a["scope"] for a in out] == [ALL, {"kind": "lengths", "lengths": [2]}, ALL]
    assert multi_agent._assignments(cfg, {}) == [{"scope": ALL, "task": "Fallback."}]
    del cfg["orchestration"]["initial_tasks"]
    assert multi_agent._assignments(cfg, {}) == [{"scope": {"kind": "lengths", "lengths": [1]}, "task": "Old fallback."}]
    with pytest.raises(RuntimeError, match="needs a task"):
        multi_agent._assignments(cfg, {"assignments": [{"scope": ALL}]})


@pytest.mark.parametrize("path", TEAM_CONFIGS)
def test_team_configs_load_and_every_role_gets_a_usable_allowance(path, tmp_path):
    cfg = run_config.load(path, {"runs_dir": str(tmp_path), "backend": "fake", "sandbox": "process"})
    workers = cfg["orchestration"]["workers"]
    assert cfg["orchestration"]["initial_tasks"][0]["scope"] == ALL
    assert cfg["verifier"]["workload_size"] == cfg["verifier"]["sample_size"] == 16384
    total = 0.0
    for role, cycle, count in (("baseline", 0, 1), ("summary", 0, 1), ("lead", 0, 1)):
        total += multi_agent._allocation(cfg, role, cycle, workers)[2] * count
    for cycle in range(1, cfg["rounds_max"] + 1):
        calls, submissions, cap, _ = multi_agent._allocation(cfg, "regional", cycle, workers)
        assert calls >= 6 and submissions >= 1 and cap > 0
        lead = multi_agent._allocation(cfg, "lead", cycle, workers)
        assert lead[0] >= 6 and lead[1] >= 1
        total += workers * cap + lead[2]
    assert total <= cfg["budget"]["run_usd"] + 1e-9


def test_a_team_of_any_size_is_accepted_but_not_an_empty_one(tmp_path):
    base = {"runs_dir": str(tmp_path), "backend": "fake", "sandbox": "process"}
    cfg = run_config.load("config/experiments/rnn-b-team.yaml", {**base, "orchestration": {"workers": 5},
                                                                 "session": {"submissions": 12}})
    assert cfg["orchestration"]["workers"] == 5
    for workers in (0, -1, True, 2.0):
        with pytest.raises(run_config.ConfigError, match="workers"):
            run_config.load("config/experiments/rnn-b-team.yaml", {**base, "orchestration": {"workers": workers}})
    with pytest.raises(run_config.ConfigError, match="initial task"):
        run_config.load("config/experiments/rnn-b-team.yaml", {**base, "orchestration": {"initial_tasks": [{"scope": ALL}]}})
    with pytest.raises(run_config.ConfigError, match="initial task"):
        run_config.load("config/experiments/rnn-b-team.yaml",
                        {**base, "orchestration": {"initial_tasks": [{"task": "x", "region": ALL}]}})


@pytest.mark.parametrize("path", TEAM_CONFIGS)
def test_team_prompts_say_nothing_of_the_task(path, tmp_path):
    cfg = run_config.load(path, {"runs_dir": str(tmp_path), "backend": "fake", "sandbox": "process"})
    model = load_model(cfg["model"])
    forbidden = loop.forbidden_strings(cfg, model)
    texts = loop.agent_surface(cfg, model)
    assert "research tool assign_task" in texts and "configured initial tasks" in texts
    for role, count in (("lead", cfg["orchestration"]["workers"]), ("regional", 0)):
        role_cfg = {**cfg, "research_task": {"role": role, "max_assignments": count, "cycle": 0}}
        texts[f"{role} prompt"] = prompt.build_multi(role_cfg, model, 0, role, "Describe units 0 and 1.", "", [])
    hygiene.check(texts, forbidden)
    assert "**researcher**" in texts["regional prompt"] and "Describe units 0 and 1." in texts["regional prompt"]
    assert "regional investigator" not in texts["regional prompt"]
    assert "assign_task(task, scope=None, entry_name=None)" in texts["lead prompt"]
    assert f"up to {cfg['orchestration']['workers']} tasks" in texts["lead prompt"]


# ---------------------------------------------------------------- a team of three over two cycles

def team_cfg(tmp_path, **overrides):
    base = {"run_id": "team-test", "runs_dir": str(tmp_path), "backend": "fake", "sandbox": "process",
            "rounds_max": 2, "patience": 3, "session": {"python_calls": 40, "submissions": 9},
            "verifier": {"check": "sample", "sample_size": 128, "reference_samples": 1},
            "orchestration": {"enabled": True, "parallel": True, "development_seed": 73, "final_summary": False,
                              "baseline_agent": False, "workers": 3, "regional_submissions": 2}}
    return run_config.load("config/pilot/lin-a-1to4.yaml", run_config.merge(base, overrides))


def team_scripts(tmp_path):
    where = tmp_path / "fake-team"
    where.mkdir()
    worker = {"calls": [
        {"tool": "run_python", "args": {"code": "g=graph()\nsave_entry('mine', g.entry(predict=model.reference(g,lv1)))"}},
        {"tool": "submit_entry", "args": {"name": "mine", "claimed_fidelity": 1.0}},
    ], "response": "## What the network does\nA researcher's notes."}
    planning = {"calls": [
        {"tool": "assign_task", "args": {"task": "Describe units 0 to 3 of the hidden layer at every input."}},
        {"tool": "assign_task", "args": {"task": "Describe units 4 to 7 of the hidden layer at every input."}},
        {"tool": "assign_task", "args": {"task": "Find the inputs on which a cheap program fails.",
                                         "scope": {"kind": "lengths", "lengths": [2]}}},
    ], "response": "## What the network does\nThree tasks are given out."}
    lead_1 = {"calls": [
        {"tool": "run_python", "args": {"code": "g=graph()\nsave_entry('whole', g.entry(predict=model.reference(g,lv1)))"}},
        {"tool": "submit_entry", "args": {"name": "whole", "claimed_fidelity": 1.0}},
        {"tool": "assign_task", "args": {"task": "Make the program 'whole' cheaper without changing any output."}},
        {"tool": "assign_task", "args": {"task": "Try to break the claim made for units 0 to 3."}},
    ], "response": "## What the network does\nTwo tasks are given out."}
    lead_2 = {"calls": [
        {"tool": "submit_entry", "args": {"name": "whole", "claimed_fidelity": 1.0}},
    ], "response": "## What the network does\nThe combined program is submitted."}
    order = [planning, worker, worker, worker, lead_1, worker, worker, lead_2]
    for index, item in enumerate(order, 1):
        write_json(where / f"{index:02d}.json", item)
    return where


@pytest.fixture(scope="module")
def team_run(tmp_path_factory):
    where = tmp_path_factory.mktemp("team")
    cfg = team_cfg(where)
    path = run_config.run_path(cfg)
    path.mkdir(parents=True)
    write_json(path / "run.json", {"run_id": cfg["run_id"], "config": cfg, "model_ops": 671, "sandbox": {"mode": "process"}})
    assert multi_agent.drive(path, fake_scripts=team_scripts(where), out=lambda *_: None) == "finished"
    return path, cfg


def test_the_lead_decides_each_cycle_what_the_researchers_do(team_run):
    path, cfg = team_run
    state = read_json(path / "multi_agent/state.json")
    roles = [(p["role"], p["cycle"]) for p in sorted(state["sessions"].values(), key=lambda p: p["index"])]
    assert roles == [("lead", 0), ("regional", 1), ("regional", 1), ("regional", 1), ("lead", 1),
                     ("regional", 2), ("regional", 2), ("lead", 2)]
    assert state["stop_reason"] == "the configured research cycles are complete"
    workers = [p for p in state["sessions"].values() if p["role"] == "regional"]
    assert [p["task"] for p in workers] == [
        "Describe units 0 to 3 of the hidden layer at every input.",
        "Describe units 4 to 7 of the hidden layer at every input.",
        "Find the inputs on which a cheap program fails.",
        "Make the program 'whole' cheaper without changing any output.",
        "Try to break the claim made for units 0 to 3."]
    assert [p["scope"] for p in workers] == [ALL, ALL, {"kind": "lengths", "lengths": [2]}, ALL, ALL]
    for plan in workers:
        text = (path / plan["directory"] / "prompt.md").read_text()
        assert "**researcher**" in text and plan["task"] in text


def test_fewer_tasks_give_each_researcher_more_of_the_cycle(team_run):
    path, _ = team_run
    state = read_json(path / "multi_agent/state.json")
    calls = {}
    for plan in state["sessions"].values():
        if plan["role"] == "regional":
            session = read_json(path / plan["directory"] / "session/session.json")
            calls.setdefault(plan["cycle"], set()).add(session["budgets"]["run_python"])
    assert len(calls[1]) == len(calls[2]) == 1
    assert calls[2].pop() > calls[1].pop()


def test_the_last_lead_combines_and_gives_out_nothing(team_run):
    path, _ = team_run
    state = read_json(path / "multi_agent/state.json")
    last = max(state["sessions"].values(), key=lambda p: p["index"])
    session = read_json(path / last["directory"] / "session/session.json")
    assert last["role"] == "lead" and session["research_task"]["max_assignments"] == 0
    assert "No further tasks can be given out in this session." in (path / last["directory"] / "prompt.md").read_text()
    archive = multi_agent.read_archive(path)
    assert archive["global"]["best"] is not None
