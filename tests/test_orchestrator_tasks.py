"""The orchestrator-tasks setup: the lead decides what each researcher works on, with any task, and a team of
any size runs through several cycles. Only the scripted fake backend is used."""

import pytest

from level1 import hygiene, lang, loop, multi_agent, prompt, regions, run_config
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
    assert set(RESEARCH_TOOLS) == {"assign_task", "finish_research"}
    schema = RESEARCH_TOOLS["assign_task"]["schema"]
    assert schema["required"] == ["task"] and set(schema["properties"]) == {"task", "scope", "entry_name"}
    assert RESEARCH_TOOLS["finish_research"]["schema"]["required"] == ["reason"]


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


# ---------------------------------------------------------------- the lead can finish the research

def constant_entry():
    g = lang.Graph({}, 4)
    return g.entry(predict=g.const(0))


def test_only_a_lead_after_a_cycle_can_finish_and_only_with_a_scored_whole_program(tmp_path):
    def lead(name, cycle, assignments=2):
        cfg = session_cfg(tmp_path, research_task={"role": "lead", "max_assignments": assignments, "cycle": cycle})
        return Session.create(tmp_path / name, cfg, cfg["network"], tmp_path / f"run-{name}", {"constant": constant_entry()})

    planning = lead("planning", 0)
    assert "assign_task" in planning.TOOLS and "finish_research" not in planning.TOOLS
    with pytest.raises(ToolError, match="cannot finish"):
        planning.finish_research("Nothing to do.")
    assert "finish_research" not in lead("last", 2, assignments=0).TOOLS

    integrating = lead("integrating", 1)
    assert "finish_research" in integrating.TOOLS
    with pytest.raises(ToolError, match="no program for the whole input distribution"):
        integrating.call("finish_research", {"reason": "Enough evidence."})
    with pytest.raises(ToolError, match="nonempty"):
        integrating.call("finish_research", {"reason": "  "})
    assert integrating.call("submit_entry", {"name": "constant", "claimed_fidelity": 0.1})["status"] == "scored"
    out = integrating.call("finish_research", {"reason": "Enough evidence."})
    assert out == {"status": "finished", "reason": "Enough evidence."}
    assert Session(integrating.path).state["finish"]["reason"] == "Enough evidence."
    with pytest.raises(ToolError, match="finished the research"):
        integrating.call("assign_task", {"task": "One more idea."})
    with pytest.raises(ToolError, match="already finished"):
        integrating.call("finish_research", {"reason": "Again."})

    assigning = lead("assigning", 1)
    assigning.call("submit_entry", {"name": "constant", "claimed_fidelity": 0.1})
    assigning.call("assign_task", {"task": "Check the remaining inputs."})
    with pytest.raises(ToolError, match="already given out tasks"):
        assigning.call("finish_research", {"reason": "Changed my mind."})


def test_an_earlier_whole_program_in_the_archive_is_enough_to_finish(tmp_path):
    cfg = session_cfg(tmp_path, research_task={"role": "lead", "max_assignments": 2, "cycle": 1})
    run = tmp_path / "run"
    lead = Session.create(tmp_path / "lead", cfg, cfg["network"], run)
    with pytest.raises(ToolError, match="no program"):
        lead.finish_research("Enough.")
    archive = multi_agent._empty_archive()
    archive["global"]["best"] = "a0003s1"
    (run / "multi_agent").mkdir(parents=True)
    write_json(run / "multi_agent" / "archive.json", archive)
    assert lead.finish_research("Enough.")["status"] == "finished"


def test_the_finish_note_is_shown_only_to_leads_after_a_cycle(tmp_path):
    cfg = run_config.load("config/experiments/rnn-b-team.yaml", {"runs_dir": str(tmp_path), "backend": "fake", "sandbox": "process"})
    model = load_model(cfg["model"])
    texts = {}
    for name, cycle, count in (("planning", 0, 4), ("after a cycle", 1, 4), ("last", 3, 0)):
        role_cfg = {**cfg, "research_task": {"role": "lead", "max_assignments": count, "cycle": cycle}}
        texts[name] = prompt.build_multi(role_cfg, model, cycle, "lead", "Plan.", "", [])
    assert "finish_research(reason)" in texts["after a cycle"]
    assert "finish_research" not in texts["planning"] and "finish_research" not in texts["last"]
    hygiene.check({**texts, **loop.agent_surface(cfg, model)}, loop.forbidden_strings(cfg, model))


def finishing_scripts(tmp_path):
    where = tmp_path / "fake-finish"
    where.mkdir(parents=True)
    worker = {"calls": [
        {"tool": "run_python", "args": {"code": "g=graph()\nsave_entry('mine', g.entry(predict=model.reference(g,lv1)))"}},
        {"tool": "submit_entry", "args": {"name": "mine", "claimed_fidelity": 1.0}},
    ], "response": "## What the network does\nA researcher's notes."}
    planning = {"calls": [
        {"tool": "assign_task", "args": {"task": "Describe units 0 to 3 of the hidden layer at every input."}},
        {"tool": "assign_task", "args": {"task": "Describe units 4 to 7 of the hidden layer at every input."}},
    ], "response": "## What the network does\nTwo tasks are given out."}
    lead = {"calls": [
        {"tool": "run_python", "args": {"code": "g=graph()\nsave_entry('whole', g.entry(predict=model.reference(g,lv1)))"}},
        {"tool": "submit_entry", "args": {"name": "whole", "claimed_fidelity": 1.0}},
        {"tool": "finish_research", "args": {"reason": "The reward has stopped rising; more cycles would not pay."}},
    ], "response": "## What the network does\nThe research is finished."}
    for index, item in enumerate((planning, worker, worker, lead), 1):
        write_json(where / f"{index:02d}.json", item)
    return where


def test_a_lead_that_finishes_sends_the_run_to_its_ordinary_end(tmp_path):
    cfg = team_cfg(tmp_path, rounds_max=3, orchestration={"workers": 2, "initial_tasks": [{"task": "Fallback that must not run."}]},
                   session={"python_calls": 40, "submissions": 6})
    path = run_config.run_path(cfg)
    path.mkdir(parents=True)
    write_json(path / "run.json", {"run_id": cfg["run_id"], "config": cfg, "model_ops": 671, "sandbox": {"mode": "process"}})
    lines = []
    assert multi_agent.drive(path, fake_scripts=finishing_scripts(tmp_path), out=lines.append) == "finished"
    state = read_json(path / "multi_agent/state.json")
    assert state["stop_reason"] == "the lead finished the research: The reward has stopped rising; more cycles would not pay."
    assert [p["role"] for p in sorted(state["sessions"].values(), key=lambda p: p["index"])] == ["lead", "regional", "regional", "lead"]
    assert len(state["cycles"]) == 1 and state["cycles"][0]["status"] == "completed"
    assert (path / "multi_agent/lock.json").exists()
    assert "Stop reason: the lead finished the research" in (path / "report.md").read_text()
    # a second drive, as after an interruption, ends the same way without starting another cycle
    assert multi_agent.drive(path, fake_scripts=finishing_scripts(tmp_path / "again"), out=lines.append) == "finished"
    assert len(read_json(path / "multi_agent/state.json")["sessions"]) == 4


# ---------------------------------------------------------------- the operator can pause between cycles

def paused_run(tmp_path):
    cfg = team_cfg(tmp_path)
    path = run_config.run_path(cfg)
    path.mkdir(parents=True)
    write_json(path / "run.json", {"run_id": cfg["run_id"], "config": cfg, "model_ops": 671, "sandbox": {"mode": "process"}})
    scripts, lines = team_scripts(tmp_path), []
    assert multi_agent.drive(path, fake_scripts=scripts, out=lines.append, pause_after=1) == "stopped"
    return path, scripts, lines


def test_a_paused_run_stops_after_its_cycle_with_nothing_locked_and_goes_on_with_the_leads_tasks(tmp_path):
    path, scripts, lines = paused_run(tmp_path)
    state = read_json(path / "multi_agent/state.json")
    assert state["stop_reason"] == "paused by the operator after cycle 1"
    assert [c["status"] for c in state["cycles"]] == ["completed"] and len(state["sessions"]) == 5
    assert not (path / "multi_agent/lock.json").exists() and not (path / "lock.json").exists()
    assert any(f"resume --run {path}" in line for line in lines)
    assert '"paused_after_cycle": 1' in (path / "log.jsonl").read_text()
    # going on: the next cycle runs the tasks the lead of cycle 1 gave out, and the run ends as an unpaused one
    assert multi_agent.drive(path, fake_scripts=scripts, out=lines.append) == "finished"
    state = read_json(path / "multi_agent/state.json")
    roles = [(p["role"], p["cycle"]) for p in sorted(state["sessions"].values(), key=lambda p: p["index"])]
    assert roles == [("lead", 0), ("regional", 1), ("regional", 1), ("regional", 1), ("lead", 1),
                     ("regional", 2), ("regional", 2), ("lead", 2)]
    assert [p["task"] for p in state["sessions"].values() if p["role"] == "regional" and p["cycle"] == 2] == [
        "Make the program 'whole' cheaper without changing any output.", "Try to break the claim made for units 0 to 3."]
    assert state["stop_reason"] == "the configured research cycles are complete"
    assert (path / "multi_agent/lock.json").exists()


def test_a_paused_run_can_end_after_its_cycle_with_the_usual_lock(tmp_path):
    path, _, lines = paused_run(tmp_path)
    assert loop.finish(path, out=lines.append)["locked"]
    state = read_json(path / "multi_agent/state.json")
    assert len(state["cycles"]) == 1 and len(state["sessions"]) == 5
    report = (path / "report.md").read_text()
    assert "Completed cycles: 1." in report and "Stop reason: paused by the operator after cycle 1" in report


def test_only_a_team_run_can_pause_and_only_after_a_cycle_number(tmp_path):
    single = ["run", "--config", "config/pilot/lin-a-1to4.yaml", "--runs-dir", str(tmp_path / "runs"),
              "--fake", str(tmp_path / "scripts")]
    assert loop.main(single + ["--pause-after", "1"]) == 1
    team = ["run", "--config", "config/experiments/rnn-b-team.yaml", "--runs-dir", str(tmp_path / "runs"),
            "--fake", str(tmp_path / "scripts"), "--sandbox", "process"]
    assert loop.main(team + ["--pause-after", "-1"]) == 1
    assert loop.main(team + ["--pause-after", "1", "--with-baseline"]) == 1
    assert not (tmp_path / "runs").exists()  # refused before any run folder is made


# ---------------------------------------------------------------- scoring during the cycles, as for a single agent

def test_a_fresh_sample_per_submission_is_only_for_runs_without_a_baseline(tmp_path):
    base = {"runs_dir": str(tmp_path), "backend": "fake", "sandbox": "process"}
    cfg = run_config.load("config/experiments/rnn-b-team.yaml", base)
    assert cfg["orchestration"]["development_seed"] is None and cfg["verifier"]["sample_seed"] is None
    assert multi_agent._scope_cfg(cfg, ALL)["verifier"]["sample_seed"] is None
    with pytest.raises(run_config.ConfigError, match="development_seed: null"):
        run_config.load("config/experiments/rnn-b-team.yaml", {**base, "orchestration": {"mode": "both", "baseline_agent": True}})
    fixed = run_config.load("config/experiments/rnn-b-team.yaml", {**base, "orchestration": {"development_seed": 5}})
    assert fixed["verifier"]["sample_seed"] == 5 and multi_agent._scope_cfg(fixed, ALL)["verifier"]["sample_seed"] == 5


def test_with_fresh_samples_each_submission_is_chosen_by_its_own_score(tmp_path):
    cfg = team_cfg(tmp_path, rounds_max=3, orchestration={"workers": 2, "development_seed": None},
                   session={"python_calls": 40, "submissions": 6})
    assert cfg["verifier"]["sample_seed"] is None
    path = run_config.run_path(cfg)
    path.mkdir(parents=True)
    write_json(path / "run.json", {"run_id": cfg["run_id"], "config": cfg, "model_ops": 671, "sandbox": {"mode": "process"}})
    assert multi_agent.drive(path, fake_scripts=finishing_scripts(tmp_path), out=lambda *_: None) == "finished"
    archive = multi_agent.read_archive(path)
    seeds = [c["submitted"]["inputs"]["seed"] for c in archive["candidates"]]
    assert len(seeds) == 3 and len(set(seeds)) == 3  # every submission drew its own inputs
    for candidate in archive["candidates"]:
        assert candidate["selection"] == candidate["submitted"]  # no second scoring on a common sample
    best = next(c for c in archive["candidates"] if c["id"] == archive["global"]["best"])
    assert best["selection"]["reward"] == max(c["selection"]["reward"] for c in archive["candidates"] if c["scope_id"] == "all")
    assert (path / "multi_agent/lock.json").exists()
