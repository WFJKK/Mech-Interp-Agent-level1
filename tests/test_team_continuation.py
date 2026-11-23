"""A team run that goes on from a finished team run at a higher price (level1/continuation.py, level1/multi_agent.py,
docs/CONTINUATION.md). Only the scripted fake backend is used."""

import hashlib
import math
from pathlib import Path

import pytest

from level1 import continuation, lang, loop, multi_agent, prompt, rewards, run_config
from level1.util import read_json, write_json

# A team of one researcher on rnn-a, kept small. The verifier samples, so the price comes with more inputs per scoring.
SMALL = {"backend": "fake", "sandbox": "process", "rounds_max": 1, "patience": 2,
         "verifier": {"sample_size": 1500, "workload_size": 1500, "reference_samples": 2},
         "session": {"python_calls": 8, "submissions": 4},
         "orchestration": {"mode": "trained", "workers": 1, "parallel": False, "final_summary": False,
                           "development_seed": None, "checkpoint_python_calls": None},
         "continuation": {"max_sample_size": 5000}}
NOTES = "\n\n## The best entry\n{best}\n\n## Next\nLook closer."
# The earlier run: the lead gives out one task; the researcher submits a program for the whole network that shifts
# the first number up by 0.05 before taking the place of the lower one, which is wrong on about one input in a
# hundred (tests/test_continuation.py): errors to price.
PLANNING = {"calls": [{"tool": "assign_task", "args": {"task": "Find a cheap program for the whole network."}}],
            "response": "## What the network does\nOne task is given out, for the whole network."}
RESEARCHER = {"calls": [
    {"tool": "run_python", "args": {"code": "g = graph()\nsave_entry('shifted', g.entry(interpretation="
                                            "(g.tokens() + g.const([0.05, 0.0])).argmin()))"}},
    {"tool": "submit_entry", "args": {"name": "shifted", "claimed_fidelity": 0.99, "expected_fidelity_other": 0.99}},
], "response": "## What the network does\nThe place of the lower number, the first shifted up a little; not always."
               + NOTES.format(best="shifted")}
LEAD = {"calls": [], "response": "## What the network does\nThe researcher's program stands as it is."}
# The run that goes on: its planning lead gives out one task, its researcher looks at what it was given and submits
# a program right where the first number is above 0, and its last lead combines nothing more.
PLANNING_AGAIN = {"calls": [{"tool": "assign_task", "args": {"task": "Find where the earlier program is wrong."}}],
                  "response": "## What the network does\nOne more task, on the earlier program's errors."}
RESEARCHER_AGAIN = {"calls": [
    {"tool": "run_python", "args": {"code": "print('carried', sorted(entries))\ng = graph()\n"
                                            "save_entry('half', g.entry(interpretation=g.tokens().argmin(), region=g.token(0) > 0.0))"}},
    {"tool": "submit_entry", "args": {"name": "half", "claimed_fidelity": 1.0, "expected_fidelity_other": 1.0}},
], "response": "## What the network does\nThe lowest number, where the first is positive." + NOTES.format(best="half")}
LEAD_AGAIN = {"calls": [], "response": "## What the network does\nThe researcher's program is the one to keep."}


def scripts(where: Path, *sessions) -> Path:
    where.mkdir(parents=True)
    for i, item in enumerate(sessions, 1):
        write_json(where / f"{i:02d}.json", item)
    return where


def config(tmp_path, run_id, **over):
    return run_config.load("config/experiments/rnn-a-team.yaml",
                           run_config.merge({**SMALL, "runs_dir": str(tmp_path), "run_id": run_id}, over))


def folder_hash(path: Path) -> str:
    h = hashlib.sha256()
    for f in sorted(p for p in path.rglob("*") if p.is_file() and p.name != "controller.lock"):
        h.update(str(f.relative_to(path)).encode())
        h.update(f.read_bytes())
    return h.hexdigest()


@pytest.fixture(scope="module")
def earlier(tmp_path_factory):
    where = tmp_path_factory.mktemp("team-continuation")
    # The earlier run's sessions are the first three scripts; the later run's sessions are numbered after them.
    plays = scripts(where / "scripts", PLANNING, RESEARCHER, LEAD, PLANNING_AGAIN, RESEARCHER_AGAIN, LEAD_AGAIN)
    path = loop.start(config(where, "team-early"), fake_scripts=plays, out=lambda *_: None)
    lock = read_json(path / "lock.json")
    assert lock["locked"] and lock["candidate_id"] == "a0002s1"
    assert lock["measured_again"]["parts"]["fidelity"] < 1.0  # there are errors left to price
    return where, plays, path


def test_the_price_and_sample_come_from_the_program_the_team_locked(earlier):
    where, _, path = earlier
    cfg = continuation.prepare(config(where, "team-later", continue_from="team-early"))
    parts = read_json(path / "lock.json")["measured_again"]["parts"]
    t = rewards.terms(parts, parts["fidelity"])
    alpha = max(10.0, math.ceil(2.0 * t["saved"] / t["errors"]))
    d = cfg["continuation"]["derived"]
    assert cfg["reward"]["alpha"] == d["alpha"] == alpha > 10
    assert cfg["verifier"]["sample_size"] == d["sample"] == max(1500, min(math.ceil(200 * alpha), 5000))
    assert d["team"] and d["runs"] == ["team-early"] and d["phases"] == 1 and d["first_round"] == 1
    assert d["next_index"] == 3  # the earlier run's three sessions
    assert d["prices"] == [{"run": "team-early", "phases": 1, "alpha": 10.0}]
    assert continuation.price_history(d) == "alpha = 10 (1 research phase)"
    assert d["locked"]["carried_as"] == "a0002s1" and d["locked"]["name"] == "shifted" and d["locked"]["measured_again"]
    assert d["locked"]["errors"] == t["errors"] and d["exact_charge"] == pytest.approx(
        alpha * (1.0 - lang.wilson_lower(d["sample"], d["sample"])))
    control = continuation.prepare(config(where, "team-x", continue_from="team-early", continuation={"memory": False}))
    assert control["continuation"]["derived"]["runs"] == [] and control["continuation"]["derived"]["next_index"] == 0
    assert control["reward"]["alpha"] == alpha


@pytest.fixture(scope="module")
def later(earlier):
    where, plays, path = earlier
    before = folder_hash(path)
    lines = []
    later_path = loop.start(config(where, "team-later", continue_from="team-early"), fake_scripts=plays, out=lines.append)
    assert folder_hash(path) == before  # the earlier run is read, never written
    return where, path, later_path, lines


def test_the_sessions_are_numbered_after_the_earlier_ones_and_see_its_archive_and_notes(later):
    _, path, later_path, lines = later
    cfg = read_json(later_path / "run.json")["config"]
    d = cfg["continuation"]["derived"]
    state = read_json(later_path / "multi_agent/state.json")
    assert sorted(state["sessions"]) == ["a0004", "a0005", "a0006"] and state["continuation"]["runs"] == ["team-early"]
    assert [c["cycle"] for c in state["cycles"]] == [1] and state["stop_reason"] == "the configured research cycles are complete"
    planning = (later_path / state["sessions"]["a0004"]["directory"] / "prompt.md").read_text()
    assert state["sessions"]["a0004"]["task"].startswith("Decide what each researcher works on in the next research phase")
    flat = " ".join(planning.split())
    assert prompt.TEAM_CONTINUATION_HEADING in planning and f"alpha = {d['alpha']:g} forward passes" in flat
    assert "alpha = 10 (1 research phase)" in flat and "go into more detail on what the earlier phases did not understand" in flat
    assert f"scored on {d['sample']:,} inputs (1,500 before)" in flat and "clearly more than" in flat
    assert planning.index(prompt.TEAM_CONTINUATION_HEADING) < planning.index("## Tools and this session's allocation")
    # the earlier archive, each reward recomputed at the new price, and the earlier notes, marked as earlier
    earlier_candidate = next(c for c in multi_agent.read_archive(path)["candidates"] if c["id"] == "a0002s1")
    receipt = earlier_candidate["selection"]
    now = rewards.compute(receipt["parts"], cfg["reward"], receipt.get("claimed_fidelity"))["reward"]
    assert prompt.TEAM_EARLIER_ARCHIVE in planning
    assert f"a0002s1: scope all; reward {now:.6f} ({receipt['reward']:.6f} at the price it was measured at)" in planning
    assert f"{d['locked']['reward_now']:.4f}" in flat and now < receipt["reward"]
    assert prompt.TEAM_EARLIER_NOTES in planning and "the first shifted up a little" in planning
    assert prompt.TEAM_LATER_NOTES not in planning  # nothing of this run's research yet
    researcher = (later_path / state["sessions"]["a0005"]["directory"] / "prompt.md").read_text()
    assert researcher.index(prompt.TEAM_EARLIER_NOTES) < researcher.index("the first shifted up a little") \
        < researcher.index(prompt.TEAM_LATER_NOTES) < researcher.index("on the earlier program's errors")
    # the earlier program reaches the sessions from the earlier run's folder
    shown = (later_path / state["sessions"]["a0005"]["directory"] / "session/python/001.txt").read_text()
    assert "carried ['a0002s1']" in shown
    assert any("going on from team-early" in line and "start at a0004" in line for line in lines)


def test_the_earlier_programs_are_shown_and_carried_but_never_selected(later):
    _, path, later_path, _ = later
    archive = multi_agent.read_archive(later_path)
    carried = [c for c in archive["candidates"] if c.get("from_run")]
    assert [c["id"] for c in carried] == ["a0002s1"] and carried[0]["from_run"] == "team-early"
    assert carried[0]["selection"]["reward_recorded"] == read_json(path / "lock.json")["reward"]
    assert archive["earlier"]["runs"] == ["team-early"]
    assert archive["global"]["best"] == "a0005s1" and "a0002s1" not in archive["global"]["frontier"]
    assert all("a0002s1" not in group["frontier"] for group in archive["regions"].values())
    lock = read_json(later_path / "lock.json")
    assert lock["candidate_id"] == "a0005s1" and [c["id"] for c in read_json(later_path / "multi_agent/lock.json")["candidates"]] == ["a0005s1"]
    # a resume carries nothing twice
    cfg = read_json(later_path / "run.json")["config"]
    assert not multi_agent._carry_earlier(later_path, cfg, archive) and len(archive["candidates"]) == 2
    # an earlier candidate with a higher reward is still not selected
    synthetic = multi_agent._empty_archive()
    synthetic["candidates"] = [{**c, "selection": {**c["selection"], "reward": 5.0}} if c.get("from_run") else c
                               for c in archive["candidates"]]
    assert multi_agent.update_frontiers(synthetic)["global"]["best"] == "a0005s1"
    report = (later_path / "report.md").read_text()
    assert "## Going on from an earlier run" in report and "after 1 research phase" in report
    assert "Understood share:" in report and "a0002s1 (from team-early)" in report


def test_the_understood_share_is_read_from_the_programs_the_two_runs_locked(later):
    _, _, later_path, _ = later
    cfg = read_json(later_path / "run.json")["config"]
    parts = read_json(later_path / "lock.json")["measured_again"]["parts"]
    share = continuation.understood_share(cfg, parts)
    if parts["fidelity"] == 1.0:
        assert share == pytest.approx(rewards.terms(parts, 1.0)["saved"] / cfg["continuation"]["derived"]["locked"]["saved"])
    else:
        assert share is None


def test_a_team_continuation_can_itself_be_continued(later):
    where, _, later_path, _ = later
    lock = read_json(later_path / "lock.json")
    if lock["measured_again"]["parts"]["fidelity"] == 1.0:
        with pytest.raises(RuntimeError, match="right on every input"):
            continuation.prepare(config(where, "team-third", continue_from="team-later"))
        return
    third = continuation.prepare(config(where, "team-third", continue_from="team-later"))
    d = third["continuation"]["derived"]
    assert d["runs"] == ["team-early", "team-later"] and d["next_index"] == 6 and d["phases"] == 2


def test_a_team_control_gets_the_price_and_sample_and_nothing_else(earlier):
    where, plays, _ = earlier
    control = loop.start(config(where, "team-control", continue_from="team-early", continuation={"memory": False}),
                         fake_scripts=plays, out=lambda *_: None)
    state = read_json(control / "multi_agent/state.json")
    assert sorted(state["sessions"]) == ["a0001", "a0002", "a0003"]
    assert state["sessions"]["a0001"]["task"].startswith("Decide what each researcher works on in the first research phase")
    text = (control / state["sessions"]["a0002"]["directory"] / "prompt.md").read_text()
    assert prompt.TEAM_CONTINUATION_HEADING not in text and prompt.TEAM_EARLIER_NOTES not in text
    assert "shifted up a little" not in text and not any(c.get("from_run") for c in multi_agent.read_archive(control)["candidates"])
    assert "as a control" in (control / "report.md").read_text()


def test_only_runs_of_one_kind_go_on_from_each_other(earlier, tmp_path):
    where, _, path = earlier
    single = run_config.load("config/pilot/rnn-a-smoke.yaml", {"runs_dir": str(where), "run_id": "single-later",
                                                                "backend": "fake", "continue_from": "team-early"})
    with pytest.raises(RuntimeError, match="is a team run and this is a single agent's run"):
        continuation.prepare(single)
    both = config(where, "team-both", continue_from="team-early")
    both["orchestration"]["mode"] = "both"  # a baseline agent first: not for a run that goes on
    with pytest.raises(RuntimeError, match="research only"):
        continuation.prepare(both)
    with pytest.raises(RuntimeError, match="not finished"):
        unfinished = where / "team-unfinished"
        unfinished.mkdir()
        write_json(unfinished / "run.json", {**read_json(path / "run.json"), "run_id": "team-unfinished"})
        continuation.prepare(config(where, "team-later-2", continue_from="team-unfinished"))


def test_the_team_note_is_on_the_agents_surface_and_other_team_prompts_are_as_they_were(tmp_path):
    from level1.networks import load_model

    cfg = config(tmp_path, "plain")
    model = load_model(cfg["model"])
    surface = loop.agent_surface(cfg, model)["continuation note"]
    for text in (prompt.TEAM_CONTINUATION_HEADING, prompt.TEAM_EARLIER_ARCHIVE, prompt.TEAM_EARLIER_NOTES):
        assert text in surface
    plain = prompt.build_multi(cfg, model, 0, "lead", "Plan.", "", [{"notes": "one"}, {"notes": "two"}])
    assert prompt.TEAM_CONTINUATION_HEADING not in plain and "{{" not in plain
    assert "one\n\ntwo" in plain and prompt.TEAM_EARLIER_NOTES not in plain


def test_the_rnn_b_team_detail_configs_go_on_from_the_team_run():
    team = run_config.load("config/experiments/rnn-b-team.yaml")
    detail, smoke = (run_config.load(f"config/experiments/rnn-b-team-detail{s}.yaml") for s in ("", "-smoke"))
    assert detail["continue_from"] == smoke["continue_from"] == team["run_id"]
    assert len({team["run_id"], detail["run_id"], smoke["run_id"]}) == 3
    assert detail["rounds_max"] == 2 and smoke["rounds_max"] == 1 and detail["continuation"]["memory"]
    for key in ("model", "network", "seed_check", "verifier", "reward", "session", "agent_model", "effort", "budget",
                "patience", "orchestration"):
        assert detail[key] == team[key]
    assert smoke["budget"]["run_usd"] < detail["budget"]["run_usd"] and smoke["orchestration"]["workers"] == 2
