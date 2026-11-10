"""The whole run with scripted rounds: prompts, sessions, records, stopping, resume, lock, seed check, report."""

from pathlib import Path

import pytest

from level1 import hygiene, loop, report, run_config
from level1.util import read_json, read_jsonl

FAKE = Path(__file__).parent / "fake_run"
FAKE_RESUME = Path(__file__).parent / "fake_resume"


def config(tmp_path, **over):
    base = {"runs_dir": str(tmp_path), "backend": "fake", "sandbox": "process", "session": {"python_calls": 3, "submissions": 2}}
    return run_config.load("config/pilot/lin-a.yaml", run_config.merge(base, over))


@pytest.fixture(scope="module")
def finished(tmp_path_factory):
    lines = []
    path = loop.start(config(tmp_path_factory.mktemp("runs")), fake_scripts=FAKE, out=lines.append)
    return path, lines


def test_the_rounds_and_why_they_end(finished):
    path, lines = finished
    rounds = loop.completed(path)
    assert [rd["improved"] for rd in rounds] == [True, True, False, False]  # patience 2 ends it; script 5 is never asked
    assert "no higher reward in 2 rounds in a row" in next(e["ended"] for e in read_jsonl(path / "log.jsonl") if e.get("ended"))
    r1, r2, r3, r4 = rounds
    assert [s["carried_as"] for s in r1["submissions"]] == ["r1s1"] and r1["submissions"][0]["reward"] == pytest.approx(0.5 * (1 - 1 / 655) - 1 / 655)
    assert r2["submissions"][0]["reward"] == pytest.approx(1 - 1 / 655 - 10 / 4096) and r2["best_reward_before"] == r1["best_reward_after"]
    assert r3["submissions"][0]["name"] == "corner" and r3["submissions"][0]["reward"] < r2["submissions"][0]["reward"]
    assert len(r4["submissions"]) == 2 and r4["tools_used"] == {"run_python": 0, "submit_entry": 2}  # the third was over budget
    assert r1["notes"].startswith("## What the network does") and "Some working text" not in r1["notes"]
    assert r4["notes"] == "No notes in the asked-for form this time."
    assert any("WARNING: the sandbox is in process mode" in line for line in lines)


def test_prompts_carry_the_history_and_nothing_withheld(finished, model):
    path, _ = finished
    first = (path / "round_001" / "prompt.md").read_text()
    third = (path / "round_003" / "prompt.md").read_text()
    assert "This is round 1 of at most 5" in first and "None: this is the first round." in first
    assert "def reference(g, lv1):" in first and "655 operations" in first and "alpha = 10" in first and "{{" not in first
    assert "| r2s1 | scored | 0.9960 | 1.0000 | 0.99976 |" in third and "is also there as `best`" in third
    assert "    The larger token, on every input but one." in third  # the agent's own words come back to it
    forbidden = loop.forbidden_strings(read_json(path / "run.json")["config"], model)
    assert not hygiene.hits({"prompt": first}, forbidden)
    assert "lin-a" not in first and "withheld" not in first and "seed" in first


def test_entries_are_carried_between_rounds(finished):
    path, _ = finished
    carried = sorted(p.stem for p in (path / "round_003" / "session" / "workspace" / "entries").glob("*.json"))
    assert carried == ["best", "corner", "r1s1", "r2s1"]
    assert read_json(path / "round_003" / "session" / "workspace" / "entries" / "best.json") == \
        read_json(path / "round_002" / "session" / "submissions" / "001" / "entry.json")


def test_the_lock_and_the_seed_check(finished):
    path, _ = finished
    lock, seed = read_json(path / "lock.json"), read_json(path / "seed_check.json")
    assert lock["locked"] and lock["carried_as"] == "r2s1" and lock["name"] == "whole"  # the earliest of the equal best
    assert read_json(path / "locked_entry.json") == read_json(path / "round_002" / "session" / "submissions" / "001" / "entry.json")
    assert seed["network"] == "lin-a-s1" and seed["entry_sha256"] == lock["entry_sha256"]
    assert seed["parts"]["n_agree"] == 4092 and seed["reward"] == pytest.approx(1 - 1 / 655 - 10 * 4 / 4096)
    assert seed["claimed_fidelity"] == 0.999 and seed["overclaim"] == 0.0  # 4092 of 4096 is just above the claim
    again = loop.finish(path, out=lambda *_: None)  # from the records alone
    assert again == lock


def test_the_report(finished):
    path, _ = finished
    rep = read_json(path / "report.json")
    text = (path / "report.md").read_text()
    assert rep["references"]["the network itself"]["lin-a-s0"]["reward"] == 0.0
    assert rep["references"]["target"]["lin-a-s1"]["parts"]["n_agree"] == 4092
    assert "| lin-a-s1 (the seed check) | 0.9887 | 1.0000 | 0.99902 |" in text
    assert "n2 = maximum(n0, n1)   <- predict" in text and "| r4s2 | `best` | 0.9960 |" in text
    assert "(weak: the agent's code could read any file" in text and "—" not in text
    rows = loop.rescore(path, {"name": "r_alpha", "alpha": 1}, out=lambda *_: None)
    assert rows[0]["rescored"] == pytest.approx(rows[0]["recorded"]) and rows[1]["rescored"] > rows[1]["recorded"]
    # the frontier of coverage, fidelity and whole cost: the half region, the whole one and the exact one each
    # hold a corner; the same entry sent again does not
    assert {s["carried_as"]: s["on_frontier"] for s in rep["submissions"]} == \
        {"r1s1": True, "r2s1": True, "r3s1": True, "r4s1": False, "r4s2": False}
    assert rep["submissions"][0]["whole_cost"] == pytest.approx(3 / 655) and "| 0.0046 | yes |" in text


def test_the_frontier_needs_no_reward():
    def sub(name, coverage, fidelity, r, r_rule=0.0, b=0.0, status="scored"):
        return {"carried_as": name, "status": status, "parts": {"coverage": coverage, "fidelity": fidelity, "r": r, "r_rule": r_rule, "b": b}}

    subs = [sub("a", 1.0, 0.90, 0.10), sub("b", 1.0, 0.95, 0.10), sub("c", 0.5, 1.0, 0.05, r_rule=0.01),
            sub("d", 0.5, 0.99, 0.08, b=0.01), sub("e", 1.0, 0.95, 0.10), sub("f", 0.0, None, None), sub("g", 1.0, 1.0, 0.0, status="refused")]
    assert report.frontier(subs) == {"b", "c"}  # a and d are dominated; e repeats b; f covers nothing; g was refused
    assert report.whole_cost(subs[2]["parts"]) == pytest.approx(0.05 + 0.01 / 0.5) and report.whole_cost(subs[5]["parts"]) is None


def test_a_stopped_run_resumes_where_it_was(tmp_path):
    cfg = config(tmp_path, run_id="stopped")
    path = run_config.run_path(cfg)
    lines = []
    assert loop.start(cfg, fake_scripts=FAKE_RESUME, out=lines.append) == path
    assert len(loop.completed(path)) == 1 and not (path / "lock.json").exists()  # stopped in round 2: nothing is locked
    assert any("stopped: usage limit (scripted)" in line for line in lines)
    assert len(list((path / "round_002" / "session" / "submissions").iterdir())) == 1
    with pytest.raises(RuntimeError, match="is a run already"):
        loop.start(cfg, fake_scripts=FAKE)
    assert loop.drive(path, fake_scripts=FAKE, out=lambda *_: None) == "finished"
    rounds = loop.completed(path)
    assert len(rounds) == 4 and len(rounds[1]["submissions"]) == 2  # the cut-off session's submission stands
    assert rounds[1]["tools_used"] == {"run_python": 2, "submit_entry": 2} and read_json(path / "lock.json")["carried_as"] == "r2s1"


def test_a_resume_refuses_changed_trusted_files(tmp_path, monkeypatch):
    cfg = config(tmp_path, run_id="changed", rounds_max=1)
    path = loop.start(cfg, fake_scripts=FAKE, out=lambda *_: None)
    monkeypatch.setattr(loop, "trusted_hash", lambda model: "another")
    with pytest.raises(RuntimeError, match="changed since this run began"):
        loop.drive(path, fake_scripts=FAKE)
    assert loop.drive(path, fake_scripts=FAKE, out=lambda *_: None, accept_changes=True) == "finished"


def test_a_run_does_not_start_when_the_agent_would_be_told(tmp_path, monkeypatch):
    cfg = config(tmp_path, run_id="told", hygiene=["forward pass"])
    with pytest.raises(hygiene.HygieneError, match="forward pass"):
        loop.start(cfg, fake_scripts=FAKE)
    assert not (run_config.run_path(cfg) / "round_001").exists() and not (run_config.run_path(cfg) / "run.json").exists()


def test_a_run_with_nothing_scored_locks_nothing(tmp_path):
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "01.json").write_text('{"calls": [], "response": "nothing", "stop_reason": "refusal"}')
    (scripts / "02.json").write_text('{"calls": [], "response": "nothing", "stop_reason": "refusal"}')
    (scripts / "03.json").write_text('{"calls": [], "response": "never asked"}')
    path = loop.start(config(tmp_path, run_id="empty", patience=5), fake_scripts=scripts, out=lambda *_: None)
    assert len(loop.completed(path)) == 2  # two refusals in a row end the rounds
    assert read_json(path / "lock.json") == {"locked": False, "reason": "no submission was scored"}
    assert "nothing was locked" in (path / "report.md").read_text() and not (path / "seed_check.json").exists()


def test_config_errors(tmp_path):
    with pytest.raises(run_config.ConfigError, match="unknown keys"):
        config(tmp_path, alpha=3)
    with pytest.raises(run_config.ConfigError, match="does not take"):
        config(tmp_path, reward={"name": "r_alpha", "alpha": 3, "floor": 0.9})
    assert config(tmp_path, reward={"alpha": 3})["reward"] == {"name": "r_alpha", "alpha": 3}  # same reward: parameters merge
    assert config(tmp_path, reward={"name": "r_alpha_claimed", "penalty": 4})["reward"] == {"name": "r_alpha_claimed", "penalty": 4}
    assert run_config.reward_from_args("r_alpha", ["alpha=2", "lower_bound=false"]) == {"name": "r_alpha", "alpha": 2.0, "lower_bound": False}
    assert run_config.reward_from_args("r_alpha_claimed", [], 5, 3) == {"name": "r_alpha_claimed", "alpha": 5.0, "penalty": 3.0}
    with pytest.raises(run_config.ConfigError, match="KEY=VALUE"):
        run_config.reward_from_args("r_alpha", ["alpha"])
    cfg = config(tmp_path, run_id="same", seed_check={"network": "lin-a-s0"})
    with pytest.raises(RuntimeError, match="another network"):
        loop.start(cfg, fake_scripts=FAKE)
