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
    assert "n2 = maximum(n0, n1)   <- interpretation" in text and "| r4s2 | `best` | 0.9960 |" in text
    assert "a guard inside Python blocked every escape the probe tried" in text and "—" not in text
    run = read_json(path / "run.json")
    assert run["sandbox"]["probe"]["critical_escapes"] and run["sandbox"]["guard_probe"]["critical_escapes"] == []   # process mode
    rows = loop.rescore(path, {"name": "r_alpha", "alpha": 1}, out=lambda *_: None)
    assert rows[0]["rescored"] == pytest.approx(rows[0]["recorded"]) and rows[1]["rescored"] > rows[1]["recorded"]
    # the frontier of right share, wrong share and cost: the half region, the whole one and the exact one each
    # hold a corner; the same entry sent again does not
    assert {s["carried_as"]: s["on_frontier"] for s in rep["submissions"]} == \
        {"r1s1": True, "r2s1": True, "r3s1": True, "r4s1": False, "r4s2": False}
    assert rep["submissions"][0]["whole_cost"] == pytest.approx(3 / 655)
    assert "| r1s1 | `half` | 0.4977 | 0.5000 | 1.00000 | 0.0046 | 0.5000 | 0.00000 | yes |" in text
    assert rep["submissions"][1]["wrong_share"] == pytest.approx(1 / 4096) and "measured again" not in text   # every input was checked
    assert rep["rounds"][0]["seconds"] is not None and "| Minutes | Output tokens | Programs run | Sandbox minutes |" in text
    assert rep["rounds"][0]["exploration"]["programs"] == 1 and "No program's text reaches for files or the system." in text
    assert "The guard inside Python refused nothing" in text and rep["guard_escapes"] == []


def test_the_frontier_needs_no_reward():
    def sub(name, coverage, fidelity, r, r_rule=0.0, b=0.0, status="scored"):
        return {"carried_as": name, "status": status, "parts": {"coverage": coverage, "fidelity": fidelity, "r": r, "r_rule": r_rule, "b": b}}

    subs = [sub("a", 1.0, 0.90, 0.10), sub("b", 1.0, 0.95, 0.10), sub("c", 0.5, 1.0, 0.05, r_rule=0.01),
            sub("d", 0.5, 0.99, 0.08, b=0.01), sub("e", 1.0, 0.95, 0.10), sub("f", 0.0, None, None), sub("g", 1.0, 1.0, 0.0, status="refused")]
    assert report.frontier(subs) == {"b", "c"}  # a and d are dominated; e repeats b; f covers nothing; g was refused
    assert report.whole_cost(subs[2]["parts"]) == pytest.approx(0.05 + 0.01 / 0.5) and report.whole_cost(subs[5]["parts"]) is None
    # the axes are the share answered rightly and the share answered wrongly, not coverage and fidelity: an entry
    # that answers the same inputs rightly and more of them wrongly is dominated, though it covers more
    careful, careless = sub("careful", 0.5, 1.0, 0.01), sub("careless", 1.0, 0.5, 0.01)
    assert report.shares(careful["parts"]) == (0.5, 0.0) and report.shares(careless["parts"]) == (0.5, 0.5)
    assert report.frontier([careless, careful]) == {"careful"}


FAKE_1TO4 = Path(__file__).parent / "fake_run_1to4"


@pytest.fixture(scope="module")
def finished_1to4(tmp_path_factory):
    """A scripted run on the data set of 1 to 4 tokens, which the verifier samples."""
    base = {"runs_dir": str(tmp_path_factory.mktemp("runs4")), "backend": "fake", "sandbox": "process", "rounds_max": 2,
            "verifier": {"sample_size": 1500, "reference_samples": 3}, "session": {"python_calls": 3, "submissions": 2}}
    lines = []
    path = loop.start(run_config.load("config/pilot/lin-a-1to4.yaml", base), fake_scripts=FAKE_1TO4, out=lines.append)
    return path, lines


def test_a_run_on_a_sampled_data_set(finished_1to4, model4):
    path, lines = finished_1to4
    run = read_json(path / "run.json")
    assert (run["check"], run["n_inputs"], run["enumerated"], run["model_ops"]) == ("sample", 1500, False, 671)
    assert any("1,500 inputs sampled per scoring" in line for line in lines)
    first = (path / "round_001" / "prompt.md").read_text()
    assert "An input is drawn like this." in first and "1,500 inputs drawn afresh from the data set for each submission" in first
    assert "Because the verifier scores a sample" in first and "n_ctx = 4" in first and "{{" not in first
    forbidden = loop.forbidden_strings(run["config"], model4)
    assert {"pairs", "two tokens", "trained on"} <= set(forbidden) and not hygiene.hits({"prompt": first}, forbidden)
    r1, r2 = loop.completed(path)
    s1, s2 = r1["submissions"][0], r2["submissions"][0]
    assert s1["inputs"]["mode"] == "sample" and s1["inputs"]["seed"] != s2["inputs"]["seed"]
    assert 0.2 < s1["parts"]["coverage"] < 0.3 and s1["parts"]["fidelity_low"] < s1["parts"]["fidelity"] and 0.1 < s1["reward"] < 0.3
    assert s2["reward"] > 0.6 and r2["improved"] and r1["notes"].startswith("## What the network does") and "## How you found it" in r1["notes"]
    out1 = (path / "round_001" / "session" / "python" / "001.txt").read_text()
    assert "inputs (1500, 4) forward operations 671 check sample tokens per input [1, 2, 3, 4]" in out1 and "True" in out1


def test_the_locked_entry_is_measured_again_and_results_are_kept(finished_1to4):
    path, _ = finished_1to4
    lock, seed = read_json(path / "lock.json"), read_json(path / "seed_check.json")
    sub = loop.completed(path)[1]["submissions"][0]
    assert lock["carried_as"] == "r2s1" and lock["measured_again"]["status"] == "scored"
    assert lock["measured_again"]["inputs"]["seed"] != sub["inputs"]["seed"]        # a sample of its own
    assert abs(lock["measured_again"]["reward"] - lock["reward"]) < 0.1 and seed["network"] == "lin-a-1to4-s1"
    again = loop.finish(path, out=lambda *_: None)                                  # nothing measured on a sample is redrawn
    assert again == lock and read_json(path / "seed_check.json") == seed
    text = (path / "report.md").read_text()
    assert "lin-a-1to4-s0 (measured again, on a fresh sample)" in text and "The first row is the best of several scorings" in text
    assert "1,500 drawn afresh for each scoring" in text and "| target_then_contraction | lin-a-1to4-s1 |" in text
    # a reference is scored on several fixed samples and its mean given, with how far the scorings lay apart
    ref = read_json(path / "report.json")["references"]["target_then_contraction"]["lin-a-1to4-s0"]
    assert ref["scorings"] == 3 and 0 < ref["reward_spread"] < 0.1 and 0.6 < ref["reward"] < 0.8
    assert "Each number is the mean of 3 scorings on fixed samples." in text and "| Cost | Reward, highest less lowest |" in text
    itself = read_json(path / "report.json")["references"]["the network itself"]["lin-a-1to4-s0"]
    assert itself["parts"]["fidelity"] == 1.0 and itself["parts"]["cost"] == 1.0 and itself["reward_spread"] == 0.0


def test_the_report_says_what_the_agent_ran(finished_1to4):
    path, _ = finished_1to4
    rep = read_json(path / "report.json")
    text = (path / "report.md").read_text()
    assert [rd["exploration"]["programs"] for rd in rep["rounds"]] == [1, 1]
    assert rep["rounds"][0]["exploration"]["flagged"] == []
    assert rep["rounds"][1]["exploration"]["flagged"] == [{"program": "round_002/session/python/001.py", "reaches": ["import machinery", "open(", "socket"]}]
    assert "- `round_002/session/python/001.py`: import machinery, open(, socket" in text and "nothing stopped such a reach" not in text
    # what the guard refused, from the program's record, in the report; and what the agent saw
    blocked = rep["rounds"][1]["exploration"]["blocked"]
    assert [(b["event"], b["target"]) for b in blocked][0] == ("open", "/var/lv1-nothing-here")
    assert blocked[1]["event"].startswith("socket.") and rep["rounds"][1]["exploration"]["incomplete"] == []
    assert "The guard inside Python refused 2 attempts to reach files, the network or other programs:" in text
    assert "- `round_002/session/python/001.py`: open /var/lv1-nothing-here" in text
    seen = (path / "round_002" / "session" / "python" / "001.txt").read_text()
    assert "blocked a file outside PermissionError" in seen and "blocked the network PermissionError" in seen


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


def test_the_smoke_config_is_the_pilot_made_small(tmp_path):
    smoke = run_config.load("config/pilot/lin-a-smoke.yaml")
    pilot = run_config.load("config/pilot/lin-a.yaml")
    assert smoke["rounds_max"] == 1 and smoke["session"] == {"python_calls": 8, "submissions": 2, "max_turns": 40, "seconds": 1200}
    same = set(pilot) - {"run_id", "rounds_max", "session"}
    assert all(smoke[k] == pilot[k] for k in same) and smoke["run_id"] != pilot["run_id"]
    wide, wide_smoke = run_config.load("config/pilot/lin-a-1to4.yaml"), run_config.load("config/pilot/lin-a-1to4-smoke.yaml")
    assert wide_smoke["rounds_max"] == 1 and wide_smoke["session"] == smoke["session"]
    assert all(wide_smoke[k] == wide[k] for k in set(wide) - {"run_id", "rounds_max", "session"}) and wide_smoke["run_id"] != wide["run_id"]
    assert (wide["model"], wide["network"], wide["seed_check"]["network"]) == ("models/lin-a-1to4", "lin-a-1to4-s0", "lin-a-1to4-s1")
    assert wide["verifier"]["check"] == "sample" and wide["verifier"]["sample_size"] == 16384
    assert all(wide[k] == pilot[k] for k in set(pilot) - {"run_id", "model", "network", "seed_check", "verifier"})   # only the data set differs


def test_the_command_line_says_what_is_wrong_without_a_traceback(tmp_path, capsys):
    assert loop.main(["run", "--config", "config/pilot/lin-a.yaml", "--runs-dir", str(tmp_path), "--sandbox", "nonsense",
                      "--fake", str(FAKE)]) == 1
    assert "error: sandbox must be auto, hardened or one of" in capsys.readouterr().err
    assert loop.main(["resume", "--run", str(tmp_path / "no-such-run")]) == 1
    assert "error:" in capsys.readouterr().err
    assert not (tmp_path / "pilot-lin-a" / "run.json").exists()


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
