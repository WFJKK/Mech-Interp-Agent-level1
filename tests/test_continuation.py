"""Going on from a finished single-agent run at a higher price (level1/continuation.py, docs/CONTINUATION.md).
Only the scripted fake backend is used."""

import hashlib
import math
from pathlib import Path

import pytest

from level1 import continuation, lang, loop, prompt, rewards, run_config
from level1.util import read_json, write_json

SMALL = {"backend": "fake", "sandbox": "process", "verifier": {"sample_size": 1500, "reference_samples": 2},
         "session": {"python_calls": 3, "submissions": 2}, "continuation": {"max_sample_size": 5000}}
NOTES = ("\n\n## How you found it\nFrom its outputs.\n\n## The best entry\n{best}\n\n## Claims\nAs submitted."
         "\n\n## Next\nLook closer.")
# On rnn-a the network's answer is nearly always the place of the lower of the two numbers (tests/test_rnn.py), so an
# entry that shifts the first number up by 0.05 first is wrong on about one input in a hundred: errors to price.
EARLIER = {"calls": [
    {"tool": "run_python", "args": {"code": "g = graph()\nsave_entry('shifted', g.entry(interpretation="
                                            "(g.tokens() + g.const([0.05, 0.0])).argmin()))"}},
    {"tool": "submit_entry", "args": {"name": "shifted", "claimed_fidelity": 0.99, "expected_fidelity_other": 0.99}},
], "response": "## What the network does\nThe place of the lower number, the first shifted up a little; not always." + NOTES.format(best="shifted")}
LATER = {"calls": [
    {"tool": "run_python", "args": {"code": "print('carried', sorted(entries))\ng = graph()\n"
                                            "save_entry('half', g.entry(interpretation=g.tokens().argmin(), region=g.token(0) > 0.0))"}},
    {"tool": "submit_entry", "args": {"name": "half", "claimed_fidelity": 1.0, "expected_fidelity_other": 1.0}},
], "response": "## What the network does\nThe lowest number, where the first is positive." + NOTES.format(best="half")}


def scripts(where: Path, *rounds) -> Path:
    where.mkdir(parents=True)
    for i, item in enumerate(rounds, 1):
        write_json(where / f"{i:02d}.json", item)
    return where


def config(tmp_path, run_id, **over):
    return run_config.load("config/pilot/rnn-a-smoke.yaml",
                           run_config.merge({**SMALL, "runs_dir": str(tmp_path), "run_id": run_id, "rounds_max": 1}, over))


def folder_hash(path: Path) -> str:
    h = hashlib.sha256()
    for f in sorted(p for p in path.rglob("*") if p.is_file()):
        h.update(str(f.relative_to(path)).encode())
        h.update(f.read_bytes())
    return h.hexdigest()


@pytest.fixture(scope="module")
def earlier(tmp_path_factory):
    where = tmp_path_factory.mktemp("continuation")
    plays = scripts(where / "scripts", EARLIER, LATER)
    path = loop.start(config(where, "early"), fake_scripts=plays, out=lambda *_: None)
    lock = read_json(path / "lock.json")
    assert lock["locked"] and lock["measured_again"]["parts"]["fidelity"] < 1.0  # there are errors left to price
    return where, plays, path


def test_the_price_and_the_sample_come_from_the_earlier_locked_entry(earlier):
    where, _, path = earlier
    cfg = continuation.prepare(config(where, "later", continue_from="early"))
    parts = read_json(path / "lock.json")["measured_again"]["parts"]
    t = rewards.terms(parts, parts["fidelity"])
    alpha = max(10.0, math.ceil(2.0 * (1.0 - (1.0 - t["saved"])) / t["errors"]))
    d = cfg["continuation"]["derived"]
    assert cfg["reward"]["alpha"] == d["alpha"] == alpha > 10
    assert cfg["verifier"]["sample_size"] == d["sample"] == max(1500, min(math.ceil(200 * alpha), 5000))
    assert d["first_round"] == 2 and d["runs"] == ["early"] and d["prices"] == [{"run": "early", "rounds": [1, 1], "alpha": 10.0}]
    assert d["locked"]["cost_over_all_inputs"] == pytest.approx(1.0 - t["saved"]) and d["locked"]["errors"] == t["errors"]
    assert d["exact_charge"] == pytest.approx(alpha * (1.0 - lang.wilson_lower(d["sample"], d["sample"])))
    # a number in the config sets the price instead, and the memory can be left out
    fixed = continuation.prepare(config(where, "later", continue_from="early", continuation={"alpha": 77, "memory": False}))
    assert fixed["reward"]["alpha"] == 77 and fixed["continuation"]["derived"]["first_round"] == 1
    assert fixed["continuation"]["derived"]["runs"] == []
    # a run without continue_from is not touched
    plain = config(where, "plain")
    assert continuation.prepare(plain) == plain and continuation.first_round(plain) == 1


@pytest.fixture(scope="module")
def later(earlier):
    where, plays, path = earlier
    before = folder_hash(path)
    lines = []
    later_path = loop.start(config(where, "later", continue_from="early"), fake_scripts=plays, out=lines.append)
    assert folder_hash(path) == before  # the earlier run is read, never written
    return where, path, later_path, lines


def test_a_continuation_numbers_its_rounds_after_the_earlier_ones_and_shows_them_at_the_new_price(later):
    where, path, later_path, lines = later
    cfg = read_json(later_path / "run.json")["config"]
    d = cfg["continuation"]["derived"]
    assert not (later_path / "round_001").exists() and (later_path / "round_002" / "round.json").is_file()
    assert [rd["round"] for rd in loop.completed(later_path)] == [2]
    text = (later_path / "round_002" / "prompt.md").read_text()
    assert "This is round 2 of at most 2." in text
    assert prompt.CONTINUATION_HEADING in text and f"alpha = {d['alpha']:g} forward passes" in text
    assert "alpha = 10 (round 1)" in text and "go into more detail on what the earlier rounds did not understand" in text
    assert f"{d['sample']:,} inputs drawn afresh (1,500 before)" in text
    assert text.index(prompt.CONTINUATION_HEADING) < text.index("## Tools, budgets and limits")
    # the earlier round's table at the new price, and its notes
    earlier_sub = read_json(path / "round_001" / "round.json")["submissions"][0]
    now = rewards.compute(earlier_sub["parts"], cfg["reward"], earlier_sub.get("claimed_fidelity"))["reward"]
    assert f"| r1s1 | scored | {now:.4f} |" in text and "the first shifted up a little" in text
    assert f"{d['locked']['reward_now']:.4f}" in text and now < earlier_sub["reward"]
    # the earlier entries reach the session, and the run ends as any run does
    assert "carried ['best', 'r1s1']" in (later_path / "round_002" / "session" / "python" / "001.txt").read_text()
    lock = read_json(later_path / "lock.json")
    assert lock["carried_as"] == "r2s1" and lock["reward"] == read_json(later_path / "round_002" / "round.json")["submissions"][0]["reward"]
    assert any("going on from early" in line for line in lines)
    report = (later_path / "report.md").read_text()
    assert "## Going on from an earlier run" in report and "after its round 1" in report and "Understood share:" in report


def test_the_understood_share_is_the_saving_kept_by_an_entry_right_everywhere(later):
    _, _, later_path, _ = later
    cfg = read_json(later_path / "run.json")["config"]
    saved = cfg["continuation"]["derived"]["locked"]["saved"]
    exact = {"coverage": 0.5, "fidelity": 1.0, "r": 0.2, "r_rule": 0.01, "b": 0.0}
    assert continuation.understood_share(cfg, exact) == pytest.approx((0.5 * 0.8 - 0.01) / saved)
    assert continuation.understood_share(cfg, {**exact, "fidelity": 0.999}) is None
    assert continuation.understood_share({}, exact) is None


def test_a_continuation_can_itself_be_continued_and_lists_every_price(later):
    where, _, later_path, _ = later
    cfg = read_json(later_path / "run.json")["config"]
    lock = read_json(later_path / "lock.json")
    if lock["measured_again"]["parts"]["fidelity"] == 1.0:
        with pytest.raises(RuntimeError, match="right on every input"):
            continuation.prepare(config(where, "third", continue_from="later"))
        return
    third = continuation.prepare(config(where, "third", continue_from="later"))
    d = third["continuation"]["derived"]
    assert d["runs"] == ["early", "later"] and d["first_round"] == 3
    assert continuation.price_history(d) == f"alpha = 10 (round 1) and alpha = {cfg['reward']['alpha']:g} (round 2)"


def test_a_control_gets_the_price_and_sample_but_none_of_the_earlier_rounds(earlier):
    where, _, path = earlier
    plays = scripts(where / "control-scripts", LATER)
    control = loop.start(config(where, "control", continue_from="early", continuation={"memory": False}),
                         fake_scripts=plays, out=lambda *_: None)
    cfg = read_json(control / "run.json")["config"]
    assert cfg["reward"]["alpha"] == continuation.prepare(config(where, "x", continue_from="early"))["reward"]["alpha"]
    text = (control / "round_001" / "prompt.md").read_text()
    assert prompt.CONTINUATION_HEADING not in text and "None: this is the first round." in text
    assert "This is round 1 of at most 1." in text and f"alpha = {cfg['reward']['alpha']:g}" in text
    assert "carried []" in (control / "round_001" / "session" / "python" / "001.txt").read_text()
    assert "as a control" in (control / "report.md").read_text()


def test_what_cannot_be_continued_is_refused(earlier, tmp_path):
    where, _, path = earlier
    with pytest.raises(RuntimeError, match="there is no run"):
        continuation.prepare(config(where, "later", continue_from="nowhere"))
    with pytest.raises(RuntimeError, match="ran on network"):
        continuation.prepare(config(where, "later", continue_from="early", network="rnn-a-s1", seed_check={"network": "rnn-a-s2"}))
    with pytest.raises(RuntimeError, match="from itself"):
        continuation.prepare(config(where, "early", continue_from="early"))
    with pytest.raises(RuntimeError, match="single agent"):
        continuation.prepare({**config(where, "later", continue_from="early"), "orchestration": {"enabled": True}})
    # an earlier run whose locked entry is right on every input measured leaves no error to price
    exact = loop.start(config(tmp_path, "exact"), fake_scripts=Path("tests/fake_run_numbers"), out=lambda *_: None)
    assert read_json(exact / "lock.json")["measured_again"]["parts"]["fidelity"] == 1.0
    with pytest.raises(RuntimeError, match="right on every input"):
        continuation.prepare(config(tmp_path, "later", continue_from="exact"))
    # an earlier run that is not finished
    unfinished = tmp_path / "unfinished"
    unfinished.mkdir()
    write_json(unfinished / "run.json", {**read_json(path / "run.json"), "run_id": "unfinished"})
    with pytest.raises(RuntimeError, match="not finished"):
        continuation.prepare(config(tmp_path, "later", continue_from="unfinished"))


def test_the_continuation_options_are_checked(tmp_path):
    for bad in ({"c_max": 0}, {"c_max": 1.5}, {"margin": 0.5}, {"inputs_per_alpha": 0}, {"alpha": -1},
                {"memory": "yes"}, {"derived": {}}):
        with pytest.raises(run_config.ConfigError):
            config(tmp_path, "later", continue_from="early", continuation=bad)
    for bad in ("../early", "", 5):
        with pytest.raises(run_config.ConfigError):
            config(tmp_path, "later", continue_from=bad)


def test_the_note_is_on_the_agents_surface_and_every_other_prompt_is_as_it_was(tmp_path):
    from level1.networks import load_model

    cfg = config(tmp_path, "plain")
    model = load_model(cfg["model"])
    assert "continuation note" in loop.agent_surface(cfg, model)
    assert prompt.continuation_note(cfg) == ""
    assert "{{" not in prompt.build(cfg, model, 1, [], 19, False)
    assert prompt.CONTINUATION_HEADING not in prompt.build(cfg, model, 1, [], 19, False)


def test_the_rnn_b_configs_go_on_from_the_pilot_and_differ_from_it_in_nothing_else():
    pilot = run_config.load("config/pilot/rnn-b.yaml")
    detail, smoke, control = (run_config.load(f"config/pilot/rnn-b-detail{s}.yaml") for s in ("", "-smoke", "-control"))
    assert detail["continue_from"] == smoke["continue_from"] == control["continue_from"] == pilot["run_id"]
    assert len({pilot["run_id"], detail["run_id"], smoke["run_id"], control["run_id"]}) == 4
    assert detail["rounds_max"] == control["rounds_max"] == 2 and smoke["rounds_max"] == 1
    assert detail["continuation"]["memory"] and smoke["continuation"]["memory"] and not control["continuation"]["memory"]
    for key in ("model", "network", "seed_check", "verifier", "reward", "session", "agent_model", "effort", "budget", "patience"):
        assert detail[key] == pilot[key] == control[key]
    assert smoke["session"]["python_calls"] < detail["session"]["python_calls"]


def test_the_rnn_g_configs_go_on_from_the_pilot_with_more_time_per_scoring():
    pilot = run_config.load("config/pilot/rnn-g.yaml")
    detail, smoke = (run_config.load(f"config/pilot/rnn-g-detail{s}.yaml") for s in ("", "-smoke"))
    assert detail["continue_from"] == smoke["continue_from"] == pilot["run_id"]
    assert detail["rounds_max"] == 2 and smoke["rounds_max"] == 1
    assert len({pilot["run_id"], detail["run_id"], smoke["run_id"]}) == 3
    assert detail["verifier"] == {**pilot["verifier"], "time_limit_s": 600, "reference_samples": 2}
    for key in ("model", "network", "seed_check", "reward", "session", "agent_model", "effort", "budget", "patience"):
        assert detail[key] == pilot[key]
    assert smoke["session"]["python_calls"] < detail["session"]["python_calls"]


def test_the_note_says_when_entries_have_other_room_than_in_the_earlier_rounds(earlier):
    where, _, path = earlier
    before = read_json(path / "run.json")["config"]["entry"]
    room = {"max_nodes": before["max_nodes"] + 100, "max_const_scalars": before["max_const_scalars"] + 64}
    cfg = continuation.prepare(config(where, "roomier", continue_from="early", entry=room))
    note = " ".join(prompt.continuation_note(cfg).split())
    assert (f"Entries may now have up to {room['max_nodes']} nodes and {room['max_const_scalars']} numbers in their "
            f"constants ({before['max_nodes']} and {before['max_const_scalars']} in the earlier rounds)") in note
    same = continuation.prepare(config(where, "same", continue_from="early"))
    assert "Entries may now have" not in prompt.continuation_note(same)


def test_the_second_rnn_g_continuation_keeps_the_price_and_doubles_the_room():
    first = run_config.load("config/pilot/rnn-g-detail.yaml")
    second, smoke = (run_config.load(f"config/pilot/rnn-g-detail-2{s}.yaml") for s in ("", "-smoke"))
    assert second["continue_from"] == smoke["continue_from"] == first["run_id"]
    assert second["continuation"]["alpha"] == 360 and second["rounds_max"] == 2 and smoke["rounds_max"] == 1
    assert second["entry"] == {"max_nodes": 800, "max_const_scalars": 512}
    assert second["verifier"] == first["verifier"] and second["session"] == first["session"]
