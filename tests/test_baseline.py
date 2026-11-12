"""The baseline (level1/baseline.py): finding the baseline run of a run's shape and conditions, freezing what it
achieved, and reading a run against it; and the command line's three ways to start (a run, its baseline, both)."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from level1 import baseline, lang, loop, report, run_config
from level1.networks import load_model
from level1.util import read_json, read_jsonl, write_json

# Some Macs' numpy warns of an overflow, an invalid value or a division by zero in every matmul, with nothing
# wrong (docs/SANDBOX.md; the agent's sandbox drops the same three messages). Any other warning still shows.
pytestmark = pytest.mark.filterwarnings("ignore:(divide by zero|overflow|invalid value) encountered in matmul:RuntimeWarning")

REPO = Path(__file__).resolve().parent.parent
FAKE_1TO4 = REPO / "tests" / "fake_run_1to4"
FAKE_ANY = REPO / "tests" / "fake_run_any"


def config(runs, path="config/pilot/lin-a-1to4.yaml", **over):
    base = {"runs_dir": str(runs), "backend": "fake", "sandbox": "process", "rounds_max": 2,
            "verifier": {"sample_size": 1500, "reference_samples": 2}, "session": {"python_calls": 3, "submissions": 2}}
    return run_config.load(path, run_config.merge(base, over))


@pytest.fixture(scope="module")
def pair(tmp_path_factory):
    """A scripted run on lin-a-1to4 that finishes before its baseline exists, then its baseline (on lin-e, the
    baseline model of that shape), made with the command's own function."""
    runs = tmp_path_factory.mktemp("runs")
    cfg = config(runs)
    lines = []
    run = loop.start(cfg, fake_scripts=FAKE_1TO4, out=lines.append)
    before = {"lines": list(lines), "report": (run / "report.md").read_text(), "reading": baseline.reading(run)}
    blines = []
    bpath = loop.run_baseline(cfg, fake_scripts=FAKE_ANY, out=blines.append)
    return {"cfg": cfg, "run": run, "baseline": bpath, "before": before, "lines": blines}


def test_which_models_have_a_baseline_and_where_it_would_be(tmp_path):
    assert baseline.baseline_models(load_model("models/att-d")) == ["att-e"]
    assert baseline.baseline_models(load_model("models/lin-b")) == ["lin-f"]
    assert baseline.baseline_models(load_model("models/lin-a")) == []  # the pilot on pairs has no baseline model
    assert baseline.twins(load_model("models/att-e")) == ["att-c", "att-d"] and not baseline.is_baseline(load_model("models/att-c"))
    cfg = config(tmp_path)
    p = baseline.plan(cfg)
    assert p["state"] == "new" and p["path"].parent == tmp_path and p["path"].name.startswith("pilot-lin-e-")
    b = p["config"]
    assert (b["model"], b["network"], b["seed_check"]) == ("models/lin-e", "lin-e-s0", {"network": "lin-e-s1"})
    assert {k for k in set(b) | set(cfg) if b.get(k) != cfg.get(k)} == {"run_id", "model", "network", "seed_check"}
    assert "baseline" not in p["path"].name  # a run's folder name says nothing of what it is for
    with pytest.raises(RuntimeError, match="no baseline model has the shape of lin-a"):
        baseline.config_for(config(tmp_path, path="config/pilot/lin-a.yaml"))


def test_a_run_without_a_baseline_says_how_to_make_one(pair):
    b = pair["before"]
    assert b["reading"]["state"] == "none" and "baseline --run" in b["reading"]["command"]
    assert any("no baseline run for this shape and these conditions yet" in line for line in b["lines"])
    assert "No baseline run for this shape and these conditions yet. To make one: `python -m level1.loop baseline --run" in b["report"]


def test_the_baseline_run_is_a_usual_run_on_the_baseline_model(pair):
    run = read_json(pair["baseline"] / "run.json")
    assert (run["config"]["model"], run["config"]["network"], run["config"]["seed_check"]["network"]) == ("models/lin-e", "lin-e-s0", "lin-e-s1")
    assert not baseline.differences(baseline.conditions(run["config"], run["prices"]), baseline.conditions(pair["cfg"], lang.PRICES))
    assert baseline.same_first_prompt(pair["run"], pair["baseline"]) is True  # the agent was given the very same prompt
    assert not (pair["baseline"] / baseline.LOCK).exists()  # the lock is let go


def test_freezing_picks_the_best_on_the_second_network_and_checks_it_on_the_others(pair):
    f = read_json(pair["baseline"] / baseline.FILE)
    on_second = [e for e in read_jsonl(pair["baseline"] / "seed_scores.jsonl") if e["status"] == "scored"]
    best = max(on_second, key=lambda e: e["reward"])
    assert (f["chosen"], f["entry_sha256"]) == (best["carried_as"], best["entry_sha256"])
    assert lang.entry_hash(read_json(pair["baseline"] / baseline.ENTRY)) == f["entry_sha256"]
    nets = f["confirmation"]["networks"]
    assert sorted(nets) == ["lin-e-s2", "lin-e-s3", "lin-e-s4", "lin-e-s5"]
    assert {r["inputs"]["seed"] for r in nets.values()} == {f["confirmation"]["inputs_seed"]}  # one fresh sample for all
    for r in nets.values():
        lo, hi = r["reward_range"]
        assert lo == r["reward"] and hi >= lo and set(r["parts"]) >= {"coverage", "fidelity", "b", "r", "build_ops", "predict_ops"}
    # the verdict is the one the recorded rewards and their intervals give; the scripted entry is a rule of the inputs
    # alone, nothing of these networks, so it cannot pass (on 1,500 inputs the intervals are wide: fails or inconclusive)
    assert f["tolerance"] == 0.01 and f["label"] == baseline.judge({n: tuple(r["reward_range"]) for n, r in nets.items()}, 0.01)[0]
    assert f["label"] in ("fails", "inconclusive")
    assert "## The baseline" in (pair["baseline"] / "report.md").read_text()


def test_a_frozen_baseline_is_never_frozen_again(pair):
    before = (pair["baseline"] / baseline.FILE).read_text()
    assert baseline.freeze(pair["baseline"], lambda *a, **k: None)["frozen_at"] == read_json(pair["baseline"] / baseline.FILE)["frozen_at"]
    loop.finish(pair["baseline"], out=lambda *a: None)
    assert (pair["baseline"] / baseline.FILE).read_text() == before


def test_the_run_is_read_against_it_on_its_own_inputs(pair):
    run = pair["run"]
    lines = read_jsonl(run / baseline.SCORES)
    assert [e["role"] for e in lines] == ["first", "held out"]
    lock, seed = read_json(run / "lock.json"), read_json(run / "seed_check.json")
    assert lines[0]["inputs"]["seed"] == lock["measured_again"]["inputs"]["seed"] and lines[0]["network"] == "lin-a-1to4-s0"
    assert lines[1]["inputs"]["seed"] == seed["inputs"]["seed"] and lines[1]["network"] == "lin-a-1to4-s1"
    r = baseline.reading(run)
    assert r["state"] == "read" and r["same_first_prompt"] is True
    s = r["split"]
    assert s["baseline"][0] == lines[0]["reward"]
    assert s["kept"][0] == pytest.approx(seed["reward"] - lines[1]["reward"])
    assert sum(v[0] for v in s.values()) == pytest.approx(lock["measured_again"]["reward"])  # the three parts add up to R0
    for value, (lo, hi) in s.values():
        assert lo <= value + 1e-12 and value <= hi + 1e-12
    text = (run / "report.md").read_text()
    assert "## Against the baseline" in text and "byte for byte: yes" in text and "accounting, not a causal split" in text
    assert any(f"{run.name} is read against this baseline" in line for line in pair["lines"])


def test_finishing_again_measures_nothing_twice(pair):
    loop.finish(pair["run"], out=lambda *a: None)
    assert len(read_jsonl(pair["run"] / baseline.SCORES)) == 2


def test_the_command_finds_a_frozen_baseline_and_runs_nothing(pair, capsys):
    rounds_before = sorted(p.name for p in pair["baseline"].glob("round_*"))
    assert loop.main(["baseline", "--run", str(pair["run"]), "--fake", str(FAKE_ANY)]) == 0
    assert "the baseline for these conditions is" in capsys.readouterr().out
    assert sorted(p.name for p in pair["baseline"].glob("round_*")) == rounds_before


def test_other_conditions_need_another_baseline(pair, tmp_path):
    other = {**pair["cfg"], "reward": {"name": "r_alpha", "alpha": 3.0}}
    assert baseline.find(other, where=pair["run"].parent) is None
    assert baseline.plan(other)["state"] == "new"
    assert baseline.find(pair["cfg"], where=pair["run"].parent)["state"] == "frozen"


def test_a_run_measured_under_other_prices_is_not_compared(pair, tmp_path):
    copy = tmp_path / "old-run"
    copy.mkdir()
    for name in ("run.json", "lock.json", "seed_check.json"):
        (copy / name).write_text((pair["run"] / name).read_text())
    run = read_json(copy / "run.json")
    write_json(copy / "run.json", {**run, "prices": "prices-v0"})
    assert baseline.find(read_json(copy / "run.json")["config"], "prices-v0", pair["run"].parent) is None
    with pytest.raises(RuntimeError, match="prices-v0"):
        loop.run_baseline(run["config"], prices="prices-v0", out=lambda *a: None)


def test_a_baseline_that_has_not_finished_is_pending(tmp_path, pair):
    cfg = config(tmp_path)
    p = baseline.plan(cfg)
    p["path"].mkdir()
    write_json(p["path"] / "run.json", {"run_id": p["config"]["run_id"], "config": p["config"], "created": "2026-11-12T00:00:00+0000",
                                        "prices": lang.PRICES})
    assert baseline.plan(cfg)["state"] == "unfinished"
    target = tmp_path / "pilot-lin-a-1to4"
    target.mkdir()
    for name in ("run.json", "lock.json", "seed_check.json"):
        (target / name).write_text((pair["run"] / name).read_text())
    write_json(target / "run.json", {**read_json(target / "run.json"), "config": cfg})
    assert baseline.compare(target, lambda *a: None) == {"state": "pending", "baseline_run": p["path"].name}
    assert "has not been frozen yet" in "\n".join(report.baseline_lines({"baseline": baseline.reading(target)}))


@pytest.mark.parametrize("scores, label", [
    ({"a": (0.95, 0.952), "b": (0.95, 0.952), "c": (0.951, 0.953)}, "passes"),
    ({"a": (0.95, 0.952), "b": (0.80, 0.81)}, "fails"),
    ({"a": (0.95, 0.952), "b": (0.94, 0.96)}, "inconclusive"),
    ({"a": (0.95, 0.952), "b": None}, "fails"),
    ({"a": (0.95, 0.952)}, "inconclusive"),
])
def test_the_verdict(scores, label):
    assert baseline.judge(scores, 0.01)[0] == label


def test_the_reward_range_is_the_fidelity_interval_carried_through():
    parts = {"coverage": 1.0, "fidelity": 1.0, "fidelity_low": lang.wilson_lower(1500, 1500), "n_agree": 1500, "n_in_region": 1500,
             "r": 0.02, "r_rule": 0.0, "b": 0.01, "cost": 0.03}
    cfg = {"name": "r_alpha", "alpha": 10.0}
    from level1 import rewards
    low = rewards.compute(parts, cfg)["reward"]
    lo, hi = baseline.reward_range({"status": "scored", "reward": low, "parts": parts, "inputs": {"mode": "sample"}}, cfg)
    assert lo == low and hi == pytest.approx(0.97)  # all right on the sample: the top of the interval is fidelity 1
    assert baseline.reward_range({"status": "scored", "reward": low, "parts": parts, "inputs": {"mode": "all"}}, cfg) == (low, low)
    assert baseline.reward_range({"status": "refused"}, cfg) == (None, None)


def test_one_process_at_a_time_runs_a_baseline(tmp_path):
    assert baseline.take_lock(tmp_path) and baseline.running(tmp_path) == os.getpid()
    other = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        (tmp_path / baseline.LOCK).write_text(f"{other.pid} now\n")
        assert baseline.running(tmp_path) == other.pid and not baseline.take_lock(tmp_path)
    finally:
        other.kill()
        other.wait()
    assert baseline.running(tmp_path) is None and baseline.take_lock(tmp_path)  # a lock left by a process that is gone
    baseline.release_lock(tmp_path)
    assert not (tmp_path / baseline.LOCK).exists()


def test_a_run_with_its_baseline_alongside(tmp_path):
    """run --with-baseline: the baseline in a second process, the run in this one, and the run read against the
    baseline at the end. Both on scripted rounds."""
    cfg = config(tmp_path, path="config/pilot/lin-c.yaml", rounds_max=1)
    lines = []
    assert loop.run_with_baseline(cfg, fake_scripts=FAKE_ANY, out=lines.append) == "finished"
    run = tmp_path / "pilot-lin-c"
    b = baseline.find(cfg, where=tmp_path)
    assert b["state"] == "frozen" and b["path"].name.startswith("pilot-lin-e-")
    assert (b["path"] / "console.txt").is_file() and (b["path"] / baseline.REQUEST).is_file()
    assert read_json(b["path"] / baseline.REQUEST)["config"]["model"] == "models/lin-c"
    assert any("its baseline goes alongside, in process" in line for line in lines)
    assert any("the baseline run ended" in line for line in lines)
    assert baseline.reading(run)["state"] == "read" and not (b["path"] / baseline.LOCK).exists()
    # a second run of the same shape and conditions finds it on record and starts nothing alongside
    lines2 = []
    cfg2 = {**cfg, "run_id": "pilot-lin-c-again"}
    assert loop.run_with_baseline(cfg2, fake_scripts=FAKE_ANY, out=lines2.append) == "finished"
    assert any("its baseline is on record" in line for line in lines2)
    assert baseline.reading(tmp_path / "pilot-lin-c-again")["state"] == "read"


FAKE_BASELINE = REPO / "tests" / "fake_baseline"  # one round on a linear model: an exact entry, and a rule of the inputs


@pytest.fixture(scope="module")
def two_entries(tmp_path_factory):
    """A baseline run on lin-e whose round submits two different entries, on the pilots' 16,384 inputs per scoring:
    on far fewer, an entry right on every input has sampling intervals wider than the tolerance, and its verdict
    is inconclusive, as it should be."""
    runs = tmp_path_factory.mktemp("runs2")
    cfg = baseline.config_for(config(runs, rounds_max=1, verifier={"sample_size": 16384}))
    path = loop.start(cfg, fake_scripts=FAKE_BASELINE, out=lambda *a: None)
    return path


def _unfrozen_copy(path, where):
    import shutil
    copy = where / path.name
    shutil.copytree(path, copy)
    for name in (baseline.FILE, baseline.ENTRY):
        (copy / name).unlink()
    return copy


def test_an_exact_entry_passes_and_is_frozen(two_entries):
    f = read_json(two_entries / baseline.FILE)
    assert f["name"] == "exact" and f["label"] == "passes" and f["established"] is True and len(f["tried"]) == 1
    rewards = {r["reward"] for r in f["confirmation"]["networks"].values()}
    assert len(rewards) == 1  # the same inputs and an exact entry: the same reward on every network


def test_when_the_best_on_the_second_network_does_not_pass_the_next_is_tried(two_entries, tmp_path, monkeypatch):
    verdicts = iter([("fails", "made up for the test"), ("passes", "made up for the test")])
    monkeypatch.setattr(baseline, "judge", lambda scores, tol: next(verdicts))
    f = baseline.freeze(_unfrozen_copy(two_entries, tmp_path), lambda *a: None)
    assert [t["name"] for t in f["tried"]] == ["exact", "high"] and [t["label"] for t in f["tried"]] == ["fails", "passes"]
    assert f["name"] == "high" and f["established"] is True
    assert f["tried"][0]["on_second_reward"] > f["tried"][1]["on_second_reward"]  # tried in that order


def test_when_none_passes_the_first_is_frozen_and_not_established(two_entries, tmp_path, monkeypatch):
    monkeypatch.setattr(baseline, "judge", lambda scores, tol: ("fails", "made up for the test"))
    copy = _unfrozen_copy(two_entries, tmp_path)
    f = baseline.freeze(copy, lambda *a: None)
    assert f["name"] == "exact" and f["label"] == "fails" and f["established"] is False and len(f["tried"]) == 2
    text = "\n".join(report.baseline_lines({"baseline": baseline.reading(copy)}))
    assert "not established" in text and "Tried, in the order of their reward on the second network" in text
