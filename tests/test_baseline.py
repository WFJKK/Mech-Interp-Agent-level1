"""The baseline (level1/baseline.py): finding the baseline run of a run's shape and conditions, freezing what it
achieved, and reading a run against it; and the command line's three ways to start (a run, its baseline, both)."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from level1 import baseline, lang, loop, prompt, report, run_config
from level1.networks import load_model
from level1.util import read_json, read_jsonl, write_json

# Some Macs' numpy warns of an overflow, an invalid value or a division by zero in every matmul, with nothing
# wrong (docs/SANDBOX.md; the agent's sandbox drops the same three messages). Any other warning still shows.
pytestmark = pytest.mark.filterwarnings("ignore:(divide by zero|overflow|invalid value) encountered in matmul:RuntimeWarning")

REPO = Path(__file__).resolve().parent.parent
FAKE_1TO4 = REPO / "tests" / "fake_run_1to4"
FAKE_ANY = REPO / "tests" / "fake_run_any"  # a rule of the inputs alone: exact on no network
FAKE_BASELINE = REPO / "tests" / "fake_baseline"  # one round on a linear model: two entries exact for any weights, and a rule


def config(runs, path="config/pilot/lin-a-1to4.yaml", **over):
    base = {"runs_dir": str(runs), "backend": "fake", "sandbox": "process", "rounds_max": 2,
            "verifier": {"sample_size": 1500, "reference_samples": 2}, "session": {"python_calls": 3, "submissions": 3}}
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
    bpath = loop.run_baseline(cfg, fake_scripts=FAKE_BASELINE, out=blines.append)
    return {"cfg": cfg, "run": run, "baseline": bpath, "before": before, "lines": blines}


def test_which_models_have_a_baseline_and_where_it_would_be(tmp_path):
    assert baseline.baseline_models(load_model("models/att-d")) == ["att-e"]
    assert baseline.baseline_models(load_model("models/lin-b")) == ["lin-f"]
    assert baseline.baseline_models(load_model("models/lin-a")) == []  # the pilot on pairs has no baseline model
    assert baseline.twins(load_model("models/att-e")) == ["att-c", "att-d"] and not baseline.is_baseline(load_model("models/att-c"))
    cfg = config(tmp_path)
    p = baseline.plan(cfg)
    assert p["state"] == "new" and p["path"].parent == tmp_path and p["path"].name.startswith("pilot-lin-e-exact-")
    b = p["config"]
    assert (b["model"], b["network"], b["seed_check"]) == ("models/lin-e", "lin-e-s0", {"network": "lin-e-s1"})
    assert {k for k in set(b) | set(cfg) if b.get(k) != cfg.get(k)} == {"run_id", "model", "network", "seed_check", baseline.EXACT}
    assert b[baseline.EXACT] is True and cfg[baseline.EXACT] is False
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
    # the agent was given the very same prompt, but for the paragraph that asks for an entry exact for any weights
    assert baseline.same_first_prompt(pair["run"], pair["baseline"]) is True
    mine, theirs = ((p / "round_001" / "prompt.md").read_text() for p in (pair["baseline"], pair["run"]))
    assert prompt.EXACT_NOTE in mine and prompt.EXACT_NOTE not in theirs and mine.replace(prompt.EXACT_NOTE, "") == theirs
    assert not (pair["baseline"] / baseline.LOCK).exists()  # the lock is let go


def test_freezing_picks_the_best_exact_entry_on_the_second_network_and_checks_it_on_the_others(pair):
    f = read_json(pair["baseline"] / baseline.FILE)
    on_second = [e for e in read_jsonl(pair["baseline"] / "seed_scores.jsonl") if e["status"] == "scored"]
    assert sorted(e["name"] for e in on_second) == ["exact", "high", "itself"]
    best = max((e for e in on_second if baseline.exact(e)), key=lambda e: e["reward"])
    assert best["name"] == "exact" and not baseline.exact(next(e for e in on_second if e["name"] == "high"))
    assert (f["chosen"], f["entry_sha256"]) == (best["carried_as"], best["entry_sha256"])
    assert lang.entry_hash(read_json(pair["baseline"] / baseline.ENTRY)) == f["entry_sha256"]
    nets = f["confirmation"]["networks"]
    assert sorted(nets) == ["lin-e-s2", "lin-e-s3", "lin-e-s4", "lin-e-s5"]
    assert {r["inputs"]["seed"] for r in nets.values()} == {f["confirmation"]["inputs_seed"]}  # one fresh sample for all
    for r in nets.values():
        lo, hi = r["reward_range"]
        assert lo == r["reward"] and hi >= lo and set(r["parts"]) >= {"coverage", "fidelity", "b", "r", "build_ops", "predict_ops"}
        assert baseline.exact(r)  # the network's output on every input of its region
    assert f["label"] == "passes" and f["established"] is True and f["asked_for_exactness"] is True and len(f["tried"]) == 1
    assert f["label"] == baseline.verdict(nets)[0]
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
    assert baseline.exact(lines[0]) and baseline.exact(lines[1])  # exact for any weights: exact on the run's networks too
    text = (run / "report.md").read_text()
    assert "## Against the baseline" in text and "accounting, not a causal split" in text
    assert "byte for byte, but for the paragraph that asks for an entry exact for any weights: yes" in text
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


def _scored(agree, region):
    return {"status": "scored", "reward": 0.9, "parts": {"n_agree": agree, "n_in_region": region}}


@pytest.mark.parametrize("receipts, label, why", [
    ({"a": _scored(100, 100), "b": _scored(40, 40)}, "passes", "every input of its region on all 2 networks"),
    ({"a": _scored(100, 100), "b": _scored(99, 100)}, "fails", "on b it gave the network's output on 99 of the 100 inputs"),
    ({"a": _scored(98, 100), "b": _scored(99, 100)}, "fails", "and it was not exact on 1 more of the 2 networks"),
    ({"a": _scored(100, 100), "b": {"status": "refused", "reason": "x"}}, "fails", "it was refused on b"),
    ({"a": _scored(0, 0)}, "fails", "0 of the 0 inputs"),  # an empty region is exact about nothing
    ({}, "fails", "on no other network"),
])
def test_the_verdict_is_exactness_on_every_other_network(receipts, label, why):
    got = baseline.verdict(receipts)
    assert got[0] == label and why in got[1]


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
    assert loop.run_with_baseline(cfg, fake_scripts=FAKE_BASELINE, out=lines.append) == "finished"
    run = tmp_path / "pilot-lin-c"
    b = baseline.find(cfg, where=tmp_path)
    assert b["state"] == "frozen" and b["path"].name.startswith("pilot-lin-e-exact-") and baseline.established(b["path"])
    assert (b["path"] / "console.txt").is_file() and (b["path"] / baseline.REQUEST).is_file()
    assert read_json(b["path"] / baseline.REQUEST)["config"]["model"] == "models/lin-c"
    assert any("its baseline goes alongside, in process" in line for line in lines)
    assert any("the baseline run ended" in line for line in lines)
    assert baseline.reading(run)["state"] == "read" and not (b["path"] / baseline.LOCK).exists()
    # a second run of the same shape and conditions finds it on record and starts nothing alongside
    lines2 = []
    cfg2 = {**cfg, "run_id": "pilot-lin-c-again"}
    assert loop.run_with_baseline(cfg2, fake_scripts=FAKE_BASELINE, out=lines2.append) == "finished"
    assert any("its baseline is on record" in line for line in lines2)
    assert baseline.reading(tmp_path / "pilot-lin-c-again")["state"] == "read"


@pytest.fixture(scope="module")
def two_entries(tmp_path_factory):
    """A baseline run on lin-e whose round submits three different entries: two exact for any weights (the three
    matrices multiplied once, and the network itself) and a rule of the inputs, which is exact on none."""
    runs = tmp_path_factory.mktemp("runs2")
    cfg = baseline.config_for(config(runs, rounds_max=1))
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
    monkeypatch.setattr(baseline, "verdict", lambda receipts: next(verdicts))
    f = baseline.freeze(_unfrozen_copy(two_entries, tmp_path), lambda *a: None)
    # the rule of the inputs was not exact on the second network, so it is never tried
    assert [t["name"] for t in f["tried"]] == ["exact", "itself"] and [t["label"] for t in f["tried"]] == ["fails", "passes"]
    assert f["name"] == "itself" and f["established"] is True
    assert f["tried"][0]["on_second_reward"] > f["tried"][1]["on_second_reward"]  # tried in that order


def test_when_none_passes_the_first_is_frozen_and_not_established(two_entries, tmp_path, monkeypatch):
    monkeypatch.setattr(baseline, "verdict", lambda receipts: ("fails", "made up for the test"))
    copy = _unfrozen_copy(two_entries, tmp_path)
    f = baseline.freeze(copy, lambda *a: None)
    assert f["name"] == "exact" and f["label"] == "fails" and f["established"] is False and len(f["tried"]) == 2
    text = "\n".join(report.baseline_lines({"baseline": baseline.reading(copy)}))
    assert "not established" in text and "Tried, in the order of their reward on the second network" in text
    assert "no run is read against it" in text


def test_a_report_written_again_scores_the_references_once(pair, monkeypatch):
    calls = []
    real = report.references
    monkeypatch.setattr(report, "references", lambda *a, **k: calls.append(1) or real(*a, **k))
    (pair["run"] / report.REFERENCES_CACHE).unlink(missing_ok=True)
    first = report.build(pair["run"])["references"]
    second = report.build(pair["run"])["references"]
    assert len(calls) == 1 and first == second
    cached = read_json(pair["run"] / report.REFERENCES_CACHE)
    write_json(pair["run"] / report.REFERENCES_CACHE, {**cached, "key": {**cached["key"], "prices": "prices-v0"}})
    report.build(pair["run"])  # anything it depends on changed: scored again
    assert len(calls) == 2


def _copy_runs(pair, where, baseline_config=None):
    """The pair's run and its baseline run, copied into another runs folder; the baseline's config changed if asked."""
    import shutil
    run, base = where / pair["run"].name, where / pair["baseline"].name
    shutil.copytree(pair["run"], run)
    shutil.copytree(pair["baseline"], base)
    if baseline_config is not None:
        r = read_json(base / "run.json")
        write_json(base / "run.json", {**r, "config": baseline_config(r["config"])})
    (run / baseline.SCORES).unlink(missing_ok=True)
    return run, base


def test_a_baseline_run_not_asked_for_exactness_serves_no_run(pair, tmp_path):
    """One made before the prompt asked for exactness (pilot-att-e, pilot-lin-e): its agent fitted its own networks."""
    run, base = _copy_runs(pair, tmp_path, lambda c: {k: v for k, v in c.items() if k != baseline.EXACT})
    assert baseline.find(read_json(run / "run.json")["config"], where=tmp_path) is None
    assert baseline.compare(run, lambda *a: None)["state"] == "none" and not (run / baseline.SCORES).exists()
    assert baseline.runs_it_serves(base) == []
    assert "Its agent was not asked for exactness" in "\n".join(report.baseline_lines({"baseline": baseline.reading(base)}))


def test_a_baseline_that_is_not_established_reads_no_run(pair, tmp_path):
    run, base = _copy_runs(pair, tmp_path)
    write_json(base / baseline.FILE, {**read_json(base / baseline.FILE), "label": "fails", "established": False,
                                      "why": "on lin-e-s3 it gave the network's output on 12 of the 16 inputs of its region"})
    assert baseline.compare(run, lambda *a: None) == {"state": "not established", "baseline_run": base.name,
                                                       "why": "on lin-e-s3 it gave the network's output on 12 of the 16 inputs of its region"}
    assert not (run / baseline.SCORES).exists()  # nothing is scored against it
    r = baseline.reading(run)
    assert r["state"] == "not established"
    text = "\n".join(report.baseline_lines({"baseline": r}))
    assert "no usable baseline for this run" in text and "| The baseline on this run's network |" not in text


def test_an_established_baseline_is_preferred_and_one_that_is_not_comes_last(pair, tmp_path):
    import shutil
    run, base = _copy_runs(pair, tmp_path)
    cfg = read_json(run / "run.json")["config"]
    failed = tmp_path / "pilot-lin-e-exact-older"
    shutil.copytree(base, failed)
    r = read_json(failed / "run.json")
    write_json(failed / "run.json", {**r, "run_id": failed.name, "created": "2000-01-01T00:00:00+0000"})  # the earlier of the two
    write_json(failed / baseline.FILE, {**read_json(failed / baseline.FILE), "label": "fails", "established": False})
    assert baseline.find(cfg, where=tmp_path)["path"] == base
    (base / baseline.FILE).unlink()  # the established one not frozen yet: it is finished, which comes before one not established
    assert baseline.find(cfg, where=tmp_path) == {"path": base, "state": "finished", "config": read_json(base / "run.json")["config"]}
    shutil.rmtree(base)
    assert baseline.find(cfg, where=tmp_path)["path"] == failed  # the last resort


def test_a_frozen_entry_that_is_not_exact_on_the_runs_networks_gives_no_split(pair, tmp_path):
    run, base = _copy_runs(pair, tmp_path)
    baseline.compare(run, lambda *a: None)
    lines = read_jsonl(run / baseline.SCORES)
    first = lines[0]
    lines[0] = {**first, "parts": {**first["parts"], "n_agree": first["parts"]["n_in_region"] - 3}}
    (run / baseline.SCORES).write_text("".join(__import__("json").dumps(x) + "\n" for x in lines))
    r = baseline.reading(run)
    assert r["state"] == "not exact here" and r["not_exact"]["first"]["n_in_region"] - r["not_exact"]["first"]["n_agree"] == 3
    text = "\n".join(report.baseline_lines({"baseline": r}))
    assert "the frozen entry is not exact" in text and "nothing is split" in text and "| The baseline on this run's network |" not in text


def test_a_baseline_run_with_no_exact_submission_freezes_nothing(tmp_path):
    cfg = baseline.config_for(config(tmp_path, rounds_max=1))
    path = loop.start(cfg, fake_scripts=FAKE_ANY, out=lambda *a: None)
    f = read_json(path / baseline.FILE)
    assert f["entry_sha256"] is None and f["established"] is False and f["label"] == "none"
    assert "no submission gave the second network's output on every input of its region" in f["why"]
    assert "This baseline is not established, and no run is read against it" in (path / "report.md").read_text()
