"""Ways to raise the score without predicting the network better or computing less, each as a test.

A hole found later gets its own test here.
"""

import pytest
import numpy as np

from level1 import lang, rewards
from level1.lang import Graph
from level1.verifier import Verifier

CFG = {"reward": {"name": "r_alpha", "alpha": 10.0}}


def test_global_baseline_cannot_rely_on_small_control_weights(tmp_path):
    """Packed values fit all controls but overflow trained twins; try beyond three entries.

    The generic claim must survive every declared trained checkpoint before it can
    provide a reward comparator, even when all untrained controls agree exactly.
    """
    from level1 import global_baseline, run_config
    from level1.networks import load_model
    from level1.util import write_json

    cfg = run_config.load("config/pilot/lin-c-smoke.yaml", {
        "runs_dir": str(tmp_path), "backend": "fake", "sandbox": "process", "rounds_max": 1,
        "verifier": {"sample_size": 64, "reference_samples": 1, "workload_size": 16384},
        "session": {"python_calls": 40, "submissions": 5},
        "orchestration": {"enabled": True, "baseline_agent": True, "baseline_validation_version": 2,
                          "development_seed": 73, "final_summary": False}})
    control_cfg = global_baseline.config_for(cfg)
    control = load_model(control_cfg["model"])
    weights = control.load_weights(control_cfg["network"])
    scale = {"embed": 2.0**65, "hidden": 2.0**64, "unembed": 2.0**64}
    programs = []
    for packed in (("unembed",), ("hidden",), ("embed",), ("hidden", "unembed")):
        graph = Graph(weights, 4)
        matrices = {}
        for name in ("embed", "hidden", "unembed"):
            value = graph.weight(name)
            if name in packed:
                value = lang.cast(lang.cast(value * scale[name], int), float) * (1.0 / scale[name])
            matrices[name] = value
        product = matrices["unembed"] @ (matrices["hidden"] @ matrices["embed"])
        programs.append(graph.entry(predict=lang.take_sum(product, graph.tokens(), axis=1).argmax()))
    programs.append(control.reference_entry(weights))
    path = tmp_path / "run"
    plan = {"id": "a0001", "role": "baseline", "directory": "multi_agent/sessions/a0001-baseline"}
    record = {"submissions": [], "meta": {"stop_reason": "end_turn"}}
    verifier = Verifier(control, control_cfg["network"], control_cfg)
    for number, program in enumerate(programs, 1):
        scored = verifier.score(program, seed=73)
        assert global_baseline._full_domain_exact(scored)
        scored.update(submission=number, name=f"candidate-{number}", claim_scope={"kind": "all"})
        record["submissions"].append(scored)
        write_json(path / plan["directory"] / "session/submissions" / f"{number:03d}" / "entry.json", program)
    frozen = global_baseline.freeze(path, cfg, control, plan, record, out=lambda _: None)
    assert frozen["status"] == "established" and frozen["chosen"] == 5
    assert set(frozen["selection"]["tried"]) == {1, 2, 3, 4, 5}
    for number in (1, 2, 3, 4):
        candidate = frozen["candidates"][str(number)]
        assert all(global_baseline._full_domain_exact(row) for row in candidate["confirmation"].values())
        assert any(row["status"] == "refused" for row in candidate["trained_confirmation"].values())
    assert all(global_baseline._full_domain_exact(row) for row in frozen["trained_receipts"].values())


def test_declared_workload_cannot_be_discounted_by_requesting_more_checks(model4):
    """The same fixed setup is not made cheaper by checking a larger sample."""
    w = model4.load_weights("lin-a-1to4-s0")
    entry = model4.reference_entries(w)["contraction"]
    parts = []
    for n in (40, 80):
        v = Verifier(model4, "lin-a-1to4-s0", {
            "verifier": {"check": "sample", "sample_size": n, "workload_size": 512},
        })
        parts.append(v.score(entry, seed=7)["parts"])
    a, b = parts
    assert a["build_ops"] == b["build_ops"] > 0
    assert a["b"] == b["b"] == pytest.approx(a["build_ops"] / (512 * 671))
    for p in parts:
        assert p["cost"] == pytest.approx(p["r"] + (p["r_rule"] + p["b"]) / p["coverage"])


def test_a_fixed_development_seed_prevents_duplicate_draw_luck(model4):
    v = Verifier(model4, "lin-a-1to4-s0", {
        "verifier": {"check": "sample", "sample_size": 50, "sample_seed": 71},
    })
    first, again = v.score(v.reference), v.score(v.reference)
    assert first["inputs"] == again["inputs"]
    assert first["inputs"]["seed"] == 71
    assert first["parts"] == again["parts"]
    assert v.score(v.reference, seed=72)["inputs"]["seed"] == 72


def test_predicate_diagnostics_cannot_redraw_the_fidelity_sample():
    """A scope's optional mass probe is not another development-sample seed."""
    from level1 import regions

    info = {"sizes": {"n_ctx": 2}, "input_space": ["real", "real"], "data_set": {"kind": "normal"}}
    g = Graph({}, 2, real=True)
    entry = g.entry(predict=g.const(0), rule=g.token(0) > 0)
    arrays = []
    for probe_n in (20, 200):
        spec = {"kind": "predicate", "entry": entry, "mass_samples": probe_n, "max_draws": 1000}
        x, _, _ = regions.prepare_inputs(info, {}, {"scope": spec, "sample_size": 64, "workload_size": 200}, seed=7)
        arrays.append(x)
    assert np.array_equal(*arrays)


def test_predicate_rejection_cap_cannot_redraw_successful_samples():
    """Increasing a safety bound keeps the same accepted input prefix."""
    from level1 import regions

    info = {"sizes": {"n_ctx": 2}, "input_space": [4, 3], "data_set": {"kind": "all"}}
    g = Graph({}, 2)
    entry = g.entry(predict=g.const(0), rule=g.token(0) >= 0)
    arrays = []
    for cap in (20, 200):
        spec = {"kind": "predicate", "entry": entry, "mass_samples": 20, "max_draws": cap}
        x, _, _ = regions.prepare_inputs(info, {}, {"scope": spec, "enumerate_up_to": 3,
                                                   "sample_size": 10, "workload_size": 100}, seed=7)
        arrays.append(x)
    assert np.array_equal(*arrays)


def test_running_the_network_is_worth_nothing_anywhere(model, nets):
    """The network as the prediction, on any region: what it saves is what it costs."""
    w = nets["lin-a-s0"][0]
    v = Verifier(model, "lin-a-s0", CFG)
    g = Graph(w, 2)
    out = model.module().reference(g, lang)
    on_half = v.score(g.entry(predict=out, rule=g.token(0) < 32))
    assert on_half["parts"]["r"] == 1.0 and on_half["reward"] == pytest.approx(-1 / 655)  # only the rule's cost


def test_work_cannot_hide_in_the_rule(model, nets):
    """A region defined by running the network (the inputs where a cheap guess is right) pays a forward
    pass on every input of the data set."""
    w = nets["lin-a-s0"][0]
    g = Graph(w, 2)
    guess = lang.maximum(g.token(0), g.token(1))
    receipt = Verifier(model, "lin-a-s0", CFG).score(g.entry(predict=guess, rule=model.module().reference(g, lang) == guess))
    assert receipt["parts"]["fidelity"] == 1.0 and receipt["parts"]["r_rule"] > 1.0 and receipt["reward"] < 0


def test_work_cannot_hide_in_the_build(model, nets):
    """Every output computed ahead and looked up: the build is charged, so the table is no free lunch, and
    the naive table (a forward pass per input, done ahead) is worth nothing."""
    w, ref, x, answers, ops = nets["lin-a-s0"]
    v = Verifier(model, "lin-a-s0", CFG)
    g = Graph(w, 2)
    e, h, u = g.weight("embed"), g.weight("hidden"), g.weight("unembed")
    cols = e.reshape(8, 64, 1) + e.reshape(8, 1, 64)  # every input's summed columns
    hidden = lang.transpose(lang.transpose(cols, [1, 2, 0]) @ h.T, [0, 1, 2])  # hidden applied, per input
    logits = hidden @ u.T
    naive = v.score(g.entry(predict=logits.argmax(axis=2)[g.token(0), g.token(1)]))
    assert naive["status"] == "scored" and naive["parts"]["fidelity"] == 1.0
    assert naive["parts"]["build_ops"] >= 4096 * (64 + 512 + 63) and naive["reward"] < 0.03


def test_a_table_written_out_does_not_fit(model, nets):
    """The outputs pasted into the entry as constants: 4,096 numbers against a limit of 256."""
    w, ref, x, answers, ops = nets["lin-a-s0"]
    g = Graph(w, 2)
    pasted = g.entry(predict=g.const(answers.reshape(64, 64))[g.token(0), g.token(1)])
    receipt = Verifier(model, "lin-a-s0", CFG).score(pasted)
    assert receipt["status"] == "refused" and "constants" in receipt["reason"]


def test_an_exception_copied_from_one_network_does_not_carry_to_the_other(model, nets):
    """One network's deviation written into the entry: perfect there, and the other network shows it."""
    w0, w1 = nets["lin-a-s0"][0], nets["lin-a-s1"][0]
    g = Graph(w0, 2)
    a, b = g.token(0), g.token(1)
    m = lang.maximum(a, b)
    copied = g.entry(predict=lang.where((a == 0) & (b == 0), 1, m))  # read off lin-a-s0's outputs
    on_first = Verifier(model, "lin-a-s0", CFG).score(copied)
    on_second = Verifier(model, "lin-a-s1", CFG).score(copied)
    assert on_first["parts"]["fidelity"] == 1.0
    assert on_second["parts"]["n_agree"] == 4092 and on_second["reward"] < on_first["reward"]
    # the same idea computed from the weights carries: the network run where the tokens are small
    refs0, refs1 = model.reference_entries(w0), model.reference_entries(w1)
    assert refs0["target_with_network_in_corner"] == refs1["target_with_network_in_corner"]
    carried = Verifier(model, "lin-a-s1", CFG).score(refs0["target_with_network_in_corner"])
    assert carried["parts"]["fidelity"] == 1.0


def test_inputs_known_to_be_wrong_never_pay(model, nets):
    """Padding a region with inputs the entry gets wrong lowers the reward whenever alpha is at least 1."""
    for alpha in (1, 2, 10):
        base = {"coverage": 0.5, "fidelity": 1.0, "fidelity_low": 1.0, "r": 0.3, "r_rule": 0.0, "b": 0.0}
        padded = {**base, "coverage": 0.6, "fidelity": 0.5 / 0.6, "fidelity_low": 0.5 / 0.6}
        cfg = {"name": "r_alpha", "alpha": alpha}
        assert rewards.compute(padded, cfg)["reward"] < rewards.compute(base, cfg)["reward"]


def test_a_low_alpha_is_called_out(model, nets):
    """With alpha at 1 or below, a free guess scores above 0 and padding stops costing: the reward says so."""
    w = nets["lin-a-s0"][0]
    g = Graph(w, 2)
    guess = g.entry(predict=g.const(63))  # a constant, right on 127 inputs of 4,096, at no cost
    low = Verifier(model, "lin-a-s0", {"reward": {"name": "r_alpha", "alpha": 1}}).score(guess)
    high = Verifier(model, "lin-a-s0", CFG).score(guess)
    assert low["reward"] == pytest.approx(127 / 4096) and high["reward"] < -8
    assert rewards.warnings({"name": "r_alpha", "alpha": 1}) and rewards.warnings({"name": "r_alpha", "alpha": 0.5})
    assert not rewards.warnings({"name": "r_alpha", "alpha": 10}) and not rewards.warnings({"name": "r_alpha_claimed"})
    assert rewards.warnings({"name": "r_alpha", "lower_bound": False}) and rewards.warnings({"name": "r_alpha_claimed", "penalty": 0})


def test_luck_in_a_sample_does_not_pay(model, nets):
    """On a data set too big to enumerate, every scoring draws its own sample, and the reward uses the
    fidelity's lower bound: an entry cannot be fitted to the inputs it will be scored on, and a sample
    that happens to miss its mistakes is not paid as if it had none."""
    w = nets["lin-a-s0"][0]
    v = Verifier(model, "lin-a-s0", {**CFG, "verifier": {"enumerate_up_to": 100, "sample_size": 300}})
    target = model.reference_entries(w)["target"]
    first, second = v.score(target), v.score(target)
    assert first["inputs"]["mode"] == "sample" and first["inputs"]["seed"] != second["inputs"]["seed"]
    for receipt in (first, second):
        p = receipt["parts"]
        assert p["fidelity_low"] < p["fidelity"] and receipt["reward_detail"]["fidelity_used"] == p["fidelity_low"]
        assert receipt["reward"] < 1 - 1 / 655 - 10 * (1 - p["fidelity"]) + 1e-12


def test_the_best_of_several_samples_is_not_the_number_reported(model4):
    """On a sampled data set the same entry scores a little differently each time, and the lock takes the
    highest: sending one entry again and again raises the recorded reward without predicting better or
    computing less. So the locked entry is scored once more, on a sample of its own, and the report gives
    that number beside the one it was locked on (tests/test_loop.py, the run on a sampled data set)."""
    w = model4.load_weights("lin-a-1to4-s0")
    v = Verifier(model4, "lin-a-1to4-s0", {**CFG, "verifier": {"sample_size": 800}})
    entry = model4.reference_entries(w)["target_then_contraction"]
    rewards_seen = [v.score(entry, seed=s)["reward"] for s in range(12)]
    assert max(rewards_seen) > sum(rewards_seen) / len(rewards_seen) + 0.002      # the best of twelve leans high
    assert len(set(rewards_seen)) > 6                                             # because every scoring is its own sample
    fresh = [v.score(entry)["inputs"]["seed"] for _ in range(3)]
    assert len(set(fresh)) == 3 and not set(fresh) & set(range(12))               # and the verifier, not the agent, draws it


def test_a_build_is_not_made_cheap_by_a_large_data_set(model4):
    """A build is spread over the inputs it is scored on, not over the size of the data set: with 17 million
    possible inputs, anything computed once would otherwise be nearly free, and an entry could run the network
    ahead on a great many inputs and look the answers up."""
    w = model4.load_weights("lin-a-1to4-s0")
    entry = model4.reference_entries(w)["contraction"]
    small = Verifier(model4, "lin-a-1to4-s0", {**CFG, "verifier": {"sample_size": 500}}).score(entry, seed=1)["parts"]
    large = Verifier(model4, "lin-a-1to4-s0", {**CFG, "verifier": {"sample_size": 2000}}).score(entry, seed=1)["parts"]
    assert small["build_ops"] == large["build_ops"] > 0
    assert small["b"] == pytest.approx(small["build_ops"] / (500 * 671)) and large["b"] == pytest.approx(large["build_ops"] / (2000 * 671))
    assert small["b"] > 1000 * small["build_ops"] / (model4.n_total * 671)


def test_a_claim_is_never_worth_more_than_the_measurement(model, nets):
    """The default reward does not read the claim at all; the variant that does pays an overclaim back twice."""
    w = nets["lin-a-s0"][0]
    target = model.reference_entries(w)["target"]
    plain = Verifier(model, "lin-a-s0", CFG)
    assert plain.score(target, claimed=1.0)["reward"] == plain.score(target, claimed=0.5)["reward"]
    assert plain.score(target, claimed=1.0)["overclaim"] == pytest.approx(1 / 4096)  # recorded all the same
    claimed = Verifier(model, "lin-a-s0", {"reward": {"name": "r_alpha_claimed", "alpha": 10.0, "penalty": 2.0}})
    honest = claimed.score(target, claimed=4095 / 4096)["reward"]
    assert claimed.score(target, claimed=1.0)["reward"] < honest and claimed.score(target, claimed=0.99)["reward"] < honest
    assert honest == pytest.approx(plain.score(target)["reward"])


def test_the_region_cannot_be_weighted(model, nets):
    """An entry names a subset, never a distribution: every input of the data set counts once, and the
    coverage is the verifier's count."""
    w = nets["lin-a-s0"][0]
    g = Graph(w, 2)
    receipt = Verifier(model, "lin-a-s0", CFG).score(g.entry(predict=g.token(0), rule=g.token(0) >= g.token(1)))
    assert receipt["parts"]["n_inputs"] == 4096 and receipt["parts"]["n_in_region"] == 64 * 65 // 2


def test_lazy_branches_are_charged_where_they_run(model, nets):
    w = nets["lin-a-s0"][0]
    g = Graph(w, 2)
    a, b = g.token(0), g.token(1)
    m = lang.maximum(a, b)
    entry = g.entry(predict=lang.cond(m < 32, model.module().reference(g, lang), m))
    parts = Verifier(model, "lin-a-s0", CFG).score(entry)["parts"]
    assert parts["predict_ops"] == 3 * 4096 + 32 * 32 * 655


def test_only_the_named_weights_and_operations_exist(model, nets):
    v = Verifier(model, "lin-a-s0", CFG)
    raw = {"format": lang.FORMAT, "rule": None, "predict": 1,
           "nodes": [{"op": "weight", "name": "answers"}, {"op": "argmax", "args": [0], "axis": None}]}
    assert "no weights named" in v.score(raw)["reason"]
    raw["nodes"][0] = {"op": "load", "path": "/etc/passwd"}
    assert "unknown operation" in v.score(raw)["reason"]


def test_an_entry_that_leans_on_small_weights_does_not_become_the_baseline(tmp_path):
    """A baseline model's networks are all alike in ways a trained network is not: their weights are small. An entry
    that keeps weights as whole numbers, times 2^64, is exact on every one of them, and a trained network's weights do
    not fit: it is refused there, and a run read against it has no baseline (pilot-rnn-f-exact's last entries, 18
    November 2026: times 2^60, on trained weights of 11 to 38). So freezing also runs a candidate on every network of
    the twins, and goes down every entry until one passes: here four such entries, each ahead of the network itself
    on the second network, fail one after the other, and the network itself is frozen."""
    from pathlib import Path

    from level1 import baseline, loop, run_config
    from level1.util import read_json, read_jsonl

    repo = Path(__file__).resolve().parent.parent
    cfg = run_config.load("config/pilot/lin-a-1to4.yaml", {
        "runs_dir": str(tmp_path), "backend": "fake", "sandbox": "process", "rounds_max": 1,
        "verifier": {"sample_size": 1500, "reference_samples": 2}, "session": {"python_calls": 3, "submissions": 5}})
    lines = []
    path = loop.start(baseline.config_for(cfg), fake_scripts=repo / "tests" / "fake_baseline_packed", out=lines.append)
    packed = {"packed_unembed", "packed_hidden", "packed_embed", "packed_two"}
    on_second = {e["name"]: e for e in read_jsonl(path / "seed_scores.jsonl")}
    assert set(on_second) == packed | {"itself"} and all(baseline.exact(e) for e in on_second.values())  # exact where the weights are small
    assert all(on_second[k]["reward"] > on_second["itself"]["reward"] for k in packed)
    f = read_json(path / baseline.FILE)
    tried = f["tried"]
    assert len(tried) == 5 and {x["name"] for x in tried[:4]} == packed and tried[4]["name"] == "itself"  # more than three, in that order
    twin_ids = ["lin-a-1to4-s0", "lin-a-1to4-s1", "lin-c-s0", "lin-c-s1"]
    for x in tried[:4]:
        assert x["label"] == "fails" and x["why"] == "it was refused on " + ", ".join(sorted(twin_ids, key=x["why"].index)), x
    assert any("no whole-number value" in line and "a network of lin-c" in line for line in lines)
    assert (f["name"], f["label"], f["established"]) == ("itself", "passes", True) and f["rule"] == baseline.RULE
    assert sorted(f["confirmation"]["twin_networks"]) == twin_ids and sorted(f["confirmation"]["networks"]) == [f"lin-e-s{k}" for k in (2, 3, 4, 5)]
    assert all(baseline.exact(r) for r in {**f["confirmation"]["networks"], **f["confirmation"]["twin_networks"]}.values())


def _shifted_scripts(where, times, sessions=1):
    """One script per session: save an entry that is wrong on about one input in a hundred (rnn-a's lower number,
    the first shifted up), then submit that same entry `times` times."""
    from level1.util import write_json

    where.mkdir(parents=True, exist_ok=True)
    calls = [{"tool": "run_python", "args": {"code": "g = graph()\nsave_entry('shifted', g.entry(interpretation="
                                                    "(g.tokens() + g.const([0.05, 0.0])).argmin()))"}}]
    calls += [{"tool": "submit_entry", "args": {"name": "shifted", "claimed_fidelity": 0.99}}] * times
    for i in range(sessions):
        write_json(where / f"{i + 1:02d}.json", {"calls": calls, "response": "## What the network does\nThe same entry, again."})
    return where


def test_submitting_one_entry_again_does_not_buy_another_draw(tmp_path):
    """Each scoring draws its own sample, so one entry scores a little differently each time, and the lock takes
    the best submission: an agent that submits one entry again and again raises the reward it is locked on without
    predicting better or computing less (pilot-rnn-g-detail-2 did, three times in one round). Pooled, every
    submission of an entry counts its samples together, and only the latest, which carries them all, can be
    locked: submitting again adds evidence, not another draw."""
    from level1 import loop, run_config
    from level1.util import read_json

    cfg = run_config.load("config/pilot/rnn-a-smoke.yaml", run_config.merge(
        {"backend": "fake", "sandbox": "process", "runs_dir": str(tmp_path), "run_id": "again", "rounds_max": 1,
         "verifier": {"sample_size": 600, "reference_samples": 2}, "session": {"python_calls": 3, "submissions": 4}}, {}))
    path = loop.start(cfg, fake_scripts=_shifted_scripts(tmp_path / "scripts", 4), out=lambda *_: None)
    subs = read_json(path / "round_001/round.json")["submissions"]
    assert [s["status"] for s in subs] == ["scored"] * 4 and len({s["entry_sha256"] for s in subs}) == 1
    draws = [s.get("draw", s)["parts"] for s in subs]
    # the reward the lock stands on is that of all four samples together: their fidelity, with one sample's bound
    agree = sum(d["n_agree"] for d in draws)
    inside = sum(d["n_in_region"] for d in draws)
    n = sum(d["n_inputs"] for d in draws)
    f = float(draws[0]["model_ops"])
    parts = {"coverage": inside / n, "r": sum(d["predict_ops"] for d in draws) / (inside * f),
             "r_rule": sum(d["rule_ops"] for d in draws) / (n * f), "b": draws[0]["b"],
             "fidelity": agree / inside, "fidelity_low": lang.wilson_lower(agree / 4, inside / 4)}  # one sample's bound
    expected = rewards.compute(parts, cfg["reward"])["reward"]
    lock = read_json(path / "lock.json")
    assert lock["carried_as"] == "r1s4" and loop.best_submission(loop.completed(path))["carried_as"] == "r1s4"
    assert lock["reward"] == pytest.approx(expected, abs=1e-12)
    assert subs[-1]["pooled"]["samples"] == 4 and subs[-1]["parts"]["n_inputs"] == 2400


def test_a_team_member_submitting_one_entry_again_does_not_buy_another_draw(tmp_path):
    """The same in a team: two submissions of one entry on one domain are one candidate whose development score
    counts both samples, and the earlier is never selected on its own."""
    from level1 import loop, multi_agent, run_config
    from level1.util import write_json

    cfg = run_config.load("config/experiments/rnn-a-team.yaml", run_config.merge(
        {"backend": "fake", "sandbox": "process", "runs_dir": str(tmp_path), "run_id": "team-again", "rounds_max": 1,
         "patience": 2, "verifier": {"sample_size": 600, "workload_size": 600, "reference_samples": 2},
         "session": {"python_calls": 8, "submissions": 4},
         "orchestration": {"mode": "trained", "workers": 1, "parallel": False, "final_summary": False,
                           "development_seed": None, "checkpoint_python_calls": None}}, {}))
    scripts = tmp_path / "team-scripts"
    scripts.mkdir()
    write_json(scripts / "01.json", {"calls": [{"tool": "assign_task", "args": {"task": "Find a cheap program."}}],
                                     "response": "## What the network does\nOne task."})
    _shifted_scripts(scripts / "researcher", 2)
    (scripts / "researcher" / "01.json").rename(scripts / "02.json")
    write_json(scripts / "03.json", {"calls": [], "response": "## What the network does\nNothing to add."})
    path = loop.start(cfg, fake_scripts=scripts, out=lambda *_: None)
    archive = multi_agent.read_archive(path)
    first, second = (next(c for c in archive["candidates"] if c["id"] == cid) for cid in ("a0002s1", "a0002s2"))
    assert first["entry_sha256"] == second["entry_sha256"]
    assert first.get("superseded_by") == "a0002s2" and second["selection"]["pooled"]["samples"] == 2
    assert second["selection"]["parts"]["n_inputs"] == 1200
    assert archive["global"]["best"] == "a0002s2" and archive["global"]["frontier"] == ["a0002s2"]


def test_submitting_one_entry_again_does_not_narrow_its_bound(model4):
    """Pooled naively, every submission again would narrow the fidelity's confidence bound and raise the reward of
    an entry that predicts no better. A pooled entry keeps the bound of one sample: the network itself, right on
    every input, scores the same however often it is submitted."""
    from level1 import pooling

    w = model4.load_weights("lin-a-1to4-s0")
    v = Verifier(model4, "lin-a-1to4-s0", {**CFG, "verifier": {"sample_size": 400}})
    entry = model4.reference_entry(w)
    once = v.score(entry, seed=1)
    draws = [pooling.draw(once)]
    for seed in (2, 3, 4, 5):
        new = v.score(entry, seed=seed)
        again = pooling.pooled(new, draws, CFG["reward"])
        draws.append(pooling.draw(new))
    assert again["pooled"]["samples"] == 5 and again["parts"]["fidelity"] == once["parts"]["fidelity"] == 1.0
    assert again["reward"] == pytest.approx(once["reward"], abs=1e-12) and once["reward"] < 0


# ---------------------------------------------------------------- a reward that prices stored numbers

PRICED = {"reward": {"name": "r_alpha_numbers", "alpha": 10.0, "price": 0.01}}


def test_a_stored_number_has_to_earn_its_price(model, nets):
    """Under r_alpha_numbers a table read at one operation is no longer free: every number stored beyond those of
    the network's own definition is charged, and the network itself, which stores no more than that, scores as
    under r_alpha."""
    w = nets["lin-a-s0"][0]
    plain, priced = Verifier(model, "lin-a-s0", CFG), Verifier(model, "lin-a-s0", PRICED)
    g = Graph(w, 2)
    guess = lang.maximum(g.token(0), g.token(1))
    table = g.const([float(i) / 8 for i in range(64)])  # 64 stored numbers, read at one operation per input
    with_table = g.entry(predict=lang.where(table[g.token(0)] < 100.0, guess, guess))
    a, b = plain.score(with_table), priced.score(with_table)
    stored = lang.stored_numbers(with_table)
    assert stored == 65 and b["parts"]["numbers"] == 65 and "numbers" not in a["parts"]
    network = b["parts"]["numbers_network"]
    assert b["reward"] == pytest.approx(a["reward"] - 0.01 * (65 - network)) and b["reward_detail"]["numbers"] == 65 - network
    g2 = Graph(w, 2)
    itself = g2.entry(predict=model.module().reference(g2, lang))
    assert priced.score(itself)["reward"] == pytest.approx(plain.score(itself)["reward"])
    with pytest.raises(ValueError, match="needs the entry's stored numbers"):  # never scored without the count
        rewards.compute(a["parts"], PRICED["reward"])


def test_numbers_packed_into_the_digits_of_one_are_counted_as_what_they_are(model, nets):
    """Two thresholds written as one long number and taken apart again by arithmetic store what two numbers store,
    and are counted so: a number counts once per four significant digits."""
    w = nets["lin-a-s0"][0]
    g = Graph(w, 2)
    two = g.entry(predict=lang.where(g.token(0) > g.const(1234), g.token(0), lang.where(g.token(1) > g.const(5678), g.token(1), g.token(0))))
    g = Graph(w, 2)
    packed = g.const(12345678)
    first, second = packed // g.const(10000), packed % g.const(10000)
    one = g.entry(predict=lang.where(g.token(0) > first, g.token(0), lang.where(g.token(1) > second, g.token(1), g.token(0))))
    assert lang.stored_numbers(two) == 2 and lang.stored_numbers(one) == 2 + 1  # the two halves, and the 10000 once
    g = Graph(w, 2)
    listed = g.const([0.5, 0.25, 0.123456, 0.1234567890123])[g.token(1) % g.const(4)]  # 1, 1, 2 and 4, and the 4 once
    assert lang.stored_numbers(g.entry(predict=lang.where(listed < g.const(0.3), g.token(0), g.token(1)))) == 1 + 1 + 2 + 4 + 1 + 1


def test_a_number_made_by_the_length_of_an_arange_is_counted(model, nets):
    """arange(n) holds n without a constant: its length is counted like a stored number, so a number is not had
    for nothing by measuring a list."""
    w = nets["lin-a-s0"][0]
    g = Graph(w, 2)
    by_list = g.entry(predict=lang.where(g.token(0) > g.arange(1235).max(), g.token(0), g.token(1)))
    g = Graph(w, 2)
    by_const = g.entry(predict=lang.where(g.token(0) > g.const(1234), g.token(0), g.token(1)))
    assert lang.stored_numbers(by_list) == lang.stored_numbers(by_const) == 1


OPEN = "not closed yet: found by the agents of pilot-rnn-g-numbers-002 and -005 (docs/REWARD.md, the table of ways to game)"


@pytest.mark.xfail(strict=True, reason=OPEN)
def test_numbers_held_as_positions_in_a_counter_are_counted(model, nets):
    """A table written as places in one long arange, a slice for each number, holds what the same table holds as a
    constant, and must be counted so. The agent of pilot-rnn-g-numbers-002 held about 290 fitted numbers this
    way and was charged for one."""
    w = nets["lin-a-s0"][0]
    values = [16.03, 2.72, 50.0, 33.97, 47.53, 46.85, 54.6, 45.4]
    g = Graph(w, 2)
    as_const = g.entry(predict=lang.where(g.const(values)[g.token(0) % g.const(8)] < 40.0, g.token(0), g.token(1)))
    g = Graph(w, 2)
    counter = g.arange(100000)
    held = lang.cast(lang.stack([counter[int(round(v * 1000))] for v in values]), "float") / lang.cast(counter[1000], "float")
    as_places = g.entry(predict=lang.where(held[g.token(0) % counter[8]] < lang.cast(counter[40], "float"), g.token(0), g.token(1)))
    assert Graph.load(as_places, w, 2)[0] is not None and Graph.load(as_const, w, 2)[0] is not None
    assert lang.stored_numbers(as_places) >= lang.stored_numbers(as_const)


@pytest.mark.xfail(strict=True, reason=OPEN)
def test_numbers_made_from_the_input_by_arithmetic_are_counted(model, nets):
    """A token minus itself is 0 on every input, e to that power is 1, and sums and quotients of 1 give any
    threshold: a number that is the same on every input and for every weight is a stored number, however it is
    made. The agent of pilot-rnn-g-numbers-005 made its thresholds this way and was charged for none."""
    w = nets["lin-a-s0"][0]
    g = Graph(w, 2)
    plain = g.entry(predict=lang.where(lang.cast(g.token(0), "float") > g.const(0.75), g.token(0), g.token(1)))
    g = Graph(w, 2)
    one = lang.exp(lang.cast(g.token(0) - g.token(0), "float"))
    two = one + one
    made = (two + one) / (two + two)                              # 0.75, with no constant
    hidden = g.entry(predict=lang.where(lang.cast(g.token(0), "float") > made, g.token(0), g.token(1)))
    assert lang.stored_numbers(plain) == 1 and lang.stored_numbers(hidden) >= lang.stored_numbers(plain)
