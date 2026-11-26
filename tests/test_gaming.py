"""Ways to raise the score without predicting the network better or computing less, each as a test.

A hole found later gets its own test here.
"""

import pytest

from level1 import lang, rewards
from level1.lang import Graph
from level1.verifier import Verifier

CFG = {"reward": {"name": "r_alpha", "alpha": 10.0}}


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
    with_table = g.entry(predict=lang.where(table[g.token(0)] < 4.0, guess, guess))
    a, b = plain.score(with_table), priced.score(with_table)
    assert lang.stored_numbers(with_table) == 65 and b["parts"]["numbers"] == 65 and "numbers" not in a["parts"]
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


def test_numbers_held_as_positions_in_a_counter_are_counted(model, nets):
    """A table written as places in one long arange, a slice for each number, holds what the same table holds as a
    constant, and must be counted so. The agent of pilot-rnn-g-numbers-002 (branch multi-agent-agnostic) held
    about 290 fitted numbers this way and was charged for one."""
    w = nets["lin-a-s0"][0]
    priced = Verifier(model, "lin-a-s0", PRICED)
    values = [16.03, 2.72, 50.0, 33.97, 47.53, 46.85, 54.6, 45.4]
    g = Graph(w, 2)
    as_const = g.entry(predict=lang.where(g.const(values)[g.token(0) % g.const(8)] < 40.0, g.token(0), g.token(1)))
    g = Graph(w, 2)
    counter = g.arange(100000)
    held = lang.cast(lang.stack([counter[int(round(v * 1000))] for v in values]), "float") / lang.cast(counter[1000], "float")
    as_places = g.entry(predict=lang.where(held[g.token(0) % counter[8]] < lang.cast(counter[40], "float"), g.token(0), g.token(1)))
    assert Graph.load(as_places, w, 2)[0] is not None and Graph.load(as_const, w, 2)[0] is not None
    assert lang.stored_numbers(as_places) >= lang.stored_numbers(as_const)
    assert priced.score(as_places)["parts"]["numbers"] >= priced.score(as_const)["parts"]["numbers"]


def test_numbers_made_from_the_input_by_arithmetic_are_counted(model, nets):
    """A token minus itself is 0 on every input, e to that power is 1, and sums and quotients of 1 give any
    threshold: a number that is the same on every input and for every weight is a stored number, however it is
    made. The agent of pilot-rnn-g-numbers-005 (branch multi-agent-agnostic) made its thresholds this way and was
    charged for none."""
    w = nets["lin-a-s0"][0]
    priced = Verifier(model, "lin-a-s0", PRICED)
    g = Graph(w, 2)
    plain = g.entry(predict=lang.where(lang.cast(g.token(0), "float") > g.const(0.75), g.token(0), g.token(1)))
    g = Graph(w, 2)
    one = lang.exp(lang.cast(g.token(0) - g.token(0), "float"))
    two = one + one
    made = (two + one) / (two + two)                              # 0.75, with no constant
    hidden = g.entry(predict=lang.where(lang.cast(g.token(0), "float") > made, g.token(0), g.token(1)))
    by_weights = g.entry(predict=lang.where(lang.cast(g.token(0), "float") > made + g.weight("embed")[0, 0] * 0.0, g.token(0), g.token(1)))
    p, h = priced.score(plain)["parts"]["numbers"], priced.score(hidden)["parts"]["numbers"]
    assert p == 1 and h >= p
    assert priced.score(by_weights)["parts"]["numbers"] >= p   # a weight times nothing is no way around it


def test_numbers_computed_from_the_weights_are_not_stored(model, nets):
    """What an entry computes from the weights is charged as computation, not as stored numbers, even when it is a
    table of whole numbers that small changes of the weights would not move: only what is the same for every weight
    counts."""
    w = nets["lin-a-s0"][0]
    priced = Verifier(model, "lin-a-s0", PRICED)
    g = Graph(w, 2)
    first = (g.weight("unembed") @ g.weight("hidden") @ g.weight("embed")).argmax(axis=0)   # an answer per token
    entry = g.entry(predict=first[g.token(0)])
    assert lang.stored_numbers(entry) == 0
    assert priced.score(entry)["parts"]["numbers"] == 0


def test_a_check_that_rarely_holds_is_not_a_stored_number(model, nets):
    """A test that holds on few inputs, or only under some weights, changes with them: it is no stored number,
    even when the inputs and weights the count happens to try never make it hold. The count once tried each value
    on 18 inputs under 4 weight sets, and pilot-rnn-g-price-01 was charged for its test of a near tie (true on 3%
    of inputs) and, on the second network, for its tests that four hidden units are all off, which hold only when
    their input weights have one sign."""
    w = nets["lin-a-s0"][0]
    priced = Verifier(model, "lin-a-s0", PRICED)
    g = Graph(w, 2)
    guess = lang.maximum(g.token(0), g.token(1))
    rare = g.entry(predict=lang.cond(g.token(0) == 37, g.token(1), guess))        # holds on 1 input in 64
    assert lang.stored_numbers(rare) == 1 and priced.score(rare)["parts"]["numbers"] == 1
    g = Graph(w, 2)
    guess = lang.maximum(g.token(0), g.token(1))
    signs = (g.weight("embed")[0, 5] > 0.0) & (g.weight("embed")[0, 0] > 0.0)    # false for this network's weights
    skip = g.entry(predict=lang.cond(signs, guess, guess))                         # a shortcut taken when it holds
    assert lang.stored_numbers(skip) == 1 and priced.score(skip)["parts"]["numbers"] == 1
