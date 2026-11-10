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
