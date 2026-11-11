"""The verifier on the pilot's two networks: the reference, the parts, the reward, what is refused."""

import numpy as np
import pytest

from level1 import lang, rewards
from level1.lang import Graph
from level1.verifier import Verifier, wilson_lower

CFG = {"reward": {"name": "r_alpha", "alpha": 10.0}}


def test_the_network_as_an_entry_is_the_network(model, nets):
    for nid, (w, ref, x, answers, ops) in nets.items():
        e, h, u = (w[k].astype(np.float64) for k in ("embed", "hidden", "unembed"))
        direct = (u @ (h @ (e[:, x[:, 0]] + e[:, x[:, 1]]))).argmax(axis=0)
        assert (answers == direct).all()
        assert ops == 655  # two columns of 8, 8 * 8, 64 * 8, 63 comparisons: the proof pipeline's forward pass
    assert model.n_total == 4096


def test_the_network_itself_scores_zero(model, nets):
    for nid, (w, ref, *_rest) in nets.items():
        receipt = Verifier(model, nid, CFG).score(ref)
        p = receipt["parts"]
        assert receipt["status"] == "scored" and receipt["inputs"] == {"mode": "all", "n": 4096, "seed": None}
        assert (p["coverage"], p["fidelity"], p["r"], p["r_rule"], p["b"]) == (1.0, 1.0, 1.0, 0.0, 0.0)
        assert receipt["reward"] == 0.0


def test_parts_of_the_withheld_references(model, nets):
    w = nets["lin-a-s0"][0]
    refs = model.reference_entries(w)
    v = Verifier(model, "lin-a-s0", CFG)
    target = v.score(refs["target"])["parts"]
    assert target["n_agree"] == 4095 and target["predict_ops"] == 4096 and target["build_ops"] == 0
    assert v.score(refs["target"])["reward"] == pytest.approx(1 - 1 / 655 - 10 / 4096)
    corner = v.score(refs["target_outside_corner"])["parts"]
    assert corner["n_in_region"] == 4096 - 9 and corner["fidelity"] == 1.0
    assert corner["rule_ops"] == 2 * 4096 and corner["predict_ops"] == 0  # the rule already computed the answer
    contracted = v.score(refs["contracted"])["parts"]
    assert contracted["fidelity"] == 1.0 and contracted["build_ops"] == 8 * 8 * 64 + 64 * 8 * 64
    assert contracted["predict_ops"] == 4096 * (2 * 64 + 63)
    table = v.score(refs["answer_table"])
    assert table["parts"]["fidelity"] == 1.0 and table["parts"]["b"] > 0.2
    mixed = v.score(refs["target_with_network_in_corner"])["parts"]
    assert mixed["fidelity"] == 1.0 and mixed["predict_ops"] == 3 * 4096 + 9 * 655


def test_an_empty_region_and_a_partial_one(model, nets):
    w = nets["lin-a-s0"][0]
    v = Verifier(model, "lin-a-s0", CFG)
    g = Graph(w, 2)
    nothing = v.score(g.entry(predict=g.token(0), rule=g.token(0) > 99))
    assert nothing["parts"]["coverage"] == 0 and nothing["parts"]["fidelity"] is None
    assert nothing["reward"] == pytest.approx(-1 / 655)  # the rule's comparison on every input, nothing gained
    half = v.score(g.entry(predict=lang.maximum(g.token(0), g.token(1)), rule=g.token(0) >= 32))
    assert half["parts"]["coverage"] == 0.5 and half["parts"]["fidelity"] == 1.0
    assert half["reward"] == pytest.approx(0.5 * (1 - 1 / 655) - 1 / 655)


def test_a_prediction_outside_the_outputs_is_a_wrong_answer(model, nets):
    w = nets["lin-a-s0"][0]
    g = Graph(w, 2)
    receipt = Verifier(model, "lin-a-s0", CFG).score(g.entry(predict=g.token(0) + 1000))
    assert receipt["status"] == "scored" and receipt["parts"]["n_agree"] == 0


def test_refusals_say_why(model, nets):
    w = nets["lin-a-s0"][0]
    v = Verifier(model, "lin-a-s0", {**CFG, "entry": {"max_nodes": 6, "max_const_scalars": 16}})
    assert "an entry is" in v.score({"nodes": []})["reason"]
    g = Graph(w, 2)
    assert "constants" in v.score(g.entry(predict=g.const(list(range(64)))[g.token(0)]))["reason"]
    g = Graph(w, 2)
    assert "nodes" in v.score(g.entry(predict=(((g.token(0) + 1) * 2 + g.token(1)) % 64) // 1))["reason"]
    g = Graph(w, 2)
    bad = v.score(g.entry(predict=g.weight("embed")[g.token(0), 0].argmax()))
    assert bad["status"] == "refused" and "out of range" in bad["reason"] and bad["parts"] is None
    g = Graph(w, 2)
    slow = Verifier(model, "lin-a-s0", {**CFG, "verifier": {"time_limit_s": -1}}).score(g.entry(predict=g.token(0)))
    assert slow["status"] == "refused" and "out of time" in slow["reason"]


def test_what_the_verifier_cannot_even_read_is_refused_and_marked(model, nets):
    v = Verifier(model, "lin-a-s0", CFG)
    odd = {"format": lang.FORMAT, "rule": None, "predict": 0, "nodes": [{"op": "token", "i": float("nan")}]}
    receipt = v.score(odd)
    assert receipt["status"] == "refused" and receipt["reason"] == "the verifier could not read or run this entry (ValueError)"
    assert receipt["verifier_error"].startswith("ValueError: ")  # the detail is kept on the verifier's side
    deep = {"format": lang.FORMAT, "rule": None, "predict": 0, "nodes": [{"op": "const", "dtype": "int", "value": 10**400}]}
    assert v.score(deep)["status"] == "refused"


def test_sampling_above_the_enumeration_limit(model, nets):
    w = nets["lin-a-s0"][0]
    cfg = {**CFG, "verifier": {"enumerate_up_to": 100, "sample_size": 500}}
    v = Verifier(model, "lin-a-s0", cfg)
    target = model.reference_entries(w)["target"]
    a, b = v.score(target, seed=7), v.score(target, seed=7)
    assert a["inputs"] == {"mode": "sample", "n": 500, "seed": 7} and a["parts"] == b["parts"]
    assert a["parts"]["n_inputs"] == 500 and v.check == "sample" and v.n_inputs() == 500
    assert a["parts"]["fidelity_low"] < a["parts"]["fidelity"] <= 1.0
    assert a["reward_detail"]["fidelity_used"] == a["parts"]["fidelity_low"]  # luck in the sample does not pay
    fresh = v.score(target)
    assert fresh["inputs"]["seed"] not in (None, 7)
    g = Graph(w, 2)
    table = v.score(g.entry(predict=(g.weight("unembed") @ (g.weight("hidden") @ g.weight("embed")))[0, 0].argmax()))
    assert table["parts"]["b"] == pytest.approx(36864 / (500 * 655))  # the build is spread over the inputs scored


def test_how_the_verifier_checks_is_a_setting(model, model4, nets):
    """Every input or a sample: verifier.check says which, and auto decides by the data set."""
    assert Verifier(model, "lin-a-s0", CFG).check == "all"                                    # auto: 4,096 inputs can be listed
    forced = Verifier(model, "lin-a-s0", {**CFG, "verifier": {"check": "sample", "sample_size": 300}})
    assert forced.check == "sample" and forced.n_inputs() == 300 and forced.score(forced.reference)["inputs"]["mode"] == "sample"
    listed = Verifier(model, "lin-a-s0", {**CFG, "verifier": {"check": "all", "enumerate_up_to": 10}})
    assert listed.check == "all" and listed.n_inputs() == 4096                                # asked for: the limit does not apply
    assert Verifier(model4, "lin-a-1to4-s0", CFG).check == "sample"                           # auto: its inputs are drawn
    with pytest.raises(RuntimeError, match="cannot be checked on every input"):
        Verifier(model4, "lin-a-1to4-s0", {**CFG, "verifier": {"check": "all"}})
    with pytest.raises(RuntimeError, match="verifier.check must be one of"):
        Verifier(model, "lin-a-s0", {**CFG, "verifier": {"check": "some"}})


def test_one_cost_and_the_reward_written_with_it(model, nets):
    """cost is the entry's whole work per input of its region; coverage * [(1 - cost) - alpha * (1 - fidelity)]
    is the reward, whatever the split into build, region test and interpretation."""
    w = nets["lin-a-s0"][0]
    v = Verifier(model, "lin-a-s0", CFG)
    for name, entry in {"the network": v.reference, **model.reference_entries(w)}.items():
        r = v.score(entry)
        p = r["parts"]
        assert p["cost"] == pytest.approx(p["r"] + (p["r_rule"] + p["b"]) / p["coverage"]), name
        assert r["reward"] == pytest.approx(p["coverage"] * ((1 - p["cost"]) - 10 * (1 - p["fidelity"]))), name
        assert rewards.cost_of(p) == p["cost"] and rewards.cost_of({k: x for k, x in p.items() if k != "cost"}) == pytest.approx(p["cost"])
    g = Graph(w, 2)
    empty = v.score(g.entry(interpretation=g.token(0), region=g.token(0) > 99))["parts"]
    assert empty["cost"] is None and rewards.cost_of(empty) is None


def test_the_network_on_one_to_four_tokens(model, model4, nets):
    """The same weights with a zero column for an empty slot: 671 operations per forward pass, the same outputs
    as lin-a wherever an input has two tokens, and a data set drawn with every number of tokens equally likely."""
    assert model4.n_total == 64 + 64**2 + 64**3 + 64**4 and not model4.can_enumerate()
    w4 = model4.load_weights("lin-a-1to4-s0")
    assert not w4["embed"][:, 64].any() and (w4["embed"][:, :64] == nets["lin-a-s0"][0]["embed"]).all()
    ref = model4.reference_entry(w4)
    x = lang.draw_inputs(model4.data_set, model4.input_space, 2000, np.random.default_rng(5))
    answers, ops = lang.run_reference(ref, w4, 4, x)
    assert ops == 671
    n_tokens = (x < 64).sum(axis=1)
    assert set(n_tokens.tolist()) == {1, 2, 3, 4} and all(0.2 < (n_tokens == k).mean() < 0.3 for k in (1, 2, 3, 4))
    assert ((x[:, :-1] == 64) <= (x[:, 1:] == 64)).all() and x.min() >= 0 and x.max() == 64   # tokens first, then empty slots
    two = n_tokens == 2
    lin_a = {(int(a), int(b)): int(o) for (a, b), o in zip(nets["lin-a-s0"][2], nets["lin-a-s0"][3])}
    assert all(lin_a[(int(r[0]), int(r[1]))] == int(o) for r, o in zip(x[two], answers[two]))
    v = Verifier(model4, "lin-a-1to4-s0", {**CFG, "verifier": {"sample_size": 3000}})
    itself = v.score(v.reference, seed=1)
    assert itself["parts"]["fidelity"] == 1.0 and itself["parts"]["cost"] == 1.0 and -0.02 < itself["reward"] < 0   # the lower bound's small charge
    scores = {name: v.score(e, seed=1) for name, e in model4.reference_entries(w4).items()}
    assert scores["target_everywhere"]["reward"] < 0 < scores["target_on_two"]["reward"] < scores["target_up_to_two"]["reward"]
    assert scores["target_up_to_two"]["reward"] < scores["contraction"]["reward"] < scores["target_then_contraction"]["reward"]
    assert scores["contraction"]["parts"]["fidelity"] == 1.0 and 0.2 < scores["target_on_two"]["parts"]["coverage"] < 0.3


def test_wilson_lower_bound():
    assert wilson_lower(0, 0) == 0.0
    assert 0.99 < wilson_lower(1000, 1000) < 1.0
    assert wilson_lower(500, 1000) == pytest.approx(0.469, abs=0.002)


def test_reward_functions():
    parts = {"coverage": 0.8, "fidelity": 0.95, "fidelity_low": 0.95, "r": 0.25, "r_rule": 0.01, "b": 0.02}
    out = rewards.compute(parts, {"name": "r_alpha", "alpha": 10})
    assert out["saved"] == pytest.approx(0.8 * 0.75 - 0.03) and out["errors"] == pytest.approx(0.04)
    assert out["reward"] == pytest.approx(0.57 - 0.4)
    assert rewards.compute(parts, {"name": "r_alpha", "alpha": 1})["reward"] == pytest.approx(0.57 - 0.04)
    honest = rewards.compute(parts, {"name": "r_alpha_claimed", "alpha": 10, "penalty": 2}, claimed=0.95)
    modest = rewards.compute(parts, {"name": "r_alpha_claimed", "alpha": 10, "penalty": 2}, claimed=0.90)
    bold = rewards.compute(parts, {"name": "r_alpha_claimed", "alpha": 10, "penalty": 2}, claimed=1.0)
    assert honest["reward"] == pytest.approx(out["reward"])
    assert modest["reward"] < honest["reward"] and bold["reward"] < honest["reward"]  # claiming the truth is best
    assert bold["fidelity_used"] == pytest.approx(0.95 - 2 * 0.05)
    with pytest.raises(ValueError, match="no reward named"):
        rewards.compute(parts, {"name": "nothing"})
    with pytest.raises(ValueError, match="does not take \\['alhpa'\\]"):  # a typo is not ignored in silence
        rewards.compute(parts, {"name": "r_alpha", "alhpa": 1})
    assert rewards.parameters("r_alpha") == {"alpha": 10.0, "lower_bound": True}
    assert rewards.parameters("r_alpha_claimed") == {"alpha": 10.0, "penalty": 2.0}
    assert "alpha = 3" in rewards.formula({"name": "r_alpha", "alpha": 3}) and "at 3 forward passes" in rewards.explanation({"name": "r_alpha", "alpha": 3})
    assert "measured - 2 * (claimed - measured)" in rewards.formula({"name": "r_alpha_claimed"})
    for name in rewards.REWARDS:  # every reward has its two texts for the prompt, and gives the network itself 0
        assert rewards.formula({"name": name}) and rewards.explanation({"name": name})
        itself = {"coverage": 1.0, "fidelity": 1.0, "fidelity_low": 1.0, "r": 1.0, "r_rule": 0.0, "b": 0.0}
        assert rewards.compute(itself, {"name": name}, claimed=1.0)["reward"] == 0.0
