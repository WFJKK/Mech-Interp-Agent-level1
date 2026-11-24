"""Pooling an entry's samples (level1/pooling.py): the arithmetic, and which submissions stand."""

import numpy as np
import pytest

from level1 import lang, pooling, regions, rewards
from level1.networks import load_model
from level1.verifier import Verifier

REWARD = {"name": "r_alpha", "alpha": 50.0}


@pytest.fixture(scope="module")
def rnn_a():
    model = load_model("models/rnn-a")
    weights = model.load_weights("rnn-a-s2")
    g = lang.Graph(weights, model.n_ctx)
    shifted = g.entry(interpretation=(g.tokens() + g.const([0.05, 0.0])).argmin(), region=g.token(0) > -1.0)
    return model, shifted


def test_pooled_parts_are_those_of_one_sample_made_of_all_of_them_but_the_bound(rnn_a):
    """Counts, coverage, fidelity and cost are those of the union of the samples; the fidelity's confidence bound
    is that of one sample (level1/pooling.py)."""
    model, entry = rnn_a
    v = Verifier(model, "rnn-a-s2", {"reward": REWARD, "verifier": {"sample_size": 700, "workload_size": 700}})
    first, second = v.score(entry, seed=11), v.score(entry, seed=12)
    pooled = pooling.pooled(second, [pooling.draw(first)], REWARD)
    x1, a1, ops, _ = v.inputs(11)
    x2, a2, _, _ = v.inputs(12)
    union = regions.measure(entry, v.weights, model.n_ctx, np.concatenate([x1, x2]), np.concatenate([a1, a2]), ops,
                            how={"mode": "sample"}, verifier_cfg=v.cfg, limits=v.limits)
    for key in ("n_inputs", "n_in_region", "n_agree", "coverage", "fidelity", "r", "r_rule", "b", "cost"):
        assert pooled["parts"][key] == pytest.approx(union[key], rel=1e-12), key
    one = union["n_in_region"] / 2
    assert pooled["parts"]["fidelity_low"] == pytest.approx(lang.wilson_lower(union["fidelity"] * one, one), rel=1e-12)
    assert pooled["parts"]["fidelity_low"] < union["fidelity_low"]  # not the narrower bound of 1,400 inputs
    assert pooled["reward"] == pytest.approx(rewards.compute(pooled["parts"], REWARD)["reward"], rel=1e-12)
    assert pooled["pooled"] == {"samples": 2, "n_inputs": 1400, "seeds": [12, 11]}
    assert pooled["draw"]["parts"] == second["parts"] and pooled["draw"]["reward"] == second["reward"]


def test_without_a_workload_each_scoring_spreads_its_own_build(rnn_a):
    model, entry = rnn_a
    v = Verifier(model, "rnn-a-s2", {"reward": REWARD, "verifier": {"sample_size": 500}})
    first, second = v.score(entry, seed=1), v.score(entry, seed=2)
    pooled = pooling.pooled(second, [pooling.draw(first)], REWARD)
    assert pooled["parts"]["b"] == pytest.approx(first["parts"]["b"]) == pytest.approx(second["parts"]["b"])
    p = pooled["parts"]
    assert p["cost"] == pytest.approx(p["r"] + (p["r_rule"] + p["b"]) / p["coverage"])


def test_one_sample_scored_again_is_not_counted_twice(rnn_a):
    model, entry = rnn_a
    v = Verifier(model, "rnn-a-s2", {"reward": REWARD, "verifier": {"sample_size": 400}})
    first, again = v.score(entry, seed=5), v.score(entry, seed=5)
    assert pooling.pooled(again, [pooling.draw(first)], REWARD) is again
    third = v.score(entry, seed=6)
    both = pooling.pooled(third, [pooling.draw(first), pooling.draw(again)], REWARD)
    assert both["pooled"]["samples"] == 2 and both["parts"]["n_inputs"] == 800


def test_only_the_latest_pooled_submission_stands(rnn_a):
    model, entry = rnn_a
    v = Verifier(model, "rnn-a-s2", {"reward": REWARD, "verifier": {"sample_size": 300}})
    a, b = v.score(entry, seed=21), v.score(entry, seed=22)
    c = pooling.pooled(v.score(entry, seed=23), [pooling.draw(a), pooling.draw(b)], REWARD)
    b_pooled = pooling.pooled(b, [pooling.draw(a)], REWARD)
    receipts = [a, b_pooled, c]
    assert pooling.superseded(receipts) == {0: 2, 1: 2}
    assert pooling.standing(receipts) == [c]
    # records made before pooling carry no 'pooled': every submission stands, and the best is taken, as it was
    assert pooling.superseded([a, b, v.score(entry, seed=24)]) == {} and len(pooling.standing([a, b])) == 2


def test_a_run_made_before_pooling_does_not_pool(tmp_path, rnn_a):
    """A session reads pool_samples from its run's config; a run started before carries none."""
    from level1 import run_config
    from level1.session import Session

    cfg = run_config.load("config/pilot/rnn-a-smoke.yaml", {"runs_dir": str(tmp_path), "backend": "fake",
                                                             "sandbox": "process"})
    assert cfg["pool_samples"] is True
    new = Session.create(tmp_path / "new", cfg, cfg["network"], tmp_path, draws=[])
    old_cfg = {k: v for k, v in cfg.items() if k != "pool_samples"}
    old = Session.create(tmp_path / "old", old_cfg, old_cfg["network"], tmp_path, draws=[])
    assert new.state["pool_samples"] is True and "pool_samples" not in old.state
