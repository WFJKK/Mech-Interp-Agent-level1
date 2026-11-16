"""The second model, lin-b: a network of the same design as lin-a, trained afresh (withheld/lin-b/train.py),
asked about inputs of 1 to 8 tokens; and lin-b-4, its networks asked only about inputs of exactly four tokens,
the inputs they were trained on."""

import numpy as np
import pytest

from level1 import hygiene, lang, loop, prompt, run_config
from level1.networks import load_model
from level1.verifier import Verifier

CFG = {"reward": {"name": "r_alpha", "alpha": 10.0}}


@pytest.fixture(scope="module")
def model_b():
    return load_model("models/lin-b")


def test_its_sizes_its_data_set_and_its_networks(model_b):
    assert model_b.n_total == sum(64 ** n for n in range(1, 9)) and not model_b.can_enumerate()
    for nid in ("lin-b-s0", "lin-b-s1"):
        w = model_b.load_weights(nid)  # the file's sha256 is checked against the config
        assert {k: w[k].shape for k in w} == {"embed": (8, 65), "hidden": (8, 8), "unembed": (64, 8)}
        assert not w["embed"][:, 64].any()  # an empty slot adds nothing
    w = model_b.load_weights("lin-b-s0")
    x = lang.draw_inputs(model_b.data_set, model_b.input_space, 4000, np.random.default_rng(2))
    n = (x < 64).sum(axis=1)
    assert set(n.tolist()) == set(range(1, 9)) and all(0.09 < (n == k).mean() < 0.16 for k in range(1, 9))
    assert ((x[:, :-1] == 64) <= (x[:, 1:] == 64)).all()  # tokens first, then empty slots
    _, ops = lang.run_reference(model_b.reference_entry(w), w, 8, x)
    assert ops == 703  # 8 columns of 8, 8 x 8, 64 x 8, 63 comparisons


def test_the_network_itself_and_the_reference_entries(model_b):
    v = Verifier(model_b, "lin-b-s0", {**CFG, "verifier": {"sample_size": 3000}})
    itself = v.score(v.reference, seed=1)
    assert itself["parts"]["fidelity"] == 1.0 and itself["parts"]["cost"] == 1.0 and -0.02 < itself["reward"] < 0
    s = {name: v.score(e, seed=1) for name, e in model_b.reference_entries(v.weights).items()}
    assert all(r["status"] == "scored" for r in s.values())
    assert s["contraction"]["parts"]["fidelity"] == 1.0 and 0.1 < s["contraction"]["reward"] < 0.3
    assert 0.1 < s["target_on_four"]["parts"]["coverage"] < 0.16 and s["target_on_four"]["parts"]["fidelity"] > 0.99
    assert s["per_length"]["reward"] > max(s["target_everywhere"]["reward"], s["contraction"]["reward"], s["target_on_four"]["reward"])


def test_the_agent_is_not_told_what_it_was_trained_on(model_b):
    cfg = run_config.load("config/pilot/lin-b.yaml")
    forbidden = loop.forbidden_strings(cfg, model_b)
    assert {"four tokens", "4 tokens", "exactly four", "trained on", "largest"} <= set(forbidden)
    texts = loop.agent_surface(cfg, model_b)
    texts["prompt"] = prompt.build(cfg, model_b, 1, [], 703, False)
    texts.update({f"model folder {p.name}": p.read_text() for p in model_b.path.iterdir() if p.is_file()})
    hygiene.check(texts, forbidden)
    assert "n_ctx = 8" in texts["prompt"] and "picked from 1 to 8" in texts["prompt"] and "16,384 inputs drawn afresh" in texts["prompt"]
    assert len(hygiene.hits({p.name: p.read_text() for p in model_b.withheld().iterdir() if p.is_file()}, forbidden)) > 5


def test_the_smoke_config_is_the_pilot_made_small():
    pilot, smoke = run_config.load("config/pilot/lin-b.yaml"), run_config.load("config/pilot/lin-b-smoke.yaml")
    assert smoke["rounds_max"] == 1 and smoke["session"] == {"python_calls": 8, "submissions": 2, "max_turns": 40, "seconds": 1200}
    assert all(smoke[k] == pilot[k] for k in set(pilot) - {"run_id", "rounds_max", "session"}) and smoke["run_id"] != pilot["run_id"]
    assert (pilot["model"], pilot["network"], pilot["seed_check"]["network"]) == ("models/lin-b", "lin-b-s0", "lin-b-s1")
    assert pilot["verifier"] == {**pilot["verifier"], "check": "sample", "sample_size": 16384}


# lin-b-4: lin-b's two networks asked only about the inputs they were trained on, four tokens in four slots ----

@pytest.fixture(scope="module")
def model_b4():
    return load_model("models/lin-b-4")


def test_lin_b_4_is_lin_b_on_four_tokens(model_b, model_b4):
    assert model_b4.n_ctx == 4 and model_b4.n_total == 64 ** 4 and not model_b4.can_enumerate()
    x = lang.draw_inputs(model_b4.data_set, model_b4.input_space, 4000, np.random.default_rng(3))
    assert x.shape == (4000, 4) and (x < 64).all() and len(np.unique(x)) == 64  # four tokens, no empty slot
    padded = np.concatenate([x, np.full((len(x), 4), 64)], axis=1)  # the same tokens in the first four of eight slots
    for s in ("s0", "s1"):
        w4, w8 = model_b4.load_weights(f"lin-b-4-{s}"), model_b.load_weights(f"lin-b-{s}")  # sha256 checked
        assert set(w4) == set(w8) and all(np.array_equal(w4[k], w8[k]) for k in w4)  # the same network, byte for byte
        out4, ops4 = lang.run_reference(model_b4.reference_entry(w4), w4, 4, x)
        out8, _ = lang.run_reference(model_b.reference_entry(w8), w8, 8, padded)
        assert ops4 == 671 and np.array_equal(out4, out8)  # 4 columns of 8, 8 x 8, 64 x 8, 63 comparisons
        right = (out4 == x.max(axis=1)).mean()
        assert 0.995 < right < 1.0  # 0.99830 and 0.99785 on every input (withheld/lin-b-4/task.yaml)


def test_lin_b_4_the_network_itself_and_the_reference_entries(model_b4):
    v = Verifier(model_b4, "lin-b-4-s0", {**CFG, "verifier": {"sample_size": 3000}})
    itself = v.score(v.reference, seed=1)
    assert itself["parts"]["fidelity"] == 1.0 and itself["parts"]["cost"] == 1.0 and -0.02 < itself["reward"] < 0
    s = {name: v.score(e, seed=1) for name, e in model_b4.reference_entries(v.weights).items()}
    assert all(r["status"] == "scored" for r in s.values())
    assert s["contraction"]["parts"]["fidelity"] == 1.0 and 0.4 < s["contraction"]["reward"] < 0.6
    assert s["target_everywhere"]["parts"]["fidelity"] > 0.995 and s["target_everywhere"]["reward"] > 0.9


def test_lin_b_4_the_agent_is_told_the_data_set_not_the_task(model_b4):
    """The data set is the inputs the network was trained on, so it says four tokens; the task stays withheld."""
    cfg = run_config.load("config/pilot/lin-b-4.yaml")
    forbidden = loop.forbidden_strings(cfg, model_b4)
    assert {"largest", "max of", "max4", "trained on", "Gross"} <= set(forbidden)
    assert not {"four tokens", "4 tokens", "exactly four"} & set(forbidden)
    texts = loop.agent_surface(cfg, model_b4)
    texts["prompt"] = prompt.build(cfg, model_b4, 1, [], 671, False)
    texts.update({f"model folder {p.name}": p.read_text() for p in model_b4.path.iterdir() if p.is_file()})
    hygiene.check(texts, forbidden)
    assert "n_ctx = 4" in texts["prompt"] and "Four tokens are picked" in texts["prompt"] and "16,384 inputs drawn afresh" in texts["prompt"]
    assert "lin-b" not in texts["prompt"]
    assert len(hygiene.hits({p.name: p.read_text() for p in model_b4.withheld().iterdir() if p.is_file()}, forbidden)) > 3


def test_lin_b_4_the_smoke_config_is_the_pilot_made_small():
    pilot, smoke = run_config.load("config/pilot/lin-b-4.yaml"), run_config.load("config/pilot/lin-b-4-smoke.yaml")
    assert smoke["rounds_max"] == 1 and smoke["session"] == {"python_calls": 8, "submissions": 2, "max_turns": 40, "seconds": 1200}
    assert all(smoke[k] == pilot[k] for k in set(pilot) - {"run_id", "rounds_max", "session"}) and smoke["run_id"] != pilot["run_id"]
    assert (pilot["model"], pilot["network"], pilot["seed_check"]["network"]) == ("models/lin-b-4", "lin-b-4-s0", "lin-b-4-s1")
    assert pilot["verifier"] == {**pilot["verifier"], "check": "sample", "sample_size": 16384}
    lin_b = run_config.load("config/pilot/lin-b.yaml")  # everything else as on lin-b
    assert {k for k in set(pilot) | set(lin_b) if pilot.get(k) != lin_b.get(k)} == {"run_id", "model", "network", "seed_check"}
