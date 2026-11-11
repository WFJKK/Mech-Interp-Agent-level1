"""The second model, lin-b: a network of the same design as lin-a, trained afresh (withheld/lin-b/train.py),
asked about inputs of 1 to 8 tokens."""

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
