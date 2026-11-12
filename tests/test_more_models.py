"""Six more models: lin-c and lin-d (the linear design of lin-a and lin-b, trained afresh on another task),
and att-a to att-d (one-layer, one-head transformers: two from the proof pipeline's zoo at 2 slots, one at
4, and one trained here at 4 by the same recipe). What they were trained to do is in withheld/<name>/."""

import importlib.util

import numpy as np
import pytest

from level1 import hygiene, lang, loop, prompt, run_config
from level1.networks import load_model
from level1.verifier import Verifier

CFG = {"reward": {"name": "r_alpha", "alpha": 10.0}}
SIZES = {"lin-c": (4, 671), "lin-d": (8, 703), "att-a": (2, 8621), "att-b": (2, 8621), "att-c": (4, 12987), "att-d": (4, 12987)}
LINEAR = {"embed": (8, 65), "hidden": (8, 8), "unembed": (64, 8)}


def attention(n_ctx):
    return {"embed": (65, 32), "pos": (n_ctx, 32), "query": (32, 32), "key": (32, 32), "value": (32, 32),
            "out": (32, 32), "unembed": (32, 64)}


def numpy_forward(name, w, x):
    """The network on inputs x (rows of slots, 64 for empty), outside the entry language, in float64."""
    w = {k: v.astype(np.float64) for k, v in w.items()}
    n = (x != 64).sum(axis=1)
    if name.startswith("lin"):
        return (w["unembed"] @ (w["hidden"] @ w["embed"]))[:, x].sum(axis=2).argmax(axis=0)
    out = np.empty(len(x), np.int64)
    for L in np.unique(n):
        rows = n == L
        res = w["embed"][x[rows, :L]] + w["pos"][None, :L]
        read = res[:, -1]
        q = (read @ w["query"]) / np.sqrt(w["query"].shape[1])
        s = np.einsum("bth,bh->bt", res @ w["key"], q)
        a = np.exp(s - s.max(axis=1, keepdims=True))
        a /= a.sum(axis=1, keepdims=True)
        head = np.einsum("bt,bth->bh", a, res @ w["value"])
        out[rows] = ((read + head @ w["out"]) @ w["unembed"]).argmax(axis=1)
    return out


@pytest.mark.parametrize("name", list(SIZES))
def test_sizes_data_set_networks_and_the_model_file(name):
    m = load_model(f"models/{name}")
    n_ctx, ops = SIZES[name]
    assert m.n_ctx == n_ctx and m.n_total == sum(64 ** k for k in range(1, n_ctx + 1)) and not m.can_enumerate()
    x = lang.draw_inputs(m.data_set, m.input_space, 3000, np.random.default_rng(4))
    lengths = (x != 64).sum(axis=1)
    assert set(lengths.tolist()) == set(range(1, n_ctx + 1)) and ((x[:, :-1] == 64) <= (x[:, 1:] == 64)).all()
    for nid in (f"{name}-s0", f"{name}-s1"):
        w = m.load_weights(nid)  # the file's sha256 is checked against the config
        assert {k: w[k].shape for k in w} == (LINEAR if name.startswith("lin") else attention(n_ctx))
        assert not (w["embed"][:, 64] if name.startswith("lin") else w["embed"][64]).any()  # an empty slot adds nothing
        answers, n_ops = lang.run_reference(m.reference_entry(w), w, n_ctx, x)
        assert n_ops == ops and (np.asarray(answers) == numpy_forward(name, w, x)).all()


@pytest.mark.parametrize("name", list(SIZES))
def test_the_network_itself_and_the_reference_entries(name):
    m = load_model(f"models/{name}")
    v = Verifier(m, f"{name}-s0", {**CFG, "verifier": {"sample_size": 3000}})
    itself = v.score(v.reference, seed=1)
    assert itself["parts"]["fidelity"] == 1.0 and itself["parts"]["cost"] == 1.0 and -0.02 < itself["reward"] < 0
    s = {k: v.score(e, seed=1) for k, e in m.reference_entries(v.weights).items()}
    assert all(r["status"] == "scored" for r in s.values()), {k: r.get("reason") for k, r in s.items()}
    if name in ("att-a", "att-b"):  # right everywhere, at a few operations against thousands
        assert s["target_everywhere"]["parts"]["fidelity"] == 1.0 and s["target_everywhere"]["reward"] > 0.98  # 0.987 at 3,000: the lower bound
    if name in ("att-c", "att-d"):
        assert s["per_length"]["parts"]["fidelity"] == 1.0 or s["per_length"]["parts"]["fidelity"] > 0.999
        assert 0.2 < s["target_on_full"]["parts"]["coverage"] < 0.3 and s["target_on_full"]["parts"]["fidelity"] > 0.999
        assert s["network_then_target"]["parts"]["fidelity"] > 0.999 and 0.15 < s["network_then_target"]["reward"] < 0.3
    if name.startswith("lin"):
        assert s["contraction"]["parts"]["fidelity"] == 1.0 and s["contraction"]["reward"] > 0


@pytest.mark.parametrize("name", list(SIZES))
def test_the_agent_is_not_told_what_it_was_trained_on(name):
    m = load_model(f"models/{name}")
    cfg = run_config.load(f"config/pilot/{name}.yaml")
    forbidden = loop.forbidden_strings(cfg, m)
    assert {"trained on", "Gross", "compact proof"} <= set(forbidden)
    assert ({"smallest", "min of"} if name in ("lin-c", "lin-d", "att-b", "att-d") else {"largest", "max of"}) <= set(forbidden)
    texts = loop.agent_surface(cfg, m)
    texts["prompt"] = prompt.build(cfg, m, 1, [], SIZES[name][1], False)
    texts.update({f"model folder {p.name}": p.read_text() for p in m.path.iterdir() if p.is_file()})
    hygiene.check(texts, forbidden)
    assert f"n_ctx = {m.n_ctx}" in texts["prompt"] and "16,384 inputs drawn afresh" in texts["prompt"]
    assert len(hygiene.hits({p.name: p.read_text() for p in m.withheld().iterdir() if p.is_file()}, forbidden)) > 3


@pytest.mark.parametrize("name", list(SIZES))
def test_the_smoke_config_is_the_pilot_made_small(name):
    pilot, smoke = run_config.load(f"config/pilot/{name}.yaml"), run_config.load(f"config/pilot/{name}-smoke.yaml")
    assert smoke["rounds_max"] == 1 and smoke["session"] == {"python_calls": 8, "submissions": 2, "max_turns": 40, "seconds": 1200}
    assert all(smoke[k] == pilot[k] for k in set(pilot) - {"run_id", "rounds_max", "session"}) and smoke["run_id"] != pilot["run_id"]
    assert (pilot["model"], pilot["network"], pilot["seed_check"]["network"]) == (f"models/{name}", f"{name}-s0", f"{name}-s1")
    assert pilot["verifier"]["check"] == "sample" and pilot["verifier"]["sample_size"] == 16384


def test_the_transformer_trainer_gets_its_gradients_right():
    """withheld/att-d/train.py writes the backward pass by hand: check it against finite differences."""
    spec = importlib.util.spec_from_file_location("att_d_train", "withheld/att-d/train.py")
    tr = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tr)
    rng = np.random.default_rng(3)
    shapes = {"E": (5, 4), "P": (3, 4), "WQ": (4, 3), "WK": (4, 3), "WV": (4, 3), "WO": (3, 4), "U": (4, 5)}
    P = tr.init(rng, shapes, np.float64)
    tok = rng.integers(0, 5, (6, 3))
    y = tr.target(tok)
    _, G = tr.loss_and_grads(P, tok, y)
    for k in tr.NAMES:
        num = np.zeros_like(P[k])
        for idx in np.ndindex(P[k].shape):
            old = P[k][idx]
            P[k][idx] = old + 1e-6
            up, _ = tr.loss_and_grads(P, tok, y)
            P[k][idx] = old - 1e-6
            down, _ = tr.loss_and_grads(P, tok, y)
            P[k][idx] = old
            num[idx] = (up - down) / 2e-6
        assert np.abs(num - G[k]).max() <= 1e-5 * max(np.abs(num).max(), 1e-12), k
