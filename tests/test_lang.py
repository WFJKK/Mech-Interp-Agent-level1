"""The entry language: values, prices, laziness, what is charged where, and what is refused."""

import json
import time

import numpy as np
import pytest

from level1 import lang
from level1.lang import EntryError, Evaluator, Graph


def run(entry, weights, x, n_ctx=2):
    ev = Evaluator(entry, weights, n_ctx)
    inside, answer, c_rule, c_pred = ev.run(x)
    return inside, answer, c_rule, c_pred, ev.build_cost


def value_and_cost(build, weights, x=(1, 2), n_ctx=2):
    """Evaluate one expression of a graph: (value, operations per input, build operations)."""
    g = Graph(weights, n_ctx, example=x)
    e = build(g)
    ev = Evaluator(_as_int_entry(g, e), weights, n_ctx)
    memo, counter = {}, [0]
    v = ev._eval(e.id, memo, counter, np.asarray(x, dtype=np.int64))
    return v, counter[0], ev.build_cost


def _as_int_entry(g, e):
    # an entry whose structure is valid whatever the expression's type: predict is a constant placed last
    nodes = [dict(n) for n in g.nodes[: e.id + 1]] + [{"op": "const", "value": 0, "dtype": "int"}]
    return {"format": lang.FORMAT, "nodes": nodes, "rule": None, "predict": len(nodes) - 1}


def test_elementwise_prices_and_values(toy):
    A, v = toy["A"], toy["v"]
    val, per_input, build = value_and_cost(lambda g: g.weight("A") * 2.0 + g.weight("v"), toy)
    assert np.allclose(val, A * 2.0 + v) and per_input == 0 and build == 24  # 12 products, 12 sums, all ahead
    val, per_input, build = value_and_cost(lambda g: g.weight("v") + g.token(0), toy)
    assert np.allclose(val, v + 1) and per_input == 4 and build == 0
    val, per_input, _ = value_and_cost(lambda g: abs(-g.token(0) - g.token(1)) // 2, toy)
    assert int(val) == 1 and per_input == 4  # neg, sub, abs, floordiv
    val, per_input, _ = value_and_cost(lambda g: (g.token(0) < g.token(1)) & ~(g.token(0) == 3), toy)
    assert bool(val) and per_input == 4  # lt, eq, not, and


def test_matmul_counts_multiply_adds(toy):
    A, B, v = toy["A"], toy["B"], toy["v"]
    val, _, build = value_and_cost(lambda g: g.weight("A") @ g.weight("B"), toy)
    assert np.allclose(val, A @ B) and build == 3 * 4 * 5
    val, _, build = value_and_cost(lambda g: g.weight("A") @ g.weight("v"), toy)
    assert np.allclose(val, A @ v) and build == 3 * 4
    val, _, build = value_and_cost(lambda g: g.weight("v") @ g.weight("v"), toy)
    assert np.isclose(val, v @ v) and build == 4


def test_reductions_cost_n_minus_one(toy):
    A = toy["A"]
    for name, fn in (("sum", np.sum), ("max", np.max), ("min", np.min), ("argmax", np.argmax), ("argmin", np.argmin)):
        val, _, build = value_and_cost(lambda g: getattr(g.weight("A"), name)(), toy)
        assert np.allclose(val, fn(A)) and build == 11
        val, _, build = value_and_cost(lambda g: getattr(g.weight("A"), name)(axis=1), toy)
        assert np.allclose(val, fn(A, axis=1)) and build == 3 * 3
        val, _, build = value_and_cost(lambda g: getattr(g.weight("A"), name)(axis=-2), toy)
        assert np.allclose(val, fn(A, axis=0)) and build == 4 * 2
    val, _, build = value_and_cost(lambda g: (g.weight("A") > 0).any(axis=0), toy)
    assert (val == (A > 0).any(axis=0)).all() and build == 12 + 4 * 2


def test_along_an_axis(toy):
    A = toy["A"]
    val, _, build = value_and_cost(lambda g: lang.cumsum(g.weight("A"), axis=1), toy)
    assert np.allclose(val, np.cumsum(A, axis=1)) and build == 3 * 3
    val, _, build = value_and_cost(lambda g: lang.sort(g.weight("A")), toy)
    assert np.allclose(val, np.sort(A, axis=1)) and build == 3 * 4 * 2
    val, _, build = value_and_cost(lambda g: lang.argsort(g.weight("A"), axis=0), toy)
    assert (val == np.argsort(A, axis=0, kind="stable")).all() and build == 4 * 3 * 2
    val, _, build = value_and_cost(lambda g: lang.softmax(g.weight("A")), toy)
    assert np.allclose(val.sum(axis=1), 1) and build == 3 * 12


def test_reading_is_charged_per_entry_read(toy):
    T = toy["T"]
    val, per_input, build = value_and_cost(lambda g: g.weight("T")[g.token(0), g.token(1)], toy)
    assert np.isclose(val, T[1, 2]) and per_input == 1 and build == 0  # one number read
    val, per_input, _ = value_and_cost(lambda g: g.weight("T")[:, g.token(0)], toy)
    assert np.allclose(val, T[:, 1]) and per_input == 6  # one column
    val, per_input, _ = value_and_cost(lambda g: g.weight("T")[g.token(1)], toy)
    assert np.allclose(val, T[2]) and per_input == 6
    val, per_input, _ = value_and_cost(lambda g: g.weight("T")[1:4, g.token(1)], toy)
    assert np.allclose(val, T[1:4, 2]) and per_input == 3  # the static slice is free, three numbers read
    val, per_input, _ = value_and_cost(lambda g: g.weight("T")[3, g.token(1)], toy)
    assert np.isclose(val, T[3, 2]) and per_input == 1
    val, per_input, _ = value_and_cost(lambda g: lang.take(g.weight("T"), g.tokens(), axis=1), toy)
    assert np.allclose(val, T[:, [1, 2]]) and per_input == 12
    val, per_input, _ = value_and_cost(lambda g: g.weight("T")[g.tokens()], toy)
    assert np.allclose(val, T[[1, 2]]) and per_input == 12
    val, per_input, _ = value_and_cost(lambda g: lang.take_sum(g.weight("T"), g.tokens(), axis=1), toy)
    assert np.allclose(val, T[:, 1] + T[:, 2]) and per_input == 12  # a matrix applied to a sum of one-hot vectors
    val, per_input, _ = value_and_cost(lambda g: g.weight("T")[g.token(0), :][g.token(1)], toy)
    assert np.isclose(val, T[1, 2]) and per_input == 7  # a row read, then one number of it


def test_moving_data_is_free(toy):
    A = toy["A"]
    for build_fn, expect in (
        (lambda g: g.weight("A").T, A.T),
        (lambda g: g.weight("A").reshape(4, 3), A.reshape(4, 3)),
        (lambda g: g.weight("A").reshape([2, -1]), A.reshape(2, -1)),
        (lambda g: g.weight("A")[1], A[1]),
        (lambda g: g.weight("A")[:, ::2], A[:, ::2]),
        (lambda g: g.weight("A")[None, :, 1], A[None, :, 1]),
        (lambda g: g.weight("A")[:, None], A[:, None]),
        (lambda g: lang.concat([g.weight("A"), g.weight("A")], axis=1), np.concatenate([A, A], axis=1)),
        (lambda g: lang.stack([g.weight("v"), g.weight("v")]), np.stack([toy["v"], toy["v"]])),
        (lambda g: lang.transpose(g.weight("A"), [1, 0]), A.T),
        (lambda g: g.arange(5), np.arange(5)),
    ):
        val, per_input, build = value_and_cost(build_fn, toy)
        assert val.shape == expect.shape and np.allclose(val, expect) and per_input == 0 and build == 0


def test_an_entry_can_be_written_with_interpretation_and_region(toy):
    g = Graph(toy, 2)
    a, b = g.token(0), g.token(1)
    one = g.entry(interpretation=lang.maximum(a, b), region=a < 3)
    g = Graph(toy, 2)
    a, b = g.token(0), g.token(1)
    other = g.entry(predict=lang.maximum(a, b), rule=a < 3)
    assert one == other and set(one) == {"format", "nodes", "rule", "predict"}        # the stored entry is the same
    assert "<- region" in lang.show(one) and "<- interpretation" in lang.show(one)
    with pytest.raises(TypeError, match="given once"):
        g.entry(predict=a, interpretation=a)
    with pytest.raises(TypeError, match="needs its interpretation"):
        g.entry(region=a < 3)
    with pytest.raises(TypeError, match="region once"):
        g.entry(interpretation=a, region=a < 3, rule=a < 2)
    assert Graph(toy, 2, example=[1, 2]).token(1).value == 2
    with pytest.raises(ValueError, match="one whole number per position of the input: 3 here"):
        Graph(toy, 3, example=[1, 2])


def test_data_sets_that_are_listed_and_data_sets_that_are_drawn():
    rng = np.random.default_rng(3)
    assert lang.count_inputs(None, [64, 64]) == 4096 and lang.count_inputs({"kind": "all"}, [3, 5]) == 15
    x = lang.draw_inputs(None, [3, 5], 400, rng)
    assert x.shape == (400, 2) and x[:, 0].max() == 2 and x[:, 1].max() == 4
    ds = {"kind": "lengths", "lengths": [1, 3], "vocab": 10, "empty": 10}
    assert lang.count_inputs(ds, [11, 11, 11]) == 10 + 1000
    y = lang.draw_inputs(ds, [11, 11, 11], 600, rng)
    n = (y < 10).sum(axis=1)
    assert set(n.tolist()) == {1, 3} and 0.4 < (n == 1).mean() < 0.6 and (y[n == 1][:, 1:] == 10).all()
    with pytest.raises(ValueError, match="kind"):
        lang.count_inputs({"kind": "weights"}, [3])
    with pytest.raises(ValueError, match="fit the model's slots"):
        lang.draw_inputs({"kind": "lengths", "lengths": [1, 4], "vocab": 10, "empty": 10}, [11, 11], 5, rng)
    assert lang.wilson_lower(0, 0) == 0.0 and 0.99 < lang.wilson_lower(1000, 1000) < 1.0


def test_a_cast_is_a_computation(toy):
    A = toy["A"]
    val, _, build = value_and_cost(lambda g: lang.cast(g.weight("A"), "int"), toy)
    assert (val == A.astype(np.int64)).all() and build == 12  # truncation
    val, per_input, _ = value_and_cost(lambda g: lang.cast(g.token(0) - g.token(1), bool), toy)
    assert bool(val) and per_input == 2  # a test against zero is not free


def test_a_cast_that_differs_between_machines_is_refused(toy):
    g = Graph(toy, 2, example=(3, 0))
    entry = g.entry(predict=lang.cast(lang.log(g.token(0) * 1.0), "int"))  # log(0) is minus infinity
    with pytest.raises(EntryError, match="no whole-number value"):
        run(entry, toy, (0, 0))
    assert run(entry, toy, (3, 0))[1] == 1


def test_what_an_entry_holds_at_once_is_bounded(toy, monkeypatch):
    monkeypatch.setattr(lang, "MAX_HELD", 100)
    g = Graph(toy, 2)
    big = g.arange(60)
    entry = g.entry(predict=((big + 1) * 2).argmax())
    with pytest.raises(EntryError, match="at once"):
        run(entry, toy, (0, 0))
    g = Graph(toy, 2)
    per_input = g.entry(predict=((g.arange(30) + g.token(0)) * 2).argmax())
    ev = Evaluator(per_input, toy, 2)
    for x in ((0, 0), (1, 1), (2, 2), (3, 3)):  # each input's values are let go before the next
        assert ev.run(x)[1] == 29


def test_cond_charges_only_the_chosen_side_and_where_both(toy):
    def entry(use_cond):
        g = Graph(toy, 2)
        a, b = g.token(0), g.token(1)
        cheap = a + 1
        dear = (g.weight("T")[:, a] + g.weight("T")[:, b]).argmax()
        pick = lang.cond if use_cond else lang.where
        return g.entry(predict=pick(a < 3, cheap, dear))

    dear_ops = 6 + 6 + 6 + 5
    assert run(entry(True), toy, (1, 2))[3] == 1 + 1 + 1  # lt, the cheap side, cond
    assert run(entry(True), toy, (4, 2))[3] == 1 + dear_ops + 1
    assert run(entry(False), toy, (1, 2))[3] == 1 + 1 + dear_ops + 1  # both sides, then where
    assert run(entry(True), toy, (1, 2))[1] == 2 and run(entry(False), toy, (1, 2))[1] == 2
    assert run(entry(True), toy, (4, 2))[1] == int((toy["T"][:, 4] + toy["T"][:, 2]).argmax())


def test_build_is_charged_once_and_only_when_needed(toy):
    g = Graph(toy, 2)
    a, b = g.token(0), g.token(1)
    table = (g.weight("A") @ g.weight("B")).argmax(axis=0)  # 60 multiply-adds and 5 * 2 comparisons, ahead
    entry = g.entry(predict=lang.cond(a < 2, a, table[b]), rule=a >= 1)
    ev = Evaluator(entry, toy, 2)
    assert ev.run((0, 0)) == (False, None, 1, 0) and ev.build_cost == 0  # outside the region: nothing predicted
    assert ev.run((1, 0)) == (True, 1, 1, 2) and ev.build_cost == 0  # the cheap side: the table is never built
    inside, answer, c_rule, c_pred = ev.run((3, 4))
    assert (inside, c_rule, c_pred) == (True, 1, 1 + 1 + 1) and ev.build_cost == 70  # lt, one number read, cond
    assert answer == int((toy["A"] @ toy["B"]).argmax(axis=0)[4])
    ev.run((4, 2))
    assert ev.build_cost == 70  # not again


def test_a_value_the_rule_computed_is_not_charged_to_the_prediction(toy):
    g = Graph(toy, 2)
    m = lang.maximum(g.token(0), g.token(1))
    entry = g.entry(predict=m, rule=m >= 2)
    assert run(entry, toy, (3, 1))[2:4] == (2, 0)
    assert run(entry, toy, (0, 1))[2:4] == (2, 0) and run(entry, toy, (0, 1))[0] is False


def test_svd_is_charged_once_for_its_three_parts(toy):
    A = toy["A"]
    g = Graph(toy, 2)
    u, s, vt = lang.svd(g.weight("A"))
    low = (u[:, :1] * s[:1]) @ vt[:1, :]
    entry = g.entry(predict=(low[0, :] + g.token(0)).argmax())
    ev = Evaluator(entry, toy, 2)
    inside, answer, _, c_pred = ev.run((1, 1))
    uu, ss, vv = np.linalg.svd(A, full_matrices=False)
    expect = (uu[:, :1] * ss[:1]) @ vv[:1, :]
    assert answer == int((expect[0] + 1).argmax()) and c_pred == 4 + 3
    assert ev.build_cost == 4 * 3 * 4 * 3 + 3 + 3 * 1 * 4  # the decomposition once, u * s, the product


def test_json_round_trip_and_hash(toy):
    g = Graph(toy, 2)
    a, b = g.token(0), g.token(1)
    entry = g.entry(predict=(g.weight("T")[:, a] * g.const([1.5, 2, 3, 4, 5, 6])).argmax(), rule=(a + b) % 2 == 0)
    again = json.loads(lang.canonical(entry))
    assert again == entry and lang.entry_hash(again) == lang.entry_hash(entry)
    for x in ((0, 0), (1, 3), (5, 5), (2, 3)):
        assert run(entry, toy, x) == run(again, toy, x)
    g2, predict, rule = Graph.load(entry, toy, 2)
    assert g2.entry(predict=predict, rule=rule) == entry
    assert lang.entry_hash({**entry, "rule": None}) != lang.entry_hash(entry)


def test_the_builder_shares_equal_nodes_and_drops_unused_ones(toy):
    g = Graph(toy, 2)
    a = g.token(0)
    unused = g.weight("A") @ g.weight("B")
    assert (a + 1).id == (a + 1).id and unused.id != a.id
    entry = g.entry(predict=(a + 1) * (a + 1))
    assert [n["op"] for n in entry["nodes"]] == ["token", "const", "add", "mul"]


def test_the_builder_shows_mistakes_where_they_are_made(toy):
    g = Graph(toy, 2)
    with pytest.raises(EntryError):
        g.weight("nothing")
    with pytest.raises((EntryError, ValueError)):
        g.weight("A") @ g.weight("A")
    with pytest.raises(TypeError):
        bool(g.token(0) < 1)
    with pytest.raises(TypeError):
        g.weight("A")[..., 0]
    with pytest.raises(IndexError):
        g.weight("A")[0, 0, 0]
    with pytest.raises(ValueError):
        Graph(toy, 2).token(0) + g.token(0)


@pytest.mark.parametrize("bad", [
    None, [], {"format": "x", "nodes": [], "predict": 0},
    {"format": lang.FORMAT, "nodes": [], "rule": None, "predict": 0},
    {"format": lang.FORMAT, "nodes": [{"op": "token", "i": 0}], "rule": None, "predict": 1},
    {"format": lang.FORMAT, "nodes": [{"op": "token", "i": 0}], "rule": None, "predict": True},
    {"format": lang.FORMAT, "nodes": [{"op": "token", "i": 0}], "rule": None, "predict": 0, "note": "x"},
    {"format": lang.FORMAT, "nodes": [{"op": "eval", "args": []}], "rule": None, "predict": 0},
    {"format": lang.FORMAT, "nodes": [{"op": "add", "args": [0, 0]}], "rule": None, "predict": 0},  # itself
    {"format": lang.FORMAT, "nodes": [{"op": "token", "i": 0}, {"op": "add", "args": [0, 2]}, {"op": "token", "i": 1}],
     "rule": None, "predict": 1},  # a later node
    {"format": lang.FORMAT, "nodes": [{"op": "token", "i": 0}, {"op": "add", "args": [0]}], "rule": None, "predict": 1},
    {"format": lang.FORMAT, "nodes": [{"op": "token", "i": 0, "code": "import os"}], "rule": None, "predict": 0},
    {"format": lang.FORMAT, "nodes": [{"op": "const", "value": [[1, 2], [3]], "dtype": "int"}], "rule": None, "predict": 0},
    {"format": lang.FORMAT, "nodes": [{"op": "const", "value": "x", "dtype": "int"}], "rule": None, "predict": 0},
    {"format": lang.FORMAT, "nodes": [{"op": "const", "value": 1, "dtype": "object"}], "rule": None, "predict": 0},
])
def test_what_is_not_an_entry_is_refused(bad):
    with pytest.raises(EntryError):
        lang.check_structure(bad)


def test_size_limits(toy):
    g = Graph(toy, 2)
    big = g.entry(predict=g.const(list(range(300)))[g.token(0)])
    assert lang.check_structure(big) == {"nodes": 3, "const_scalars": 300}
    with pytest.raises(EntryError, match="constants"):
        lang.check_structure(big, max_const_scalars=256)
    with pytest.raises(EntryError, match="nodes"):
        lang.check_structure(big, max_nodes=2)
    assert not json.dumps(big).count("NaN")
    with pytest.raises(EntryError):
        lang.check_structure({"format": lang.FORMAT, "rule": None, "predict": 0,
                              "nodes": [{"op": "const", "value": float("inf"), "dtype": "float"}]})


def test_failures_on_an_input_name_the_node(toy):
    g = Graph(toy, 2)
    entry = g.entry(predict=g.weight("T")[g.token(0) + 3, 0].argmax())
    assert run(entry, toy, (1, 1))[0] is True
    with pytest.raises(EntryError, match=r"node \d+ \(gather\).*out of range"):
        run(entry, toy, (4, 0))
    with pytest.raises(EntryError, match="one whole number"):
        run(g.entry(predict=g.weight("v")[0]), toy, (0, 0))  # a float, not an output
    with pytest.raises(EntryError, match="one boolean"):
        run(g.entry(predict=g.token(0), rule=g.token(1)), toy, (0, 0))
    with pytest.raises(EntryError, match="2 tokens"):
        run(g.entry(predict=g.token(0)), toy, (0, 0, 0))
    raw = {"format": lang.FORMAT, "rule": None, "predict": 1, "nodes": [{"op": "token", "i": 5}, {"op": "neg", "args": [0]}]}
    with pytest.raises(EntryError, match="no position 5"):
        run(raw, toy, (0, 0))
    raw = {"format": lang.FORMAT, "rule": None, "predict": 2,
           "nodes": [{"op": "weight", "name": "A"}, {"op": "weight", "name": "T"}, {"op": "matmul", "args": [0, 1]}]}
    with pytest.raises(EntryError, match=r"node 2 \(matmul\)"):
        run(raw, toy, (0, 0))


def test_values_cannot_grow_without_bound(toy):
    n = 3000
    raw = {"format": lang.FORMAT, "rule": None, "predict": 4, "nodes": [
        {"op": "arange", "n": n}, {"op": "reshape", "args": [0], "shape": [n, 1]}, {"op": "reshape", "args": [0], "shape": [1, n]},
        {"op": "add", "args": [1, 2]}, {"op": "argmax", "args": [3], "axis": None}]}
    with pytest.raises(EntryError, match="above the limit"):
        run(raw, toy, (0, 0))


def test_an_entry_out_of_time_is_stopped(toy):
    g = Graph(toy, 2)
    entry = g.entry(predict=g.token(0) + g.token(1))
    ev = Evaluator(entry, toy, 2, deadline=time.monotonic() - 1)
    with pytest.raises(EntryError, match="out of time"):
        ev.run((0, 0))


def test_all_inputs():
    x = lang.all_inputs([2, 3])
    assert x.tolist() == [[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2]]
