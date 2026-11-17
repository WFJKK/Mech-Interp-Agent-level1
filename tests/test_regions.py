"""Conditional evaluation keeps the existing reward and records its outer scope."""

import copy

import numpy as np
import pytest

from level1 import lang, rewards
from level1.lang import Graph
from level1.verifier import Verifier


def test_assigned_scope_and_candidate_region_have_separate_coverage(model4):
    v = Verifier(model4, "lin-a-1to4-s0", {
        "verifier": {"scope": {"kind": "lengths", "lengths": [2]}},
    })
    whole = v.score(v.reference)
    assert whole["status"] == "scored"
    assert whole["inputs"]["mode"] == "all"
    assert whole["parts"]["n_inputs"] == v.n_inputs() == 4096
    assert whole["parts"]["coverage"] == whole["parts"]["fidelity"] == 1.0
    assert whole["reward"] == 0.0
    assert whole["scope"]["cardinality"] == 4096
    assert whole["scope"]["mass"] == 0.25
    assert whole["scope"]["mass_exact"]
    assert whole["scope"]["workload_size"] == 4096
    g = Graph(v.weights, 4)
    entry = g.entry(predict=model4.module().reference(g, lang), rule=g.token(0) < 32)
    narrower = v.score(entry)
    assert narrower["parts"]["coverage"] == 0.5
    assert narrower["parts"]["n_in_region"] == 2048
    assert narrower["scope"]["mass"] == 0.25
    assert narrower["reward"] == pytest.approx(-1 / 671)


def test_multilength_conditioning_preserves_distribution_not_uniform_strings(model4):
    from level1 import regions
    cfg = {"scope": {"kind": "lengths", "lengths": [1, 3]}, "sample_size": 4000}
    x, how, scope = regions.prepare_inputs(model4.info(), {}, cfg, seed=12)
    lengths = np.sum(x != 64, axis=1)
    assert how["mode"] == "sample"
    assert set(lengths) == {1, 3}
    assert 0.47 < np.mean(lengths == 1) < 0.53
    assert scope["mass"] == 0.5
    assert scope["cardinality"] == 64 + 64**3
    assert scope["workload_size"] == scope["cardinality"]


def finite_info():
    return {"sizes": {"n_ctx": 2}, "input_space": [4, 3], "data_set": {"kind": "all"}}


def predicate(predicate_fn, real=False):
    g = Graph({}, 2, real=real)
    return g.entry(predict=g.const(0), rule=predicate_fn(g))


def test_arbitrary_finite_predicate_has_exact_mass_and_count():
    from level1 import regions
    spec = {"kind": "predicate", "entry": predicate(lambda g: g.token(0) < 2)}
    x, how, meta = regions.prepare_inputs(finite_info(), {}, {"scope": spec}, seed=2)
    assert how["mode"] == "all"
    assert len(x) == 6 and np.all(x[:, 0] < 2)
    assert meta["mass"] == 0.5 and meta["mass_exact"]
    assert meta["cardinality"] == meta["workload_size"] == 6
    assert meta["conditioning"]["rule_ops"] == 12
    g = Graph({}, 2)
    entry = g.entry(predict=g.token(0) + g.token(1))
    truth, ops = lang.run_reference(entry, {}, 2, x)
    parts = regions.measure(entry, {}, 2, x, truth, ops, how=how, scope=meta)
    assert rewards.compute(parts, {"name": "r_alpha"})["reward"] == 0.0


def test_real_scope_requires_a_declared_workload_and_uses_bounded_rejection():
    from level1 import regions
    info = {"sizes": {"n_ctx": 2}, "input_space": ["real", "real"], "data_set": {"kind": "normal"}}
    spec = {"kind": "predicate", "entry": predicate(lambda g: g.token(0) > 0, real=True),
            "mass_samples": 400, "max_draws": 2000}
    with pytest.raises(lang.EntryError, match="workload_size"):
        regions.prepare_inputs(info, {}, {"scope": spec, "sample_size": 100}, seed=3)
    cfg = {"scope": spec, "sample_size": 100, "workload_size": 300}
    x, how, meta = regions.prepare_inputs(info, {}, cfg, seed=3)
    assert how["mode"] == "sample" and len(x) == 100 and np.all(x[:, 0] > 0)
    assert meta["cardinality"] is None and not meta["mass_exact"]
    assert meta["workload_size"] == 300
    assert 0.35 < meta["mass"] < 0.65
    assert meta["mass_estimation"]["n"] == 400
    impossible = {**spec, "entry": predicate(lambda g: g.token(0) < -1e100, real=True),
                  "max_draws": 30, "mass_samples": 10}
    with pytest.raises(lang.EntryError, match="rejection"):
        regions.prepare_inputs(info, {}, {"scope": impossible, "sample_size": 20, "workload_size": 40}, seed=3)


def test_unknown_finite_predicate_count_also_requires_workload():
    from level1 import regions
    spec = {"kind": "predicate", "entry": predicate(lambda g: g.token(0) < 2)}
    with pytest.raises(lang.EntryError, match="workload_size"):
        regions.prepare_inputs(finite_info(), {}, {"scope": spec, "enumerate_up_to": 3}, seed=0)


def test_scope_cannot_change_price_reward_or_reuse_horizon():
    from level1 import regions
    for key, value in (("workload_size", 10**30), ("alpha", 0), ("reward", "r_alpha"), ("name", "arbitrary")):
        with pytest.raises(lang.EntryError, match="scope.*does not take"):
            regions.normalize_scope({"kind": "all", key: value})
    for invalid in (0, -3, True, 2.5):
        with pytest.raises(lang.EntryError, match="workload_size"):
            regions.prepare_inputs(finite_info(), {}, {"scope": {"kind": "all"}, "workload_size": invalid})


def test_scope_ids_are_semantic_not_sampler_settings():
    from level1 import regions
    assert regions.scope_id(None) == regions.scope_id({"kind": "all"}) == "all"
    assert regions.scope_id({"kind": "lengths", "lengths": [3, 1, 1]}) == regions.scope_id({"kind": "lengths", "lengths": [1, 3]})
    p = predicate(lambda g: g.token(0) > 0)
    a = {"kind": "predicate", "entry": p, "max_draws": 100}
    b = {**a, "max_draws": 1000, "mass_samples": 300}
    assert regions.scope_id(a) == regions.scope_id(b)
    other_prediction = copy.deepcopy(p)
    other_prediction["nodes"].append({"op": "weight", "name": "unavailable"})
    other_prediction["predict"] = len(other_prediction["nodes"]) - 1
    assert regions.scope_id(a) == regions.scope_id({"kind": "predicate", "entry": other_prediction})


def test_scope_cannot_enlarge_the_declared_data_set(model4):
    from level1 import regions
    with pytest.raises(lang.EntryError, match="subset"):
        regions.prepare_inputs(model4.info(), {}, {"scope": {"kind": "lengths", "lengths": [5]}})


def test_local_helper_and_verifier_recompute_the_same_regional_measurements(model4):
    from level1 import regions
    cfg = {"scope": {"kind": "lengths", "lengths": [2]}, "check": "sample", "sample_size": 80, "sample_seed": 73}
    v = Verifier(model4, "lin-a-1to4-s0", {"verifier": cfg})
    entry = model4.reference_entries(v.weights)["contraction"]
    receipt = v.score(entry)
    x, how, meta = regions.prepare_inputs(model4.info(), v.weights, cfg)
    answers, ops = lang.run_reference(v.reference, v.weights, 4, x)
    parts = regions.measure(entry, v.weights, 4, x, answers, ops, how=how, scope=meta, verifier_cfg=cfg)
    assert receipt["parts"] == parts
    assert receipt["inputs"] == how
    assert receipt["scope"] == meta


def test_intersect_scopes_cannot_broaden_the_assignment():
    from level1 import regions
    assigned = {"kind": "lengths", "lengths": [1, 2]}
    assert regions.intersect_scopes(assigned, {"kind": "all"}) == assigned
    assert regions.intersect_scopes(assigned, {"kind": "lengths", "lengths": [2, 3]}) == {
        "kind": "lengths", "lengths": [2],
    }
    with pytest.raises(lang.EntryError, match="empty"):
        regions.intersect_scopes(assigned, {"kind": "lengths", "lengths": [3]})


def test_intersection_identity_flattens_deduplicates_and_ignores_sampler_settings():
    from level1 import regions
    a = {"kind": "predicate", "entry": predicate(lambda g: g.token(0) < 3), "max_draws": 100}
    b = {"kind": "predicate", "entry": predicate(lambda g: g.token(1) > 0), "mass_samples": 300}
    simple = regions.intersect_scopes(a, b)
    nested = regions.intersect_scopes({"kind": "all"}, {
        "kind": "intersection", "scopes": [b, {**a, "max_draws": 900}, a],
    })
    assert regions.scope_id(simple) == regions.scope_id(nested)
    assert regions.normalize_scope(nested) == nested
    assert regions.intersect_scopes(a, {**a, "max_draws": 900}) == regions.normalize_scope(a)
    assert len(nested["scopes"]) == 2
    assert next(item for item in nested["scopes"] if "max_draws" in item)["max_draws"] == 100


def test_overlapping_predicates_use_the_intersection_mass_not_a_product():
    from level1 import regions
    a = {"kind": "predicate", "entry": predicate(lambda g: g.token(0) < 3)}
    b = {"kind": "predicate", "entry": predicate(lambda g: g.token(0) > 0)}
    spec = regions.intersect_scopes(a, b)
    x, how, meta = regions.prepare_inputs(finite_info(), {}, {"scope": spec}, seed=2)
    assert how["mode"] == "all"
    assert len(x) == 6 and set(x[:, 0]) == {1, 2}
    assert meta["mass"] == 0.5 and meta["cardinality"] == meta["workload_size"] == 6
    assert meta["predicate_binding"] == "current_weights"
    with pytest.raises(lang.EntryError, match="empty"):
        regions.prepare_inputs(finite_info(), {}, {"scope": regions.intersect_scopes(
            a, {"kind": "predicate", "entry": predicate(lambda g: g.token(0) > 2)},
        )})


def test_mixed_intersection_keeps_declared_probabilities_and_outer_mass():
    from level1 import regions
    info = {"sizes": {"n_ctx": 3}, "input_space": [3] * 3,
            "data_set": {"kind": "lengths", "lengths": [1, 1, 2, 3], "vocab": 2, "empty": 2}}
    spec = regions.intersect_scopes(
        {"kind": "lengths", "lengths": [1, 2]},
        {"kind": "predicate", "entry": predicate(lambda g: g.token(0) == 0)},
    )
    x, how, meta = regions.prepare_inputs(info, {}, {"scope": spec, "sample_size": 4000}, seed=4)
    lengths = np.sum(x != 2, axis=1)
    assert how["mode"] == "sample" and set(lengths) == {1, 2} and np.all(x[:, 0] == 0)
    assert 0.63 < np.mean(lengths == 1) < 0.70
    assert meta["mass"] == 3 / 8 and meta["mass_exact"]
    assert meta["cardinality"] == meta["workload_size"] == 3


def test_small_selected_support_can_be_counted_without_enumerating_the_full_domain():
    from level1 import regions
    info = {"sizes": {"n_ctx": 6}, "input_space": [4] * 6,
            "data_set": {"kind": "lengths", "lengths": [1, 6], "vocab": 3, "empty": 3}}
    spec = regions.intersect_scopes(
        {"kind": "lengths", "lengths": [1]},
        {"kind": "predicate", "entry": predicate(lambda g: g.token(0) < 2)},
    )
    x, how, meta = regions.prepare_inputs(info, {}, {"scope": spec, "enumerate_up_to": 10}, seed=0)
    assert how["mode"] == "all" and len(x) == 2
    assert meta["mass"] == pytest.approx(1 / 3) and meta["mass_exact"]
    assert meta["conditioning"]["n_evaluated"] == 3


def test_mixed_rejection_estimate_is_scaled_to_the_original_distribution():
    from level1 import regions
    info = {"sizes": {"n_ctx": 3}, "input_space": [3] * 3,
            "data_set": {"kind": "lengths", "lengths": [1, 2, 3, 3], "vocab": 2, "empty": 2}}
    spec = regions.intersect_scopes(
        {"kind": "lengths", "lengths": [1]},
        {"kind": "predicate", "entry": predicate(lambda g: g.token(0) == 0), "mass_samples": 400},
    )
    cfg = {"scope": spec, "sample_size": 100, "enumerate_up_to": 1, "workload_size": 100}
    x, _, meta = regions.prepare_inputs(info, {}, cfg, seed=3)
    estimate = meta["mass_estimation"]
    assert meta["cardinality"] is None and not meta["mass_exact"]
    assert np.all(x[:, 0] == 0) and np.all(x[:, 1:] == 2)
    assert estimate["probe_base_mass"] == 0.25
    assert meta["mass"] == 0.25 * estimate["accepted"] / estimate["n"]
    assert estimate["low"] < 0.125 < estimate["high"] < 0.25


def test_continuous_intersections_retain_workload_and_rejection_limits():
    from level1 import regions
    info = {"sizes": {"n_ctx": 2}, "input_space": ["real", "real"], "data_set": {"kind": "normal"}}
    a = {"kind": "predicate", "entry": predicate(lambda g: g.token(0) > 0, real=True),
         "max_draws": 1000, "mass_samples": 400}
    b = {"kind": "predicate", "entry": predicate(lambda g: g.token(0) < 1, real=True),
         "max_draws": 2000, "mass_samples": 400}
    spec = regions.intersect_scopes(a, b)
    with pytest.raises(lang.EntryError, match="workload_size"):
        regions.prepare_inputs(info, {}, {"scope": spec, "sample_size": 100}, seed=3)
    x, _, meta = regions.prepare_inputs(info, {}, {"scope": spec, "sample_size": 100, "workload_size": 100}, seed=3)
    assert np.all((0 < x[:, 0]) & (x[:, 0] < 1))
    assert 0.25 < meta["mass"] < 0.45 and meta["cardinality"] is None
    capped = regions.intersect_scopes({**a, "max_draws": 10}, b)
    with pytest.raises(lang.EntryError, match="within 10 draws"):
        regions.prepare_inputs(info, {}, {"scope": capped, "sample_size": 20, "workload_size": 100}, seed=3)


def test_intersections_share_the_existing_predicate_structure_allowance():
    from level1 import regions
    a = {"kind": "predicate", "entry": predicate(lambda g: g.token(0) < 3)}
    b = {"kind": "predicate", "entry": predicate(lambda g: g.token(1) > 0)}
    limits = {"max_nodes": 6, "max_const_scalars": 256}
    regions.prepare_inputs(finite_info(), {}, {"scope": a}, limits=limits)
    regions.prepare_inputs(finite_info(), {}, {"scope": b}, limits=limits)
    with pytest.raises(lang.EntryError, match="nodes"):
        regions.prepare_inputs(finite_info(), {}, {"scope": regions.intersect_scopes(a, b)}, limits=limits)
    nested = {"kind": "all"}
    for _ in range(70):
        nested = {"kind": "intersection", "scopes": [nested]}
    with pytest.raises(lang.EntryError, match="too large"):
        regions.normalize_scope(nested)
    with pytest.raises(lang.EntryError, match="takes only"):
        regions.intersect_scopes(a, {"kind": "intersection", "scopes": [b], "workload_size": 100})
