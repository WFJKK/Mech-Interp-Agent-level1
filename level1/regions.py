"""Conditional input domains, shared by the verifier and the local sandbox.

A scope changes the distribution being measured, not the reward or the prices.
The entry's own region is measured inside that scope. The scope's probability
in the original distribution and the work of conditioning are reported
separately; a program that routes global inputs must pay for its own routing.
"""

from __future__ import annotations

import copy
import hashlib
import math
import os
import time
from collections import Counter

import numpy as np

try:
    from . import lang
except ImportError:  # the same file is copied next to lv1.py in the sandbox
    import lv1 as lang


MAX_DRAWS = 1_000_000
DEFAULT_DRAWS = 200_000
DEFAULT_MASS_SAMPLES = 1000
DEFAULT_LIMITS = {"max_nodes": 400, "max_const_scalars": 256}
MAX_SCOPE_COMPONENTS = 8


def _positive_int(value, name, maximum=None):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise lang.EntryError(f"{name} must be a positive whole number")
    if maximum is not None and value > maximum:
        raise lang.EntryError(f"{name} must be at most {maximum}")
    return value


def _predicate_entry(entry):
    """Keep the predicate and its ancestors, with a harmless integer output."""
    lang.check_structure(entry)
    if entry.get("rule") is None:
        raise lang.EntryError("a predicate scope needs an entry with a region test")
    needed, stack = set(), [entry["rule"]]
    while stack:
        i = stack.pop()
        if i not in needed:
            needed.add(i)
            stack.extend(entry["nodes"][i].get("args", []))
    order = sorted(needed)
    mapped = {old: new for new, old in enumerate(order)}
    nodes = []
    for old in order:
        node = copy.deepcopy(entry["nodes"][old])
        if "args" in node:
            node["args"] = [mapped[i] for i in node["args"]]
        nodes.append(node)
    nodes.append({"op": "const", "dtype": "int", "value": 0})
    return {"format": lang.FORMAT, "nodes": nodes, "rule": mapped[entry["rule"]], "predict": len(nodes) - 1}


def normalize_scope(spec=None):
    """Validate a scope without allowing it to change evaluation policy."""
    if spec is None:
        return {"kind": "all"}
    if not isinstance(spec, dict):
        raise lang.EntryError("scope must be an object")
    kind = spec.get("kind")
    keys = {"all": {"kind"}, "lengths": {"kind", "lengths"},
            "predicate": {"kind", "entry", "max_draws", "mass_samples"},
            "intersection": {"kind", "scopes"}}
    if not isinstance(kind, str) or kind not in keys:
        raise lang.EntryError("scope.kind must be all, lengths, predicate or intersection")
    extra = set(spec) - keys[kind]
    if extra:
        raise lang.EntryError(f"scope {kind} does not take {sorted(extra)}")
    if kind == "intersection":
        return _normalize_intersection(spec.get("scopes"))
    out = {"kind": kind}
    if kind == "lengths":
        lengths = spec.get("lengths")
        if not isinstance(lengths, list) or not lengths:
            raise lang.EntryError("scope.lengths must be a nonempty list")
        out["lengths"] = sorted({_positive_int(n, "scope.lengths") for n in lengths})
    if kind == "predicate":
        out["entry"] = _predicate_entry(spec.get("entry"))
        for key in ("max_draws", "mass_samples"):
            if key in spec:
                out[key] = _positive_int(spec[key], f"scope.{key}", MAX_DRAWS)
    return out


def _scope_identity(spec):
    if spec["kind"] == "intersection":
        return {"kind": "intersection", "scopes": [_scope_identity(item) for item in spec["scopes"]]}
    return {k: v for k, v in spec.items() if k not in ("max_draws", "mass_samples")}


def _normalize_intersection(scopes):
    """Flatten conjunctions and retain the strictest requested rejection cap."""
    if not isinstance(scopes, list) or not scopes:
        raise lang.EntryError("scope.scopes must be a nonempty list")
    pending = list(scopes)
    seen = 0
    lengths = None
    predicates = {}
    while pending:
        seen += 1
        # Bound raw nesting as well as the number of distinct predicates.
        if seen > MAX_SCOPE_COMPONENTS ** 2 or len(pending) > MAX_SCOPE_COMPONENTS ** 2:
            raise lang.EntryError("scope intersection is too large")
        item = pending.pop()
        if isinstance(item, dict) and item.get("kind") == "intersection":
            if set(item) != {"kind", "scopes"}:
                raise lang.EntryError("scope intersection takes only kind and scopes")
            children = item["scopes"]
            if not isinstance(children, list) or not children:
                raise lang.EntryError("scope.scopes must be a nonempty list")
            pending.extend(children)
            continue
        item = normalize_scope(item)
        if item["kind"] == "lengths":
            values = set(item["lengths"])
            lengths = values if lengths is None else lengths & values
            if not lengths:
                raise lang.EntryError("the assigned scope is empty")
        elif item["kind"] == "predicate":
            key = lang.canonical(_scope_identity(item))
            if key in predicates:
                prior = predicates[key]
                for option, default, combine in (("max_draws", DEFAULT_DRAWS, min),
                                                  ("mass_samples", DEFAULT_MASS_SAMPLES, max)):
                    if option in item or option in prior:
                        prior[option] = combine(prior.get(option, default), item.get(option, default))
            else:
                predicates[key] = item
            if len(predicates) > MAX_SCOPE_COMPONENTS:
                raise lang.EntryError(f"a scope intersection has at most {MAX_SCOPE_COMPONENTS} components")
    components = list(predicates.values())
    if lengths is not None:
        components.append({"kind": "lengths", "lengths": sorted(lengths)})
    if len(components) > MAX_SCOPE_COMPONENTS:
        raise lang.EntryError(f"a scope intersection has at most {MAX_SCOPE_COMPONENTS} components")
    if not components:
        return {"kind": "all"}
    if len(components) == 1:
        return components[0]
    components.sort(key=lambda item: lang.canonical(_scope_identity(item)))
    return {"kind": "intersection", "scopes": components}


def intersect_scopes(parent, child):
    """Condition a claim on both its assignment and its requested narrower scope."""
    return normalize_scope({"kind": "intersection", "scopes": [parent, child]})


def _combined_predicate(specs, limits):
    """Conjoin predicates without granting each component a fresh entry allowance."""
    nodes, rules = [], []
    for spec in specs:
        entry = spec["entry"]
        lang.check_structure(entry, limits["max_nodes"], limits["max_const_scalars"])
        offset = len(nodes)
        # The normalized entry's last node is its unused constant prediction.
        for node in entry["nodes"][:-1]:
            node = copy.deepcopy(node)
            if "args" in node:
                node["args"] = [i + offset for i in node["args"]]
            nodes.append(node)
        rules.append(entry["rule"] + offset)
    rule = rules[0]
    for following in rules[1:]:
        nodes.append({"op": "and", "args": [rule, following]})
        rule = len(nodes) - 1
    nodes.append({"op": "const", "dtype": "int", "value": 0})
    entry = {"format": lang.FORMAT, "nodes": nodes, "rule": rule, "predict": len(nodes) - 1}
    lang.check_structure(entry, limits["max_nodes"], limits["max_const_scalars"])
    return entry


def scope_id(spec=None):
    """Stable scope identity, independent of rejection and diagnostic settings."""
    spec = normalize_scope(spec)
    if spec["kind"] == "all":
        return "all"
    identity = _scope_identity(spec)
    return "scope-" + hashlib.sha256(lang.canonical(identity).encode()).hexdigest()[:20]


def _deadline(deadline):
    if deadline is not None and time.monotonic() > deadline:
        raise lang.EntryError("scope preparation ran out of time")


class Scope:
    """An input distribution conditioned on a validated, fixed scope.

    Enumeration is used only when every surviving input has equal probability.
    For other finite distributions the cardinality is still useful as a declared
    workload size, but measurements retain the original conditional probabilities.
    """

    def __init__(self, input_space, data_set, weights, n_ctx, spec=None, *,
                 limits=None, enumerate_up_to=200_000, deadline=None):
        self.input_space = list(input_space)
        self.data_set = copy.deepcopy(data_set or {"kind": "all"})
        self.n_ctx = int(n_ctx)
        self.spec = normalize_scope(spec)
        self.id = scope_id(self.spec)
        self.limits = {**DEFAULT_LIMITS, **(limits or {})}
        self.enumerate_up_to = _positive_int(enumerate_up_to, "enumerate_up_to")
        self.deadline = deadline if deadline is not None else time.monotonic() + 120.0
        self.kind = lang.data_set_kind(self.data_set)
        self.cardinality = None
        self.mass = 1.0
        self.mass_exact = True
        self.can_enumerate = False
        self._points = self._probabilities = None
        self._predicate = None
        self._predicate_entry = None
        self._base_mass = 1.0
        self._max_draws = DEFAULT_DRAWS
        self._mass_samples = DEFAULT_MASS_SAMPLES
        self._conditioning = {"rule_ops": 0, "build_ops": 0, "n_evaluated": 0}
        self._mass_estimation = None
        self._draw_set = self.data_set
        self._length_counts = None

        if self.kind == "lengths":
            lengths = self.data_set.get("lengths", [])
            if not lengths or any(isinstance(n, bool) or not isinstance(n, int) or not 1 <= n <= self.n_ctx for n in lengths):
                raise lang.EntryError("the data set's lengths must fit the input slots")
            vocab = _positive_int(self.data_set.get("vocab"), "data_set.vocab")
            if 0 <= int(self.data_set["empty"]) < vocab:
                raise lang.EntryError("scoped length data needs a padding value outside its vocabulary")
            self._length_counts = Counter(lengths)
            parent_count = sum(vocab ** n for n in self._length_counts)
        else:
            parent_count = lang.count_inputs(self.data_set, self.input_space)

        components = self.spec["scopes"] if self.spec["kind"] == "intersection" else [self.spec]
        length_spec = next((item for item in components if item["kind"] == "lengths"), None)
        predicates = [item for item in components if item["kind"] == "predicate"]
        base_count = parent_count
        if length_spec is not None:
            if self.kind != "lengths":
                raise lang.EntryError("a lengths scope needs a data set declaring lengths")
            selected = length_spec["lengths"]
            if not set(selected) <= set(self._length_counts):
                raise lang.EntryError("scope.lengths must be a subset of the data set's lengths")
            selected_draws = [n for n in self.data_set["lengths"] if n in selected]
            self._draw_set = {**self.data_set, "lengths": selected_draws}
            self.cardinality = sum(vocab ** n for n in selected)
            base_count = self.cardinality
            self.mass = len(selected_draws) / len(self.data_set["lengths"])
            self._base_mass = self.mass
            self.can_enumerate = self._uniform_lengths(Counter(selected_draws))
        else:
            self.cardinality = parent_count
            self.can_enumerate = self.kind == "all" or (self.kind == "lengths" and self._uniform_lengths(self._length_counts))
        if predicates:
            entry = predicates[0]["entry"] if len(predicates) == 1 else _combined_predicate(predicates, self.limits)
            lang.check_structure(entry, self.limits["max_nodes"], self.limits["max_const_scalars"])
            self._predicate_entry = entry
            self._predicate = lang.Evaluator(entry, weights, self.n_ctx, self.deadline)
            self._max_draws = min(item.get("max_draws", DEFAULT_DRAWS) for item in predicates)
            self._mass_samples = max(item.get("mass_samples", DEFAULT_MASS_SAMPLES) for item in predicates)
            self.cardinality = None
            self.can_enumerate = False
            self.mass = None
            self.mass_exact = False
            if base_count is not None and base_count <= self.enumerate_up_to and base_count * self.n_ctx <= lang.MAX_ELEMENTS:
                points, probabilities = self._support(self._draw_set)
                accepted = self._filter(points)
                self._points = points[accepted]
                self.cardinality = int(accepted.sum())
                if not self.cardinality:
                    raise lang.EntryError("the assigned scope is empty")
                probability = probabilities[accepted]
                conditional_mass = self.cardinality / base_count if self.kind == "all" else math.fsum(probability.tolist())
                self.mass = self._base_mass * conditional_mass
                self.mass_exact = True
                self._probabilities = probability / conditional_mass
                self.can_enumerate = bool(np.all(probability == probability[0]))

    def _uniform_lengths(self, counts):
        vocab = int(self.data_set["vocab"])
        items = list(counts.items())
        first, multiplicity = items[0]
        return all(multiplicity * vocab ** n == count * vocab ** first for n, count in items)

    def _support(self, data_set):
        """Finite support and its probabilities under the declared distribution."""
        _deadline(self.deadline)
        if self.kind == "all":
            n = lang.count_inputs(data_set, self.input_space)
            if n * self.n_ctx > lang.MAX_ELEMENTS:
                raise lang.EntryError("scope enumeration is too large; use sampling")
            return lang.all_inputs(self.input_space), np.full(n, 1.0 / n)
        if self.kind != "lengths":
            raise lang.EntryError("this scope cannot be checked on every input")
        counts = Counter(data_set["lengths"])
        vocab = int(data_set["vocab"])
        n_total = sum(vocab ** n for n in counts)
        if n_total * self.n_ctx > lang.MAX_ELEMENTS:
            raise lang.EntryError("scope enumeration is too large; use sampling")
        rows, probabilities = [], []
        for length, count in sorted(counts.items()):
            _deadline(self.deadline)
            n = vocab ** length
            x = np.full((n, self.n_ctx), int(data_set["empty"]), dtype=np.int64)
            x[:, :length] = lang.all_inputs([vocab] * length)
            rows.append(x)
            probabilities.append(np.full(n, count / (len(data_set["lengths"]) * n)))
        return np.concatenate(rows), np.concatenate(probabilities)

    def _filter(self, points):
        accepted = np.zeros(len(points), dtype=bool)
        for i, x in enumerate(points):
            _deadline(self.deadline)
            inside, _, cost, _ = self._predicate.run(x)
            accepted[i] = inside
            self._conditioning["rule_ops"] += cost
        self._conditioning["n_evaluated"] += len(points)
        self._conditioning["build_ops"] = self._predicate.build_cost
        return accepted

    def resolve_check(self, cfg):
        check = cfg.get("check", "auto")
        if check not in ("auto", "all", "sample"):
            raise lang.EntryError("verifier.check must be one of ('auto', 'all', 'sample')")
        if check == "all" and not self.can_enumerate:
            raise lang.EntryError("this scope cannot be checked on every input at equal weight; use sampling")
        if check == "auto":
            return "all" if self.can_enumerate and self.cardinality <= int(cfg.get("enumerate_up_to", 200_000)) else "sample"
        return check

    def enumerate(self):
        if not self.can_enumerate:
            raise lang.EntryError("this scope cannot be checked on every input at equal weight")
        if self._points is None:
            self._points, _ = self._support(self._draw_set)
        return self._points

    def draw(self, n, rng, mass_rng=None):
        _deadline(self.deadline)
        if self._predicate is None:
            return lang.draw_inputs(self._draw_set, self.input_space, n, rng)
        if self._points is not None:
            return self._points[rng.choice(len(self._points), size=n, p=self._probabilities)]
        self._predicate = lang.Evaluator(self._predicate_entry, self._predicate.weights, self.n_ctx, self.deadline)
        self._conditioning = {"rule_ops": 0, "build_ops": 0, "n_evaluated": 0}
        # A fixed-size independent probe estimates outer mass. The acceptance
        # ratio at a rejection stopping time is not used as a mass estimate.
        if mass_rng is None:
            encoded = lang.canonical(rng.bit_generator.state).encode()
            mass_rng = np.random.default_rng(int.from_bytes(hashlib.sha256(encoded).digest()[:8], "big"))
        probe_n = self._mass_samples
        successes = 0
        batch_limit = max(1, min(4096, lang.MAX_ELEMENTS // self.n_ctx))
        for start in range(0, probe_n, batch_limit):
            probe = lang.draw_inputs(self._draw_set, self.input_space, min(batch_limit, probe_n - start), mass_rng)
            successes += int(self._filter(probe).sum())
        self.mass = self._base_mass * successes / probe_n
        self._mass_estimation = {"n": probe_n, "accepted": successes,
                                 "low": self._base_mass * lang.wilson_lower(successes, probe_n),
                                 "high": self._base_mass * (1.0 - lang.wilson_lower(probe_n - successes, probe_n))}
        if self._base_mass != 1.0:
            # Probes are conditional on the length support, whose outer mass is
            # known. Scale the estimate and interval back to the original input
            # distribution; do not multiply two separately estimated masses.
            self._mass_estimation["probe_base_mass"] = self._base_mass
        remaining = n
        drawn = 0
        chunks = []
        cap = self._max_draws
        while remaining and drawn < cap:
            amount = min(max(remaining * 2, 64), cap - drawn, batch_limit)
            # Draw one row at a time so changing the cap cannot change how a
            # column-wise data sampler groups random numbers into inputs.
            proposals = np.concatenate([lang.draw_inputs(self._draw_set, self.input_space, 1, rng) for _ in range(amount)])
            keep = proposals[self._filter(proposals)][:remaining]
            chunks.append(keep)
            remaining -= len(keep)
            drawn += amount
        if remaining:
            raise lang.EntryError(f"scope rejection sampling found only {n - remaining} of {n} inputs within {cap} draws")
        return np.concatenate(chunks)

    def metadata(self):
        out = {"id": self.id, "spec": copy.deepcopy(self.spec), "cardinality": self.cardinality,
               "mass": self.mass, "mass_exact": self.mass_exact,
               "conditioning": dict(self._conditioning)}
        if self._mass_estimation is not None:
            out["mass_estimation"] = dict(self._mass_estimation)
        if self._predicate is not None:
            out["predicate_binding"] = "current_weights"
        return out


def _workload(cfg, scope):
    if cfg.get("workload_size") is not None:
        return _positive_int(cfg["workload_size"], "verifier.workload_size"), "declared"
    if scope.cardinality is None:
        raise lang.EntryError("this scope has no known finite cardinality; declare verifier.workload_size")
    return scope.cardinality, ("all_once" if scope.can_enumerate else "cardinality_at_declared_distribution")


def prepare_inputs(info, weights, verifier_cfg=None, *, seed=None, limits=None, deadline=None, scope=None):
    """Return inputs, sampling details and optional scope metadata.

    With no scope, the historical enumeration and workload conventions remain
    unchanged. An explicit scope uses its finite cardinality, or a workload
    supplied by the controller, independently of the verification sample size.
    """
    cfg = verifier_cfg or {}
    if cfg.get("workload_size") is not None:
        _positive_int(cfg["workload_size"], "verifier.workload_size")
    deadline = deadline if deadline is not None else time.monotonic() + float(cfg.get("time_limit_s", 120.0))
    _deadline(deadline)
    data_set, space = info.get("data_set") or {"kind": "all"}, info["input_space"]
    if cfg.get("scope") is not None:
        scope = scope or Scope(space, data_set, weights, info["sizes"]["n_ctx"], cfg["scope"],
                               limits=limits, enumerate_up_to=int(cfg.get("enumerate_up_to", 200_000)), deadline=deadline)
        scope.deadline = deadline
        if scope._predicate is not None:
            scope._predicate.deadline = deadline
        workload, basis = _workload(cfg, scope)
        check = scope.resolve_check(cfg)
    else:
        check = cfg.get("check", "auto")
        if check not in ("auto", "all", "sample"):
            raise lang.EntryError("verifier.check must be one of ('auto', 'all', 'sample')")
        can_enumerate = lang.data_set_kind(data_set) == "all"
        if check == "all" and not can_enumerate:
            raise lang.EntryError("the data set cannot be checked on every input; use sampling")
        if check == "auto":
            check = "all" if can_enumerate and lang.count_inputs(data_set, space) <= int(cfg.get("enumerate_up_to", 200_000)) else "sample"
        scope = None
    if check == "all":
        x = scope.enumerate() if scope is not None else lang.all_inputs(space)
        how = {"mode": "all", "n": len(x), "seed": None}
    else:
        n = _positive_int(cfg.get("sample_size", 4000), "verifier.sample_size")
        if seed is None:
            seed = cfg.get("sample_seed")
        if seed is None:
            seed = int.from_bytes(os.urandom(8), "big")
        rng = np.random.default_rng(seed)
        mass_rng = np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(1,))) if scope is not None else None
        x = scope.draw(n, rng, mass_rng) if scope is not None else lang.draw_inputs(data_set, space, n, rng)
        how = {"mode": "sample", "n": n, "seed": int(seed)}
    meta = None
    if scope is not None:
        meta = scope.metadata()
        meta.update(workload_size=workload, workload_basis=basis,
                    build_measurement="observed_lazy_dependencies")
        how["scope_id"] = scope.id
    return x, how, meta


def measure(entry, weights, n_ctx, inputs, answers, model_ops, *, how, scope=None,
            verifier_cfg=None, limits=None, deadline=None):
    """Measure within prepared scope inputs, retaining the existing reward parts."""
    limits = {**DEFAULT_LIMITS, **(limits or {})}
    lang.check_structure(entry, limits["max_nodes"], limits["max_const_scalars"])
    cfg = verifier_cfg or {}
    workload = cfg.get("workload_size")
    if workload is None and scope is not None:
        workload = scope["workload_size"]
    parts = lang.measure(entry, weights, n_ctx, inputs, answers, model_ops,
                         deadline=deadline, workload_size=workload)
    parts["fidelity_low"] = (lang.wilson_lower(parts["n_agree"], parts["n_in_region"])
                             if how["mode"] == "sample" and parts["n_in_region"] else parts["fidelity"])
    return parts
