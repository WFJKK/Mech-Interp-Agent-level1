"""The entry language of level 1: programs in counted operations.

An entry is a small program, stored as JSON data, with two results per input:

    rule      the region test, a boolean: is this input in the entry's region S?  (absent: every input is)
    predict   the interpretation, an integer: the network's output on this input, as the entry predicts it

It is a graph of operations on tensors. The sources are the input tokens, the
network's weights (by name), constants and ranges; every other node applies one
operation to earlier nodes. Nothing in an entry is code: the verifier reads the
JSON and interprets it with the table below, so an entry can do no work that is
not counted.

Prices (PRICES = "prices-v1"): each operation on numbers counts 1. Per result
entry unless said otherwise:

    add sub mul div floordiv mod neg abs maximum minimum exp log sqrt     1
    lt le gt ge eq ne and or not                                          1
    where(c, x, y)        both sides are computed                         1
    cond(c, x, y)         only the chosen side is computed                1 in all
    matmul                one per multiply-add: (result entries) x (the contracted length)
    sum max min argmax argmin any all over n entries                      n - 1
    cumsum                                                                n - 1 per vector
    sort argsort          of n entries                                    n * ceil(log2 n) per vector
    softmax                                                               3
    take(x, idx, axis)    reading numbers                                 1 per entry read
    gather(x, i.., axes)  reading at scalar positions                     1 per entry read
    take_sum(x, idx, axis)  the sum of the slices at idx: a matrix applied to a sum of
                          one-hot vectors                                 (slice entries) x len(idx)
    svd_u svd_s svd_vt    of an m x n matrix, charged once for the three  4 * m * n * min(m, n)
    cast                  to whole numbers, floats or booleans           1
    token tokens weight const arange reshape transpose slice concat stack         0

What does not depend on the input is computed once per entry and charged to the
build; the rest is charged per input, to the rule (for every input) or to the
prediction (for the inputs of the region). A value the rule already computed for
an input is not charged again to the prediction.

In the words used with the agent: the prediction is the entry's interpretation,
the rule is the test for its region, and an entry's cost is all three charges
together, per input of the region, over one forward pass.

The same file gives the builder that writes entries (Graph and its expressions),
the interpreter (Evaluator) and measure(), which the verifier and the agent's
sandbox both use. It needs numpy and the standard library only.
"""

from __future__ import annotations

import builtins
import hashlib
import json
import math
import time

import numpy as np

FORMAT = "level1-entry-v1"
PRICES = "prices-v1"

MAX_NODES = 2000  # absolute ceilings; a run's own limits are lower (config: entry.max_nodes, entry.max_const_scalars)
MAX_CONST_SCALARS = 1_000_000
MAX_ELEMENTS = 4_000_000  # entries of any one value
MAX_HELD = 48_000_000  # entries of all the values an entry holds at once (the build's, or one input's)
MAX_RANK = 6
SVD_FACTOR = 4

ELEMENTWISE_2 = {
    "add": np.add, "sub": np.subtract, "mul": np.multiply, "div": np.true_divide, "floordiv": np.floor_divide,
    "mod": np.mod, "maximum": np.maximum, "minimum": np.minimum, "lt": np.less, "le": np.less_equal,
    "gt": np.greater, "ge": np.greater_equal, "eq": np.equal, "ne": np.not_equal, "and": np.logical_and,
    "or": np.logical_or,
}
ELEMENTWISE_1 = {"neg": np.negative, "abs": np.abs, "exp": np.exp, "log": np.log, "sqrt": np.sqrt, "not": np.logical_not}
REDUCE = {"sum": np.sum, "max": np.max, "min": np.min, "argmax": np.argmax, "argmin": np.argmin, "any": np.any, "all": np.all}
ALONG = ("cumsum", "sort", "argsort", "softmax")
SVD = ("svd_u", "svd_s", "svd_vt")
SOURCES = ("token", "tokens", "weight", "const", "arange")
DTYPES = {"int": np.int64, "float": np.float64, "bool": np.bool_}

ARITY = {**{k: 2 for k in ELEMENTWISE_2}, **{k: 1 for k in ELEMENTWISE_1}, **{k: 1 for k in REDUCE},
         **{k: 1 for k in ALONG}, **{k: 1 for k in SVD}, "where": 3, "cond": 3, "matmul": 2, "take": 2,
         "take_sum": 2, "gather": None, "reshape": 1, "transpose": 1, "slice": 1, "concat": None, "stack": None,
         "cast": 1, **{k: 0 for k in SOURCES}}
ATTRS = {"token": {"i"}, "weight": {"name"}, "const": {"value", "dtype"}, "arange": {"n"}, "take": {"axis"},
         "take_sum": {"axis"}, "gather": {"axes"}, "reshape": {"shape"}, "transpose": {"perm"}, "slice": {"index"},
         "concat": {"axis"}, "stack": {"axis"}, "cast": {"dtype"}, **{k: {"axis"} for k in REDUCE},
         **{k: {"axis"} for k in ALONG}}


class EntryError(ValueError):
    """The entry is not a valid program, or it failed on an input. The message is for its author."""


# values ---------------------------------------------------------------------------------
def _norm(v) -> np.ndarray:
    a = np.asarray(v)
    if a.dtype == np.bool_:
        return a
    if np.issubdtype(a.dtype, np.integer):
        return a.astype(np.int64, copy=False)
    if np.issubdtype(a.dtype, np.floating):
        return a.astype(np.float64, copy=False)
    raise EntryError(f"values are integers, floats or booleans, not {a.dtype}")


def _guard(n: int) -> None:
    if n > MAX_ELEMENTS:
        raise EntryError(f"a value with {n} entries is above the limit of {MAX_ELEMENTS}")


def _axis(node: dict, ndim: int, key: str = "axis") -> int:
    ax = node.get(key)
    if not isinstance(ax, int) or isinstance(ax, bool) or not 0 <= ax < max(ndim, 1) or ndim == 0:
        raise EntryError(f"{node['op']}: axis {ax!r} does not fit a value with {ndim} dimensions")
    return ax


def _int_index(v: np.ndarray, n: int, what: str) -> np.ndarray:
    if not np.issubdtype(v.dtype, np.integer):
        raise EntryError(f"{what}: an index must be whole numbers, not {v.dtype}")
    if v.size and (int(v.min()) < 0 or int(v.max()) >= n):
        raise EntryError(f"{what}: index out of range for a length of {n}")
    return v


def _const(node: dict) -> np.ndarray:
    dtype = node.get("dtype")
    if dtype not in DTYPES:
        raise EntryError(f"const: dtype must be one of {sorted(DTYPES)}")
    try:
        a = np.array(node.get("value"), dtype=DTYPES[dtype])
    except (ValueError, TypeError, OverflowError) as exc:
        raise EntryError(f"const: not a rectangular array of numbers ({exc})") from None
    if a.dtype == object or a.ndim > MAX_RANK:
        raise EntryError("const: not a rectangular array of numbers")
    if a.dtype == np.float64 and not np.isfinite(a).all():
        raise EntryError("const: numbers must be finite")
    return a


def _svd(x: np.ndarray):
    if x.ndim != 2 or 0 in x.shape:
        raise EntryError("svd: needs a matrix")
    u, s, vt = np.linalg.svd(x.astype(np.float64), full_matrices=False)
    m, n = x.shape
    return (u, s, vt), SVD_FACTOR * m * n * min(m, n)


def apply(node: dict, args: list, tokens: np.ndarray | None, weights: dict, svd_cache: dict | None = None):
    """The value and the cost of one node, given its arguments' values. cond is not handled here."""
    op = node["op"]
    if op == "token":
        if tokens is None:
            raise EntryError("token used where no input is given")
        i = node.get("i")
        if not isinstance(i, int) or isinstance(i, bool) or not 0 <= i < len(tokens):
            raise EntryError(f"token: no position {i!r} in an input of {len(tokens)} tokens")
        return np.asarray(tokens[i], dtype=np.int64), 0
    if op == "tokens":
        if tokens is None:
            raise EntryError("tokens used where no input is given")
        return np.asarray(tokens, dtype=np.int64), 0
    if op == "weight":
        name = node.get("name")
        if name not in weights:
            raise EntryError(f"weight: the network has no weights named {name!r}; it has {sorted(weights)}")
        return weights[name], 0
    if op == "const":
        return _const(node), 0
    if op == "arange":
        n = node.get("n")
        if not isinstance(n, int) or isinstance(n, bool) or not 0 < n <= MAX_ELEMENTS:
            raise EntryError("arange: n must be a positive whole number")
        return np.arange(n, dtype=np.int64), 0

    if op in ELEMENTWISE_2:
        x, y = args
        shape = np.broadcast_shapes(x.shape, y.shape)
        size = int(np.prod(shape, dtype=np.int64))
        _guard(size)
        if op in ("and", "or") and not (x.dtype == np.bool_ and y.dtype == np.bool_):
            raise EntryError(f"{op}: needs booleans")
        if op in ("add", "sub", "mul", "div", "floordiv", "mod") and (x.dtype == np.bool_ or y.dtype == np.bool_):
            x, y = x.astype(np.int64), y.astype(np.int64)
        return _norm(ELEMENTWISE_2[op](x, y)), size
    if op in ELEMENTWISE_1:
        (x,) = args
        if op == "not" and x.dtype != np.bool_:
            raise EntryError("not: needs booleans")
        if op != "not" and x.dtype == np.bool_:
            x = x.astype(np.int64)
        if op in ("exp", "log", "sqrt"):
            x = x.astype(np.float64)
        return _norm(ELEMENTWISE_1[op](x)), int(x.size)
    if op == "where":
        c, x, y = args
        if c.dtype != np.bool_:
            raise EntryError("where: the condition must be boolean")
        shape = np.broadcast_shapes(c.shape, x.shape, y.shape)
        size = int(np.prod(shape, dtype=np.int64))
        _guard(size)
        return _norm(np.where(c, x, y)), size
    if op == "matmul":
        x, y = args
        if x.ndim == 0 or y.ndim == 0:
            raise EntryError("matmul: needs vectors or matrices; use mul for a scalar")
        k = x.shape[-1]
        if k != (y.shape[0] if y.ndim == 1 else y.shape[-2]):
            raise EntryError(f"matmul: shapes {x.shape} and {y.shape} do not fit")
        lead = np.broadcast_shapes(x.shape[:-2] if x.ndim > 1 else (), y.shape[:-2] if y.ndim > 1 else ())
        size = int(np.prod(lead, dtype=np.int64)) * (x.shape[-2] if x.ndim > 1 else 1) * (y.shape[-1] if y.ndim > 1 else 1)
        _guard(size)
        _guard(size * max(k, 1))
        if x.dtype == np.bool_ or y.dtype == np.bool_:
            x, y = x.astype(np.int64), y.astype(np.int64)
        return _norm(np.matmul(x, y)), size * k
    if op in REDUCE:
        (x,) = args
        ax = node.get("axis")
        if ax is None:
            n, out = int(x.size), 1
        else:
            ax = _axis(node, x.ndim)
            n = x.shape[ax]
            out = int(x.size // n) if n else 0
        if n == 0:
            raise EntryError(f"{op}: nothing to reduce")
        if op in ("any", "all") and x.dtype != np.bool_:
            raise EntryError(f"{op}: needs booleans")
        return _norm(REDUCE[op](x, axis=ax)), out * (n - 1)
    if op in ALONG:
        (x,) = args
        ax = _axis(node, x.ndim)
        n = x.shape[ax]
        if n == 0:
            raise EntryError(f"{op}: an empty axis")
        vectors = int(x.size // n)
        if op == "cumsum":
            return _norm(np.cumsum(x.astype(np.int64) if x.dtype == np.bool_ else x, axis=ax)), vectors * (n - 1)
        if op == "softmax":
            z = x.astype(np.float64)
            z = np.exp(z - z.max(axis=ax, keepdims=True))
            return z / z.sum(axis=ax, keepdims=True), 3 * int(x.size)
        cost = vectors * n * math.ceil(math.log2(n)) if n > 1 else 0
        if op == "sort":
            return _norm(np.sort(x, axis=ax, kind="stable")), cost
        return _norm(np.argsort(x, axis=ax, kind="stable")), cost
    if op == "take":
        x, idx = args
        ax = _axis(node, x.ndim)
        _int_index(idx, x.shape[ax], "take")
        _guard(int(idx.size) * int(x.size // max(x.shape[ax], 1)))
        out = np.take(x, idx, axis=ax)
        return out, int(out.size)
    if op == "take_sum":
        x, idx = args
        ax = _axis(node, x.ndim)
        if idx.ndim != 1 or idx.size == 0:
            raise EntryError("take_sum: the index must be a non-empty vector")
        _int_index(idx, x.shape[ax], "take_sum")
        _guard(int(idx.size) * int(x.size // max(x.shape[ax], 1)))
        out = np.take(x, idx, axis=ax).sum(axis=ax)
        return _norm(out), int(out.size) * int(idx.size)
    if op == "gather":
        x, *idxs = args
        axes = node.get("axes")
        if (not isinstance(axes, list) or len(axes) != len(idxs) or not idxs or len(set(axes)) != len(axes)
                or any(not isinstance(a, int) or isinstance(a, bool) or not 0 <= a < x.ndim for a in axes)):
            raise EntryError("gather: axes must name one distinct axis of the value per index")
        index = [slice(None)] * x.ndim
        for a, i in zip(axes, idxs):
            if i.ndim != 0:
                raise EntryError("gather: every index must be a single whole number; use take for a vector of them")
            index[a] = int(_int_index(i, x.shape[a], "gather"))
        out = x[tuple(index)]
        return out, max(int(out.size), 1)
    if op in SVD:
        (x,) = args
        key = ("svd", node["args"][0])
        cost = 0
        if svd_cache is None or key not in svd_cache:
            parts, cost = _svd(x)
            if svd_cache is not None:
                svd_cache[key] = parts
        else:
            parts = svd_cache[key]
        return parts[SVD.index(op)], cost
    if op == "reshape":
        (x,) = args
        shape = node.get("shape")
        if (not isinstance(shape, list) or len(shape) > MAX_RANK
                or any(not isinstance(s, int) or isinstance(s, bool) or s < -1 for s in shape)):
            raise EntryError("reshape: shape must be a list of sizes")
        try:
            return x.reshape(shape), 0
        except ValueError:
            raise EntryError(f"reshape: {x.shape} does not fit {shape}") from None
    if op == "transpose":
        (x,) = args
        perm = node.get("perm")
        if perm is not None and (not isinstance(perm, list) or sorted(perm) != list(range(x.ndim))):
            raise EntryError("transpose: perm must reorder the value's axes")
        return np.transpose(x, perm), 0
    if op == "slice":
        (x,) = args
        index = node.get("index")
        if not isinstance(index, list) or len(index) > x.ndim:
            raise EntryError("slice: one item per axis at most")
        idx = []
        for it in index:
            if it is None:
                idx.append(slice(None))
            elif isinstance(it, int) and not isinstance(it, bool):
                idx.append(it)
            elif isinstance(it, dict) and set(it) <= {"start", "stop", "step"} and all(
                    v is None or (isinstance(v, int) and not isinstance(v, bool)) for v in it.values()) and it.get("step") != 0:
                idx.append(slice(it.get("start"), it.get("stop"), it.get("step")))
            else:
                raise EntryError("slice: an item is null, a whole number or {start, stop, step}")
        try:
            return x[tuple(idx)], 0
        except IndexError:
            raise EntryError(f"slice: out of range for shape {x.shape}") from None
    if op in ("concat", "stack"):
        if not args:
            raise EntryError(f"{op}: needs at least one value")
        _guard(sum(int(a.size) for a in args))
        ax = node.get("axis")
        nd = args[0].ndim + (1 if op == "stack" else 0)
        if not isinstance(ax, int) or isinstance(ax, bool) or not 0 <= ax < max(nd, 1) or nd > MAX_RANK:
            raise EntryError(f"{op}: axis {ax!r} does not fit")
        try:
            return _norm((np.concatenate if op == "concat" else np.stack)(args, axis=ax)), 0
        except ValueError:
            raise EntryError(f"{op}: shapes {[a.shape for a in args]} do not fit") from None
    if op == "cast":
        (x,) = args
        if node.get("dtype") not in DTYPES:
            raise EntryError(f"cast: dtype must be one of {sorted(DTYPES)}")
        if node["dtype"] == "int" and x.dtype == np.float64 and x.size and not (np.abs(x) < 2.0**63).all():
            # what a machine makes of such a float differs from one processor to another
            raise EntryError("cast: a float that is not finite, or too large, has no whole-number value")
        with np.errstate(all="ignore"):
            return x.astype(DTYPES[node["dtype"]]), int(x.size)
    raise EntryError(f"unknown operation {op!r}")


# entries as data --------------------------------------------------------------------------
def check_structure(entry, max_nodes: int = MAX_NODES, max_const_scalars: int = MAX_CONST_SCALARS) -> dict:
    """Refuse anything that is not an entry: the JSON shape, the operations, their arguments and attributes.
    Returns the entry's size: {"nodes", "const_scalars"}."""
    if not isinstance(entry, dict) or set(entry) - {"format", "nodes", "rule", "predict"} or entry.get("format") != FORMAT:
        raise EntryError(f'an entry is {{"format": "{FORMAT}", "nodes": [...], "rule": n or null, "predict": n}}')
    nodes = entry.get("nodes")
    if not isinstance(nodes, list) or not nodes:
        raise EntryError("an entry needs a list of nodes")
    if len(nodes) > min(max_nodes, MAX_NODES):
        raise EntryError(f"the entry has {len(nodes)} nodes, above the limit of {min(max_nodes, MAX_NODES)}")
    scalars = 0
    for i, node in enumerate(nodes):
        if not isinstance(node, dict) or not isinstance(node.get("op"), str) or node["op"] not in ARITY:
            raise EntryError(f"node {i}: unknown operation {node.get('op') if isinstance(node, dict) else node!r}")
        op = node["op"]
        extra = set(node) - {"op", "args"} - ATTRS.get(op, set())
        if extra:
            raise EntryError(f"node {i} ({op}): does not take {sorted(extra)}")
        args = node.get("args", [])
        if not isinstance(args, list) or any(not isinstance(a, int) or isinstance(a, bool) or not 0 <= a < i for a in args):
            raise EntryError(f"node {i} ({op}): arguments must be earlier nodes")
        if ARITY[op] is not None and len(args) != ARITY[op]:
            raise EntryError(f"node {i} ({op}): takes {ARITY[op]} arguments, not {len(args)}")
        if ARITY[op] is None and not 1 <= len(args) <= 64:
            raise EntryError(f"node {i} ({op}): takes between 1 and 64 arguments")
        if op == "const":
            scalars += int(_const(node).size)
            if scalars > max_const_scalars:
                raise EntryError(f"the entry's constants hold more than the limit of {max_const_scalars} numbers")
    for key in ("rule", "predict"):
        r = entry.get(key)
        if r is None and key == "rule":
            continue
        if not isinstance(r, int) or isinstance(r, bool) or not 0 <= r < len(nodes):
            raise EntryError(f"{key} must name a node")
    return {"nodes": len(nodes), "const_scalars": scalars}


def canonical(entry: dict) -> str:
    return json.dumps(entry, sort_keys=True, separators=(",", ":"), allow_nan=False)


def entry_hash(entry: dict) -> str:
    return hashlib.sha256(canonical(entry).encode()).hexdigest()


def show(entry: dict) -> str:
    """An entry as readable lines, one per node: n3 = take_sum(n0, n1, axis=1)."""
    check_structure(entry)
    lines = []
    for i, node in enumerate(entry["nodes"]):
        op = node["op"]
        parts = [f"n{a}" for a in node.get("args", [])]
        for k in sorted(set(node) - {"op", "args"}):
            v = node[k]
            if op == "const" and k == "value":
                text = json.dumps(v)
                v = text if len(text) <= 60 else f"{text[:57]}..."
                parts.append(f"value={v}")
            else:
                parts.append(f"{k}={json.dumps(v)}")
        mark = "".join(f"   <- {label}" for name, label in (("rule", "region"), ("predict", "interpretation")) if entry.get(name) == i)
        lines.append(f"n{i} = {op}({', '.join(parts)}){mark}")
    return "\n".join(lines)


# the interpreter ----------------------------------------------------------------------------
class Evaluator:
    """Runs an entry on inputs and counts its operations.

    build_cost grows as the values that do not depend on the input are first needed; run(x) returns what
    one input costs on top of that."""

    def __init__(self, entry: dict, weights: dict, n_ctx: int, deadline: float | None = None):
        check_structure(entry)
        self.nodes = entry["nodes"]
        self.rule, self.predict = entry.get("rule"), entry["predict"]
        self.weights = {k: _norm(v) for k, v in weights.items()}
        self.n_ctx = int(n_ctx)
        self.deadline = deadline
        self.dep = []
        for node in self.nodes:
            self.dep.append(node["op"] in ("token", "tokens") or any(self.dep[a] for a in node.get("args", [])))
        self.build: dict = {}
        self.build_cost = 0
        self._held = [0, 0]  # entries held by the build's values, and by the current input's

    def _eval(self, root: int, memo: dict, counter: list, tokens):
        dep, build, nodes = self.dep, self.build, self.nodes
        stack = [root]
        while stack:
            i = stack[-1]
            store = memo if dep[i] else build
            if i in store:
                stack.pop()
                continue
            if self.deadline is not None and time.monotonic() > self.deadline:
                raise EntryError("the entry ran out of time")
            node = nodes[i]
            args = node.get("args", [])
            if node["op"] == "cond":
                c, a, b = args
                if c not in (memo if dep[c] else build):
                    stack.append(c)
                    continue
                pred = (memo if dep[c] else build)[c]
                if pred.dtype != np.bool_ or pred.size != 1:
                    raise EntryError(f"node {i} (cond): the condition must be one boolean")
                chosen = a if bool(pred.reshape(())) else b
                if chosen not in (memo if dep[chosen] else build):
                    stack.append(chosen)
                    continue
                value, cost = (memo if dep[chosen] else build)[chosen], 1
            else:
                missing = [j for j in args if j not in (memo if dep[j] else build)]
                if missing:
                    stack.extend(reversed(missing))
                    continue
                vals = [(memo if dep[j] else build)[j] for j in args]
                try:
                    with np.errstate(all="ignore"):
                        value, cost = apply(node, vals, tokens if dep[i] else None, self.weights, store)
                except EntryError as exc:
                    raise EntryError(f"node {i} ({node['op']}): {exc}") from None
                except Exception as exc:  # whatever numpy made of it: the node failed, and its author is told which
                    raise EntryError(f"node {i} ({node['op']}): {type(exc).__name__}: {str(exc)[:300]}") from None
                if value.size > MAX_ELEMENTS or value.ndim > MAX_RANK:
                    raise EntryError(f"node {i} ({node['op']}): the value is too big")
            store[i] = value
            self._held[1 if dep[i] else 0] += int(value.size)
            if self._held[0] + self._held[1] > MAX_HELD:
                raise EntryError(f"the entry holds more than {MAX_HELD} numbers at once")
            if dep[i]:
                counter[0] += int(cost)
            else:
                self.build_cost += int(cost)
            stack.pop()
        return (memo if dep[root] else build)[root]

    def run(self, x) -> tuple:
        """(in the region?, the predicted output or None, operations of the rule, operations of the prediction)"""
        tokens = np.asarray(x, dtype=np.int64).reshape(-1)
        if len(tokens) != self.n_ctx:
            raise EntryError(f"an input has {self.n_ctx} tokens")
        memo: dict = {}
        self._held[1] = 0
        c_rule, c_pred = [0], [0]
        inside = True
        if self.rule is not None:
            v = self._eval(self.rule, memo, c_rule, tokens)
            if v.dtype != np.bool_ or v.size != 1:
                raise EntryError("the region test must give one boolean per input")
            inside = bool(v.reshape(()))
        answer = None
        if inside:
            v = self._eval(self.predict, memo, c_pred, tokens)
            if not np.issubdtype(v.dtype, np.integer) or v.size != 1:
                raise EntryError("the interpretation must give one whole number per input")
            answer = int(v.reshape(()))
        return inside, answer, c_rule[0], c_pred[0]


def all_inputs(input_space) -> np.ndarray:
    """Every input of a product space [n0, n1, ...], one row per input, in lexicographic order."""
    grids = np.indices([int(n) for n in input_space]).reshape(len(input_space), -1)
    return grids.T.astype(np.int64)


# data sets -----------------------------------------------------------------------------------
# A model says what its data set is (config.yaml: data_set). Two kinds:
#   all       every input of the input space, each as likely as any other (the default)
#   lengths   an input has a varying number of tokens: a number is drawn from `lengths` (each as likely as
#             any other), then that many tokens below `vocab`; they fill the first slots, and the rest hold
#             the value `empty`
def data_set_kind(data_set) -> str:
    kind = (data_set or {}).get("kind", "all")
    if kind not in ("all", "lengths"):
        raise ValueError(f"a data set is of kind all or lengths, not {kind!r}")
    return kind


def count_inputs(data_set, input_space) -> int:
    """How many different inputs the data set holds."""
    if data_set_kind(data_set) == "all":
        return int(np.prod([int(n) for n in input_space], dtype=object))
    return int(sum(int(data_set["vocab"]) ** int(k) for k in data_set["lengths"]))


def draw_inputs(data_set, input_space, n: int, rng) -> np.ndarray:
    """n inputs drawn from the data set, one row each. The verifier and the agent's sandbox both draw with
    this, from different seeds."""
    slots = len(input_space)
    if data_set_kind(data_set) == "all":
        return np.stack([rng.integers(0, int(k), size=n) for k in input_space], axis=1).astype(np.int64)
    lengths = np.array([int(k) for k in data_set["lengths"]])
    if lengths.min() < 1 or lengths.max() > slots:
        raise ValueError("a data set's lengths must fit the model's slots")
    length = rng.choice(lengths, size=n)
    x = rng.integers(0, int(data_set["vocab"]), size=(n, slots))
    x[np.arange(slots)[None, :] >= length[:, None]] = int(data_set["empty"])
    return x.astype(np.int64)


def wilson_lower(k: int, n: int, z: float = 1.96) -> float:
    """The lower end of the Wilson interval for k successes in n draws."""
    if n == 0:
        return 0.0
    p = k / n
    centre = p + z * z / (2 * n)
    spread = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, (centre - spread) / (1 + z * z / n))


def run_reference(reference: dict, weights: dict, n_ctx: int, inputs) -> tuple:
    """The network's own outputs on these inputs and what one forward pass costs: the network written as an
    entry, interpreted like any other. Its cost may not depend on the input, and it has no build."""
    ev = Evaluator(reference, weights, n_ctx)
    answers, costs = [], set()
    for x in inputs:
        inside, answer, c_rule, c_pred = ev.run(x)
        if not inside or c_rule:
            raise EntryError("the reference must answer every input, with no rule")
        answers.append(answer)
        costs.add(c_pred)
    if len(costs) != 1 or ev.build_cost:
        raise EntryError("the reference must cost the same on every input and compute nothing ahead")
    return np.array(answers, dtype=np.int64), costs.pop()


def measure(entry: dict, weights: dict, n_ctx: int, inputs, model_answers, model_ops: int,
            deadline: float | None = None) -> dict:
    """What the verifier measures of an entry on these inputs.

    model_answers and model_ops come from run_reference on the same inputs. The job is these inputs: the
    entry's build is done once for them, so it is spread over as many forward passes as there are inputs
    (all of the data set when every input is checked, the sample when one is drawn).

    cost is the entry's whole work per input of its region, over one forward pass: r + (r_rule + b) /
    coverage. With it, coverage * (1 - cost) is the share of the forward passes the entry saves."""
    ev = Evaluator(entry, weights, n_ctx, deadline)
    n = len(inputs)
    inside_n = agree = rule_ops = predict_ops = 0
    for x, truth in zip(inputs, model_answers):
        inside, answer, c_rule, c_pred = ev.run(x)
        rule_ops += c_rule
        if inside:
            inside_n += 1
            predict_ops += c_pred
            agree += int(answer == int(truth))
    f = float(model_ops)
    return {
        "n_inputs": n, "n_in_region": inside_n, "n_agree": agree,
        "model_ops": int(model_ops), "rule_ops": int(rule_ops), "predict_ops": int(predict_ops), "build_ops": int(ev.build_cost),
        "coverage": inside_n / n,
        "fidelity": (agree / inside_n) if inside_n else None,
        "cost": ((predict_ops + rule_ops + ev.build_cost) / (inside_n * f)) if inside_n else None,
        "r": (predict_ops / (inside_n * f)) if inside_n else None,
        "r_rule": rule_ops / (n * f),
        "b": ev.build_cost / (n * f),
    }


# the builder ---------------------------------------------------------------------------------
class Expr:
    """A node of a Graph being built. Its .value is the node's value on the graph's example input, so shapes
    are known and a mistake shows where it is made."""

    __slots__ = ("g", "id")
    __array_priority__ = 1000

    def __init__(self, g: "Graph", id: int):
        self.g, self.id = g, id

    @property
    def value(self) -> np.ndarray:
        return self.g.values[self.id]

    @property
    def shape(self) -> tuple:
        return self.value.shape

    @property
    def T(self) -> "Expr":
        return self.g.op("transpose", self, perm=None)

    def __repr__(self) -> str:
        return f"<Expr {self.id} {self.g.nodes[self.id]['op']} shape={self.shape}>"

    def __hash__(self) -> int:
        return hash((id(self.g), self.id))

    def __bool__(self):
        raise TypeError("an expression has no truth value while the entry is being built; use where or cond")

    def _b(self, op, other, swap=False):
        return self.g.op(op, other, self) if swap else self.g.op(op, self, other)

    def __add__(self, o): return self._b("add", o)
    def __radd__(self, o): return self._b("add", o, True)
    def __sub__(self, o): return self._b("sub", o)
    def __rsub__(self, o): return self._b("sub", o, True)
    def __mul__(self, o): return self._b("mul", o)
    def __rmul__(self, o): return self._b("mul", o, True)
    def __truediv__(self, o): return self._b("div", o)
    def __rtruediv__(self, o): return self._b("div", o, True)
    def __floordiv__(self, o): return self._b("floordiv", o)
    def __rfloordiv__(self, o): return self._b("floordiv", o, True)
    def __mod__(self, o): return self._b("mod", o)
    def __rmod__(self, o): return self._b("mod", o, True)
    def __matmul__(self, o): return self._b("matmul", o)
    def __rmatmul__(self, o): return self._b("matmul", o, True)
    def __lt__(self, o): return self._b("lt", o)
    def __le__(self, o): return self._b("le", o)
    def __gt__(self, o): return self._b("gt", o)
    def __ge__(self, o): return self._b("ge", o)
    def __eq__(self, o): return self._b("eq", o)
    def __ne__(self, o): return self._b("ne", o)
    def __and__(self, o): return self._b("and", o)
    def __rand__(self, o): return self._b("and", o, True)
    def __or__(self, o): return self._b("or", o)
    def __ror__(self, o): return self._b("or", o, True)
    def __invert__(self): return self.g.op("not", self)
    def __neg__(self): return self.g.op("neg", self)
    def __abs__(self): return self.g.op("abs", self)

    def reshape(self, *shape):
        shape = list(shape[0]) if len(shape) == 1 and isinstance(shape[0], (list, tuple)) else list(shape)
        return self.g.op("reshape", self, shape=[int(s) for s in shape])

    def astype(self, dtype):
        return cast(self, dtype)

    def sum(self, axis=None): return _reduce("sum", self, axis)
    def max(self, axis=None): return _reduce("max", self, axis)
    def min(self, axis=None): return _reduce("min", self, axis)
    def argmax(self, axis=None): return _reduce("argmax", self, axis)
    def argmin(self, axis=None): return _reduce("argmin", self, axis)
    def any(self, axis=None): return _reduce("any", self, axis)
    def all(self, axis=None): return _reduce("all", self, axis)

    def __getitem__(self, index):
        """x[...] with whole numbers, slices, None and expressions. Expressions that are one whole number read
        at those positions together (gather: 1 per entry read); one vector of whole numbers is allowed (take)."""
        items = list(index) if isinstance(index, tuple) else [index]
        if builtins.any(it is Ellipsis for it in items):
            raise TypeError("write the axes out; ... is not supported")
        has_new = builtins.any(it is None for it in items)
        if has_new and builtins.any(isinstance(it, Expr) for it in items):
            raise TypeError("None together with an expression index is not supported; reshape instead")
        static = []  # one item per indexed axis of this value: None (all of it), a whole number, or a slice
        for it in items:
            if it is None:
                continue
            if isinstance(it, Expr):
                static.append(None)
            elif isinstance(it, slice):
                spec = {}
                for k, v in (("start", it.start), ("stop", it.stop), ("step", it.step)):
                    if v is not None and (isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer))):
                        raise TypeError("slices take whole numbers")
                    spec[k] = None if v is None else int(v)
                static.append(None if builtins.all(v is None for v in spec.values()) else spec)
            elif isinstance(it, (int, np.integer)) and not isinstance(it, (bool, np.bool_)):
                static.append(int(it))
            else:
                raise TypeError(f"cannot index with {type(it).__name__}")
        if len(static) > self.value.ndim:
            raise IndexError(f"too many indices for a value of shape {self.shape}")
        cur = self
        if builtins.any(s is not None for s in static):
            cur = self.g.op("slice", self, index=static)
        pos, k = [], 0  # where each indexed axis sits after the static part (a whole number removed its axis)
        for s in static:
            if isinstance(s, int):
                pos.append(None)
            else:
                pos.append(k)
                k += 1
        indexed = [it for it in items if it is not None]
        dyn = [(pos[j], it) for j, it in enumerate(indexed) if isinstance(it, Expr)]
        scalars = [(a, e) for a, e in dyn if e.value.ndim == 0]
        vectors = [(a, e) for a, e in dyn if e.value.ndim != 0]
        if len(vectors) > 1:
            raise TypeError("at most one index may be a vector; use take for more")
        if scalars:
            cur = self.g.op("gather", cur, *[e for _, e in scalars], axes=[a for a, _ in scalars])
        if vectors:
            a, e = vectors[0]
            a -= builtins.sum(1 for sa, _ in scalars if sa < a)
            cur = self.g.op("take", cur, e, axis=a)
        if has_new:
            shape, dims, k = [], list(cur.value.shape), 0
            for it in items:
                if it is None:
                    shape.append(1)
                elif isinstance(it, slice):
                    shape.append(dims[k])
                    k += 1
            cur = cur.reshape(shape + dims[k:])
        return cur


class Graph:
    """Builds an entry. Give it the network's weights and the number of tokens per input; every expression
    is also computed on an example input (all zeros unless given), which is how shapes are known."""

    def __init__(self, weights: dict, n_ctx: int, example=None):
        self.weights = {k: _norm(v) for k, v in weights.items()}
        self.n_ctx = int(n_ctx)
        self.example = np.zeros(self.n_ctx, dtype=np.int64) if example is None else np.asarray(example, dtype=np.int64)
        if self.example.shape != (self.n_ctx,):
            raise ValueError(f"an example input has one whole number per position of the input: {self.n_ctx} here")
        self.nodes: list[dict] = []
        self.values: list[np.ndarray] = []
        self._seen: dict[str, int] = {}
        self._svd: dict = {}

    # sources
    def token(self, i: int) -> Expr:
        return self._add({"op": "token", "i": int(i)}, [])

    def tokens(self) -> Expr:
        return self._add({"op": "tokens"}, [])

    def weight(self, name: str) -> Expr:
        return self._add({"op": "weight", "name": str(name)}, [])

    def arange(self, n: int) -> Expr:
        return self._add({"op": "arange", "n": int(n)}, [])

    def const(self, value) -> Expr:
        a = np.asarray(value)
        if a.dtype == np.bool_:
            dtype = "bool"
        elif np.issubdtype(a.dtype, np.integer):
            dtype = "int"
        elif np.issubdtype(a.dtype, np.floating):
            dtype = "float"
        else:
            raise TypeError(f"a constant is a number or an array of numbers, not {a.dtype}")
        return self._add({"op": "const", "value": a.tolist(), "dtype": dtype}, [])

    def wrap(self, x) -> Expr:
        if isinstance(x, Expr):
            if x.g is not self:
                raise ValueError("expressions of two graphs cannot be mixed")
            return x
        return self.const(x)

    def op(self, name: str, *args, **attrs) -> Expr:
        args = [self.wrap(a) for a in args]
        node = {"op": name, "args": [a.id for a in args], **{k: v for k, v in attrs.items() if v is not None or k == "axis"}}
        if name in REDUCE and node.get("axis") is None:
            node["axis"] = None
        return self._add(node, args)

    def _add(self, node: dict, args: list) -> Expr:
        if "args" in node and not node["args"] and node["op"] in SOURCES:
            node = {k: v for k, v in node.items() if k != "args"}
        key = json.dumps(node, sort_keys=True)
        if key in self._seen:
            return Expr(self, self._seen[key])
        vals = [a.value for a in args]
        with np.errstate(all="ignore"):
            if node["op"] == "cond":
                c = vals[0]
                if c.dtype != np.bool_ or c.size != 1:
                    raise EntryError("cond: the condition must be one boolean")
                value = vals[1] if bool(c.reshape(())) else vals[2]
            else:
                value, _ = apply(node, vals, self.example, self.weights, self._svd)
        self.nodes.append(node)
        self.values.append(value)
        self._seen[key] = len(self.nodes) - 1
        return Expr(self, len(self.nodes) - 1)

    def entry(self, predict=None, rule=None, *, interpretation=None, region=None) -> dict:
        """The entry as JSON data: the nodes the two results need, renumbered. The interpretation may be
        given as `interpretation` or `predict`, the region test as `region` or `rule`; the stored entry
        names them predict and rule either way."""
        if (predict is None) == (interpretation is None):
            raise TypeError("an entry needs its interpretation, given once (interpretation=... or predict=...)")
        if rule is not None and region is not None:
            raise TypeError("give the region once (region=... or rule=...)")
        predict = interpretation if predict is None else predict
        rule = region if rule is None else rule
        predict = self.wrap(predict)
        roots = [predict.id] + ([self.wrap(rule).id] if rule is not None else [])
        need, stack = set(), list(roots)
        while stack:
            i = stack.pop()
            if i not in need:
                need.add(i)
                stack.extend(self.nodes[i].get("args", []))
        order = sorted(need)
        new = {old: k for k, old in enumerate(order)}
        nodes = []
        for old in order:
            node = dict(self.nodes[old])
            if "args" in node:
                node["args"] = [new[a] for a in node["args"]]
            nodes.append(node)
        entry = {"format": FORMAT, "nodes": nodes, "rule": new[roots[1]] if rule is not None else None, "predict": new[roots[0]]}
        check_structure(entry)
        return entry

    @classmethod
    def load(cls, entry: dict, weights: dict, n_ctx: int, example=None) -> tuple:
        """An entry opened for more building: (graph, predict, rule or None)."""
        check_structure(entry)
        g = cls(weights, n_ctx, example)
        exprs = []
        for node in entry["nodes"]:
            node = dict(node)
            args = [exprs[a] for a in node.get("args", [])]
            if "args" in node:
                node["args"] = [a.id for a in args]
            exprs.append(g._add(node, args))
        rule = entry.get("rule")
        return g, exprs[entry["predict"]], (exprs[rule] if rule is not None else None)


def _g(*xs) -> Graph:
    for x in xs:
        if isinstance(x, Expr):
            return x.g
        if isinstance(x, (list, tuple)):
            for y in x:
                if isinstance(y, Expr):
                    return y.g
    raise TypeError("at least one argument must be an expression of a graph")


def _ax(x: Expr, axis) -> int:
    nd = x.value.ndim
    axis = int(axis)
    if not -nd <= axis < nd:
        raise ValueError(f"axis {axis} does not fit a value with {nd} dimensions")
    return axis % nd


def _reduce(op: str, x, axis) -> Expr:
    g = _g(x)
    x = g.wrap(x)
    return g.op(op, x, axis=None if axis is None else _ax(x, axis))


def where(c, x, y) -> Expr: return _g(c, x, y).op("where", c, x, y)
def cond(c, x, y) -> Expr: return _g(c, x, y).op("cond", c, x, y)
def matmul(x, y) -> Expr: return _g(x, y).op("matmul", x, y)
def maximum(x, y) -> Expr: return _g(x, y).op("maximum", x, y)
def minimum(x, y) -> Expr: return _g(x, y).op("minimum", x, y)
def exp(x) -> Expr: return _g(x).op("exp", x)
def log(x) -> Expr: return _g(x).op("log", x)
def sqrt(x) -> Expr: return _g(x).op("sqrt", x)
def argmax(x, axis=None) -> Expr: return _reduce("argmax", x, axis)
def argmin(x, axis=None) -> Expr: return _reduce("argmin", x, axis)


def _along(op: str, x, axis) -> Expr:
    g = _g(x)
    x = g.wrap(x)
    return g.op(op, x, axis=_ax(x, axis))


def cumsum(x, axis=-1) -> Expr: return _along("cumsum", x, axis)
def sort(x, axis=-1) -> Expr: return _along("sort", x, axis)
def argsort(x, axis=-1) -> Expr: return _along("argsort", x, axis)
def softmax(x, axis=-1) -> Expr: return _along("softmax", x, axis)


def take(x, idx, axis=0) -> Expr:
    g = _g(x, idx)
    x = g.wrap(x)
    return g.op("take", x, idx, axis=_ax(x, axis))


def take_sum(x, idx, axis=0) -> Expr:
    g = _g(x, idx)
    x = g.wrap(x)
    return g.op("take_sum", x, idx, axis=_ax(x, axis))


def gather(x, idxs, axes) -> Expr:
    g = _g(x, idxs)
    x = g.wrap(x)
    return g.op("gather", x, *idxs, axes=[_ax(x, a) for a in axes])


def concat(xs, axis=0) -> Expr:
    g = _g(xs)
    xs = [g.wrap(x) for x in xs]
    return g.op("concat", *xs, axis=_ax(xs[0], axis))


def stack(xs, axis=0) -> Expr:
    g = _g(xs)
    xs = [g.wrap(x) for x in xs]
    nd = xs[0].value.ndim + 1
    return g.op("stack", *xs, axis=int(axis) % nd)


def reshape(x, shape) -> Expr: return _g(x).wrap(x).reshape(shape)
def transpose(x, perm=None) -> Expr: return _g(x).op("transpose", x, perm=None if perm is None else [int(p) for p in perm])


def cast(x, dtype) -> Expr:
    name = {int: "int", float: "float", bool: "bool"}.get(dtype, dtype)
    return _g(x).op("cast", x, dtype=name)


def svd(x) -> tuple:
    """(u, s, vt) of a matrix, as numpy's reduced decomposition. Charged once for the three."""
    g = _g(x)
    return g.op("svd_u", x), g.op("svd_s", x), g.op("svd_vt", x)
