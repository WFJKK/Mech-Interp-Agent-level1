An entry is a program with two results per input:

- its **region**: a boolean. Is this input one the entry speaks about? Leave it out and the region is
  every input.
- its **interpretation**: a whole number, the network's output on this input as the entry predicts
  it. It is only computed for inputs in the region.

An entry is a graph of operations on tensors, stored as JSON data. You write it in Python with the
entry language `lv1`, and the verifier interprets the JSON; nothing of an entry is run as code, so
only the operations below exist.

### Writing one

```python
g = graph()                      # a new lv1.Graph on this network's weights
a, b = g.token(0), g.token(1)    # the input's tokens, whole numbers
t = g.tokens()                   # all of them, as a vector
W = g.weight("embed")            # a weight matrix, by name
c = g.const([1.0, 2.0])          # a constant; plain Python numbers and numpy arrays become constants too
r = g.arange(8)                  # 0, 1, ..., 7

x = W[:, a] * 2.0 + W[:, b]      # arithmetic and comparisons as in numpy, with broadcasting
entry = g.entry(interpretation=(a + b) % info["n_outputs"], region=a < 8)   # only an example of the format
save_entry("example", entry)     # keep it; submit_entry takes the name
print(measure(entry))            # coverage, fidelity, cost and the reward, as the verifier measures them
```

Every expression is also computed on an example input (all tokens 0) while you build, so `x.shape`
and `x.value` exist and a wrong shape fails at the line that makes it. `graph(example=[3, 5])`
builds on another example input, given as one number per position of the input.
`lv1.Graph.load(entry, weights, n_ctx)` opens a saved entry again and returns
`(graph, interpretation, region)`. In the stored JSON the interpretation is called `predict` and
the region `rule`.

Values are whole numbers, floats or booleans (64 bits). There are no loops and no Python `if` on
expressions: use `lv1.where(c, x, y)` (both sides are computed) or `lv1.cond(c, x, y)` (c is one
boolean, and only the chosen side is computed and charged).

### Operations and what they cost

Each operation on numbers costs 1. Per entry of the result unless said otherwise:

| Operation | Cost |
| --- | --- |
| `+ - * / // %`, unary `-`, `abs(x)`, `lv1.maximum`, `lv1.minimum`, `lv1.exp`, `lv1.log`, `lv1.sqrt` | 1 |
| `< <= > >= == !=`, `&`, `\|`, `~` (booleans) | 1 |
| `lv1.where(c, x, y)` | 1 |
| `lv1.cond(c, x, y)` | 1 in all, plus the chosen side |
| `x @ y`, `lv1.matmul(x, y)` | 1 per multiply-add: result entries times the contracted length |
| `x.sum(axis)`, `.max`, `.min`, `.argmax`, `.argmin`, `.any`, `.all` over n entries | n - 1 |
| `lv1.cumsum(x, axis)` | n - 1 per vector |
| `lv1.sort(x, axis)`, `lv1.argsort(x, axis)` of n entries | n * ceil(log2 n) per vector |
| `lv1.softmax(x, axis)` | 3 |
| `x[i]`, `x[i, j]`, `x[:, i]` with expressions that are one whole number (`lv1.gather`) | 1 per entry read |
| `x[v]` or `lv1.take(x, v, axis)` with a vector of whole numbers | 1 per entry read |
| `lv1.take_sum(x, v, axis)`: the sum of the slices of x at the positions v | slice entries times len(v) |
| `lv1.svd(x)` of an m x n matrix: `(u, s, vt)` | 4 * m * n * min(m, n), once for the three |
| `lv1.cast(x, "int")`, `"float"`, `"bool"` | 1 |
| `g.token`, `g.tokens`, `g.weight`, `g.const`, `g.arange` | 0 |
| `x.reshape(...)`, `x.T`, `lv1.transpose`, `x[2]`, `x[1:5]`, `x[None]`, `lv1.concat`, `lv1.stack` | 0 |

Reading costs 1 per number read: `T[a, b]` reads one number, `T[:, a]` a whole column.

### What an entry's cost is made of

Everything an entry computes is counted, in three parts:

- **Build.** Whatever does not depend on the input is computed once, when first needed. A table
  computed from the weights is paid for in full here.
- **Region test.** What the region computes from the input, charged for every input the verifier
  runs, in the region or not.
- **Interpretation.** What the interpretation computes from the input, charged for the inputs in the
  region. A value the region test already computed for an input is not charged again.

The entry's cost is the three together, per input of the region, as a share of one forward pass.
`measure(entry)` gives it as `cost`, and the parts as `b`, `r_rule` and `r`.

### Limits

The run limits an entry's size and the time the verifier gives it: the numbers are under "Tools,
budgets and limits" in your message. A constant counts every number it holds, and no single value
may hold more than 4,000,000 numbers. The verifier refuses an entry, and says why, when on any
input an index is out of range, shapes do not fit, the region is not one boolean, the
interpretation is not one whole number, or a float that is not finite is cast to a whole number.
An interpretation that gives something other than one of the network's outputs counts as a wrong
answer.
