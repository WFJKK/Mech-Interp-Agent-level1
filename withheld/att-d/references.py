"""Reference entries for att-d, for reports and tests. Withheld from the agent: no tool reaches this folder.

Each function takes a graph g and the entry language lv1 and returns {"predict": ..., "rule": ... or absent}.
An empty slot holds EMPTY; tokens fill the first slots, so "all 4 tokens" is "the last slot holds a token".
"""

import math

EMPTY = 64
N_CTX = 4


def _task(g, lv1):
    """The smallest token: an empty slot (64) is above every token, so it never is the smallest."""
    return g.tokens().min()


def _network(g, lv1):
    """The network itself, as models/att-d/model.py writes it: read at the slot of the last token given."""
    t = g.tokens()
    x = lv1.take(g.weight("embed"), t, axis=0) + g.weight("pos")
    n = (t != EMPTY).sum()
    read = x[n - 1]
    q = (read @ g.weight("query")) * (1 / math.sqrt(g.weight("query").shape[1]))
    scores = lv1.where(g.arange(N_CTX) < n, (x @ g.weight("key")) @ q, -1e30)
    head = lv1.softmax(scores) @ (x @ g.weight("value"))
    return ((read + head @ g.weight("out")) @ g.weight("unembed")).argmax()


def _one_token_table(g, lv1):
    """Each token's output when it is the only one, computed once from the weights: the read slot attends to
    itself alone."""
    x = g.weight("embed")[0:EMPTY] + g.weight("pos")[0]
    return ((x + (x @ g.weight("value")) @ g.weight("out")) @ g.weight("unembed")).argmax(axis=1)


def target_everywhere(g, lv1):
    """The task's rule on every input, trained on or not."""
    return {"predict": _task(g, lv1)}


def target_on_full(g, lv1):
    """The task's rule, claimed only for the inputs the networks were trained on: every slot filled."""
    return {"predict": _task(g, lv1), "rule": g.token(N_CTX - 1) < EMPTY}


def table_then_target(g, lv1):
    """One token: the table computed from the weights. More tokens: the task's rule."""
    one = g.token(1) == EMPTY
    return {"predict": lv1.cond(one, _one_token_table(g, lv1)[g.token(0)], _task(g, lv1))}


def network_then_target(g, lv1):
    """Fewer than four tokens: the network itself. Four tokens: the task's rule."""
    return {"predict": lv1.cond(g.token(N_CTX - 1) < EMPTY, _task(g, lv1), _network(g, lv1))}


def per_length(g, lv1):
    """One token: the table computed from the weights. Two or three: the network itself. Four: the task's rule."""
    one = g.token(1) == EMPTY
    full = g.token(N_CTX - 1) < EMPTY
    return {"predict": lv1.cond(one, _one_token_table(g, lv1)[g.token(0)],
                                lv1.cond(full, _task(g, lv1), _network(g, lv1)))}


ENTRIES = {"target_everywhere": target_everywhere, "target_on_full": target_on_full, "table_then_target": table_then_target,
           "network_then_target": network_then_target, "per_length": per_length}
