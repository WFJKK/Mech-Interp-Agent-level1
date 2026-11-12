"""Reference entries for att-e, for reports and tests. Withheld from the agent: no tool reaches this folder.

Each function takes a graph g and the entry language lv1 and returns {"predict": ..., "rule": ... or absent}.
att-e's networks have nothing learned in them (task.yaml), so its references use nothing but the weights'
algebra, which holds for any network of this design: they are right on any weights, up to rounding.
An empty slot holds EMPTY; tokens fill the first slots, so an input has L tokens when slot L is empty.
"""

import math

EMPTY = 64


def _network(g, lv1):
    """The network itself, as models/att-e/model.py writes it: read at the slot of the last token given."""
    t = g.tokens()
    n_ctx = g.weight("pos").shape[0]
    x = lv1.take(g.weight("embed"), t, axis=0) + g.weight("pos")
    n = (t != EMPTY).sum()
    read = x[n - 1]
    q = (read @ g.weight("query")) * (1 / math.sqrt(g.weight("query").shape[1]))
    scores = lv1.where(g.arange(n_ctx) < n, (x @ g.weight("key")) @ q, -1e30)
    head = lv1.softmax(scores) @ (x @ g.weight("value"))
    return ((read + head @ g.weight("out")) @ g.weight("unembed")).argmax()


def _one_token_table(g, lv1):
    """Each token's output when it is the only one, computed once from the weights: the read slot attends to
    itself alone."""
    x = g.weight("embed")[0:EMPTY] + g.weight("pos")[0]
    return ((x + (x @ g.weight("value")) @ g.weight("out")) @ g.weight("unembed")).argmax(axis=1)


def one_token_table_then_network(g, lv1):
    """One token: the table computed from the weights. More: the network itself."""
    return {"predict": lv1.cond(g.token(1) == EMPTY, _one_token_table(g, lv1)[g.token(0)], _network(g, lv1))}


def _per_slot(g, w, k):
    """(n_ctx, rows of embed, k): row t of slot s is (embed[t] + pos[s]) @ w, as embed @ w + pos @ w."""
    e, p = g.weight("embed"), g.weight("pos")
    return (e @ w).reshape(1, e.shape[0], k) + (p @ w).reshape(p.shape[0], 1, k)


def tables(g, lv1):
    """The forward pass rewritten as tables of the weights, built once: per (slot, token) the query, the key,
    the row the token adds through value, out and unembed, and the row it adds directly through unembed; per
    length L, the scores of the read slot's token against every (slot, token) of the first L slots. Per input:
    L scores read and softmaxed, L value rows mixed by them, the read token's direct row added, the argmax.
    One token: a table of answers. Exact for any weights."""
    e, p = g.weight("embed"), g.weight("pos")
    n_ctx, rows = p.shape[0], e.shape[0]
    dh, n_out = g.weight("query").shape[1], g.weight("unembed").shape[1]
    q = _per_slot(g, g.weight("query") * (1 / math.sqrt(dh)), dh)
    k = _per_slot(g, g.weight("key"), dh)
    vou = _per_slot(g, (g.weight("value") @ g.weight("out")) @ g.weight("unembed"), n_out)
    direct = _per_slot(g, g.weight("unembed"), n_out)
    answers = {1: (direct[0] + vou[0]).argmax(axis=1)[g.token(0)]}
    for L in range(2, n_ctx + 1):
        r = L - 1
        scores = q[r] @ k[0:L].reshape(L * rows, dh).T                      # (rows, L * rows)
        read = g.token(r)
        s = lv1.stack([lv1.gather(scores, [read, g.token(j) + j * rows], axes=[0, 1]) for j in range(L)])
        mixed = lv1.softmax(s) @ lv1.stack([vou[j][g.token(j)] for j in range(L)])
        answers[L] = (direct[r][read] + mixed).argmax()
    out = answers[n_ctx]
    for L in range(n_ctx - 1, 0, -1):
        out = lv1.cond(g.token(L) == EMPTY, answers[L], out)
    return {"predict": out}


ENTRIES = {"one_token_table_then_network": one_token_table_then_network, "tables": tables}
