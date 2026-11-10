"""Reference entries for lin-a, for reports and tests. Withheld from the agent: no tool reaches this folder.

Each function takes a graph g and the entry language lv1 and returns {"predict": ..., "rule": ... or absent}.
The report scores them next to the agent's best entry, so that a score can be read against what is known.
"""

CORNER = 3  # the networks' outputs differ from the task only where both tokens are below this


def target(g, lv1):
    """The task itself: the larger token. Never looks at the weights."""
    return {"predict": lv1.maximum(g.token(0), g.token(1))}


def target_outside_corner(g, lv1):
    """The task, claimed only where both networks follow it: at least one token is CORNER or more."""
    m = lv1.maximum(g.token(0), g.token(1))
    return {"predict": m, "rule": m >= CORNER}


def target_with_network_in_corner(g, lv1, reference):
    """The task everywhere, except in the corner of small tokens, where the network itself is run."""
    m = lv1.maximum(g.token(0), g.token(1))
    return {"predict": lv1.cond(m < CORNER, reference(g, lv1), m)}


def contracted(g, lv1):
    """The three matrices multiplied once; per input, two columns of the product, added, and their argmax."""
    table = g.weight("unembed") @ (g.weight("hidden") @ g.weight("embed"))
    return {"predict": lv1.take_sum(table, g.tokens(), axis=1).argmax()}


def two_candidates(g, lv1):
    """The product once; per input, only the two tokens' own logits are compared."""
    table = g.weight("unembed") @ (g.weight("hidden") @ g.weight("embed"))
    a, b = g.token(0), g.token(1)
    return {"predict": lv1.where(table[b, a] + table[b, b] > table[a, a] + table[a, b], b, a)}


def answer_table(g, lv1):
    """Brute force moved into the build: every input's output computed ahead, then looked up."""
    table = g.weight("unembed") @ (g.weight("hidden") @ g.weight("embed"))
    n = table.shape[0]
    logits = table.reshape(n, n, 1) + table.reshape(n, 1, n)
    return {"predict": logits.argmax(axis=0)[g.token(0), g.token(1)]}


ENTRIES = {"target": target, "target_outside_corner": target_outside_corner, "contracted": contracted,
           "two_candidates": two_candidates, "answer_table": answer_table}
ENTRIES_WITH_REFERENCE = {"target_with_network_in_corner": target_with_network_in_corner}
