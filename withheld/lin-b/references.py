"""Reference entries for lin-b, for reports and tests. Withheld from the agent: no tool reaches this folder.

Each function takes a graph g and the entry language lv1 and returns {"predict": ..., "rule": ... or absent}.
An empty slot holds EMPTY; tokens fill the first slots, so "at most k tokens" is "slot k is empty".
"""

EMPTY = 64


def _largest(g, lv1):
    """The largest token of an input with any number of tokens: an empty slot counts as 0 (64 mod 64)."""
    return (g.tokens() % EMPTY).max()


def _product(g):
    return g.weight("unembed") @ (g.weight("hidden") @ g.weight("embed"))


def _contraction(g, lv1):
    return lv1.take_sum(_product(g), g.tokens(), axis=1).argmax()


def target_everywhere(g, lv1):
    """The task's rule on every input, trained on or not."""
    return {"predict": _largest(g, lv1)}


def target_on_four(g, lv1):
    """The task's rule, claimed only for the inputs the networks were trained on: exactly four tokens."""
    t = [g.token(i) for i in range(4)]
    return {"predict": lv1.maximum(lv1.maximum(t[0], t[1]), lv1.maximum(t[2], t[3])),
            "rule": (t[3] < EMPTY) & (g.token(4) == EMPTY)}


def target_on_three_to_five(g, lv1):
    """The task's rule on inputs of three to five tokens, where it holds best."""
    return {"predict": _largest(g, lv1), "rule": (g.token(2) < EMPTY) & (g.token(5) == EMPTY)}


def contraction(g, lv1):
    """The three matrices multiplied once; per input, the product's columns at the slots, added, and their
    argmax. Exact for any number of tokens."""
    return {"predict": _contraction(g, lv1)}


def per_length(g, lv1):
    """One token: each token's own output, a table computed from the weights. Two to seven tokens: the task's
    rule. Eight tokens: the contraction."""
    T1 = _product(g).argmax(axis=0)
    one = g.token(1) == EMPTY
    eight = g.token(7) < EMPTY
    return {"predict": lv1.cond(one, T1[g.token(0)], lv1.cond(eight, _contraction(g, lv1), _largest(g, lv1)))}


ENTRIES = {"target_everywhere": target_everywhere, "target_on_four": target_on_four,
           "target_on_three_to_five": target_on_three_to_five, "contraction": contraction, "per_length": per_length}
