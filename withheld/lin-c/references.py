"""Reference entries for lin-c, for reports and tests. Withheld from the agent: no tool reaches this folder.

Each function takes a graph g and the entry language lv1 and returns {"predict": ..., "rule": ... or absent}.
An empty slot holds EMPTY, which is above every token, so the smallest slot of an input is its smallest token;
tokens fill the first slots, so "at most two tokens" is "the third slot is empty".
"""

EMPTY = 64


def _product(g):
    return g.weight("unembed") @ (g.weight("hidden") @ g.weight("embed"))


def target_everywhere(g, lv1):
    """The task's rule on every input, trained on or not."""
    return {"predict": g.tokens().min()}


def target_on_two(g, lv1):
    """The task's rule, claimed only for the inputs the networks were trained on: exactly two tokens."""
    a, b = g.token(0), g.token(1)
    return {"predict": lv1.minimum(a, b), "rule": (b < EMPTY) & (g.token(2) == EMPTY)}


def target_up_to_two(g, lv1):
    """The task's rule on inputs of one or two tokens."""
    return {"predict": lv1.minimum(g.token(0), g.token(1)), "rule": g.token(2) == EMPTY}


def contraction(g, lv1):
    """The three matrices multiplied once; per input, the product's columns at the slots, added, and their
    argmax. Exact for any number of tokens."""
    return {"predict": lv1.take_sum(_product(g), g.tokens(), axis=1).argmax()}


def target_then_contraction(g, lv1):
    """The task's rule on one or two tokens, the contraction on three or four."""
    exact = lv1.take_sum(_product(g), g.tokens(), axis=1).argmax()
    return {"predict": lv1.cond(g.token(2) == EMPTY, lv1.minimum(g.token(0), g.token(1)), exact)}


ENTRIES = {"target_everywhere": target_everywhere, "target_on_two": target_on_two, "target_up_to_two": target_up_to_two,
           "contraction": contraction, "target_then_contraction": target_then_contraction}
