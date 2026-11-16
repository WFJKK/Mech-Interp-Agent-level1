"""Reference entries for lin-b-4, for reports and tests. Withheld from the agent: no tool reaches this folder.

Each function takes a graph g and the entry language lv1 and returns {"predict": ..., "rule": ... or absent}.
Every input of this data set has four tokens and no empty slot.
"""


def _product(g):
    return g.weight("unembed") @ (g.weight("hidden") @ g.weight("embed"))


def target_everywhere(g, lv1):
    """The task's rule on every input: the largest of the four tokens."""
    return {"predict": g.tokens().max()}


def contraction(g, lv1):
    """The three matrices multiplied once; per input, the product's columns at the four slots, added, and their
    argmax. Exact for any weights."""
    return {"predict": lv1.take_sum(_product(g), g.tokens(), axis=1).argmax()}


ENTRIES = {"target_everywhere": target_everywhere, "contraction": contraction}
