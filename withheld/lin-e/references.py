"""Reference entries for lin-e, for reports and tests. Withheld from the agent: no tool reaches this folder.

Each function takes a graph g and the entry language lv1 and returns {"predict": ..., "rule": ... or absent}.
lin-e's networks have nothing learned in them (task.yaml), so its references use nothing but the weights'
algebra, which holds for any network of this design.
"""


def _product(g):
    return g.weight("unembed") @ (g.weight("hidden") @ g.weight("embed"))


def contraction(g, lv1):
    """The three matrices multiplied once; per input, the product's columns at the slots, added, and their
    argmax. Exact for any weights and any number of tokens."""
    return {"predict": lv1.take_sum(_product(g), g.tokens(), axis=1).argmax()}


ENTRIES = {"contraction": contraction}
