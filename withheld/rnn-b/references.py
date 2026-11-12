"""Reference entries for rnn-b, for reports and tests. Withheld from the agent: no tool reaches this folder.

Each function takes a graph g and the entry language lv1 and returns {"predict": ..., "rule": ... or absent}.
The networks were trained to give the position of the second-largest number (task.yaml). The numbers are drawn
from a continuous distribution, so two of them are equal with probability zero and ties do not matter.
"""


def target(g, lv1):
    """The task's rule on every input: the position of the second-largest number, the last but one of the
    positions put in the order of their numbers."""
    return {"predict": lv1.argsort(g.tokens())[g.n_ctx - 2]}


def _steps(g, lv1, h, start):
    """The network's steps from position start on, from the state h before it, as models/rnn-b/model.py takes them."""
    w, rec = g.weight("input")[:, 0], g.weight("recurrent")
    for t in range(start, g.n_ctx):
        h = lv1.maximum(w * g.token(t) + rec @ h, 0.0)
    return h


def folded_first_step(g, lv1):
    """The network, exact on any weights, with its first step folded into its second: plain algebra. The state
    after x_0 is relu(input x_0), which is x_0 relu(input) when x_0 > 0 and x_0 (-relu(-input)) otherwise, so
    recurrent times it is x_0 times one of two vectors computed once."""
    w, rec = g.weight("input")[:, 0], g.weight("recurrent")
    up, down = rec @ lv1.maximum(w, 0.0), rec @ -lv1.maximum(-w, 0.0)
    x0 = g.token(0)
    h = lv1.maximum(w * g.token(1) + x0 * lv1.cond(x0 > 0, up, down), 0.0)
    return {"predict": (g.weight("readout") @ _steps(g, lv1, h, 2)).argmax()}


ENTRIES = {"target": target, "folded_first_step": folded_first_step}
