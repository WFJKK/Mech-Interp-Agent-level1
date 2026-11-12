"""The network, written in the entry language.

An input is n_ctx real numbers, x_0 to x_(n_ctx - 1). The network reads them one at a time, in that order.
In an entry, g.token(t) is x_t, as a float, and g.tokens() is all of them; the output is still a whole number.

The network is a recurrent network with one layer of d_hidden units, a ReLU and no biases. Its weights:

    input       d_hidden x 1
    recurrent   d_hidden x d_hidden
    readout     n_outputs x d_hidden

Its state h, a vector of d_hidden numbers, starts at zero. At each position t, in order, it becomes

    h = relu(input x_t + recurrent h)

where relu sets every negative number to zero. After the last position, readout h gives n_outputs numbers,
and the output is the position of the highest (the first one on a tie).
"""


def reference(g, lv1):
    """The network's output on the graph's input. g is an lv1.Graph, lv1 the entry language."""
    w, rec = g.weight("input")[:, 0], g.weight("recurrent")
    h = lv1.maximum(w * g.token(0), 0.0)  # the first position: the state before it is zero, and so is recurrent h
    for t in range(1, g.n_ctx):
        h = lv1.maximum(w * g.token(t) + rec @ h, 0.0)
    return (g.weight("readout") @ h).argmax()
