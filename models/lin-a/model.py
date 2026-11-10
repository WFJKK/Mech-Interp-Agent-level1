"""The network lin-a, written in the entry language.

An input is n_ctx tokens, each a whole number from 0 to d_vocab - 1. The network has three weight
matrices and no biases:

    embed     d_model x d_vocab
    hidden    d_model x d_model
    unembed   d_vocab x d_model

It adds up the columns of embed at the input's tokens, applies hidden, applies unembed, and its
output is the position of the highest of the d_vocab resulting numbers (the first one on a tie).
"""


def reference(g, lv1):
    """The network's output on the graph's input. g is an lv1.Graph, lv1 the entry language."""
    h = lv1.take_sum(g.weight("embed"), g.tokens(), axis=1)
    logits = g.weight("unembed") @ (g.weight("hidden") @ h)
    return logits.argmax()
