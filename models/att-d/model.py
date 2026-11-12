"""The network, written in the entry language.

An input is n_ctx slots. A slot holds a token, a whole number from 0 to d_vocab - 1, or it is empty,
which is written as the number d_vocab. An input has at least one token; its tokens fill the first
slots and the empty slots come after them.

The network is a transformer with one layer, one attention head and no MLP, no layer norm and no biases.
Its weights:

    embed     (d_vocab + 1) x d_model   a row per token; the last row, an empty slot's, is all zeros
    pos       n_ctx x d_model            a row per slot
    query     d_model x d_head
    key       d_model x d_head
    value     d_model x d_head
    out       d_head x d_model
    unembed   d_model x d_vocab

Each slot's vector is its token's row of embed plus its own row of pos. The network is read at the slot of
the last token given: that slot's vector times query, divided by the square root of d_head, is scored
against every given slot's vector times key (a dot product per slot; empty slots take no part); the softmax
of the scores weights the given slots' vectors times value; the sum, times out, is added to the read slot's
vector; that times unembed gives d_vocab numbers, and the output is the position of the highest (the first
one on a tie).
"""

import math


def reference(g, lv1):
    """The network's output on the graph's input. g is an lv1.Graph, lv1 the entry language."""
    t = g.tokens()
    embed, pos = g.weight("embed"), g.weight("pos")
    n_ctx, empty = pos.shape[0], embed.shape[0] - 1
    x = lv1.take(embed, t, axis=0) + pos                     # a vector per slot
    n = (t != empty).sum()                                    # the number of tokens given
    read = x[n - 1]                                           # the slot of the last token given
    q = (read @ g.weight("query")) * (1 / math.sqrt(g.weight("query").shape[1]))
    scores = (x @ g.weight("key")) @ q
    scores = lv1.where(g.arange(n_ctx) < n, scores, -1e30)    # empty slots take no part
    head = lv1.softmax(scores) @ (x @ g.weight("value"))
    logits = (read + head @ g.weight("out")) @ g.weight("unembed")
    return logits.argmax()
