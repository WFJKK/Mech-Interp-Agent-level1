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


def target_with_pieces_near_ties(g, lv1):
    """The task's rule away from ties and, near a tie, the network as the linear map it is there: a short entry
    worked out from rnn-b-s0's weights (2026-11-27), read off the weights of whatever network it is scored on.

    The network has no biases, so its output depends only on the ratios p = x_0/x_2 and q = x_1/x_2. Away from
    the ties (q below lo, or above hi) the entry gives the second-largest by those ratios. Near a tie the network
    is linear in a few pieces, each named by a direction lying in it; the units on after each number (its masks)
    are read off the network's own states at that direction, and the piece's map from the input to outputs 1 and
    2 against output 0 is computed from the weights, so the answer there is the argmax of (0, map x).
      near x_1 = x_2: the piece of (-1, 0, 0) where x_0 < x_2, of (1, 0, 0) elsewhere. Between outputs 1 and 2
        the readout passes output 0, which gives a band of 0 around the tie.
      near x_0 = x_2 with x_1 below: the piece of (1, 0, 1) for x_2 > 0, of (-1, 0, -1) otherwise; its tie is
        tilted, p = 0.993 + 0.011 q on rnn-b-s0.
      near x_0 = x_2 with x_1 above: the rule, its tie moved to that piece's tie at x_1 = 0 (p = 0.993).
      the label 2 is the network's own output at (1, -1, 0).
    lo and hi, the two margins, are the only numbers it writes. On rnn-b-s0 it agrees with the network on 99.4%
    of inputs at 7.9 operations each; what it misses is mostly where all three numbers are close in ratio, where
    54 pieces share 0.7% of the inputs."""
    lo, hi = 0.92, 1.1
    zero = g.const(0.0)
    w, R, U = g.weight("input")[:, 0], g.weight("recurrent"), g.weight("readout")
    RT = R.T
    D = U[1:] - U[0]                                           # outputs 1 and 2 against output 0
    relu = lambda v: lv1.maximum(v, zero)  # noqa: E731
    # the states at the naming directions, [c = 0, c = a] x [a = 1, a = -1]: (1,0,0), (-1,0,0), (1,0,1), (-1,0,-1)
    S1 = lv1.stack([w, -w])
    H1 = relu(S1)
    H2 = relu(H1 @ RT)
    RH2 = H2 @ RT
    H3 = lv1.stack([relu(RH2), relu(RH2 + S1)])
    M1, M2, M3 = H1 > zero, H2 > zero, H3 > zero
    # within a piece the last state is va x_0 + vb x_1 + vc x_2
    VA = lv1.where(M3, lv1.where(M2, lv1.where(M1, w, zero) @ RT, zero) @ RT, zero)
    VB = lv1.where(M3, lv1.where(M2, w, zero) @ RT, zero)
    VC = lv1.where(M3, w, zero)
    MAPS = D @ lv1.stack([VA, VB, VC], axis=3)                 # [c type, a sign] x 2 outputs x 3 numbers
    tie_above = -MAPS[1, 0, 1, 2] / MAPS[1, 0, 1, 0]           # the tie of the piece of (1,0,1) at x_1 = 0, as p
    two = (U @ relu(relu(H1[0] @ RT - w) @ RT)).argmax()       # the network's output at (1, -1, 0)
    a, b, c = g.token(0), g.token(1), g.token(2)
    x = g.tokens()
    q, p = b / c, a / c
    pq = lv1.stack([p, q])
    zcol = zero.reshape([1])

    def linear(M):
        return lv1.concat([zcol, M @ x]).argmax()

    lo, hi = g.const(lo), g.const(hi)
    below = lv1.cond(p < lo, pq.argmax(), lv1.cond(p > hi, two, linear(lv1.cond(c > zero, MAPS[1, 0], MAPS[1, 1]))))
    above = lv1.cond(p < tie_above, two, pq.argmin())
    band = linear(lv1.cond(a < c, MAPS[0, 1], MAPS[0, 0]))
    return {"predict": lv1.cond(q < lo, below, lv1.cond(q > hi, above, band))}


ENTRIES = {"target": target, "folded_first_step": folded_first_step, "target_with_pieces_near_ties": target_with_pieces_near_ties}
