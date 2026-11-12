"""Train the networks of models/att-d: the transformer min-of-4. Withheld from the agent, like everything here.

    python withheld/att-d/train.py [--seeds 0 1] [--work DIR]

The proof pipeline's attention network and training recipe (Compact-Proof-Agent, models/attn-max4/model.py,
config.yaml and core/train.py), in numpy, as attn-max4's v64 networks were trained (models/att-c): a
one-layer, one-head, attention-only transformer with no biases and learned positional embeddings, read at
the last position, n_ctx 4, d_vocab 64, d_model = d_head = 32. Each position's residual is its token's
row of E plus its row of P; the last position's query, scaled by 1/sqrt(d_head), scores every position's
key; the softmax of the scores weights the values; the head's output goes through WO onto the last
position's residual; U gives one logit per token. Cross-entropy against the smallest of the 4 tokens, on a
fresh batch of 512 uniformly random inputs of 4 tokens at every step (drawn column by column, as the
proof pipeline draws them); AdamW (lr 0.001, betas 0.9 and 0.999, eps 1e-8, weight decay 0.01, decoupled);
4,000 steps; float32. Initial weights uniform in +-1/sqrt(shape[0]) for every matrix, in the order E, P,
WQ, WK, WV, WO, U, as the proof pipeline's model draws them. Everything random comes from
numpy.random.default_rng(seed), so the seeds are not the proof pipeline's: torch drew its weights.

Resumable: every 500 steps the state (weights, AdamW moments, step, the random generator) is saved to
--work/<network>.ckpt.npz, and a stopped run continues from there. The finished weights go to
models/att-d/zoo/<network>.npz under the level-1 names (embed with one more row, all zeros, for an empty
slot; pos, query, key, value, out, unembed). Then update models/att-d/config.yaml (sha256) and the record
in task.yaml.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
N_CTX, D_VOCAB, D_MODEL, D_HEAD = 4, 64, 32, 32
LR, BETAS, EPS, WD, BATCH, STEPS, EVERY = 1e-3, (0.9, 0.999), 1e-8, 0.01, 512, 4000, 500
NAMES = ("E", "P", "WQ", "WK", "WV", "WO", "U")
SHAPES = {"E": (D_VOCAB, D_MODEL), "P": (N_CTX, D_MODEL), "WQ": (D_MODEL, D_HEAD), "WK": (D_MODEL, D_HEAD),
          "WV": (D_MODEL, D_HEAD), "WO": (D_HEAD, D_MODEL), "U": (D_MODEL, D_VOCAB)}
LEVEL1 = {"E": "embed", "P": "pos", "WQ": "query", "WK": "key", "WV": "value", "WO": "out", "U": "unembed"}


def init(rng, shapes=SHAPES, dtype=np.float32):
    return {k: ((rng.random(shapes[k]) * 2 - 1) * shapes[k][0] ** -0.5).astype(dtype) for k in NAMES}


def target(tok):
    return tok.min(axis=1)


def forward(P, tok):
    """Logits at the last position, and what the backward pass needs."""
    n_ctx = tok.shape[1]
    scale = P["WQ"].shape[1] ** -0.5
    res = P["E"][tok] + P["P"][None, :n_ctx]                 # (B, T, d)
    last = res[:, -1]                                         # (B, d)
    q = (last @ P["WQ"]) * scale                              # (B, h)
    K = res @ P["WK"]                                         # (B, T, h)
    V = res @ P["WV"]                                         # (B, T, h)
    s = np.einsum("bth,bh->bt", K, q)
    s = s - s.max(axis=1, keepdims=True)
    a = np.exp(s)
    a /= a.sum(axis=1, keepdims=True)                         # (B, T)
    head = np.einsum("bt,bth->bh", a, V)                      # (B, h)
    out = last + head @ P["WO"]                               # (B, d)
    return out @ P["U"], (res, last, q, K, V, a, head, out, scale)


def loss_and_grads(P, tok, y):
    logits, (res, last, q, K, V, a, head, out, scale) = forward(P, tok)
    B = len(y)
    z = logits - logits.max(axis=1, keepdims=True)
    p = np.exp(z)
    p /= p.sum(axis=1, keepdims=True)
    loss = float(-np.log(p[np.arange(B), y] + 1e-30).mean())
    dlog = p
    dlog[np.arange(B), y] -= 1
    dlog /= B
    G = {"U": out.T @ dlog}
    dout = dlog @ P["U"].T
    dlast = dout.copy()
    dhead = dout @ P["WO"].T
    G["WO"] = head.T @ dout
    da = np.einsum("bh,bth->bt", dhead, V)
    dV = a[:, :, None] * dhead[:, None, :]
    ds = a * (da - (a * da).sum(axis=1, keepdims=True))
    dK = ds[:, :, None] * q[:, None, :]
    dq = np.einsum("bt,bth->bh", ds, K) * scale
    G["WQ"] = last.T @ dq
    dlast += dq @ P["WQ"].T
    G["WK"] = np.einsum("btd,bth->dh", res, dK)
    G["WV"] = np.einsum("btd,bth->dh", res, dV)
    dres = dK @ P["WK"].T + dV @ P["WV"].T
    dres[:, -1] += dlast
    G["P"] = np.zeros_like(P["P"])
    G["P"][: tok.shape[1]] = dres.sum(axis=0)
    G["E"] = np.zeros_like(P["E"])
    np.add.at(G["E"], tok.ravel(), dres.reshape(-1, dres.shape[-1]))
    return loss, G


def adamw(P, G, m, v, t):
    b1, b2 = BETAS
    for k in NAMES:
        P[k] *= (1 - LR * WD)
        m[k] = b1 * m[k] + (1 - b1) * G[k]
        v[k] = b2 * v[k] + (1 - b2) * G[k] ** 2
        P[k] -= (LR * (m[k] / (1 - b1 ** t)) / (np.sqrt(v[k] / (1 - b2 ** t)) + EPS)).astype(P[k].dtype)


def draw(rng, n):
    return np.stack([rng.integers(0, D_VOCAB, size=n) for _ in range(N_CTX)], axis=1).astype(np.int64)


def accuracy(P, batch=1 << 16) -> float:
    """The share of all 64^4 inputs of 4 tokens on which the network's output is their smallest token."""
    right, total = 0, D_VOCAB ** N_CTX
    for start in range(0, total, batch):
        idx = np.arange(start, min(start + batch, total))
        tok = np.stack([(idx // D_VOCAB ** (N_CTX - 1 - i)) % D_VOCAB for i in range(N_CTX)], axis=1)
        logits, _ = forward(P, tok)
        right += int((logits.argmax(axis=1) == target(tok)).sum())
    return right / total


def train(seed: int, work: Path, log=print) -> dict:
    nid = f"att-d-s{seed}"
    ckpt = work / f"{nid}.ckpt.npz"
    if ckpt.exists():
        z = np.load(ckpt, allow_pickle=False)
        P = {k: z[f"P_{k}"] for k in NAMES}
        m = {k: z[f"m_{k}"] for k in NAMES}
        v = {k: z[f"v_{k}"] for k in NAMES}
        t, losses = int(z["step"]), json.loads(str(z["losses"]))
        rng = np.random.default_rng(seed)
        rng.bit_generator.state = json.loads(str(z["rng"]))
        log(f"{nid}: resuming at step {t}")
    else:
        rng = np.random.default_rng(seed)
        P = init(rng)
        m = {k: np.zeros_like(x) for k, x in P.items()}
        v = {k: np.zeros_like(x) for k, x in P.items()}
        t, losses = 0, []
    t0, block = time.time(), []
    while t < STEPS:
        t += 1
        tok = draw(rng, BATCH)
        loss, G = loss_and_grads(P, tok, target(tok))
        adamw(P, G, m, v, t)
        block.append(loss)
        if t % EVERY == 0:
            losses.append({"step": t, "loss": round(float(np.mean(block)), 6)})
            log(f"{nid}: step {t:5d}  loss {np.mean(block):.5f}  ({time.time() - t0:.0f} s)")
            block, t0 = [], time.time()
            work.mkdir(parents=True, exist_ok=True)
            tmp = work / f"{nid}.ckpt.tmp.npz"
            np.savez(tmp, step=t, losses=json.dumps(losses), rng=json.dumps(rng.bit_generator.state),
                     **{f"P_{k}": x for k, x in P.items()}, **{f"m_{k}": x for k, x in m.items()},
                     **{f"v_{k}": x for k, x in v.items()})
            tmp.replace(ckpt)
    acc = accuracy(P)
    log(f"{nid}: smallest token on {acc:.6f} of all four-token inputs")
    out = REPO / "models" / "att-d" / "zoo" / f"{nid}.npz"
    level1 = {LEVEL1[k]: P[k] for k in NAMES}
    level1["embed"] = np.concatenate([P["E"], np.zeros((1, D_MODEL), P["E"].dtype)], axis=0)  # an empty slot
    np.savez(out, **level1)
    return {"network": nid, "steps": t, "losses": losses, "accuracy": acc, "file": str(out.relative_to(REPO))}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--work", default=str(Path.home() / "att-d-training"))
    args = ap.parse_args(argv)
    for seed in args.seeds:
        r = train(seed, Path(args.work))
        print(json.dumps({k: r[k] for k in ("network", "steps", "accuracy", "file")}), "last", r["losses"][-1], flush=True)


if __name__ == "__main__":
    main()
