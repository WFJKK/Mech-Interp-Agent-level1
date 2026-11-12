"""Train the networks of models/lin-c: the linear min-of-2 network. Withheld from the agent, like everything here.

    python withheld/lin-c/train.py [--seeds 0 1] [--work DIR]

The proof pipeline's linear network and training recipe (Compact-Proof-Agent, models/max2/model.py and
core/train.py), in numpy: the tokens' one-hot vectors summed, then three linear maps without biases (embed
d_model x d_vocab, hidden d_model x d_model, unembed d_vocab x d_model); cross-entropy against the smallest of
the 2 tokens, on fresh uniformly random inputs of exactly 2 tokens at every step; AdamW (lr 0.001, betas 0.9
and 0.999, eps 1e-8, weight decay 0.01, decoupled), batch 1,024; initial weights uniform in +-1/sqrt(fan_in),
as torch.nn.Linear draws them. Everything random comes from numpy.random.default_rng(seed).

How long: 4,000 steps, as the proof pipeline trained lin-a's networks (its max2-v64 zoo), the max-of-2
counterpart, so that the two pairs of networks differ in their task and their seeds only. Both of lin-a's
networks were then right on 99.9% or more of their two-token inputs; these are checked against the same
(Joshua, 11 November 2026, on training: "until the loss is acceptable or the performance").

Resumable: after every block the state (weights, AdamW moments, step, the random generator) is saved to
--work/<network>.ckpt.npz, and a stopped run continues from there. The finished weights go to
models/lin-c/zoo/<network>.npz, with embed given one more column, all zeros, for an empty slot; the
networks are asked about inputs of 1 to 4 tokens. Then update models/lin-c/config.yaml (sha256) and the
record in task.yaml.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
N_CTX, D_VOCAB, D_MODEL = 2, 64, 8
LR, BETAS, EPS, WD, BATCH = 1e-3, (0.9, 0.999), 1e-8, 0.01, 1024
BLOCK, STEPS = 1_000, 4_000


def init(rng):
    p = {"embed": rng.uniform(-1, 1, (D_MODEL, D_VOCAB)) / np.sqrt(D_VOCAB),
         "hidden": rng.uniform(-1, 1, (D_MODEL, D_MODEL)) / np.sqrt(D_MODEL),
         "unembed": rng.uniform(-1, 1, (D_VOCAB, D_MODEL)) / np.sqrt(D_MODEL)}
    return {k: v.astype(np.float32) for k, v in p.items()}


def step(P, m, v, t, rng):
    tok = rng.integers(0, D_VOCAB, (BATCH, N_CTX))
    y = tok.min(axis=1)
    cnt = np.zeros((BATCH, D_VOCAB), np.float32)
    np.add.at(cnt, (np.repeat(np.arange(BATCH), N_CTX), tok.ravel()), 1.0)
    h1 = cnt @ P["embed"].T
    h2 = h1 @ P["hidden"].T
    z = h2 @ P["unembed"].T
    z = z - z.max(axis=1, keepdims=True)
    p = np.exp(z)
    p /= p.sum(axis=1, keepdims=True)
    loss = float(-np.log(p[np.arange(BATCH), y] + 1e-30).mean())
    g = p
    g[np.arange(BATCH), y] -= 1
    g /= BATCH
    G = {"unembed": g.T @ h2}
    g2 = g @ P["unembed"]
    G["hidden"] = g2.T @ h1
    G["embed"] = (g2 @ P["hidden"]).T @ cnt
    b1, b2 = BETAS
    for k in P:
        P[k] *= (1 - LR * WD)
        m[k] = b1 * m[k] + (1 - b1) * G[k]
        v[k] = b2 * v[k] + (1 - b2) * G[k] ** 2
        P[k] -= (LR * (m[k] / (1 - b1 ** t)) / (np.sqrt(v[k] / (1 - b2 ** t)) + EPS)).astype(np.float32)
    return loss


def accuracy_on_two(P) -> float:
    """The share of all 64^2 inputs of 2 tokens on which the network's output is their smallest token."""
    E, H, U = (P[k].astype(np.float64) for k in ("embed", "hidden", "unembed"))
    M = U @ (H @ E)
    out = (M[:, :, None] + M[:, None, :]).argmax(0)
    r = np.arange(D_VOCAB)
    return float((out == np.minimum(r[:, None], r[None, :])).mean())


def train(seed: int, work: Path, log=print) -> dict:
    nid = f"lin-c-s{seed}"
    ckpt = work / f"{nid}.ckpt.npz"
    if ckpt.exists():
        z = np.load(ckpt, allow_pickle=False)
        P = {k: z[f"P_{k}"] for k in ("embed", "hidden", "unembed")}
        m = {k: z[f"m_{k}"] for k in P}
        v = {k: z[f"v_{k}"] for k in P}
        t = int(z["step"])
        history = json.loads(str(z["history"]))
        rng = np.random.default_rng(seed)
        rng.bit_generator.state = json.loads(str(z["rng"]))
        log(f"{nid}: resuming at step {t}")
    else:
        rng = np.random.default_rng(seed)
        P = init(rng)
        m = {k: np.zeros_like(x) for k, x in P.items()}
        v = {k: np.zeros_like(x) for k, x in P.items()}
        t, history = 0, []
    while t < STEPS:
        t0, losses = time.time(), []
        for _ in range(BLOCK):
            t += 1
            losses.append(step(P, m, v, t, rng))
        acc = accuracy_on_two(P)
        history.append({"step": t, "loss": round(float(np.mean(losses)), 5), "accuracy": acc})
        log(f"{nid}: step {t:6d}  loss {np.mean(losses):.4f}  smallest token on {acc:.5f} of all two-token inputs  ({time.time() - t0:.0f} s)")
        work.mkdir(parents=True, exist_ok=True)
        tmp = work / f"{nid}.ckpt.tmp.npz"
        np.savez(tmp, step=t, history=json.dumps(history), rng=json.dumps(rng.bit_generator.state),
                 **{f"P_{k}": x for k, x in P.items()}, **{f"m_{k}": x for k, x in m.items()}, **{f"v_{k}": x for k, x in v.items()})
        tmp.replace(ckpt)
    embed = np.concatenate([P["embed"], np.zeros((D_MODEL, 1), np.float32)], axis=1)  # the empty slot adds nothing
    out = REPO / "models" / "lin-c" / "zoo" / f"{nid}.npz"
    np.savez(out, embed=embed, hidden=P["hidden"], unembed=P["unembed"])
    return {"network": nid, "steps": t, "history": history, "file": str(out.relative_to(REPO))}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--work", default=str(Path.home() / "lin-c-training"))
    args = ap.parse_args(argv)
    for seed in args.seeds:
        r = train(seed, Path(args.work))
        print(json.dumps({k: r[k] for k in ("network", "steps", "file")}), "final", r["history"][-1], flush=True)


if __name__ == "__main__":
    main()
