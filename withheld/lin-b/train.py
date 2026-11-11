"""Train the networks of models/lin-b: the linear max-of-4 network. Withheld from the agent, like everything here.

    python withheld/lin-b/train.py [--seeds 0 1] [--work DIR]

The proof pipeline's linear network and training recipe (Compact-Proof-Agent, models/max2/model.py and
core/train.py), in numpy: the tokens' one-hot vectors summed, then three linear maps without biases (embed
d_model x d_vocab, hidden d_model x d_model, unembed d_vocab x d_model); cross-entropy against the largest of
the 4 tokens, on fresh uniformly random inputs of exactly 4 tokens at every step; AdamW (lr 0.001, betas 0.9
and 0.999, eps 1e-8, weight decay 0.01, decoupled), batch 1,024; initial weights uniform in +-1/sqrt(fan_in),
as torch.nn.Linear draws them. Everything random comes from numpy.random.default_rng(seed).

How long: in blocks of 10,000 steps, until the share of all 64^4 four-token inputs on which the network
outputs their largest token gains less than 0.05 percentage points over 20,000 steps, at most 200,000 steps
(Joshua, 11 November 2026: "until the loss is acceptable or the performance"). The max-2 recipe's 4,000 steps
leave this network at 97.2%.

Resumable: after every block the state (weights, AdamW moments, step, the random generator) is saved to
--work/<network>.ckpt.npz, and a stopped run continues from there. The finished weights go to
models/lin-b/zoo/<network>.npz, with embed given one more column, all zeros, for an empty slot; the
networks are asked about inputs of 1 to 8 tokens. Then update models/lin-b/config.yaml (sha256) and the
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
N_CTX, D_VOCAB, D_MODEL = 4, 64, 8
LR, BETAS, EPS, WD, BATCH = 1e-3, (0.9, 0.999), 1e-8, 0.01, 1024
BLOCK, MAX_STEPS, PLATEAU_PP = 10_000, 200_000, 0.05


def init(rng):
    p = {"embed": rng.uniform(-1, 1, (D_MODEL, D_VOCAB)) / np.sqrt(D_VOCAB),
         "hidden": rng.uniform(-1, 1, (D_MODEL, D_MODEL)) / np.sqrt(D_MODEL),
         "unembed": rng.uniform(-1, 1, (D_VOCAB, D_MODEL)) / np.sqrt(D_MODEL)}
    return {k: v.astype(np.float32) for k, v in p.items()}


def step(P, m, v, t, rng):
    tok = rng.integers(0, D_VOCAB, (BATCH, N_CTX))
    y = tok.max(axis=1)
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


def accuracy_on_four(P) -> float:
    """The share of all 64^4 inputs of 4 tokens on which the network's output is their largest token."""
    E, H, U = (P[k].astype(np.float64) for k in ("embed", "hidden", "unembed"))
    M = U @ (H @ E)
    S3 = (M[:, :, None] + M[:, None, :])[:, :, :, None] + M[:, None, None, :]
    r = np.arange(D_VOCAB)
    mx3 = np.maximum(np.maximum(r[:, None, None], r[None, :, None]), r[None, None, :])
    right = 0
    for a in range(D_VOCAB):
        right += int(((S3 + M[:, a][:, None, None, None]).argmax(0) == np.maximum(mx3, a)).sum())
    return right / D_VOCAB ** 4


def train(seed: int, work: Path, log=print) -> dict:
    nid = f"lin-b-s{seed}"
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
    while True:
        if len(history) >= 3 and (history[-1]["accuracy"] - history[-3]["accuracy"]) * 100 < PLATEAU_PP:
            break
        if t >= MAX_STEPS:
            break
        t0, losses = time.time(), []
        for _ in range(BLOCK):
            t += 1
            losses.append(step(P, m, v, t, rng))
        acc = accuracy_on_four(P)
        history.append({"step": t, "loss": round(float(np.mean(losses)), 5), "accuracy": acc})
        log(f"{nid}: step {t:6d}  loss {np.mean(losses):.4f}  largest token on {acc:.5f} of all four-token inputs  ({time.time() - t0:.0f} s)")
        work.mkdir(parents=True, exist_ok=True)
        tmp = work / f"{nid}.ckpt.tmp.npz"
        np.savez(tmp, step=t, history=json.dumps(history), rng=json.dumps(rng.bit_generator.state),
                 **{f"P_{k}": x for k, x in P.items()}, **{f"m_{k}": x for k, x in m.items()}, **{f"v_{k}": x for k, x in v.items()})
        tmp.replace(ckpt)
    embed = np.concatenate([P["embed"], np.zeros((D_MODEL, 1), np.float32)], axis=1)  # the empty slot adds nothing
    out = REPO / "models" / "lin-b" / "zoo" / f"{nid}.npz"
    np.savez(out, embed=embed, hidden=P["hidden"], unembed=P["unembed"])
    return {"network": nid, "steps": t, "history": history, "file": str(out.relative_to(REPO))}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--work", default=str(Path.home() / "lin-b-training"))
    args = ap.parse_args(argv)
    for seed in args.seeds:
        r = train(seed, Path(args.work))
        print(json.dumps({k: r[k] for k in ("network", "steps", "file")}), "final", r["history"][-1], flush=True)


if __name__ == "__main__":
    main()
