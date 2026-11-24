"""Every scoring of one entry on one domain in a run, taken together (run config: pool_samples).

When the verifier samples, each scoring draws its own inputs, so one entry scores a little differently each time.
Were the best submission simply taken, an entry submitted again and again would rise on luck alone
(tests/test_gaming.py). So in a run that pools, a submission of an entry that was scored before on the same domain
is scored on all of its samples together: the counts of the distinct samples are added up, and the coverage, the
fidelity and the cost are worked out from the sums. A fixed sample, or every input, is one sample however often it
is scored, so it is never counted twice. Each scoring's own build is spread over its own inputs, as when it stands
alone, so pooling leaves an entry's cost per input as it was.

The fidelity's confidence bound stays that of one sample: the pooled fidelity, on as many inputs as one scoring
has in the entry's region. Otherwise every submission again would narrow the bound and raise the reward without
predicting better, and an entry submitted often could not be compared with one submitted once. So submitting again
makes an entry's reward less noisy, and buys neither another chance nor a narrower bound.

The pooled submission keeps its own measurement under 'draw', and says under 'pooled' how many samples went in and
which ones (their seeds). It then stands for the entry: an earlier submission whose sample it took in is not
selected on its own (superseded). Runs made before pooling carry no pool_samples, so their records read as before.
"""

from __future__ import annotations

from . import lang, rewards

SUMMED = ("n_inputs", "n_in_region", "n_agree", "rule_ops", "predict_ops")


def enabled(cfg: dict) -> bool:
    return bool(cfg.get("pool_samples"))


def draw(receipt: dict) -> dict:
    """One scoring's own measurement: its parts, how its inputs were drawn and its reward."""
    if receipt.get("draw"):
        return receipt["draw"]
    return {"parts": receipt["parts"], "inputs": receipt["inputs"], "reward": receipt["reward"]}


def domain(receipt: dict) -> str:
    """The domain a receipt was scored on: a team session's scope, or all inputs."""
    return (draw(receipt)["inputs"] or {}).get("scope_id") or "all"


def _sample(how: dict) -> tuple:
    return (how.get("mode"), how.get("seed"), how.get("n"), how.get("scope_id"))


def pool_parts(parts_list: list[dict], sampled: bool) -> dict:
    """The parts of several distinct samples of one entry: counts added up, the bound that of one sample."""
    out = dict(parts_list[0])
    for key in SUMMED:
        out[key] = sum(int(p[key]) for p in parts_list)
    n, inside, f = out["n_inputs"], out["n_in_region"], float(out["model_ops"])
    out["coverage"] = inside / n
    out["fidelity"] = out["n_agree"] / inside if inside else None
    out["r"] = out["predict_ops"] / (inside * f) if inside else None
    out["r_rule"] = out["rule_ops"] / (n * f)
    out["b"] = sum(float(p["b"]) * int(p["n_inputs"]) for p in parts_list) / n  # each scoring's build over its own inputs
    out["cost"] = out["r"] + (out["r_rule"] + out["b"]) / out["coverage"] if inside else None
    if sampled and inside:  # the bound of one sample: the pooled fidelity on one scoring's share of the region
        one = inside / len(parts_list)
        out["fidelity_low"] = lang.wilson_lower(out["fidelity"] * one, one)
    else:
        out["fidelity_low"] = out["fidelity"]
    return out


def pooled(receipt: dict, earlier: list[dict], reward_cfg: dict) -> dict:
    """A scored receipt with the earlier measurements of the same entry on the same domain (draw dicts, oldest
    first) taken into it. Only distinct samples count; with none, the receipt comes back as it was."""
    own = draw(receipt)
    seen = {_sample(own["inputs"] or {})}
    use, seeds = [own["parts"]], [(own["inputs"] or {}).get("seed")]
    for d in earlier:
        key = _sample(d["inputs"] or {})
        if key not in seen:
            seen.add(key)
            use.append(d["parts"])
            seeds.append((d["inputs"] or {}).get("seed"))
    if len(use) == 1:
        return receipt
    parts = pool_parts(use, (own["inputs"] or {}).get("mode") == "sample")
    r = rewards.compute(parts, reward_cfg, receipt.get("claimed_fidelity"))
    out = {k: v for k, v in receipt.items() if k not in ("draw", "pooled")}
    out.update(draw={**own, "reward_detail": receipt.get("reward_detail")} if not receipt.get("draw") else receipt["draw"],
               parts=parts, reward=r["reward"], reward_detail={k: v for k, v in r.items() if k != "reward"},
               pooled={"samples": len(use), "n_inputs": parts["n_inputs"], "seeds": seeds})
    if receipt.get("claimed_fidelity") is not None and parts["fidelity"] is not None:
        out["overclaim"] = max(0.0, float(receipt["claimed_fidelity"]) - parts["fidelity"])
    return out


def superseded(receipts: list[dict]) -> dict:
    """{position: position of the later receipt} for every scored receipt, in a list in submission order, whose
    sample a later receipt of the same entry on the same domain took in. Those are not selected on their own."""
    out = {}
    for j, later in enumerate(receipts):
        seeds = set((later.get("pooled") or {}).get("seeds") or [])
        if not seeds:
            continue
        for i in range(j):
            r = receipts[i]
            if (r.get("status") == "scored" and r.get("entry_sha256") == later.get("entry_sha256")
                    and domain(r) == domain(later) and (draw(r)["inputs"] or {}).get("seed") in seeds):
                out[i] = j
    return out


def standing(receipts: list[dict]) -> list[dict]:
    """The scored receipts that can be selected: each entry's latest, with its samples pooled."""
    gone = superseded(receipts)
    return [r for i, r in enumerate(receipts) if r.get("status") == "scored" and i not in gone]
