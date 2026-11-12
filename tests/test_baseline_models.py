"""The baseline models lin-e, lin-f and att-e: networks of the shapes of lin-c, lin-d and att-c (and of lin-a-1to4,
lin-b and att-d, which share them) that were never trained, drawn the way their twins' trainers start
(withheld/<name>/draw.py). A run on one is a usual run in everything the agent is given, so what it achieves
there owes nothing to learning: it is the baseline that runs of that shape are read against."""

import importlib.util
import itertools
import subprocess
from pathlib import Path

import numpy as np
import pytest

from level1 import hygiene, lang, loop, prompt, run_config
from level1.networks import load_model
from level1.util import read_yaml
from level1.verifier import Verifier

# Some Macs' numpy warns of an overflow, an invalid value or a division by zero in every matmul, with nothing
# wrong (docs/SANDBOX.md; the agent's sandbox drops the same three messages). Any other warning still shows.
pytestmark = pytest.mark.filterwarnings("ignore:(divide by zero|overflow|invalid value) encountered in matmul:RuntimeWarning")

REPO = Path(__file__).resolve().parent.parent
BASELINES = {"lin-e": ("lin-c", "lin-a-1to4"), "lin-f": ("lin-d", "lin-b"), "att-e": ("att-c", "att-d")}
NETWORKS = [f"s{i}" for i in range(6)]
# What these networks are, in the words the agent must never meet. They are said in withheld/ and here only.
UNTRAINED_WORDS = ["untrained", "not trained", "never trained", "before training", "random weights", "randomly initialized",
                   "randomly initialised", "random init", "initialization", "initialisation"]
CFG = {"reward": {"name": "r_alpha", "alpha": 10.0}, "verifier": {"check": "sample", "sample_size": 3000}}


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _forward(name, w, x):
    """The network on inputs x (rows of slots, 64 for empty), outside the entry language, in float64."""
    w = {k: v.astype(np.float64) for k, v in w.items()}
    n = (x != 64).sum(axis=1)
    if name.startswith("lin"):
        return (w["unembed"] @ (w["hidden"] @ w["embed"]))[:, x].sum(axis=2).argmax(axis=0)
    out = np.empty(len(x), np.int64)
    for L in np.unique(n):
        rows = n == L
        res = w["embed"][x[rows, :L]] + w["pos"][None, :L]
        read = res[:, -1]
        q = (read @ w["query"]) / np.sqrt(w["query"].shape[1])
        s = np.einsum("bth,bh->bt", res @ w["key"], q)
        a = np.exp(s - s.max(axis=1, keepdims=True))
        a /= a.sum(axis=1, keepdims=True)
        out[rows] = ((read + np.einsum("bt,bth->bh", a, res @ w["value"]) @ w["out"]) @ w["unembed"]).argmax(axis=1)
    return out


@pytest.mark.parametrize("name", list(BASELINES))
def test_the_networks_are_drawn_as_the_withheld_folder_says(name):
    m = load_model(f"models/{name}")
    task = m.task()
    draw = _load(REPO / "withheld" / name / "draw.py", f"draw_{name.replace('-', '_')}")
    assert [n["id"] for n in m.config["networks"]] == [f"{name}-{s}" for s in NETWORKS]
    assert {k: v["seed"] for k, v in task["initialization"]["networks"].items()} == draw.SEEDS
    assert sorted(task["baseline_of"]) == sorted(BASELINES[name])
    trainer = _load(REPO / "withheld" / draw.TRAINER / "train.py", f"trainer_{draw.TRAINER.replace('-', '_')}")
    for nid, seed in draw.SEEDS.items():
        stored = m.load_weights(nid)  # the file's sha256 is checked against the config
        again = draw.draw(nid)
        assert set(stored) == set(again) and all(stored[k].dtype == again[k].dtype == np.float32 for k in stored)
        assert all(np.array_equal(stored[k], again[k]) for k in stored), nid
        assert seed not in (0, 1)  # not the starting point of a trained twin
        first = trainer.init(np.random.default_rng(seed))  # the trainer's own first step, read directly
        if name.startswith("lin"):
            assert np.array_equal(stored["embed"][:, :64], first["embed"])
            assert all(np.array_equal(stored[k], first[k]) for k in ("hidden", "unembed"))
        else:
            assert np.array_equal(stored["embed"][:64], first["E"])
            assert all(np.array_equal(stored[trainer.LEVEL1[k]], first[k]) for k in trainer.NAMES if k != "E")
    empty = stored["embed"][:, 64] if name.startswith("lin") else stored["embed"][64]
    assert not empty.any()  # an empty slot adds nothing, as in the trained twins


@pytest.mark.parametrize("name", list(BASELINES))
def test_a_baseline_model_is_its_twins_design_sizes_and_data_set(name):
    m = load_model(f"models/{name}")
    for twin in BASELINES[name]:
        t = load_model(f"models/{twin}")
        assert m.source() == t.source()
        for key in ("sizes", "input_space", "n_outputs", "data_set", "weights"):
            assert m.config[key] == t.config[key], (twin, key)


@pytest.mark.parametrize("name", list(BASELINES))
def test_the_agent_is_given_what_a_run_on_the_twin_gives_it(name):
    """The same prompt, the same tools, files and information: only the weights differ."""
    twin = BASELINES[name][0]
    cfg, cfg_twin = run_config.load(f"config/pilot/{name}.yaml"), run_config.load(f"config/pilot/{twin}.yaml")
    assert {k for k in set(cfg) | set(cfg_twin) if cfg.get(k) != cfg_twin.get(k)} == {"run_id", "model", "network", "seed_check"}
    m, t = load_model(cfg["model"]), load_model(cfg_twin["model"])
    w, wt = m.load_weights(cfg["network"]), t.load_weights(cfg_twin["network"])
    x = lang.draw_inputs(m.data_set, m.input_space, 20, np.random.default_rng(0))
    ops, ops_twin = lang.run_reference(m.reference_entry(w), w, m.n_ctx, x)[1], lang.run_reference(t.reference_entry(wt), wt, t.n_ctx, x)[1]
    assert ops == ops_twin
    assert prompt.build(cfg, m, 1, [], ops, False) == prompt.build(cfg_twin, t, 1, [], ops_twin, False)
    assert loop.agent_surface(cfg, m) == loop.agent_surface(cfg_twin, t)


@pytest.mark.parametrize("name", list(BASELINES))
def test_the_agent_is_not_told_what_the_networks_are(name):
    m = load_model(f"models/{name}")
    cfg = run_config.load(f"config/pilot/{name}.yaml")
    forbidden = loop.forbidden_strings(cfg, m)
    assert set(UNTRAINED_WORDS) | {"baseline", "trained on", "Gross"} <= set(forbidden)
    texts = loop.agent_surface(cfg, m)
    texts["prompt"] = prompt.build(cfg, m, 1, [], 1000, False)
    texts.update({f"model folder {p.name}": p.read_text() for p in m.path.iterdir() if p.is_file()})
    hygiene.check(texts, forbidden)
    assert len(hygiene.hits({p.name: p.read_text() for p in m.withheld().iterdir() if p.is_file()}, forbidden)) > 3


def test_what_the_networks_are_is_said_in_withheld_and_the_tests_only():
    r = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard"], cwd=REPO, capture_output=True, text=True)
    files = [Path(f) for f in r.stdout.splitlines()] if r.returncode == 0 and r.stdout.strip() else []
    if not files:
        pytest.skip("no git to list the repository's files")
    texts = {}
    for rel in files:
        if rel.parts[0] not in ("withheld", "tests") and rel.suffix != ".npz" and (REPO / rel).is_file():
            try:
                texts[rel.as_posix()] = (REPO / rel).read_text()
            except UnicodeDecodeError:
                pass
    assert "config/pilot/att-e.yaml" in texts and "models/att-e/config.yaml" in texts
    assert hygiene.hits(texts, UNTRAINED_WORDS) == []


@pytest.mark.parametrize("name", list(BASELINES))
def test_the_networks_have_learned_nothing(name):
    """Where their trained twins answer with the largest or the smallest token, these do not."""
    m = load_model(f"models/{name}")
    trained_on = 2 if name == "lin-e" else 4  # the inputs the twins were trained on
    x = np.full((4000, m.n_ctx), 64)
    x[:, :trained_on] = np.random.default_rng(1).integers(0, 64, (4000, trained_on))
    toks = x[:, :trained_on]
    for s in NETWORKS[:2]:
        y = _forward(name, m.load_weights(f"{name}-{s}"), x)
        assert np.mean(y == toks.max(axis=1)) < 0.2 and np.mean(y == toks.min(axis=1)) < 0.2
    for twin in BASELINES[name]:
        t = load_model(f"models/{twin}")
        y = _forward(twin, t.load_weights(f"{twin}-s0"), x)
        assert max(np.mean(y == toks.max(axis=1)), np.mean(y == toks.min(axis=1))) > 0.95


@pytest.mark.parametrize("name", list(BASELINES))
def test_the_references_are_right_on_any_weights_and_score_the_same_everywhere(name):
    """The withheld references use only the weights' algebra: on the baseline's own networks and on the
    trained twins' alike they are right on every input and cost the same."""
    m = load_model(f"models/{name}")
    best = "tables" if name == "att-e" else "contraction"
    seen = set()
    for model, nid in [(name, f"{name}-{s}") for s in NETWORKS] + [(t, f"{t}-s{i}") for t in BASELINES[name] for i in (0, 1)]:
        v = Verifier(load_model(f"models/{model}"), nid, CFG)
        r = v.score(m.reference_entries(v.weights)[best], seed=5)
        assert r["status"] == "scored" and r["parts"]["fidelity"] == 1.0, (nid, r.get("reason"))
        seen.add(round(r["reward"], 6))
    assert len(seen) == 1
    if name == "att-e":  # 0.9548 at the pilot's 16,384 inputs; on 3,000 the build weighs more
        assert 0.90 < seen.pop() < 0.92
    v = Verifier(m, f"{name}-s0", CFG)
    itself = v.score(v.reference, seed=1)
    assert itself["parts"]["fidelity"] == 1.0 and itself["parts"]["cost"] == 1.0 and -0.02 < itself["reward"] < 0


@pytest.mark.parametrize("name", list(BASELINES))
def test_the_smoke_config_is_the_baseline_pilot_made_small(name):
    pilot, smoke = run_config.load(f"config/pilot/{name}.yaml"), run_config.load(f"config/pilot/{name}-smoke.yaml")
    assert smoke["rounds_max"] == 1 and smoke["session"] == {"python_calls": 8, "submissions": 2, "max_turns": 40, "seconds": 1200}
    assert all(smoke[k] == pilot[k] for k in set(pilot) - {"run_id", "rounds_max", "session"}) and smoke["run_id"] != pilot["run_id"]
    assert (pilot["model"], pilot["network"], pilot["seed_check"]["network"]) == (f"models/{name}", f"{name}-s0", f"{name}-s1")
    twin_smoke = run_config.load(f"config/pilot/{BASELINES[name][0]}-smoke.yaml")
    assert {k for k in set(smoke) | set(twin_smoke) if smoke.get(k) != twin_smoke.get(k)} == {"run_id", "model", "network", "seed_check"}


def test_a_scripted_run_on_a_baseline_model(tmp_path):
    """The whole loop on att-e with the scripted round that fits any model: it starts (the hygiene check and the
    sandbox probe pass), runs, locks, checks the seed and writes the report."""
    over = {"runs_dir": str(tmp_path), "backend": "fake", "sandbox": "process", "verifier": {"sample_size": 1500, "reference_samples": 2},
            "session": {"python_calls": 3, "submissions": 2}}
    cfg = run_config.load("config/pilot/att-e-smoke.yaml", over)
    lines = []
    path = loop.start(cfg, fake_scripts=REPO / "tests" / "fake_run_any", out=lines.append)
    assert (path / "report.md").is_file() and (path / "lock.json").is_file() and (path / "seed_check.json").is_file()
    assert any("seed check on the second network" in line for line in lines)
    report = (path / "report.md").read_text()
    assert "tables" in report and "Untrained" in report  # the withheld references and what the networks are, for people
    assert read_yaml(REPO / "withheld" / "att-e" / "task.yaml")["task"].startswith("Untrained")
