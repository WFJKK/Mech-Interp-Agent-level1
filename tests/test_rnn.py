"""The recurrent models rnn-a to rnn-f, whose inputs are real numbers: ARC's AlgZoo 2nd argmax networks (one-layer ReLU
RNNs without biases, trained to give the position of the second-largest of n standard normal numbers) at three of
its sizes, and the same seeds before training as their baselines. What they are is in withheld/<name>/ and
withheld/rnn/; the networks are converted from AlgZoo's own files (withheld/rnn/pth) by withheld/rnn/convert.py."""

import hashlib
import importlib.util
import subprocess
from pathlib import Path

import numpy as np
import pytest

from level1 import baseline, hygiene, lang, loop, prompt, run_config
from level1.networks import ModelError, Model, load_model
from level1.util import read_yaml
from level1.verifier import Verifier, resolve_check

# Some Macs' numpy warns of an overflow, an invalid value or a division by zero in every matmul, with nothing
# wrong (docs/SANDBOX.md; the agent's sandbox drops the same three messages). Any other warning still shows.
pytestmark = pytest.mark.filterwarnings("ignore:(divide by zero|overflow|invalid value) encountered in matmul:RuntimeWarning")

REPO = Path(__file__).resolve().parent.parent
TRAINED = {"rnn-a": (2, 2), "rnn-b": (4, 3), "rnn-c": (16, 10)}   # name: (hidden units, numbers per input)
BASELINES = {"rnn-d": "rnn-a", "rnn-e": "rnn-b", "rnn-f": "rnn-c"}
SIZES = {**TRAINED, **{b: TRAINED[t] for b, t in BASELINES.items()}}
PILOT = {"rnn-a": ("s2", "s1"), "rnn-b": ("s0", "s4"), "rnn-c": ("s2", "s3")}  # rounds, seed check: the two best seeds
ALGZOO_CHART = {"rnn-a": 1.000, "rnn-b": 0.985, "rnn-c": 0.948}  # AlgZoo's README chart: its best seed, per size
ALGZOO_PICK = {"rnn-a": 2, "rnn-b": 0, "rnn-c": 2}                 # zoo.best_of_5_seed_2nd_argmax
TASK_WORDS = ["2nd argmax", "second argmax", "2nd_argmax", "second-largest", "second largest", "runner-up", "AlgZoo",
              "alg-zoo", "alg_zoo", "Alignment Research", "arc-ml"]
CFG = {"reward": {"name": "r_alpha", "alpha": 10.0}, "verifier": {"check": "sample", "sample_size": 3000}}


def _convert():
    spec = importlib.util.spec_from_file_location("rnn_convert", REPO / "withheld" / "rnn" / "convert.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def ops(d, n):
    """One forward pass: the first step (no recurrent product: the state before it is zero), n - 1 full steps,
    the readout and its argmax."""
    return 2 * d + (n - 1) * (d * d + 3 * d) + n * d + (n - 1)


def numpy_forward(w, x):
    """The network on inputs x (one row of numbers each), outside the entry language, in float64."""
    w = {k: v.astype(np.float64) for k, v in w.items()}
    h = np.zeros((len(x), w["recurrent"].shape[0]))
    for t in range(x.shape[1]):
        h = np.maximum(x[:, t:t + 1] * w["input"][:, 0] + h @ w["recurrent"].T, 0.0)
    return (h @ w["readout"].T).argmax(axis=1)


def second_largest(x):
    return np.argsort(x, axis=1)[:, -2]


@pytest.mark.parametrize("name", list(SIZES))
def test_sizes_data_set_networks_and_the_model_file(name):
    m = load_model(f"models/{name}")
    d, n = SIZES[name]
    assert m.real and m.n_ctx == n and m.n_total is None and not m.can_enumerate() and m.input_space == ["real"] * n
    assert resolve_check(m, {"check": "auto"}) == "sample"
    with pytest.raises(RuntimeError, match="drawn, not listed"):
        resolve_check(m, {"check": "all"})
    assert m.source() == load_model("models/rnn-a").source()  # one design
    x = lang.draw_inputs(m.data_set, m.input_space, 2000, np.random.default_rng(4))
    assert x.dtype == np.float64 and x.shape == (2000, n)
    assert len(m.config["networks"]) == 5
    for net in m.config["networks"]:
        w = m.load_weights(net["id"])  # the file's sha256 is checked against the config
        assert {k: v.shape for k, v in w.items()} == {"input": (d, 1), "recurrent": (d, d), "readout": (n, d)}
        assert all(v.dtype == np.float32 for v in w.values())
        answers, n_ops = lang.run_reference(m.reference_entry(w), w, n, x)
        assert n_ops == ops(d, n) and (np.asarray(answers) == numpy_forward(w, x)).all(), net["id"]
    assert {name: ops(*SIZES[name]) for name in TRAINED} == {"rnn-a": 19, "rnn-b": 78, "rnn-c": 2937}


def test_a_data_set_of_numbers_must_say_so_everywhere(tmp_path):
    """A model whose data set is numbers says `real` for every position and tells the agent its data set in words."""
    good = read_yaml(REPO / "models" / "rnn-a" / "config.yaml")
    for change, match in [({"input_space": ["real", 65]}, "`real` for every position"),
                          ({"input_space": [3, 3]}, "`real` for every position"),
                          ({"data_set": {"kind": "normal"}}, "data_set.says"),
                          ({"data_set": {"kind": "lengths", "lengths": [1, 2], "vocab": 3, "empty": 3}}, "`real` for every position")]:
        folder = tmp_path / f"m{len(list(tmp_path.iterdir()))}"
        folder.mkdir()
        import yaml
        (folder / "config.yaml").write_text(yaml.safe_dump({**good, **change}))
        with pytest.raises(ModelError, match=match):
            Model(folder)


@pytest.mark.parametrize("name", list(SIZES))
def test_the_networks_are_algzoo_s_own(name):
    """Each network is AlgZoo's file, converted again here and compared, and the file is the one task.yaml names."""
    conv = _convert()
    m = load_model(f"models/{name}")
    task = m.task()
    ids = [n["id"] for n in m.config["networks"]]
    assert sorted(ids) == [f"{name}-s{s}" for s in range(5)]
    hashes = task["provenance"]["sha256"] if name in TRAINED else {k: v["sha256"] for k, v in task["initialization"]["networks"].items()}
    for nid in ids:
        seed = int(nid[-1])
        path = conv.source(name, seed)
        assert hashlib.sha256(path.read_bytes()).hexdigest() == hashes[nid]
        assert f"_{conv.TRAINED if name in TRAINED else 0}_{seed}.pth" in path.name and path.name.startswith(f"2nd_argmax_{SIZES[name][0]}_{SIZES[name][1]}_")
        stored, again = m.load_weights(nid), conv.convert(name, seed)
        assert set(stored) == set(again) and all(np.array_equal(stored[k], again[k]) and stored[k].dtype == again[k].dtype for k in stored)


def test_the_converter_runs_nothing_from_the_files(tmp_path):
    """The .pth files are pickles: the converter's unpickler takes tensors and an ordered dict and refuses anything else."""
    import pickle
    import zipfile

    conv = _convert()

    class Boom:
        def __reduce__(self):
            return (print, ("ran",))

    bad = tmp_path / "bad.pth"
    with zipfile.ZipFile(bad, "w") as z:
        z.writestr("archive/data.pkl", pickle.dumps({"rnn.weight_ih_l0": Boom()}))
    with pytest.raises(pickle.UnpicklingError, match="refused to load builtins.print"):
        conv.load_state_dict(bad)


def _state_dict_file(path, size, stride, numbers):
    """A state-dict file in PyTorch's zip format, one float tensor over one storage, written without torch: the
    pickle names torch's classes by module and name, which a stand-in for torch, put in sys.modules, provides."""
    import collections
    import io
    import pickle
    import sys
    import types
    import zipfile

    torch, utils = types.ModuleType("torch"), types.ModuleType("torch._utils")

    class FloatStorage:
        pass

    def _rebuild_tensor_v2(*args):
        pass

    FloatStorage.__module__, FloatStorage.__qualname__ = "torch", "FloatStorage"
    _rebuild_tensor_v2.__module__, _rebuild_tensor_v2.__qualname__ = "torch._utils", "_rebuild_tensor_v2"
    torch.FloatStorage, utils._rebuild_tensor_v2 = FloatStorage, _rebuild_tensor_v2

    class Storage:
        pass

    class Tensor:
        def __reduce__(self):
            return (_rebuild_tensor_v2, (Storage(), 0, size, stride, False, collections.OrderedDict()))

    class Writer(pickle.Pickler):
        def persistent_id(self, obj):
            return ("storage", FloatStorage, "0", "cpu", len(numbers)) if isinstance(obj, Storage) else None

    saved = {k: sys.modules.get(k) for k in ("torch", "torch._utils")}
    sys.modules.update({"torch": torch, "torch._utils": utils})
    try:
        buf = io.BytesIO()
        Writer(buf, protocol=2).dump(collections.OrderedDict([("w", Tensor())]))
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("archive/data.pkl", buf.getvalue())
        z.writestr("archive/byteorder", "little")
        z.writestr("archive/data/0", np.asarray(numbers, dtype="<f4").tobytes())
    return path


def test_the_converter_reads_tensors_and_nothing_outside_them(tmp_path):
    conv = _convert()
    six = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
    w = conv.load_state_dict(_state_dict_file(tmp_path / "a.pth", (2, 3), (3, 1), six))["w"]
    assert w.dtype == np.float32 and w.tolist() == [[0.5, 1.5, 2.5], [3.5, 4.5, 5.5]]
    t = conv.load_state_dict(_state_dict_file(tmp_path / "b.pth", (3, 2), (1, 3), six))["w"]  # a transposed view
    assert t.tolist() == [[0.5, 3.5], [1.5, 4.5], [2.5, 5.5]]
    with pytest.raises(ValueError, match="outside its storage"):
        conv.load_state_dict(_state_dict_file(tmp_path / "c.pth", (3, 3), (3, 1), six))
    with pytest.raises(ValueError, match="outside its storage"):
        conv.load_state_dict(_state_dict_file(tmp_path / "d.pth", (2, 3), (300, 1), six))


@pytest.mark.parametrize("name", list(TRAINED))
def test_the_trained_networks_do_what_algzoo_says(name):
    """On the inputs task.yaml was measured on, each network agrees with the task as often as it says, and the best
    seed, AlgZoo's own pick, comes out at AlgZoo's chart (measured on another sample, in float32)."""
    m = load_model(f"models/{name}")
    task = m.task()
    x = np.random.default_rng(0).standard_normal((2**20, m.n_ctx))
    target = second_largest(x)
    acc = {f"{name}-s{s}": float(np.mean(numpy_forward(m.load_weights(f"{name}-s{s}"), x) == target)) for s in range(5)}
    assert all(abs(acc[k] - float(task["how_often_the_output_is_the_second_largest"][k])) < 1e-4 for k in acc), acc
    best = max(acc, key=acc.get)
    assert best == f"{name}-s{ALGZOO_PICK[name]}" and abs(acc[best] - ALGZOO_CHART[name]) < 0.002
    first, second = PILOT[name]
    assert sorted(acc.values())[-2:] == sorted([acc[f"{name}-{first}"], acc[f"{name}-{second}"]])  # the pilot uses the two best


@pytest.mark.parametrize("name", list(BASELINES))
def test_the_baseline_networks_are_the_starting_points_and_have_learned_nothing(name):
    """Each is AlgZoo's file of 0 training sequences: PyTorch's initialization, inside +-1/sqrt(hidden units), far from
    where training took it, and no better at the task than chance allows."""
    m, twin = load_model(f"models/{name}"), load_model(f"models/{BASELINES[name]}")
    d, n = SIZES[name]
    x = np.random.default_rng(0).standard_normal((2**16, n))
    target = second_largest(x)
    for s in range(5):
        w, t = m.load_weights(f"{name}-s{s}"), twin.load_weights(f"{BASELINES[name]}-s{s}")
        assert all(np.abs(v).max() <= 1 / np.sqrt(d) for v in w.values())
        assert all(np.linalg.norm(t[k] - w[k]) > 0.9 * np.linalg.norm(w[k]) for k in w)
        assert np.mean(numpy_forward(w, x) == target) < 0.5 + 0.05 * (n == 2) and np.mean(numpy_forward(w, x) == target) < 1.5 / n + 0.05
    assert read_yaml(REPO / "withheld" / name / "task.yaml")["task"].startswith("Untrained")


@pytest.mark.parametrize("name", list(BASELINES))
def test_a_baseline_model_is_its_twins_design_sizes_and_data_set(name):
    m, t = load_model(f"models/{name}"), load_model(f"models/{BASELINES[name]}")
    assert m.source() == t.source()
    for key in ("sizes", "input_space", "n_outputs", "data_set", "weights"):
        assert m.config[key] == t.config[key], key
    assert baseline.twins(m) == [t.name] and baseline.baseline_models(t) == [name]


@pytest.mark.parametrize("name", list(BASELINES))
def test_the_baseline_starts_where_the_twins_pilot_started(name):
    """The baseline model lists first the starting points of the networks its twin's pilot uses, so the baseline run
    (by its own config, or made for a run's config) works on where the run's network started and checks its seed on
    where the seed check's network started."""
    twin = BASELINES[name]
    pilot = run_config.load(f"config/pilot/{twin}.yaml")
    first, second = PILOT[twin]
    assert (pilot["network"], pilot["seed_check"]["network"]) == (f"{twin}-{first}", f"{twin}-{second}")
    made = baseline.config_for(pilot)
    assert (made["model"], made["network"], made["seed_check"]["network"]) == (f"models/{name}", f"{name}-{first}", f"{name}-{second}")
    own = run_config.load(f"config/pilot/{name}.yaml")
    assert (own["network"], own["seed_check"]["network"]) == (made["network"], made["seed_check"]["network"])
    assert {k for k in set(own) | set(made) if own.get(k) != made.get(k)} == {"run_id"}


@pytest.mark.parametrize("name", list(BASELINES))
def test_the_agent_is_given_what_a_run_on_the_twin_gives_it(name):
    """The same prompt, the same tools, files and information: only the weights differ."""
    twin = BASELINES[name]
    cfg, cfg_twin = run_config.load(f"config/pilot/{name}.yaml"), run_config.load(f"config/pilot/{twin}.yaml")
    assert {k for k in set(cfg) | set(cfg_twin) if cfg.get(k) != cfg_twin.get(k)} == {"run_id", "model", "network", "seed_check"}
    m, t = load_model(cfg["model"]), load_model(cfg_twin["model"])
    w, wt = m.load_weights(cfg["network"]), t.load_weights(cfg_twin["network"])
    x = lang.draw_inputs(m.data_set, m.input_space, 20, np.random.default_rng(0))
    o, o_twin = lang.run_reference(m.reference_entry(w), w, m.n_ctx, x)[1], lang.run_reference(t.reference_entry(wt), wt, t.n_ctx, x)[1]
    assert o == o_twin
    assert prompt.build(cfg, m, 1, [], o, False) == prompt.build(cfg_twin, t, 1, [], o_twin, False)
    assert loop.agent_surface(cfg, m) == loop.agent_surface(cfg_twin, t)


@pytest.mark.parametrize("name", list(SIZES))
def test_the_agent_is_not_told_what_the_networks_are(name):
    m = load_model(f"models/{name}")
    cfg = run_config.load(f"config/pilot/{name}.yaml")
    forbidden = loop.forbidden_strings(cfg, m)
    assert set(TASK_WORDS) | {"largest", "trained on", "Gross", "compact proof"} <= set(forbidden)
    if name in BASELINES:
        assert {"untrained", "initialization", "baseline"} <= set(forbidden)
    texts = loop.agent_surface(cfg, m)
    texts["prompt"] = prompt.build(cfg, m, 1, [], ops(*SIZES[name]), False)
    texts.update({f"model folder {p.name}": p.read_text() for p in m.path.iterdir() if p.is_file()})
    hygiene.check(texts, forbidden)
    d, n = SIZES[name]
    assert f"n_ctx = {n}, d_hidden = {d}, outputs: {n}" in texts["prompt"] and "16,384 inputs drawn afresh" in texts["prompt"]
    assert "standard normal distribution" in texts["prompt"] and '"kind": "normal"' in texts["sandbox info"]
    assert '"n_total": null' in texts["sandbox info"] and '"says"' not in texts["sandbox info"]
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
    assert "config/pilot/rnn-a.yaml" in texts and "models/rnn-c/config.yaml" in texts and "README.md" in texts
    words = TASK_WORDS + ["second biggest", "second highest", "next largest", "alignment.org"]
    assert hygiene.hits(texts, words) == []
    assert not any(rel.parts[0] not in ("withheld", "tests") and rel.suffix == ".pth" for rel in files)


@pytest.mark.parametrize("name", list(SIZES))
def test_the_network_itself_and_the_reference_entries(name):
    m = load_model(f"models/{name}")
    nets = [n["id"] for n in m.config["networks"]]
    v = Verifier(m, nets[0], CFG)
    itself = v.score(v.reference, seed=1)
    assert itself["parts"]["fidelity"] == 1.0 and itself["parts"]["cost"] == 1.0 and -0.02 < itself["reward"] < 0
    # the plain algebra: right on every network, at the same cost
    seen = set()
    for nid in nets:
        vn = Verifier(m, nid, CFG)
        r = vn.score(m.reference_entries(vn.weights)["folded_first_step"], seed=5)
        assert r["status"] == "scored" and r["parts"]["fidelity"] == 1.0, (nid, r.get("reason"))
        seen.add(round(r["reward"], 6))
    assert len(seen) == 1 and 0.05 < seen.pop() < 0.25
    if name in TRAINED:
        first = f"{name}-{PILOT[name][0]}"
        r = Verifier(m, first, CFG).score(m.reference_entries(m.load_weights(first))["target"], seed=5)
        assert r["status"] == "scored" and r["parts"]["cost"] < 0.11
        low = {"rnn-a": 0.85, "rnn-b": 0.65, "rnn-c": 0.35}[name]  # the task's rule, wrong where the network is
        assert low < r["reward"] < low + 0.15, r["reward"]


@pytest.mark.parametrize("name", list(SIZES))
def test_the_smoke_config_is_the_pilot_made_small(name):
    pilot, smoke = run_config.load(f"config/pilot/{name}.yaml"), run_config.load(f"config/pilot/{name}-smoke.yaml")
    assert smoke["rounds_max"] == 1 and smoke["session"] == {"python_calls": 8, "submissions": 2, "max_turns": 40, "seconds": 1200}
    assert all(smoke[k] == pilot[k] for k in set(pilot) - {"run_id", "rounds_max", "session"}) and smoke["run_id"] != pilot["run_id"]
    first, second = PILOT[BASELINES.get(name, name)]
    assert (pilot["model"], pilot["network"], pilot["seed_check"]["network"]) == (f"models/{name}", f"{name}-{first}", f"{name}-{second}")
    assert pilot["verifier"]["check"] == "sample" and pilot["verifier"]["sample_size"] == 16384


def test_a_scripted_run_on_numbers(tmp_path):
    """The whole loop on rnn-a with a scripted round written for inputs of numbers: it starts (the hygiene check and
    the sandbox probe pass), runs, locks, checks the seed and writes the report."""
    over = {"runs_dir": str(tmp_path), "backend": "fake", "sandbox": "process", "verifier": {"sample_size": 1500, "reference_samples": 2},
            "session": {"python_calls": 3, "submissions": 2}}
    cfg = run_config.load("config/pilot/rnn-a-smoke.yaml", over)
    lines = []
    path = loop.start(cfg, fake_scripts=REPO / "tests" / "fake_run_numbers", out=lines.append)
    assert (path / "report.md").is_file() and (path / "lock.json").is_file() and (path / "seed_check.json").is_file()
    printed = (path / "round_001" / "session" / "python" / "001.txt").read_text()
    assert "(1500, 2) float64 forward operations 19 check sample kind normal n_total None" in printed, printed
    receipts = [read_json(p) for p in sorted((path / "round_001" / "session" / "submissions").glob("*/receipt.json"))]
    assert [r["status"] for r in receipts] == ["scored", "scored"]
    top, half = (r["parts"] for r in receipts)
    assert top["coverage"] == 1.0 and top["fidelity"] < 0.01  # the highest number is never the network's answer here
    assert 0.4 < half["coverage"] < 0.6 and half["fidelity"] > 0.99  # where x_0 > 0, the lowest of two is
    report = (path / "report.md").read_text()
    assert "real-valued inputs" in report and "folded_first_step" in report and "target" in report
    assert any("seed check on the second network" in line for line in lines)


def read_json(p):
    import json
    return json.loads(Path(p).read_text())
