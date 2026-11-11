"""The sandbox as the agent's code meets it. The modes' own tests are the proof pipeline's; here: that the
vendored code runs, that Landlock (where there is one) keeps the withheld folder and the second network out
of reach, and that limits hold."""

import pytest

from level1 import run_config, sandbox
from level1.runner import Limits, run_recipe
from level1.session import Session

RECIPE = "def make_result(weights, info, extra):\n    import helper\n    return {'sum': helper.f(extra['x']), 'names': sorted(weights)}\n"


def test_a_recipe_runs_and_only_data_comes_back(model):
    ex = run_recipe(RECIPE, {"helper.py": "def f(x):\n    return x + 1\n"}, model.load_weights("lin-a-s0"), model.info(), {"x": 41},
                    Limits(time_s=60, memory_mb=2048, max_result_bytes=1 << 20))
    assert ex.status == "ok" and ex.result == {"sum": 42, "names": ["embed", "hidden", "unembed"]} and ex.mode == "process"
    with pytest.raises(ValueError, match="cannot be a file"):
        run_recipe(RECIPE, {"../escape.py": ""}, {}, {}, None, Limits(time_s=5, memory_mb=512, max_result_bytes=1024))
    with pytest.raises(ValueError, match="cannot be a file"):
        run_recipe(RECIPE, {"harness.py": ""}, {}, {}, None, Limits(time_s=5, memory_mb=512, max_result_bytes=1024))


def test_the_time_limit_holds(model):
    ex = run_recipe("def make_result(weights, info, extra):\n    while True:\n        pass\n", {}, model.load_weights("lin-a-s0"), model.info(),
                    None, Limits(time_s=1, memory_mb=2048, max_result_bytes=1 << 20))
    assert ex.status == "timeout"


landlock = pytest.mark.skipif(not sandbox.landlock_available(), reason="no Landlock on this machine")


@landlock
def test_landlock_keeps_everything_in(model, tmp_path):
    cfg = {"sandbox": "landlock", "runs_dir": str(tmp_path)}
    sb = sandbox.choose(cfg, sandbox.protected_paths(cfg, model, tmp_path / "run"))
    report = sandbox.probe(sb, model, "lin-a-s0", tmp_path / "run")
    assert sb.hardened and report["critical_escapes"] == [] and report["attempts"]["read_withheld"].startswith("blocked")
    sandbox.check_probe(sb, report)
    assert sandbox.choose({"sandbox": "hardened"}).hardened


@landlock
def test_the_agents_code_cannot_read_what_is_withheld(model, tmp_path):
    cfg = run_config.load("config/pilot/lin-a.yaml", {"runs_dir": str(tmp_path), "sandbox": "landlock"})
    session = Session.create(tmp_path / "run" / "round_001" / "session", cfg, "lin-a-s0", tmp_path / "run")
    code = f"""
import os
for path in ({str(model.withheld() / "task.yaml")!r}, {str(model.weights_path("lin-a-s1"))!r}, {str(model.path / "config.yaml")!r}):
    try:
        open(path).read()
        print("READ", path)
    except OSError as exc:
        print("blocked", type(exc).__name__)
print(sorted(os.listdir(".")))
g = graph()
print(measure(g.entry(predict=g.token(0)))["n_inputs"])
"""
    out = session.run_python(code)
    assert out["sandbox"] == "landlock" and "READ" not in out["output"] and out["output"].count("blocked PermissionError") == 3
    assert "['harness.py', 'input.json', 'lv1.py', 'model.py', 'out', 'recipe.py', 'rewards.py', 'tmp', 'weights.npz']" in out["output"]
    assert out["output"].strip().endswith("4096")


def test_the_requirements_install_on_every_python_the_pipeline_supports():
    """requirements.txt pins one numpy per Python: a pin that exists only for a recent Python made
    `pip install -r requirements.txt` fail on Python 3.10 (11 November 2026)."""
    lines = [line.split("#")[0].strip() for line in (sandbox.REPO_ROOT / "requirements.txt").read_text().splitlines()]
    numpy_lines = [line for line in lines if line.startswith("numpy")]
    assert len(numpy_lines) == 2 and all(";" in line for line in numpy_lines)         # no pin without its condition
    for python in ((3, 10), (3, 11), (3, 12), (3, 13), (3, 14)):
        holding = [line for line in numpy_lines if sandbox._marker_holds(line.partition(";")[2], python)]
        assert len(holding) == 1, python                                               # exactly one numpy per Python
    old, new = sandbox.requirement_versions((3, 10)), sandbox.requirement_versions((3, 13))
    assert old["numpy"] == "2.2.6" and new["numpy"] == "2.5.3"                       # numpy 2.2 is the last for Python 3.10
    assert {k: v for k, v in old.items() if k != "numpy"} == {k: v for k, v in new.items() if k != "numpy"}
    assert set(old) == {"numpy", "pyyaml", "psutil", "pytest"} and ";" not in "".join(old.values())
    assert sandbox._marker_holds("") and sandbox._marker_holds('sys_platform == "linux"')


# the guard inside Python ---------------------------------------------------------------------------
ESCAPES = '''
import os, sys

def attempt(name, fn, out):
    try:
        fn()
        out[name] = "reached"
    except BaseException as exc:
        out[name] = "blocked " + type(exc).__name__

def make_result(weights, info, extra):
    import numpy as np
    # reading is tried on the real withheld file; anything that would change a file is tried on decoys outside
    # the sandbox folder, so that nothing real is touched even if the guard failed
    secret, other, repo, decoy, elsewhere = extra["secret"], extra["other"], extra["repo"], extra["decoy"], extra["elsewhere"]
    if extra.get("only_read"):
        out = {}
        attempt("open", lambda: open(secret).read(), out)
        return {"escapes": out, "ordinary": {}}
    out, ok = {}, {}
    attempt("open", lambda: open(secret).read(), out)
    attempt("os.open", lambda: os.read(os.open(secret, os.O_RDONLY), 9), out)
    attempt("pathlib", lambda: __import__("pathlib").Path(secret).read_bytes(), out)
    attempt("numpy fromfile", lambda: np.fromfile(secret, dtype=np.uint8), out)
    attempt("numpy load", lambda: dict(np.load(other)), out)
    attempt("list the repository", lambda: os.listdir(repo), out)
    attempt("symlink", lambda: (os.symlink(secret, "tmp/s"), open("tmp/s").read()), out)
    attempt("hard link", lambda: (os.link(secret, "tmp/h"), open("tmp/h").read()), out)
    attempt("move it", lambda: os.rename(decoy, "tmp/m"), out)
    attempt("climb from a folder descriptor", lambda: os.open("../../" + os.path.relpath(secret, "/"), os.O_RDONLY,
                                                             dir_fd=os.open(sys.prefix, os.O_RDONLY)), out)
    attempt("subprocess", lambda: __import__("subprocess").run(["cat", secret]), out)
    attempt("os.system", lambda: os.system("true"), out)
    attempt("socket", lambda: __import__("socket").create_connection(("127.0.0.1", 9), timeout=1), out)
    attempt("ctypes", lambda: __import__("ctypes").CDLL(None), out)
    attempt("signal", lambda: os.kill(os.getppid(), 0), out)
    attempt("write outside", lambda: open(elsewhere, "w"), out)
    attempt("numpy", lambda: (weights["unembed"] @ weights["hidden"]).sum(), ok)
    attempt("standard library", lambda: [__import__(n) for n in ("collections", "itertools", "math", "statistics", "fractions")], ok)
    attempt("numpy's own parts", lambda: (np.linalg.svd(np.eye(3)), np.fft.fft(np.ones(4)), np.random.default_rng().integers(3)), ok)
    attempt("its own tmp", lambda: (np.save("tmp/a.npy", np.arange(3)), np.load("tmp/a.npy")), ok)
    attempt("tempfile", lambda: __import__("tempfile").NamedTemporaryFile(), ok)
    attempt("its own folder", lambda: (os.listdir("."), open("helper.py").read()), ok)
    return {"escapes": out, "ordinary": ok}
'''


def test_the_guard_blocks_and_records_what_the_code_has_no_business_doing(model, tmp_path):
    """In process mode nothing but the guard stands between the agent's code and your files: every way to the
    withheld file and the other network's weights that the code could take is refused, and recorded, while
    what exploring needs still works. The probe below tries the same and more."""
    secret = str(model.withheld() / "task.yaml")
    decoy, elsewhere = tmp_path / "decoy.txt", tmp_path / "written.txt"
    decoy.write_text("a decoy\n")
    extra = {"secret": secret, "other": str(model.weights_path("lin-a-s1")), "repo": str(sandbox.REPO_ROOT),
             "decoy": str(decoy), "elsewhere": str(elsewhere)}
    cfg = {"sandbox": "process"}
    sb = sandbox.choose(cfg, sandbox.protected_paths(cfg, model))
    ex = run_recipe(ESCAPES, {"helper.py": "x = 1\n"}, model.load_weights("lin-a-s0"), model.info(), extra,
                    Limits(time_s=60, memory_mb=2048, max_result_bytes=1 << 20), sandbox=sb)
    assert ex.status == "ok", ex.error
    assert {k: v for k, v in ex.result["escapes"].items() if not v.startswith("blocked")} == {}
    assert set(ex.result["ordinary"].values()) == {"reached"}, ex.result["ordinary"]
    assert ex.audit["complete"] and ex.audit["n_blocked"] >= len(ex.result["escapes"])
    events = {b["event"] for b in ex.audit["blocked"]}
    assert {"open", "os.listdir", "os.symlink", "os.link", "os.rename", "subprocess.Popen", "os.system", "os.kill"} <= events
    assert any(e.startswith("socket.") for e in events) and any(e.startswith("ctypes.") for e in events)
    assert any(b["target"] == secret for b in ex.audit["blocked"])
    assert (model.withheld() / "task.yaml").is_file() and decoy.read_text() == "a decoy\n" and not elsewhere.exists()
    # without the guard (only the probe ever asks for that) a read reaches the file in process mode
    bare = run_recipe(ESCAPES, {}, model.load_weights("lin-a-s0"), model.info(), {**extra, "only_read": True},
                      Limits(time_s=60, memory_mb=2048, max_result_bytes=1 << 20), sandbox=sb, guard=False)
    assert bare.result["escapes"]["open"] == "reached" and bare.audit == {"present": False, "complete": False, "n_blocked": 0, "blocked": []}


def test_the_probe_tries_the_system_alone_and_then_the_guard(model, tmp_path):
    cfg = {"sandbox": "process", "runs_dir": str(tmp_path)}
    sb = sandbox.choose(cfg, sandbox.protected_paths(cfg, model, tmp_path / "run"))
    alone = sandbox.probe(sb, model, "lin-a-s0", tmp_path / "run")
    guarded = sandbox.probe(sb, model, "lin-a-s0", tmp_path / "run", guard=True)
    assert not alone["guard"] and guarded["guard"]
    assert {"read_withheld", "read_withheld_numpy", "read_withheld_hardlink", "read_withheld_symlink", "read_withheld_subprocess",
            "read_withheld_ctypes", "write_outside"} <= set(alone["critical_escapes"])     # process mode on its own keeps nothing in
    assert guarded["critical_escapes"] == [] and guarded["minor_escapes"] == []
    sandbox.check_probe(sb, alone)                                                         # process mode is not hardened: no refusal
    sandbox.check_probe(sb, guarded)
    with pytest.raises(sandbox.SandboxError, match="the guard inside Python is not keeping"):
        sandbox.check_probe(sb, {**guarded, "critical_escapes": ["read_withheld"]})        # a guard that leaks stops the run
    assert not list((tmp_path / "run").iterdir()) and not list(sandbox.Path.home().glob(".lv1-probe-*"))


# seatbelt (macOS) ----------------------------------------------------------------------------------
def test_the_seatbelt_profile(model, tmp_path):
    """What the macOS profile allows, read as text (it can only be run on a Mac): later rules win, so reading
    the home folder and the protected folders is refused before Python's folders and the sandbox folder are
    allowed again; writing only to out/ and tmp/; no network, no other programs, no signals but to itself."""
    import os
    import sys

    box = tmp_path / "box"
    protected = sandbox.protected_paths({"runs_dir": str(tmp_path / "runs")}, model, tmp_path / "runs" / "r")
    profile = sandbox.seatbelt_profile(box, protected, sys.executable)
    rules = profile.splitlines()
    assert rules[:2] == ["(version 1)", "(allow default)"] and "(deny network*)" in rules and "(deny process-fork)" in rules
    assert "(deny process-exec)" in rules and f'(literal "{os.path.realpath(sys.executable)}")' in profile
    assert "(deny signal)" in rules and "(allow signal (target self))" in rules
    deny_read = next(i for i, r in enumerate(rules) if r.startswith("(deny file-read*"))
    allow_read = next(i for i, r in enumerate(rules) if r.startswith("(allow file-read* "))
    assert deny_read < allow_read and rules[-1] == "(allow file-read-metadata)"
    assert f'(subpath "{os.path.realpath(sandbox.Path.home())}")' in rules[deny_read]
    assert all(f'(subpath "{p}")' in rules[deny_read] for p in protected)
    assert all(f'(subpath "{p}")' not in rules[allow_read] for p in protected)
    assert f'(subpath "{os.path.realpath(sys.prefix)}")' in rules[allow_read] and f'(subpath "{box}")' in rules[allow_read]
    write = next(r for r in rules if r.startswith("(allow file-write*"))
    assert f'(subpath "{box / "out"}")' in write and f'(subpath "{box / "tmp"}")' in write and '(literal "/dev/null")' in write
    assert rules.index("(deny file-write*)") < rules.index(write)
    assert sandbox._sb('a "b" \\c') == '"a \\"b\\" \\\\c"'
    with pytest.raises(sandbox.SandboxError, match="would let recipes read"):
        sandbox.seatbelt_profile(box, (os.path.realpath(sys.prefix),))   # a protected folder Python must read: refused


@pytest.mark.skipif(sandbox.seatbelt_available(), reason="this is a Mac")
def test_seatbelt_mode_is_for_macs():
    with pytest.raises(sandbox.SandboxError, match="needs macOS"):
        sandbox.choose({"sandbox": "seatbelt"})
    assert sandbox.choose({"sandbox": "auto"}).mode != "seatbelt"


@pytest.mark.skipif(not sandbox.seatbelt_available(), reason="seatbelt is macOS only")
def test_seatbelt_keeps_everything_in(model, tmp_path):
    """Runs only on a Mac: the probe, with the guard left out, against the system's own sandbox."""
    cfg = {"sandbox": "seatbelt", "runs_dir": str(tmp_path)}
    sb = sandbox.choose(cfg, sandbox.protected_paths(cfg, model, tmp_path / "run"))
    report = sandbox.probe(sb, model, "lin-a-s0", tmp_path / "run")
    assert sb.hardened and report["critical_escapes"] == [], report
    ex = run_recipe(RECIPE, {"helper.py": "def f(x):\n    return x + 1\n"}, model.load_weights("lin-a-s0"), model.info(), {"x": 41},
                    Limits(time_s=60, memory_mb=2048, max_result_bytes=1 << 20), sandbox=sb)
    assert ex.status == "ok" and ex.result["sum"] == 42 and ex.audit["complete"], ex.error
