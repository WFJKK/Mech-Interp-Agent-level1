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
