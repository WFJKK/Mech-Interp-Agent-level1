"""Role tools, sandbox evaluation and prompt boundaries for scoped research."""

import copy

import pytest

from level1 import hygiene, loop, prompt, run_config
from level1.session import Session, ToolError
from level1.util import read_json, write_json


def config(tmp_path, **changes):
    cfg = run_config.load("config/experiments/lin-c-regional.yaml", {
        "runs_dir": str(tmp_path), "sandbox": "process", "backend": "fake",
        "session": {"python_calls": 10, "submissions": 4},
    })
    cfg.update(changes)
    return cfg


def test_lead_assigns_a_frozen_scope_but_workers_cannot(tmp_path):
    cfg = config(tmp_path, research_task={"role": "lead", "max_assignments": 2, "cycle": 0})
    lead = Session.create(tmp_path / "lead", cfg, cfg["network"], tmp_path / "run")
    assert "assign_region" in lead.TOOLS
    assigned = lead.call("assign_region", {"scope": {"kind": "lengths", "lengths": [2]},
                                           "task": "Investigate a simple rule and its exceptions."})
    assert assigned["status"] == "assigned"
    reopened = Session(lead.path)
    assert reopened.state["assignments"][0]["scope"] == {"kind": "lengths", "lengths": [2]}
    assert reopened.state["reward"] == cfg["reward"]
    worker_cfg = copy.deepcopy(cfg)
    worker_cfg["research_task"] = {"role": "regional", "max_assignments": 0, "cycle": 1}
    worker = Session.create(tmp_path / "worker", worker_cfg, cfg["network"], tmp_path / "run")
    assert "assign_region" not in worker.TOOLS
    with pytest.raises(ToolError, match="no tool"):
        worker.call("assign_region", {"scope": {"kind": "all"}, "task": "Change the task"})
    with pytest.raises(ToolError):
        lead.assign_region({"kind": "all", "workload_size": 10**12}, "Discount setup")
    for invalid_kind in ([], {}):
        with pytest.raises(ToolError, match="scope.kind"):
            lead.assign_region({"kind": invalid_kind}, "Invalid domain")
    assert len(read_json(lead.state_path)["assignments"]) == 1
    lead.assign_region({"kind": "lengths", "lengths": [1]}, "An independent investigation")
    with pytest.raises(ToolError, match="already assigned"):
        lead.assign_region({"kind": "all"}, "One more")


def test_sandbox_and_verifier_use_the_same_assigned_domain(tmp_path):
    cfg = config(tmp_path, research_task={"role": "regional", "max_assignments": 0, "cycle": 1})
    cfg["verifier"]["scope"] = {"kind": "lengths", "lengths": [2]}
    session = Session.create(tmp_path / "worker", cfg, cfg["network"], tmp_path / "run")
    result = session.run_python("""
import json
x, answers, operations = network_outputs()
assert len(x) == 4096
assert np.all(np.sum(x != 64, axis=1) == 2)
g = graph()
entry = g.entry(predict=g.token(0), rule=g.token(0) == g.token(1))
save_entry('conditional', entry)
print(json.dumps(measure(entry)))
""")
    assert result["status"] == "ok", result
    assert result["saved_entries"] == ["conditional"], result["output"]
    import json

    measured = json.loads(result["output"])
    scored = session.submit_entry("conditional", 0.9)
    assert scored["status"] == "scored", scored
    for field in ("coverage", "fidelity", "cost", "b", "r_rule", "r"):
        assert measured[field] == pytest.approx(scored["parts"][field])
    assert scored["parts"]["coverage"] == pytest.approx(1 / 64)
    assert measured["reward"] == pytest.approx(scored["reward"])


def test_research_prompts_do_not_reveal_the_pilot_task(tmp_path):
    cfg = config(tmp_path, research_task={"role": "lead", "max_assignments": 2, "cycle": 0})
    from level1.networks import load_model

    model = load_model(cfg["model"])
    for role in ("lead", "regional"):
        text = prompt.build_multi(cfg, model, 0, role, "Investigate the assigned domain.", "", [])
        hygiene.check({"prompt": text, **loop.agent_surface(cfg, model)}, loop.forbidden_strings(cfg, model))
        assert "coverage 1" in text
        assert "reward = coverage" in text
    original = run_config.load("config/pilot/lin-c.yaml")
    assert original["orchestration"]["enabled"] is False
    assert original["verifier"]["scope"] is None


def test_local_sample_uses_the_configured_development_seed(tmp_path):
    cfg = config(tmp_path, research_task={"role": "regional", "max_assignments": 0, "cycle": 1})
    cfg["verifier"].update(scope={"kind": "all"}, check="sample", sample_size=128, sample_seed=17)
    session = Session.create(tmp_path / "worker", cfg, cfg["network"], tmp_path / "run")
    result = session.run_python("""
import json
g = graph()
entry = g.entry(predict=g.token(0))
save_entry('first', entry)
print(json.dumps(measure(entry)))
""")
    import json

    assert result["saved_entries"] == ["first"], result["output"]
    measured = json.loads(result["output"])
    scored = session.submit_entry("first", 0.5)
    assert measured["n_agree"] == scored["parts"]["n_agree"]
    assert measured["reward"] == pytest.approx(scored["reward"])
    assert scored["inputs"]["seed"] == 17
    assert scored["scope"]["id"] == "all"


def test_orchestration_requires_shared_quotas_and_valid_scopes(tmp_path):
    with pytest.raises(run_config.ConfigError, match="global integration"):
        run_config.load("config/experiments/lin-c-regional.yaml", {
            "session": {"submissions": 2}})
    with pytest.raises(run_config.ConfigError, match="positive share"):
        run_config.load("config/experiments/lin-c-regional.yaml", {
            "orchestration": {"planning_fraction": 0.5, "regional_fraction": 0.5}})
    with pytest.raises(run_config.ConfigError, match="unknown orchestration"):
        run_config.load("config/experiments/lin-c-regional.yaml", {
            "orchestration": {"per_worker_budget": 60}})
    with pytest.raises(run_config.ConfigError, match="too small after sharing"):
        run_config.load("config/experiments/lin-c-regional.yaml", {
            "session": {"python_calls": 3, "submissions": 3}})
    with pytest.raises(run_config.ConfigError, match="positive whole numbers"):
        run_config.load("config/experiments/lin-c-regional.yaml", {"rounds_max": 2.5})


def test_sampled_single_scope_has_fresh_locked_checks(tmp_path):
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    write_json(scripts / "01.json", {"calls": [
        {"tool": "run_python", "args": {"code": "save_entry('full', reference_entry())"}},
        {"tool": "submit_entry", "args": {"name": "full", "claimed_fidelity": 1.0,
                                            "expected_fidelity_other": 1.0}},
    ], "response": "## What the network does\nThe complete computation was checked."})
    cfg = run_config.load("config/experiments/lin-c-length2.yaml", {
        "runs_dir": str(tmp_path / "runs"), "backend": "fake", "sandbox": "process", "rounds_max": 1,
        "session": {"python_calls": 1, "submissions": 1},
        "verifier": {"check": "sample", "sample_size": 64, "sample_seed": 101, "reference_samples": 1},
    })
    path = loop.start(cfg, fake_scripts=scripts, out=lambda *_: None)
    development = read_json(path / "round_001/session/submissions/001/receipt.json")
    lock = read_json(path / "lock.json")
    other = read_json(path / "seed_check.json")
    assert development["inputs"]["seed"] == 101
    assert lock["measured_again"]["inputs"]["seed"] != 101
    assert other["inputs"]["seed"] != 101
    assert read_json(path / "run.json")["n_total"] == 4096
    assert "fixed development sample" in (path / "report.md").read_text()
    again = loop.finish(path, out=lambda *_: None)
    assert again == lock
    assert read_json(path / "seed_check.json") == other
