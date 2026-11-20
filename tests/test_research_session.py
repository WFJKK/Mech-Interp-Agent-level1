"""Role tools, sandbox evaluation and prompt boundaries for scoped research."""

import copy

import pytest

from level1 import hygiene, loop, prompt, run_config
from level1.session import Session, ToolError
from level1.util import read_json, write_json


def config(tmp_path, **changes):
    cfg = run_config.load("config/experiments/lin-c-regional.yaml", {
        "runs_dir": str(tmp_path), "sandbox": "process", "backend": "fake",
        "session": {"python_calls": 10, "submissions": 6},
    })
    cfg.update(changes)
    return cfg


def test_lead_assigns_a_frozen_scope_but_workers_cannot(tmp_path):
    cfg = config(tmp_path, research_task={"role": "lead", "max_assignments": 2, "cycle": 0})
    lead = Session.create(tmp_path / "lead", cfg, cfg["network"], tmp_path / "run")
    assert "assign_task" in lead.TOOLS
    assigned = lead.call("assign_task", {"scope": {"kind": "lengths", "lengths": [2]},
                                           "task": "Investigate a simple rule and its exceptions."})
    assert assigned["status"] == "assigned"
    reopened = Session(lead.path)
    assert reopened.state["assignments"][0]["scope"] == {"kind": "lengths", "lengths": [2]}
    assert reopened.state["reward"] == cfg["reward"]
    worker_cfg = copy.deepcopy(cfg)
    worker_cfg["research_task"] = {"role": "regional", "max_assignments": 0, "cycle": 1}
    worker = Session.create(tmp_path / "worker", worker_cfg, cfg["network"], tmp_path / "run")
    assert "assign_task" not in worker.TOOLS
    with pytest.raises(ToolError, match="no tool"):
        worker.call("assign_task", {"scope": {"kind": "all"}, "task": "Change the task"})
    with pytest.raises(ToolError):
        lead.assign_task("Discount setup", {"kind": "all", "workload_size": 10**12})
    for invalid_kind in ([], {}):
        with pytest.raises(ToolError, match="scope.kind"):
            lead.assign_task("Invalid domain", {"kind": invalid_kind})
    assert len(read_json(lead.state_path)["assignments"]) == 1
    lead.assign_task("An independent investigation", {"kind": "lengths", "lengths": [1]})
    with pytest.raises(ToolError, match="already assigned"):
        lead.assign_task("One more", {"kind": "all"})


def test_expired_attempt_cannot_assign_a_new_investigation(tmp_path, monkeypatch):
    cfg = config(tmp_path, research_task={"role": "lead", "max_assignments": 2, "cycle": 0})
    lead = Session.create(tmp_path / "lead", cfg, cfg["network"], tmp_path / "run")
    monkeypatch.setattr("level1.session.time.time", lambda: 100.0)
    lead.begin_attempt(10)
    monkeypatch.setattr("level1.session.time.time", lambda: 111.0)
    with pytest.raises(ToolError, match="wall-clock deadline"):
        lead.assign_task("Too late to start.", {"kind": "all"})
    assert lead.state["assignments"] == []


def test_summary_session_has_no_tools_even_when_called_directly(tmp_path, monkeypatch):
    from level1 import lang, mcp_server

    cfg = config(tmp_path, research_task={"role": "summary", "max_assignments": 9})
    g = lang.Graph({}, 4)
    entry = g.entry(predict=g.const(0))
    session = Session.create(tmp_path / "summary", cfg, cfg["network"], tmp_path / "run", {"saved": entry})
    assert session.TOOLS == {}
    assert session.state["budgets"] == {"run_python": 0, "submit_entry": 0}
    assert session.state["research_task"]["max_assignments"] == 0
    assert mcp_server.tool_list(session) == []

    def cannot_spend(*_):
        raise AssertionError("a final summary tried to spend a research tool allowance")

    monkeypatch.setattr(session, "_spend", cannot_spend)
    calls = {
        "run_python": {"code": "print('new research')"},
        "submit_entry": {"name": "saved", "claimed_fidelity": 1.0},
        "assign_task": {"scope": {"kind": "all"}, "task": "Change the investigation"},
    }
    for name, arguments in calls.items():
        with pytest.raises(ToolError, match="no tool"):
            session.call(name, arguments)
        with pytest.raises(ToolError, match="summary.*no research tools"):
            getattr(session, name)(**arguments)
        reply = mcp_server.handle(session, {"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                                            "params": {"name": name, "arguments": arguments}})
        assert reply["result"]["isError"]
    assert session.receipts() == []
    assert session.state["used"] == {"run_python": 0, "submit_entry": 0}
    assert session.state["assignments"] == []
    assert not (session.path / "python").exists()
    assert not (session.path / "submissions").exists()
    assert session.entries() == {"saved": entry}
    reopened = Session.create(session.path, cfg, cfg["network"], tmp_path / "run")
    assert reopened.TOOLS == {}
    assert reopened.remaining() == {"run_python": 0, "submit_entry": 0}


def test_a_summary_cannot_reopen_a_research_session_with_tools(tmp_path):
    cfg = config(tmp_path, research_task={"role": "regional", "max_assignments": 0})
    session = Session.create(tmp_path / "existing", cfg, cfg["network"], tmp_path / "run")
    summary_cfg = copy.deepcopy(cfg)
    summary_cfg["research_task"]["role"] = "summary"
    before = session.state_path.read_bytes()
    with pytest.raises(ToolError, match="summary role"):
        Session.create(session.path, summary_cfg, cfg["network"], tmp_path / "run")
    assert session.state_path.read_bytes() == before


@pytest.mark.parametrize("model_path", ["models/lin-c", "models/att-c", "models/rnn-a", "models/lin-e"])
def test_summary_prompt_is_generic_and_uses_only_supplied_evidence(tmp_path, monkeypatch, model_path):
    import json

    from level1.networks import load_model

    cfg = config(tmp_path, research_task={"role": "summary", "max_assignments": 0})
    model = load_model(model_path)
    forbidden = loop.forbidden_strings(cfg, model)

    def not_a_summary_input(*_):
        raise AssertionError("the summary attempted to obtain new or withheld evidence")

    monkeypatch.setattr(model, "load_weights", not_a_summary_input)
    monkeypatch.setattr(model, "task", not_a_summary_input)
    monkeypatch.setattr(model, "withheld", not_a_summary_input)
    evidence = {"frozen_selection": {"global": "g01s1", "regional": {}},
                "candidates": [{"id": "g01s1", "development": {"reward": 0.7},
                                "final_audits": {"fresh": {"reward": 0.6}}}],
                "prior_notes": {"items": [{"notes": "An unresolved observation with literal {{braces}}."}]},
                "comparison": {"provided": False}}
    before = copy.deepcopy(evidence)
    text = prompt.build_summary(cfg, model, evidence)
    assert json.dumps(prompt.summary_evidence(evidence), indent=2, sort_keys=True, allow_nan=False) in text
    assert evidence == before
    assert "literal {{braces}}" in text
    assert prompt.ENTRY_DOC.read_text().strip() in text
    assert "best found" in text
    assert "formal proof" in text
    assert "No tools are available" in text
    assert "independent_comparison_search" in text
    assert "does not mean that no comparison search occurred" in text
    assert model.name not in text
    surface = loop.agent_surface(cfg, model)
    assert "summary prompt template" in surface
    hygiene.check({"prompt": prompt.build_summary(cfg, model, {}), **surface}, forbidden)


@pytest.mark.parametrize("model_path", ["models/lin-c", "models/rnn-d"])
def test_summary_view_hides_operator_provenance_but_preserves_discoveries(tmp_path, model_path):
    from level1 import lang
    from level1.networks import load_model

    cfg = config(tmp_path, research_task={"role": "summary", "max_assignments": 0})
    model = load_model(model_path)
    g = lang.Graph({}, model.n_ctx, real=model.real)
    program = g.entry(predict=g.const(0))
    operator_label = "OPERATOR minimum of random weights"
    discovery = "DISCOVERY minimum of random weights"
    receipt = {"status": "scored", "network": operator_label, "reward": 0.7,
               "parts": {"coverage": 1.0, "fidelity": 1.0, "cost": 0.3},
               "verifier_error": "PRIVATE_ERROR minimum of random weights"}
    raw = {"version": 1, "run_id": operator_label, "network": operator_label,
           "second_network": operator_label, "research_stop_reason": operator_label,
           "prices": lang.PRICES, "frozen_selection": {"global": "g01s1", "regional": {}},
           "candidates": [{"id": "g01s1", "role": "lead", "scope_id": "all", "scope": {"kind": "all"},
                           "program": program, "development": receipt,
                           "final_audits": {"fresh": receipt, "held_out": receipt,
                                            "fresh_baseline": receipt, "held_out_baseline": receipt}}],
           "comparison": {"provided": True, "program": program, "provenance": operator_label,
                          "development": {"all": receipt}, "random_weight_control_provided": False},
           "prior_notes": {"items": [{"role": "regional", "notes": discovery}]}}
    before = copy.deepcopy(raw)
    view = prompt.summary_evidence(raw)
    assert prompt.summary_evidence(view) == view
    text = prompt.build_summary(cfg, model, raw)
    assert operator_label not in text
    assert "PRIVATE_ERROR" not in text
    assert "_baseline" not in text
    assert discovery in text
    assert "fresh_comparison" in text and "held_out_comparison" in text
    assert view["candidates"][0]["program"] == program
    assert view["comparison"]["program"] == program
    assert "random_weight_control_provided" not in view["comparison"]
    assert raw == before
    masked = prompt.summary_evidence(raw, include_discoveries=False)
    assert discovery not in str(masked)
    assert masked["comparison"]["program"] == program
    hygiene.check({"prompt": prompt.build_summary(cfg, model, masked)}, loop.forbidden_strings(cfg, model))

    # Retained controller fields and the supplied comparison program are checked,
    # rather than exempted along with researcher-authored findings.
    unsafe = copy.deepcopy(raw)
    unsafe["prices"] = operator_label
    with pytest.raises(hygiene.HygieneError):
        hygiene.check({"prompt": prompt.build_summary(cfg, model, prompt.summary_evidence(unsafe, include_discoveries=False))},
                      loop.forbidden_strings(cfg, model))
    unsafe = copy.deepcopy(raw)
    unsafe["comparison"]["program"]["nodes"].append({"op": "weight", "name": operator_label})
    with pytest.raises(hygiene.HygieneError):
        hygiene.check({"prompt": prompt.build_summary(cfg, model, prompt.summary_evidence(unsafe, include_discoveries=False))},
                      loop.forbidden_strings(cfg, model))


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
