"""The single global comparison agent has a bounded, full-domain task."""

import copy
import json

import pytest

from level1 import hygiene, lang, loop, prompt, run_config
from level1.networks import load_model
from level1.session import Session, ToolError


def make_session(tmp_path, *, checkpoint=1, submissions=2):
    cfg = run_config.load("config/pilot/lin-c.yaml", {
        "runs_dir": str(tmp_path), "sandbox": "process", "backend": "fake",
        "model": "models/lin-e", "network": "lin-e-s0", "exact_on_any_weights": True,
        "session": {"python_calls": 4, "submissions": submissions},
        "verifier": {"scope": {"kind": "all"}, "check": "sample", "sample_size": 32, "sample_seed": 17},
    })
    cfg["research_task"] = {"role": "baseline", "cycle": 0, "max_assignments": 0}
    if checkpoint != "missing":
        cfg["research_task"]["checkpoint_python_calls"] = checkpoint
    model = load_model(cfg["model"])
    weights = model.load_weights(cfg["network"])
    reference = model.reference_entry(weights)
    g, prediction, _ = lang.Graph.load(reference, weights, model.n_ctx)
    partial = g.entry(predict=prediction, rule=g.token(0) < 32)
    bad = lang.Graph(weights, model.n_ctx)
    wrong = bad.entry(predict=bad.const(0))
    session = Session.create(tmp_path / "session", cfg, cfg["network"], tmp_path / "run",
                             {"reference": reference, "partial": partial, "wrong": wrong})
    return session, cfg, model


def test_baseline_prompt_is_global_generic_and_does_not_disclose_control_identity(tmp_path):
    session, cfg, model = make_session(tmp_path)
    text = prompt.build_multi(cfg, model, 0, "baseline", "Improve the complete program.", "", [])
    assert prompt.EXACT_NOTE in text
    assert "single investigator" in text
    assert "entire assigned input domain" in text
    assert "every evaluated input" in text
    assert "No further investigations can be scheduled" in text
    assert "Your role in this session: **comparison**" in text
    assert "scope=" not in text and "scope_entry_name" not in text
    assert "You can declare a narrower evaluation domain" not in text
    assert "`submit_entry(name, claimed_fidelity, expected_fidelity_other=None)`" in text
    assert "Use `network_outputs()` and `measure(entry)` on the complete assigned domain" in text
    assert "Those checks select one frozen comparison program" in text
    assert model.name not in text and cfg["network"] not in text
    reply = session.run_python("import json\nprint(json.dumps(info, sort_keys=True))")
    assert reply["status"] == "ok"
    assert "research_task" not in reply["output"]
    hygiene.check({"prompt": text, **loop.agent_surface(cfg, model),
                   "session tool descriptions": json.dumps(session.TOOLS),
                   "session info": json.dumps(session.info()), "tool reply": json.dumps(reply)},
                  loop.forbidden_strings(cfg, model))


@pytest.mark.parametrize("role", ["lead", "regional"])
def test_research_roles_keep_their_narrower_domain_tools(tmp_path, role):
    _, cfg, model = make_session(tmp_path)
    cfg["research_task"] = {"role": role, "max_assignments": 0, "cycle": 1}
    cfg["exact_on_any_weights"] = False
    text = prompt.build_multi(cfg, model, 0, role, "Investigate the assigned domain.", "", [])
    assert "You can declare a narrower evaluation domain with `submit_entry(..., scope=...)`" in text
    assert "optional `scope=...`" in text
    assert "expected_fidelity_other=None, scope=None, scope_entry_name=None" in text
    assert "pass a saved entry defining its region test as `scope_entry_name`" in text
    assert "are not returned to researchers and do not select the development winners" in text


def test_baseline_tools_reject_regional_assignments_and_claims_without_spending(tmp_path):
    session, _, _ = make_session(tmp_path)
    assert "assign_region" not in session.TOOLS
    assert "scope" not in session.TOOLS["submit_entry"]["schema"]["properties"]
    with pytest.raises(ToolError, match="no tool"):
        session.call("assign_region", {"scope": {"kind": "all"}, "task": "Another investigation."})
    with pytest.raises(ToolError, match="cannot assign"):
        session.assign_region({"kind": "all"}, "Another investigation.")
    with pytest.raises(ToolError, match="does not take"):
        session.call("submit_entry", {"name": "reference", "claimed_fidelity": 1.0,
                                      "scope": {"kind": "lengths", "lengths": [2]}})
    with pytest.raises(ToolError, match="only in research sessions"):
        session.submit_entry("reference", 1.0, scope={"kind": "lengths", "lengths": [2]})
    assert session.state["used"] == {"run_python": 0, "submit_entry": 0}
    assert session.receipts() == []
    result = session.run_python("measure(reference_entry(), scope={'kind':'lengths','lengths':[2]})")
    assert result["status"] == "error"
    assert session.state["used"]["submit_entry"] == 0


@pytest.mark.parametrize("name", ["partial", "wrong"])
def test_baseline_checkpoint_needs_full_coverage_and_exact_agreement_and_allows_bounded_repair(tmp_path, name):
    session, _, _ = make_session(tmp_path)
    assert session.run_python("print(sorted(entries))")["status"] == "ok"
    assert session.remaining()["checkpoint"]["status"] == "blocked"
    before = copy.deepcopy(session.state["used"])
    with pytest.raises(ToolError, match="checkpoint"):
        session.run_python("print('extra exploration')")
    assert session.state["used"] == before

    result = session.submit_entry(name, 0.0)
    assert result["status"] == "scored"
    if name == "partial":
        assert result["parts"]["fidelity"] == 1.0 and result["parts"]["coverage"] < 1.0
    else:
        assert result["parts"]["coverage"] == 1.0 and result["parts"]["fidelity"] < 1.0
    checkpoint = session.remaining()["checkpoint"]
    assert checkpoint["status"] == "pending" and checkpoint["repair_calls"] == 1
    assert session.remaining()["submit_entry"] == 1
    assert session.run_python("print('one construction repair')")["status"] == "ok"
    with pytest.raises(ToolError, match="checkpoint"):
        session.run_python("print('a second repair')")

    assert session.submit_entry("reference", 1.0)["status"] == "scored"
    assert session.remaining()["submit_entry"] == 0
    assert session.remaining()["checkpoint"]["status"] == "scored"
    assert session.remaining()["checkpoint"]["submission"] == 2
    assert session.run_python("print('further saved refinements')")["status"] == "ok"


def test_last_failed_baseline_submission_does_not_grant_unbounded_research(tmp_path):
    session, _, _ = make_session(tmp_path, submissions=1)
    result = session.submit_entry("partial", 1.0)
    assert result["status"] == "scored"
    assert session.remaining()["checkpoint"]["status"] == "blocked"
    assert session.remaining()["submit_entry"] == 0
    with pytest.raises(ToolError, match="checkpoint"):
        session.run_python("print('cannot finish a checkpoint now')")
    assert session.state["used"]["run_python"] == 0


def test_baseline_checkpoint_and_quotas_survive_reopen(tmp_path):
    session, cfg, _ = make_session(tmp_path)
    session.run_python("print('initial call')")
    changed = copy.deepcopy(cfg)
    changed["research_task"]["checkpoint_python_calls"] = 100
    changed["session"]["submissions"] = 100
    reopened = Session.create(session.path, changed, cfg["network"], tmp_path / "run")
    assert reopened.remaining()["checkpoint"]["initial_python_calls"] == 1
    assert reopened.remaining()["submit_entry"] == 2
    with pytest.raises(ToolError, match="checkpoint"):
        reopened.run_python("print('new attempt does not reset exploration')")
    reopened.submit_entry("reference", 1.0)
    again = Session(session.path)
    assert again.remaining()["checkpoint"]["submission"] == 1


@pytest.mark.parametrize("checkpoint", [None, "missing"])
def test_baseline_session_without_policy_does_not_acquire_checkpoint(tmp_path, checkpoint):
    session, _, _ = make_session(tmp_path, checkpoint=checkpoint)
    assert "checkpoint" not in session.remaining()


def test_generated_comparison_summary_preserves_neutral_provenance_and_masks_only_discoveries(tmp_path):
    _, cfg, model = make_session(tmp_path)
    program = {"format": lang.FORMAT, "nodes": [{"op": "weight", "name": "random weights"}], "predict": 0, "rule": None}
    raw = {"comparison": {
        "provided": True, "source": "single_agent_search", "program": program,
        "search_sessions": 1, "same_search_conditions": False, "validation_established": True,
        "validation_note": "Full coverage and exact agreement were checked on other parameter settings.",
        "baseline_search": True, "candidate_parameter_control_provided": False,
        "random_weight_control_provided": False, "provenance": "PRIVATE lin-e-s0 random weights",
        "development": {}, "control_model": "models/lin-e",
    }}
    view = prompt.summary_evidence(raw)
    assert prompt.summary_evidence(view) == view
    comparison = view["comparison"]
    for key in ("source", "search_sessions", "same_search_conditions", "validation_established", "validation_note",
                "candidate_parameter_control_provided"):
        assert comparison[key] == raw["comparison"][key]
    assert "random_weight_control_provided" not in comparison
    assert comparison["independent_comparison_search"] is True
    assert "baseline_search" not in comparison
    assert comparison["program"] == program
    assert comparison["provenance"] == (
        "A frozen full-domain comparison program produced by one independent session.")
    assert "PRIVATE" not in str(view) and "lin-e-s0" not in str(view)
    assert "control_model" not in comparison
    masked = prompt.summary_evidence(raw, include_discoveries=False)
    assert masked["comparison"]["program"] != program
    assert "random weights" not in str(masked)
    hygiene.check({"prompt": prompt.build_summary(cfg, model, masked)}, loop.forbidden_strings(cfg, model))

    provided = copy.deepcopy(raw)
    provided["comparison"]["source"] = "provided"
    masked = prompt.summary_evidence(provided, include_discoveries=False)
    assert masked["comparison"]["program"] == program
    with pytest.raises(hygiene.HygieneError):
        hygiene.check({"prompt": prompt.build_summary(cfg, model, masked)}, loop.forbidden_strings(cfg, model))
