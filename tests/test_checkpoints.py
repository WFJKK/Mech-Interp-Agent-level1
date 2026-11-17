"""A measured integration checkpoint is required before extended exploration."""

import copy

import pytest

from level1 import lang, regions, run_config
from level1.networks import load_model
from level1.session import Session, ToolError
from level1.util import read_json, read_jsonl, write_json


def make_session(tmp_path, *, submissions=3, checkpoint=2, carried=True):
    cfg = run_config.load("config/pilot/lin-a-1to4.yaml", {
        "runs_dir": str(tmp_path), "sandbox": "process", "backend": "fake",
        "session": {"python_calls": 8, "submissions": submissions},
        "verifier": {"scope": {"kind": "all"}, "check": "sample", "sample_size": 32,
                     "sample_seed": 17},
    })
    cfg["research_task"] = {"role": "lead", "cycle": 1, "max_assignments": 0}
    if checkpoint != "missing":
        cfg["research_task"]["checkpoint_python_calls"] = checkpoint
    model = load_model(cfg["model"])
    weights = model.load_weights(cfg["network"])
    g = lang.Graph(weights, model.n_ctx)
    # The interpreter refuses this out-of-bounds table on the sampled inputs.
    invalid = g.entry(predict=g.const([0])[g.token(0)])
    entries = {"reference": model.reference_entry(weights), "invalid": invalid} if carried else {}
    session = Session.create(tmp_path / "session", cfg, cfg["network"], tmp_path / "run", entries)
    return session, cfg


def use_initial_calls(session):
    for _ in range(2):
        result = session.run_python("print(sorted(entries))")
        assert result["status"] == "ok"


def test_checkpoint_blocks_more_python_without_spending_and_accepts_a_measured_program(tmp_path):
    session, _ = make_session(tmp_path)
    start = session.remaining()["checkpoint"]
    assert start["status"] == "pending"
    assert start["initial_python_calls"] == 2
    assert start["scope_id"] == "all"
    assert start["python_calls_until_checkpoint"] == 2
    use_initial_calls(session)
    assert session.remaining()["checkpoint"]["status"] == "blocked"
    before = copy.deepcopy(session.state["used"])
    with pytest.raises(ToolError, match="checkpoint"):
        session.run_python("print('extended optimization')")
    assert session.state["used"] == before
    assert not (session.path / "python" / "003.py").exists()
    assert "checkpoint" in read_jsonl(session.path / "tools.jsonl")[-1]["outcome"]["error"]

    receipt = session.submit_entry("reference", 1.0)
    assert receipt["status"] == "scored"
    # A reference is an evaluated fallback, without a reward or mechanism bonus.
    assert receipt["parts"]["fidelity"] == 1.0
    assert receipt["parts"]["cost"] == pytest.approx(1.0)
    assert receipt["reward"] <= 0
    saved = session.receipts()[0]
    assert saved["claim_scope"] == {"kind": "all"}
    assert saved["assigned_scope_id"] == "all"
    checkpoint = session.remaining()["checkpoint"]
    assert checkpoint["status"] == "scored" and checkpoint["submission"] == 1
    assert session.run_python("print('continue')")["status"] == "ok"


def test_child_claim_does_not_unlock_checkpoint_and_cannot_spend_its_last_slot(tmp_path):
    session, _ = make_session(tmp_path, submissions=2)
    child = {"kind": "lengths", "lengths": [1]}
    receipt = session.submit_entry("reference", 1.0, scope=child)
    assert receipt["status"] == "scored"
    assert receipt["scope"]["id"] == regions.scope_id(child)
    assert session.receipts()[0]["claim_scope"] == child
    assert session.remaining()["checkpoint"]["status"] == "pending"
    assert session.remaining()["submit_entry"] == 1
    with pytest.raises(ToolError, match="checkpoint"):
        session.submit_entry("reference", 1.0, scope=child)
    assert session.remaining()["submit_entry"] == 1
    assert len(session.receipts()) == 1
    use_initial_calls(session)
    with pytest.raises(ToolError, match="checkpoint"):
        session.run_python("print('still unmeasured globally')")
    assert session.submit_entry("reference", 1.0)["status"] == "scored"
    assert session.remaining()["submit_entry"] == 0
    assert session.run_python("print('save further drafts')")["status"] == "ok"


def test_refused_checkpoints_allow_only_bounded_repair_before_resubmission(tmp_path):
    session, _ = make_session(tmp_path, submissions=3)
    use_initial_calls(session)
    for attempt in (1, 2):
        result = session.submit_entry("invalid", 1.0)
        assert result["status"] == "refused"
        assert session.remaining()["submit_entry"] == 3 - attempt
        checkpoint = session.remaining()["checkpoint"]
        assert checkpoint["repair_calls"] == attempt
        assert checkpoint["python_calls_until_checkpoint"] == 1
        assert session.run_python("print('inspect the refusal')")["status"] == "ok"
        with pytest.raises(ToolError, match="checkpoint"):
            session.run_python("print('another optimization')")
    assert session.submit_entry("invalid", 1.0)["status"] == "refused"
    assert session.remaining()["submit_entry"] == 0
    assert session.remaining()["checkpoint"]["status"] == "blocked"
    with pytest.raises(ToolError, match="checkpoint"):
        session.run_python("print('no submission slots remain')")
    assert session.state["used"] == {"run_python": 4, "submit_entry": 3}


def test_a_refused_child_claim_does_not_grant_global_repair_calls(tmp_path):
    session, _ = make_session(tmp_path)
    use_initial_calls(session)
    refused = session.submit_entry("invalid", 1.0, scope={"kind": "lengths", "lengths": [1]})
    assert refused["status"] == "refused"
    assert session.remaining()["checkpoint"]["repair_calls"] == 0
    with pytest.raises(ToolError, match="checkpoint"):
        session.run_python("print('child refusal is not a global checkpoint')")
    assert session.state["used"]["run_python"] == 2


def test_interrupted_submission_has_no_checkpoint_or_repair_credit(tmp_path, monkeypatch):
    from level1.verifier import Verifier

    session, _ = make_session(tmp_path)
    use_initial_calls(session)

    def interrupted(*args, **kwargs):
        raise SystemExit("simulated interruption before a receipt")

    with monkeypatch.context() as patch:
        patch.setattr(Verifier, "score", interrupted)
        with pytest.raises(SystemExit, match="simulated interruption"):
            session.submit_entry("reference", 1.0)
    assert session.remaining()["submit_entry"] == 2
    assert (session.path / "submissions" / "001" / "entry.json").is_file()
    assert session.receipts() == []
    reopened = Session(session.path)
    assert reopened.remaining()["checkpoint"]["repair_calls"] == 0
    with pytest.raises(ToolError, match="checkpoint"):
        reopened.run_python("print('incomplete evaluation')")
    assert reopened.submit_entry("reference", 1.0)["status"] == "scored"
    assert reopened.remaining()["checkpoint"]["submission"] == 2


def test_reopen_and_new_attempt_do_not_refresh_checkpoint_or_tool_quotas(tmp_path):
    session, cfg = make_session(tmp_path)
    use_initial_calls(session)
    changed = copy.deepcopy(cfg)
    changed["research_task"]["checkpoint_python_calls"] = 100
    changed["session"]["submissions"] = 100
    reopened = Session.create(session.path, changed, cfg["network"], tmp_path / "run")
    assert reopened.remaining()["checkpoint"]["initial_python_calls"] == 2
    assert reopened.remaining()["submit_entry"] == 3
    reopened.begin_attempt(60)
    assert 0 < reopened.remaining()["wall_seconds_remaining"] <= 60
    with pytest.raises(ToolError, match="checkpoint"):
        reopened.run_python("print('a retry is not a new exploration budget')")
    assert reopened.state["used"]["run_python"] == 2
    assert reopened.submit_entry("reference", 1.0)["status"] == "scored"
    again = Session(session.path)
    assert again.remaining()["checkpoint"]["status"] == "scored"
    assert again.run_python("print('checkpoint survived')")["status"] == "ok"


@pytest.mark.parametrize("checkpoint", [None, "missing"])
def test_old_session_without_checkpoint_policy_keeps_legacy_behavior(tmp_path, checkpoint):
    session, _ = make_session(tmp_path, checkpoint=checkpoint)
    assert "checkpoint" not in session.remaining()
    for _ in range(3):
        assert session.run_python("print('legacy exploration')")["status"] == "ok"
    assert session.receipts() == []


def test_initial_window_can_build_a_checkpoint_without_inherited_drafts(tmp_path):
    session, _ = make_session(tmp_path, carried=False)
    result = session.run_python("g = graph()\nsave_entry('first', g.entry(predict=g.const(0)))")
    assert result["saved_entries"] == ["first"]
    assert session.submit_entry("first", 0.0)["status"] == "scored"
    assert session.remaining()["checkpoint"]["status"] == "scored"
    assert session.run_python("print(sorted(entries))")["status"] == "ok"


def test_no_saved_draft_gets_one_construction_repair_before_checkpoint(tmp_path):
    session, _ = make_session(tmp_path, carried=False)
    for _ in range(2):
        failed = session.run_python("raise RuntimeError('candidate construction failed')")
        assert failed["status"] == "error" and failed["saved_entries"] == []
    checkpoint = session.remaining()["checkpoint"]
    assert checkpoint["bootstrap_repair_calls"] == 1
    assert checkpoint["python_calls_until_checkpoint"] == 1
    repaired = session.run_python("g = graph()\nsave_entry('repaired', g.entry(predict=g.const(0)))")
    assert repaired["saved_entries"] == ["repaired"]
    assert session.remaining()["checkpoint"]["bootstrap_repair_calls"] == 1
    assert session.remaining()["checkpoint"]["status"] == "blocked"
    with pytest.raises(ToolError, match="checkpoint"):
        session.run_python("print('a saved repair now requires evaluation')")
    assert session.state["used"]["run_python"] == 3
    assert session.submit_entry("repaired", 0.0)["status"] == "scored"
    assert session.remaining()["checkpoint"]["status"] == "scored"


def test_failed_construction_repair_does_not_refresh_on_reopen_or_retry(tmp_path):
    session, cfg = make_session(tmp_path, carried=False)
    for _ in range(2):
        assert session.run_python("raise RuntimeError('no usable draft')")["status"] == "error"
    reopened = Session.create(session.path, cfg, cfg["network"], tmp_path / "run")
    assert reopened.run_python("raise RuntimeError('repair also failed')")["status"] == "error"
    retried = Session(session.path)
    retried.begin_attempt(60)
    assert retried.entries() == {} and retried.receipts() == []
    checkpoint = retried.remaining()["checkpoint"]
    assert checkpoint["bootstrap_repair_calls"] == 1
    assert checkpoint["python_calls_until_checkpoint"] == 0
    assert checkpoint["status"] == "blocked"
    with pytest.raises(ToolError, match="checkpoint"):
        retried.run_python("print('there is no second construction repair')")
    assert retried.state["used"] == {"run_python": 3, "submit_entry": 0}


def test_a_refusal_after_construction_repair_still_grants_its_own_repair_call(tmp_path):
    session, _ = make_session(tmp_path, carried=False)
    for _ in range(2):
        assert session.run_python("raise RuntimeError('no draft yet')")["status"] == "error"
    saved = session.run_python("g = graph()\nsave_entry('candidate', g.entry(predict=g.const([0])[g.token(0)]))")
    assert saved["saved_entries"] == ["candidate"]
    assert session.submit_entry("candidate", 1.0)["status"] == "refused"
    reopened = Session(session.path)
    checkpoint = reopened.remaining()["checkpoint"]
    assert checkpoint["repair_calls"] == 1
    assert checkpoint["python_calls_until_checkpoint"] == 1
    assert reopened.run_python("g = graph()\nsave_entry('candidate', g.entry(predict=g.const(0)))")["status"] == "ok"
    with pytest.raises(ToolError, match="checkpoint"):
        reopened.run_python("print('the refusal granted only one repair')")
    assert reopened.state["used"]["run_python"] == 4
    assert reopened.submit_entry("candidate", 0.0)["status"] == "scored"


def test_legacy_same_scope_receipt_remains_a_checkpoint_after_upgrade(tmp_path):
    session, _ = make_session(tmp_path)
    assert session.submit_entry("reference", 1.0)["status"] == "scored"
    path = session.path / "submissions" / "001" / "receipt.json"
    receipt = read_json(path)
    receipt.pop("claim_scope")
    receipt.pop("assigned_scope_id")
    write_json(path, receipt)
    reopened = Session(session.path)
    assert reopened.remaining()["checkpoint"]["status"] == "scored"
