"""Shared API caps for concurrent research roles, with no real CLI or API calls."""

from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from level1 import budget, run_config
from level1.backends import StopRun
from level1.backends.claude_code import ClaudeCodeBackend
from level1.session import Session
from level1.util import read_jsonl


def grouped(**over):
    return {"run_usd": 5, "round_usd": 3, "cycle_id": "cycle-001", "cycle_usd": 2, **over}


def _process_claim(args):
    run, k, cfg = args
    return budget.claim(Path(run), k, 1, cfg)


def test_claims_reserve_the_run_session_and_cycle_and_release_only_reported_savings(tmp_path):
    cfg = grouped(run_usd=3, round_usd=1.25)
    assert budget.claim(tmp_path, 1, 1, cfg) == 1.25
    st = budget.state(tmp_path, cfg)
    assert st["spent"] == 1.25 and st["rounds"] == {1: 1.25} and st["cycles"] == {"cycle-001": 1.25}
    assert st["unreported"] == st["stopped"] == 1
    # A stopped role cannot receive its reserved allowance again, even before it wrote a stream file.
    assert budget.claim(tmp_path, 1, 1, cfg) == 0
    assert budget.claim(tmp_path, 1, 2, cfg) == 0
    assert budget.claim(tmp_path, 2, 1, cfg) == 0.75
    assert budget.claim(tmp_path, 3, 1, cfg) == 0
    assert budget.left(budget.state(tmp_path, cfg)) == 1  # later cycles may still run

    budget.end(tmp_path, 1, 1, 0.4, "success", atomic=True)
    assert budget.claim(tmp_path, 1, 2, cfg) == 0.85
    budget.end(tmp_path, 2, 1, None, "timed out", atomic=True)
    st = budget.state(tmp_path, cfg)
    assert st["spent"] == 2 and st["rounds"] == {1: 1.25, 2: 0.75}
    assert st["cycles"] == {"cycle-001": 2}
    assert "cycle cycle-001's API budget is spent" in budget.spent_why(st, 3)
    assert budget.claim(tmp_path, 3, 1, {**cfg, "cycle_id": "cycle-002"}) == 1
    assert len([e for e in read_jsonl(tmp_path / budget.LOG) if e["event"] == "start"]) == 4


def test_parallel_process_claims_cannot_each_reserve_the_same_money(tmp_path):
    cfg = grouped(run_usd=2, round_usd=0.6, cycle_usd=1.31)
    with ProcessPoolExecutor(max_workers=6, mp_context=multiprocessing.get_context("spawn")) as pool:
        caps = list(pool.map(_process_claim, [(str(tmp_path), k, cfg) for k in range(1, 13)]))
    assert sorted(caps) == [0.0] * 9 + [0.11, 0.6, 0.6]
    st = budget.state(tmp_path, cfg)
    assert st["spent"] == pytest.approx(1.31)
    assert sum(st["rounds"].values()) == pytest.approx(1.31)
    assert st["cycles"]["cycle-001"] == pytest.approx(1.31)
    # A fresh cycle has only the run's remaining 69 cents, shared across two new roles.
    next_cfg = {**cfg, "cycle_id": "cycle-002"}
    assert budget.claim(tmp_path, 20, 1, next_cfg) == 0.6
    assert budget.claim(tmp_path, 21, 1, next_cfg) == 0.09
    assert budget.claim(tmp_path, 22, 1, next_cfg) == 0


def test_a_cycle_cap_is_persistent_and_legacy_interrupted_rounds_are_unchanged(tmp_path):
    cfg = grouped(cycle_usd=1)
    assert budget.claim(tmp_path, 1, 1, cfg) == 1
    assert budget.claim(tmp_path, 2, 1, {**cfg, "cycle_usd": 4}) == 0
    assert budget.state(tmp_path, {**cfg, "cycle_usd": 4})["cycle_usd"] == 1
    budget.set_run_budget(tmp_path, 10, atomic=True)
    assert budget.state(tmp_path, cfg)["run_usd"] == 10
    assert budget.claim(tmp_path, 2, 1, cfg) == 0

    legacy = tmp_path / "legacy"
    old_cfg = {"run_usd": 5, "round_usd": 2}
    budget.start(legacy, 1, 1, 2)
    st = budget.state(legacy, old_cfg)
    assert st["spent"] == 2 and st["rounds"] == {} and st["cycles"] == {}
    assert budget.cap(st, 1) == 2
    assert not (legacy / budget.LOCK).exists()


@pytest.mark.parametrize("cost", [None, True, -0.2, float("nan"), float("inf")])
def test_uncertain_grouped_cost_keeps_the_reservation(tmp_path, cost):
    cfg = grouped(cycle_usd=0.6)
    assert budget.claim(tmp_path, 1, 1, cfg) == 0.6
    budget.end(tmp_path, 1, 1, cost, "ended", atomic=True)
    st = budget.state(tmp_path, cfg)
    assert st["spent"] == 0.6 and st["rounds"] == {1: 0.6} and st["cycles"] == {"cycle-001": 0.6}
    assert st["unreported"] == 1 and st["stopped"] == 0
    assert budget.claim(tmp_path, 2, 1, cfg) == 0


def test_reported_overshoot_is_counted_in_all_scopes(tmp_path):
    cfg = grouped(round_usd=0.5, cycle_usd=0.7)
    assert budget.claim(tmp_path, 1, 1, cfg) == 0.5
    budget.end(tmp_path, 1, 1, 0.8, "error_max_budget_usd", atomic=True)
    st = budget.state(tmp_path, cfg)
    assert st["spent"] == st["rounds"][1] == st["cycles"]["cycle-001"] == 0.8
    assert budget.claim(tmp_path, 2, 1, cfg) == 0


def test_a_partial_log_line_cannot_hide_the_next_reservation(tmp_path):
    cfg = grouped(round_usd=0.5)
    assert budget.claim(tmp_path, 1, 1, cfg) == 0.5
    with (tmp_path / budget.LOG).open("a") as handle:
        handle.write('{"event": "end", "round": 1, "try":')
    assert budget.claim(tmp_path, 2, 1, cfg) == 0.5
    st = budget.state(tmp_path, cfg)
    assert st["spent"] == 1 and st["rounds"] == {1: 0.5, 2: 0.5}
    assert st["cycles"] == {"cycle-001": 1} and st["stopped"] == 2


@pytest.mark.parametrize("over", [{"cycle_id": None}, {"cycle_id": " "}, {"cycle_usd": 0},
                                  {"cycle_usd": True}, {"cycle_usd": float("inf")}])
def test_grouped_config_is_validated(over):
    assert budget.check(grouped(**over))


def test_atomic_claim_fails_clearly_without_a_group_or_os_lock(tmp_path, monkeypatch):
    assert budget.check({"run_usd": 3, "round_usd": 1}) == []
    assert budget.check({"run_usd": 3, "round_usd": 1, "cycle_usd": 1})
    with pytest.raises(ValueError, match="requires cycle_id"):
        budget.claim(tmp_path, 1, 1, {"run_usd": 3, "round_usd": 1})
    monkeypatch.setattr(budget, "fcntl", None)
    with pytest.raises(RuntimeError, match="OS file locking"):
        budget.claim(tmp_path, 1, 1, grouped())
    budget.start(tmp_path, 1, 1, 1)  # the legacy path does not require fcntl


def _fake_role(tmp_path, index, outcomes, *, cfg_over=None):
    cfg = run_config.load("config/pilot/lin-a.yaml", {"runs_dir": str(tmp_path), "sandbox": "process",
                          "auth": "api_key", "budget": grouped(), **(cfg_over or {})})
    run = tmp_path / "run"
    where = run / f"role_{index:03d}"
    session = Session.create(where / "session", cfg, "lin-a-s0", run)
    calls = []

    def runner(cmd, prompt_text, env, cwd, timeout_s):
        calls.append(cmd)
        outcome = outcomes[len(calls) - 1]
        if isinstance(outcome, Exception):
            raise outcome
        data = {"type": "result", "subtype": "success", "is_error": False, "result": "done",
                "total_cost_usd": outcome, "modelUsage": {cfg["agent_model"]: {"outputTokens": 1}}}
        init = {"type": "system", "subtype": "init", "apiKeySource": "apiKeyHelper"}
        return SimpleNamespace(stdout=json.dumps(init) + "\n" + json.dumps(data), stderr="", returncode=0)

    backend = ClaudeCodeBackend(cfg, runner=runner, log=lambda *_: None, sleep=lambda _: None,
                                environ={"PATH": "/usr/bin", "CPL_ANTHROPIC_KEY": "test-key-no-api-access"})
    return backend, session, where, calls


def test_backend_roles_share_a_cycle_and_preserve_its_cost_on_timeout(tmp_path):
    first, session, where, calls = _fake_role(tmp_path, 0, [0.5])
    response = first.respond("work", 0, where, session)
    assert calls[0][calls[0].index("--max-budget-usd") + 1] == "2.00"
    assert response.meta["budget"]["cycle_counted_usd"] == 0.5
    second, session, where, calls = _fake_role(tmp_path, 1, [subprocess.TimeoutExpired("fake", 1)])
    response = second.respond("work", 1, where, session)
    assert calls[0][calls[0].index("--max-budget-usd") + 1] == "1.50"
    assert response.meta["budget"]["cycle_counted_usd"] == 2
    assert response.meta["budget"]["cycle_id"] == "cycle-001"
    assert response.meta["budget"]["round_counted_usd"] == 1.5
    third, session, where, calls = _fake_role(tmp_path, 2, [])
    response = third.respond("work", 2, where, session)
    assert calls == [] and response.meta["stop_reason"] == "budget"
    assert "cycle cycle-001's API budget is spent" in response.text
    assert third.exhausted(tmp_path / "run") is None
    starts = [e for e in read_jsonl(tmp_path / "run" / budget.LOG) if e["event"] == "start"]
    assert [(e["round"], e["try"], e["cycle_id"]) for e in starts] == [(1, 1, "cycle-001"), (2, 1, "cycle-001")]


def test_backend_releases_a_claim_when_the_cli_could_not_start(tmp_path):
    backend, session, where, _ = _fake_role(tmp_path, 0, [OSError("fake launch failure")])
    with pytest.raises(StopRun, match="could not run"):
        backend.respond("work", 0, where, session)
    st = budget.state(tmp_path / "run", backend.cfg["budget"])
    assert st["spent"] == 0 and st["rounds"] == {1: 0} and st["cycles"] == {"cycle-001": 0}
    assert st["stopped"] == 0


def test_backend_allows_only_the_tools_available_to_that_session(tmp_path):
    backend, session, where, calls = _fake_role(tmp_path, 0, [0.1])
    default = backend.command_line(where / "mcp.json")
    assert "assign_task" not in default[default.index("--allowedTools") + 1]
    session.TOOLS = {**session.TOOLS, "assign_task": {}}
    backend.respond("work", 0, where, session)
    allowed = calls[0][calls[0].index("--allowedTools") + 1]
    assert allowed == "mcp__lv1__run_python,mcp__lv1__submit_entry,mcp__lv1__assign_task"


def test_open_reservations_and_finished_unreported_costs_are_distinct_in_logs(tmp_path):
    cfg = grouped(round_usd=1.25, cycle_usd=5)
    backend, session, where, _ = _fake_role(tmp_path, 0, [0.5], cfg_over={"budget": cfg})
    run = tmp_path / "run"
    assert budget.claim(run, 99, 1, cfg) == 1.25
    logs = []
    backend.log = logs.append
    response = backend.respond("work", 0, where, session)
    st = budget.state(run, cfg)
    assert st["spent"] == 1.75
    assert st["unreported"] == st["stopped"] == st["open_tries"] == 1
    assert st["open_reserved_usd"] == 1.25 and st["completed_unreported_tries"] == 0
    assert "1 open attempt reserved at its cap ($1.25)" in logs[-1]
    assert "may still be running or interrupted" in logs[-1]
    assert "CLI reported no cost" not in logs[-1] and "completed attempt" not in logs[-1]
    assert response.meta["budget"]["open_tries"] == 1

    budget.end(run, 99, 1, None, "timed out", atomic=True)
    st = budget.state(run, cfg)
    assert st["spent"] == 1.75  # presentation changes do not alter what is counted
    assert st["unreported"] == st["completed_unreported_tries"] == 1
    assert st["stopped"] == st["open_tries"] == 0
    assert st["completed_unreported_usd"] == 1.25
    note = budget.accounting_note(st)
    assert "1 completed attempt without a usable reported cost counted at its cap ($1.25)" == note
    assert "open attempt" not in note
    assert budget.accounting_note({"unreported": 2, "stopped": 1}).startswith("1 open attempt")
