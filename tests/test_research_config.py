"""Model-neutral research setup and the reserved final-report allowance."""

import copy

import pytest

from level1 import multi_agent, run_config


def research(**overrides):
    return run_config.load("config/experiments/lin-c-regional.yaml", overrides)


@pytest.mark.parametrize("model,network", [
    ("models/lin-a", "lin-a-s0"),
    ("models/att-a", "att-a-s0"),
    ("models/rnn-a", "rnn-a-s0"),
])
def test_shared_profile_accepts_different_model_interfaces(model, network):
    cfg = run_config.load("config/experiments/research.yaml", {
        "run_id": "model-independent-research", "model": model, "network": network,
        "verifier": {"workload_size": 4096},
    })
    assert cfg["model"] == model and cfg["network"] == network
    assert cfg["orchestration"]["enabled"] and cfg["orchestration"]["final_summary"]
    assert cfg["verifier"]["scope"] == {"kind": "all"}
    assert all(task["scope"] == {"kind": "all"} for task in cfg["orchestration"]["initial_regions"])


def test_standard_example_does_not_preassign_input_lengths():
    cfg = research()
    assert cfg["verifier"]["sample_size"] == 16384
    assert cfg["seed_check"]["network"] == "lin-c-s1"
    assert all(task["scope"] == {"kind": "all"} for task in cfg["orchestration"]["initial_regions"])
    control = run_config.load("config/experiments/lin-c-length2.yaml")
    assert not control["orchestration"]["enabled"]
    assert control["verifier"]["scope"] == {"kind": "lengths", "lengths": [2]}


def test_shared_profile_declares_regional_claim_slots_and_an_integration_checkpoint():
    cfg = research()
    assert cfg["session"]["submissions"] == 6
    assert cfg["orchestration"]["regional_submissions"] == 2
    assert cfg["orchestration"]["checkpoint_python_calls"] == 2
    allocations = [multi_agent._allocation(cfg, role, 1, 2) for role in ("regional", "regional", "lead")]
    assert [a[1] for a in allocations] == [2, 2, 2]
    assert sum(a[1] for a in allocations) == cfg["session"]["submissions"]
    plan = {"scope": {"kind": "all"}, "role": "lead", "cycle": 1,
            "worker_count": 2, "task": "Integrate checked findings.", "id": "a0004"}
    assert multi_agent.role_config(cfg, plan)["research_task"]["checkpoint_python_calls"] == 2
    plan["cycle"] = 0
    assert "checkpoint_python_calls" not in multi_agent.role_config(cfg, plan)["research_task"]


def test_saved_configs_keep_their_old_submission_allocation_and_checkpoint_policy():
    cfg = copy.deepcopy(research())
    cfg["orchestration"].pop("regional_submissions")
    cfg["orchestration"].pop("checkpoint_python_calls")
    cfg["session"]["submissions"] = 4
    run_config.check_orchestration(cfg)
    assert [multi_agent._allocation(cfg, role, 1, 2)[1] for role in ("regional", "regional", "lead")] == [1, 1, 2]
    plan = {"scope": {"kind": "all"}, "role": "lead", "cycle": 1,
            "worker_count": 2, "task": "Integrate checked findings.", "id": "a0004"}
    assert "checkpoint_python_calls" not in multi_agent.role_config(cfg, plan)["research_task"]


@pytest.mark.parametrize("value", [0, -1, True, 1.5, "2"])
def test_regional_submission_quota_is_explicit_and_validated(value):
    with pytest.raises(run_config.ConfigError, match="regional_submissions"):
        research(orchestration={"regional_submissions": value})


@pytest.mark.parametrize("value", [0, -1, True, 1.5, "2"])
def test_checkpoint_window_is_positive_or_disabled(value):
    with pytest.raises(run_config.ConfigError, match="checkpoint_python_calls"):
        research(orchestration={"checkpoint_python_calls": value})


def test_worker_slots_cannot_consume_the_integrators_declared_allowance():
    with pytest.raises(run_config.ConfigError, match="workers \\* regional_submissions"):
        research(session={"submissions": 4})
    cfg = research(session={"submissions": 4}, orchestration={"regional_submissions": 1, "checkpoint_python_calls": None})
    assert multi_agent._allocation(cfg, "lead", 1, 2)[1] == 2


@pytest.mark.parametrize("value", [-1, 1, 1.5, True, "0.1", float("nan"), float("inf")])
def test_summary_fraction_rejects_invalid_reservations(value):
    with pytest.raises(run_config.ConfigError, match="summary_fraction"):
        research(orchestration={"summary_fraction": value})


def test_summary_needs_a_positive_reservation_and_an_actual_turn():
    with pytest.raises(run_config.ConfigError, match="positive summary_fraction"):
        research(orchestration={"summary_fraction": 0})
    with pytest.raises(run_config.ConfigError, match="summary allocation"):
        research(orchestration={"summary_fraction": 0.0001})
    cfg = research(orchestration={"final_summary": False, "summary_fraction": 0})
    assert not cfg["orchestration"]["final_summary"]


def test_summary_reservation_cannot_leave_research_without_turns():
    with pytest.raises(run_config.ConfigError, match="too small after sharing"):
        research(orchestration={"summary_fraction": 0.999, "baseline_agent": False})


def test_old_saved_options_do_not_implicitly_request_a_paid_summary():
    cfg = copy.deepcopy(research())
    cfg["orchestration"].pop("final_summary")
    cfg["orchestration"].pop("summary_fraction")
    run_config.check_orchestration(cfg)
    assert "final_summary" not in cfg["orchestration"]


@pytest.mark.parametrize("value", [1, "true", None])
def test_summary_switch_is_boolean(value):
    with pytest.raises(run_config.ConfigError, match="final_summary"):
        research(orchestration={"final_summary": value})
