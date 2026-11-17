"""Model-neutral research setup and the reserved final-report allowance."""

import copy

import pytest

from level1 import run_config


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
        research(orchestration={"summary_fraction": 0.999})


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
