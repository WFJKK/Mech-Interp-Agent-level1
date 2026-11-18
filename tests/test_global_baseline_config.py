"""Configuration gates for the optional single global baseline session."""

import copy

import pytest

from level1 import run_config


def config(tmp_path, **overrides):
    values = {
        "run_id": "global-baseline-config",
        "runs_dir": str(tmp_path),
        "backend": "fake",
        "rounds_max": 1,
        "session": {"python_calls": 80, "max_turns": 150, "submissions": 6},
        "orchestration": {"enabled": True},
    }
    return run_config.load("config/pilot/lin-a-1to4.yaml", run_config.merge(values, overrides))


def test_new_research_config_enables_one_global_baseline(tmp_path):
    cfg = config(tmp_path)
    options = cfg["orchestration"]
    assert options["baseline_agent"] is True
    assert options["baseline_fraction"] == 0.2
    assert options["baseline_submissions"] == 2
    assert options["baseline_entry"] is None
    assert cfg["session"]["submissions"] == 6
    assert cfg["budget"]["run_usd"] == 60


def test_saved_config_without_baseline_keys_remains_unchanged(tmp_path):
    cfg = config(tmp_path)
    for key in ("baseline_agent", "baseline_fraction", "baseline_submissions"):
        del cfg["orchestration"][key]
    original = copy.deepcopy(cfg)
    run_config.check_orchestration(cfg)
    assert cfg == original
    assert "baseline_agent" not in cfg["orchestration"]


@pytest.mark.parametrize("value", [None, "true", 1, 0])
def test_baseline_agent_flag_requires_bool(tmp_path, value):
    with pytest.raises(run_config.ConfigError, match="baseline_agent must be true or false"):
        config(tmp_path, orchestration={"baseline_agent": value})


@pytest.mark.parametrize("value", [0, -0.1, 1, float("nan"), float("inf"), True, "0.2"])
def test_automatic_baseline_fraction_must_reserve_a_valid_share(tmp_path, value):
    with pytest.raises(run_config.ConfigError, match="baseline_fraction"):
        config(tmp_path, orchestration={"baseline_fraction": value})


@pytest.mark.parametrize("value", [None, 0, 1, True, 2.5])
def test_baseline_submission_quota_leaves_room_after_checkpoint(tmp_path, value):
    with pytest.raises(run_config.ConfigError, match="baseline_submissions"):
        config(tmp_path, orchestration={"baseline_submissions": value})


def test_baseline_and_summary_cannot_consume_the_entire_allowance(tmp_path):
    with pytest.raises(run_config.ConfigError, match="leave a positive share for research"):
        config(tmp_path, orchestration={"baseline_fraction": 0.9, "summary_fraction": 0.1})


@pytest.mark.parametrize("key,value", [("python_calls", 4), ("max_turns", 4)])
def test_baseline_receives_at_least_one_call_and_turn(tmp_path, key, value):
    with pytest.raises(run_config.ConfigError, match=f"baseline allocation.*{key}"):
        config(tmp_path, session={key: value})


@pytest.mark.parametrize("key,value", [("python_calls", 7), ("max_turns", 30)])
def test_reservation_cannot_starve_research_roles(tmp_path, key, value):
    with pytest.raises(run_config.ConfigError, match=f"session.{key} is too small"):
        config(tmp_path, session={key: value}, orchestration={"baseline_fraction": 0.8})


def test_supplied_program_consumes_no_baseline_share(tmp_path):
    cfg = config(tmp_path, orchestration={"baseline_entry": "/provided/program.json", "baseline_fraction": 0.95})
    assert cfg["orchestration"]["baseline_agent"] is True
    assert cfg["orchestration"]["baseline_entry"] == "/provided/program.json"
    # It would exceed the total if the disabled generation fraction were reserved.
    assert cfg["orchestration"]["baseline_fraction"] + cfg["orchestration"]["summary_fraction"] > 1


def test_disabling_agent_consumes_no_baseline_share(tmp_path):
    cfg = config(tmp_path, orchestration={"baseline_agent": False, "baseline_fraction": 0.95})
    assert cfg["orchestration"]["baseline_agent"] is False


def test_disabling_summary_releases_its_share(tmp_path):
    cfg = config(tmp_path, orchestration={"final_summary": False, "baseline_fraction": 0.8},
                 session={"python_calls": 100, "max_turns": 200})
    assert cfg["orchestration"]["baseline_fraction"] == 0.8


@pytest.mark.parametrize("value", ["", " ", 1, {}])
def test_invalid_supplied_entry_does_not_disable_required_baseline(tmp_path, value):
    with pytest.raises(run_config.ConfigError, match="baseline_entry must be a path or null"):
        config(tmp_path, orchestration={"baseline_entry": value})
