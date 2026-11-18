"""Stage selection and resource validation for independent research modes."""

import copy

import pytest

from level1 import research_modes, run_config


def config(**overrides):
    values = {
        "run_id": "research-modes-config",
        "backend": "fake",
        "rounds_max": 1,
        "session": {"python_calls": 80, "max_turns": 150, "submissions": 6},
        "orchestration": {"enabled": True},
    }
    return run_config.load("config/pilot/lin-a-1to4.yaml", run_config.merge(values, overrides))


@pytest.mark.parametrize("mode,old_flag,requires,generates,fraction", [
    ("trained", True, False, False, 0.0),
    ("baseline", False, True, True, 1.0),
    ("both", False, True, True, 0.2),
])
def test_explicit_mode_wins_legacy_flag(mode, old_flag, requires, generates, fraction):
    cfg = config(orchestration={"mode": mode, "baseline_agent": old_flag})
    assert research_modes.mode(cfg) == mode
    assert research_modes.requires_baseline(cfg) is requires
    assert research_modes.generates_baseline(cfg) is generates
    assert research_modes.baseline_fraction(cfg) == fraction


@pytest.mark.parametrize("options,expected,generates", [
    ({"baseline_agent": True}, "both", True),
    ({"baseline_agent": False}, "trained", False),
    ({"baseline_agent": False, "baseline_entry": "/provided/program.json"}, "both", False),
])
def test_null_mode_keeps_legacy_stage_selection(options, expected, generates):
    cfg = config(orchestration={"mode": None, **options})
    assert research_modes.mode(cfg) == expected
    assert research_modes.generates_baseline(cfg) is generates


def test_saved_config_does_not_acquire_a_validation_policy_or_new_phase():
    cfg = config(orchestration={"baseline_agent": False})
    for key in ("mode", "baseline_agent", "baseline_fraction", "baseline_submissions", "baseline_validation_version"):
        cfg["orchestration"].pop(key)
    original = copy.deepcopy(cfg)
    run_config.check_orchestration(cfg)
    assert cfg == original
    assert research_modes.mode(cfg) == "trained"
    assert not research_modes.requires_baseline(cfg)
    assert research_modes.validation_version(cfg) == 1


def test_saved_automatic_baseline_keeps_its_original_validation_contract():
    cfg = config()
    cfg["orchestration"].pop("mode")
    cfg["orchestration"].pop("baseline_validation_version")
    original = copy.deepcopy(cfg)
    run_config.check_orchestration(cfg)
    assert cfg == original
    assert research_modes.mode(cfg) == "both"
    assert research_modes.generates_baseline(cfg)
    assert research_modes.validation_version(cfg) == 1


def test_old_supplied_comparison_remains_available_without_paid_generation():
    cfg = config(orchestration={"baseline_entry": "/provided/program.json"})
    for key in ("mode", "baseline_agent", "baseline_fraction", "baseline_submissions", "baseline_validation_version"):
        cfg["orchestration"].pop(key)
    original = copy.deepcopy(cfg)
    run_config.check_orchestration(cfg)
    assert cfg == original
    assert research_modes.mode(cfg) == "both"
    assert research_modes.requires_baseline(cfg)
    assert not research_modes.generates_baseline(cfg)
    assert research_modes.baseline_fraction(cfg) == 0


@pytest.mark.parametrize("mode", ["trained", "baseline", "both"])
def test_new_modes_require_current_validation_policy(mode):
    cfg = config(orchestration={"mode": mode})
    assert research_modes.validation_version(cfg) == 2


@pytest.mark.parametrize("version", [None, 0, 1, 3, True, 2.0, "2"])
def test_new_config_cannot_downgrade_or_invent_validation_version(version):
    with pytest.raises(run_config.ConfigError, match="baseline_validation_version: 2"):
        config(orchestration={"baseline_validation_version": version})


@pytest.mark.parametrize("mode", ["", "other", True, 1, [], {}])
def test_invalid_modes_are_configuration_errors(mode):
    with pytest.raises(run_config.ConfigError, match="orchestration.mode"):
        config(orchestration={"mode": mode})


@pytest.mark.parametrize("mode", ["trained", "baseline", "both"])
def test_explicit_mode_requires_orchestration_opt_in(mode):
    with pytest.raises(run_config.ConfigError, match="requires orchestration.enabled"):
        config(orchestration={"enabled": False, "mode": mode})


def test_original_single_agent_configs_do_not_acquire_a_research_baseline():
    cfg = run_config.load("config/pilot/lin-a-1to4.yaml")
    assert not cfg["orchestration"]["enabled"]
    assert research_modes.mode(cfg) == "trained"
    assert not research_modes.requires_baseline(cfg)
    assert not research_modes.generates_baseline(cfg)
    assert research_modes.baseline_fraction(cfg) == 0


def test_explicit_trained_mode_rejects_a_supplied_comparison():
    with pytest.raises(run_config.ConfigError, match="baseline_entry conflicts with explicit trained mode"):
        config(orchestration={"mode": "trained", "baseline_entry": "/provided/program.json"})


@pytest.mark.parametrize("mode", ["baseline", "both"])
def test_supplied_entry_requires_validation_without_a_generation_reservation(mode):
    cfg = config(orchestration={"mode": mode, "baseline_agent": False,
                                "baseline_entry": "/provided/program.json", "baseline_fraction": 0.95})
    assert research_modes.requires_baseline(cfg)
    assert not research_modes.generates_baseline(cfg)
    assert research_modes.baseline_fraction(cfg) == 0
    assert research_modes.validation_version(cfg) == 2


def test_baseline_only_uses_full_share_and_no_research_constraints():
    cfg = config(session={"submissions": 0, "python_calls": 1, "max_turns": 1},
                 orchestration={"mode": "baseline", "workers": 0, "regional_submissions": 0,
                                "planning_fraction": 0, "regional_fraction": 1, "initial_regions": None,
                                "archive_per_scope": 0, "baseline_fraction": 1,
                                "final_summary": True, "summary_fraction": 1})
    assert research_modes.baseline_fraction(cfg) == 1
    assert cfg["orchestration"]["baseline_submissions"] == 2
    assert cfg["session"]["submissions"] == 0
    assert cfg["verifier"]["scope"] == {"kind": "all"}


@pytest.mark.parametrize("key", ["python_calls", "max_turns"])
def test_baseline_only_still_requires_actual_agent_resources(key):
    with pytest.raises(run_config.ConfigError, match=f"session.{key} must be a positive"):
        config(session={key: 0}, orchestration={"mode": "baseline"})


@pytest.mark.parametrize("value", [-1, True, 1.5])
def test_baseline_only_unused_research_submission_count_is_nonnegative_integer(value):
    with pytest.raises(run_config.ConfigError, match="session.submissions must be a nonnegative"):
        config(session={"submissions": value}, orchestration={"mode": "baseline"})


def test_trained_only_does_not_reserve_legacy_baseline_share():
    cfg = config(orchestration={"mode": "trained", "baseline_agent": True,
                                "baseline_fraction": 0.95, "summary_fraction": 0.1})
    assert research_modes.baseline_fraction(cfg) == 0
    assert cfg["orchestration"]["baseline_fraction"] + cfg["orchestration"]["summary_fraction"] > 1


def test_both_mode_must_leave_research_a_positive_share_even_with_legacy_flag_off():
    with pytest.raises(run_config.ConfigError, match="leave a positive share for research"):
        config(orchestration={"mode": "both", "baseline_agent": False,
                              "baseline_fraction": 0.9, "summary_fraction": 0.1})


def test_both_mode_keeps_research_submission_requirements():
    with pytest.raises(run_config.ConfigError, match="session.submissions must be a positive"):
        config(session={"submissions": 0}, orchestration={"mode": "both"})


def test_global_scope_is_required_in_baseline_only_mode():
    with pytest.raises(run_config.ConfigError, match="global task must use scope kind all"):
        config(orchestration={"mode": "baseline"}, verifier={"scope": {"kind": "lengths", "lengths": [2]}})
