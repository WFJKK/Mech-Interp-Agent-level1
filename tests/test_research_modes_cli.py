"""Explicit CLI stage choices do not silently start a different workflow."""

import pytest

from level1 import loop, research_modes, run_config
from level1.util import write_json


@pytest.mark.parametrize("mode", ["trained", "baseline", "both"])
def test_mode_flag_opts_a_model_config_into_the_research_controller(tmp_path, monkeypatch, mode):
    received = []

    def start(cfg, **kwargs):
        received.append(cfg)
        path = run_config.run_path(cfg)
        write_json(path / "multi_agent/lock.json", {"mode": mode})
        return path

    monkeypatch.setattr(loop, "start", start)
    assert loop.main(["run", "--config", "config/pilot/lin-c.yaml", "--mode", mode,
                      "--runs-dir", str(tmp_path), "--run-id", "cli-mode", "--budget-usd", "25"]) == 0
    assert len(received) == 1
    cfg = received[0]
    assert cfg["orchestration"]["enabled"] is True and research_modes.mode(cfg) == mode
    assert cfg["orchestration"]["baseline_validation_version"] == 2
    assert cfg["budget"]["run_usd"] == 25


def test_without_mode_flag_the_original_single_agent_config_stays_single_agent(tmp_path, monkeypatch):
    received = []

    def start(cfg, **kwargs):
        received.append(cfg)
        return run_config.run_path(cfg)

    monkeypatch.setattr(loop, "start", start)
    assert loop.main(["run", "--config", "config/pilot/lin-c-smoke.yaml", "--runs-dir", str(tmp_path)]) == 0
    assert received[0]["orchestration"]["enabled"] is False


@pytest.mark.parametrize("mode", ["trained", "baseline", "both"])
def test_incomplete_research_workflow_does_not_exit_successfully(tmp_path, monkeypatch, mode):
    monkeypatch.setattr(loop, "start", lambda cfg, **kw: run_config.run_path(cfg))
    assert loop.main(["run", "--config", "config/pilot/lin-c.yaml", "--mode", mode,
                      "--runs-dir", str(tmp_path)]) == 2


def test_with_baseline_cannot_override_explicit_trained_mode(tmp_path, monkeypatch, capsys):
    def no_start(*args, **kwargs):
        pytest.fail("contradictory stage flags must be rejected before starting")

    monkeypatch.setattr(loop, "start", no_start)
    assert loop.main(["run", "--config", "config/pilot/lin-c.yaml", "--mode", "trained",
                      "--with-baseline", "--runs-dir", str(tmp_path)]) == 1
    assert "trained-only run keeps its saved stages" in capsys.readouterr().err


def test_resume_has_no_mode_override(tmp_path):
    with pytest.raises(SystemExit) as result:
        loop.main(["resume", "--run", str(tmp_path / "old-run"), "--mode", "baseline"])
    assert result.value.code == 2
