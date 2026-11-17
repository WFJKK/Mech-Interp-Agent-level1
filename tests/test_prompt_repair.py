"""Research reports describe the supplied target and respect each role's real allowance."""

import pytest

from level1 import hygiene, loop, prompt, run_config
from level1.networks import load_model


def config(model="lin-a-1to4", **over):
    research = over.pop("research_task", {"role": "lead", "max_assignments": 2})
    cfg = run_config.load(f"config/pilot/{model}.yaml", run_config.merge({
        "backend": "fake", "auth": "api_key", "budget": {"run_usd": 60, "round_usd": 3.24},
        "verifier": {"scope": {"kind": "all"}, "sample_size": 16, "workload_size": 16384},
    }, over))
    cfg["research_task"] = research
    return cfg


@pytest.mark.parametrize("model_name", ["lin-a-1to4", "lin-e", "att-c", "rnn-a"])
def test_research_reporting_and_allowance_are_generic_across_models(model_name):
    cfg = config(model_name)
    model = load_model(cfg["model"])
    text = prompt.build_multi(cfg, model, 0, "lead", "Investigate the supplied target.", "", [])
    assert "separate from the assistant conducting the investigation" in text
    assert "independently checkable mathematical justifications" in text
    assert "## Experiments and evidence" in text
    assert "experiments and reasoning" not in text and "## How you found it" not in text
    assert "nominal API allowance for this session is $3.24" in text
    assert "Submit a promising candidate with `submit_entry` while enough" in text
    assert "promising carried draft" in text
    assert "does not give it a formal score" in text
    assert "simple explanations on subdomains" in text
    assert "configured fidelity and cost tradeoff" in text
    surface = loop.agent_surface(cfg, model)
    assert surface["API attempt allowance"] == prompt.API_ATTEMPT_NOTE
    hygiene.check({**surface, "research prompt": text, "actual allowance": prompt.api_attempt_note(1.84)},
                  loop.forbidden_strings(cfg, model))


def test_planning_saves_a_usable_artifact_without_requesting_an_unavailable_submission():
    cfg = config(session={"submissions": 0})
    text = prompt.build_multi(cfg, load_model(cfg["model"]), 0, "lead", "Plan investigations.", "", [])
    assert "Save a usable executable entry" in text and "before extended optimization" in text
    assert "`save_entry(name, entry)`" in text and "`reference_entry()`" in text
    assert "This session has no formal submission allowance" in text
    assert "Submit a promising candidate with `submit_entry`" not in text


def test_a_single_submission_is_reserved_for_a_promising_candidate():
    text = prompt.research_deliverable(config(session={"submissions": 1}))
    assert "one formal submission; keep a starting reference as a draft" in text
    assert "save further improvements for a later session" in text


def test_login_does_not_advertise_an_enforced_dollar_allowance():
    cfg = config(auth="login")
    text = prompt.research_allocation(cfg)
    assert "does not enforce an API dollar cap" in text
    assert "$3.24" not in text
    assert str(cfg["session"]["max_turns"]) in text


def test_final_synthesis_requests_evidence_without_new_work_or_private_reasoning():
    cfg = config(research_task={"role": "summary", "max_assignments": 0})
    model = load_model(cfg["model"])
    text = prompt.build_summary(cfg, model, {})
    assert "supplied numerical target network" in text
    assert "independently checkable mathematical justifications" in text
    assert "No tools are available" in text
    assert "Submit a promising candidate with `submit_entry`" not in text
    assert "do not reproduce or request private internal reasoning" in text
    hygiene.check({"summary prompt": text}, loop.forbidden_strings(cfg, model))
