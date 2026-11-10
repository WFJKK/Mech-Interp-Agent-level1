"""The agent is not told what the model does: everything it can reach, searched for the forbidden strings."""

from pathlib import Path

import pytest

from level1 import hygiene, loop, prompt, run_config
from level1.session import surface

REPO = Path(__file__).parent.parent


@pytest.fixture(scope="module")
def cfg():
    return run_config.load("config/pilot/lin-a.yaml")


def test_the_surface_and_the_prompt_are_clean(model, cfg):
    forbidden = loop.forbidden_strings(cfg, model)
    assert {"max of", "larger", "Gross"} <= set(forbidden)
    texts = surface(model, cfg["verifier"])
    assert {"sandbox file lv1.py", "sandbox file model.py", "sandbox file rewards.py", "sandbox file harness.py", "sandbox wrapper",
            "tool run_python", "tool submit_entry", "sandbox info"} <= set(texts)
    texts["prompt"] = prompt.build(cfg, model, 1, [], 655, True)
    texts["prompt on a sampled data set"] = prompt.build(cfg, model, 1, [], 655, False)
    # the history's own words (the table and the lines around it), with a made-up round in the agent's place
    made_up = [{"round": 1, "notes": "", "submissions": [
        {"status": "scored", "carried_as": "r1s1", "name": "e", "reward": 0.5, "parts": {"coverage": 1, "fidelity": 1, "r": 0, "r_rule": 0, "b": 0},
         "claimed_fidelity": 1.0, "expected_fidelity_other": 1.0},
        {"status": "refused", "carried_as": "r1s2", "name": "e", "reason": "why"}]}, {"round": 2, "notes": "", "submissions": []}]
    texts["history"] = prompt.history(made_up)
    hygiene.check(texts, forbidden)


def test_the_restart_note_and_the_sandbox_folder_are_checked_too(model, cfg):
    forbidden = loop.forbidden_strings(cfg, model)
    texts = loop.agent_surface(cfg, model)
    assert {"restart note", "sandbox folder's location", "python's location", "temporary folder's location"} <= set(texts)
    assert "{python_left}" in texts["restart note"]
    hygiene.check(texts, forbidden)
    named = loop.agent_surface({**cfg, "sandbox_dir": "/data/max2-boxes"}, model)
    assert hygiene.hits(named, forbidden) == [("sandbox folder's location", "max2", "/data/max2-boxes")]
    # a container has its own paths: the host's sandbox folder and Python do not show in it
    assert "sandbox folder's location" not in loop.agent_surface({**cfg, "sandbox_dir": "/data/max2-boxes"}, model, "container")


def test_the_check_finds_what_it_looks_for(model, cfg):
    forbidden = loop.forbidden_strings(cfg, model)
    found = hygiene.hits({"a file": "line one\nIt outputs the LARGER token.\nmax OF two"}, forbidden)
    assert ("a file", "larger", "It outputs the LARGER token.") in found and ("a file", "max of", "max OF two") in found
    with pytest.raises(hygiene.HygieneError, match="2 places"):
        hygiene.check({"a file": "the largest\nGross et al."}, forbidden)
    # the list is not empty words: the withheld folder itself is full of them
    withheld = {p.name: p.read_text() for p in model.withheld().iterdir() if p.is_file()}
    assert len(hygiene.hits(withheld, forbidden)) > 5


def test_nothing_names_the_networks_or_the_pipeline_to_the_agent(model, cfg):
    text = prompt.build(cfg, model, 1, [], 655, True) + "\n".join(surface(model, cfg["verifier"]).values())
    for word in ("lin-a", "withheld", "seed_check", "Compact", "level1/", "github"):
        assert word not in text, word


def test_the_agent_facing_files_are_the_ones_checked(model, cfg):
    """Every file the prompt is built from is under prompts/, agent_docs/ or the model folder's model.py."""
    assert prompt.TEMPLATES == REPO / "prompts" and prompt.ENTRY_DOC == REPO / "agent_docs" / "ENTRY.md"
    built = prompt.build(cfg, model, 1, [], 655, True)
    assert prompt.ENTRY_DOC.read_text().strip() in built and model.source().strip() in built
