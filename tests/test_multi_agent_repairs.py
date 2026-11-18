"""Saved-program handoff and refusal barriers, without paid model calls."""

import copy
import threading

import pytest

from level1 import artifacts, lang, multi_agent, run_config, synthesis
from level1.backends.fake import FakeBackend
from level1.networks import load_model
from level1.session import ENTRY_NAME
from level1.util import read_json, write_json


def prepare(tmp_path, **overrides):
    base = {"run_id": "repair-test", "runs_dir": str(tmp_path), "backend": "fake", "sandbox": "process",
            "rounds_max": 1, "max_consecutive_refusals": 2,
            "session": {"python_calls": 30, "submissions": 4},
            "verifier": {"check": "sample", "sample_size": 32, "reference_samples": 1},
            "orchestration": {"enabled": True, "parallel": True, "final_summary": False, "baseline_agent": False}}
    cfg = run_config.load("config/pilot/lin-a-1to4.yaml", run_config.merge(base, overrides))
    path = run_config.run_path(cfg)
    path.mkdir(parents=True)
    write_json(path / "run.json", {"config": cfg, "run_id": cfg["run_id"], "sandbox": {"mode": "process"}})
    return path, cfg


def write_scripts(tmp_path, items):
    directory = tmp_path / "scripts"
    directory.mkdir()
    for index, item in enumerate(items, 1):
        write_json(directory / f"{index:02d}.json", item)
    return directory


def python(code):
    return {"tool": "run_python", "args": {"code": code}}


def assignment(length):
    return {"tool": "assign_region", "args": {"scope": {"kind": "lengths", "lengths": [length]},
                                                 "task": "Investigate this class using the saved draft."}}


def submit(name):
    return {"tool": "submit_entry", "args": {"name": name, "claimed_fidelity": 1.0}}


def state():
    return {"version": 1, "next_index": 0, "sessions": {}, "cycles": [], "stop_reason": None}


def plan(st, role="lead", cycle=0, snapshot=None, history=None):
    return multi_agent._new_plan(st, role, cycle, multi_agent.ALL, "Investigate saved evidence.", 2,
                                 multi_agent._empty_archive() if snapshot is None else snapshot, history or [])


def record(path, p, reason="end_turn", notes="A bounded observation.", **fields):
    value = {"id": p["id"], "index": p["index"], "role": p["role"], "cycle": p["cycle"],
             "scope": p["scope"], "notes": notes, "meta": {"stop_reason": reason},
             "submissions": [], "assignments": [], "tools_used": {}, **fields}
    write_json(path / p["directory"] / "record.json", value)
    return value


@pytest.mark.parametrize("reason", ["timeout", "turn_limit", "budget"])
def test_partial_notes_remain_labeled_in_history_without_rewriting_evidence(tmp_path, reason):
    path, _ = prepare(tmp_path)
    st = state()
    p = plan(st)
    notes = "## What the network does\nA partial observation that must not look complete."
    record(path, p, reason=reason, notes=notes)
    source = path / p["directory"] / "record.json"
    before = source.read_bytes()
    history = multi_agent._history(path, st, p["index"] + 1)
    assert history[0]["notes"].startswith(f"Partial research notes: this session ended with {reason}.")
    assert notes in history[0]["notes"]
    frozen = [{"session_id": p["id"], "notes": notes}]
    safe = multi_agent._safe_history(path, st, frozen)
    assert safe[0]["notes"] == history[0]["notes"]
    assert frozen[0]["notes"] == notes and source.read_bytes() == before


def entry(index=0):
    g = lang.Graph({}, 4)
    return g.entry(predict=g.token(index))


def saved(path, p, name, value):
    target = path / p["directory"] / "session/workspace/entries" / f"{name}.json"
    write_json(target, value)
    return target


@pytest.fixture(scope="module")
def handed_off(tmp_path_factory):
    where = tmp_path_factory.mktemp("saved-drafts")
    path, cfg = prepare(where)
    scripts = write_scripts(where, [
        {"calls": [python("g=graph()\nsave_entry('tables_v5',g.entry(predict=model.reference(g,lv1)))"),
                   assignment(2), assignment(1)], "response": "Use the saved tables_v5 program."},
        {"calls": [python("""assert 'tables_v5' in entries
assert 'worker_beta' not in entries
assert len([name for name in entries if name.startswith('draft_')]) == 1
save_entry('same_table',entries['tables_v5'])
g=graph()
save_entry('worker_alpha',g.entry(predict=g.token(0)))
""")], "response": "A draft remains available for later checking."},
        {"calls": [python("""assert 'tables_v5' in entries
assert 'worker_alpha' not in entries and 'same_table' not in entries
g=graph()
save_entry('worker_beta',g.entry(predict=g.token(1)))
""")], "response": "An independent draft is ready."},
        {"calls": [python("""assert {'tables_v5','same_table','worker_alpha','worker_beta'} <= entries.keys()
assert len([name for name in entries if name.startswith('draft_')]) == 3
save_entry('ready',entries['tables_v5'])
"""), submit("ready")], "response": "The carried program is now formally checked."},
    ])
    assert multi_agent.drive(path, fake_scripts=scripts, out=lambda *_: None) == "finished"
    return path, cfg


def test_zero_submission_planning_draft_reaches_workers_and_formal_global_selection(handed_off):
    path, _ = handed_off
    archive = multi_agent.read_archive(path)
    st = read_json(path / "multi_agent/state.json")
    planning = read_json(path / st["sessions"]["a0001"]["directory"] / "record.json")
    assert planning["submissions"] == []
    assert len(archive["candidates"]) == 1
    assert archive["global"]["best"] == "a0004s1"
    assert archive["candidates"][0]["selection"]["parts"]["fidelity"] == 1
    assert len(archive["drafts"]) == 3
    first = next(draft for draft in archive["drafts"] if draft["saved_name"] == "tables_v5")
    assert first["entry_sha256"] == archive["candidates"][0]["entry_sha256"]
    assert {(origin["session_id"], origin["saved_name"]) for origin in first["origins"]} == {
        ("a0001", "tables_v5"), ("a0002", "same_table")}
    assert all(draft["status"] == "unscored_draft" and "selection" not in draft for draft in archive["drafts"])
    frozen = read_json(path / "multi_agent/lock.json")
    assert [candidate["id"] for candidate in frozen["candidates"]] == ["a0004s1"]
    assert "not formal candidates" in multi_agent.archive_summary(archive, 8)


def test_draft_context_is_frozen_for_parallel_workers_and_no_recursive_copies(handed_off):
    path, cfg = handed_off
    st = read_json(path / "multi_agent/state.json")
    first, second, lead = (st["sessions"][key] for key in ("a0002", "a0003", "a0004"))
    assert first["snapshot"] == second["snapshot"]
    for worker in (first, second):
        carried = multi_agent._carried(path, cfg, worker["snapshot"])
        assert "tables_v5" in carried
        assert not {"worker_alpha", "worker_beta", "same_table"} & carried.keys()
        assert len(worker["snapshot"]["drafts"]) == 1
    assert len(lead["snapshot"]["drafts"]) == 3
    assert {"worker_alpha", "worker_beta", "same_table"} <= multi_agent._carried(path, cfg, lead["snapshot"]).keys()
    archive = multi_agent.read_archive(path)
    before = copy.deepcopy(archive)
    for p in st["sessions"].values():
        artifacts.collect(path, cfg, archive, p)
    assert archive == before


def test_deduplicated_artifacts_preserve_names_versions_and_immutable_content(tmp_path):
    path, cfg = prepare(tmp_path)
    archive, st = multi_agent._empty_archive(), state()
    first = plan(st)
    original = entry()
    source = saved(path, first, "foo", original)
    saved(path, first, "foo_copy", original)
    artifacts.collect(path, cfg, archive, first)
    assert len(archive["drafts"]) == 1
    old = copy.deepcopy(archive["drafts"][0])
    assert ENTRY_NAME.fullmatch(old["id"]) and len(old["id"]) <= 40
    assert {origin["saved_name"] for origin in old["origins"]} == {"foo", "foo_copy"}
    second = plan(st, "regional", 1, copy.deepcopy(archive))
    for name, value in artifacts.carried(path, archive).items():
        saved(path, second, name, value)
    saved(path, second, "bar", original)
    artifacts.collect(path, cfg, archive, second)
    assert len(archive["drafts"]) == 1
    assert {origin["saved_name"] for origin in archive["drafts"][0]["origins"]} == {"foo", "foo_copy", "bar"}
    assert artifacts.carried(path, archive)["bar"] == original
    write_json(source, entry(1))
    artifacts.collect(path, cfg, archive, first)
    assert len(archive["drafts"]) == 2
    aliases = artifacts.carried(path, archive)
    assert "foo" not in aliases  # two different saved versions require their immutable aliases
    assert aliases[old["id"]] == aliases["foo_copy"] == aliases["bar"] == original
    assert read_json(path / old["entry_path"]) == original
    write_json(path / old["entry_path"], entry(2))
    with pytest.raises(RuntimeError, match="changed after it was snapshotted"):
        artifacts.carried(path, archive)


def test_legacy_planning_drafts_are_backfilled_without_replaying_completed_session(tmp_path, monkeypatch):
    path, cfg = prepare(tmp_path)
    st, archive = state(), multi_agent._empty_archive()
    archive.pop("drafts")  # persisted before draft support
    planning = plan(st, snapshot=copy.deepcopy(archive))
    st["planning"] = planning["id"]
    record(path, planning, assignments=[assignment(2)["args"], assignment(1)["args"]])
    saved(path, planning, "tables_v5", entry())
    write_json(path / "multi_agent/state.json", st)
    write_json(path / "multi_agent/archive.json", archive)
    scripts = write_scripts(tmp_path, [
        {"crash": "completed planning must not replay"},
        {"calls": [python("assert 'tables_v5' in entries\nsave_entry('confirmed_one',entries['tables_v5'])")]},
        {"calls": [python("assert 'tables_v5' in entries\nsave_entry('confirmed_two',entries['tables_v5'])")]},
        {"calls": [submit("tables_v5")], "response": "A recovered artifact was checked."},
    ])
    calls, respond = [], FakeBackend.respond

    def track(self, text, index, where, session):
        calls.append(index)
        return respond(self, text, index, where, session)

    monkeypatch.setattr(FakeBackend, "respond", track)
    assert multi_agent.drive(path, fake_scripts=scripts, out=lambda *_: None) == "finished"
    assert sorted(calls) == [1, 2, 3]
    result = multi_agent.read_archive(path)
    assert len(result["drafts"]) == 1
    assert {origin["saved_name"] for origin in result["drafts"][0]["origins"]} == {
        "tables_v5", "confirmed_one", "confirmed_two"}
    assert result["global"]["best"] == "a0004s1"
    assert "tables_v5" not in multi_agent._carried(path, cfg, planning["snapshot"])


def test_interrupted_planning_preserves_saved_draft_without_promoting_it(tmp_path):
    path, _ = prepare(tmp_path)
    scripts = write_scripts(tmp_path, [{
        "calls": [python("g=graph()\nsave_entry('unfinished',g.entry(predict=g.token(0)))")],
        "stop_run": "a budget interruption before closing notes",
    }])
    assert multi_agent.drive(path, fake_scripts=scripts, out=lambda *_: None) == "stopped"
    archive = multi_agent.read_archive(path)
    assert archive["candidates"] == [] and archive["global"]["best"] is None
    assert len(archive["drafts"]) == 1 and archive["drafts"][0]["saved_name"] == "unfinished"
    assert not (path / "multi_agent/lock.json").exists()
    assert (path / archive["drafts"][0]["entry_path"]).exists()


def test_parallel_refusals_stop_before_lead_preserve_saved_work_and_latch_on_resume(tmp_path, monkeypatch):
    path, _ = prepare(tmp_path, rounds_max=2)
    raw = "PROVIDER_REQUEST_ID_opaque: protected reasoning refusal text."
    scripts = write_scripts(tmp_path, [
        {"calls": [assignment(2), assignment(1)], "response": "Two ordinary investigations."},
        {"calls": [python("g=graph()\nsave_entry('checked',g.entry(predict=g.token(0)))\nsave_entry('draft',g.entry(predict=g.token(1)))"),
                   submit("checked")], "response": raw, "stop_reason": "refusal"},
        {"response": raw, "stop_reason": "refusal"},
        {"crash": "the integration session must not start"},
    ])
    second_done, order, respond = threading.Event(), [], FakeBackend.respond

    def reverse_workers(self, text, index, where, session):
        if index == 1:
            assert second_done.wait(timeout=10)
        result = respond(self, text, index, where, session)
        order.append(index)
        if index == 2:
            second_done.set()
        return result

    monkeypatch.setattr(FakeBackend, "respond", reverse_workers)
    assert multi_agent.drive(path, fake_scripts=scripts, out=lambda *_: None) == "stopped"
    assert order == [0, 2, 1]
    st = read_json(path / "multi_agent/state.json")
    assert st["refusal_stop"]["session_ids"] == ["a0002", "a0003"]
    assert st["sessions"]["a0004"]["status"] == "planned"
    assert not (path / st["sessions"]["a0004"]["directory"]).exists()
    assert not (path / "multi_agent/lock.json").exists()
    archive = multi_agent.read_archive(path)
    assert [candidate["id"] for candidate in archive["candidates"]] == ["a0002s1"]
    assert len(archive["drafts"]) == 1 and archive["drafts"][0]["saved_name"] == "draft"
    histories = multi_agent._history(path, st, 4)
    assert raw not in str(histories)
    assert histories[1]["submissions"][0]["status"] == "scored"
    records = {p: p.read_bytes() for p in (path / "multi_agent/sessions").glob("*/record.json")}
    monkeypatch.setattr(multi_agent, "make", lambda *a, **k: pytest.fail("a refusal latch forbids new sessions"))
    assert multi_agent.drive(path, fake_scripts=scripts, out=lambda *_: None) == "stopped"
    assert all(p.read_bytes() == data for p, data in records.items())
    assert raw in (path / st["sessions"]["a0002"]["directory"] / "response.md").read_text()


def test_legacy_refusal_streak_stays_latched_despite_later_success(tmp_path, monkeypatch):
    path, _ = prepare(tmp_path, rounds_max=2)
    st = state()
    for index, reason in enumerate(("end_turn", "refusal", "refusal", "end_turn")):
        p = plan(st, "lead" if index in (0, 3) else "regional", int(index > 0))
        record(path, p, reason)
    st["planning"] = "a0001"
    write_json(path / "multi_agent/state.json", st)
    monkeypatch.setattr(multi_agent, "make", lambda *a, **k: pytest.fail("legacy refusal thresholds must stop before a call"))
    assert multi_agent.drive(path, out=lambda *_: None) == "stopped"
    assert read_json(path / "multi_agent/state.json")["refusal_stop"]["session_ids"] == ["a0002", "a0003"]


def test_legacy_pending_prompt_uses_neutral_history_without_editing_raw_evidence(tmp_path, monkeypatch):
    path, cfg = prepare(tmp_path)
    st = state()
    first = plan(st)
    record(path, first, notes="Ordinary earlier research notes.")
    refused = plan(st, "regional", 1)
    raw = "PRIVATE_PROVIDER_REQUEST_ID and diagnostic advice."
    checked = {"status": "scored", "submission": 1, "reward": 0.1}
    record(path, refused, notes=raw, submissions=[checked],
           meta={"stop_reason": "error", "model_events": [{"subtype": "model_refusal_no_fallback"}]})
    history = [{"session_id": first["id"], "notes": "Ordinary earlier research notes."},
               {"session_id": refused["id"], "notes": raw, "submissions": [checked]}]
    lead = plan(st, "lead", 1, history=history)
    old_prompt = path / lead["directory"] / "prompt.md"
    old_prompt.parent.mkdir(parents=True)
    old_prompt.write_text("Legacy prompt containing " + raw)
    scripts = write_scripts(tmp_path, [{}, {}, {"response": "A new bounded observation."}])
    seen, respond = [], FakeBackend.respond

    def inspect(self, text, index, where, session):
        seen.append(text)
        return respond(self, text, index, where, session)

    monkeypatch.setattr(FakeBackend, "respond", inspect)
    result = multi_agent._run_session(path, cfg, load_model(cfg["model"]), st, lead, scripts, lambda *_: None)
    assert len(seen) == 1 and raw not in seen[0]
    assert "Ordinary earlier research notes." in seen[0]
    assert "Provider diagnostics are retained separately" in seen[0]
    assert result["prompt_file"] == "prompt.history_safe.md"
    assert raw in old_prompt.read_text() and lead["history"] == history
    safe = multi_agent._safe_history(path, st, history)
    assert safe[1]["submissions"] == [checked] and safe[0] == history[0]


@pytest.mark.parametrize("cached, expected", [(None, "stopped"), ("refusal", "incomplete"), ("error", "incomplete"),
                                               ("end_turn", "completed")])
def test_refusal_gate_blocks_new_summary_calls_but_recovers_cached_output(tmp_path, monkeypatch, cached, expected):
    path, cfg = prepare(tmp_path, orchestration={"final_summary": True})
    st, archive = state(), multi_agent._empty_archive()
    for index, reason in enumerate(("end_turn", "refusal", "refusal", "end_turn")):
        p = plan(st, "lead" if index in (0, 3) else "regional", int(index > 0))
        record(path, p, reason)
    st["planning"] = "a0001"
    if cached:
        summary = plan(st, "summary", 2)
        summary["directory"] = "multi_agent/summary"
        st["summary"] = summary["id"]
        write_json(path / "multi_agent/summary/response.json", {"text": "# Saved synthesis\nAn evidence-based report.\n",
                                                               "meta": {"stop_reason": cached}})
    write_json(path / "multi_agent/state.json", st)
    write_json(path / "multi_agent/archive.json", archive)
    frozen = {"version": 1, "global": None, "regional": {}, "candidates": [], "at": "2026-01-01T00:00:00Z"}
    write_json(path / "multi_agent/lock.json", frozen)
    preserved = (path / "multi_agent/lock.json").read_bytes()
    monkeypatch.setattr(multi_agent, "make", lambda *a, **k: pytest.fail("locked research cannot call an agent"))
    monkeypatch.setattr(synthesis, "make", lambda *a, **k: pytest.fail("a refusal threshold forbids new summary calls"))
    assert multi_agent.drive(path, out=lambda *_: None) == "finished"
    assert read_json(path / "multi_agent/summary/status.json")["status"] == expected
    assert (path / "multi_agent/lock.json").read_bytes() == preserved
    assert multi_agent.read_archive(path) == archive
    after = read_json(path / "multi_agent/state.json")
    assert after["refusal_stop"]["session_ids"] == ["a0002", "a0003"]
    assert after["next_index"] == st["next_index"]
