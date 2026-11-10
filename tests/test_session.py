"""A session's two tools against the real sandbox (process mode) and the real verifier."""

from pathlib import Path

import pytest

from level1 import run_config
from level1.session import Session, ToolError
from level1.util import read_json, read_jsonl


@pytest.fixture()
def session(tmp_path):
    cfg = run_config.load("config/pilot/lin-a.yaml", {"runs_dir": str(tmp_path), "sandbox": "process",
                                                      "session": {"python_calls": 3, "submissions": 2}})
    return Session.create(tmp_path / "run" / "round_001" / "session", cfg, "lin-a-s0", tmp_path / "run")


SAVE = """
g = graph()
a, b = g.token(0), g.token(1)
e = g.entry(predict=a, rule=a == b)
print(save_entry("diagonal", e), measure(e)["coverage"], info["n_total"], sorted(weights), model.reference.__name__)
"""


def test_run_python_sees_the_network_and_saves_entries(session):
    out = session.run_python(SAVE)
    assert out["status"] == "ok" and out["saved_entries"] == ["diagonal"] and out["sandbox"] == "process"
    assert "{'nodes': 3, 'const_scalars': 0} 0.015625 4096 ['embed', 'hidden', 'unembed'] reference" in out["output"]
    assert out["remaining"] == {"run_python": 2, "submit_entry": 2}
    assert read_json(session.entries_dir / "diagonal.json")["format"] == "level1-entry-v1"
    again = session.run_python("print(sorted(entries), entries['diagonal']['predict'])")
    assert "['diagonal'] 0" in again["output"]
    # every program and what went back to the agent are kept in full, whatever the log line shortens
    long = "x = 0\n" + "x += 1\n" * 3000 + "print(x)"
    session.run_python(long)
    assert (session.path / "python" / "001.py").read_text() == SAVE and (session.path / "python" / "003.py").read_text() == long
    assert (session.path / "python" / "003.txt").read_text() == "3000\n" and (session.path / "python" / "002.txt").read_text() == again["output"]


def test_the_guide_s_example_runs_as_written(session):
    """The code the agent is shown in agent_docs/ENTRY.md, pasted into run_python unchanged."""
    guide = (Path(__file__).parent.parent / "agent_docs" / "ENTRY.md").read_text()
    code = guide.split("```python\n", 1)[1].split("```", 1)[0]
    out = session.run_python(code + "\nprint(x.shape, x.value.dtype, graph(example=[3, 5]).token(1).value)")
    assert out["status"] == "ok" and "Traceback" not in out["output"], out["output"]
    assert out["saved_entries"] == ["example"] and "'coverage': 0.125" in out["output"] and "(8,) float64 5" in out["output"]
    # what the guide says of the costs, on that example: W[:, a] reads a column (8), times 2.0 (8), plus a column (8 + 8)
    from level1 import lang
    from level1.networks import load_model
    w = load_model("models/lin-a").load_weights("lin-a-s0")
    g = lang.Graph(w, 2)
    a, b = g.token(0), g.token(1)
    x = g.weight("embed")[:, a] * 2.0 + g.weight("embed")[:, b]
    _, _, _, c_pred = lang.Evaluator(g.entry(predict=x.argmax()), w, 2).run((1, 2))
    assert c_pred == 8 + 8 + 8 + 8 + 7


def test_mistakes_in_the_code_come_back_as_text(session):
    out = session.run_python("print('before')\nprint(1 / 0)")
    assert out["status"] == "ok" and "before" in out["output"] and "ZeroDivisionError" in out["output"]
    out = session.run_python("g = graph()\nsave_entry('bad name!', g.entry(predict=g.token(0)))\nsave_entry('fine', g.entry(predict=g.token(0)))")
    assert out["saved_entries"] == ["fine"] and "bad name!" in out["problems"][0]


def test_submit_scores_and_keeps_the_receipt(session):
    session.run_python(SAVE)
    r = session.submit_entry("diagonal", 0.9, 0.8)
    assert r["status"] == "scored" and r["parts"]["n_in_region"] == 64 and r["parts"]["n_agree"] == 63
    assert r["overclaim"] == 0.0 and r["remaining"]["submit_entry"] == 1
    receipt = session.receipts()[0]
    assert receipt["name"] == "diagonal" and receipt["submission"] == 1 and receipt["expected_fidelity_other"] == 0.8
    assert read_json(session.path / "submissions" / "001" / "entry.json") == read_json(session.entries_dir / "diagonal.json")
    log = read_jsonl(session.path / "tools.jsonl")
    assert [line["tool"] for line in log] == ["run_python", "submit_entry"] and log[1]["outcome"]["status"] == "scored"


def test_refused_calls_spend_nothing_and_budgets_hold(session):
    with pytest.raises(ToolError, match="no saved entry"):
        session.submit_entry("missing", 1.0)
    with pytest.raises(ToolError, match="from 0 to 1"):
        session.submit_entry("missing", 1.5)
    with pytest.raises(ToolError, match="does not take"):
        session.call("submit_entry", {"name": "x", "claimed_fidelity": 1.0, "network": "lin-a-s1"})
    with pytest.raises(ToolError, match="no tool"):
        session.call("read_file", {"path": "withheld/lin-a/task.yaml"})
    assert session.remaining() == {"run_python": 3, "submit_entry": 2}
    for _ in range(3):
        session.run_python("g = graph()\nsave_entry('e', g.entry(predict=g.token(0)))")
    with pytest.raises(ToolError, match="budget"):
        session.run_python("print(1)")
    session.submit_entry("e", 0.1)
    session.submit_entry("e", 0.1)
    with pytest.raises(ToolError, match="budget"):
        session.submit_entry("e", 0.1)
    reopened = Session(session.path)  # the tool server's process, or a resumed run
    assert reopened.remaining() == {"run_python": 0, "submit_entry": 0} and len(reopened.receipts()) == 2


def test_an_entry_too_big_is_not_saved(session):
    out = session.run_python("g = graph()\nsave_entry('big', g.entry(predict=g.const(list(range(1000)))[g.token(0)]))")
    assert out["saved_entries"] == [] and "constants" in out["output"]
