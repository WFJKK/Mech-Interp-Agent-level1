"""A round's prompt: the template, the network's definition, the entry guide, the scoring and the history.

Everything the agent is told goes through here, and through hygiene.check before it is sent. The prompt
never names a network, a model folder or anything of the pipeline: the agent has "the network" and "the
second network".
"""

from __future__ import annotations

from pathlib import Path

from . import rewards
from .util import REPO_ROOT

TEMPLATES = REPO_ROOT / "prompts"
ENTRY_DOC = REPO_ROOT / "agent_docs" / "ENTRY.md"


def _fmt(v, digits=4) -> str:
    return "none" if v is None else f"{v:.{digits}f}"


def history(rounds: list) -> str:
    """Earlier rounds as the agent sees them: every submission's receipt in a table, then the round's notes."""
    if not rounds:
        return "None: this is the first round."
    out = []
    best = None
    for rd in rounds:
        out.append(f"### Round {rd['round']}")
        subs = rd.get("submissions") or []
        if subs:
            out.append("")
            out.append("| saved as | status | reward | coverage | fidelity | cost | claimed | expected on the second |")
            out.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
            for s in subs:
                p = s.get("parts") or {}
                if s["status"] == "scored":
                    row = [s["carried_as"], "scored", _fmt(s["reward"]), _fmt(p.get("coverage")), _fmt(p.get("fidelity"), 5),
                           _fmt(rewards.cost_of(p)), _fmt(s.get("claimed_fidelity"), 5), _fmt(s.get("expected_fidelity_other"), 5)]
                    if best is None or s["reward"] > best[1]:
                        best = (s["carried_as"], s["reward"])
                else:
                    row = [s.get("name") or "", f"refused: {str(s.get('reason'))[:120]}", "", "", "", "", "", ""]
                out.append("| " + " | ".join(row) + " |")
        else:
            out.append("")
            out.append("No submission.")
        out.append("")
        out.append("Notes of that round:")
        out.append("")
        notes = (rd.get("notes") or "").strip() or "(none)"
        out.append("\n".join("    " + line for line in notes.splitlines()))
        out.append("")
    if best is not None:
        out.append(f"The scored entries above are in `entries` under the names in the first column, and the best so far "
                   f"(reward {_fmt(best[1])}) is also there as `best`.")
    return "\n".join(out)


RESTART_NOTE = """

## This session was started before

It was cut off, and it starts again here with the same task and what is left of its budgets:
{python_left} of {python_all} `run_python` calls and {submit_left} of {submit_all} submissions. The
saved entries are still in `entries`: {saved}.
{submitted}"""


def restart_note(session) -> str:
    """What a session that was cut off is told when it starts again: its budgets as they stand, the entries
    it saved and the submissions it already made. Empty for a session that has used nothing."""
    used, budgets = session.state["used"], session.state["budgets"]
    if not any(used.values()):
        return ""
    lines = []
    for r in session.receipts():
        p = r.get("parts") or {}
        if r["status"] == "scored":
            lines.append(f"- `{r.get('name')}`: reward {_fmt(r['reward'])}, coverage {_fmt(p.get('coverage'))}, fidelity "
                         f"{_fmt(p.get('fidelity'), 5)}, cost {_fmt(rewards.cost_of(p))}")
        else:
            lines.append(f"- `{r.get('name')}`: refused: {str(r.get('reason'))[:200]}")
    submitted = ("It already submitted:\n\n" + "\n".join(lines) + "\n") if lines else "It had submitted nothing yet.\n"
    saved = ", ".join(f"`{n}`" for n in sorted(session.entries())) or "none"
    return RESTART_NOTE.format(python_left=budgets["run_python"] - used["run_python"], python_all=budgets["run_python"],
                               submit_left=budgets["submit_entry"] - used["submit_entry"], submit_all=budgets["submit_entry"],
                               saved=saved, submitted=submitted)


SAMPLE_NOTE = """
Because the verifier scores a sample, two things follow. The fidelity the reward uses is the lower end
of its 95% confidence interval on that sample: a lucky sample does not pay, and even an entry that is
right on every sampled input is charged a little (the network itself then scores just under 0). And
the same entry submitted twice gets two slightly different rewards.
"""


# A baseline run (level1/baseline.py) asks for an entry that is exact whatever the weights; a usual run's prompt
# has nothing in this place. It goes to the agent, so it says nothing of what the networks are.
EXACT_NOTE = """

This run asks one more thing of an entry: that it be exact whatever the weights. When the rounds are
over, the entries are also run, unchanged, on several other networks of the same definition and sizes,
with other weights, and an entry counts only if it gives each network's output on every input of its
region that is checked there. So whatever depends on the weights has to be computed from the weights
inside the entry, and every shortcut has to hold for any weights: a rule that holds only for this
network makes the entry fail on the others."""


def build(cfg: dict, model, round_no: int, rounds: list, model_ops: int, enumerated: bool) -> str:
    template = (TEMPLATES / f"{cfg['prompt_template']}.md").read_text()
    sizes = ", ".join(f"{k} = {v}" for k, v in model.sizes.items()) + f", outputs: {model.config['n_outputs']}"
    inputs_note = (f"every input of the data set ({model.n_total:,} inputs)" if enumerated else
                   f"{int(cfg['verifier']['sample_size']):,} inputs drawn afresh from the data set for each submission")
    fields = {
        "ROUND": round_no, "ROUNDS_MAX": cfg["rounds_max"], "MODEL_SOURCE": model.source().strip(), "SIZES": sizes,
        "DATA_SET": model.data_set_text(), "MODEL_OPS": f"{model_ops:,}", "ENTRY_DOC": ENTRY_DOC.read_text().strip(),
        "INPUTS_NOTE": inputs_note, "SAMPLE_NOTE": "" if enumerated else SAMPLE_NOTE,
        "REWARD_FORMULA": rewards.formula(cfg["reward"]), "REWARD_EXPLANATION": rewards.explanation(cfg["reward"]),
        "PYTHON_CALLS": cfg["session"]["python_calls"], "PYTHON_SECONDS": cfg["python"]["time_limit_s"],
        "SUBMISSIONS": cfg["session"]["submissions"], "MAX_NODES": cfg["entry"]["max_nodes"],
        "MAX_CONST": cfg["entry"]["max_const_scalars"], "VERIFIER_SECONDS": f"{float(cfg['verifier']['time_limit_s']):g}",
        "HISTORY": history(rounds), "EXACT_NOTE": EXACT_NOTE if cfg.get("exact_on_any_weights") else "",
    }
    text = template
    for k, v in fields.items():
        text = text.replace("{{" + k + "}}", str(v))
    if "{{" in text:
        raise ValueError(f"the prompt template has a field nobody fills: {text[text.index('{{'):][:40]}")
    return text
