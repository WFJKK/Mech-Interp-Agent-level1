"""A round's prompt: the template, the network's definition, the entry guide, the scoring and the history.

Everything the agent is told goes through here, and through hygiene.check before it is sent. The prompt
never names a network, a model folder or anything of the pipeline: the agent has "the network" and "the
second network".
"""

from __future__ import annotations

from pathlib import Path
import copy
import json
import re
import textwrap

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


API_ATTEMPT_NOTE = """

## This attempt's API allowance

The controller has allocated ${cap_usd:.2f} for this API attempt after accounting for earlier costs and
open reservations. This is the limit passed to the CLI for this attempt; it may be less than the
session's nominal allocation. Keep the remaining work and final deliverable within this allowance.
The CLI checks spending after an answer, so it can stop without a final-turn warning and the last
answer can exceed the limit.
"""


def api_attempt_note(cap_usd: float) -> str:
    """The actual claimed allowance, also valid for a tool-free synthesis session."""
    return API_ATTEMPT_NOTE.format(cap_usd=cap_usd)


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


# A run that goes on from an earlier one with the earlier rounds in its prompt (level1/continuation.py) says what
# changed, in these paragraphs, filled from the run's config. Every other run's prompt has nothing here.
CONTINUATION_HEADING = "## These rounds go on from earlier ones"
CONTINUATION_NOTE = (
    "The earlier rounds on this network were scored with {prices}. The entry locked after round {last} "
    "({carried_as}), {measured}, has coverage {coverage:.4f}, fidelity {fidelity:.5f} and cost {cost:.4f}: it saves "
    "{saved:.4f} of the forward passes and is wrong on {errors:.6f} of all inputs. From round {first} on, the aim is "
    "to go into more detail on what the earlier rounds did not understand: the inputs on which their entries "
    "disagree with the network, why the network answers as it does there, and how to predict that cheaply.",
    "To make that count, a wrong answer is now charged at alpha = {alpha:g} forward passes. {rule} At this price "
    "the entry locked after round {last} scores {reward_now:.4f}. {exact}",
    "The rewards of the earlier rounds under \"Earlier rounds\" are recomputed at this price from what was "
    "measured then. Their notes were written at the earlier prices, so a fix they judged not worth its cost may be "
    "worth it now. Their scored entries are in `entries` as before.",
)
CONTINUATION_RULE = ("That is {margin:g} times what makes removing all of that entry's errors worth it even if "
                     "that cost {allowed:.4f} of its saving{whole}, rounded up and never below the earlier price: "
                     "{margin:g} x {allowed:.4f} / {errors:.6f}.")
CONTINUATION_FIXED = "That price is set for these rounds."
CONTINUATION_SAMPLED = ("An entry right on every input of its sample is charged only {charge:.3f} at full coverage "
                        "for the inputs not drawn, so among such entries the saving decides. To keep that charge "
                        "small, each submission is now scored on {sample:,} inputs drawn afresh ({sample_before:,} "
                        "before).")
CONTINUATION_SAMPLED_SAME = ("An entry right on every input of its sample is charged only {charge:.3f} at full coverage "
                             "for the inputs not drawn, so among such entries the saving decides. Each submission is "
                             "scored on {sample:,} inputs drawn afresh, as in round {last}.")
CONTINUATION_EVERY = "An entry right on every input loses nothing to errors, so among such entries the saving decides."
CONTINUATION_LIMITS = ("Entries may now have up to {nodes} nodes and {consts} numbers in their constants ({nodes_before} "
                       "and {consts_before} in the earlier rounds), so an idea that did not fit then may fit now.")


def continuation_texts() -> str:
    """The continuation note's parts as written, for the hygiene check of the agent's surface."""
    return "\n\n".join((CONTINUATION_HEADING, *CONTINUATION_NOTE, CONTINUATION_RULE, CONTINUATION_FIXED,
                         CONTINUATION_SAMPLED, CONTINUATION_SAMPLED_SAME, CONTINUATION_EVERY, CONTINUATION_LIMITS))


def continuation_note(cfg: dict) -> str:
    """What a run that goes on from earlier rounds tells its agent; empty for any other run."""
    from . import continuation

    d = continuation.derived(cfg)
    if not d or not d["runs"]:
        return ""
    o, lk = continuation.options(cfg), d["locked"]
    c_max = float(o["c_max"])
    allowed = c_max - lk["cost_over_all_inputs"]
    rule = (CONTINUATION_RULE.format(margin=float(o["margin"]), allowed=allowed, whole=" (all of it)" if c_max == 1 else "",
                                     errors=lk["errors"])
            if o.get("alpha") is None else CONTINUATION_FIXED)
    if not d["sampled"]:
        exact = CONTINUATION_EVERY
    elif d["sample"] == d["sample_before"]:
        exact = CONTINUATION_SAMPLED_SAME.format(charge=d["exact_charge"], sample=d["sample"], last=d["first_round"] - 1)
    else:
        exact = CONTINUATION_SAMPLED.format(charge=d["exact_charge"], sample=d["sample"], sample_before=d["sample_before"])
    fields = {"prices": continuation.price_history(d), "last": d["first_round"] - 1, "first": d["first_round"],
              "carried_as": lk["carried_as"], "measured": "measured again on fresh inputs" if lk["measured_again"] else "on every input",
              "coverage": lk["coverage"], "fidelity": lk["fidelity"], "cost": lk["cost"], "saved": lk["saved"],
              "errors": lk["errors"], "alpha": d["alpha"], "rule": rule, "reward_now": lk["reward_now"], "exact": exact}
    notes = list(CONTINUATION_NOTE)
    before = d.get("limits_before") or {}
    now = {k: cfg["entry"][k] for k in ("max_nodes", "max_const_scalars")}
    if all(before.get(k) is not None for k in now) and before != now:  # the earlier rounds' entries had other room
        notes.insert(2, CONTINUATION_LIMITS)
        fields.update(nodes=now["max_nodes"], consts=now["max_const_scalars"], nodes_before=before["max_nodes"],
                      consts_before=before["max_const_scalars"])
    paragraphs = [textwrap.fill(p.format(**fields), width=100, break_long_words=False, break_on_hyphens=False)
                  for p in notes]
    return "\n\n" + "\n\n".join([CONTINUATION_HEADING, *paragraphs])


def build(cfg: dict, model, round_no: int, rounds: list, model_ops: int, enumerated: bool) -> str:
    if cfg["verifier"].get("scope") is not None:
        return build_multi(cfg, model, round_no - 1, "regional", "Improve the interpretation on the assigned input domain.",
                           "", history(rounds))
    template = (TEMPLATES / f"{cfg['prompt_template']}.md").read_text()
    sizes = ", ".join(f"{k} = {v}" for k, v in model.sizes.items()) + f", outputs: {model.config['n_outputs']}"
    inputs_note = (f"every input of the data set ({model.n_total:,} inputs)" if enumerated else
                   f"{int(cfg['verifier']['sample_size']):,} inputs drawn afresh from the data set for each submission")
    from .continuation import first_round

    fields = {
        "ROUND": round_no, "ROUNDS_MAX": first_round(cfg) - 1 + int(cfg["rounds_max"]), "MODEL_SOURCE": model.source().strip(),
        "SIZES": sizes,
        "DATA_SET": model.data_set_text(), "MODEL_OPS": f"{model_ops:,}", "ENTRY_DOC": ENTRY_DOC.read_text().strip(),
        "INPUTS_NOTE": inputs_note, "SAMPLE_NOTE": "" if enumerated else SAMPLE_NOTE,
        "REWARD_FORMULA": rewards.formula(cfg["reward"]), "REWARD_EXPLANATION": rewards.explanation(cfg["reward"]),
        "PYTHON_CALLS": cfg["session"]["python_calls"], "PYTHON_SECONDS": cfg["python"]["time_limit_s"],
        "SUBMISSIONS": cfg["session"]["submissions"], "MAX_NODES": cfg["entry"]["max_nodes"],
        "MAX_CONST": cfg["entry"]["max_const_scalars"], "VERIFIER_SECONDS": f"{float(cfg['verifier']['time_limit_s']):g}",
        "HISTORY": history(rounds), "EXACT_NOTE": EXACT_NOTE if cfg.get("exact_on_any_weights") else "",
        "CONTINUATION_NOTE": continuation_note(cfg),
    }
    text = template
    for k, v in fields.items():
        text = text.replace("{{" + k + "}}", str(v))
    if "{{" in text:
        raise ValueError(f"the prompt template has a field nobody fills: {text[text.index('{{'):][:40]}")
    return text


RESEARCH_ROLES = {
    "comparison": (
        "You are the single investigator for a global comparison program. Find an inexpensive executable "
        "program that predicts the network over the entire assigned input domain. Every submitted "
        "candidate must cover that whole domain and compute the network's output for any weights of "
        "this definition and these sizes. Quantities that depend on weights must be computed from the "
        "supplied weights inside the program. Optimize the ordinary reward subject to these requirements. "
        "Submit a valid complete program early, then improve it within the remaining allowance. "
        "Explain the computational savings and any numerical limitations. This session does not "
        "schedule other investigators or submit narrower domain claims."),
    "lead": (
        "You are the research lead. In each research phase you decide what each researcher works on and give "
        "out the tasks with assign_task; after the phase you see their checked results and notes and combine "
        "them into one program for the whole network, evaluated on the original input distribution. Your aim "
        "is the highest reward for that program. A task can concern anything you judge useful: some of the "
        "network's units or another part of its computation, a group of inputs, making a saved program "
        "cheaper, or testing a claim. Researchers are scored by the same reward on their assigned domain; a "
        "program that computes part of the network its own way and the rest as the network does is an "
        "ordinary entry and is scored like any other. Seek an overall mechanistic understanding of the "
        "network: connect the researchers' findings, investigate shared mechanisms, and explain how the parts "
        "of the computation produce the network's behavior. Preserve the best global interpretation even "
        "when the mechanistic understanding remains incomplete. Mechanistic explanation is a research goal, "
        "not an additional submission, selection or retention requirement. Distinguish supported "
        "explanations, hypotheses and unresolved questions. Submit global candidates for evaluation; a score "
        "on a narrower domain is never a substitute for measuring the composed program. Preserve useful "
        "findings even when their integration does not improve the global score. Use the researchers' checked "
        "results to identify useful combinations, shared computations, counterexamples and missing tasks. "
        "Investigate simple explanations on subdomains where measurements show strong agreement, and examine "
        "their failure cases separately. Tasks should use the configured fidelity and cost tradeoff; require "
        "exact agreement only when it is an explicit, bounded hypothesis or a requirement of this task."),
    "researcher": (
        "You are a researcher on a team led by a research lead. Carry out the task the lead gave you, shown "
        "below, and optimize the ordinary reward on the assigned domain. Every program you submit is an "
        "ordinary entry scored by that reward; if your task concerns part of the network, a program that "
        "computes that part its own way and the rest as the network does is such an entry. Seek a "
        "mechanistic explanation of why an interpretation works, relating the observed behavior to the "
        "network's weights and internal computations. Retain and submit strong interpretations even when "
        "their mechanism remains unclear; do not delay a required submission to finish an explanation. "
        "Mechanistic explanation is a research goal, not an additional submission, selection or retention "
        "requirement. Distinguish supported explanations, hypotheses and unresolved questions. Test "
        "competing explanations, and refine an entry's region when needed. Return executable candidates, "
        "the evidence for their behavior, failure cases and ideas that the lead can reuse. You can submit "
        "directly to the evaluator; your results are retained independently of the global score. When a "
        "simpler explanation works on a narrower domain, submit it with an explicit scope so its evidence "
        "is retained separately. Use the remaining submission allowance for a broader candidate or a "
        "competing explanation."),
}

ASSIGNMENT_NOTE = """
You may give out up to {count} tasks for the next research phase with
`assign_task(task, scope=None, entry_name=None)`, one per researcher. Giving out fewer runs fewer
researchers, each with more of the allowance. You decide what each researcher does, for example
some of the network's units or another part of its computation, a group of inputs, making a
saved program cheaper, or testing a claim. Give each a concrete question, the relevant evidence and
the output you want back. Without `scope`, a researcher's submissions are scored on the whole input
distribution. To score them on a narrower domain, use scope kind `lengths` with a `lengths` list, or
`predicate` with a saved entry whose region test defines the domain, passing its name as `entry_name`.
The scope is fixed before the researcher begins. This tool cannot change the reward, operation prices
or workload policy. When giving out more than one task, leave room for an alternative explanation
instead of making every researcher follow the same hypothesis.
"""

# A lead after a cycle may also end the research (session.finish_research); the planning lead may not.
FINISH_NOTE = """
If you judge that further research is unlikely to raise the reward, call `finish_research(reason)`
instead of giving out tasks. The run then goes straight to its final checks and report: the best
program scored on the whole input distribution is locked, as after the last cycle. A program for the
whole network must have been scored first, by you or earlier in the run, and finishing cannot be
combined with tasks given out in this session.
"""


def research_allocation(cfg: dict) -> str:
    """Describe the configured session quota without treating a login as dollar-capped."""
    session = cfg["session"]
    if cfg.get("auth") == "api_key":
        api = (f"The nominal API allowance for this session is ${float(cfg['budget']['round_usd']):.2f}. "
               "It shares the experiment's budget with other sessions. The actual allowance for each "
               "attempt is supplied with that request and may be below the nominal amount.")
    else:
        api = "This session does not enforce an API dollar cap; its tool, turn and time allocations still apply."
    return (api + f" Each backend attempt has at most {int(session['max_turns'])} turns and "
            f"{float(session['seconds']):g} seconds. Retries do not refresh the tool or submission quotas. "
            "Tool replies report wall_seconds_remaining for the current attempt. This is a deadline, "
            "not a live dollar balance; reported API cost is reconciled when the attempt ends. "
            "Complete a useful deliverable early; "
            "a budget limit can end the session before closing notes are written.")


def research_deliverable(cfg: dict) -> str:
    """Saving a draft and submitting a scored candidate are distinct operations."""
    saving = ("Save a usable executable entry with `save_entry(name, entry)` before extended optimization. "
              "Use `reference_entry()` as a clearly labeled starting point if needed. A program described "
              "only in prose or printed to the console is not a saved artifact.")
    submissions = int(cfg["session"]["submissions"])
    research = cfg.get("research_task") or {}
    checkpoint = research.get("checkpoint_python_calls")
    if checkpoint is not None:
        saving += (f" This session permits {int(checkpoint)} initial run_python calls. "
                   "Before further Python exploration, the controller requires a completed scored "
                   "submit_entry receipt on the assigned domain. A promising carried draft can be used "
                   "for this checkpoint, but the controller does not choose or submit one for you. ")
        if research.get("role") == "baseline":
            saving += ("This checkpoint must cover every evaluated input and agree with every network output "
                       "there. An incorrect, partial-coverage or refused submission permits one additional "
                       "repair call while a submission slot remains. ")
        else:
            saving += ("A refused same-domain submission permits one additional repair call while a submission slot "
                       "remains. ")
        saving += ("If the initial window produced no saved entry, one bounded construction "
                   "repair call is available within the original Python quota. Child-domain claims do "
                   "not satisfy the checkpoint, and the last submission "
                   "slot is reserved for it until it succeeds. Check remaining.checkpoint in tool replies. "
                   "This is an operational checkpoint; the ordinary score determines the program's quality.")
    if submissions > 0:
        quota = ("You have one formal submission; keep a starting reference as a draft and spend the "
                 "submission on a promising candidate." if submissions == 1 else
                 f"You have {submissions} formal submissions; account for that limit when deciding which candidates to submit.")
        return (saving + " " + quota + " Submit a promising candidate with `submit_entry` while enough "
                "API, time and tool allowance remains to finish reliably. A promising carried draft can "
                "be a useful first candidate. If the submission allowance is exhausted, save further "
                "improvements for a later session. Saving or locally measuring an entry does not submit "
                "it and does not give it a formal score.")
    return (saving + " This session has no formal submission allowance. Hand off saved drafts, concise "
            "experimental results and concrete assignments so a later investigator can test and submit them.")


def research_domain_notes(role: str, submissions: int) -> dict:
    """Describe only the domain controls exposed to this session's role."""
    if role in ("baseline", "comparison"):
        return {
            "CLAIM_NOTE": ("This session evaluates the complete assigned domain. Submit a program with full coverage; "
                           "the task cannot be replaced by a narrower evaluation domain. An entry's region test "
                           "still contributes to its measured coverage and cost."),
            "DISTRIBUTION_NOTE": "The assigned domain is the complete original input distribution for this session.",
            "COST_DOMAIN_NOTE": ("The runtime includes the entry's region test and prediction. The complete program "
                                 "must pay for any routing and shared setup. Setup is charged once over the declared "
                                 "workload. Each receipt reports that workload separately from the verification count. "
                                 "Changing a program does not change the workload policy or operation prices."),
            "FINAL_CHECK_NOTE": ("After this session, the submitted programs are checked unchanged on other parameter "
                                 "settings of the same definition and sizes. Those checks select one frozen comparison "
                                 "program and are not returned to this session. Numerical agreement is measured evidence; "
                                 "state separately any mathematical justification and numerical limitations."),
            "LOCAL_MEASUREMENT_NOTE": ("Use `network_outputs()` and `measure(entry)` on the complete assigned domain.\n"
                                       "  These local measurements spend no formal submission and cannot satisfy the required checkpoint."),
            "SUBMISSION_TOOL_NOTE": (
                "- `submit_entry(name, claimed_fidelity, expected_fidelity_other=None)`:\n"
                f"  up to {submissions} submissions in total, including the checkpoint. Each submission is evaluated\n"
                "  on the complete assigned domain; the program must cover it and reproduce the network's output."),
            "SHARED_RECORD_NOTE": ("These allocations are a share of one experiment budget. This independent session "
                                   "has its own workspace and saved programs. Its formal submissions are the record "
                                   "used for the subsequent checks and selection of one frozen comparison program."),
        }
    return {
        "CLAIM_NOTE": """You can declare a narrower evaluation domain with `submit_entry(..., scope=...)`. The evaluator
intersects it with the assignment, freezes that domain, and retains an independent conditional score
and its probability in the original distribution. Each claim consumes one submission. An entry's
region test alone does not declare such a separate task; it still determines coverage inside the
domain being evaluated. Neither operation changes the reward or operation prices.""",
        "DISTRIBUTION_NOTE": """When the assigned domain is narrower than all inputs, its distribution is this distribution
conditioned on that domain. The original distribution remains the task of the lead's program for the
whole network.""",
        "COST_DOMAIN_NOTE": """The runtime includes the entry's region test on every task input and its prediction on covered inputs.
Setup is charged once over the declared workload. Dividing the whole cost by conditional coverage gives
the cost per covered input. The membership of an externally assigned domain is known for this
task. A global composition must implement and pay for its own routing and any shared setup.
For a narrower claim, the configured workload policy is unchanged: an explicitly fixed workload stays
fixed; otherwise the claim's known finite support sets its workload. Unknown support requires a workload
already declared by the run. Each receipt reports this value separately from the verification count.""",
        "FINAL_CHECK_NOTE": """Final candidates are locked before evaluation on fresh inputs and the second network. Those results
are not returned to researchers and do not select the development winners. A successful claim on a
narrower domain is retained even if it does not improve the global program. Numerical agreement is measured evidence;
state separately any derivation from weights, counterexamples, or additional mechanistic evidence.""",
        "LOCAL_MEASUREMENT_NOTE": """The last two use the assigned domain by default; optional `scope=...`
  tests a narrower claim under the same policy. These local measurements spend no formal submission and
  cannot satisfy the integration checkpoint.""",
        "SUBMISSION_TOOL_NOTE": (
            "- `submit_entry(name, claimed_fidelity, expected_fidelity_other=None, scope=None, scope_entry_name=None)`:\n"
            f"  up to {submissions} submissions in total, including checkpoints and all narrower claims. `scope` can\n"
            "  have kind `all`, `lengths`, `predicate`, or `intersection` with a list named `scopes`. For a predicate,\n"
            "  pass a saved entry defining its region test as `scope_entry_name`, or include that entry in the scope."),
        "SHARED_RECORD_NOTE": """These allocations are a share of one experiment budget. Saved entries and the checked archive are the
shared record; conversations are separate. A comparison entry, when provided, is available in `entries`
as `comparison_entry` and is evaluated under the same conditions as other entries on this domain.""",
    }


def build_multi(cfg: dict, model, session_index: int, role: str, task: str,
                archive_summary, history) -> str:
    """The research prompt uses the same language and reward, with an explicit conditional domain."""
    import numpy as np

    from . import lang
    from .verifier import Verifier

    public_role = {"baseline": "comparison", "regional": "researcher"}.get(role, role)
    if public_role not in RESEARCH_ROLES:
        raise ValueError(f"unknown research role: {role}")
    verifier = Verifier(model, cfg["network"], cfg)
    example = lang.draw_inputs(model.data_set, model.input_space, 1, np.random.default_rng(0))
    _, model_ops = lang.run_reference(verifier.reference, verifier.weights, model.n_ctx, example)
    scope = cfg["verifier"].get("scope") or {"kind": "all"}
    scope_meta = verifier.scope.metadata() if getattr(verifier, "scope", None) is not None else {}
    workload = cfg["verifier"].get("workload_size") or scope_meta.get("cardinality")
    if workload is None and getattr(verifier, "scope", None) is not None:
        workload = verifier.scope.cardinality
    if workload is None:
        raise ValueError("a scoped research session needs a known finite support size or an explicit workload_size")
    fixed = cfg["verifier"].get("sample_seed") is not None
    inputs_note = (f"Every input of this task is checked ({verifier.n_inputs():,} inputs)." if verifier.enumerated else
                   f"Development uses {'a fixed' if fixed else 'a fresh'} sample of {verifier.n_inputs():,} inputs "
                   "from the conditional distribution.")
    sample_note = ("The exact fidelity is used when every task input is checked." if verifier.enumerated else
                   "The reward uses the lower end of the fidelity's 95% confidence interval. " +
                   ("Repeating an identical entry uses the same development inputs and does not draw a luckier score. " if fixed else
                    "Submissions receive fresh development samples. ") +
                   "Adaptive search can fit development evidence, so final locked evaluation is reported separately.")
    research = cfg.get("research_task") or {}
    count = int(research.get("max_assignments", 0))
    if isinstance(history, str):
        notes = history
    else:
        notes = "\n\n".join(str(item.get("notes", "")) if isinstance(item, dict) else str(item) for item in (history or []))
    if not isinstance(archive_summary, str):
        archive_summary = json.dumps(archive_summary, indent=2, sort_keys=True)
    fields = {
        "ROLE": public_role, "ROLE_INSTRUCTIONS": RESEARCH_ROLES[public_role], "TASK": task,
        "SCOPE": json.dumps(scope, indent=2, sort_keys=True), "MODEL_SOURCE": model.source().strip(),
        "SIZES": ", ".join(f"{k} = {v}" for k, v in model.sizes.items()) + f", outputs: {model.config['n_outputs']}",
        "MODEL_OPS": f"{model_ops:,}", "DATA_SET": model.data_set_text(), "ENTRY_DOC": ENTRY_DOC.read_text().strip(),
        "REWARD_FORMULA": rewards.formula(cfg["reward"]), "REWARD_EXPLANATION": rewards.explanation(cfg["reward"]),
        "INPUTS_NOTE": inputs_note,
        "WORKLOAD_NOTE": f"Setup is amortized over a declared workload of {int(workload):,} inputs at the task's distribution. "
                         "This is separate from the number of verification inputs. For an exhaustively checked uniform "
                         "finite domain, the default workload is one evaluation of each input.",
        "SAMPLE_NOTE": sample_note, "EXACT_NOTE": EXACT_NOTE if cfg.get("exact_on_any_weights") else "",
        "PYTHON_CALLS": cfg["session"]["python_calls"], "PYTHON_SECONDS": cfg["python"]["time_limit_s"],
        "SUBMISSIONS": cfg["session"]["submissions"], "MAX_NODES": cfg["entry"]["max_nodes"],
        "MAX_CONST": cfg["entry"]["max_const_scalars"], "VERIFIER_SECONDS": f"{float(cfg['verifier']['time_limit_s']):g}",
        "API_ALLOCATION": research_allocation(cfg), "DELIVERABLE_NOTE": research_deliverable(cfg),
        "ASSIGNMENT_NOTE": (ASSIGNMENT_NOTE.format(count=count) + (FINISH_NOTE if int(research.get("cycle", 0)) >= 1 else "")
                            if role == "lead" and count else "No further tasks can be given out in this session."),
        "ARCHIVE": archive_summary or "No checked submissions yet.", "HISTORY": notes or "No earlier research notes.",
        **research_domain_notes(role, int(cfg["session"]["submissions"])),
    }
    text = (TEMPLATES / "research_v1.md").read_text()
    for key, value in fields.items():
        text = text.replace("{{" + key + "}}", str(value))
    if "{{" in text:
        raise ValueError("the research prompt has an unfilled template field")
    return text


def summary_evidence(evidence: dict, *, include_discoveries: bool = True) -> dict:
    """An agent-facing projection; raw provenance remains in the saved evidence.

    The masked form is for hygiene inspection of the complete coordinator-owned
    surface. Only researcher-authored programs, predicate definitions and notes
    are masked. An independently generated comparison program is also a
    researcher discovery; an operator-supplied comparison is checked in full.
    """
    if not isinstance(evidence, dict):
        raise ValueError("summary evidence must be an object")
    if not evidence:
        return {}

    def pick(value, keys):
        return {key: copy.deepcopy(value[key]) for key in keys if key in value} if isinstance(value, dict) else {}

    def discovery(value):
        return copy.deepcopy(value) if include_discoveries else "Researcher-authored content omitted for hygiene inspection."

    def scope(value):
        if not isinstance(value, dict):
            return copy.deepcopy(value)
        out = pick(value, ("kind", "lengths", "max_draws", "mass_samples", "id", "cardinality", "mass", "mass_exact",
                           "conditioning", "mass_estimation", "predicate_binding", "workload_size", "workload_basis",
                           "build_measurement"))
        if "entry" in value:
            out["entry"] = discovery(value["entry"])
        if "spec" in value:
            out["spec"] = scope(value["spec"])
        return out

    def receipt(value, network):
        if not isinstance(value, dict):
            return copy.deepcopy(value)
        out = pick(value, ("status", "reason", "entry_sha256", "size", "parts", "reward", "prices", "reward_detail",
                           "claimed_fidelity", "overclaim", "expected_fidelity_other"))
        out["network"] = network
        if "inputs" in value:
            out["inputs"] = pick(value["inputs"], ("mode", "n", "seed", "scope_id"))
        if "scope" in value:
            out["scope"] = scope(value["scope"])
        return out

    def audits(value):
        value = value or {}
        out = pick(value, ("candidate_id", "entry_sha256"))
        if "scope" in value:
            out["scope"] = scope(value["scope"])
        for label, network in (("fresh", "the network"), ("held_out", "the second network")):
            if label in value:
                out[label] = receipt(value[label], network)
            comparison = value.get(f"{label}_comparison", value.get(f"{label}_baseline"))
            if comparison is not None:
                out[f"{label}_comparison"] = receipt(comparison, network)
        return out

    out = pick(evidence, ("version", "reward", "prices"))
    out["network"] = "the network"
    out["second_network"] = "the second network" if evidence.get("second_network") else None
    out["evaluation_policy"] = pick(evidence.get("evaluation_policy"),
                                    ("check", "enumerate_up_to", "sample_size", "sample_seed", "workload_size"))
    out["frozen_selection"] = pick(evidence.get("frozen_selection"), ("global", "regional", "locked_at"))
    out["candidates"] = []
    for candidate in evidence.get("candidates", []):
        item = pick(candidate, ("id", "role", "scope_id", "entry_sha256", "expected_fidelity_other"))
        if "scope" in candidate:
            item["scope"] = scope(candidate["scope"])
        if "program" in candidate:
            item["program"] = discovery(candidate["program"])
        if "development" in candidate:
            item["development"] = receipt(candidate["development"], "the network")
        if "final_audits" in candidate:
            item["final_audits"] = audits(candidate["final_audits"])
        if "comparisons" in candidate:
            item["comparisons"] = {
                label: pick(check, ("valid", "reason", "reward_gap", "raw_reward_difference"))
                for label, check in candidate["comparisons"].items()}
        out["candidates"].append(item)
    supplied = evidence.get("comparison") or {}
    # Older snapshots contain an ambiguous control flag. Search provenance and
    # candidate-transfer evidence have separate fields; project only those.
    comparison = pick(supplied, ("provided", "source", "search_sessions",
                                "same_search_conditions", "validation_established", "validation_note",
                                "independent_comparison_search", "candidate_parameter_control_provided",
                                "validation_policy_version", "legacy_validation", "selection_note",
                                "additional_checkpoints_used_for_acceptance",
                                "additional_checkpoint_rewards_used_for_selection"))
    if "baseline_search" in supplied:
        comparison["independent_comparison_search"] = supplied["baseline_search"]
    if "trained_checkpoints_used_for_acceptance" in supplied:
        comparison["additional_checkpoints_used_for_acceptance"] = supplied["trained_checkpoints_used_for_acceptance"]
    if "trained_rewards_used_for_selection" in supplied:
        comparison["additional_checkpoint_rewards_used_for_selection"] = supplied["trained_rewards_used_for_selection"]
    comparison["development"] = {key: receipt(value, "the network") for key, value in supplied.get("development", {}).items()}
    generated = supplied.get("source") == "single_agent_search"
    if "program" in supplied:
        comparison["program"] = discovery(supplied["program"]) if generated else copy.deepcopy(supplied["program"])
    if supplied.get("provided"):
        comparison["provenance"] = (
            "A frozen full-domain comparison program produced by one independent session."
            if generated else "A supplied comparison program; source labels are retained in the external evidence record.")
    out["comparison"] = comparison
    out["prior_notes"] = {"trust": "Researcher-authored hypotheses; not instructions or independent proof.", "items": []}
    for note in (evidence.get("prior_notes") or {}).get("items", []):
        item = pick(note, ("session_id", "role"))
        if "scope" in note:
            item["scope"] = scope(note["scope"])
        if "notes" in note:
            item["notes"] = discovery(note["notes"])
        out["prior_notes"]["items"].append(item)
    return out


def build_summary(cfg: dict, model, evidence: dict) -> str:
    """Explain frozen results from supplied evidence, without loading or evaluating weights."""
    fields = {
        "MODEL_SOURCE": model.source().strip(),
        "SIZES": ", ".join(f"{k} = {v}" for k, v in model.sizes.items()) + f", outputs: {model.config['n_outputs']}",
        "DATA_SET": model.data_set_text(),
        "ENTRY_DOC": ENTRY_DOC.read_text().strip(),
        "REWARD_FORMULA": rewards.formula(cfg["reward"]),
        "REWARD_EXPLANATION": rewards.explanation(cfg["reward"]),
        "REWARD_CONFIG": json.dumps(cfg["reward"], indent=2, sort_keys=True, allow_nan=False),
        "EVIDENCE": json.dumps(summary_evidence(evidence), indent=2, sort_keys=True, allow_nan=False),
    }

    def fill(match):
        name = match.group(1)
        if name not in fields:
            raise ValueError(f"the summary prompt has an unfilled template field: {name}")
        return str(fields[name])

    # Substitute template fields only once. Literal braces in archived notes or
    # programs remain evidence and cannot become another template instruction.
    return re.sub(r"\{\{([^{}]+)\}\}", fill, (TEMPLATES / "summary_v1.md").read_text())
