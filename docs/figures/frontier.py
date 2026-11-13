"""The README's picture of the pilots' Pareto frontiers.

    python docs/figures/frontier.py [RUNS FOLDER]     (default ~/mech-interp-level1-runs; needs matplotlib)

The frontier of a run is over three numbers per submission: the share of the inputs answered rightly,
the share answered wrongly and the cost. Every submission of these runs covers the whole data set, so
its right share is one less its wrong share, and the frontier is a frontier in two: cost against wrong
answers. One panel per run shows every scored submission on the run's own network, the ones the report
marks as on the frontier, the locked entry, the line of equal reward through it, and the frozen
baseline entry scored on the same network. It reads each run's report.json and baseline_scores.jsonl
and writes frontier.png next to this file.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

RUNS = [("pilot-lin-a-1to4", "lin-a-1to4", "linear, 4 slots"), ("pilot-lin-c", "lin-c", "linear, 4 slots"),
        ("pilot-lin-b", "lin-b", "linear, 8 slots"), ("pilot-lin-d", "lin-d", "linear, 8 slots"),
        ("pilot-att-c", "att-c", "transformer, 4 slots"), ("pilot-att-d", "att-d", "transformer, 4 slots")]
ALPHA = 10.0  # the runs' reward, r_alpha
PER = 10_000  # wrong answers are shown per this many inputs
# the light reference palette of the dataviz guide: two series, the muted grey, chrome and ink
BLUE, ORANGE, GREY = "#2a78d6", "#eb6834", "#b9b8b0"
SURFACE, GRID, AXIS, INK, INK2 = "#fcfcfb", "#e1e0d9", "#c3c2b7", "#0b0b0b", "#52514e"
OUT = Path(__file__).resolve().parent / "frontier.png"


def load(folder: Path, run: str) -> tuple:
    """(the run's scored submissions, the locked one's name, the frozen baseline entry's receipt on the run's
    own network or None)."""
    report = json.loads((folder / run / "report.json").read_text())
    subs = [s for s in report["submissions"] if s.get("status") == "scored"]
    for s in subs:
        if abs(s["parts"]["coverage"] - 1.0) > 1e-12:
            raise SystemExit(f"{run} {s['carried_as']} covers less than the data set: two axes are not enough")
    baseline = report.get("baseline") or {}
    first, frozen = (baseline.get("networks") or {}).get("first"), baseline.get("baseline_run")
    base = None
    scores = folder / run / "baseline_scores.jsonl"
    if scores.exists() and first:
        for line in scores.read_text().splitlines():
            e = json.loads(line)
            if e.get("network") == first and e.get("baseline_run") == frozen and e.get("status") == "scored":
                base = e
    return subs, report["lock"]["carried_as"], base


def panel(ax, subs: list, locked: str, base: dict | None, title: str) -> None:
    pts = [(s["whole_cost"], s["wrong_share"] * PER, bool(s.get("on_frontier")), s["carried_as"] == locked) for s in subs]
    bc = bw = None
    if base:
        p = base["parts"]
        bc, bw = p["cost"], p["coverage"] * (1.0 - p["fidelity"]) * PER
    xmax = 1.1 * max([c for c, *_ in pts] + ([bc] if bc is not None else []))
    ytop = max(1.0, 1.18 * max(w for _, w, *_ in pts))
    ax.set_xlim(0, xmax)
    ax.set_ylim(-0.07 * ytop, ytop)

    lc, lw = next((c, w) for c, w, _, is_locked in pts if is_locked)
    xs = [0, xmax]  # equal reward through the locked entry: one wrong answer weighs alpha forward passes
    ax.plot(xs, [lw + PER * (lc - x) / ALPHA for x in xs], color=INK2, lw=1, alpha=0.55, zorder=1)

    rest = [(c, w) for c, w, on, _ in pts if not on]
    ax.scatter([c for c, _ in rest], [w for _, w in rest], s=34, color=GREY, edgecolors=SURFACE, linewidths=1.5, zorder=3)
    front = sorted((c, w) for c, w, on, _ in pts if on)
    if len(front) > 1:
        ax.step([c for c, _ in front], [w for _, w in front], where="post", color=BLUE, lw=2, zorder=2)
    ax.scatter([c for c, _ in front], [w for _, w in front], s=46, color=BLUE, edgecolors=SURFACE, linewidths=1.5, zorder=4)
    # the two labels go on the sides facing away from each other when the points are close
    near_right = bc is not None and bc > lc and bc - lc < 0.15 * xmax
    ax.annotate("locked", (lc, lw), xytext=(-7, 7) if near_right else (7, 7), textcoords="offset points",
                ha="right" if near_right else "left", fontsize=8, color=INK2, zorder=5)
    if bc is not None:
        ax.scatter([bc], [bw], s=46, marker="D", color=ORANGE, edgecolors=SURFACE, linewidths=1.5, zorder=4)
        ax.annotate("baseline", (bc, bw), xytext=(7, 7) if near_right else (-7, 7), textcoords="offset points",
                    ha="left" if near_right else "right", fontsize=8, color=INK2, zorder=5)

    ax.set_title(title, loc="left", fontsize=9.5, color=INK, pad=6)
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=INK2, labelsize=8, length=0)


def main(argv=None) -> int:
    args = sys.argv[1:] if argv is None else argv
    folder = Path(args[0]).expanduser() if args else Path("~/mech-interp-level1-runs").expanduser()
    fig, axes = plt.subplots(3, 2, figsize=(8.2, 9.4), dpi=200, facecolor=SURFACE)
    for ax, (run, name, design) in zip(axes.flat, RUNS):
        subs, locked, base = load(folder, run)
        panel(ax, subs, locked, base, f"{name}   {design}")
    handles = [Line2D([], [], marker="o", ls="", color=GREY, markeredgecolor=SURFACE, markersize=7, label="a submission"),
               Line2D([], [], marker="o", color=BLUE, markeredgecolor=SURFACE, markersize=7, lw=2, label="on the Pareto frontier"),
               Line2D([], [], marker="D", ls="", color=ORANGE, markeredgecolor=SURFACE, markersize=7,
                      label="the frozen baseline entry, same network"),
               Line2D([], [], color=INK2, alpha=0.55, lw=1, label="equal reward through the locked entry (alpha = 10)")]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False, fontsize=8.5, labelcolor=INK2,
               bbox_to_anchor=(0.5, 0.995))
    fig.supxlabel("cost: operations per input, in forward passes", fontsize=9, color=INK2)
    fig.supylabel("wrong answers per 10,000 inputs", fontsize=9, color=INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.94), h_pad=1.6, w_pad=1.4)
    fig.savefig(OUT, facecolor=SURFACE)
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
