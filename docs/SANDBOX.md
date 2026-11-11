# The sandbox

The only code of the agent's that ever runs is what it sends to `run_python`. An entry is data and
is interpreted by the verifier; it never runs as code. So the sandbox has one job: run the agent's
exploring code and keep it away from everything it must not see or touch.

What it must not see: the withheld folder (what the network was trained to do, the reference
entries), the second network's weights, the repository, the run folder. What it must not do: reach
the network, write outside its own folder.

## Modes

| Mode | Needs | What it does |
| --- | --- | --- |
| `container` | Docker or Podman, and the image | No network, a read-only root, only the sandbox folder mounted, runs as nobody with no capabilities, memory, CPU and process limits |
| `landlock` | Linux with Landlock (x86_64 or aarch64) | Before the code is imported the process gives up its rights for good: no sockets, reading only Python, system libraries and the sandbox folder, writing only the folder's `out/` and `tmp/` |
| `process` | nothing | A separate process with an empty environment, a fresh folder and resource limits. Weak: the code runs as you and can read whatever you can |

The run config's `sandbox` picks one: a mode by name, `auto` for the strongest that works here
(falling back to `process`), or `hardened` for the strongest of the first two, refusing to start if
neither works. `--sandbox hardened` on the run command overrides the config.

    python -m level1.sandbox info      # which modes work on this machine
    python -m level1.sandbox build     # build the container image (Docker or Podman must be running)
    python -m level1.sandbox probe --model models/lin-a

## On a Mac

Landlock is a Linux feature, so a Mac has two choices:

- **Docker Desktop (or Podman) running, and the image built once** with
  `python -m level1.sandbox build`. Then `auto` picks container mode.
- **Nothing**: `auto` falls back to process mode.

In process mode the rule that the agent is not told what the model does rests on the agent not
looking: its code could open `withheld/lin-a/task.yaml`. The run says so when it starts, `run.json`
records the mode, and the report marks it. A process-mode run is fine for trying the pipeline out.
For a run that is meant to count, use container or Landlock mode. Every program the agent ran is
kept in full in the session folder (`round_k/session/python/`), so what a process-mode run actually
read can be checked afterwards. The report helps with that: under "What the agent ran" it lists
the programs that reach for files or the system (`open(`, `os.`, `pathlib`, `subprocess`, sockets,
the withheld folder's name). A match is a reason to read the program, not a verdict, and a program
that reads a file in another way is not listed.

## The probe

Every run starts by trying real escapes in the chosen mode, and judges them by evidence on the
outside (bytes that match the real file, a marker file that appears, a token that reaches a local
listener):

- read the verifier's code, the second network's weights, the withheld task file, the run folder,
  the driving process's environment;
- write into the repository, the model folder, the run folder, the home folder;
- open a TCP and a UDP connection;
- import the verifier.

A hardened mode that lets any of these out is refused. In process mode they all get out, which is
what the warning is about. The result is in `run.json` under `sandbox.probe`.

## What the agent's code is given

A fresh folder per call:

    recipe.py      the agent's code inside a wrapper that sets up its scope
    lv1.py         the entry language (level1/lang.py)
    rewards.py     the reward functions
    model.py       the network's definition (models/<name>/model.py)
    harness.py     starts the recipe and writes the result
    weights.npz    the weights of the rounds' network, and no other
    input.json     sizes, the data set and how it is checked, limits, the entries saved so far

and limits from the run config (`python`): 300 s, 4 GB, 2 threads, a result of at most 8 MB. The
result comes back as one JSON file that is parsed as data: nothing the sandbox writes is executed or
unpickled, and an entry it hands back is checked again on the outside before it is saved.

On some Macs numpy's matrix product warns of a division by zero, an overflow and an invalid value
in every matmul, with nothing wrong; on Linux the same code prints nothing. The wrapper does not
pass those three messages on, so the agent sees the same output on every machine. Any other warning
still reaches it.

All of these files are searched by the hygiene check, since the agent can read them.

Landlock mode refuses to start if a folder the code may read (the Python installation, for
example) contains the repository or the runs folder. Keep them apart, or use container mode.

## Where the files came from

`level1/sandbox.py`, `level1/runner.py` and `level1/harness.py` were taken from the proof pipeline
(`core/sandbox.py`, `core/runner.py`, `core/harness.py` of Compact-Proof-Agent) and changed in what
goes into the folder, what is protected and what the probe aims at. `level1/mcp_server.py` and
`level1/backends/claude_code.py` came from there too. `harness.py` sits in every sandbox folder,
which is why its own text says nothing about where it came from.
