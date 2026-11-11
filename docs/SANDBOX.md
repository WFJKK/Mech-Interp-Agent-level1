# The sandbox

The only code of the agent's that ever runs is what it sends to `run_python`. An entry is data and
is interpreted by the verifier; it never runs as code. So the sandbox has one job: run the agent's
exploring code and keep it away from everything it must not see or touch.

What it must not see: the withheld folder (what the network was trained to do, the reference
entries), the second network's weights, the repository, the run folder. What it must not do: reach
the network, start other programs, write outside its own folder.

There are two layers: the system's sandbox (a mode, below), and a guard inside Python that is on in
every mode.

## Modes

| Mode | Needs | What it does |
| --- | --- | --- |
| `container` | Docker or Podman, and the image | No network, a read-only root, only the sandbox folder mounted, runs as nobody with no capabilities, memory, CPU and process limits |
| `landlock` | Linux with Landlock (x86_64 or aarch64) | Before the code is imported the process gives up its rights for good: no sockets, reading only Python, system libraries and the sandbox folder, writing only the folder's `out/` and `tmp/` |
| `seatbelt` | macOS (`sandbox-exec`) | The process starts under a macOS sandbox profile: no reading in the home folder except Python's own folders, no reading the repository, model or runs folders, writing only the folder's `out/` and `tmp/`, no network, no other programs, no signals to other processes. Opt-in for now |
| `process` | nothing | A separate process with an empty environment, a fresh folder and resource limits. The system does not keep the code in: only the guard does |

The run config's `sandbox` picks one: a mode by name, `auto` for the strongest of container,
landlock and process that works here, or `hardened` for the strongest of container and landlock,
refusing to start if neither works. `--sandbox` on the run command overrides the config.
`seatbelt` is used only when asked for by name, until it has passed the probe on a Mac.

    python -m level1.sandbox info      # which modes work on this machine
    python -m level1.sandbox build     # build the container image (Docker or Podman must be running)
    python -m level1.sandbox probe --model models/lin-a [--mode seatbelt]

## The guard inside Python

In every mode, just before the agent's code is imported, the harness installs an audit hook
(Python's own mechanism for watching what code does: it is called before a file is opened, a
folder listed, a socket made, a program started, a signal sent or native code loaded). It refuses:

- opening or listing anything outside Python's own folders, system files and the sandbox folder;
- writing, moving, linking or deleting anything outside the folder's `out/` and `tmp/`;
- relative paths that climb with `..` (they could be meant for a folder the hook cannot see);
- sockets, other programs, signals, `ctypes`, and adding another audit hook.

The code gets a `PermissionError`, and the refusal is written to the folder's `out/audit.jsonl`.
The session keeps that record next to the program (`python/NNN.audit.json`) whenever something was
refused or the record is incomplete (a time limit, or code that ended its process its own way),
and the report lists every refusal under "What the agent ran". Ordinary exploring is untouched:
numpy, the standard library, files in its own folder. On the smoke run's programs it costs no
measurable time.

The guard shares its process with the code it watches, so it is a fence and a record, not a wall:
code written to get around it might. That is what the system's sandbox is for. In process mode the
guard is the only fence, and a run that is meant to count needs one of the other modes.

## On a Mac

Landlock is a Linux feature, so a Mac has three choices:

- **Docker Desktop (or Podman) running, and the image built once** with
  `python -m level1.sandbox build`. Then `auto` picks container mode.
- **seatbelt**, macOS's own sandbox, asked for by name: `--sandbox seatbelt`. Try it first with
  `python -m level1.sandbox probe --model models/lin-a --mode seatbelt`; the tests also try it on
  a Mac (`test_seatbelt_keeps_everything_in`). It was written without a Mac to run it on, so its
  first test on one is the real one.
- **Nothing**: `auto` falls back to process mode, with the guard.

Every program the agent ran is also kept in full in the session folder (`round_k/session/python/`),
and the report lists the programs whose text reaches for files or the system (`open(`, `os.`,
`pathlib`, `subprocess`, sockets, the withheld folder's name). A match is a reason to read the
program, not a verdict.

## The probe

Every run starts by trying real escapes in the chosen mode, and judges them by evidence on the
outside (bytes that match the real file, a marker file that appears, a token that reaches a local
listener):

- read the verifier's code, the second network's weights, the withheld task file, the run folder,
  the driving process's environment;
- read the withheld task file another way: with numpy, through a hard link or a symbolic link
  made in the folder, with `cat` in another program, with `open` called through `ctypes`;
- write into the repository, the model folder, the run folder, the home folder;
- open a TCP and a UDP connection;
- import the verifier.

It runs twice: once with the guard left out, to test the system's sandbox on its own, and once with
the guard, as the agent's code meets it. A hardened mode that lets anything out on its own is
refused, and so is any mode in which something gets past the guard. In process mode everything gets
out without the guard, and nothing with it. The results are in `run.json` under `sandbox.probe`
and `sandbox.guard_probe`.

## What the agent's code is given

A fresh folder per call:

    recipe.py      the agent's code inside a wrapper that sets up its scope
    lv1.py         the entry language (level1/lang.py)
    rewards.py     the reward functions
    model.py       the network's definition (models/<name>/model.py)
    harness.py     starts the recipe, installs the guard and writes the result
    weights.npz    the weights of the rounds' network, and no other
    input.json     sizes, the data set and how it is checked, limits, the guard's policy (the
                   folders it may read and write; never a protected one), the entries saved so far

and limits from the run config (`python`): 300 s, 4 GB, 2 threads, a result of at most 8 MB. The
result comes back as one JSON file that is parsed as data: nothing the sandbox writes is executed or
unpickled, and an entry it hands back is checked again on the outside before it is saved.

On some Macs numpy's matrix product warns of a division by zero, an overflow and an invalid value
in every matmul, with nothing wrong; on Linux the same code prints nothing. The wrapper does not
pass those three messages on, so the agent sees the same output on every machine. Any other warning
still reaches it.

All of these files are searched by the hygiene check, since the agent can read them.

Landlock mode, the guard and seatbelt mode all refuse to start if a folder the code may read (the
Python installation, for example) contains the repository or the runs folder. Keep them apart, or
use container mode. A virtual environment inside the repository is fine: it is inside the
repository, not around it, and only its own folder becomes readable.

## Where the files came from

`level1/sandbox.py`, `level1/runner.py` and `level1/harness.py` were taken from the proof pipeline
(`core/sandbox.py`, `core/runner.py`, `core/harness.py` of Compact-Proof-Agent) and changed in what
goes into the folder, what is protected and what the probe aims at. `level1/mcp_server.py` and
`level1/backends/claude_code.py` came from there too. `harness.py` sits in every sandbox folder,
which is why its own text says nothing about where it came from.
