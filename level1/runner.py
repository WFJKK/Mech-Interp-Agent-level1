"""Run one recipe on one network, in a sandbox.

The runner makes a fresh sandbox folder holding only what the recipe may see:
recipe.py, read-only copies of the files it may import (the entry language and
the model's model.py), the harness, the network's weights (weights.npz, no
pickle) and input.json (info, extra and limits). It starts `python -I -B
harness.py` in the sandbox mode it is given (level1.sandbox: container, landlock
or process) with an empty environment, enforces the wall-clock and memory limits
from outside, and reads back one result file as JSON data. Nothing the sandbox
writes is executed or unpickled, and nothing it reports about itself is trusted.

Taken from the proof pipeline (core/runner.py of Compact-Proof-Agent); changed
only in what goes into the folder.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import psutil

from .sandbox import Sandbox, container_command, landlock_policy

HARNESS = Path(__file__).with_name("harness.py")
STATUSES = ("ok", "crash", "timeout", "bad_output")
PROCESS = Sandbox("process")

MIB = 1 << 20
POLL_S = 0.05
CPU_MARGIN_S = 10
SANDBOX_ID_VAR = "LV1_SANDBOX_ID"
MAX_RECIPE_PROCESSES = 2048  # a ceiling against a fork bomb; one recipe needs only a few
# The memory limit is enforced on resident memory, from outside. The data-size
# limit set inside is only a backstop, with headroom because importing numpy
# with a threaded BLAS maps memory that it never touches.
DATA_HEADROOM_MB = 1024
MAX_ERROR_CHARS = 2000
ERROR_FILE_BYTES = 1 << 14
STDERR_TAIL_BYTES = 1 << 13

# Must match the harness's exit codes; they are hints only.
EXIT_RAISED, EXIT_BAD_OUTPUT, EXIT_SANDBOX = 3, 4, 5
# A container runtime's own failures, and a container killed by SIGKILL.
CONTAINER_FAILURES, CONTAINER_KILLED = (125, 126, 127), 137


@dataclass(frozen=True)
class Limits:
    time_s: float
    memory_mb: float
    max_result_bytes: int
    threads: int = 1

    @classmethod
    def from_config(cls, cfg: dict) -> "Limits":
        return cls(
            time_s=float(cfg["time_limit_s"]),
            memory_mb=float(cfg["memory_limit_mb"]),
            max_result_bytes=int(float(cfg["max_result_mb"]) * MIB),
            threads=int(cfg.get("threads", 1)),
        )


@dataclass
class Execution:
    status: str
    result: dict | None = None
    error: str | None = None
    runtime_s: float = 0.0
    peak_rss_mb: float = 0.0
    result_bytes: int | None = None
    mode: str = "process"


class BadOutput(ValueError):
    pass


# reading what the sandbox wrote ------------------------------------------------
def _no_constant(name):
    raise ValueError(f"{name} is not a JSON number")


def _no_duplicates(pairs):
    out = {}
    for k, v in pairs:
        if k in out:
            raise ValueError(f"duplicate key {k[:40]!r}")
        out[k] = v
    return out


def parse_result_bytes(data: bytes, max_bytes: int) -> dict:
    """A result file as JSON data only: UTF-8, no NaN or infinities, no duplicate keys, an object at the top."""
    if len(data) > max_bytes:
        raise BadOutput(f"the result file has {len(data)} bytes, above the limit of {max_bytes}")
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        raise BadOutput("the result file is not UTF-8 text") from None
    try:
        obj = json.loads(text, parse_constant=_no_constant, object_pairs_hook=_no_duplicates)
    except RecursionError:
        raise BadOutput("the result file is nested too deeply") from None
    except ValueError as exc:
        raise BadOutput(f"the result file is not valid JSON: {str(exc)[:200]}") from None
    if not isinstance(obj, dict):
        raise BadOutput(f"the result file holds a JSON {type(obj).__name__}, not an object")
    return obj


def read_regular(path: Path, max_bytes: int) -> bytes | None:
    """Read a file the sandbox wrote, refusing links, pipes and anything else that is not a regular file."""
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    try:
        fd = os.open(path, flags)
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise BadOutput(f"{path.name} cannot be read ({exc.strerror}); links are refused") from None
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode):
            raise BadOutput(f"{path.name} is not a regular file")
        if st.st_size > max_bytes:
            raise BadOutput(f"{path.name} has {st.st_size} bytes, above the limit of {max_bytes}")
        chunks, remaining = [], max_bytes + 1
        while remaining > 0:
            b = os.read(fd, min(remaining, MIB))
            if not b:
                break
            chunks.append(b)
            remaining -= len(b)
        data = b"".join(chunks)
        if len(data) > max_bytes:
            raise BadOutput(f"{path.name} is above the limit of {max_bytes} bytes")
        return data
    finally:
        os.close(fd)


def _tail(f, n: int) -> str:
    try:
        f.flush()
        size = os.fstat(f.fileno()).st_size
        f.seek(max(0, size - n))
        return f.read().decode("utf-8", errors="replace")
    except OSError:
        return ""


def _clean(text: str, box: Path) -> str:
    text = text.replace(str(box) + os.sep, "").strip()
    return text[-MAX_ERROR_CHARS:]


# keeping a recipe's processes together ------------------------------------------
_SUBREAPER_SET = False


def _become_subreaper() -> None:
    """Orphaned descendants reparent to this process, not init, so a child that calls setsid() can still be found."""
    global _SUBREAPER_SET
    if _SUBREAPER_SET or not sys.platform.startswith("linux"):
        return
    _SUBREAPER_SET = True
    try:
        import ctypes

        ctypes.CDLL(None, use_errno=True).prctl(36, 1, 0, 0, 0)  # PR_SET_CHILD_SUBREAPER
    except (OSError, AttributeError):
        pass


def _reap_marked(sandbox_id: str, deadline_s: float = 5.0) -> int:
    """Kill every surviving process that carries this sandbox's id, including any that escaped the process group."""
    killed, end = 0, time.monotonic() + deadline_s
    while True:
        found = []
        try:
            for p in psutil.Process().children(recursive=True):
                try:
                    if p.environ().get(SANDBOX_ID_VAR) == sandbox_id:
                        found.append(p)
                except (psutil.Error, OSError):
                    pass
        except psutil.Error:
            break
        for p in found:
            try:
                p.kill()
            except psutil.Error:
                pass
        killed += len(found)
        if not found or time.monotonic() > end:
            return killed
        time.sleep(0.05)


# the sandbox folder -------------------------------------------------------------
def _environment(box: Path, threads: int, sandbox_id: str) -> dict:
    t = str(threads)
    return {
        "PATH": os.defpath,
        "HOME": str(box / "tmp"),
        "TMPDIR": str(box / "tmp"),
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PYTHONDONTWRITEBYTECODE": "1",
        SANDBOX_ID_VAR: sandbox_id,  # every process the recipe spawns inherits this, so teardown can find them all
        "OMP_NUM_THREADS": t,
        "MKL_NUM_THREADS": t,
        "OPENBLAS_NUM_THREADS": t,
        "NUMEXPR_NUM_THREADS": t,
        "VECLIB_MAXIMUM_THREADS": t,
    }


RESERVED = ("recipe.py", "harness.py", "weights.npz", "input.json", "out", "tmp")


def _prepare(box: Path, recipe_source: str, files: dict, weights: dict, info: dict, extra, limits: Limits, sb: Sandbox):
    box.mkdir()
    (box / "out").mkdir()
    (box / "tmp").mkdir()
    (box / "recipe.py").write_text(recipe_source)
    for name, source in files.items():
        if name in RESERVED or "/" in name or name.startswith(".") or not name.endswith(".py"):
            raise ValueError(f"{name!r} cannot be a file of the sandbox folder")
        (box / name).write_text(source)
    shutil.copyfile(HARNESS, box / "harness.py")
    np.savez(box / "weights.npz", **{k: np.asarray(v) for k, v in weights.items()})
    job = {
        "info": info,
        "extra": extra,
        "limits": {
            "data_bytes": int((limits.memory_mb + DATA_HEADROOM_MB) * MIB),
            "file_bytes": limits.max_result_bytes + MIB,
            "cpu_s": int(limits.time_s * max(1, limits.threads)) + CPU_MARGIN_S,
            "max_result_bytes": limits.max_result_bytes,
            "nproc": MAX_RECIPE_PROCESSES,
        },
        "lockdown": landlock_policy(box, sb.protected) if sb.mode == "landlock" else None,
    }
    (box / "input.json").write_text(json.dumps(job))
    for name in ("recipe.py", *files, "harness.py", "weights.npz", "input.json"):
        os.chmod(box / name, 0o444)
    if sb.mode == "container":
        os.chmod(box / "out", 0o777)  # the container runs as nobody
    os.chmod(box, 0o555)


def _remove(root: Path) -> None:
    """Delete a sandbox folder, whatever permissions the recipe left on it (links are not followed)."""
    try:
        os.chmod(root, 0o700)
    except OSError:
        pass
    for dirpath, dirnames, _ in os.walk(root):
        for d in dirnames:
            p = os.path.join(dirpath, d)
            if not os.path.islink(p):
                try:
                    os.chmod(p, 0o700)
                except OSError:
                    pass
    shutil.rmtree(root, ignore_errors=True)


def _tree_rss(proc) -> int:
    if proc is None:
        return 0
    try:
        procs = [proc] + proc.children(recursive=True)
    except psutil.Error:
        return 0
    total = 0
    for p in procs:
        try:
            total += p.memory_info().rss
        except psutil.Error:
            pass
    return total


def _kill_group(proc: subprocess.Popen) -> None:
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass


def _signal_name(rc: int) -> str:
    try:
        return signal.Signals(-rc).name
    except ValueError:
        return str(-rc)


# one execution --------------------------------------------------------------------
def run_recipe(
    recipe_source: str,
    files: dict,
    weights: dict,
    info: dict,
    extra,
    limits: Limits,
    sandbox_root: str | Path | None = None,
    sandbox: Sandbox | None = None,
) -> Execution:
    """Run make_result(weights, info, extra) from recipe_source in a fresh sandbox; return what came back.
    files are the modules the recipe may import, {file name: source}."""
    sb = sandbox or PROCESS
    root = Path(tempfile.mkdtemp(prefix="lv1-sandbox-", dir=sandbox_root))
    box = root / "box"
    sandbox_id = root.name
    if sb.mode != "container":
        _become_subreaper()
    try:
        _prepare(box, recipe_source, files, weights, info, extra, limits, sb)
        (root / "logs").mkdir()
        with open(root / "logs" / "stdout.txt", "w+b") as out_f, open(root / "logs" / "stderr.txt", "w+b") as err_f:
            start = time.monotonic()
            if sb.mode == "container":
                rc, verdict, peak = _wait_container(sb, box, limits, out_f, err_f)
            else:
                rc, verdict, peak = _wait_process(box, limits, out_f, err_f, sandbox_id)
            ex = Execution(status="crash", runtime_s=round(time.monotonic() - start, 3), peak_rss_mb=round(peak / MIB, 1), mode=sb.mode)
            return _interpret(ex, rc, verdict, box, limits, sb, _clean(_tail(err_f, STDERR_TAIL_BYTES), box))
    finally:
        if sb.mode != "container":
            _reap_marked(sandbox_id)  # kill anything that escaped the process group (for example via setsid)
        _remove(root)


def _wait_process(box: Path, limits: Limits, out_f, err_f, sandbox_id: str):
    """Start the harness as a separate process in its own process group; watch its time and resident memory."""
    proc = subprocess.Popen(
        [sys.executable, "-I", "-B", str(box / "harness.py")],
        cwd=box,
        env=_environment(box, limits.threads, sandbox_id),
        stdin=subprocess.DEVNULL,
        stdout=out_f,
        stderr=err_f,
        start_new_session=True,
        close_fds=True,
    )
    start = time.monotonic()
    try:
        ps = psutil.Process(proc.pid)
    except psutil.Error:
        ps = None
    peak, verdict = 0, None
    try:
        while proc.poll() is None:
            if time.monotonic() - start > limits.time_s:
                verdict = ("timeout", f"the recipe ran longer than the time limit of {limits.time_s:g} s")
                break
            rss = _tree_rss(ps)
            peak = max(peak, rss)
            if rss > limits.memory_mb * MIB:
                verdict = ("crash", f"the recipe used more than the memory limit of {limits.memory_mb:g} MB")
                break
            time.sleep(POLL_S)
    finally:
        _kill_group(proc)
        proc.wait()
    return proc.returncode, verdict, peak


def _wait_container(sb: Sandbox, box: Path, limits: Limits, out_f, err_f):
    """Start the harness in a container; the runtime enforces memory, CPU and process limits, the runner the time."""
    name = f"lv1-{os.urandom(8).hex()}"
    proc = subprocess.Popen(
        container_command(sb, box, name, limits.memory_mb, limits.threads),
        stdin=subprocess.DEVNULL,
        stdout=out_f,
        stderr=err_f,
        start_new_session=True,
        close_fds=True,
    )
    start, verdict = time.monotonic(), None
    try:
        while proc.poll() is None:
            if time.monotonic() - start > limits.time_s:
                verdict = ("timeout", f"the recipe ran longer than the time limit of {limits.time_s:g} s")
                break
            time.sleep(POLL_S)
    finally:
        if proc.poll() is None:
            subprocess.run([sb.runtime, "kill", name], capture_output=True, timeout=60)
            try:
                proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                _kill_group(proc)
                proc.wait()
    rc = proc.returncode
    if verdict is None and rc == CONTAINER_KILLED:
        verdict = (
            "crash",
            f"the recipe process was killed by signal SIGKILL; in a container that usually means it went over "
            f"the memory limit of {limits.memory_mb:g} MB",
        )
    return rc, verdict, 0


def _interpret(ex: Execution, rc: int, verdict, box: Path, limits: Limits, sb: Sandbox, stderr: str) -> Execution:
    if verdict is not None:
        ex.status, ex.error = verdict
        return ex
    if rc == 0:
        try:
            data = read_regular(box / "out" / "result.json", limits.max_result_bytes)
            if data is None:
                raise BadOutput("the recipe process wrote no result file")
            ex.result_bytes = len(data)
            ex.result = parse_result_bytes(data, limits.max_result_bytes)
            ex.status = "ok"
        except BadOutput as exc:
            ex.status, ex.error = "bad_output", str(exc)
        return ex

    try:
        err = read_regular(box / "out" / "error.txt", ERROR_FILE_BYTES)
    except BadOutput:
        err = None
    message = _clean(err.decode("utf-8", errors="replace"), box) if err else ""
    if rc == EXIT_RAISED:
        ex.status, ex.error = "crash", message or stderr or "the recipe raised an exception"
    elif rc == EXIT_BAD_OUTPUT:
        ex.status, ex.error = "bad_output", message or "the recipe returned no usable result"
    elif rc == EXIT_SANDBOX:
        ex.error = message or f"the {sb.mode} sandbox could not be set up"
    elif sb.mode == "container" and rc in CONTAINER_FAILURES:
        ex.error = f"the container could not run the recipe (exit code {rc})" + (f"\n{stderr}" if stderr else "")
    elif rc < 0:
        ex.error = f"the recipe process was killed by signal {_signal_name(rc)}" + (f"\n{stderr}" if stderr else "")
    else:
        ex.error = f"the recipe process exited with code {rc}" + (f"\n{stderr}" if stderr else "")
    ex.error = ex.error[-MAX_ERROR_CHARS:]
    return ex
