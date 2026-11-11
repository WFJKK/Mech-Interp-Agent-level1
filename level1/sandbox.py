"""Where the agent's Python runs, and how well it is kept in.

Four modes, strongest first:

    container  docker or podman: no network, a read-only root filesystem, only
               the sandbox folder mounted (its out/ writable), running as nobody
               with no capabilities, and the runtime's memory, CPU and process
               limits. Needs the image from `python -m level1.sandbox build`.
    landlock   Linux without a container. Before the code is imported, the
               harness takes away its own rights: no new privileges; seccomp
               allows no sockets at all; Landlock allows reading only the Python
               installation, system libraries, the process's own /proc entry and
               the sandbox folder, writing only the sandbox's out/ and tmp/, and
               no ptrace or signals outside the sandbox. Needs no root.
    seatbelt   macOS without a container: the harness runs under sandbox-exec
               with a profile that refuses reading the home folder (except
               Python's own folders), reading the repository, the model folder
               and the runs folder, writing anywhere but the sandbox's out/ and
               tmp/, the network, other processes and signals. Opt-in for now
               (--sandbox seatbelt): it is used only when asked for, until it has
               passed the probe on a Mac.
    process    a plain separate process: python -I -B, an empty environment, a
               fresh folder and resource limits. The operating system does not
               keep the code in: it runs as you.

In every mode the harness also installs a guard inside Python (harness.py): an
audit hook that refuses file access outside Python and the sandbox folder,
sockets, other processes, signals and native code, and records each refusal. In
process mode it is the only fence; it shares its process with the code it
watches, so it is a fence and a record, not a wall.

`auto` takes the strongest of container, landlock and process that works here;
`hardened` never falls back to process: with neither container nor landlock
available (nor seatbelt, when asked for by name) it refuses to start. Every run
starts with probe(), which tries real escapes in the chosen mode; a hardened mode
that lets anything critical out is refused.

    python -m level1.sandbox info                  which modes work here
    python -m level1.sandbox build [--runtime R]   build the container image
    python -m level1.sandbox probe --model F [--mode M]   try to escape, and report

Taken from the proof pipeline (core/sandbox.py of Compact-Proof-Agent); changed in
what is protected and what the probe aims at.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import platform
import shutil
import site
import socket
import subprocess
import sys
import sysconfig
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from .util import REPO_ROOT

MODES = ("container", "landlock", "seatbelt", "process")
DEFAULT_IMAGE = "mech-interp-level1-sandbox:1"
DOCKERFILE_DIR = REPO_ROOT / "sandbox"
NOBODY = "65534:65534"
CONTAINER_PIDS = 256
CONTAINER_TMP_MB = 256

SYSTEM_READ = ("/usr", "/lib", "/lib32", "/lib64", "/libx32", "/bin", "/sbin", "/etc", "/sys")
DEVICE_READ = ("/dev/urandom", "/dev/random", "/dev/zero")
DEVICE_READ_WRITE = ("/dev/null",)
PROC_READ = ("/proc/cpuinfo", "/proc/meminfo", "/proc/stat")
MAC_SYSTEM_READ = ("/System", "/usr", "/bin", "/sbin", "/private/etc", "/etc")
SEATBELT = "/usr/bin/sandbox-exec"
SECCOMP_ARCHES = ("x86_64", "aarch64", "arm64")


class SandboxError(RuntimeError):
    pass


@dataclass(frozen=True)
class Sandbox:
    mode: str
    runtime: str | None = None
    image: str | None = None
    image_id: str | None = None
    protected: tuple = field(default_factory=tuple)

    @property
    def hardened(self) -> bool:
        return self.mode != "process"

    def describe(self) -> dict:
        d = {"mode": self.mode, "hardened": self.hardened}
        if self.mode == "container":
            d.update(runtime=self.runtime, image=self.image, image_id=self.image_id)
        return d


# what works here ------------------------------------------------------------------------
def landlock_abi() -> int:
    if not sys.platform.startswith("linux"):
        return 0
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        libc.syscall.restype = ctypes.c_long
        v = libc.syscall(444, None, ctypes.c_size_t(0), ctypes.c_uint32(1))  # LANDLOCK_CREATE_RULESET_VERSION
    except (OSError, AttributeError):
        return 0
    return int(v) if v > 0 else 0


def landlock_available() -> bool:
    return landlock_abi() >= 1 and platform.machine() in SECCOMP_ARCHES


def seatbelt_available() -> bool:
    return sys.platform == "darwin" and os.path.exists(SEATBELT)


# Docker Desktop can take a while to answer after its resource saver paused it.
CONTAINER_INFO_TIMEOUT_S = 90


def container_runtime(preferred: str | None = None) -> str | None:
    for name in [preferred] if preferred else ["docker", "podman"]:
        if name and shutil.which(name):
            try:
                if subprocess.run([name, "info"], capture_output=True, timeout=CONTAINER_INFO_TIMEOUT_S).returncode == 0:
                    return name
            except (OSError, subprocess.TimeoutExpired):
                pass
    return None


def image_id(runtime: str, image: str) -> str | None:
    """The image's ID, or None if it is not there. image inspect first; then the image list, since Docker
    Desktop's containerd image store can list (and run) an image that image inspect calls missing."""
    for cmd in (
        [runtime, "image", "inspect", "--format", "{{.Id}}", image],
        [runtime, "images", "--no-trunc", "--quiet", image],
    ):
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        except (OSError, subprocess.TimeoutExpired):
            continue
        out = r.stdout.strip().splitlines()
        if r.returncode == 0 and out and out[0].strip():
            return out[0].strip()
    return None


def protected_paths(cfg: dict | None = None, model=None, run_path: Path | None = None) -> tuple:
    """Paths the agent's code must never see: the repository (with the withheld folder), the model folder
    (with the other networks' weights) and the run folder."""
    out = [REPO_ROOT]
    if model is not None:
        out += [model.path, model.withheld()]
    if run_path is not None:
        out.append(Path(run_path))
    if cfg is not None and cfg.get("runs_dir"):
        out.append(Path(str(cfg["runs_dir"])).expanduser())
    return tuple(sorted({str(Path(p).resolve()) for p in out}))


def choose(cfg: dict, protected: tuple = ()) -> Sandbox:
    """The sandbox a run config asks for: a mode name, auto for the strongest that works here, or
    hardened for the strongest of container and landlock, refusing to fall back to process."""
    want = cfg.get("sandbox", "auto")
    image = cfg.get("sandbox_image") or DEFAULT_IMAGE
    if want not in ("auto", "hardened") + MODES:
        raise SandboxError(f"sandbox must be auto, hardened or one of {MODES}, not {want!r}")
    if want in ("auto", "hardened", "container"):
        rt = container_runtime()
        iid = image_id(rt, image) if rt else None
        if rt and iid:
            return Sandbox("container", runtime=rt, image=image, image_id=iid, protected=protected)
        if want == "container":
            why = "no docker or podman is running" if not rt else f"the image {image} is missing"
            raise SandboxError(f"container mode needs a container runtime and its image: {why}. Build it with python -m level1.sandbox build")
    if want == "seatbelt":
        if not seatbelt_available():
            raise SandboxError("seatbelt mode needs macOS with sandbox-exec")
        seatbelt_profile(Path(tempfile.gettempdir()) / "box", protected)  # refuses if the profile would expose a protected path
        return Sandbox("seatbelt", protected=protected)
    if want in ("auto", "hardened", "landlock"):
        if landlock_available():
            sb = Sandbox("landlock", protected=protected)
            landlock_policy(Path(tempfile.gettempdir()) / "box", protected)  # refuses if the policy would expose a protected path
            return sb
        if want == "landlock":
            raise SandboxError("landlock mode needs Linux with Landlock enabled on x86_64 or aarch64")
    if want == "hardened":
        why = "no docker or podman answered" if not rt else f"{rt} is running but the image {image} was not found"
        raise SandboxError(
            f"no hardened sandbox is available here ({why}): start Docker (or Podman) with the image built "
            "(python -m level1.sandbox build), or use Linux with Landlock. The run refuses to fall back to process mode."
        )
    return Sandbox("process", protected=protected)


# landlock -----------------------------------------------------------------------------------
def python_paths() -> list[str]:
    """Where this Python's standard library and installed packages live."""
    paths = {sys.prefix, sys.base_prefix, sys.exec_prefix, sys.base_exec_prefix}
    paths.add(os.path.dirname(os.path.realpath(sys.executable)))
    paths.update(v for k, v in sysconfig.get_paths().items() if k in ("stdlib", "platstdlib", "purelib", "platlib"))
    try:
        paths.update(site.getsitepackages())
    except AttributeError:
        pass
    out = set()
    for p in paths:
        if p and os.path.exists(p):
            out.add(os.path.realpath(p))
            out.add(os.path.abspath(p))
    return sorted(out)


def _refuse_overlap(read: list, protected: tuple, what: str) -> None:
    """Refuse if a path the code may read is, or contains, a protected one."""
    for r in read:
        rp = Path(r).resolve()
        for p in protected:
            pp = Path(p)
            if rp == pp or rp in pp.parents:
                raise SandboxError(
                    f"{what} would let recipes read {r}, which contains {p}; "
                    "move the repository, the runs folder or the Python installation apart, or use container mode"
                )


def system_reads() -> list[str]:
    paths = MAC_SYSTEM_READ if sys.platform == "darwin" else SYSTEM_READ + PROC_READ
    return [p for p in paths + DEVICE_READ if os.path.exists(p)]


def landlock_policy(box: Path, protected: tuple = ()) -> dict:
    """What a recipe may touch in landlock mode. Refuses if a readable path would contain a protected one."""
    read = python_paths() + [p for p in SYSTEM_READ + DEVICE_READ + PROC_READ if os.path.exists(p)] + [str(box)]
    _refuse_overlap(read, protected, "landlock mode")
    read_write = [str(box / "out"), str(box / "tmp")] + [p for p in DEVICE_READ_WRITE if os.path.exists(p)]
    return {"read": read, "read_write": read_write, "own_proc": True}


def guard_policy(box: Path, protected: tuple = ()) -> dict:
    """What the guard inside Python lets a recipe touch (harness.py): reading Python's folders, system files and
    the sandbox folder; writing its out/ and tmp/. Refuses if a readable path would contain a protected one."""
    read = python_paths() + system_reads() + [str(box)]
    _refuse_overlap(read, protected, "the guard")
    write = [str(box / "out"), str(box / "tmp")] + [p for p in DEVICE_READ_WRITE if os.path.exists(p)]
    return {"read": read, "write": write, "own_proc": True}


# seatbelt (macOS) ----------------------------------------------------------------------------
def _sb(path) -> str:
    return '"' + str(path).replace("\\", "\\\\").replace('"', '\\"') + '"'


def _both(paths) -> list[str]:
    """Each path as given and with its links resolved: the profile is matched against real paths."""
    out = set()
    for p in paths:
        out.add(os.path.abspath(str(p)))
        out.add(os.path.realpath(str(p)))
    return sorted(out)


def seatbelt_profile(box: Path, protected: tuple = (), python: str | None = None) -> str:
    """The sandbox-exec profile for one recipe. Everything not named stays as macOS allows it (allow default),
    and the rules after it take that away: in seatbelt profiles a later rule wins over an earlier one. So the
    order matters: reading the home folder and the protected paths is refused, then Python's own folders and
    the sandbox folder are allowed again (a virtual environment may sit inside the repository). The guard
    policy's overlap check refuses a profile in which a readable folder would contain a protected one."""
    policy = guard_policy(box, protected)
    python = python or sys.executable
    home = os.path.realpath(str(Path.home()))
    readable = _both(python_paths() + [str(box)])
    rules = [
        "(version 1)",
        "(allow default)",
        "(deny network*)",
        "(deny process-fork)",
        "(deny process-exec)",
        "(allow process-exec " + " ".join(f"(literal {_sb(p)})" for p in _both([python])) + ")",
        "(deny signal)",
        "(allow signal (target self))",
        "(deny file-write*)",
        "(allow file-write* " + " ".join(f"(subpath {_sb(p)})" for p in _both(policy["write"][:2]))
        + "".join(f" (literal {_sb(p)})" for p in DEVICE_READ_WRITE) + ")",
        "(deny file-read* " + " ".join(f"(subpath {_sb(p)})" for p in _both([home, *protected])) + ")",
        "(allow file-read* " + " ".join(f"(subpath {_sb(p)})" for p in readable) + ")",
        "(allow file-read-metadata)",  # finding one's way along a path; no contents, no listings
    ]
    return "\n".join(rules) + "\n"


def seatbelt_command(box: Path, protected: tuple, argv: list) -> list:
    return [SEATBELT, "-p", seatbelt_profile(box, protected, argv[0])] + list(argv)


# containers ------------------------------------------------------------------------------
def container_command(sb: Sandbox, box: Path, name: str, memory_mb: float, threads: int) -> list[str]:
    t = str(threads)
    env = {
        "HOME": "/tmp",
        "TMPDIR": "/tmp",
        "LANG": "C.UTF-8",
        "PYTHONDONTWRITEBYTECODE": "1",
        "OMP_NUM_THREADS": t,
        "MKL_NUM_THREADS": t,
        "OPENBLAS_NUM_THREADS": t,
    }
    cmd = [sb.runtime, "run", "--rm", "--init", "--name", name, "--network", "none", "--read-only"]
    cmd += ["--tmpfs", f"/tmp:rw,size={CONTAINER_TMP_MB}m,mode=1777"]
    cmd += ["--memory", f"{int(memory_mb)}m", "--memory-swap", f"{int(memory_mb)}m"]
    cmd += ["--cpus", t, "--pids-limit", str(CONTAINER_PIDS), "--user", NOBODY]
    cmd += ["--cap-drop", "ALL", "--security-opt", "no-new-privileges"]
    for k, v in env.items():
        cmd += ["-e", f"{k}={v}"]
    cmd += ["-v", f"{box}:/box:ro", "-v", f"{box / 'out'}:/box/out:rw", "-w", "/box"]
    cmd += [sb.image, "python", "-I", "-B", "/box/harness.py"]
    return cmd


def _marker_holds(marker: str, python: tuple | None = None) -> bool:
    """Whether a requirement line's condition holds on this Python, for the one form requirements.txt uses:
    python_version < "X.Y" or python_version >= "X.Y". Any other condition counts as holding."""
    import re

    m = re.fullmatch(r"""python_version\s*(<|>=)\s*["'](\d+)\.(\d+)["']""", marker.strip())
    if not m:
        return True
    here, there = tuple(python or sys.version_info[:2]), (int(m.group(2)), int(m.group(3)))
    return here < there if m.group(1) == "<" else here >= there


def requirement_versions(python: tuple | None = None) -> dict:
    """{package: pinned version} from requirements.txt, taking the lines whose condition holds on this Python."""
    out = {}
    for line in (REPO_ROOT / "requirements.txt").read_text().splitlines():
        line = line.split("#")[0].strip()
        requirement, _, marker = line.partition(";")
        if "==" in requirement and _marker_holds(marker, python):
            k, v = requirement.split("==", 1)
            out[k.strip().lower()] = v.strip()
    return out


def image_versions() -> dict:
    """The numpy version for the image: the one installed here, so the agent's code computes with the
    same library as the verifier, else the pin in requirements.txt."""
    from importlib import metadata

    out = requirement_versions()
    for name in ("numpy",):
        try:
            out[name] = metadata.version(name).split("+", 1)[0]
        except metadata.PackageNotFoundError:
            pass
    return out


def build_image(runtime: str | None = None, image: str = DEFAULT_IMAGE, log=print) -> str:
    rt = container_runtime(runtime)
    if rt is None:
        raise SandboxError("no docker or podman is running")
    v = image_versions()
    py = f"{sys.version_info.major}.{sys.version_info.minor}"
    cmd = [rt, "build", "-t", image, "--build-arg", f"PYTHON_VERSION={py}"]
    cmd += ["--build-arg", f"NUMPY_VERSION={v['numpy']}", str(DOCKERFILE_DIR)]
    log(" ".join(cmd))
    if subprocess.run(cmd).returncode != 0:
        raise SandboxError("the image build failed")
    return image_id(rt, image)


# the probe ------------------------------------------------------------------------------
VIA = ("numpy", "hardlink", "symlink", "subprocess", "ctypes")  # other ways to the withheld file than open()
CRITICAL = ("read_trusted_code", "read_other_weights", "read_withheld", "read_run_folder", "write_trusted_folder",
            "write_model_folder", "write_run_folder", "write_outside", "network_tcp", "network_udp",
            "read_driver_environment", "import_verifier") + tuple(f"read_withheld_{v}" for v in VIA)
MINOR = ("signal_driver",)

PROBE_RECIPE = '''
import json, os, socket, sys

TARGETS = json.loads({targets!r})


def attempt(fn):
    try:
        return fn()
    except BaseException as exc:  # anything that stops the attempt
        return "blocked: " + type(exc).__name__


def read(path):
    with open(path, "rb") as f:
        return f.read(32).hex()


def read_numpy(path):
    import numpy as np

    return bytes(np.fromfile(path, dtype=np.uint8)[:32]).hex()


def read_hardlink(path):
    os.link(path, os.path.join("tmp", "probe-hardlink"))
    return read(os.path.join("tmp", "probe-hardlink"))


def read_symlink(path):
    os.symlink(path, os.path.join("tmp", "probe-symlink"))
    return read(os.path.join("tmp", "probe-symlink"))


def read_subprocess(path):
    import subprocess

    return subprocess.run(["cat", path], capture_output=True, timeout=10).stdout[:32].hex()


def read_ctypes(path):
    import ctypes

    libc = ctypes.CDLL(None, use_errno=True)
    fd = libc.open(path.encode(), 0)
    if fd < 0:
        raise OSError(ctypes.get_errno(), "open")
    buf = ctypes.create_string_buffer(32)
    n = libc.read(fd, buf, 32)
    return buf.raw[:max(n, 0)].hex()


VIA = {{"numpy": read_numpy, "hardlink": read_hardlink, "symlink": read_symlink, "subprocess": read_subprocess,
       "ctypes": read_ctypes}}


def write(folder):
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, TARGETS["marker"]), "w") as f:
        f.write(TARGETS["token"])
    return "written"


def tcp():
    s = socket.create_connection(("127.0.0.1", TARGETS["tcp_port"]), timeout=2)
    s.sendall(TARGETS["token"].encode())
    s.close()
    return "sent"


def udp():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.sendto(TARGETS["token"].encode(), ("127.0.0.1", TARGETS["udp_port"]))
    s.close()
    return "sent"


def import_verifier():
    sys.path.insert(0, TARGETS["repo"])
    import level1.verifier as c

    return "imported " + c.__file__


def signal_driver():
    os.kill(TARGETS["driver_pid"], 0)
    return "signalled"


def make_result(weights, info, extra):
    r = {{}}
    for name, path in TARGETS["reads"].items():
        r[name] = attempt(lambda: read(path))
    for via, fn in VIA.items():
        r["read_withheld_" + via] = attempt(lambda: fn(TARGETS["reads"]["read_withheld"]))
    for name, folder in TARGETS["writes"].items():
        r[name] = attempt(lambda: write(folder))
    r["network_tcp"] = attempt(tcp)
    r["network_udp"] = attempt(udp)
    r["import_verifier"] = attempt(import_verifier)
    r["signal_driver"] = attempt(signal_driver)
    return {{"probe": r}}
'''


class _Listener:
    """A TCP and a UDP socket on this machine's localhost that record whether the probe's token reached them."""

    def __init__(self, token: bytes):
        self.token = token
        self.tcp = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.tcp.bind(("127.0.0.1", 0))
        self.tcp.listen(4)
        self.tcp.settimeout(0.2)
        self.udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.udp.bind(("127.0.0.1", 0))
        self.udp.settimeout(0.2)
        self.got = {"network_tcp": False, "network_udp": False}
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        while not self._stop.is_set():
            try:
                conn, _ = self.tcp.accept()
                self.got["network_tcp"] = True  # a connection is enough
                conn.close()
            except OSError:
                pass
            try:
                data, _ = self.udp.recvfrom(256)
                self.got["network_udp"] = self.got["network_udp"] or self.token in data
            except OSError:
                pass

    def ports(self):
        return self.tcp.getsockname()[1], self.udp.getsockname()[1]

    def close(self):
        self._stop.set()
        self._thread.join(timeout=2)
        self.tcp.close()
        self.udp.close()


def _head(path: Path) -> str | None:
    try:
        with open(path, "rb") as f:
            return f.read(32).hex()  # the first 32 bytes; not a size
    except OSError:
        return None


def probe(sb: Sandbox, model, network_id: str, run_path: Path, limits=None, guard: bool = False) -> dict:
    """Try every escape in the probe recipe inside sandbox sb, and report what got out.

    Escapes are judged by evidence on this side only: bytes read that match the
    real file, marker files that appear, tokens that reach a local listener, a
    successful import of the verifier or a signal to this process. run_path is a
    folder the code must not reach (normally the run's folder).

    guard=False (the default) tests the system's sandbox alone: the guard inside
    Python would otherwise stop most attempts before they reach it. guard=True
    tests the guard, as the agent's code meets it.
    """
    from .runner import Limits, run_recipe

    others = [n for n in model.config["networks"] if n["id"] != network_id]
    other_weights = model.path / others[0]["weights"] if others else model.path / "config.yaml"
    withheld = model.withheld() / "task.yaml"
    token = os.urandom(12).hex()
    run_path = Path(run_path)
    run_path.mkdir(parents=True, exist_ok=True)
    run_file = run_path / ".probe-target"
    run_file.write_text(f"{token}: the probe must not read this\n")
    reads = {
        "read_trusted_code": REPO_ROOT / "level1" / "verifier.py",
        "read_other_weights": other_weights,
        "read_withheld": withheld,
        "read_run_folder": run_file,
        "read_driver_environment": Path(f"/proc/{os.getpid()}/environ"),
    }
    writes = {
        "write_trusted_folder": REPO_ROOT / "level1" / "__pycache__",
        "write_model_folder": model.path,
        "write_run_folder": run_path,
        "write_outside": Path.home(),  # only the marker file the probe drops is ever removed
    }
    marker = f".lv1-probe-{os.getpid()}-{token[:8]}"
    truth = {k: _head(v) for k, v in reads.items()}
    listener = _Listener(token.encode())
    tcp_port, udp_port = listener.ports()
    targets = {
        "reads": {k: str(v) for k, v in reads.items()},
        "writes": {k: str(v) for k, v in writes.items()},
        "marker": marker,
        "token": token,
        "tcp_port": tcp_port,
        "udp_port": udp_port,
        "driver_pid": os.getpid(),
        "repo": str(REPO_ROOT),
    }
    try:
        ex = run_recipe(
            PROBE_RECIPE.format(targets=json.dumps(targets)),
            {},
            model.load_weights(network_id),
            model.info(),
            None,
            limits or Limits(time_s=120, memory_mb=2048, max_result_bytes=1 << 20),
            sandbox=sb,
            guard=guard,
        )
    finally:
        time.sleep(0.3)  # let the listener see late packets
        listener.close()
        landed = {}
        for name, d in writes.items():
            f = d / marker
            if f.exists() or f.is_symlink():
                landed[name] = True
                f.unlink()
        run_file.unlink(missing_ok=True)
    if ex.status != "ok" or not isinstance(ex.result.get("probe"), dict):
        raise SandboxError(f"the sandbox probe did not run in {sb.mode} mode: {ex.status}: {ex.error}")
    seen = ex.result["probe"]
    escaped = {k for k in reads if truth[k] is not None and seen.get(k) == truth[k]}
    escaped |= {f"read_withheld_{v}" for v in VIA if truth["read_withheld"] is not None and seen.get(f"read_withheld_{v}") == truth["read_withheld"]}
    escaped |= set(landed)
    escaped |= {k for k, got in listener.got.items() if got}
    if str(seen.get("import_verifier", "")).startswith("imported"):
        escaped.add("import_verifier")
    if seen.get("signal_driver") == "signalled":
        escaped.add("signal_driver")
    return {
        "mode": sb.mode,
        "guard": guard,
        "critical_escapes": sorted(escaped & set(CRITICAL)),
        "minor_escapes": sorted(escaped & set(MINOR)),
        "attempts": {k: ("escaped" if k in escaped else ("blocked" if not str(v).startswith("blocked") else v)) for k, v in seen.items()},
    }


def check_probe(sb: Sandbox, report: dict) -> None:
    if report.get("guard") and report["critical_escapes"]:
        raise SandboxError(f"the guard inside Python is not keeping the agent's code in here: {', '.join(report['critical_escapes'])}")
    if sb.hardened and report["critical_escapes"]:
        raise SandboxError(f"{sb.mode} mode is not keeping the agent's code in here: {', '.join(report['critical_escapes'])}")


# command line -----------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("info")
    b = sub.add_parser("build")
    b.add_argument("--runtime", choices=["docker", "podman"])
    b.add_argument("--image", default=DEFAULT_IMAGE)
    p = sub.add_parser("probe")
    p.add_argument("--mode", default="auto", choices=("auto", "hardened") + MODES)
    p.add_argument("--model", required=True, help="a model folder")
    p.add_argument("--image", default=DEFAULT_IMAGE)
    args = ap.parse_args(argv)
    if args.cmd == "info":
        rt = container_runtime()
        print(f"container: runtime {rt or 'none'}; image {DEFAULT_IMAGE} {'present' if rt and image_id(rt, DEFAULT_IMAGE) else 'missing'}")
        print(f"landlock: {'available' if landlock_available() else 'not available'} (ABI {landlock_abi()}, {platform.machine()})")
        print(f"seatbelt: {'available, used only when asked for (--sandbox seatbelt); try it with probe --mode seatbelt' if seatbelt_available() else 'not available (macOS only)'}")
        print("process: always available (weak: a guard inside Python, no wall)")
        return 0
    if args.cmd == "build":
        print(build_image(args.runtime, args.image))
        return 0
    from .networks import load_model

    model = load_model(args.model)
    nid = model.config["networks"][0]["id"]
    with tempfile.TemporaryDirectory() as tmp:
        cfg = {"sandbox": args.mode, "sandbox_image": args.image, "runs_dir": tmp}
        sb = choose(cfg, protected_paths(cfg, model, Path(tmp)))
        report = probe(sb, model, nid, Path(tmp) / "run")
        guarded = probe(sb, model, nid, Path(tmp) / "run", guard=True)
    print(json.dumps({"system": report, "with the guard": guarded}, indent=2))
    print(f"{sb.mode} mode on its own: " + (", ".join(report["critical_escapes"]) + " got out" if report["critical_escapes"] else "nothing got out"))
    print("with the guard inside Python: " + (", ".join(guarded["critical_escapes"]) + " got out" if guarded["critical_escapes"] else "nothing got out"))
    return 1 if guarded["critical_escapes"] or (sb.hardened and report["critical_escapes"]) else 0


if __name__ == "__main__":
    sys.exit(main())
