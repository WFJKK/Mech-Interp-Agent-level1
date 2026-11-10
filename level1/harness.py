"""Runs one recipe inside a sandbox folder. Not trusted: it shares its process with the recipe.

level1.runner copies this file into a fresh sandbox folder next to recipe.py, the
files the recipe may import, weights.npz and input.json, and starts it with
`python -I -B harness.py`. It sets its own resource limits before the recipe is
imported, calls make_result(weights, info, extra), and writes out/result.json.

(Where this file comes from is said in docs/SANDBOX.md; a copy of it sits in every
sandbox folder, so it says nothing about the pipeline around it.)

In landlock mode, input.json carries a policy, and before anything else the
harness takes away its own rights for good (no new privileges, seccomp with no
sockets, Landlock), failing closed with exit code 5 if any step fails.

Its exit code is only a hint to the runner, which trusts nothing written here:
0 the result was written, 3 the recipe raised, 4 the result is not a JSON object
within the size limit, 5 the sandbox could not be set up.

This file uses only the standard library and numpy, and imports nothing from
the pipeline.
"""

import json
import os
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
MAX_ERROR_CHARS = 4000

EXIT_OK, EXIT_RAISED, EXIT_BAD_OUTPUT, EXIT_SANDBOX = 0, 3, 4, 5
# Landlock's system calls have the same numbers on every architecture.
SYS_LANDLOCK_CREATE_RULESET, SYS_LANDLOCK_ADD_RULE, SYS_LANDLOCK_RESTRICT_SELF = 444, 445, 446


# landlock mode: no new privileges, no sockets, and a file-system allow-list ----------------
def _libc():
    import ctypes

    libc = ctypes.CDLL(None, use_errno=True)
    libc.syscall.restype = ctypes.c_long
    return ctypes, libc


def _check(ctypes, result, what):
    if result != 0:
        err = ctypes.get_errno()
        raise OSError(err, f"{what}: {os.strerror(err)}")


def _no_sockets(ctypes, libc):
    """A seccomp filter: socket() fails with EACCES; any other architecture or the x32 ABI fails everything."""
    import platform
    import struct

    arch, nr = {"x86_64": (0xC000003E, 41), "aarch64": (0xC00000B7, 198), "arm64": (0xC00000B7, 198)}[platform.machine()]
    load, jeq, jge, ret = 0x20, 0x15, 0x35, 0x06  # BPF opcodes; not a size
    allow, deny = 0x7FFF0000, 0x00050000 | 13
    prog = [
        (load, 0, 0, 4),  # architecture
        (jeq, 1, 0, arch),
        (ret, 0, 0, deny),
        (load, 0, 0, 0),  # system call number
        (jge, 2, 0, 0x40000000),  # x32 system calls
        (jeq, 1, 0, nr),
        (ret, 0, 0, allow),
        (ret, 0, 0, deny),
    ]
    raw = b"".join(struct.pack("HBBI", *ins) for ins in prog)
    filt = ctypes.create_string_buffer(raw, len(raw))

    class Prog(ctypes.Structure):
        _fields_ = [("len", ctypes.c_ushort), ("filter", ctypes.c_void_p)]

    fprog = Prog(len(prog), ctypes.cast(filt, ctypes.c_void_p))
    _check(ctypes, libc.prctl(22, 2, ctypes.byref(fprog), 0, 0), "seccomp")  # PR_SET_SECCOMP, SECCOMP_MODE_FILTER


def _landlock(ctypes, libc, read, read_write):
    import stat
    import struct

    abi = libc.syscall(SYS_LANDLOCK_CREATE_RULESET, None, ctypes.c_size_t(0), ctypes.c_uint32(1))  # the ABI version
    if abi < 1:
        raise OSError(ctypes.get_errno(), "Landlock is not available")
    handled = (1 << 13) - 1
    if abi >= 2:
        handled |= 1 << 13  # refer
    if abi >= 3:
        handled |= 1 << 14  # truncate
    if abi >= 5:
        handled |= 1 << 15  # ioctl on devices
    net = 3 if abi >= 4 else 0  # bind and connect TCP
    scoped = 3 if abi >= 6 else 0  # abstract unix sockets and signals
    size = 8 if abi < 4 else (16 if abi < 6 else 24)
    attr = ctypes.create_string_buffer(struct.pack("QQQ", handled, net, scoped), 24)
    fd = libc.syscall(SYS_LANDLOCK_CREATE_RULESET, attr, ctypes.c_size_t(size), ctypes.c_uint32(0))
    if fd < 0:
        raise OSError(ctypes.get_errno(), "landlock_create_ruleset")
    execute, write_file, read_file, read_dir = 1, 2, 4, 8
    make_char, make_sock, make_fifo, make_block, make_sym = 1 << 6, 1 << 9, 1 << 10, 1 << 11, 1 << 12
    file_rights = execute | write_file | read_file | (1 << 14) | (1 << 15)
    ro = read_file | read_dir | execute
    rw = handled & ~(make_char | make_sock | make_fifo | make_block | make_sym | execute)
    for paths, access in ((read, ro), (read_write, rw)):
        for path in paths:
            try:
                pfd = os.open(path, os.O_PATH | os.O_CLOEXEC)
            except OSError:
                continue  # a path that does not exist grants nothing
            try:
                a = access & handled
                if not stat.S_ISDIR(os.fstat(pfd).st_mode):
                    a &= file_rights
                rule = ctypes.create_string_buffer(struct.pack("=Qi", a, pfd), 12)
                rc = libc.syscall(SYS_LANDLOCK_ADD_RULE, fd, ctypes.c_int(1), rule, ctypes.c_uint32(0))  # a path rule
                _check(ctypes, rc, f"landlock_add_rule {path}")
            finally:
                os.close(pfd)
    _check(ctypes, libc.syscall(SYS_LANDLOCK_RESTRICT_SELF, fd, ctypes.c_uint32(0)), "landlock_restrict_self")
    os.close(fd)
    return abi


def _lockdown(policy):
    ctypes, libc = _libc()
    _check(ctypes, libc.prctl(38, 1, 0, 0, 0), "no_new_privs")  # PR_SET_NO_NEW_PRIVS
    _no_sockets(ctypes, libc)
    read = list(policy["read"])
    if policy.get("own_proc"):
        read.append(os.path.realpath("/proc/self"))  # this process's own entry, nobody else's
    return _landlock(ctypes, libc, read, policy["read_write"])


def _limit(name, value):
    try:
        import resource
    except ImportError:  # not available on this platform; the runner still watches from outside
        return
    if value is None or not hasattr(resource, name):
        return
    which = getattr(resource, name)
    try:
        _, hard = resource.getrlimit(which)
        v = int(value)
        if hard != resource.RLIM_INFINITY:
            v = min(v, hard)
        resource.setrlimit(which, (v, v))
    except (ValueError, OSError):
        pass


def _finish(code, message=None):
    if message is not None:
        with open(os.path.join(OUT, "error.txt"), "w") as f:
            f.write(message[-MAX_ERROR_CHARS:])
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)


def _plain(obj):
    """Numpy numbers and arrays in a result become JSON numbers and lists."""
    import numpy as np

    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"a {type(obj).__name__} is not JSON-serialisable")


def main():
    with open(os.path.join(HERE, "input.json")) as f:
        job = json.load(f)
    if job.get("lockdown") is not None:
        try:
            _lockdown(job["lockdown"])
        except BaseException as exc:  # fail closed: no recipe runs in a half-built sandbox
            _finish(EXIT_SANDBOX, f"the sandbox could not be set up: {exc!r}")
    limits = job["limits"]
    _limit("RLIMIT_DATA", limits.get("data_bytes"))
    _limit("RLIMIT_FSIZE", limits.get("file_bytes"))
    _limit("RLIMIT_CPU", limits.get("cpu_s"))
    _limit("RLIMIT_CORE", 0)
    if limits.get("nproc"):
        _limit("RLIMIT_NPROC", limits["nproc"])  # a ceiling against a fork bomb
    sys.path.insert(0, HERE)  # -I leaves the script's folder off the path

    import numpy as np

    with np.load(os.path.join(HERE, "weights.npz"), allow_pickle=False) as z:
        weights = {k: z[k] for k in z.files}
    try:
        import recipe

        result = recipe.make_result(weights, job["info"], job.get("extra"))
    except BaseException:  # includes SystemExit and KeyboardInterrupt from the recipe
        _finish(EXIT_RAISED, traceback.format_exc())
    if not isinstance(result, dict):
        _finish(EXIT_BAD_OUTPUT, f"make_result returned a {type(result).__name__}, not a dict")
    try:
        text = json.dumps(result, allow_nan=False, default=_plain, separators=(",", ":"))
    except (TypeError, ValueError, RecursionError) as exc:
        _finish(EXIT_BAD_OUTPUT, f"the result is not valid JSON data: {exc}")
    data = text.encode()
    if len(data) > limits["max_result_bytes"]:
        _finish(EXIT_BAD_OUTPUT, f"the result file has {len(data)} bytes, above the limit of {limits['max_result_bytes']}")
    tmp = os.path.join(OUT, "result.json.tmp")
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, os.path.join(OUT, "result.json"))
    _finish(EXIT_OK)


if __name__ == "__main__":
    main()
