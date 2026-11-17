"""Run configs: config/defaults.yaml, then the files a config extends, then the config itself.

A config may name one file it extends (`extends`, relative to itself); mappings are merged key by key, and
anything else is replaced. One exception: a `reward` block that names another reward than the one it would
be merged into replaces it whole, since the parameters of one reward mean nothing to another.
"""

from __future__ import annotations

from pathlib import Path
import math

from . import budget, rewards
from .util import REPO_ROOT, read_yaml

DEFAULTS = REPO_ROOT / "config" / "defaults.yaml"
REQUIRED = ("run_id", "model", "network")
KEY_VARIABLES = ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN")  # the CLI's own; the backend removes them


class ConfigError(ValueError):
    pass


def merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        if k == "reward" and isinstance(v, dict) and isinstance(out.get(k), dict) and v.get("name", out[k].get("name")) != out[k].get("name"):
            out[k] = dict(v)
        else:
            out[k] = merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def _chain(path: Path, seen: tuple = ()) -> dict:
    path = Path(path).resolve()
    if path in seen:
        raise ConfigError(f"{path} extends itself")
    cfg = read_yaml(path) or {}
    if not isinstance(cfg, dict):
        raise ConfigError(f"{path} is not a mapping")
    parent = cfg.pop("extends", None)
    if parent is None:
        return cfg
    return merge(_chain(path.parent / parent, seen + (path,)), cfg)


def load(path, overrides: dict | None = None) -> dict:
    cfg = merge(read_yaml(DEFAULTS), _chain(Path(path)))
    cfg = merge(cfg, overrides or {})
    unknown = sorted(set(cfg) - set(read_yaml(DEFAULTS)) - {"model", "network", "fake_scripts"})
    if unknown:
        raise ConfigError(f"unknown keys in the run config: {unknown}")
    missing = [k for k in REQUIRED if not cfg.get(k)]
    if missing:
        raise ConfigError(f"the run config needs {missing}")
    for key in ("rounds_max", "patience"):
        if isinstance(cfg[key], bool) or not isinstance(cfg[key], int) or cfg[key] < 1:
            raise ConfigError("rounds_max and patience must be positive whole numbers")
    check_orchestration(cfg)
    try:
        rewards.check(cfg["reward"])
    except ValueError as exc:
        raise ConfigError(str(exc)) from None
    if cfg.get("auth") not in ("login", "api_key"):
        raise ConfigError(f"auth must be login or api_key, not {cfg.get('auth')!r}")
    for key in ("api_key_env", "api_workspace_env"):
        var = cfg.get(key)
        if var is not None and (not isinstance(var, str) or not var.isidentifier() or var in KEY_VARIABLES):
            raise ConfigError(f"{key} must name an environment variable other than {' and '.join(KEY_VARIABLES)}, not {var!r}")
    if cfg["auth"] == "api_key":
        if not cfg.get("api_key_env"):
            raise ConfigError("auth api_key needs api_key_env: the variable that holds the key")
        errors = budget.check(cfg.get("budget"))
        if errors:
            raise ConfigError("; ".join(errors))
    return cfg


def check_orchestration(cfg: dict) -> None:
    """Validate the optional research loop without changing old runs' reward or measurement policy."""
    options = cfg.get("orchestration") or {}
    defaults = read_yaml(DEFAULTS).get("orchestration", {})
    if not isinstance(options, dict):
        raise ConfigError("orchestration must be a mapping")
    unknown = sorted(set(options) - set(defaults))
    if unknown:
        raise ConfigError(f"unknown orchestration options: {unknown}")
    for key in ("enabled", "parallel", "final_summary"):
        if not isinstance(options.get(key, False), bool):
            raise ConfigError(f"orchestration.{key} must be true or false")
    if not options.get("enabled"):
        return
    workers = options.get("workers")
    if isinstance(workers, bool) or not isinstance(workers, int) or workers not in (1, 2):
        raise ConfigError("orchestration.workers must be 1 or 2")
    fractions = []
    for key in ("planning_fraction", "regional_fraction"):
        value = options.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 < value < 1:
            raise ConfigError(f"orchestration.{key} must be between 0 and 1")
        fractions.append(value)
    if sum(fractions) >= 1:
        raise ConfigError("planning_fraction + regional_fraction must leave a positive share for global integration")
    summary = options.get("summary_fraction", 0.0)
    if isinstance(summary, bool) or not isinstance(summary, (int, float)) or not math.isfinite(summary) or not 0 <= summary < 1:
        raise ConfigError("orchestration.summary_fraction must be a finite number from 0 up to but not including 1")
    if options.get("final_summary", False) and summary <= 0:
        raise ConfigError("final_summary needs a positive summary_fraction reserved within the run budget")
    summary = summary if options.get("final_summary", False) else 0.0
    research_share = 1.0 - summary
    for key in ("python_calls", "submissions", "max_turns"):
        value = cfg["session"][key]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ConfigError(f"session.{key} must be a positive whole number in research mode")
    seconds = cfg["session"]["seconds"]
    if isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or seconds <= 0:
        raise ConfigError("session.seconds must be a positive finite number")
    planning, regional = fractions
    global_fraction = 1 - planning - regional
    for key in ("python_calls", "max_turns"):
        amount = cfg["session"][key] * (research_share if key == "max_turns" else 1.0)
        if math.floor(amount * regional) // workers < 1 or math.floor(amount * global_fraction) < 1:
            raise ConfigError(f"session.{key} is too small after sharing its allocation between regional and global integration roles")
    if math.floor(cfg["session"]["max_turns"] * cfg["rounds_max"] * planning * research_share) < 1:
        raise ConfigError("the planning allocation must allow at least one turn")
    if summary and math.floor(cfg["session"]["max_turns"] * cfg["rounds_max"] * summary) < 1:
        raise ConfigError("the final summary allocation must allow at least one turn")
    for key in ("development_seed", "archive_per_scope"):
        value = options.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < (1 if key == "archive_per_scope" else 0):
            raise ConfigError(f"orchestration.{key} must be a valid nonnegative integer (archive_per_scope must be positive)")
    if int(cfg["session"]["submissions"]) <= workers:
        raise ConfigError("session.submissions must leave a submission for global integration after the regional workers")
    if int(cfg["session"]["python_calls"]) <= workers:
        raise ConfigError("session.python_calls must leave exploration calls for global integration")
    initial = options.get("initial_regions")
    if not isinstance(initial, list):
        raise ConfigError("orchestration.initial_regions must be a list of scope/task mappings")
    from .regions import normalize_scope

    for assignment in initial:
        if not isinstance(assignment, dict) or set(assignment) != {"scope", "task"}:
            raise ConfigError("each initial region needs only scope and task")
        if not isinstance(assignment["task"], str) or not assignment["task"].strip():
            raise ConfigError("each initial region needs a nonempty task")
        try:
            assignment["scope"] = normalize_scope(assignment["scope"])
        except (ValueError, RuntimeError) as exc:
            raise ConfigError(f"invalid initial region: {exc}") from None
    baseline = options.get("baseline_entry")
    if baseline is not None and (not isinstance(baseline, str) or not baseline.strip()):
        raise ConfigError("orchestration.baseline_entry must be a path or null")
    # An explicit scope opts into stable workload accounting. The absent scope retains the old evaluator.
    if cfg["verifier"].get("scope") is None:
        cfg["verifier"]["scope"] = {"kind": "all"}
    elif normalize_scope(cfg["verifier"]["scope"]) != {"kind": "all"}:
        raise ConfigError("the orchestration's global task must use scope kind all; put regional scopes in initial_regions")
    cfg["verifier"]["sample_seed"] = options["development_seed"]


def reward_from_args(name: str = "r_alpha", params: list | None = None, alpha: float | None = None,
                     penalty: float | None = None) -> dict:
    """A reward block from command-line pieces: a name, KEY=VALUE parameters (numbers, true, false), and the
    two common parameters by their own flags."""
    out: dict = {"name": name}
    for item in params or []:
        key, sep, value = str(item).partition("=")
        if not sep or not key:
            raise ConfigError(f"a reward parameter is KEY=VALUE, not {item!r}")
        low = value.strip().lower()
        if low in ("true", "false"):
            out[key.strip()] = low == "true"
        else:
            try:
                out[key.strip()] = float(value)
            except ValueError:
                raise ConfigError(f"the reward parameter {key!r} needs a number, true or false, not {value!r}") from None
    if alpha is not None:
        out["alpha"] = float(alpha)
    if penalty is not None:
        out["penalty"] = float(penalty)
    return out


def run_path(cfg: dict) -> Path:
    return Path(str(cfg["runs_dir"])).expanduser().resolve() / str(cfg["run_id"])
