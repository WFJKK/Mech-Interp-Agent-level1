"""Run configs: config/defaults.yaml, then the files a config extends, then the config itself.

A config may name one file it extends (`extends`, relative to itself); mappings are merged key by key, and
anything else is replaced. One exception: a `reward` block that names another reward than the one it would
be merged into replaces it whole, since the parameters of one reward mean nothing to another.
"""

from __future__ import annotations

from pathlib import Path
import math

from . import budget, research_modes, rewards
from .util import REPO_ROOT, read_yaml

DEFAULTS = REPO_ROOT / "config" / "defaults.yaml"
REQUIRED = ("run_id", "model", "network")
KEY_VARIABLES = ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN")  # the CLI's own; the backend removes them
OPENAI_KEY_VARIABLES = ("OPENAI_API_KEY", "CODEX_API_KEY")  # the Codex CLI's own; a run's key has a variable of its own
BACKENDS = ("claude_code", "codex", "openai_api", "fake")


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
    check_continuation(cfg)
    try:
        rewards.check(cfg["reward"])
    except ValueError as exc:
        raise ConfigError(str(exc)) from None
    if cfg.get("auth") not in ("login", "api_key"):
        raise ConfigError(f"auth must be login or api_key, not {cfg.get('auth')!r}")
    if cfg.get("backend") not in BACKENDS:
        raise ConfigError(f"no backend named {cfg.get('backend')!r}; there are {', '.join(BACKENDS)}")
    reserved = KEY_VARIABLES + OPENAI_KEY_VARIABLES
    for key in ("api_key_env", "api_workspace_env"):
        var = cfg.get(key)
        if var is not None and (not isinstance(var, str) or not var.isidentifier() or var in reserved):
            raise ConfigError(f"{key} must name an environment variable other than {' and '.join(reserved)}, not {var!r}")
    if cfg["backend"] == "codex" and cfg["auth"] != "login":
        raise ConfigError("the codex backend runs under a ChatGPT login only (auth: login): the Codex CLI has no spending cap, "
                          "so a key is billed through the openai_api backend, which has one")
    if cfg["backend"] == "openai_api":
        if cfg["auth"] != "api_key":
            raise ConfigError("the openai_api backend bills an API key: set auth: api_key (or --auth api_key) and api_key_env")
        check_openai(cfg)
    if cfg["auth"] == "api_key":
        if not cfg.get("api_key_env"):
            raise ConfigError("auth api_key needs api_key_env: the variable that holds the key")
        errors = budget.check(cfg.get("budget"))
        if errors:
            raise ConfigError("; ".join(errors))
    return cfg


def openai_prices(cfg: dict) -> dict:
    """The prices of the run's model, dollars per million tokens (config: openai.prices.<agent_model>)."""
    table = (cfg.get("openai") or {}).get("prices") or {}
    prices = table.get(cfg.get("agent_model")) if isinstance(table, dict) else None
    if not isinstance(prices, dict):
        raise ConfigError(f"openai.prices has no entry for the model {cfg.get('agent_model')!r}: without its prices "
                          "(dollars per million tokens: input, cached_input, output) the run's budget cannot be counted")
    return prices


def check_openai(cfg: dict) -> None:
    """The openai block of a run billed through OpenAI's API: where it is, and the prices its budget is counted at."""
    opts = cfg.get("openai")
    defaults = read_yaml(DEFAULTS)["openai"]
    if not isinstance(opts, dict) or sorted(set(opts) - set(defaults)):
        raise ConfigError(f"openai must be a mapping with the options {sorted(defaults)}")
    if not isinstance(opts.get("base_url"), str) or not opts["base_url"].startswith(("https://", "http://127.0.0.1:", "http://localhost:")):
        raise ConfigError("openai.base_url must be an https address (or a stand-in on this machine)")
    if not isinstance(opts.get("request"), dict):
        raise ConfigError("openai.request must be a mapping of request fields (empty for none)")
    held = sorted(set(opts["request"]) & {"model", "input", "tools", "stream", "store", "include"})
    if held:
        raise ConfigError(f"openai.request cannot set {held}: the backend sets them")
    idle = opts.get("idle_seconds")
    if isinstance(idle, bool) or not isinstance(idle, (int, float)) or not idle > 0:
        raise ConfigError("openai.idle_seconds must be a positive number")

    def priced(block, where, need):
        for key in need:
            v = block.get(key)
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0 or (key in ("input", "output") and v == 0):
                raise ConfigError(f"{where}.{key} must be a price in dollars per million tokens, not {v!r}")

    prices = openai_prices(cfg)
    where = f"openai.prices.{cfg['agent_model']}"
    unknown = sorted(set(prices) - {"input", "cached_input", "cache_write", "output", "long_context"})
    if unknown:
        raise ConfigError(f"{where} has unknown keys {unknown}")
    priced(prices, where, ["input", "output"] + [k for k in ("cached_input", "cache_write") if k in prices])
    long = prices.get("long_context")
    if long is not None:
        if not isinstance(long, dict) or sorted(set(long) - {"above_input_tokens", "input", "cached_input", "cache_write", "output"}):
            raise ConfigError(f"{where}.long_context must be a mapping with above_input_tokens, input, output (and cached_input, cache_write)")
        n = long.get("above_input_tokens")
        if isinstance(n, bool) or not isinstance(n, int) or n < 1:
            raise ConfigError(f"{where}.long_context.above_input_tokens must be a positive whole number of tokens")
        priced(long, where + ".long_context", ["input", "output"] + [k for k in ("cached_input", "cache_write") if k in long])


def check_continuation(cfg: dict) -> None:
    """continue_from names an earlier run; the continuation block sets how the new price and sample follow from it
    (level1/continuation.py). What the run's start works out is not the config's to set."""
    earlier = cfg.get("continue_from")
    if earlier is not None and (not isinstance(earlier, str) or not earlier or "/" in earlier or earlier in (".", "..")):
        raise ConfigError("continue_from must be the run_id of a run in runs_dir, or null")
    opts = cfg.get("continuation")
    defaults = read_yaml(DEFAULTS)["continuation"]
    if not isinstance(opts, dict) or sorted(set(opts) - set(defaults)):
        raise ConfigError(f"continuation must be a mapping with the options {sorted(defaults)}")
    if not isinstance(opts["memory"], bool):
        raise ConfigError("continuation.memory must be true or false")

    def number(key, low, high=math.inf, low_open=True):
        v = opts[key]
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v > high or v < low or (low_open and v == low):
            raise ConfigError(f"continuation.{key} must be a number {'above' if low_open else 'from'} {low:g}"
                              + ("" if high == math.inf else f" up to {high:g}"))

    number("c_max", 0.0, 1.0)
    number("margin", 1.0, low_open=False)
    number("inputs_per_alpha", 0.0)
    number("max_sample_size", 1.0, low_open=False)
    if opts["alpha"] is not None:
        number("alpha", 0.0)


def check_orchestration(cfg: dict) -> None:
    """Validate the optional research loop without changing old runs' reward or measurement policy."""
    options = cfg.get("orchestration") or {}
    defaults = read_yaml(DEFAULTS).get("orchestration", {})
    if not isinstance(options, dict):
        raise ConfigError("orchestration must be a mapping")
    unknown = sorted(set(options) - set(defaults))
    if unknown:
        raise ConfigError(f"unknown orchestration options: {unknown}")
    for key in ("enabled", "baseline_agent"):
        if not isinstance(options.get(key, False), bool):
            raise ConfigError(f"orchestration.{key} must be true or false")
    selected = options.get("mode")
    if selected is not None and (not isinstance(selected, str) or selected not in research_modes.MODES):
        raise ConfigError("orchestration.mode must be trained, baseline, both or null")
    if selected is not None and not options.get("enabled"):
        raise ConfigError("orchestration.mode requires orchestration.enabled: true")
    if not options.get("enabled"):
        return
    mode = research_modes.mode(cfg)
    version = options.get("baseline_validation_version")
    if "baseline_validation_version" in options and (isinstance(version, bool) or not isinstance(version, int) or version != 2):
        raise ConfigError("new research configurations require orchestration.baseline_validation_version: 2")
    baseline = options.get("baseline_entry")
    if baseline is not None and (not isinstance(baseline, str) or not baseline.strip()):
        raise ConfigError("orchestration.baseline_entry must be a path or null")
    if selected == "trained" and baseline is not None:
        raise ConfigError("orchestration.baseline_entry conflicts with explicit trained mode")
    automatic_baseline = research_modes.generates_baseline(cfg)
    if mode != "baseline":
        baseline_fraction = options.get("baseline_fraction", 0.0)
        if (isinstance(baseline_fraction, bool) or not isinstance(baseline_fraction, (int, float))
                or not math.isfinite(baseline_fraction) or not 0 <= baseline_fraction < 1):
            raise ConfigError("orchestration.baseline_fraction must be a finite number from 0 up to but not including 1")
        if automatic_baseline and baseline_fraction <= 0:
            raise ConfigError("baseline_agent needs a positive baseline_fraction reserved within the run budget")
    baseline_fraction = research_modes.baseline_fraction(cfg)
    if automatic_baseline:
        baseline_submissions = options.get("baseline_submissions")
        if isinstance(baseline_submissions, bool) or not isinstance(baseline_submissions, int) or baseline_submissions < 2:
            raise ConfigError("orchestration.baseline_submissions must allow at least two submissions for a checkpoint and an improvement")
    checkpoint_calls = options.get("checkpoint_python_calls")
    if checkpoint_calls is not None and (isinstance(checkpoint_calls, bool) or not isinstance(checkpoint_calls, int) or checkpoint_calls < 1):
        raise ConfigError("orchestration.checkpoint_python_calls must be a positive whole number or null")
    for key in ("python_calls", "max_turns", "submissions"):
        value = cfg["session"][key]
        minimum = 0 if mode == "baseline" and key == "submissions" else 1
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            requirement = "a nonnegative whole number" if minimum == 0 else "a positive whole number"
            raise ConfigError(f"session.{key} must be {requirement} in research mode")
    seconds = cfg["session"]["seconds"]
    if isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or seconds <= 0:
        raise ConfigError("session.seconds must be a positive finite number")
    if automatic_baseline:
        for key in ("python_calls", "max_turns"):
            if math.floor(cfg["session"][key] * cfg["rounds_max"] * baseline_fraction) < 1:
                raise ConfigError(f"the baseline allocation must allow at least one session.{key}")
    seed = options.get("development_seed")
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int) or seed < 0):
        raise ConfigError("orchestration.development_seed must be a valid nonnegative integer, or null")
    if seed is None and mode != "trained":
        raise ConfigError("orchestration.development_seed: null (a fresh sample for every submission) needs "
                          "orchestration.mode: trained; a baseline comparison is scored on one fixed sample")
    from .regions import normalize_scope

    # Every active stage uses the original global input distribution.
    if cfg["verifier"].get("scope") is None:
        cfg["verifier"]["scope"] = {"kind": "all"}
    elif normalize_scope(cfg["verifier"]["scope"]) != {"kind": "all"}:
        raise ConfigError("the orchestration's global task must use scope kind all; put narrower scopes in initial_tasks")
    cfg["verifier"]["sample_seed"] = seed
    if mode == "baseline":
        return  # workers, research fractions and synthesis have no stage in this mode
    for key in ("parallel", "final_summary"):
        if not isinstance(options.get(key, False), bool):
            raise ConfigError(f"orchestration.{key} must be true or false")
    workers = options.get("workers")
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ConfigError("orchestration.workers must be a positive whole number")
    regional_submissions = options.get("regional_submissions", 1)
    if isinstance(regional_submissions, bool) or not isinstance(regional_submissions, int) or regional_submissions < 1:
        raise ConfigError("orchestration.regional_submissions must be a positive whole number")
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
    if summary + baseline_fraction >= 1:
        raise ConfigError("baseline_fraction + summary_fraction must leave a positive share for research")
    research_share = 1.0 - summary - baseline_fraction
    python_share = 1.0 - baseline_fraction
    planning, regional = fractions
    global_fraction = 1 - planning - regional
    for key in ("python_calls", "max_turns"):
        amount = cfg["session"][key] * (research_share if key == "max_turns" else python_share)
        if math.floor(amount * regional) // workers < 1 or math.floor(amount * global_fraction) < 1:
            raise ConfigError(f"session.{key} is too small after sharing its allocation between regional and global integration roles")
    if math.floor(cfg["session"]["max_turns"] * cfg["rounds_max"] * planning * research_share) < 1:
        raise ConfigError("the planning allocation must allow at least one turn")
    if automatic_baseline and math.floor(cfg["session"]["python_calls"] * cfg["rounds_max"] * planning * python_share) < 1:
        raise ConfigError("the planning allocation must allow at least one Python call after reserving the baseline")
    if summary and math.floor(cfg["session"]["max_turns"] * cfg["rounds_max"] * summary) < 1:
        raise ConfigError("the final summary allocation must allow at least one turn")
    for key in ("archive_per_scope",):
        value = options.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < (1 if key == "archive_per_scope" else 0):
            raise ConfigError(f"orchestration.{key} must be a valid nonnegative integer (archive_per_scope must be positive)")
    if int(cfg["session"]["submissions"]) <= workers * regional_submissions:
        raise ConfigError("session.submissions must leave a submission for global integration after "
                          "workers * regional_submissions; increase submissions or lower regional_submissions")
    if int(cfg["session"]["python_calls"]) <= workers:
        raise ConfigError("session.python_calls must leave exploration calls for global integration")
    tasks = options.get("initial_tasks", [])
    if not isinstance(tasks, list):
        raise ConfigError("orchestration.initial_tasks must be a list of task mappings")
    for assignment in tasks:
        if not isinstance(assignment, dict) or "task" not in assignment or not set(assignment) <= {"scope", "task"}:
            raise ConfigError("each initial task needs a task and at most a scope")
        if not isinstance(assignment["task"], str) or not assignment["task"].strip():
            raise ConfigError("each initial task needs a nonempty task")
        try:
            assignment["scope"] = normalize_scope(assignment.get("scope") or {"kind": "all"})
        except (ValueError, RuntimeError) as exc:
            raise ConfigError(f"invalid initial task: {exc}") from None
    initial = options.get("initial_regions")
    if not isinstance(initial, list):
        raise ConfigError("orchestration.initial_regions must be a list of scope/task mappings")
    for assignment in initial:
        if not isinstance(assignment, dict) or set(assignment) != {"scope", "task"}:
            raise ConfigError("each initial region needs only scope and task")
        if not isinstance(assignment["task"], str) or not assignment["task"].strip():
            raise ConfigError("each initial region needs a nonempty task")
        try:
            assignment["scope"] = normalize_scope(assignment["scope"])
        except (ValueError, RuntimeError) as exc:
            raise ConfigError(f"invalid initial region: {exc}") from None


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
