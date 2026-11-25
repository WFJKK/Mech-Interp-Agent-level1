"""Run configs: config/defaults.yaml, then the files a config extends, then the config itself.

A config may name one file it extends (`extends`, relative to itself); mappings are merged key by key, and
anything else is replaced. One exception: a `reward` block that names another reward than the one it would
be merged into replaces it whole, since the parameters of one reward mean nothing to another.
"""

from __future__ import annotations

import math
from pathlib import Path

from . import budget, rewards
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
    if cfg["rounds_max"] < 1 or cfg["patience"] < 1:
        raise ConfigError("rounds_max and patience must be at least 1")
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
