"""Run configs: config/defaults.yaml, then the files a config extends, then the config itself.

A config may name one file it extends (`extends`, relative to itself); mappings are merged key by key, and
anything else is replaced. One exception: a `reward` block that names another reward than the one it would
be merged into replaces it whole, since the parameters of one reward mean nothing to another.
"""

from __future__ import annotations

from pathlib import Path

from . import rewards
from .util import REPO_ROOT, read_yaml

DEFAULTS = REPO_ROOT / "config" / "defaults.yaml"
REQUIRED = ("run_id", "model", "network")


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
    return cfg


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
