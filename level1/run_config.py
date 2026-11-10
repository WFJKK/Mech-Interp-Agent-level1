"""Run configs: config/defaults.yaml, then the files a config extends, then the config itself.

A config may name one file it extends (`extends`, relative to itself); mappings are merged key by key, and
anything else is replaced.
"""

from __future__ import annotations

from pathlib import Path

from .util import REPO_ROOT, read_yaml

DEFAULTS = REPO_ROOT / "config" / "defaults.yaml"
REQUIRED = ("run_id", "model", "network")


class ConfigError(ValueError):
    pass


def merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
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
    return cfg


def run_path(cfg: dict) -> Path:
    return Path(str(cfg["runs_dir"])).expanduser().resolve() / str(cfg["run_id"])
