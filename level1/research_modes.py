"""Select research stages without changing the meaning of saved configurations."""

from __future__ import annotations


MODES = ("trained", "baseline", "both")


def mode(cfg: dict) -> str:
    """An explicit mode wins; absent legacy choices retain their original stages."""
    options = cfg.get("orchestration") or {}
    if not options.get("enabled"):
        return "trained"
    selected = options.get("mode")
    if selected is None:
        return "both" if options.get("baseline_agent") is True or options.get("baseline_entry") else "trained"
    if not isinstance(selected, str) or selected not in MODES:
        raise ValueError("orchestration.mode must be trained, baseline, both or null")
    return selected


def requires_baseline(cfg: dict) -> bool:
    options = cfg.get("orchestration") or {}
    return options.get("enabled") is True and mode(cfg) in ("baseline", "both")


def generates_baseline(cfg: dict) -> bool:
    return requires_baseline(cfg) and not (cfg.get("orchestration") or {}).get("baseline_entry")


def baseline_fraction(cfg: dict) -> float:
    """Only generated programs reserve a paid session; a standalone one gets the full share."""
    if not generates_baseline(cfg):
        return 0.0
    if mode(cfg) == "baseline":
        return 1.0
    return float(cfg["orchestration"].get("baseline_fraction", 0.0))


def validation_version(cfg: dict) -> int:
    """Saved runs without a policy field retain their earlier validation contract."""
    return int((cfg.get("orchestration") or {}).get("baseline_validation_version", 1))
