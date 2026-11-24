"""Immutable saved drafts, kept separate from formally submitted candidates."""

from __future__ import annotations

from pathlib import Path

from . import lang
from .session import ENTRY_NAME
from .util import read_json, write_json


def home(path: Path, item: dict) -> Path:
    """The run folder an archived entry's path is relative to: this run's or, for one carried from an earlier run
    that this one goes on from (continue_from), that run's, beside it."""
    return Path(path).parent / item["from_run"] if item.get("from_run") else Path(path)


def collect(path: Path, cfg: dict, archive: dict, plan: dict) -> None:
    """Snapshot content once, recording new names but not unchanged inheritance."""
    drafts = archive.setdefault("drafts", [])
    by_hash = {item["entry_sha256"]: item for item in drafts}
    snapshot = plan.get("snapshot") or {"candidates": [], "global": {"best": None}}
    inherited = {name: item["entry_sha256"] for name, item in aliases(snapshot).items()}
    for item in snapshot["candidates"]:
        if item["selection"].get("status") == "scored":
            inherited[item["id"]] = item["entry_sha256"]
            if item["id"] == snapshot["global"]["best"]:
                inherited["best_global"] = item["entry_sha256"]
    comparison = Path(path) / "multi_agent/baseline_entry.json"
    if comparison.exists():
        inherited["comparison_entry"] = lang.entry_hash(read_json(comparison))
    submitted = {(item["submitted"].get("name"), item["entry_sha256"])
                 for item in archive["candidates"]
                 if item["session_id"] == plan["id"] and item["selection"].get("status") == "scored"}
    workspace = Path(path) / plan["directory"] / "session/workspace/entries"
    for source in sorted(workspace.glob("*.json")):
        if source.is_symlink() or not source.is_file() or not ENTRY_NAME.fullmatch(source.stem):
            continue
        entry = read_json(source)
        digest = lang.entry_hash(entry)
        if inherited.get(source.stem) == digest or (source.stem, digest) in submitted:
            continue  # carried or scored as it is: not a new draft (a carried one may be over this run's limits)
        lang.check_structure(entry, cfg["entry"]["max_nodes"], cfg["entry"]["max_const_scalars"])
        alias = "draft_" + digest[:32]
        if any(item["id"] == alias and item["entry_sha256"] != digest for item in drafts):
            raise RuntimeError("two saved drafts have conflicting immutable aliases")
        relative = f"multi_agent/artifacts/{digest}.json"
        target = Path(path) / relative
        if target.exists():
            if lang.entry_hash(read_json(target)) != digest:
                raise RuntimeError("a saved draft snapshot changed after collection")
        else:
            write_json(target, entry)
        origin = {"session_id": plan["id"], "role": plan["role"], "cycle": plan["cycle"],
                  "saved_name": source.stem, "scope": plan["scope"]}
        if digest not in by_hash:
            item = {"id": alias, "entry_sha256": digest, "entry_path": relative,
                    **origin, "status": "unscored_draft", "origins": []}
            drafts.append(item)
            by_hash[digest] = item
        item = by_hash[digest]
        origins = item.setdefault("origins", _origins(item))
        if origin not in origins:
            origins.append(origin)
            origins.sort(key=lambda value: (value["session_id"], value["saved_name"]))
        item.update(origins[0])
    drafts.sort(key=lambda item: (item["session_id"], item["saved_name"], item["id"]))


def _origins(item: dict) -> list[dict]:
    """Read the initial draft format without rewriting its immutable content."""
    if "origins" in item:
        return item["origins"]
    return [{key: item[key] for key in ("session_id", "role", "cycle", "saved_name", "scope")}]


def aliases(archive: dict, reserved=()) -> dict[str, dict]:
    """Stable content aliases plus original names only when they are unambiguous."""
    drafts = archive.get("drafts", [])
    reserved = {*reserved, "best_global", "comparison_entry", *(c["id"] for c in archive["candidates"])}
    out = {}
    for item in drafts:
        if item["id"] in reserved:
            raise RuntimeError("a draft's immutable alias conflicts with a reserved entry name")
        out[item["id"]] = item
    names = {}
    for item in drafts:
        for origin in _origins(item):
            names.setdefault(origin["saved_name"], {})[item["entry_sha256"]] = item
    for name, items in names.items():
        if len(items) == 1 and name not in reserved and name not in out:
            out[name] = next(iter(items.values()))
    return out


def carried(path: Path, archive: dict, reserved=()) -> dict:
    loaded = {}
    out = {}
    for alias, item in aliases(archive, reserved).items():
        key = item["id"]
        if key not in loaded:
            entry = read_json(home(path, item) / item["entry_path"])
            if lang.entry_hash(entry) != item["entry_sha256"]:
                raise RuntimeError(f"saved draft {key} changed after it was snapshotted")
            loaded[key] = entry
        out[alias] = loaded[key]
    return out


def describe(archive: dict) -> str:
    if not archive.get("drafts"):
        return ""
    available = aliases(archive)
    rows = ["Saved draft artifacts (not formal candidates):",
            "These programs were saved without a formal score at their origin. Their aliases are available in entries. "
            "Measurement claims in notes are unverified here; submit a viable draft to make it eligible for selection."]
    for item in archive["drafts"]:
        sources = "; ".join(f"{origin['saved_name']} by {origin['session_id']} on {origin['scope']}"
                            for origin in _origins(item))
        original_names = sorted({origin["saved_name"] for origin in _origins(item)
                                 if available.get(origin["saved_name"], {}).get("id") == item["id"]})
        alias = "; original-name aliases " + ", ".join(original_names) if original_names else ""
        rows.append(f"{item['id']}: saved as {sources}; unscored draft at those origins{alias}.")
    return "\n".join(rows)
