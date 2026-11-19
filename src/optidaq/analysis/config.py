"""Helpers for reading blocks of an analysis config; no other logic."""

from __future__ import annotations

from typing import Any, Mapping, Sequence, Tuple


def strip(block: Any) -> Any:
    """The config without keys that start with "_" (comments), at every level."""

    if isinstance(block, dict):
        return {k: strip(v) for k, v in block.items() if not str(k).startswith("_")}
    if isinstance(block, list):
        return [strip(v) for v in block]
    return block


def fill(block: Mapping, defaults: Mapping, where: str, required: Sequence[str] = ()) -> dict:
    """``block`` with ``defaults`` filled in, one level deep.

    Keys that are not in ``defaults`` and ``required`` keys without a value are
    errors, so a misspelt key cannot be ignored silently.
    """

    unknown = sorted(set(block) - set(defaults))
    out = {**defaults, **block}
    missing = [k for k in required if out.get(k) is None]
    if unknown or missing:
        raise ValueError(
            "%s: unknown keys %s, missing keys %s; known keys: %s" % (where, unknown, missing, sorted(defaults))
        )
    return out


def named(entry) -> Tuple[str, dict]:
    """A pluggable part of a config, written as "name" or as
    {"name": ..., "params": {...}}. Returns (name, params)."""

    if isinstance(entry, str):
        return entry, {}
    if not isinstance(entry, Mapping) or "name" not in entry or set(entry) - {"name", "params"}:
        raise ValueError('expected "name" or {"name": ..., "params": {...}}, got %r' % (entry,))
    return entry["name"], dict(entry.get("params") or {})


def pick(registry: Mapping, name, what: str):
    """``registry[name]``, or an error that lists the known names."""

    if name not in registry:
        raise ValueError("unknown %s %r; known: %s" % (what, name, sorted(registry)))
    return registry[name]
