"""Spec-declared per-run overrides: what a caller may vary without editing the harness.

A harness lists the dotted paths a caller may set for one run in its top-level
``overridable`` field. The paths come from a closed allowlist in this module —
prompt text and sampling knobs, nothing that widens what the run may do — so
declaring a path can never hand a caller part of the safety layer. Overrides
apply to an in-memory copy of the spec that is fully re-validated; the
``harness.yaml`` on disk is never written.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from hiveloom.errors import SpecError

#: Overridable paths that name one field.
_FIXED_PATHS = frozenset({"system_prompt", "model.temperature", "model.max_tokens"})
_MODEL_IDENTITY = frozenset({"model", "model.provider", "model.id"})
_PARAMS_RE = re.compile(r"model\.params\.[A-Za-z0-9_-]+\Z")
_STEP_RE = re.compile(r"loop\.steps\.([A-Za-z0-9][A-Za-z0-9_-]*)\.instruction\Z")

#: Top-level fields that are the safety layer or grant capability. They fall
#: outside the allowlist anyway; naming them only makes the error say why.
_SAFETY_ROOTS = frozenset(
    {
        "guardrails",
        "logging",
        "egress",
        "confinement",
        "extensions",
        "hooks",
        "mcp_servers",
        "tools",
        "verify",
        "evolution",
        "memory",
        "overridable",
        "id",
    }
)


def is_overridable_path(path: str) -> bool:
    """Whether ``path`` is a shape a harness may declare overridable."""
    return path in _FIXED_PATHS or bool(_PARAMS_RE.fullmatch(path) or _STEP_RE.fullmatch(path))


def check_overridable_paths(paths: list[str]) -> list[str]:
    """Validate a spec's ``overridable`` list, raising ``ValueError`` on a bad entry."""
    for path in paths:
        if is_overridable_path(path):
            continue
        root = path.split(".", 1)[0]
        if root in _SAFETY_ROOTS or path in _MODEL_IDENTITY:
            raise ValueError(
                f"'{path}' can never be overridable: it is part of the safety layer "
                "or selects the executor (use --model/--provider for that)"
            )
        raise ValueError(
            f"'{path}' is not an overridable path (allowed: system_prompt, "
            "model.temperature, model.max_tokens, model.params.<key>, "
            "loop.steps.<step_id>.instruction)"
        )
    if len(set(paths)) != len(paths):
        raise ValueError("overridable lists a path more than once")
    return paths


def override_digests(overrides: dict[str, Any]) -> dict[str, str]:
    """A short hash per applied value: provenance without echoing the value."""
    return {
        path: hashlib.sha256(
            json.dumps(value, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()[:16]
        for path, value in sorted(overrides.items())
    }


def apply_overrides(spec, overrides: dict[str, Any] | None):
    """Return a re-validated in-memory spec with ``overrides`` applied.

    Every path must be listed in ``spec.overridable``; an unlisted path or a
    value the schema rejects raises :class:`SpecError`.
    """
    if not overrides:
        return spec

    from hiveloom.spec.loader import spec_from_dict, spec_to_dict

    unlisted = sorted(set(overrides) - set(spec.overridable))
    if unlisted:
        declared = ", ".join(spec.overridable) or "(none)"
        raise SpecError(
            f"not overridable: {', '.join(unlisted)}. The harness declares: {declared}"
        )
    data = spec_to_dict(spec)
    for path, value in overrides.items():
        step = _STEP_RE.fullmatch(path)
        if step:
            target = next(
                (
                    item
                    for item in data.get("loop", {}).get("steps", [])
                    if isinstance(item, dict) and item.get("id") == step.group(1)
                ),
                None,
            )
            if target is None:
                raise SpecError(f"no structured loop step with id '{step.group(1)}'")
            target["instruction"] = value
        elif path.startswith("model."):
            model = data["model"]
            if path.startswith("model.params."):
                model = model.setdefault("params", {})
            model[path.rsplit(".", 1)[1]] = value
        else:
            data[path] = value
    return spec_from_dict(data, source="run overrides")
