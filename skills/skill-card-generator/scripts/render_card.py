#!/usr/bin/env python3
"""
render_card.py — Render a skill card from a validated context JSON
using the Jinja template.

Usage:
  python3 render_card.py --context <context.json> \
                         --template <path/to/skill-card.md.j2> \
                         --out <output.md>

The template is in references/skill-card.md.j2. The agent does not
author layout — it only produces the context JSON. Rendering is
deterministic so two identical contexts always produce identical cards.
"""

import argparse
import json
import sys
from pathlib import Path

try:
    from jinja2 import Environment, FileSystemLoader, StrictUndefined
except ImportError:
    print(
        "ERROR: jinja2 not installed. Install with:\n"
        "  pip install jinja2 --break-system-packages",
        file=sys.stderr,
    )
    sys.exit(2)


# ─── Minimal context schema ───────────────────────────────────────────────
# Key: (type, required). required=True means missing key = error.
# Lists can be empty; strings can be "" but must be present.

SCHEMA = {
    "skill_name":                (str,   True),
    "skill_kind":                (str,   True),   # "Agent" or similar
    "description_sentence":      (str,   True),
    "usage_posture":             (str,   True),   # commercial | research_dev | demonstration
    "owner":                     (dict,  True),   # {kind, verify?, verify_reason?, name?, card_link?}
    "license_identifier":        ((str, type(None)), False),
    "license_verify":            (bool,  False),  # True → wrap rendered license in red VERIFY span
    "license_verify_reason":     (str,   False),  # short explanation, shown in HTML comment
    "use_case":                  (str,   True),
    "deployment_geography":      (str,   True),
    "references":                (list,  True),   # [{label, url}]
    "output":                    (dict,  True),   # {types: [str], format, parameters, other_properties}
    "skill_version":             (str,   True),
    "evaluation":                (dict,  False),  # optional: evaluation details
}

VALID_USAGE = {"commercial", "research_dev", "demonstration"}
VALID_OWNER_KINDS = {"nvidia", "third_party"}
EVALUATION_STRING_FIELDS = ("agent", "tasks", "results_markdown")
EVALUATION_METRIC_GROUPS = ("dimensions", "signals")
TESTING_COMPLETED_FIELDS = (
    "agent_red_teaming",
    "network_security",
    "product_security",
)


def validate(ctx: dict) -> list[str]:
    errors = []
    for key, (typ, required) in SCHEMA.items():
        if key not in ctx:
            if required:
                errors.append(f"missing required key: '{key}'")
            continue
        if not isinstance(ctx[key], typ):
            expected = typ if not isinstance(typ, tuple) else " or ".join(t.__name__ for t in typ)
            errors.append(f"'{key}' should be {expected}, got {type(ctx[key]).__name__}")

    if "usage_posture" in ctx and ctx["usage_posture"] not in VALID_USAGE:
        errors.append(
            f"'usage_posture' must be one of {sorted(VALID_USAGE)}, got {ctx['usage_posture']!r}"
        )
    if "owner" in ctx and isinstance(ctx["owner"], dict):
        kind = ctx["owner"].get("kind")
        if kind not in VALID_OWNER_KINDS:
            errors.append(
                f"'owner.kind' must be one of {sorted(VALID_OWNER_KINDS)}, got {kind!r}"
            )
        if kind == "third_party":
            for k in ("name", "card_link"):
                if not ctx["owner"].get(k):
                    errors.append(f"'owner.{k}' required when owner.kind == 'third_party'")

    # Nested shape checks
    if "output" in ctx and isinstance(ctx["output"], dict):
        for k in ("types", "format", "parameters", "other_properties"):
            if k not in ctx["output"]:
                errors.append(f"'output.{k}' missing")
    if "evaluation" in ctx and isinstance(ctx["evaluation"], dict):
        evaluation = ctx["evaluation"]
        for key in EVALUATION_STRING_FIELDS:
            if key in evaluation and not isinstance(evaluation[key], str):
                errors.append(
                    f"'evaluation.{key}' should be str, got "
                    f"{type(evaluation[key]).__name__}"
                )
        if "agents" in evaluation:
            agents = evaluation["agents"]
            if not isinstance(agents, list):
                errors.append(
                    "'evaluation.agents' should be list, got "
                    f"{type(agents).__name__}"
                )
            else:
                for idx, agent in enumerate(agents):
                    if not isinstance(agent, str):
                        errors.append(
                            f"'evaluation.agents[{idx}]' should be str, got "
                            f"{type(agent).__name__}"
                        )
        if "metrics" in evaluation:
            metrics = evaluation["metrics"]
            if isinstance(metrics, str):
                pass
            elif isinstance(metrics, dict):
                for group in EVALUATION_METRIC_GROUPS:
                    if group not in metrics:
                        continue
                    entries = metrics[group]
                    if not isinstance(entries, list):
                        errors.append(
                            f"'evaluation.metrics.{group}' should be list, got "
                            f"{type(entries).__name__}"
                        )
                        continue
                    for idx, entry in enumerate(entries):
                        if not isinstance(entry, dict):
                            errors.append(
                                f"'evaluation.metrics.{group}[{idx}]' should be dict, got "
                                f"{type(entry).__name__}"
                            )
                            continue
                        for field in ("name", "description"):
                            if field not in entry:
                                errors.append(
                                    f"'evaluation.metrics.{group}[{idx}].{field}' missing"
                                )
                            elif not isinstance(entry[field], str):
                                errors.append(
                                    f"'evaluation.metrics.{group}[{idx}].{field}' should be str, got "
                                    f"{type(entry[field]).__name__}"
                                )
            else:
                errors.append(
                    "'evaluation.metrics' should be str or dict, got "
                    f"{type(metrics).__name__}"
                )
        if "testing_completed" in evaluation:
            testing_completed = evaluation["testing_completed"]
            if not isinstance(testing_completed, dict):
                errors.append(
                    "'evaluation.testing_completed' should be dict, got "
                    f"{type(testing_completed).__name__}"
                )
            else:
                for key in TESTING_COMPLETED_FIELDS:
                    if key not in testing_completed:
                        errors.append(f"'evaluation.testing_completed.{key}' missing")
                    elif not isinstance(testing_completed[key], bool):
                        errors.append(
                            f"'evaluation.testing_completed.{key}' should be bool, got "
                            f"{type(testing_completed[key]).__name__}"
                        )
    for item in ctx.get("references", []):
        if not isinstance(item, dict) or "label" not in item or "url" not in item:
            errors.append("each 'references' item needs 'label' and 'url'")
            break
    return errors


def _load_catalog(template_dir: Path, name: str) -> list:
    """Load a canned-entries catalog from references/catalog/<name>.json.

    Missing catalog file is tolerated (returns []) so the renderer still works
    for stripped-down skill directories, but the normal path is that both
    limitations.json and risks.json exist.
    """
    catalog_path = template_dir / "catalog" / f"{name}.json"
    if not catalog_path.exists():
        return []
    try:
        data = json.loads(catalog_path.read_text())
    except json.JSONDecodeError as e:
        print(f"ERROR: catalog {catalog_path} is not valid JSON: {e}", file=sys.stderr)
        sys.exit(4)
    if not isinstance(data, list):
        print(f"ERROR: catalog {catalog_path} must be a JSON array", file=sys.stderr)
        sys.exit(4)
    return data


def _apply_marker_defaults(ctx: dict) -> None:
    """Ensure optional verify-marker fields exist so StrictUndefined doesn't bite."""
    ctx.setdefault("license_verify", False)
    ctx.setdefault("license_verify_reason", "")
    if isinstance(ctx.get("owner"), dict):
        ctx["owner"].setdefault("verify", False)
        ctx["owner"].setdefault("verify_reason", "")


def render(context_path: Path, template_path: Path, out_path: Path) -> None:
    ctx = json.loads(context_path.read_text())
    errors = validate(ctx)
    if errors:
        print("Context validation failed:", file=sys.stderr)
        for e in errors:
            print(f"  - {e}", file=sys.stderr)
        sys.exit(3)

    _apply_marker_defaults(ctx)

    template_dir = template_path.parent

    env = Environment(
        loader=FileSystemLoader(str(template_dir)),
        undefined=StrictUndefined,
        keep_trailing_newline=True,
        trim_blocks=False,
        lstrip_blocks=False,
    )
    tmpl = env.get_template(template_path.name)
    rendered = tmpl.render(**ctx)
    out_path.write_text(rendered)
    print(f"Rendered card: {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--context", required=True, type=Path)
    p.add_argument("--template", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()
    render(args.context, args.template, args.out)


if __name__ == "__main__":
    main()
