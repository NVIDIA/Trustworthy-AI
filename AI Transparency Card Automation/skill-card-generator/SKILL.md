---
name: "skill-card-generator"
description: "Reads an agent skill's source files and produces a skill card plus a review table. Use when a skill directory exists and a governance card needs to be generated or updated."
license: https://creativecommons.org/licenses/by/4.0/deed.en & https://www.apache.org/licenses/LICENSE-2.0
compatibility: "Any agent that can run Python scripts and write files"
metadata:
  author: "Trustworthy AI Projects <trustworthyaiprojects@nvidia.com>"
  tags:
    - skill-card
    - governance
    - documentation
    - trustworthy-ai
  domain: documentation
---

# Generate Skill Card

**Skill directory to analyze**: $ARGUMENTS

## Overview

Produces a filled skill card from a target skill's source files and the surrounding repo. The card is rendered deterministically from a Jinja template driven by a JSON context you author. A separate review table flags every inferred or human-required field.

## When to use

- A skill directory exists and needs a governance card
- An existing card is out of date after changes to the skill
- Before submitting a skill for legal/safety review

## Workflow

### Step 1 — Resolve the target

If `$ARGUMENTS` is provided, use it. Otherwise default to the current working directory. The target should be a skill directory (typically `<repo>/.agents/skills/<name>/` or `.claude/skills/<name>/`).

### Step 2 — Run the discovery script

```
python3 <skill-dir>/scripts/discover_assets.py <target>
```

The output contains, in order:
- Discovery report (file roles)
- Extracted file contents from the skill directory
- Extracted file contents from the repo root (README, evaluation docs)
- **Structured signal summary** (primary input for your context)
- Style guide (verbatim from `references/style-guide.md`)
- Jinja template (verbatim from `references/skill-card.md.j2`)

All inputs you need are in this one output — do not issue additional Read calls for source files.

### Step 3 — Build the context JSON

Read the style guide and build a context object for this skill. Every field is defined there. Key rules:

- The **signal summary** is your first stop for each field — frontmatter, license, version.
- When the summary doesn't cover a field, read the raw extracted file contents.
- When neither source supports a field, choose the honest default the style guide specifies; use `HUMAN-REQUIRED` placeholders only as a last resort.
- Do not leave `use_case` empty.
- Set `owner.verify: true` whenever ownership is inferred or defaulted (see style guide for when `verify: false` is appropriate). Set `license_verify: true` unless the license identifier was extracted verbatim from a documentation file.

Write the context to a temp file, e.g. `/tmp/<skill-name>-context.json`.

### Step 4 — Render the card

```
python3 <skill-dir>/scripts/render_card.py \
  --context /tmp/<skill-name>-context.json \
  --template <skill-dir>/references/skill-card.md.j2 \
  --out <target>/<skill-name>-skill-card.md
```

The script validates the context against a minimal schema and refuses to render if required fields are missing or typed wrong. Fix reported errors before proceeding.

### Step 5 — Self-verify

Before finishing:
- Cross-field consistency checks from the style guide must pass.
- The rendered card should not contain unrendered `{{ ... }}` or `{% ... %}` fragments.


## Files in this skill

- `SKILL.md` — this file (orchestration)
- `references/style-guide.md` — per-context-field guidance (the substantive instructions)
- `references/skill-card.md.j2` — Jinja template (exact card layout)
- `references/catalog/limitations.json` — canned technical-limitations catalog (buffet list)
- `references/catalog/risks.json` — canned risk-management catalog (buffet list)
- `scripts/discover_assets.py` — discovery + signal extraction
- `scripts/render_card.py` — Jinja renderer with context validation and catalog injection
- `scripts/validate_submission.py` — pre-submission marker validator
