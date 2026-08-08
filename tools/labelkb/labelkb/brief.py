"""The justification sheet that ships with every generated design.

The design says *what*; the brief says *why* — element by element, each row
citing either precedent jobs (with counts) or a placeholder that a human must
fill. It is deliberately the same element list that drew the SVG/PDF, so the
two cannot drift apart.
"""

from __future__ import annotations

import json

from .config import Config
from .design import Design


def _evidence_lines(design: Design, config: Config) -> list[str]:
    path = config.kb / "rules-evidence.json"
    if not path.exists():
        return []
    evidence = json.loads(path.read_text(encoding="utf-8"))
    regime_rules = evidence.get("allopathic") or []
    lines = []
    for rule in regime_rules:
        if not rule["total"]:
            continue
        lines.append(
            f"- {rule['statement']}: **{rule['support']}/{rule['total']}** approved "
            f"labels (e.g. jobs {', '.join(rule['example_jobs'][:3])})"
        )
    return lines


def build_brief(design: Design, config: Config | None = None) -> str:
    config = config or Config.load()
    header = [
        f"# Label design brief — {design.generic_display}",
        "",
        f"> {design.disclaimer}",
        "",
        "| | |",
        "|---|---|",
        f"| Generic | {design.generic_display} |",
        f"| Dosage form | {design.dosage_form or 'CONFIRM'} |",
        f"| Die size | {design.die_w_mm:g} × {design.die_h_mm:g} mm |",
        f"| Precedent | {design.status} — jobs {', '.join(design.precedent_jobs) or 'none'} |",
        f"| Layout archetype | {design.archetype_key or 'default grid'} "
        f"(n={design.archetype_n}) |",
        f"| Inks | {', '.join(design.inks)} |",
        f"| Substrate | {design.substrate or 'CONFIRM'} |",
        "",
    ]

    if design.confirm_items:
        header.append("## ⚠ CONFIRM before release")
        header.append("")
        for item in design.confirm_items:
            header.append(f"- [ ] {item}")
        header.append("")

    body = [
        "## Elements",
        "",
        "| Element | Content | Zone (mm) | Size (mm) | Source |",
        "|---|---|---|---|---|",
    ]
    for element in design.elements:
        if element.kind == "dieline":
            continue
        x0, y0, x1, y1 = element.bbox_mm
        zone = f"{x0:g},{y0:g} → {x1:g},{y1:g}"
        size = f"{element.font_mm:g}" if element.font_mm else "—"
        content = (element.text or "—").replace("|", "\\|")
        flag = " ⚠" if element.confirm else ""
        body.append(
            f"| {element.role}{flag} | {content} | {zone} | {size} | {element.source} |"
        )
    body.append("")

    evidence = _evidence_lines(design, config)
    if evidence:
        body.append("## What approved labels in this regime carry")
        body.append("")
        body.extend(evidence)
        body.append("")

    body += [
        "## Statutory references",
        "",
        "- D&C Rules 96/97 labelling particulars — `VERIFY` against the current bare act.",
        "- Schedule H / H1 / X box wording and Rx symbol — `VERIFY`.",
        "- DPCO/MRP declaration and Legal Metrology particulars — `VERIFY`.",
        "",
        "Corpus evidence above is observational: it states what this printer's",
        "approved labels actually carry, with job numbers to check against.",
        "Statutory citations are reminders for a human, not legal assertions.",
        "",
    ]
    return "\n".join(header + body)
