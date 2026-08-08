"""Write a generated design to files: SVG (canonical), PDF, brief, DOCX.

DOCX only if python-docx is present (the [export] extra); its absence costs
the .docx file, nothing else.
"""

from __future__ import annotations

import re
from pathlib import Path

from .brief import build_brief
from .config import Config
from .design import Design, to_pdf, to_svg


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:60]


def export_design(design: Design, out_dir: Path, config: Config | None = None) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = _slug(design.query) or "design"

    svg_path = out_dir / f"{stem}.svg"
    svg_path.write_text(to_svg(design), encoding="utf-8")

    pdf_path = out_dir / f"{stem}.pdf"
    to_pdf(design, pdf_path)

    brief_md = build_brief(design, config)
    md_path = out_dir / f"{stem}-brief.md"
    md_path.write_text(brief_md, encoding="utf-8")

    json_path = out_dir / f"{stem}.json"
    json_path.write_text(design.model_dump_json(indent=2), encoding="utf-8")

    written = {
        "svg": str(svg_path),
        "pdf": str(pdf_path),
        "brief_md": str(md_path),
        "design_json": str(json_path),
    }

    docx_path = _try_docx(brief_md, out_dir / f"{stem}-brief.docx")
    if docx_path:
        written["brief_docx"] = str(docx_path)
    return written


def _try_docx(markdown: str, out_path: Path) -> Path | None:
    try:
        import docx
    except ImportError:
        return None

    document = docx.Document()
    table = None
    for line in markdown.splitlines():
        stripped = line.strip()
        if not stripped:
            table = None
            continue
        if stripped.startswith("# "):
            document.add_heading(stripped[2:], level=1)
        elif stripped.startswith("## "):
            document.add_heading(stripped[3:], level=2)
        elif stripped.startswith("> "):
            document.add_paragraph(stripped[2:], style="Intense Quote")
        elif re.fullmatch(r"\|[-| ]+\|", stripped):
            continue  # separator row
        elif stripped.startswith("|"):
            cells = [c.strip() for c in stripped.strip("|").split("|")]
            if table is None or len(table.columns) != len(cells):
                table = document.add_table(rows=0, cols=len(cells))
                table.style = "Light Grid Accent 1"
            row = table.add_row()
            for cell, text in zip(row.cells, cells):
                cell.text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
        elif stripped.startswith(("- ", "- [ ] ")):
            document.add_paragraph(
                stripped.removeprefix("- [ ] ").removeprefix("- "), style="List Bullet"
            )
        else:
            document.add_paragraph(re.sub(r"\*\*(.+?)\*\*", r"\1", stripped))
    document.save(out_path)
    return out_path
