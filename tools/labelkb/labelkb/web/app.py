"""Local web app: search the KB, generate designs, review extractions.

Run with:  labelkb web   (or: uvicorn labelkb.web.app:app)

Design choices, deliberate:
- Server-rendered HTML, zero external assets - it must work on a studio
  machine with no internet and never leak a request anywhere.
- Every page is a thin skin over the same engine the CLI uses; nothing here
  has behaviour of its own.
- The review queue is the learning loop: corrections land in
  kb/corrections.jsonl, which `kb-build` overlays on the records - and which
  doubles as the labelled dataset if a LoRA is ever trained.
"""

from __future__ import annotations

import html
import json
from pathlib import Path

from fastapi import FastAPI, Form
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse

from ..config import Config
from ..design import generate_design
from ..export import export_design
from ..kb import load_records, lookup_generic

app = FastAPI(title="labelkb", docs_url=None, redoc_url=None)

_PAGE = """<!doctype html>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>labelkb</title>
<style>
  body {{ font: 15px/1.5 system-ui, sans-serif; margin: 2rem auto; max-width: 60rem;
         padding: 0 1rem; color: #1a1a1a; }}
  h1 a {{ color: inherit; text-decoration: none; }}
  input[type=text] {{ font: inherit; padding: .4rem .6rem; width: 24rem; max-width: 90%; }}
  button {{ font: inherit; padding: .4rem .9rem; cursor: pointer; }}
  table {{ border-collapse: collapse; width: 100%; margin: 1rem 0; }}
  th, td {{ border: 1px solid #ccc; padding: .35rem .6rem; text-align: left;
            vertical-align: top; }}
  th {{ background: #f2f2f2; }}
  .warn {{ background: #fff3cd; border: 1px solid #e0c060; padding: .6rem .9rem;
           border-radius: 4px; margin: .8rem 0; }}
  .muted {{ color: #666; }}
  .disclaimer {{ font-size: .85em; color: #8a1f1f; margin-top: 2rem;
                 border-top: 1px solid #ddd; padding-top: .8rem; }}
</style>
<h1><a href="/">labelkb</a></h1>
{body}
<p class="disclaimer">Design aid, not regulatory sign-off - a qualified person
must verify and approve every artwork. Local use only: this knowledge base is
customer data.</p>
"""


def _page(body: str) -> HTMLResponse:
    return HTMLResponse(_PAGE.format(body=body))


def _e(value) -> str:
    return html.escape(str(value if value is not None else "—"))


@app.get("/", response_class=HTMLResponse)
def index():
    config = Config.load()
    generics_path = config.kb / "generics.json"
    generics = (
        json.loads(generics_path.read_text(encoding="utf-8"))
        if generics_path.exists()
        else {}
    )
    records = len(list(config.records_dir.glob("*.json"))) if config.records_dir.exists() else 0
    return _page(f"""
      <form action="/lookup" method="get">
        <input type="text" name="q" placeholder="e.g. pantoprazole 40mg injection" required>
        <button>Look up</button>
      </form>
      <form action="/design" method="post" style="margin-top:.6rem">
        <input type="text" name="q" placeholder="generic to design" required>
        <input type="text" name="die" placeholder="die WxH (optional)" style="width:9rem">
        <button>CREATE design</button>
      </form>
      <p class="muted">{records} label records &middot; {len(generics)} generics
      &middot; <a href="/review">review queue</a></p>
    """)


@app.get("/lookup", response_class=HTMLResponse)
def lookup(q: str):
    result = lookup_generic(q, Config.load())
    status = result["status"]
    rows = ""
    products = []
    if status == "exact":
        products = result["profile"]["products"]
    elif status == "partial":
        products = [p for entry in result["related"] for p in entry["products"]]
    for product in products[:25]:
        rows += (
            f"<tr><td>{_e(product['job_no'])}</td><td>{_e(product['brand'])}</td>"
            f"<td>{_e(product['dosage_form'])}</td><td>{_e(product['die'])}</td>"
            f"<td>{_e(product['customer'])}</td><td>{_e(product['rx_status'])}</td></tr>"
        )
    table = (
        f"<table><tr><th>Job</th><th>Brand</th><th>Form</th><th>Die</th>"
        f"<th>Customer</th><th>Rx</th></tr>{rows}</table>"
        if rows
        else "<p class='muted'>No precedent labels.</p>"
    )
    return _page(f"""
      <h2>{_e(q)}</h2>
      <p>Precedent status: <strong>{_e(status)}</strong></p>
      {table}
      <form action="/design" method="post">
        <input type="hidden" name="q" value="{_e(q)}">
        <button>CREATE design from this</button>
      </form>
    """)


@app.post("/design", response_class=HTMLResponse)
def design(q: str = Form(...), die: str = Form(default="")):
    config = Config.load()
    result = generate_design(q, die=die.strip() or None, config=config)
    files = export_design(result, config.kb / "designs", config=config)

    confirms = "".join(f"<div class='warn'>⚠ {_e(item)}</div>" for item in result.confirm_items)
    links = "".join(
        f"<li><a href='/file?path={_e(path)}'>{_e(Path(path).name)}</a></li>"
        for path in files.values()
    )
    rows = "".join(
        f"<tr><td>{_e(el.role)}</td><td>{_e(el.text)}</td>"
        f"<td>{_e(el.font_mm)}</td><td>{_e(el.source)}</td></tr>"
        for el in result.elements
        if el.kind != "dieline"
    )
    return _page(f"""
      <h2>Design draft — {_e(result.generic_display)}</h2>
      <p>Die {result.die_w_mm:g} × {result.die_h_mm:g} mm &middot;
         precedent: {_e(result.status)}
         ({_e(', '.join(result.precedent_jobs[:6]) or 'none')}) &middot;
         archetype {_e(result.archetype_key or 'default grid')}</p>
      {confirms}
      <ul>{links}</ul>
      <table><tr><th>Element</th><th>Content</th><th>Size mm</th><th>Source</th></tr>
      {rows}</table>
    """)


@app.get("/file")
def file(path: str):
    config = Config.load()
    resolved = Path(path).resolve()
    if not str(resolved).startswith(str(config.kb.resolve())):
        return HTMLResponse("outside kb", status_code=403)
    return FileResponse(resolved)


@app.get("/review", response_class=HTMLResponse)
def review():
    config = Config.load()
    records = load_records(config)
    flagged = [
        r
        for r in records
        if (r.provenance.ocr_mean_confidence or 1.0) < 0.9
        or r.provenance.role_disagreements
        or r.provenance.notes
    ]
    flagged.sort(key=lambda r: r.provenance.ocr_mean_confidence or 0)
    rows = "".join(
        f"<tr><td>{_e(r.job_no)}</td><td>{_e(r.brand)}</td>"
        f"<td>{_e(r.provenance.ocr_mean_confidence)}</td>"
        f"<td>{len(r.provenance.role_disagreements)}</td>"
        f"<td>{_e('; '.join(r.provenance.notes))}</td>"
        f"""<td><form action="/correct" method="post">
            <input type="hidden" name="job_no" value="{_e(r.job_no)}">
            <input type="text" name="field" placeholder="field e.g. brand" required>
            <input type="text" name="value" placeholder="corrected value" required>
            <button>fix</button></form></td></tr>"""
        for r in flagged[:50]
    )
    return _page(f"""
      <h2>Review queue</h2>
      <p class="muted">{len(flagged)} of {len(records)} records flagged
      (low OCR confidence, engine disagreement, or geometry notes).
      Corrections overlay the records at kb-build time and never edit the
      extraction files.</p>
      <table><tr><th>Job</th><th>Brand</th><th>OCR conf</th><th>Disagreements</th>
      <th>Notes</th><th>Correct a field</th></tr>{rows}</table>
    """)


@app.post("/correct")
def correct(job_no: str = Form(...), field: str = Form(...), value: str = Form(...)):
    config = Config.load()
    config.kb.mkdir(parents=True, exist_ok=True)
    with (config.kb / "corrections.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps({"job_no": job_no, "field": field, "value": value}) + "\n"
        )
    return RedirectResponse("/review", status_code=303)


def main() -> None:  # pragma: no cover - thin runner
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8377)


if __name__ == "__main__":  # pragma: no cover
    main()
