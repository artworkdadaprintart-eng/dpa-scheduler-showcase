"""Command line for labelkb.

This is the engine. The Claude Code skill and the web app are both meant to
shell out to these commands rather than reimplement anything, so that there is
one behaviour to test and one place a fix has to land.

Every subcommand takes ``--json`` for machine consumption and prints a readable
summary otherwise.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from . import __version__
from .config import DEFAULT_RENDER_DPI, Config
from .corpus.esko_info import parse_info_file
from .corpus.proof import parse_proof
from .corpus.walk import iter_job_folders, load_manifest
from .probe import probe
from .render import has_extractable_text, page_geometry, render_page


def _emit(data, as_json: bool, human: str) -> None:
    print(json.dumps(data, indent=2, ensure_ascii=False) if as_json else human)


def cmd_probe(args) -> int:
    result = probe()
    if args.save:
        config = Config.load()
        config.engine_profile = result.profile
        path = config.save()
        result.warnings.append(f"engine profile saved to {path}")
    _emit(result.to_dict(), args.json, result.summary())
    return 0


def cmd_walk(args) -> int:
    root = Path(args.root).expanduser()
    if not root.is_dir():
        print(f"error: corpus root not found: {root}", file=sys.stderr)
        return 2

    config = Config.load()
    out = Path(args.out).expanduser() if args.out else config.manifest_path

    # One walk, not two: on Google Drive for Desktop the tree is streamed from
    # the cloud, so every extra pass is minutes, not milliseconds. Progress
    # goes to stderr so it never pollutes --json output.
    jobs = []
    for job in iter_job_folders(root):
        jobs.append(job)
        if len(jobs) % 25 == 0:
            print(f"  ... {len(jobs)} job folders found so far", file=sys.stderr)
    count = len(jobs)

    if not args.dry_run:
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as handle:
            for job in jobs:
                handle.write(json.dumps(job.to_dict(), ensure_ascii=False) + "\n")

    customers = Counter(j.customer for j in jobs)
    dies = Counter(
        f"{j.parsed['die_w_mm']:g}x{j.parsed['die_h_mm']:g}"
        for j in jobs
        if j.parsed.get("die_w_mm")
    )
    with_artwork = sum(1 for j in jobs if j.artwork_pdf)
    with_proof = sum(1 for j in jobs if j.proof_pdf)
    with_info = sum(1 for j in jobs if j.info_sidecar)

    summary = {
        "root": str(root),
        "jobs": count,
        "customers": len(customers),
        "with_artwork_pdf": with_artwork,
        "with_proof_pdf": with_proof,
        "with_esko_sidecar": with_info,
        "top_customers": customers.most_common(10),
        "top_die_sizes": dies.most_common(10),
        "manifest": None if args.dry_run else str(out),
    }

    lines = [
        f"root              {root}",
        f"job folders       {count}",
        f"customers         {len(customers)}",
        f"with artwork pdf  {with_artwork}",
        f"with proof pdf    {with_proof}",
        f"with esko sidecar {with_info}",
    ]
    if count and with_artwork < count:
        lines.append(f"\n!  {count - with_artwork} job folders have no artwork PDF")
    if dies:
        lines.append("\ntop die sizes     " + ", ".join(f"{d} ({n})" for d, n in dies.most_common(8)))
    if customers:
        lines.append("top customers     " + ", ".join(f"{c} ({n})" for c, n in customers.most_common(5)))
    lines.append("" if args.dry_run else f"\nmanifest written  {out}")

    _emit(summary, args.json, "\n".join(lines))
    return 0


def cmd_info(args) -> int:
    info = parse_info_file(args.path)
    data = info.to_dict()
    lines = [
        f"creator       {info.creator or '-'}",
        f"dieline       {info.dieline_ard or '-'}",
        f"source        {info.source_path or '-'}",
        f"inks          {', '.join(info.ink_names) or '-'}",
        f"ink types     {', '.join(t or '-' for t in info.ink_types) or '-'}",
        f"layers        {', '.join(info.layers) or '-'}",
        f"printed inks  {info.colour_count}",
    ]
    if not info.strings:
        lines.append("\n!  no strings recovered - is this really an Esko .info sidecar?")
    _emit(data, args.json, "\n".join(lines))
    return 0


def cmd_proof(args) -> int:
    ticket = parse_proof(args.path)
    data = ticket.to_dict()
    width = max((len(k) for k in ticket.fields), default=0)
    lines = [f"{k:<{width}}  {v}" for k, v in sorted(ticket.fields.items())]
    if ticket.die_w_mm:
        lines.append(f"\ndie size      {ticket.die_w_mm:g} x {ticket.die_h_mm:g} mm")
    if ticket.colours:
        lines.append(f"colours       {', '.join(ticket.colours)}")
    if not ticket.fields:
        lines.append(
            "!  no ticket fields matched. If this is a real proof, its template "
            "uses labels labelkb does not know yet - send the layout and they "
            "can be added to _ANCHORS in corpus/proof.py."
        )
    _emit(data, args.json, "\n".join(lines))
    return 0


def cmd_geometry(args) -> int:
    geometry = page_geometry(args.path)
    live_text = has_extractable_text(args.path)
    data = geometry.to_dict() | {"has_extractable_text": live_text}
    lines = [
        f"pages         {geometry.page_count}",
        f"page size     {geometry.width_mm:g} x {geometry.height_mm:g} mm",
        f"live text     {'yes' if live_text else 'no (outlines - vision required)'}",
    ]
    if args.expect:
        try:
            w, h = (float(v) for v in args.expect.lower().split("x"))
        except ValueError:
            print(f"error: --expect must look like 30x19, got {args.expect!r}", file=sys.stderr)
            return 2
        bleed = geometry.bleed_mm(w, h)
        data["matches_expected"] = bleed is not None
        data["bleed_mm_per_side"] = bleed
        if bleed is None:
            verdict = "MISMATCH"
        elif bleed == 0.0:
            verdict = "match"
        else:
            verdict = f"match (die + ~{bleed:g} mm bleed per side)"
        lines.append(f"expected      {w:g} x {h:g} mm -> {verdict}")
    _emit(data, args.json, "\n".join(lines))
    return 0


def cmd_render(args) -> int:
    config = Config.load()
    source = Path(args.path)
    out = Path(args.out) if args.out else config.renders_dir / f"{source.stem}.png"
    render = render_page(source, out, dpi=args.dpi)
    lines = [
        f"written       {render.path}",
        f"size          {render.width_px} x {render.height_px} px at {render.dpi:g} dpi",
        f"scale         {render.px_per_mm:.3f} px/mm",
    ]
    if render.dpi < args.dpi:
        lines.append(f"\n!  dpi reduced from {args.dpi} to keep the bitmap manageable")
    _emit(render.to_dict(), args.json, "\n".join(lines))
    return 0


def cmd_extract(args) -> int:
    from .extract import run_batch

    config = Config.load()
    rows = load_manifest(Path(args.manifest) if args.manifest else config.manifest_path)
    if not rows:
        print("error: manifest is empty - run `labelkb walk` first", file=sys.stderr)
        return 2
    if args.job:
        rows = [r for r in rows if r["job_no"] in set(args.job)]
    result = run_batch(rows, config=config, limit=args.limit)
    lines = [
        f"extracted   {result['extracted']}",
        f"skipped     {result['skipped']} (already current)",
        f"failed      {result['failed']}",
    ]
    for failure in result["failures"][:10]:
        lines.append(f"  !  {failure['job_no']}: {failure['error']}")
    _emit(result, args.json, "\n".join(lines))
    return 0 if result["failed"] == 0 else 1


def cmd_kb_build(args) -> int:
    from .kb import build_kb

    summary = build_kb(Config.load())
    _emit(
        summary,
        args.json,
        "\n".join(
            [
                f"records     {summary['records']}",
                f"generics    {summary['generics']}",
                f"archetypes  {summary['archetypes']}",
                f"regimes     {', '.join(summary['regimes']) or '-'}",
            ]
        ),
    )
    return 0


def cmd_lookup(args) -> int:
    from .kb import lookup_generic

    result = lookup_generic(args.generic, Config.load())
    if args.json:
        _emit(result, True, "")
        return 0
    status = result["status"]
    lines = [f"status      {status}"]
    if status == "exact":
        profile = result["profile"]
        lines.append(f"precedents  {profile['n']} approved label(s)")
        for product in profile["products"][:8]:
            lines.append(
                f"  {product['job_no']}  {product['brand'] or '-':20} "
                f"{product['dosage_form'] or '-':10} {product['die'] or '-':8} "
                f"{product['customer']}"
            )
    elif status == "partial":
        lines.append("no exact precedent; molecules shared with:")
        for entry in result["related"]:
            lines.append(f"  {entry['key']}  (n={entry['n']}, overlap: {', '.join(entry['overlap'])})")
    else:
        lines.append("no precedent for this molecule")
        for entry in result.get("same_form", []):
            lines.append(f"  same form: {entry['key']} (n={entry['n']})")
    print("\n".join(lines))
    return 0


def cmd_design(args) -> int:
    from .design import generate_design
    from .export import export_design

    config = Config.load()
    design = generate_design(args.generic, die=args.die, config=config)
    out_dir = Path(args.out) if args.out else config.kb / "designs"
    written = export_design(design, out_dir, config=config)

    data = {"design": design.model_dump(), "files": written}
    lines = [
        f"generic     {design.generic_display}",
        f"die         {design.die_w_mm:g} x {design.die_h_mm:g} mm",
        f"precedent   {design.status} ({', '.join(design.precedent_jobs[:5]) or 'none'})",
        f"archetype   {design.archetype_key or 'default grid'} (n={design.archetype_n})",
        f"elements    {len(design.elements)}",
    ]
    for item in design.confirm_items:
        lines.append(f"\n⚠  CONFIRM: {item}")
    lines.append("")
    for kind, path in written.items():
        lines.append(f"{kind:12} {path}")
    lines.append(f"\n{design.disclaimer}")
    _emit(data, args.json, "\n".join(lines))
    return 0


def cmd_web(args) -> int:
    try:
        from .web.app import main as run_web
    except ImportError:
        print("error: web extras not installed - pip install -e '.[web]'", file=sys.stderr)
        return 2
    run_web()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="labelkb",
        description="Learn label-artwork conventions and compliance from approved artwork.",
    )
    parser.add_argument("--version", action="version", version=f"labelkb {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add(name, help_text, handler):
        sub = subparsers.add_parser(name, help=help_text)
        sub.add_argument("--json", action="store_true", help="emit JSON")
        sub.set_defaults(handler=handler)
        return sub

    probe_parser = add("probe", "report hardware and pick a vision engine profile", cmd_probe)
    probe_parser.add_argument("--save", action="store_true", help="persist the profile to the KB config")

    walk_parser = add("walk", "scan the corpus and write a job manifest", cmd_walk)
    walk_parser.add_argument("--root", required=True, help=r'corpus root, e.g. "G:\My Drive\Artworks"')
    walk_parser.add_argument("--out", help="manifest path (default: <kb>/jobs.jsonl)")
    walk_parser.add_argument("--dry-run", action="store_true", help="report without writing")

    add("info", "decode an Esko .info sidecar", cmd_info).add_argument("path")
    add("proof", "read the job ticket from an approval PDF", cmd_proof).add_argument("path")

    geometry_parser = add("geometry", "report an artwork PDF's true size in mm", cmd_geometry)
    geometry_parser.add_argument("path")
    geometry_parser.add_argument("--expect", help="expected die size, e.g. 30x19")

    render_parser = add("render", "rasterise an artwork PDF for the vision pipeline", cmd_render)
    render_parser.add_argument("path")
    render_parser.add_argument("--out", help="output PNG path")
    render_parser.add_argument(
        "--dpi", type=int, default=DEFAULT_RENDER_DPI, help=f"default {DEFAULT_RENDER_DPI}"
    )

    extract_parser = add("extract", "run the vision pipeline over the manifest", cmd_extract)
    extract_parser.add_argument("--manifest", help="default: <kb>/jobs.jsonl")
    extract_parser.add_argument("--limit", type=int, help="stop after N new extractions")
    extract_parser.add_argument("--job", action="append", help="only these job numbers")

    add("kb-build", "aggregate records into the knowledge base", cmd_kb_build)

    lookup_parser = add("lookup", "what the corpus knows about a generic", cmd_lookup)
    lookup_parser.add_argument("generic", help='e.g. "pantoprazole 40mg injection"')

    design_parser = add("design", "CREATE a label design draft for a generic", cmd_design)
    design_parser.add_argument("generic", help='e.g. "diclofenac 25mg injection"')
    design_parser.add_argument("--die", help="die size WxH in mm, e.g. 30x19")
    design_parser.add_argument("--out", help="output directory (default: <kb>/designs)")

    add("web", "run the local web app (127.0.0.1:8377)", cmd_web)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.handler(args)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except (ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
