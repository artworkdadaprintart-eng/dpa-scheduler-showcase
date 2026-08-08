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
from .corpus.walk import build_manifest, iter_job_folders
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

    if args.dry_run:
        jobs = list(iter_job_folders(root))
        count = len(jobs)
    else:
        count = build_manifest(root, out)
        jobs = list(iter_job_folders(root))

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
        ok = geometry.matches(w, h)
        data["matches_expected"] = ok
        lines.append(f"expected      {w:g} x {h:g} mm -> {'match' if ok else 'MISMATCH'}")
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
