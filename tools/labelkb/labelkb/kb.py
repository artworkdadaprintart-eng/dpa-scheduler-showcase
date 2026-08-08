"""Aggregate LabelRecords into the knowledge base CREATE draws from.

Three indexes, three questions answered:

* ``generics.json``  — "what do approved labels for THIS molecule look like?"
* ``archetypes.json`` — "how is a label of THIS form at THIS die size laid out?"
* ``rules-evidence.json`` — "what does an approved label in THIS regime always
  carry?" — with counts and job numbers, never bare assertions.

Corrections: ``corrections.jsonl`` holds human fixes from the review queue,
``{"job_no": ..., "field": "brand", "value": "..."}``, applied over records at
load time. The extraction output is never edited in place — corrections are a
visible overlay, so what the machine said and what a human fixed stay separable
(that separation *is* the future fine-tuning dataset).
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median

from .config import Config
from .extract import LabelRecord
from .normalize import generic_key, parse_composition, parse_query


def _set_path(data: dict, dotted: str, value):
    keys = dotted.split(".")
    target = data
    for key in keys[:-1]:
        target = target.setdefault(key, {})
    target[keys[-1]] = value


def load_records(config: Config) -> list[LabelRecord]:
    """All extraction records, with human corrections applied on top."""
    corrections: dict[str, list[dict]] = defaultdict(list)
    corrections_path = config.kb / "corrections.jsonl"
    if corrections_path.exists():
        for line in corrections_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                entry = json.loads(line)
                corrections[str(entry["job_no"])].append(entry)

    records = []
    for path in sorted(config.records_dir.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        for fix in corrections.get(str(data.get("job_no")), []):
            _set_path(data, fix["field"], fix["value"])
        records.append(LabelRecord.model_validate(data))
    return records


def _record_key(record: LabelRecord) -> str:
    """KB key for a record: parsed composition first, generic line second."""
    ingredients = []
    for line in record.composition_raw:
        ingredients.extend(parse_composition(line))
    if not ingredients and record.generic_name:
        ingredients = parse_composition(record.generic_name)
    return generic_key(ingredients)


def _die_key(record: LabelRecord) -> str | None:
    spec = record.print_spec
    if not (spec.die_w_mm and spec.die_h_mm):
        return None
    return f"{round(spec.die_w_mm):g}x{round(spec.die_h_mm):g}"


def build_generics(records: list[LabelRecord]) -> dict:
    generics: dict[str, dict] = {}
    for record in records:
        key = _record_key(record)
        if not key:
            continue
        entry = generics.setdefault(key, {"key": key, "products": []})
        entry["products"].append(
            {
                "job_no": record.job_no,
                "customer": record.customer,
                "brand": record.brand,
                "generic_text": record.generic_name,
                "composition_raw": record.composition_raw,
                "dosage_form": record.dosage_form,
                "pack_size": record.pack_size,
                "regime": record.regime,
                "rx_status": record.compliance.rx_status,
                "red_band": record.compliance.red_band,
                "statutory_warnings": record.compliance.statutory_warnings,
                "storage": record.compliance.storage,
                "die": _die_key(record),
                "prominence_ratio": record.compliance.prominence.ratio,
                "inks": record.print_spec.inks,
                "substrate": record.print_spec.substrate,
            }
        )
    for entry in generics.values():
        statuses = [p["rx_status"] for p in entry["products"]]
        entry["n"] = len(entry["products"])
        entry["rx_status"] = Counter(statuses).most_common(1)[0][0]
        entry["red_band_n"] = sum(1 for p in entry["products"] if p["red_band"])
    return generics


def build_archetypes(records: list[LabelRecord]) -> dict:
    """Median layout per (dosage form, die size).

    Zone medians are computed per role across labels; a role present on fewer
    than half the members is reported with its support so CREATE can treat it
    as optional rather than mandatory.
    """
    groups: dict[str, list[LabelRecord]] = defaultdict(list)
    for record in records:
        die = _die_key(record)
        if record.dosage_form and die:
            groups[f"{record.dosage_form}|{die}"].append(record)

    archetypes: dict[str, dict] = {}
    for key, members in groups.items():
        role_boxes: dict[str, list] = defaultdict(list)
        for record in members:
            seen: set[str] = set()
            for zone in record.zones:
                if zone.role in ("other",) or zone.role in seen:
                    continue  # first zone per role per label keeps medians honest
                seen.add(zone.role)
                role_boxes[zone.role].append((zone.bbox_norm, zone.height_mm))

        zones = {}
        for role, entries in role_boxes.items():
            boxes = [e[0] for e in entries]
            zones[role] = {
                "bbox_norm": [round(median(b[i] for b in boxes), 4) for i in range(4)],
                "height_mm": round(median(e[1] for e in entries), 3),
                "support": len(entries),
            }

        ratios = [
            r.compliance.prominence.ratio
            for r in members
            if r.compliance.prominence.ratio
        ]
        inks = Counter(tuple(r.print_spec.inks) for r in members if r.print_spec.inks)
        substrates = Counter(
            r.print_spec.substrate for r in members if r.print_spec.substrate
        )
        archetypes[key] = {
            "n": len(members),
            "zones": zones,
            "prominence_ratio_median": round(median(ratios), 3) if ratios else None,
            "common_inks": list(inks.most_common(1)[0][0]) if inks else [],
            "common_substrate": substrates.most_common(1)[0][0] if substrates else None,
            "example_jobs": [r.job_no for r in members[:8]],
        }
    return archetypes


def build_rules_evidence(records: list[LabelRecord]) -> dict:
    """Observed regularities, counted. Every entry is 'n of N, e.g. these jobs'
    — the defensible form of a rule, because these labels were approved."""
    by_regime: dict[str, list[LabelRecord]] = defaultdict(list)
    for record in records:
        by_regime[record.regime].append(record)

    def rule(members, statement, predicate, scope=None):
        pool = [m for m in members if scope(m)] if scope else members
        hits = [m for m in pool if predicate(m)]
        return {
            "statement": statement,
            "support": len(hits),
            "total": len(pool),
            "confidence": round(len(hits) / len(pool), 3) if pool else None,
            "example_jobs": [m.job_no for m in hits[:6]],
        }

    evidence: dict[str, list] = {}
    for regime, members in by_regime.items():
        rules = [
            rule(
                members,
                "Statutory warning text present",
                lambda m: bool(m.compliance.statutory_warnings),
            ),
            rule(
                members,
                "Manufacturing licence number present",
                lambda m: bool(m.compliance.licence_no),
            ),
            rule(members, "MRP statement present", lambda m: bool(m.compliance.mrp_text)),
            rule(members, "Storage condition present", lambda m: bool(m.compliance.storage)),
            rule(
                members,
                "Batch/Mfg/Exp block present",
                lambda m: bool(m.compliance.batch_block),
            ),
            rule(
                members,
                "Red band present on Rx labels",
                lambda m: m.compliance.red_band,
                scope=lambda m: m.compliance.rx_status == "rx",
            ),
            rule(
                members,
                "Generic name set smaller than brand (ratio < 1)",
                lambda m: bool(
                    m.compliance.prominence.ratio and m.compliance.prominence.ratio < 1
                ),
                scope=lambda m: m.compliance.prominence.ratio is not None,
            ),
        ]
        evidence[regime] = [r for r in rules if r["total"]]
    return evidence


def build_kb(config: Config | None = None) -> dict:
    config = config or Config.load()
    records = load_records(config)

    generics = build_generics(records)
    archetypes = build_archetypes(records)
    evidence = build_rules_evidence(records)

    config.kb.mkdir(parents=True, exist_ok=True)
    for name, data in [
        ("generics.json", generics),
        ("archetypes.json", archetypes),
        ("rules-evidence.json", evidence),
    ]:
        (config.kb / name).write_text(
            json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    return {
        "records": len(records),
        "generics": len(generics),
        "archetypes": len(archetypes),
        "regimes": sorted(evidence),
    }


def lookup_generic(query_text: str, config: Config | None = None) -> dict:
    """Resolve a query to a profile, or to nearest precedents when unseen.

    Nearest = shares at least one base molecule; then same dosage form. The
    result always says which case it is — CREATE must never silently treat a
    neighbour as the thing itself.
    """
    config = config or Config.load()
    generics_path = config.kb / "generics.json"
    generics = (
        json.loads(generics_path.read_text(encoding="utf-8"))
        if generics_path.exists()
        else {}
    )
    query = parse_query(query_text)

    if query.key and query.key in generics:
        return {"status": "exact", "query": query.__dict__, "profile": generics[query.key]}

    related = []
    for key, entry in generics.items():
        overlap = set(key.split("+")) & set(query.bases)
        if overlap:
            related.append({"overlap": sorted(overlap), **entry})
    if related:
        related.sort(key=lambda e: (-len(e["overlap"]), -e["n"]))
        return {"status": "partial", "query": query.__dict__, "related": related[:5]}

    same_form = []
    if query.dosage_form:
        for entry in generics.values():
            products = [
                p for p in entry["products"] if p["dosage_form"] == query.dosage_form
            ]
            if products:
                same_form.append({**entry, "products": products})
    return {
        "status": "none",
        "query": query.__dict__,
        "same_form": same_form[:5],
    }
