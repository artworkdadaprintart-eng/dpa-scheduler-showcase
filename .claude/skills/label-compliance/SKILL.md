---
name: label-compliance
description: >
  Design pharmaceutical label artwork and answer compliance questions from
  Dada Print Art's approved-artwork knowledge base. Use when asked to design,
  create, or draft a label for a generic/molecule name (e.g. "design a label
  for pantoprazole 40mg injection"), when asked what compliance, statutory
  text, schedule warnings, or mandatory elements apply to a product, when
  asked what precedent labels exist for a drug, or to ingest/learn from the
  artwork corpus. Wraps the local `labelkb` CLI in tools/labelkb.
---

# Label design & compliance (labelkb)

This skill is a thin wrapper. All knowledge lives in the `labelkb` tool and
the knowledge base it builds on this machine — never answer from memory what
the KB can answer from approved precedent.

## Setup check (run once per session, silently)

```bash
cd tools/labelkb && python -m labelkb.cli --version
```

If that fails: `pip install -e tools/labelkb` first. If the KB is empty
(`labelkb lookup` says so), tell the user to run ingestion — see "Learning"
below — and continue with what exists.

## Answering "design a label for <generic>"

1. `python -m labelkb.cli lookup "<generic and form>"` — see what precedent
   exists. Report the status honestly: exact / partial / none.
2. `python -m labelkb.cli design "<generic>" [--die WxH]` — generates
   SVG + PDF + brief into the KB's designs folder and prints the paths.
3. Show the user: precedent jobs used, the CONFIRM items verbatim, and the
   file paths. CONFIRM items are not decoration — repeat them and say a
   qualified person must resolve each before release.
4. If the user gives a die size, brand name, or customer, pass/relay them.
   Never invent a schedule (H/H1/X) classification: if the tool flags it
   unknown, ask the user.

## Answering "what compliance applies to <generic>"

1. `python -m labelkb.cli lookup "<generic>" --json`
2. Summarise from the profile: rx status across precedent, statutory warnings
   verbatim, red band frequency, storage wording, die sizes and substrates in
   use — always with job numbers so the user can pull the real artwork.
3. `kb/rules-evidence.json` holds regime-wide counts; quote them as
   "n of N approved labels", never as legal assertions. Statutory citations
   (D&C Rules 96/97, Schedules) are `VERIFY` items for a human.

## Learning (ingestion — local machine with G:\ mounted)

```bash
python -m labelkb.cli walk --root "G:\My Drive\Artworks"   # manifest
python -m labelkb.cli extract [--limit N]                   # vision pipeline
python -m labelkb.cli kb-build                               # aggregate
```

`probe --save` first on a new machine (picks the vision engine).
`labelkb web` serves the local review queue at 127.0.0.1:8377.

## Hard rules

- The KB and everything derived from artwork is customer data: never commit
  it, never upload it, never paste licence numbers or full label text into
  anything that leaves this machine.
- Every output is a design aid; approval authority stays with a qualified
  person. Do not soften or omit the disclaimer or CONFIRM items.
