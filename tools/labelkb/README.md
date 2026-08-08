# labelkb

A local tool that reads Dada Print Art's approved label artwork, builds a
knowledge base from it, and — eventually — produces a compliance spec and
layout brief for a new product given its generic name.

Every artwork in the corpus was signed off by a customer's regulatory function
and, for drug products, cleared the drug department. So the corpus is evidence:
whatever the approved labels consistently do is, by definition, what gets
approved. `labelkb` turns that from something in designers' heads into
something on disk that can be queried, corrected, and cited.

## Status

The **deterministic layer is complete and tested** — corpus discovery, Esko
sidecar decoding, proof-ticket reading, rendering and measurement. This is the
foundation the vision pipeline sits on, and it needs no GPU and no ML.

| Stage | State |
|---|---|
| Corpus walk / job manifest | done |
| Esko `.info` sidecar decode | done |
| Proof job-ticket extraction | done |
| Render + millimetre calibration | done |
| Hardware probe / engine profile | done |
| OCR layer | not started |
| OpenCV layer (red band, boxes, barcode) | not started |
| VLM role assignment | not started |
| Knowledge base + brief generation | not started |

## Why it runs locally

The corpus lives in Google Drive, but the machine that matters is the one with
Drive for Desktop mounted — the Esko sidecars record paths like
`G:/My Drive/Artworks/...`, so the whole corpus is already there as ordinary
files.

That is not just convenient, it is the only workable option. Python cannot call
the Drive connector; only the model can. Pulling artwork through the connector
means every file passes through the model's context as base64, and these PDFs
run from 0.6 MB to 38 MB — hundreds of thousands of tokens each, for one label.
Locally the same file is a path. Ingest where the files are.

## Install

```bash
cd tools/labelkb
python -m venv .venv && .venv/Scripts/activate     # Windows
pip install -e .
```

The base install needs only PyMuPDF and pydantic. The heavier stacks are opt-in
and chosen per machine — run `labelkb probe` first, then install what it
recommends:

```bash
pip install -e ".[ocr]"      # RapidOCR + OpenCV, CPU is fine
pip install -e ".[vlm]"      # local vision-language model client
pip install -e ".[web]"      # local web app
pip install -e ".[dev]"      # pytest
```

On Windows use **RapidOCR (ONNXRuntime)**, which is what `[ocr]` installs.
PaddleOCR is the same model family but its Windows install is a recurring
source of pain.

## Commands

```bash
labelkb probe --save                          # what can this machine run?
labelkb walk --root "G:\My Drive\Artworks" --dry-run
labelkb walk --root "G:\My Drive\Artworks"    # writes kb/jobs.jsonl
labelkb info  "<job>/.metadata/.<job>.pdf.info"
labelkb proof "<job>/<job> Artwork for approval.pdf"
labelkb geometry "<job>/<job>.pdf" --expect 30x19
labelkb render   "<job>/<job>.pdf" --dpi 600
```

All of them take `--json`.

## Validating against the real corpus

The suite runs against reconstructed fixtures, because neither a 5 KB Esko blob
nor a 600 KB proof PDF can be moved into a cloud session reliably. Three
commands confirm the real thing behaves the same. Run them first.

**1. Sidecar decoding.** On job 88116 this must report dieline
`Label 30 x 19.ARD`, creator `Adobe Illustrator 30.1 (Windows)`, and six inks
(`cyan, magenta, yellow, black, Cut, Outside Bleed`):

```bash
labelkb info ".../88116 - JARODOL 1ML 30X19/.metadata/.88116 - JARODOL 1ML 30X19.pdf.info"
```

**2. Millimetre calibration.** This is load-bearing: every type-size measurement
downstream is derived from it, so if the page does not measure 30 × 19 mm,
stop and fix it before ingesting anything.

```bash
labelkb geometry ".../88116 - JARODOL 1ML 30X19.pdf" --expect 30x19
```

**3. Proof ticket.** Field *names* differ between customer templates. The
parser locates values geometrically and reports what it matched; if a template
uses labels it does not know, they are simply absent — add them to `_ANCHORS`
in `labelkb/corpus/proof.py`.

```bash
labelkb proof ".../88116 - JARODOL 1ML 30X19 ARTWORK FOR APPROVAL.pdf"
```

Then sanity-check the sweep — the counts should look like your filing, and
`with_esko_sidecar` tells you how much free technical data the corpus carries:

```bash
labelkb walk --root "G:\My Drive\Artworks" --dry-run
```

## What the corpus gives up, and how

| Source | Cost | Yields |
|---|---|---|
| Folder name | free | job no., brand, pack size, die size, repeat flag |
| Proof PDF (live text) | text extraction | customer, size, repeat, colours, substrate, die no., date |
| Esko `.info` sidecar | ~5 KB read | dieline ARD, separations, ink types, creator, source path |
| Artwork PDF page box | text extraction | true die size in mm — the measurement scale |
| Artwork PDF pixels | render + OCR/VLM | everything on the label itself |

Only the last row needs the vision pipeline. Production artwork carries **no
extractable text** — all type is converted to curves — which is why the label
content can only be reached as pixels. `labelkb geometry` reports this per file
(`live text: no`), and on the rare customer-supplied PDF that *does* have live
text, that text should be preferred over anything OCR produces.

## Data handling

The repository this lives in is **public**. Anything derived from the corpus —
brand names, compositions, manufacturer addresses, manufacturing licence
numbers — belongs to Dada Print Art's customers and must not be committed. The
repo `.gitignore` excludes `kb/`, `renders/`, and artwork file types.

Tool code and rule packs are committed. Extracted data stays on the machine
that produced it. If the knowledge base ever needs to be shared across the
team, move it to a private repository rather than relaxing these rules.

## Testing

```bash
cd tools/labelkb && python -m pytest -q
```

Fixtures are reconstructions, not captures: the Esko fixture is rebuilt with
the documented encoder from the string table observed in the real job-88116
sidecar, and the proof fixtures are generated in both plausible ticket layouts.
That keeps the tests readable and lets them assert the binary framing itself —
which was subtly wrong on the first attempt and is now pinned by a regression
test.
