# labelkb

A local tool that **sees** Dada Print Art's approved label artwork, **learns**
from it, and **creates new label designs** from a generic name:

```bash
labelkb design "pantoprazole 40mg injection" --die 34x25
# -> SVG + PDF draft at true die size, every mandatory element placed in its
#    learned zone at its learned size, statutory wording verbatim from
#    approved precedent, plus a brief justifying each element with job numbers
```

Every artwork in the corpus was signed off by a customer's regulatory function
and, for drug products, cleared the drug department. So the corpus is evidence:
whatever the approved labels consistently do is, by definition, what gets
approved. `labelkb` turns that from something in designers' heads into
something on disk that can be queried, corrected, and cited.

## Status

| Stage | State |
|---|---|
| Corpus walk / job manifest | done |
| Esko `.info` sidecar decode | done |
| Proof job-ticket extraction | done |
| Render + millimetre calibration | done |
| Hardware probe / engine profile | done |
| OCR layer (RapidOCR, boxes → mm) | done |
| OpenCV layer (red band, ruled boxes, barcode) | done |
| Role assignment (local VLM via Ollama + rule engine) | done |
| Extraction fusion → LabelRecord | done |
| Knowledge base (generics / archetypes / rules-evidence) | done |
| CREATE: design generator + brief + SVG/PDF/DOCX export | done |
| Local web app (search, design, review queue) | done |
| Claude Code skill wrapper | done |
| Corpus pilot on the studio machine | **next — yours** |

## How LEARN works

Render each artwork at 600 dpi → **OCR reads the type** (pixel boxes convert
to real millimetres, so type sizes are measurements, not guesses) → **OpenCV
measures the furniture** (Schedule-H red band, ruled warning boxes, barcode
zones) → **a local VLM assigns each text line a role** (brand / generic /
composition / statutory warning / licence / MRP / …) from a closed enum, with
a keyword rule engine as the CPU-only path, the prior, and the fallback →
everything fuses with the folder name, proof ticket and Esko sidecar into one
`LabelRecord` per job, with per-field confidence and provenance.

`kb-build` then aggregates records into three indexes:

- `generics.json` — per-molecule profiles (salt forms collapse to one key,
  combinations key as sorted `a+b`)
- `archetypes.json` — median zone map and type-size ratios per
  (dosage form × die size): what makes a 30×19 injection label look right
- `rules-evidence.json` — observed regularities with counts and job numbers,
  e.g. "red band on 43/47 approved Rx labels"

Corrections from the review queue overlay the records (never edit them) and
double as a labelled dataset if a LoRA fine-tune is ever wanted.

## How CREATE works

`labelkb design "<generic>"` resolves the molecule against `generics.json`
(exact → precedent products; partial → shared-molecule neighbours, flagged;
none → same-form neighbours, flagged), picks the die (requested, or the most
common among precedent), pulls the archetype for that form and die, and places
every mandatory element into its learned zone at its learned size — statutory
wording verbatim from precedent, generic type sized by the median
brand-to-generic ratio, red band only when precedent Rx labels carry it. The
result renders to SVG and PDF at exact millimetre scale (opens in
CorelDRAW/Illustrator at die size) with a markdown/DOCX brief citing the
precedent job for every element.

What it will not do: guess. Schedule status of an unseen molecule, missing
statutory wording, unconfirmed composition — each becomes a **CONFIRM** item
printed on the design, in the brief, and in the CLI output.

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
# once per machine
labelkb probe --save                          # picks the vision engine profile

# LEARN
labelkb walk --root "G:\My Drive\Artworks"    # corpus -> kb/jobs.jsonl
labelkb extract [--limit N] [--job 88116]     # vision pipeline -> kb/labels/
labelkb kb-build                              # -> generics/archetypes/rules

# ASK and CREATE
labelkb lookup "pantoprazole 40mg injection"
labelkb design "pantoprazole 40mg injection" --die 34x25
labelkb web                                   # 127.0.0.1:8377 - search, design,
                                              # review queue for corrections

# per-file inspection
labelkb info  "<job>/.metadata/.<job>.pdf.info"
labelkb proof "<job>/<job> Artwork for approval.pdf"
labelkb geometry "<job>/<job>.pdf" --expect 30x19
labelkb render   "<job>/<job>.pdf" --dpi 600
```

All of them take `--json`. The Claude Code skill in
`.claude/skills/label-compliance/` wraps these same commands, so
"design a label for cefixime 200mg tablet" works conversationally.

## Recommended pilot (before the full corpus)

1. `labelkb probe --save`, install the extras it recommends.
2. `labelkb walk --root "G:\My Drive\Artworks" --dry-run` — sanity-check counts.
3. `labelkb extract --limit 50` — about 50 jobs across a few pharma customers.
4. `labelkb kb-build`, then `labelkb design` for a generic that IS in those 50,
   and diff the draft element-by-element against the real approved artwork.
5. Work the review queue (`labelkb web`), re-run `kb-build`, then let
   `extract` run over the whole corpus.

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
