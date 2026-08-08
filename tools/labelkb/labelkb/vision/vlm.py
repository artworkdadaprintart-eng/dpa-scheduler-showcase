"""Role assignment: deciding what each piece of text on the label *is*.

OCR delivers boxes and text; nothing yet says which box is the brand, which is
the generic name, which is the statutory warning. That classification is this
module. Two engines provide it:

* **VLM** (Ollama, local): sees a downscaled image of the whole label plus the
  numbered OCR lines, and returns one role per line as strict JSON. It never
  reads small print — the text is already read; it only classifies.

* **Rules**: keyword heuristics that need no model at all. On a CPU-only
  machine they are the engine; on a GPU machine they run anyway, as the prior
  the VLM answer is checked against — the fusion layer records when the two
  disagree, and low agreement is what routes a label into the review queue.

Roles are a closed set. A model free to invent categories produces a knowledge
base nothing can aggregate, so anything outside the enum is coerced to
``other`` and counted as a validation failure (one retry with the error fed
back, then fall back to rules).
"""

from __future__ import annotations

import base64
import json
import re
import urllib.request
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable

from ..config import Config
from .ocr import Line, OcrResult


class Role(str, Enum):
    BRAND = "brand"
    GENERIC_NAME = "generic_name"
    COMPOSITION = "composition"
    DOSAGE_FORM = "dosage_form"
    STATUTORY_WARNING = "statutory_warning"
    RX_SYMBOL = "rx_symbol"
    LICENCE_NO = "licence_no"
    MRP = "mrp"
    STORAGE = "storage"
    BATCH_MFG_EXP = "batch_mfg_exp"
    MANUFACTURER = "manufacturer"
    MARKETER = "marketer"
    NET_QTY = "net_qty"
    BARCODE_NUMBER = "barcode_number"
    OTHER = "other"


@dataclass
class Assignment:
    line_index: int
    role: Role
    confidence: float
    engine: str  # "vlm" | "rules"


@dataclass
class RoleResult:
    assignments: list[Assignment]
    engine: str
    disagreements: list[int] = field(default_factory=list)  # line indices

    def role_of(self, line_index: int) -> Role:
        for assignment in self.assignments:
            if assignment.line_index == line_index:
                return assignment.role
        return Role.OTHER

    def lines_with(self, role: Role) -> list[int]:
        return [a.line_index for a in self.assignments if a.role == role]


# --------------------------------------------------------------------------
# Rule engine
# --------------------------------------------------------------------------

# Order matters: first match wins, and the specific must precede the general.
# Patterns run against lowercased text with whitespace collapsed (RapidOCR
# merges inter-word spaces unpredictably, so rules cannot rely on them).
_RULES: list[tuple[Role, re.Pattern]] = [
    (Role.STATUTORY_WARNING, re.compile(
        r"schedule\s*h1?|schedule\s*x|not\s*to\s*be\s*sold|"
        r"without\s*(the)?\s*prescription|to\s*be\s*sold\s*by\s*retail|"
        r"registered\s*medical\s*practitioner|keep\s*out\s*of\s*(the)?\s*reach"
    )),
    (Role.LICENCE_NO, re.compile(r"m(fg)?\.?\s*lic(ence)?\.?\s*no|licen[cs]e\s*no")),
    (Role.MRP, re.compile(r"m\.?\s*r\.?\s*p|retail\s*price|price\s*not\s*exceed")),
    (Role.BATCH_MFG_EXP, re.compile(
        r"batch\s*no|b\.?\s*no\.?|lot\s*no|mfg\.?\s*d(ate|t)|exp\.?\s*d(ate|t)|expiry"
    )),
    (Role.STORAGE, re.compile(
        r"store\s|storage|protect(ed)?\s*from\s*light|cool\s*(and|&)?\s*dry|"
        r"below\s*\d+\s*.?c|do\s*not\s*freeze"
    )),
    (Role.MARKETER, re.compile(r"marketed\s*by")),
    (Role.MANUFACTURER, re.compile(r"manufactured\s*by|mfd\.?\s*by|mfg\.?\s*by")),
    (Role.COMPOSITION, re.compile(
        r"^(each|every)\s*(ml|tablet|capsule|5\s*ml|gm?|sachet)|composition|contains?\s*:"
    )),
    (Role.NET_QTY, re.compile(r"net\s*(qty|quantity|content|wt|weight|vol)")),
    (Role.RX_SYMBOL, re.compile(r"^rx$|^℞$")),
    (Role.BARCODE_NUMBER, re.compile(r"^\d{12,14}$")),
    (Role.DOSAGE_FORM, re.compile(
        r"^(oral\s*)?(injection|tablets?|capsules?|syrup|suspension|drops|"
        r"ointment|cream|gel|lotion|solution|infusion)$"
    )),
    # Pharmacopoeia suffix inside a product-style line: the generic name.
    (Role.GENERIC_NAME, re.compile(
        r"\b(ip|bp|usp)\b.*\b(injection|tablets?|capsules?|syrup|suspension|drops|solution)|"
        r"\b(injection|tablets?|capsules?|syrup|suspension|drops|solution)\b.*\b(ip|bp|usp)\b"
    )),
]


def _squash(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def assign_by_rules(lines: list[Line]) -> list[Assignment]:
    """Keyword classification, plus one structural rule: of the lines left
    unclaimed, the tallest is the brand — on these labels the brand is
    reliably the largest type."""
    assignments: list[Assignment] = []
    unclaimed: list[int] = []

    for index, line in enumerate(lines):
        text = _squash(line.text)
        for role, pattern in _RULES:
            if pattern.search(text):
                assignments.append(Assignment(index, role, 0.7, "rules"))
                break
        else:
            unclaimed.append(index)

    if unclaimed:
        tallest = max(unclaimed, key=lambda i: lines[i].height_px)
        # Guard: a brand is prominent. If the tallest leftover is no taller
        # than the median claimed line, nothing here looks like a brand.
        claimed_heights = sorted(
            lines[a.line_index].height_px for a in assignments
        )
        median = claimed_heights[len(claimed_heights) // 2] if claimed_heights else 0.0
        if lines[tallest].height_px > median:
            assignments.append(Assignment(tallest, Role.BRAND, 0.6, "rules"))
            unclaimed.remove(tallest)

    for index in unclaimed:
        assignments.append(Assignment(index, Role.OTHER, 0.4, "rules"))

    assignments.sort(key=lambda a: a.line_index)
    return assignments


# --------------------------------------------------------------------------
# VLM engine
# --------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "You classify text found on a printed pharmaceutical label. You are given "
    "the label image and a numbered list of text lines already read by OCR "
    "(with their heights in mm and positions). For every line, assign exactly "
    "one role from this list: "
    + ", ".join(role.value for role in Role)
    + ". Do not read new text from the image; classify only the given lines. "
    "Reply with JSON only: {\"assignments\": [{\"line\": <int>, \"role\": "
    "\"<role>\", \"confidence\": <0..1>}]} with one entry per line."
)


def build_line_listing(result: OcrResult) -> str:
    scale = result.px_per_mm or 1.0
    rows = []
    for index, line in enumerate(result.lines):
        rows.append(
            {
                "line": index,
                "text": line.text,
                "height_mm": round(line.height_px / scale, 2),
                "y_mm": round(line.y0 / scale, 1),
                "x_mm": round(line.x0 / scale, 1),
            }
        )
    return json.dumps(rows, ensure_ascii=False)


Transport = Callable[[dict], str]


def _ollama_transport(endpoint: str, timeout: float = 180.0) -> Transport:
    def send(payload: dict) -> str:  # pragma: no cover - needs a live server
        request = urllib.request.Request(
            f"{endpoint.rstrip('/')}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
        return body["message"]["content"]

    return send


def _parse_assignments(raw: str, line_count: int) -> list[Assignment]:
    """Strict: bad JSON or an out-of-range line index raises; an unknown role
    becomes OTHER (recorded via low confidence) rather than poisoning the enum."""
    data = json.loads(raw)
    assignments = []
    for entry in data["assignments"]:
        index = int(entry["line"])
        if not 0 <= index < line_count:
            raise ValueError(f"line index {index} out of range 0..{line_count - 1}")
        try:
            role = Role(str(entry["role"]).strip().lower())
            confidence = float(entry.get("confidence", 0.5))
        except ValueError:
            role, confidence = Role.OTHER, 0.1
        assignments.append(Assignment(index, role, confidence, "vlm"))
    seen = {a.line_index for a in assignments}
    missing = set(range(line_count)) - seen
    if missing:
        raise ValueError(f"lines not classified: {sorted(missing)}")
    assignments.sort(key=lambda a: a.line_index)
    return assignments


def assign_by_vlm(
    result: OcrResult,
    image_path: str | Path,
    model: str,
    transport: Transport,
    max_retries: int = 1,
) -> list[Assignment]:
    image_b64 = base64.b64encode(Path(image_path).read_bytes()).decode("ascii")
    listing = build_line_listing(result)

    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": f"Text lines:\n{listing}", "images": [image_b64]},
    ]
    last_error: Exception | None = None
    for _ in range(max_retries + 1):
        raw = transport({"model": model, "messages": messages, "stream": False, "format": "json"})
        try:
            return _parse_assignments(raw, len(result.lines))
        except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            last_error = exc
            messages.append({"role": "assistant", "content": raw})
            messages.append(
                {"role": "user", "content": f"Invalid: {exc}. Reply with corrected JSON only."}
            )
    raise RuntimeError(f"VLM returned unusable assignments: {last_error}")


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------

def assign_roles(
    result: OcrResult,
    image_path: str | Path,
    config: Config | None = None,
    transport: Transport | None = None,
) -> RoleResult:
    """Classify every OCR line, with the engine the machine supports.

    GPU profiles use the VLM and keep the rule verdicts as a cross-check;
    disagreements are recorded so extraction can lower field confidence and
    the review queue can prioritise. Any VLM failure degrades to rules —
    a weaker answer, never no answer.
    """
    config = config or Config.load()
    rule_assignments = assign_by_rules(result.lines)

    use_vlm = bool(config.vlm_model) and (config.engine_profile or "").startswith("gpu")
    if not use_vlm:
        return RoleResult(assignments=rule_assignments, engine="rules")

    try:
        vlm_assignments = assign_by_vlm(
            result,
            image_path,
            model=config.vlm_model,
            transport=transport or _ollama_transport(config.vlm_endpoint),
        )
    except (RuntimeError, OSError):
        return RoleResult(assignments=rule_assignments, engine="rules")

    rules_by_line = {a.line_index: a.role for a in rule_assignments}
    disagreements = [
        a.line_index
        for a in vlm_assignments
        if rules_by_line.get(a.line_index, Role.OTHER) not in (a.role, Role.OTHER)
    ]
    return RoleResult(assignments=vlm_assignments, engine="vlm", disagreements=disagreements)
