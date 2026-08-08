"""Paths and machine-local settings.

Nothing here is committed: the corpus root and the chosen vision engine differ
per machine, so they live in a config file beside the KB rather than in git.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from pathlib import Path

# 600 dpi keeps 1 mm statutory type legible: a 19 mm-tall label renders ~450 px
# high, and the smallest mandatory text still lands around 24 px. Dropping below
# this is the fastest way to lose the fine print the whole tool exists to read.
DEFAULT_RENDER_DPI = 600

# Guard against a pathological MediaBox turning into a gigapixel render.
MAX_RENDER_LONG_EDGE_PX = 6000

ENV_ROOT = "LABELKB_ROOT"
ENV_KB = "LABELKB_KB"


def default_kb_dir() -> Path:
    """Where derived data lives. Kept out of git — see the repo .gitignore."""
    if env := os.environ.get(ENV_KB):
        return Path(env).expanduser()
    return Path(__file__).resolve().parents[1] / "kb"


@dataclass
class Config:
    """Machine-local configuration, persisted to ``<kb>/config.json``."""

    corpus_root: str | None = None
    kb_dir: str | None = None
    render_dpi: int = DEFAULT_RENDER_DPI

    # Set by `labelkb probe`; consumed by the vision layer.
    engine_profile: str | None = None
    vlm_model: str | None = None
    vlm_endpoint: str = "http://127.0.0.1:11434"

    @property
    def kb(self) -> Path:
        return Path(self.kb_dir).expanduser() if self.kb_dir else default_kb_dir()

    @property
    def root(self) -> Path | None:
        if self.corpus_root:
            return Path(self.corpus_root).expanduser()
        if env := os.environ.get(ENV_ROOT):
            return Path(env).expanduser()
        return None

    @property
    def records_dir(self) -> Path:
        return self.kb / "labels"

    @property
    def renders_dir(self) -> Path:
        return self.kb / "renders"

    @property
    def manifest_path(self) -> Path:
        return self.kb / "jobs.jsonl"

    @classmethod
    def load(cls, kb_dir: Path | None = None) -> "Config":
        path = (kb_dir or default_kb_dir()) / "config.json"
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            known = {f for f in cls.__dataclass_fields__}
            return cls(**{k: v for k, v in data.items() if k in known})
        return cls()

    def save(self) -> Path:
        self.kb.mkdir(parents=True, exist_ok=True)
        path = self.kb / "config.json"
        path.write_text(json.dumps(asdict(self), indent=2) + "\n", encoding="utf-8")
        return path
