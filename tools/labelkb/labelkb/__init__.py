"""labelkb — learn label-artwork conventions and compliance from approved artwork.

The corpus is the source of truth: every record, archetype and rule the tool
emits traces back to job numbers whose artwork was actually approved.
"""

__version__ = "0.1.0"

# Bumped whenever extraction semantics change, so stale records in the KB can be
# spotted and re-run rather than silently mixed with new ones.
EXTRACTOR_VERSION = 1
