"""Right-sizing for Hermes: fingerprint each Task, choose the cheapest rung that should still pass,
record every attempt and outcome, and learn from the records with a held-out check."""

from .chooser import Decision, Pool, Rung, audit, choose, choose_tools, split_of, tune_presets, wilson
from .engine import RightSizer, load_config
from .fingerprint import fingerprint, fuse, level_similarities
from .store import RecordStore, read_snapshot

__all__ = ["Decision", "Pool", "RecordStore", "RightSizer", "Rung", "audit", "choose", "choose_tools", "fingerprint",
           "fuse", "level_similarities", "load_config", "read_snapshot", "split_of", "tune_presets", "wilson"]
