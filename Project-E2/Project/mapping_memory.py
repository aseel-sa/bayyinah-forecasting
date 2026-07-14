"""
mapping_memory.py — Confirmed column-mapping memory | Bayyina Platform

Simple file-based (JSON) storage for user-confirmed mappings, keyed two ways:
  1. Schema fingerprint: hash of the sorted normalized column names → the full
     confirmed mapping. A new upload with the same structure gets an instant
     proposal (no rules, no LLM).
  2. Per-column memory: normalized name → most recently confirmed role + count.

No detection logic here — storage and lookup only; intake.run_intake decides
how to use the results. Upgrade path: SQLite + tenant_id scoping when the
platform becomes multi-client; JSON is enough for now.
"""

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

# Default store lives next to the code; pass memory_path to isolate per client.
DEFAULT_PATH = Path(__file__).parent / "mapping_memory.json"


def _normalize(name: str) -> str:
    """Lowercases and squashes separators — must mirror intake._normalize."""
    return re.sub(r"[\s_\-./]+", "", str(name).strip().lower())


def schema_fingerprint(columns: List[str]) -> str:
    """SHA1 of sorted normalized column names — identifies a file structure."""
    joined = "|".join(sorted(_normalize(c) for c in columns))
    return hashlib.sha1(joined.encode("utf-8")).hexdigest()


def load_memory(path: Optional[str] = None) -> Dict[str, Any]:
    """
    Loads the memory file. Missing or corrupt file → empty structure:
    a broken memory must never block intake.
    """
    p = Path(path) if path else DEFAULT_PATH
    if not p.exists():
        return {"schemas": {}, "columns": {}}
    try:
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
        data.setdefault("schemas", {})
        data.setdefault("columns", {})
        return data
    except Exception:
        return {"schemas": {}, "columns": {}}


def lookup(columns: List[str], memory: Dict[str, Any]) -> Dict[str, Any]:
    """
    Looks up a new upload against memory.

    Returns:
      full_match  : {original_column → role} when the schema fingerprint is
                    known (stored normalized keys re-bound to the CURRENT
                    upload's original names), else None.
      column_hits : {normalized_name → {role, count}} for individually known columns.
    """
    stored = memory["schemas"].get(schema_fingerprint(columns))

    full_match = None
    if stored:
        by_norm = {_normalize(c): c for c in columns}
        full_match = {by_norm[n]: role for n, role in stored.items() if n in by_norm}

    column_hits = {_normalize(c): memory["columns"][_normalize(c)]
                   for c in columns if _normalize(c) in memory["columns"]}
    return {"full_match": full_match, "column_hits": column_hits}


def save_confirmed(confirmed: Dict[str, str], path: Optional[str] = None) -> None:
    """
    Persists a user-confirmed mapping {column → role}: stores the full schema
    under its fingerprint and updates per-column confirmation counts.
    Called ONLY from intake.confirm_mapping — never on mere proposal edits.
    A role change for a known column adopts the newest role and resets the
    count (the user is the source of truth, and their truth evolves).
    """
    p = Path(path) if path else DEFAULT_PATH
    memory = load_memory(p)

    fp = schema_fingerprint(list(confirmed.keys()))
    memory["schemas"][fp] = {_normalize(c): role for c, role in confirmed.items()}

    for col, role in confirmed.items():
        key = _normalize(col)
        entry = memory["columns"].get(key, {"role": role, "count": 0})
        if entry["role"] == role:
            entry["count"] += 1
        else:
            entry = {"role": role, "count": 1}
        memory["columns"][key] = entry

    with open(p, "w", encoding="utf-8") as f:
        json.dump(memory, f, ensure_ascii=False, indent=2)
