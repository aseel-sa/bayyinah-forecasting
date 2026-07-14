"""
llm_mapper.py — Optional LLM fallback for ambiguous column mapping | Bayyina

Prepared but DISABLED by default: llm_client=None → everything works without it
(ambiguous columns simply stay in needs_review).

Hard rules (see intake.py developer note [6]):
  - The LLM never sees data — metadata + sample values only (from profile_columns).
  - Called only for ambiguous/conflicting columns, never for all columns.
  - Its output is a SUGGESTION: confidence capped at 0.70 and needs_review
    always stays True. The user decides; the LLM never finalizes.
  - Suggestions naming unknown columns or roles outside the closed list are
    silently rejected (anti-hallucination).

llm_client contract: callable(payload: dict) → str (JSON). Any provider
(Anthropic / OpenAI / local) wrapped to this signature works — this module has
zero API dependencies. The client is expected to run at temperature 0.
"""

import json
from typing import Any, Callable, Dict, List, Optional

# Closed set of roles the LLM may suggest (mirrors intake's roles + meanings).
VALID_ROLES = {
    "date", "product", "demand_quantity",
    "price", "revenue", "customer", "region", "category", "promotion", "discount",
    "temperature", "holiday",
    "stock_on_hand", "available_inventory", "inventory_balance", "safety_stock",
    "lead_time", "reorder_point", "moq", "supplier", "stockout", "backorders",
    "fill_rate", "lost_sales", "capacity",
    "ignore",
}

# Hard cap: an LLM suggestion can never reach auto-accept territory.
LLM_CONFIDENCE_CAP = 0.70


def build_llm_payload(
    profiles: Dict[str, Dict[str, Any]],
    proposals: List[Dict[str, Any]],
    uncertain: List[Dict[str, Any]],
    n_rows: int,
) -> Dict[str, Any]:
    """
    Builds the EXACT metadata-only payload sent to the LLM. No raw data.

    Includes dataset context (row count, all column names, already-resolved
    mappings so the LLM cannot reassign them) and, per uncertain column, its
    profile (dtype, null rate, uniqueness, parse rates, ≤5 truncated samples)
    plus the rule-based guess — the LLM reviews rather than guesses cold.
    """
    resolved = {p["role"]: p["column"] for p in proposals
                if p["role"] not in ("unknown", "ignore") and not p["needs_review"]}
    return {
        "task": ("Map dataset columns to demand-forecasting roles. Return ONLY JSON "
                 "matching the response schema. Suggest roles for the uncertain "
                 "columns; never reassign resolved ones."),
        "response_schema": {
            "mappings": [{"column": "str", "role": f"one of {sorted(VALID_ROLES)}",
                          "confidence": "float 0..1", "reason": "str"}],
            "warnings": ["str"],
        },
        "dataset_context": {
            "domain": "manufacturing demand forecasting",
            "n_rows": int(n_rows),
            "all_columns": list(profiles.keys()),
            "resolved_mappings": resolved,
            "languages": ["ar", "en"],
        },
        "uncertain_columns": [
            {**profiles.get(p["column"], {"name": p["column"]}),
             "rule_based_guess": {"role": p["role"], "confidence": p["confidence"],
                                  "reasons": p["reasons"][:2]}}
            for p in uncertain
        ],
    }


def parse_llm_suggestions(raw: str, valid_columns: set) -> List[Dict[str, Any]]:
    """
    Parses and STRICTLY validates the LLM's JSON reply.

    Silently rejected: invalid JSON (→ []), unknown columns, roles outside
    VALID_ROLES, malformed confidence. Confidence is clamped to
    [0, LLM_CONFIDENCE_CAP]. The LLM can never finalize anything.
    """
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return []

    suggestions: List[Dict[str, Any]] = []
    for item in data.get("mappings", []):
        if not isinstance(item, dict):
            continue
        col, role = item.get("column"), item.get("role")
        if col not in valid_columns or role not in VALID_ROLES:
            continue  # hallucinated column/role → silent reject
        try:
            conf = min(max(float(item.get("confidence", 0.5)), 0.0), LLM_CONFIDENCE_CAP)
        except (TypeError, ValueError):
            conf = 0.5
        suggestions.append({"column": str(col), "role": str(role), "confidence": conf,
                            "reason": str(item.get("reason", ""))[:200]})
    return suggestions


def suggest_mappings(
    payload: Dict[str, Any],
    llm_client: Optional[Callable[[Dict[str, Any]], str]] = None,
) -> List[Dict[str, Any]]:
    """
    Calls the injected llm_client and returns validated suggestions.
    llm_client=None or ANY client failure → [] — the flow degrades to manual
    review, it never breaks intake.
    """
    if llm_client is None:
        return []
    try:
        raw = llm_client(payload)
    except Exception:
        return []
    return parse_llm_suggestions(raw, set(payload["dataset_context"]["all_columns"]))


def merge_suggestions(
    proposals: List[Dict[str, Any]],
    suggestions: List[Dict[str, Any]],
    warnings_list: List[str],
) -> None:
    """
    Merges LLM suggestions into proposals (in place) under hard constraints:
    only still-uncertain/unknown columns are touched; user- and memory-sourced
    proposals are never overridden; needs_review stays True ALWAYS.
    (User-facing warning strings are Arabic — the product UI is Arabic.)
    """
    by_col = {s["column"]: s for s in suggestions}
    for p in proposals:
        s = by_col.get(p["column"])
        if s is None:
            continue
        if p["source"] in ("user", "memory"):
            continue  # the LLM never outranks the user or the memory
        if not (p["needs_review"] or p["role"] == "unknown"):
            continue
        p.update({
            "role": s["role"],
            "confidence": s["confidence"],
            "source": "llm",
            "reasons": [f"اقتراح LLM: {s['reason']}"] if s["reason"] else ["اقتراح LLM."],
            "needs_review": True,  # always — the LLM suggests, the user decides
        })
        warnings_list.append(
            f"العمود «{p['column']}»: اقتراح LLM ({s['role']}) — يتطلّب تأكيد المستخدم.")
