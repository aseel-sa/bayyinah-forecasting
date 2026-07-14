"""
insights.py — Deterministic Decision-Support Insights | Bayyina Platform

Position: after model + forecast_performance. A read-only CONSUMER of validated
pipeline outputs. It translates structured forecasting results into planning
insights for humans (and, later, for chatbot.py).

Hard rules (by design — see DEV NOTES):
  - Deterministic only. No LLM here. Same inputs → same insights.
  - Never invents a number: every figure traces to a structured input field.
  - Non-blocking: any failure returns a valid 'failed' payload; it can never
    break the forecasting pipeline.
  - Explainability is a PLACEHOLDER hook today (no SHAP/LIME) but the contract
    already accepts a future explainability_result.

Main entry point: generate_insights(model_result, performance_result=None,
                  quality_result=None, scenario_result=None,
                  explainability_result=None) -> dict (JSON-serializable)
"""

# ===========================================================================
# DEVELOPER NOTES
# ===========================================================================
#
# [1] Inputs are tolerated in any reasonable shape: DataFrame, list-of-dicts,
#     or dict. `_records()` normalizes tabular inputs; `_num()`/`_get()` guard
#     scalars and nested keys. Nothing assumes an exact DataFrame schema.
#
# [2] "Growth" requires a real historical comparison (trailing actual mean vs
#     forecast mean, equal windows). When history is unavailable we rank by
#     forecasted VOLUME and label it "highest forecasted volume" — never call
#     a volume ranking "growth" (spec rule). The basis is reported per item.
#
# [3] Confidence anchors on the model layer's own `confidence_label`
#     (excellent/good/needs_attention/poor → High/Medium/Low) then is adjusted
#     DOWN (never up) for data-quality warnings, failed models, and very short
#     evaluation. Conservative fallback to "Low + missing-info note" when the
#     anchor is absent — we never claim confidence we cannot support.
#
# [4] Thresholds (_WMAPE_*, _BIAS_*) are practical labels consistent with the
#     model layer's metric cards, NOT universal forecasting truth. One place,
#     easy to change.
#
# [5] Risk scoring is additive over available signals (per-product wMAPE, bias,
#     segment, scenario sensitivity, failed models, high-volume×weak-confidence).
#     Only medium/high-risk products are surfaced, capped, to avoid flooding.
#
# [6] Output is run through `_json_safe()` (Timestamps→ISO, numpy→python,
#     NaN→None) so chatbot_context and the whole payload are JSON-serializable.
#
# ===========================================================================

from typing import Any, Dict, List, Optional

try:
    import pandas as pd
except Exception:  # pandas should exist, but insights must never hard-depend
    pd = None  # type: ignore


# ─── Heuristic thresholds (labels, not universal truth — note [4]) ───────────

_WMAPE_GOOD = 0.20          # ≤ → good portfolio accuracy
_WMAPE_WEAK = 0.35          # > → weak; product-level high-error flag
_BIAS_STRONG = 0.30        # |bias| > → strong directional skew
_BIAS_NOTABLE = 0.15       # |bias| > → notable skew
_TOP_N = 8                 # cap for surfaced product lists

_EXPLAINABILITY_PLACEHOLDER = {
    "status": "not_available",
    "method": None,
    "top_drivers": [],
    "notes": [
        "Technical explainability such as SHAP can be added later for supported models.",
        "Current insights use validation performance, bias, warnings, and available model metadata.",
    ],
}

_INVENTORY_LIMITATION = ("Inventory recommendations require current inventory, "
                         "lead time, and service-level inputs and are not "
                         "included in this version.")

# Mandatory scope disclaimer — insights are demand/validation only; they do not
# assert operational, inventory, or capacity risk (no such data exists yet).
_SCOPE_LIMITATION = ("Current insights are based on demand forecasts and "
                     "validation signals only. Production feasibility, inventory "
                     "risk, and capacity constraints require additional "
                     "operational data.")


# ─── Safe accessors (note [1]) ───────────────────────────────────────────────

def _get(obj: Any, *keys: str, default: Any = None) -> Any:
    """Nested dict get; returns default on any missing key or non-dict."""
    cur = obj
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _records(obj: Any) -> List[Dict[str, Any]]:
    """Normalizes a DataFrame / list-of-dicts / None into a list of dicts."""
    if obj is None:
        return []
    if pd is not None and isinstance(obj, pd.DataFrame):
        if obj.empty:
            return []
        return obj.to_dict("records")
    if isinstance(obj, list):
        return [r for r in obj if isinstance(r, dict)]
    return []


def _num(x: Any) -> Optional[float]:
    """Coerces to float or None (handles NaN, strings, numpy scalars)."""
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if v != v:  # NaN
        return None
    return v


def _pct(x: Optional[float]) -> str:
    """Formats a fraction as a percent string, or 'n/a'."""
    return f"{x:.0%}" if isinstance(x, (int, float)) and x == x else "n/a"


def _json_safe(obj: Any) -> Any:
    """Recursively converts Timestamps/numpy/NaN into JSON-serializable values."""
    if obj is None:
        return None
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if pd is not None and isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, (int, float)):
        return None if obj != obj else obj  # NaN → None
    if hasattr(obj, "item"):  # numpy scalar
        try:
            return obj.item()
        except Exception:
            return str(obj)
    if hasattr(obj, "isoformat"):  # date/datetime
        try:
            return obj.isoformat()
        except Exception:
            return str(obj)
    if isinstance(obj, str):
        return obj
    return str(obj)


def _tail_mean(series_like: Any, n: int) -> Optional[float]:
    """Mean of the last n values of a Series/list. None if unusable."""
    if n <= 0:
        return None
    vals: List[float] = []
    if pd is not None and isinstance(series_like, pd.Series):
        seq = list(series_like.values)
    elif isinstance(series_like, (list, tuple)):
        seq = list(series_like)
    else:
        return None
    for v in seq[-n:]:
        fv = _num(v)
        if fv is not None:
            vals.append(fv)
    return sum(vals) / len(vals) if vals else None


# ─── Per-product extraction from the model leaderboard ───────────────────────

def _selected_perf_by_product(model_result: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    {product → {wmape, mae, forecast_bias, status}} for the SELECTED model of
    each product, read from the model leaderboard (scope == 'product').
    Empty when the leaderboard is absent.
    """
    out: Dict[str, Dict[str, Any]] = {}
    for r in _records(model_result.get("model_leaderboard")):
        if r.get("scope") != "product" or not r.get("selected"):
            continue
        prod = r.get("product")
        if prod in (None, "ALL"):
            continue
        out[str(prod)] = {"wmape": _num(r.get("wmape")), "mae": _num(r.get("mae")),
                          "forecast_bias": _num(r.get("forecast_bias")),
                          "status": r.get("status")}
    return out


def _forecast_volume_by_product(model_result: Dict[str, Any]) -> Dict[str, float]:
    """{product → summed forecast volume} from product_forecasts."""
    vol: Dict[str, float] = {}
    for r in _records(model_result.get("product_forecasts")):
        prod = r.get("product")
        f = _num(r.get("forecast"))
        if prod is None or f is None:
            continue
        vol[str(prod)] = vol.get(str(prod), 0.0) + f
    return vol


# ─── Section: top growth / volume (note [2]) ─────────────────────────────────

def _top_growth_products(model_result: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Highest forecasted demand INCREASE vs trailing history when history is
    available; otherwise highest forecasted VOLUME, explicitly labeled.
    """
    forecasts = _forecast_volume_by_product(model_result)
    if not forecasts:
        return []

    # Forecast mean per product (volume / n_steps).
    steps: Dict[str, int] = {}
    for r in _records(model_result.get("product_forecasts")):
        prod = r.get("product")
        if prod is not None:
            steps[str(prod)] = steps.get(str(prod), 0) + 1

    history = model_result.get("history")
    growth_items: List[Dict[str, Any]] = []
    if isinstance(history, dict) and history:
        for prod, fc_total in forecasts.items():
            n = steps.get(prod, 0)
            if n <= 0:
                continue
            fc_mean = fc_total / n
            recent_mean = _tail_mean(history.get(prod), n)
            if recent_mean is None or recent_mean <= 0:
                continue
            growth_pct = (fc_mean - recent_mean) / recent_mean
            growth_items.append({
                "product": str(prod), "basis": "growth_vs_history",
                "forecast_mean": round(fc_mean, 2),
                "recent_actual_mean": round(recent_mean, 2),
                "growth_pct": round(growth_pct, 4),
                "forecast_total": round(fc_total, 2)})

    if growth_items:
        growth_items.sort(key=lambda x: x["growth_pct"], reverse=True)
        return growth_items[:_TOP_N]

    # Fallback: rank by volume, clearly labeled (NOT growth).
    vol_items = [{"product": p, "basis": "highest_forecasted_volume",
                  "forecast_total": round(v, 2)}
                 for p, v in forecasts.items()]
    vol_items.sort(key=lambda x: x["forecast_total"], reverse=True)
    return vol_items[:_TOP_N]


# ─── Section: model confidence (note [3]) ────────────────────────────────────

def _label_to_level(label: Optional[str]) -> Optional[str]:
    return {"excellent": "High", "good": "High",
            "needs_attention": "Medium", "poor": "Low"}.get(label or "")


def _model_confidence(model_result: Dict[str, Any],
                      performance_result: Optional[Dict[str, Any]],
                      quality_result: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    fs = model_result.get("forecast_summary") or {}
    wmape = _num(fs.get("portfolio_wmape"))
    label = fs.get("confidence_label") or _get(performance_result, "summary", "confidence_label")
    bias_dir = fs.get("forecast_bias_direction")

    reasons: List[str] = []
    limitations: List[str] = []

    base_level = _label_to_level(label)
    if base_level:
        reasons.append(f"Model layer confidence label: '{label}'"
                       + (f" (portfolio wMAPE {_pct(wmape)})." if wmape is not None else "."))
    else:
        limitations.append("No validation-based confidence label available; "
                           "treat the forecast as a decision-support signal.")

    # Downgrade-only adjustments (note [3]).
    downgrade = 0
    q_issues = _records(_get(quality_result, "issues", default=[]))
    n_critical = sum(1 for i in q_issues if i.get("severity") == "critical")
    n_warning = sum(1 for i in q_issues if i.get("severity") == "warning")
    if n_critical:
        downgrade += 2
        reasons.append(f"{n_critical} critical data-quality issue(s) reduce confidence.")
    elif n_warning >= 3:
        downgrade += 1
        reasons.append(f"{n_warning} data-quality warnings reduce confidence.")

    failed = _records(model_result.get("failed_models"))
    n_failed = sum(1 for f in failed if f.get("status") == "failed")
    if n_failed:
        downgrade += 1
        reasons.append(f"{n_failed} model run(s) failed during evaluation.")

    eval_sum = model_result.get("evaluation_summary") or {}
    n_eval = _num(eval_sum.get("n_products_evaluated"))
    n_total = _num(eval_sum.get("n_products_forecasted"))
    if n_eval is not None and n_total and n_eval < n_total:
        unvalidated = int(n_total - n_eval)
        downgrade += 1 if unvalidated > 0 else 0
        limitations.append(f"{unvalidated} product(s) lacked enough history for "
                           "walk-forward validation; their forecasts rest on a baseline.")

    if bias_dir in ("over-forecasting", "under-forecasting"):
        reasons.append(f"Portfolio tends toward {bias_dir}.")

    # Map base level + downgrades to a final level (downgrade only).
    order = ["Low", "Medium", "High"]
    start = order.index(base_level) if base_level in order else 0  # missing → Low base
    final = order[max(0, start - downgrade)]
    if base_level is None and downgrade == 0:
        final = "Low"  # conservative fallback when we know nothing

    return {"level": final, "reasons": reasons or ["Insufficient signals to assess confidence."],
            "limitations": limitations}


# ─── Section: high-risk products (note [5]) ──────────────────────────────────

def _high_risk_products(model_result: Dict[str, Any],
                        scenario_result: Optional[Dict[str, Any]],
                        confidence_level: str) -> List[Dict[str, Any]]:
    perf = _selected_perf_by_product(model_result)
    volume = _forecast_volume_by_product(model_result)
    segments = {str(r.get("product")): r for r in _records(model_result.get("product_segments"))}
    failed_by_prod: Dict[str, int] = {}
    for f in _records(model_result.get("failed_models")):
        p = f.get("product")
        if p not in (None, "ALL"):
            failed_by_prod[str(p)] = failed_by_prod.get(str(p), 0) + 1
    scenario_affected = {str(p) for p in
                         _get(scenario_result, "impact_summary", "most_affected_products",
                              default=[]) if p is not None}

    vol_sorted = sorted(volume.values(), reverse=True)
    high_vol_cut = vol_sorted[max(0, len(vol_sorted) // 5 - 1)] if vol_sorted else None

    candidates = set(perf) | set(volume) | set(segments) | set(failed_by_prod) | scenario_affected
    scored: List[Dict[str, Any]] = []

    for prod in candidates:
        drivers: List[str] = []
        score = 0
        pe = perf.get(prod, {})
        wmape = pe.get("wmape")
        bias = pe.get("forecast_bias")

        # "error" is retained intentionally: wMAPE is a measured VALIDATION
        # error metric. The framing leads with forecast uncertainty, not
        # operational risk.
        if wmape is not None and wmape > _WMAPE_WEAK:
            drivers.append(f"high forecast uncertainty (validation error wMAPE {_pct(wmape)})"); score += 2
        elif wmape is not None and wmape > _WMAPE_GOOD:
            drivers.append(f"elevated forecast uncertainty (validation error wMAPE {_pct(wmape)})"); score += 1

        if bias is not None and abs(bias) > _BIAS_STRONG:
            drivers.append(f"strong validation bias ({bias:+.0%})"); score += 2
        elif bias is not None and abs(bias) > _BIAS_NOTABLE:
            drivers.append(f"notable validation bias ({bias:+.0%})"); score += 1

        seg = (segments.get(prod) or {}).get("segment")
        if seg == "intermittent":
            drivers.append("intermittent demand pattern"); score += 1
        elif seg == "short_history":
            drivers.append("limited historical data"); score += 1

        if prod in scenario_affected:
            drivers.append("high sensitivity in what-if scenario"); score += 1

        if failed_by_prod.get(prod):
            drivers.append("a candidate model failed during evaluation"); score += 1

        if (high_vol_cut is not None and volume.get(prod, 0) >= high_vol_cut
                and confidence_level == "Low"):
            drivers.append("high forecasted volume under low overall confidence"); score += 1

        if score <= 0:
            continue
        level = "high" if score >= 3 else "medium" if score >= 1 else "low"
        action = _risk_action(drivers)
        scored.append({"product": prod, "risk_level": level,
                       "risk_drivers": drivers, "recommended_action": action,
                       "_score": score, "_vol": volume.get(prod, 0.0)})

    scored.sort(key=lambda x: (x["_score"], x["_vol"]), reverse=True)
    for s in scored:
        s.pop("_score", None)
        s.pop("_vol", None)
    return scored[:_TOP_N]


def _risk_action(drivers: List[str]) -> str:
    joined = " ".join(drivers)
    if "forecast error" in joined or "model failed" in joined:
        return "Investigate this product's forecast error before using it operationally."
    if "bias" in joined:
        return "Review the forecast for this product; bias may distort planning quantities."
    if "scenario" in joined:
        return "Use what-if scenarios for this demand-sensitive product before committing plans."
    if "intermittent" in joined or "limited historical" in joined:
        return "Treat this forecast as a decision-support signal requiring planner review."
    return "Flag this product for planner review before production planning."


# ─── Section: key findings ───────────────────────────────────────────────────

def _key_findings(model_result: Dict[str, Any],
                  quality_result: Optional[Dict[str, Any]],
                  scenario_result: Optional[Dict[str, Any]],
                  high_risk: List[Dict[str, Any]],
                  growth: List[Dict[str, Any]],
                  confidence: Dict[str, Any],
                  explainability_available: bool) -> List[Dict[str, Any]]:
    findings: List[Dict[str, Any]] = []

    # Concentration of demand.
    volume = _forecast_volume_by_product(model_result)
    if volume:
        total = sum(volume.values()) or 1.0
        top = sorted(volume.values(), reverse=True)
        top3_share = sum(top[:3]) / total
        if top3_share > 0.6 and len(volume) > 3:
            findings.append({"title": "Demand is concentrated in a few products",
                             "description": f"The top 3 products account for {top3_share:.0%} "
                                            "of forecasted demand; planning accuracy for them "
                                            "matters disproportionately.",
                             "severity": "medium", "source": "forecast"})

    # Forecast error.
    n_high_err = sum(1 for r in high_risk
                     if any("forecast error" in d for d in r["risk_drivers"]))
    if n_high_err:
        findings.append({"title": "Some products show high forecast error",
                         "description": f"{n_high_err} product(s) have elevated validation "
                                        "error and should be reviewed before operational use.",
                         "severity": "high" if n_high_err >= 3 else "medium",
                         "source": "performance"})

    # Quality limiting confidence.
    q_issues = _records(_get(quality_result, "issues", default=[]))
    n_q = sum(1 for i in q_issues if i.get("severity") in ("critical", "warning"))
    if n_q and confidence["level"] != "High":
        findings.append({"title": "Model confidence is limited by data-quality issues",
                         "description": f"{n_q} data-quality issue(s) were detected; they lower "
                                        "confidence in the forecast.",
                         "severity": "high" if confidence["level"] == "Low" else "medium",
                         "source": "quality"})

    # Scenario sensitivity.
    delta_pct = _num(_get(scenario_result, "impact_summary", "total_demand_delta_pct"))
    if delta_pct is not None and abs(delta_pct) > 0:
        findings.append({"title": "Scenario results indicate demand sensitivity",
                         "description": f"A declared what-if scenario shifts total demand by "
                                        f"{delta_pct:+.0%}, indicating planning sensitivity.",
                         "severity": "medium", "source": "scenario"})

    # Growth direction (only when growth-basis available).
    if growth and growth[0].get("basis") == "growth_vs_history":
        leader = growth[0]
        findings.append({"title": "A product shows strong forecasted growth",
                         "description": f"'{leader['product']}' is forecast to grow "
                                        f"{leader['growth_pct']:+.0%} versus recent demand.",
                         "severity": "low", "source": "forecast"})

    # Explainability status (always informative, low severity).
    if not explainability_available:
        findings.append({"title": "Technical explainability is not yet available",
                         "description": "Insights rely on validation performance, bias, and "
                                        "model metadata; feature-level explanations (e.g. SHAP) "
                                        "can be added later for supported models.",
                         "severity": "low", "source": "explainability"})

    return findings[:6]


# ─── Section: forecast quality notes ─────────────────────────────────────────

def _forecast_quality_notes(model_result: Dict[str, Any],
                            performance_result: Optional[Dict[str, Any]]) -> List[str]:
    notes: List[str] = []
    fs = model_result.get("forecast_summary") or {}
    psum = _get(performance_result, "summary", default={}) or {}

    wmape = _num(fs.get("portfolio_wmape")) if _num(fs.get("portfolio_wmape")) is not None \
        else _num(psum.get("portfolio_wmape"))
    if wmape is not None:
        notes.append(f"Measured portfolio weighted error (wMAPE) is {_pct(wmape)} "
                     "from walk-forward validation.")
    bias_dir = fs.get("forecast_bias_direction") or psum.get("forecast_bias_direction")
    if bias_dir:
        notes.append(f"Validation indicates the forecast tends toward {bias_dir}.")

    n_over = _num(psum.get("n_over_forecast"))
    n_under = _num(psum.get("n_under_forecast"))
    if n_over is not None and n_under is not None:
        notes.append(f"Across validation periods, {int(n_over)} were over-forecast and "
                     f"{int(n_under)} under-forecast.")

    failed = _records(model_result.get("failed_models"))
    n_failed = sum(1 for f in failed if f.get("status") == "failed")
    n_skipped = sum(1 for f in failed if f.get("status") == "skipped")
    if n_failed or n_skipped:
        notes.append(f"{n_failed} model run(s) failed and {n_skipped} were skipped "
                     "(e.g. an optional package was unavailable); these did not block the forecast.")

    notes.append("These are MEASURED validation signals, distinct from any hypothetical "
                 "what-if scenario assumptions.")
    return notes


# ─── Section: data limitations ───────────────────────────────────────────────

def _data_limitations(model_result: Dict[str, Any],
                      quality_result: Optional[Dict[str, Any]],
                      performance_result: Optional[Dict[str, Any]],
                      explainability_available: bool) -> List[str]:
    lims: List[str] = []

    for i in _records(_get(quality_result, "issues", default=[])):
        if i.get("severity") in ("critical", "warning") and i.get("description"):
            lims.append(str(i["description"]))
        if len(lims) >= 4:
            break

    eval_sum = model_result.get("evaluation_summary") or {}
    n_eval = _num(eval_sum.get("n_products_evaluated"))
    n_total = _num(eval_sum.get("n_products_forecasted"))
    if n_eval is not None and n_total and n_eval < n_total:
        lims.append(f"{int(n_total - n_eval)} product(s) had insufficient history for "
                    "walk-forward validation.")

    if not _get(performance_result, "summary", "n_validation_rows"):
        if performance_result is not None:
            lims.append("Product-level validation performance was limited or unavailable.")

    # External-feature warnings from the model layer.
    for w in _records(model_result.get("warnings")):
        if w.get("affected_scope") == "external_features" and w.get("message"):
            lims.append(str(w["message"]))
            break

    lims.append(_INVENTORY_LIMITATION)
    if not explainability_available:
        lims.append("Technical (feature-level) explainability is not yet available for "
                    "the selected models.")
    return lims


# ─── Section: recommended actions ────────────────────────────────────────────

def _recommended_actions(high_risk: List[Dict[str, Any]],
                         growth: List[Dict[str, Any]],
                         confidence: Dict[str, Any],
                         scenario_result: Optional[Dict[str, Any]],
                         explainability_available: bool) -> List[Dict[str, Any]]:
    actions: List[Dict[str, Any]] = []

    growth_products = [g["product"] for g in growth
                       if g.get("basis") == "growth_vs_history"][:5]
    if growth_products:
        actions.append({"action": "Review high-growth products before production planning.",
                        "priority": "high", "reason": "These products are forecast to grow "
                        "versus recent demand and may need capacity ahead of time.",
                        "related_products": growth_products})

    err_products = [r["product"] for r in high_risk
                    if any("forecast error" in d for d in r["risk_drivers"])][:5]
    if err_products:
        actions.append({"action": "Investigate products with high forecast error before "
                        "using results operationally.",
                        "priority": "high", "reason": "Elevated validation error means these "
                        "forecasts are less reliable for committing plans.",
                        "related_products": err_products})

    if _get(scenario_result, "impact_summary", "total_demand_delta_pct") is not None:
        affected = [str(p) for p in
                    _get(scenario_result, "impact_summary", "most_affected_products",
                         default=[])][:5]
        actions.append({"action": "Use what-if scenarios for demand-sensitive products.",
                        "priority": "medium", "reason": "Declared scenarios show meaningful "
                        "demand shifts; planners should stress-test before committing.",
                        "related_products": affected})

    if confidence["level"] == "Low":
        actions.append({"action": "Treat low-confidence forecasts as decision-support signals "
                        "requiring planner review.",
                        "priority": "high", "reason": "Overall confidence is limited by "
                        "validation error and/or data-quality issues.",
                        "related_products": []})

    actions.append({"action": "Collect inventory, lead time, and capacity data before adding "
                    "inventory planning.",
                    "priority": "low", "reason": "Inventory and operational-risk features "
                    "require these inputs, which are not part of this version.",
                    "related_products": []})

    if not explainability_available:
        actions.append({"action": "Add technical explainability later for supported tree-based "
                        "models if deeper interpretation is required.",
                        "priority": "low", "reason": "Feature-level drivers (e.g. SHAP) would "
                        "explain WHY a forecast moves, complementing current performance signals.",
                        "related_products": []})

    return actions


# ─── Section: executive summary ──────────────────────────────────────────────

def _direction_phrase(model_result: Dict[str, Any]) -> Optional[str]:
    """Overall direction from total forecast vs trailing-equal-window actuals.
    Returns None when no honest comparison is possible (never guesses)."""
    history = model_result.get("history")
    total_fc = _records(model_result.get("total_forecast"))
    if not (isinstance(history, dict) and history and total_fc):
        return None
    fc_total = sum(_num(r.get("total_forecast")) or 0.0 for r in total_fc)
    n = len(total_fc)
    recent_total = 0.0
    have = False
    for s in history.values():
        tm = _tail_mean(s, n)
        if tm is not None:
            recent_total += tm * n
            have = True
    if not have or recent_total <= 0:
        return None
    change = (fc_total - recent_total) / recent_total
    if change > 0.05:
        return f"expected to increase (~{change:+.0%}) over the forecast horizon"
    if change < -0.05:
        return f"expected to decrease (~{change:+.0%}) over the forecast horizon"
    return "expected to remain broadly stable over the forecast horizon"


def _executive_summary(model_result: Dict[str, Any], confidence: Dict[str, Any],
                       high_risk: List[Dict[str, Any]], growth: List[Dict[str, Any]]
                       ) -> str:
    parts: List[str] = []
    direction = _direction_phrase(model_result)
    if direction:
        parts.append(f"Overall demand is {direction}.")
    else:
        total = _num(_get(model_result, "forecast_summary", "total_expected_demand"))
        horizon = _num(_get(model_result, "forecast_summary", "forecast_horizon"))
        if total is not None and horizon is not None:
            parts.append(f"Total forecasted demand over the next {int(horizon)} periods is "
                         f"about {total:,.0f} units.")
        else:
            parts.append("A forecast was produced across the available products.")

    parts.append(f"Forecast confidence is {confidence['level'].lower()}"
                 + (f" ({confidence['reasons'][0].rstrip('.')})." if confidence.get("reasons")
                    else "."))

    n_high = sum(1 for r in high_risk if r["risk_level"] == "high")
    if n_high:
        parts.append(f"{n_high} product(s) carry high forecast risk and need review.")
    elif high_risk:
        parts.append(f"{len(high_risk)} product(s) warrant a closer look before planning.")

    if growth and growth[0].get("basis") == "growth_vs_history":
        parts.append("The planning team should review high-growth and high-error products "
                     "before finalizing production plans.")
    else:
        parts.append("The planning team should review high-error products and treat the "
                     "forecast as decision support before finalizing production plans.")

    return " ".join(parts)


# ─── Section: chatbot context ────────────────────────────────────────────────

def _chatbot_context(executive_summary: str, high_risk: List[Dict[str, Any]],
                     model_result: Dict[str, Any], confidence: Dict[str, Any],
                     recommended_actions: List[Dict[str, Any]],
                     data_limitations: List[str], explainability: Dict[str, Any],
                     sources_used: List[str]) -> Dict[str, Any]:
    fs = model_result.get("forecast_summary") or {}
    key_metrics: Dict[str, Any] = {}
    for k in ("portfolio_wmape", "forecast_bias_direction", "confidence_label",
              "total_expected_demand", "forecast_horizon", "granularity"):
        if fs.get(k) is not None:
            key_metrics[k] = fs.get(k)

    return {
        "executive_summary": executive_summary,
        "top_risks": [{"product": r["product"], "risk_level": r["risk_level"],
                       "drivers": r["risk_drivers"]} for r in high_risk[:5]],
        "key_metrics": key_metrics,
        "recommended_actions": [a["action"] for a in recommended_actions],
        "limitations": data_limitations[:5],
        "explainability_status": explainability.get("status"),
        "confidence_level": confidence.get("level"),
        "source_fields_used": sources_used,
    }


# ─── Failure payload (shared with the pipeline integration) ──────────────────

def _failed_result(error: str) -> Dict[str, Any]:
    """The exact non-blocking failure shape (spec) — also used by pipeline."""
    return {
        "status": "failed",
        "error": str(error),
        "executive_summary": "Insights could not be generated, but the forecast "
                             "completed successfully.",
        "key_findings": [],
        "top_growth_products": [],
        "high_risk_products": [],
        "model_confidence": {"level": "Low", "reasons": [], "limitations": []},
        "forecast_quality_notes": [],
        "data_limitations": [],
        "recommended_actions": [],
        "explainability": dict(_EXPLAINABILITY_PLACEHOLDER),
        "chatbot_context": {},
    }


# ─── Explainability passthrough (future hook) ────────────────────────────────

def _explainability(explainability_result: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not explainability_result:
        return dict(_EXPLAINABILITY_PLACEHOLDER)
    return {
        "status": explainability_result.get("status", "available"),
        "method": explainability_result.get("method"),
        "top_drivers": explainability_result.get("top_drivers", []),
        "notes": explainability_result.get("notes", []),
        "limitations": explainability_result.get("limitations", []),
        "supported_model_type": explainability_result.get("supported_model_type"),
    }


# ─── Main entry point ────────────────────────────────────────────────────────

def generate_insights(
    model_result: Dict[str, Any],
    performance_result: Optional[Dict[str, Any]] = None,
    quality_result: Optional[Dict[str, Any]] = None,
    scenario_result: Optional[Dict[str, Any]] = None,
    explainability_result: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Translates validated pipeline outputs into deterministic decision-support
    insights. Read-only consumer; never mutates inputs; never raises (returns a
    'failed' payload instead). All numbers trace to structured input fields.

    Returns a JSON-serializable dict with: status, executive_summary,
    key_findings, top_growth_products, high_risk_products, model_confidence,
    forecast_quality_notes, data_limitations, recommended_actions,
    explainability, chatbot_context.
    """
    try:
        if not isinstance(model_result, dict) or not model_result:
            # No usable forecast to summarize — honest 'partial', not a crash.
            placeholder = _failed_result("model_result missing or empty")
            placeholder["status"] = "partial"
            placeholder["executive_summary"] = ("No forecast outputs were available to "
                                                "generate insights.")
            return _json_safe(placeholder)

        explainability = _explainability(explainability_result)
        explainability_available = explainability.get("status") not in (None, "not_available")

        # Track which optional inputs were actually used (for chatbot provenance).
        sources_used = ["model_result"]
        if performance_result:
            sources_used.append("performance_result")
        if quality_result:
            sources_used.append("quality_result")
        if scenario_result:
            sources_used.append("scenario_result")
        if explainability_available:
            sources_used.append("explainability_result")

        confidence = _model_confidence(model_result, performance_result, quality_result)
        growth = _top_growth_products(model_result)
        high_risk = _high_risk_products(model_result, scenario_result, confidence["level"])
        findings = _key_findings(model_result, quality_result, scenario_result,
                                 high_risk, growth, confidence, explainability_available)
        quality_notes = _forecast_quality_notes(model_result, performance_result)
        limitations = _data_limitations(model_result, quality_result, performance_result,
                                        explainability_available)
        actions = _recommended_actions(high_risk, growth, confidence, scenario_result,
                                       explainability_available)
        exec_summary = _executive_summary(model_result, confidence, high_risk, growth)
        chatbot = _chatbot_context(exec_summary, high_risk, model_result, confidence,
                                   actions, limitations, explainability, sources_used)

        # 'partial' when the core forecast exists but validation signals are thin.
        has_validation = bool(_num(_get(model_result, "forecast_summary", "portfolio_wmape"))
                              is not None or _selected_perf_by_product(model_result))
        status = "success" if has_validation else "partial"

        result = {
            "status": status,
            "executive_summary": exec_summary,
            "key_findings": findings,
            "top_growth_products": growth,
            "high_risk_products": high_risk,
            "model_confidence": confidence,
            "forecast_quality_notes": quality_notes,
            "data_limitations": limitations,
            "recommended_actions": actions,
            "explainability": explainability,
            "chatbot_context": chatbot,
        }
        return _json_safe(result)

    except Exception as exc:  # never break the pipeline (note: non-blocking)
        return _json_safe(_failed_result(f"{type(exc).__name__}: {exc}"))
