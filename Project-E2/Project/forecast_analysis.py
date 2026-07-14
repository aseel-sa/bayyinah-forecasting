"""
forecast_analysis.py — تحليل أداء التوقّع ومحاكاة السيناريوهات | Bayyina Platform

قدرتان منفصلتان في ملف واحد (قرار معتمد):

  القسم 1 — تحليل أداء التوقّع (forecast performance — مُقاس):
      فعلي مقابل متوقّع، الخطأ لكل فترة، اتجاه الانحياز (رفع/خفض)،
      وملخّص دقّة يُعاد استخدامه من مخرجات model.py كما هي.

  القسم 2 — محاكاة ماذا-لو (what-if scenario — افتراضي):
      تعديل توقّع الأساس وفق افتراضات معلنة (تغيير الطلب ٪، قوة الموسمية،
      إزاحة الذروة) ومقارنة الأساس بالسيناريو مع ملخّص أثر تجاري.

مبدأ حاكم: القسمان مستقلان تماماً — لا استدعاءات متقاطعة، ويمكن حذف أيّ
قسم دون كسر الآخر. كل مخرجات السيناريو تحمل تحذيراً إلزامياً بأنها
افتراض لا توقّعاً مؤكّداً.
"""

# ===========================================================================
# DEVELOPER NOTES
# ===========================================================================
#
# [1] One file, two sealed sections (approved design): every private helper
#     is prefixed _perf_* or _scen_*; a cross-prefix call is a review error.
#     Either section can be deleted whole; the other keeps working. The only
#     shared items are the imports and the section-agnostic warning shape
#     (the same {level, affected_scope, affected_item, message} dict used by
#     model.py warnings — duplicated per section ON PURPOSE, not shared).
#
# [2] Read-only consumers: both entry points take model_result (the
#     model.run_forecast output contract) as a plain dict and never mutate
#     it — all frames are defensively copied before any column is added.
#     This module imports NO project module (no circular-import risk).
#
# [3] Section 1 recomputes NOTHING that model.py already provides: summary
#     accuracy (metric_cards, portfolio_wmape, confidence_label, bias
#     direction, evaluation_summary) is echoed verbatim. The only new
#     numbers are the row-level error = predicted - actual (same sign
#     convention as model._metrics: positive = over-forecast), error_pct
#     (only where actual != 0 — NaN otherwise, never inf), and their
#     per-date totals. Each row is traceable 1:1 to a validation_predictions
#     row ("predicted" renames "validation_prediction" for display).
#
# [4] Section 2 is post-forecast arithmetic (phase A/B of the roadmap):
#     never retrains, never re-predicts, never touches model internals.
#     Per product (rows ordered by horizon_step):
#         level = mean(forecast); dev = forecast - level
#         dev' = roll(seasonal_strength_factor * dev, peak_shift_periods)
#         scenario = clip(multiplier * (level + dev'), 0, None)
#     Application order (seasonal reshape -> demand multiplier -> clip) is
#     part of the contract and echoed in scenario_assumptions.
#
# [5] Scenario bounds are INHERITED, not recomputed: baseline bounds are
#     shifted in parallel by (scenario - baseline) so the width — the
#     baseline's residual-based uncertainty — is preserved, then the lower
#     bound is clipped at 0. A warning states they are approximations
#     inherited from the baseline, not scenario-specific intervals.
#
# [6] Graceful degradation everywhere: missing/empty inputs return a
#     well-formed empty contract plus a structured warning — never raise.
#     Invalid scenario parameters fall back to neutral defaults, loudly.
#
# [7] The fe_output parameter of run_scenario is RESERVED for the future
#     feature-value scenario tier (weather "+2°C" re-prediction through the
#     model layer's climatology hook). Accepted now so the signature is
#     stable; unused in phase A/B.
#
# ===========================================================================

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


# ═════════════════════════════════════════════════════════════════════════════
# القسم 1 — تحليل أداء التوقّع (مُقاس)  |  Section 1 — Forecast Performance
#           (independent — deletable without affecting Section 2)
# ═════════════════════════════════════════════════════════════════════════════

_PERF_ANALYSIS_TYPE = "forecast_performance_measured"

_PERF_PER_PERIOD_COLS = ["product", "date", "fold", "horizon_step",
                         "model_name", "actual", "predicted",
                         "error", "error_pct", "direction"]
_PERF_TOTAL_COLS = ["date", "actual", "predicted", "error", "error_pct",
                    "direction", "n_products"]


def _perf_warning(level: str, item: str, message: str) -> Dict[str, str]:
    """Structured warning in the model.py shape (note [1])."""
    return {"level": level, "affected_scope": "forecast_performance",
            "affected_item": item, "message": message}


def _perf_direction(error: pd.Series) -> pd.Series:
    """Sign-based direction code: over / under / exact (positive = over)."""
    return pd.Series(np.where(error > 0, "over",
                              np.where(error < 0, "under", "exact")),
                     index=error.index)


def _perf_error_pct(error: pd.Series, actual: pd.Series) -> pd.Series:
    """Percentage error ONLY where mathematically valid (actual != 0);
    NaN otherwise — never inf (note [3])."""
    with np.errstate(divide="ignore", invalid="ignore"):
        pct = np.where(actual != 0, error / actual * 100.0, np.nan)
    return pd.Series(pct, index=error.index)


def _perf_empty(warnings: List[Dict[str, str]]) -> Dict[str, Any]:
    """Well-formed empty contract (note [6])."""
    return {
        "analysis_type": _PERF_ANALYSIS_TYPE,
        "per_period": pd.DataFrame(columns=_PERF_PER_PERIOD_COLS),
        "per_period_total": pd.DataFrame(columns=_PERF_TOTAL_COLS),
        "summary": {"portfolio_wmape": None, "forecast_bias_direction": None,
                    "confidence_label": None, "metric_cards": [],
                    "evaluation_summary": {}, "n_validation_rows": 0,
                    "n_products_evaluated": 0, "n_periods_evaluated": 0,
                    "n_over_forecast": 0, "n_under_forecast": 0, "n_exact": 0},
        "warnings": warnings,
    }


def analyze_forecast_performance(model_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Forecast performance analysis — MEASURED accuracy (actual vs predicted).

    Consumes model.run_forecast output read-only. Derives the per-period
    error view (the "demand, prediction, and error" chart) from
    validation_predictions and echoes — never recomputes — the summary
    accuracy already in the contract (note [3]).

    Returns {analysis_type, per_period, per_period_total, summary, warnings}.
      per_period       : product, date, fold, horizon_step, model_name,
                         actual, predicted, error (=predicted-actual),
                         error_pct (NaN when actual=0), direction
      per_period_total : the same view pooled across products per date
      summary          : echoed metric_cards / portfolio metrics + row counts
    Degrades to an empty contract with a warning when no validation history
    exists (short-history datasets) — never raises.
    """
    mr = model_result or {}
    vp = mr.get("validation_predictions")
    if vp is None or not isinstance(vp, pd.DataFrame) or len(vp) == 0:
        return _perf_empty([_perf_warning(
            "warning", "ALL",
            "No validation history available (insufficient history for "
            "walk-forward validation) — actual-vs-predicted performance "
            "cannot be measured.")])

    warnings: List[Dict[str, str]] = []
    pp = vp.copy(deep=True)
    pp = pp.rename(columns={"validation_prediction": "predicted"})
    pp["error"] = pp["predicted"] - pp["actual"]          # positive = over
    pp["error_pct"] = _perf_error_pct(pp["error"], pp["actual"])
    pp["direction"] = _perf_direction(pp["error"])
    keep = [c for c in _PERF_PER_PERIOD_COLS if c in pp.columns]
    pp = pp[keep].sort_values(["product", "date", "fold"]).reset_index(drop=True)

    # Pooled across products per date (the "ALL" view of the same rows).
    tot = (pp.groupby("date", as_index=False)
             .agg(actual=("actual", "sum"), predicted=("predicted", "sum"),
                  n_products=("product", "nunique")))
    tot["error"] = tot["predicted"] - tot["actual"]
    tot["error_pct"] = _perf_error_pct(tot["error"], tot["actual"])
    tot["direction"] = _perf_direction(tot["error"])
    tot = tot[_PERF_TOTAL_COLS].sort_values("date").reset_index(drop=True)

    fs = mr.get("forecast_summary") or {}
    summary = {
        # Echoed verbatim from the model contract — no recomputation (note [3]).
        "portfolio_wmape": fs.get("portfolio_wmape"),
        "forecast_bias_direction": fs.get("forecast_bias_direction"),
        "confidence_label": fs.get("confidence_label"),
        "metric_cards": [dict(c) for c in (mr.get("metric_cards") or [])],
        "evaluation_summary": dict(mr.get("evaluation_summary") or {}),
        # New, traceable row counts over per_period.
        "n_validation_rows": int(len(pp)),
        "n_products_evaluated": int(pp["product"].nunique()),
        "n_periods_evaluated": int(pp["date"].nunique()),
        "n_over_forecast": int((pp["direction"] == "over").sum()),
        "n_under_forecast": int((pp["direction"] == "under").sum()),
        "n_exact": int((pp["direction"] == "exact").sum()),
    }

    return {"analysis_type": _PERF_ANALYSIS_TYPE, "per_period": pp,
            "per_period_total": tot, "summary": summary, "warnings": warnings}


# ═════════════════════════════════════════════════════════════════════════════
# القسم 2 — محاكاة ماذا-لو (افتراضي)  |  Section 2 — Scenario Simulation
#           (independent — deletable without affecting Section 1)
# ═════════════════════════════════════════════════════════════════════════════

_SCEN_ANALYSIS_TYPE = "scenario_simulation_assumption"

_SCEN_KNOWN_KEYS = {"name", "demand_change_pct", "per_product_change_pct",
                    "seasonal_strength_factor", "peak_shift_periods"}

_SCEN_APPLICATION_ORDER = ("seasonal_strength_factor -> peak_shift_periods "
                           "-> demand_change_pct -> clip(>=0)")

_SCEN_DISCLAIMER = ("This is a hypothetical scenario result built on declared "
                    "assumptions — it is NOT a confirmed prediction and does "
                    "not modify the baseline forecast.")


def _scen_warning(level: str, item: str, message: str) -> Dict[str, str]:
    """Structured warning in the model.py shape (note [1])."""
    return {"level": level, "affected_scope": "scenario",
            "affected_item": item, "message": message}


def _scen_disclaimer_warning() -> Dict[str, str]:
    """The mandatory assumption-not-prediction label — present in EVERY
    scenario output, including identity and degraded ones."""
    return _scen_warning("warning", "disclaimer", _SCEN_DISCLAIMER)


def _scen_resolve(scenario: Optional[Dict[str, Any]],
                  known_products: List[str],
                  warnings: List[Dict[str, str]]) -> Dict[str, Any]:
    """
    Validates the user's declared assumptions; invalid values fall back to
    neutral defaults LOUDLY (note [6]). Returns the resolved assumptions dict
    that is also echoed in the output contract.
    """
    sc = dict(scenario or {})

    unknown = sorted(set(sc) - _SCEN_KNOWN_KEYS)
    if unknown:
        warnings.append(_scen_warning(
            "warning", ",".join(unknown),
            f"Unknown scenario keys were ignored: {unknown}"))

    def _num(key: str, default: float) -> float:
        v = sc.get(key, default)
        try:
            v = float(v)
            if not np.isfinite(v):
                raise ValueError
            return v
        except (TypeError, ValueError):
            warnings.append(_scen_warning(
                "warning", key,
                f"Invalid value for {key} ({sc.get(key)!r}) — "
                f"default {default} used instead."))
            return default

    demand_pct = _num("demand_change_pct", 0.0)
    factor = _num("seasonal_strength_factor", 1.0)
    if factor < 0:
        warnings.append(_scen_warning(
            "warning", "seasonal_strength_factor",
            f"Negative seasonal strength factor ({factor}) is not accepted — 1.0 used."))
        factor = 1.0
    shift = int(round(_num("peak_shift_periods", 0)))

    per_product: Dict[str, float] = {}
    raw_pp = sc.get("per_product_change_pct") or {}
    if not isinstance(raw_pp, dict):
        warnings.append(_scen_warning(
            "warning", "per_product_change_pct",
            "per_product_change_pct must be a {product: percent} dict — ignored."))
        raw_pp = {}
    known = set(known_products)
    for prod, pct in raw_pp.items():
        if str(prod) not in known:
            warnings.append(_scen_warning(
                "warning", str(prod),
                f"Product '{prod}' does not exist in the baseline forecast — its adjustment was ignored."))
            continue
        try:
            per_product[str(prod)] = float(pct)
        except (TypeError, ValueError):
            warnings.append(_scen_warning(
                "warning", str(prod),
                f"Invalid percentage for product '{prod}' ({pct!r}) — ignored."))

    return {
        "name": sc.get("name"),
        "demand_change_pct": demand_pct,
        "per_product_change_pct": per_product,
        "seasonal_strength_factor": factor,
        "peak_shift_periods": shift,
        "method": "post_forecast_adjustment",
        "application_order": _SCEN_APPLICATION_ORDER,
    }


def _scen_transform(forecast: np.ndarray, factor: float, shift: int,
                    multiplier: float) -> np.ndarray:
    """The documented per-product transform (note [4]): seasonal reshape
    around the horizon mean level, then the demand multiplier, then clip."""
    level = forecast.mean()
    dev = np.roll(factor * (forecast - level), shift)
    return np.clip(multiplier * (level + dev), 0, None)


def _scen_empty(assumptions: Dict[str, Any],
                warnings: List[Dict[str, str]]) -> Dict[str, Any]:
    """Well-formed empty contract — still carries the disclaimer (note [6])."""
    empty_fc = pd.DataFrame(columns=["product", "date", "forecast",
                                     "lower_bound", "upper_bound", "model_used",
                                     "horizon_step", "forecast_start_date"])
    return {
        "analysis_type": _SCEN_ANALYSIS_TYPE,
        "baseline_forecast": empty_fc.copy(),
        "scenario_forecast": empty_fc.copy(),
        "scenario_comparison": pd.DataFrame(
            columns=["date", "baseline_total", "scenario_total",
                     "delta", "delta_pct"]),
        "impact_summary": {"total_demand_baseline": 0.0,
                           "total_demand_scenario": 0.0,
                           "total_demand_delta": 0.0,
                           "total_demand_delta_pct": None,
                           "peak_period_baseline": None,
                           "peak_period_scenario": None,
                           "most_affected_products": []},
        "scenario_assumptions": assumptions,
        "warnings": warnings,
    }


def run_scenario(model_result: Dict[str, Any],
                 scenario: Optional[Dict[str, Any]],
                 fe_output: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    What-if scenario simulation — ASSUMPTION-based forecast adjustment.

    Consumes the baseline model.run_forecast output read-only (the baseline
    is NEVER mutated — note [2]) and applies the user's declared assumptions
    as post-forecast arithmetic (note [4]). Never retrains or re-predicts.

    scenario keys (all optional; identity scenario == baseline):
      name                     : free label, echoed back
      demand_change_pct        : global ±% (e.g. 10 → ×1.10, -15 → ×0.85)
      per_product_change_pct   : {product: ±%} — overrides the global % for
                                 those products
      seasonal_strength_factor : 1.0 neutral; >1 amplifies the seasonal
                                 deviation around each product's horizon mean;
                                 <1 dampens it
      peak_shift_periods       : roll the seasonal deviation ±k periods
                                 (positive = peak arrives later)

    fe_output is reserved for the future feature-value tier (note [7]).

    Returns {analysis_type, baseline_forecast, scenario_forecast,
    scenario_comparison, impact_summary, scenario_assumptions, warnings}.
    Every output carries the mandatory assumption-not-prediction disclaimer.
    """
    warnings: List[Dict[str, str]] = [_scen_disclaimer_warning()]
    mr = model_result or {}
    pf = mr.get("product_forecasts")

    if pf is None or not isinstance(pf, pd.DataFrame) or len(pf) == 0:
        assumptions = _scen_resolve(scenario, [], warnings)
        warnings.append(_scen_warning(
            "warning", "ALL",
            "No baseline forecast available (product_forecasts is empty) — a scenario cannot be built."))
        return _scen_empty(assumptions, warnings)

    baseline = pf.copy(deep=True)
    products = [str(p) for p in baseline["product"].unique()]
    a = _scen_resolve(scenario, products, warnings)

    global_mult = 1.0 + a["demand_change_pct"] / 100.0
    factor, shift = a["seasonal_strength_factor"], a["peak_shift_periods"]

    # ── Per-product transform on a defensive copy (notes [2],[4]) ──
    scen = baseline.copy(deep=True)
    for prod, g in scen.groupby("product", sort=False):
        order = g.sort_values("horizon_step").index
        f = scen.loc[order, "forecast"].to_numpy(dtype=float)
        mult = 1.0 + a["per_product_change_pct"].get(str(prod),
                                                     a["demand_change_pct"]) / 100.0
        scen.loc[order, "forecast"] = _scen_transform(f, factor, shift, mult)

    # ── Bounds: inherited from the baseline, shifted in parallel (note [5]) ──
    has_bounds = ("lower_bound" in scen.columns and "upper_bound" in scen.columns)
    if has_bounds:
        delta_vals = scen["forecast"] - baseline["forecast"]
        scen["lower_bound"] = np.clip(baseline["lower_bound"] + delta_vals, 0, None)
        scen["upper_bound"] = baseline["upper_bound"] + delta_vals
        if np.isfinite(baseline["lower_bound"].to_numpy(dtype=float)).any():
            warnings.append(_scen_warning(
                "info", "bounds",
                "Scenario bounds are inherited from the baseline bounds "
                "(parallel shift) — approximations, not scenario-specific "
                "confidence intervals."))

    # ── Comparison per date: baseline vs scenario totals ──
    base_tot = (baseline.groupby("date", as_index=False)
                        .agg(baseline_total=("forecast", "sum")))
    scen_tot = (scen.groupby("date", as_index=False)
                    .agg(scenario_total=("forecast", "sum")))
    comp = base_tot.merge(scen_tot, on="date", how="outer").sort_values("date")
    comp["delta"] = comp["scenario_total"] - comp["baseline_total"]
    with np.errstate(divide="ignore", invalid="ignore"):
        comp["delta_pct"] = np.where(comp["baseline_total"] != 0,
                                     comp["delta"] / comp["baseline_total"] * 100.0,
                                     np.nan)
    comp = comp.reset_index(drop=True)

    # ── Business impact summary ──
    base_sum = float(baseline["forecast"].sum())
    scen_sum = float(scen["forecast"].sum())
    per_prod = (baseline.groupby("product")["forecast"].sum().rename("baseline_total")
                .to_frame()
                .join(scen.groupby("product")["forecast"].sum().rename("scenario_total")))
    per_prod["delta"] = per_prod["scenario_total"] - per_prod["baseline_total"]
    with np.errstate(divide="ignore", invalid="ignore"):
        per_prod["delta_pct"] = np.where(
            per_prod["baseline_total"] != 0,
            per_prod["delta"] / per_prod["baseline_total"] * 100.0, np.nan)
    affected = (per_prod.reindex(per_prod["delta"].abs()
                                 .sort_values(ascending=False).index)
                .head(5).reset_index())
    most_affected = [
        {"product": str(r["product"]),
         "baseline_total": float(r["baseline_total"]),
         "scenario_total": float(r["scenario_total"]),
         "delta": float(r["delta"]),
         "delta_pct": (float(r["delta_pct"])
                       if np.isfinite(r["delta_pct"]) else None)}
        for _, r in affected.iterrows()]

    impact = {
        "total_demand_baseline": base_sum,
        "total_demand_scenario": scen_sum,
        "total_demand_delta": scen_sum - base_sum,
        "total_demand_delta_pct": ((scen_sum - base_sum) / base_sum * 100.0
                                   if base_sum != 0 else None),
        "peak_period_baseline": (comp.loc[comp["baseline_total"].idxmax(), "date"]
                                 if len(comp) else None),
        "peak_period_scenario": (comp.loc[comp["scenario_total"].idxmax(), "date"]
                                 if len(comp) else None),
        "most_affected_products": most_affected,
    }

    return {
        "analysis_type": _SCEN_ANALYSIS_TYPE,
        "baseline_forecast": baseline,      # untouched copy — echo (note [2])
        "scenario_forecast": scen,
        "scenario_comparison": comp,
        "impact_summary": impact,
        "scenario_assumptions": a,
        "warnings": warnings,
    }
