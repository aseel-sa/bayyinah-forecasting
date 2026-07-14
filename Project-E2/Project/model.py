"""
model.py — Forecasting Engine | Bayyina Platform

Position: after feature_engineering.py. This is the forecasting core of the
product, not a model script.

Owns: model execution/training/evaluation/comparison, forecast generation,
output formatting, fallback logic, business-friendly metric interpretation.
Does NOT own: feature creation (feature_engineering does), UI, LLM insights.

Main entry point: run_forecast(...) → the full output contract (see DEV NOTES).
"""

# ===========================================================================
# DEVELOPER NOTES
# ===========================================================================
#
# [1] No internal feature creation (non-negotiable): the main path consumes
#     feature_engineering output (fe_output: feature_matrices + metadata).
#     When fe_output is not passed (direct calls), build_features is invoked
#     ONCE — never per fold. Recursive multi-step prediction reuses
#     fe._add_temporal_features/_add_seasonal_features so train and predict
#     share one feature definition.
#
# [2] Two horizons, separated by design:
#       validation_horizon — steps per walk-forward fold (default 3);
#       forecast_horizon   — final user-facing forecast (default 12 monthly /
#                            26 weekly). Both user-overridable.
#
# [3] Scalable walk-forward: features are built ONCE; each fold slices matrix
#     ROWS at the fold cutoff. Lag/rolling values only look backward, so row
#     slicing is leakage-free. KNOWN CAVEAT: detected-seasonality flags and
#     panel-level feature values are derived from the full history (static-
#     feature compromise); the prior per-fold full rebuild was pure but O(folds
#     × products) feature builds — impossible at 1000+ products. Documented
#     trade-off, deliberately taken.
#
# [4] Safety: every model runs through a wrapper. Missing package → status
#     "skipped" (reason package_not_available); runtime error → status
#     "failed" + message; the pipeline NEVER crashes because a model died.
#     naive always works with ≥1 observation → a usable forecast always exists.
#
# [5] Selection: primary wMAPE, tie-break |forecast_bias|, fallback MAE.
#     Never MAPE-only (unstable near zero demand). Simple models win when
#     they score better — complexity is not rewarded.
#
# [6] hybrid_ensemble: inverse-wMAPE weighted blend of the top ≤3 valid models
#     per product. Its validation predictions are blended from the components'
#     STORED fold predictions (no retraining). Skipped when <2 valid
#     components. Never forced as winner.
#
# [7] External future availability (from fe feature_metadata):
#       scenario_only / historical_context → EXCLUDED from training features
#         (cannot be known in production forecasting; scenario engine later);
#       requires_future_values → kept in training; future steps are filled
#         with per-product CLIMATOLOGY (same-calendar-month historical mean) —
#         NEVER silently zero-filled;
#       safe_for_future → used normally.
#     Every exclusion/adjustment emits a structured warning.
#
# [8] Metric thresholds (metric_cards labels) are engineering judgments, NOT
#     universal truths: wMAPE/MAPE ≤10% excellent, ≤20% good, ≤35%
#     needs_attention, else poor; |bias| ≤5% balanced/excellent, ≤15% good,
#     ≤30% needs_attention; MAE/RMSE labeled relative to mean demand
#     (≤15%/30%/50%). Calibrate per business once real accuracy data exists.
#
# [9] Confidence intervals are residual-based (forecast ± 1.28·validation
#     RMSE ≈ 80% under normality) — approximate and honest, available for any
#     model with validation history. Summed bounds for the portfolio total
#     OVERSTATE uncertainty (no diversification effect) — documented caveat.
#
# [10] Totals: bottom_up_total (sum of product forecasts) always;
#      direct_total_forecast (models trained on the aggregated series) also
#      implemented — cheap because it is a single series.
#
# [11] LSTM is metadata-only (status future_extension) — not implemented:
#      needs larger datasets, tuning, and runtime validation.
#
# [12] Backward-compat aliases preserved: forecasts (dict product→Series),
#      comparison_table (product/model/mape/rmse/mae/wmape/n_folds),
#      summary_table, history (dict product→Series), best_model.
#
# ===========================================================================

import time
import warnings
from importlib.util import find_spec
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import feature_engineering as fe

warnings.filterwarnings("ignore")

# Package availability — checked once; missing → models skipped, never crash.
_HAS = {
    "statsmodels": find_spec("statsmodels") is not None,
    "prophet": find_spec("prophet") is not None,
    "xgboost": find_spec("xgboost") is not None,
    "sklearn": find_spec("sklearn") is not None,
    "statsforecast": find_spec("statsforecast") is not None,
}

_DEFAULT_FORECAST_H = {"monthly": 12, "weekly": 26}
_DEFAULT_VALIDATION_H = 3
_MA_WINDOW = {"monthly": 3, "weekly": 4}
_CI_Z = 1.28  # ≈80% interval under normality (note [9])

# Model registry: requirements + tier. 'ml' models need the tree matrix.
_MODEL_SPECS: Dict[str, Dict[str, Any]] = {
    "naive":                 dict(requires=None, kind="fast"),
    "seasonal_naive":        dict(requires=None, kind="fast"),
    "moving_average":        dict(requires=None, kind="fast"),
    "exponential_smoothing": dict(requires=None, kind="fast"),
    "sarima":                dict(requires="statsmodels", kind="stat"),
    "prophet":               dict(requires="prophet", kind="stat"),
    "croston":               dict(requires="statsforecast", kind="stat"),
    "random_forest":         dict(requires="sklearn", kind="ml"),
    "xgboost":               dict(requires="xgboost", kind="ml"),
}
_FAST = ["naive", "seasonal_naive", "moving_average", "exponential_smoothing"]
_ADVANCED = ["sarima", "prophet", "random_forest", "xgboost"]

# Fixed ordinal codes for the categorical seasonal pattern (sklearn needs numbers).
_PATTERN_CODES = {"summer_peak": 0, "winter_peak": 1, "shoulder_peak": 2,
                  "year_round": 3, "irregular": 4, "undetected": 5}

_MODE_FOLDS = {"fast": 2, "balanced": 3, "full": 4}


def _freq_of(granularity: str) -> str:
    return fe._GRAN_CONFIG[granularity]["freq"]


def _season_of(granularity: str) -> int:
    return fe._GRAN_CONFIG[granularity]["season"]


# ─── Metrics (note [5]) ──────────────────────────────────────────────────────

def _metrics(actual: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    """MAE / RMSE / MAPE (zero-skipping) / wMAPE / forecast_bias from arrays."""
    actual = np.asarray(actual, dtype=float)
    pred = np.asarray(pred, dtype=float)
    err = pred - actual
    abs_err = np.abs(err)
    nz = actual != 0
    denom = float(np.abs(actual).sum())
    return {
        "mae": float(abs_err.mean()),
        "rmse": float(np.sqrt((err ** 2).mean())),
        "mape": float((abs_err[nz] / np.abs(actual[nz])).mean()) if nz.any() else np.inf,
        "wmape": float(abs_err.sum() / denom) if denom > 0 else np.inf,
        "forecast_bias": float(err.sum() / denom) if denom > 0 else np.inf,
    }


def _label_pct(v: float) -> str:
    """wMAPE/MAPE label (thresholds: note [8])."""
    if not np.isfinite(v):
        return "poor"
    return ("excellent" if v <= 0.10 else "good" if v <= 0.20
            else "needs_attention" if v <= 0.35 else "poor")


def _label_bias(v: float) -> str:
    if not np.isfinite(v):
        return "poor"
    a = abs(v)
    return ("excellent" if a <= 0.05 else "good" if a <= 0.15
            else "needs_attention" if a <= 0.30 else "poor")


def _label_relative(v: float, mean_demand: float) -> str:
    """MAE/RMSE label relative to mean demand (no universal unit scale)."""
    if not np.isfinite(v) or mean_demand <= 0:
        return "needs_attention"
    r = v / mean_demand
    return ("excellent" if r <= 0.15 else "good" if r <= 0.30
            else "needs_attention" if r <= 0.50 else "poor")


# ─── Model runners (uniform: history in → h-step prediction out) ─────────────
# Each raises on failure; the safe wrapper converts to failed/skipped status.

def _predict_naive(y: pd.Series, h: int, season: int) -> np.ndarray:
    return np.full(h, float(y.iloc[-1]))


def _predict_seasonal_naive(y: pd.Series, h: int, season: int) -> np.ndarray:
    """Repeats the last full seasonal cycle; falls back to naive when shorter."""
    if len(y) < season:
        return _predict_naive(y, h, season)
    last_cycle = y.iloc[-season:].values
    return np.array([last_cycle[i % season] for i in range(h)], dtype=float)


def _predict_moving_average(y: pd.Series, h: int, season: int,
                            window: int = 3) -> np.ndarray:
    w = min(window, len(y))
    return np.full(h, float(y.iloc[-w:].mean()))


def _predict_ses(y: pd.Series, h: int, season: int, alpha: float = 0.3) -> np.ndarray:
    """Simple exponential smoothing, dependency-free. Flat multi-step forecast."""
    level = float(y.iloc[0])
    for v in y.iloc[1:]:
        level = alpha * float(v) + (1 - alpha) * level
    return np.full(h, level)


def _predict_sarima(y: pd.Series, h: int, season: int) -> np.ndarray:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    if len(y) < 2 * season + 4:
        raise ValueError(f"history too short for SARIMA (<{2 * season + 4})")
    fit = SARIMAX(y, order=(1, 1, 1), seasonal_order=(1, 1, 1, season),
                  enforce_stationarity=False, enforce_invertibility=False
                  ).fit(disp=False, method="lbfgs", maxiter=150)
    return np.asarray(fit.forecast(steps=h), dtype=float)


def _predict_prophet(y: pd.Series, h: int, season: int, freq: str = "MS") -> np.ndarray:
    from prophet import Prophet
    m = Prophet(yearly_seasonality="auto", weekly_seasonality=False,
                daily_seasonality=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.fit(pd.DataFrame({"ds": y.index, "y": y.values}))
    future = m.make_future_dataframe(periods=h, freq=freq)
    return m.predict(future)["yhat"].iloc[-h:].values.astype(float)


def _predict_croston(y: pd.Series, h: int, season: int, freq: str = "MS") -> np.ndarray:
    from statsforecast import StatsForecast
    from statsforecast.models import CrostonClassic
    sf = StatsForecast(models=[CrostonClassic()], freq=freq, verbose=False)
    sf.fit(pd.DataFrame({"unique_id": "p", "ds": y.index,
                         "y": y.values.astype(float)}))
    pred = sf.predict(h=h)
    col = [c for c in pred.columns if c not in ("unique_id", "ds")][0]
    return pred[col].values.astype(float)


def _encode_tree(frame: pd.DataFrame, feature_cols: List[str]) -> np.ndarray:
    """Numeric matrix in exact feature_cols order; categorical pattern → fixed
    ordinal codes; residual NaN of SAFE internal features → 0."""
    d = frame.copy()
    if "seasonal_pattern_type" in d.columns:
        d["seasonal_pattern_type"] = (d["seasonal_pattern_type"].astype(str)
                                      .map(_PATTERN_CODES)
                                      .fillna(_PATTERN_CODES["undetected"]))
    X = d.reindex(columns=feature_cols)
    return X.apply(pd.to_numeric, errors="coerce").fillna(0.0).astype(float).values


def _climatology(frame: pd.DataFrame, cols: List[str]) -> Dict[str, Dict[Any, float]]:
    """Per-feature {calendar_month → historical mean} (+'__mean__' fallback) —
    the explicit future-fill strategy for requires_future_values (note [7])."""
    clim: Dict[str, Dict[Any, float]] = {}
    months = frame.index.month
    for col in cols:
        if col not in frame.columns:
            continue
        v = pd.to_numeric(frame[col], errors="coerce")
        by_month = v.groupby(months).mean().to_dict()
        by_month["__mean__"] = float(v.mean()) if v.notna().any() else 0.0
        clim[col] = by_month
    return clim


def _predict_tree(learner: Any, tree_frame: pd.DataFrame, h: int,
                  granularity: str, feature_cols: List[str],
                  future_fill_cols: List[str], sector: Optional[str]) -> np.ndarray:
    """
    Trains the learner on the (already sliced) tree frame and forecasts h steps
    recursively. Future values of requires_future_values features come from
    climatology — never zero (note [7]). Temporal/seasonal features are rebuilt
    each step via fe helpers (one feature definition — note [1]).
    """
    tcol = "y"
    train = tree_frame.dropna(subset=[c for c in feature_cols
                                      if tree_frame[c].dtype != object
                                      and c != "seasonal_pattern_type"])
    if len(train) < 5:
        raise ValueError("fewer than 5 trainable rows (lags incomplete)")
    X = _encode_tree(train, feature_cols)
    learner.fit(X, pd.to_numeric(train[tcol], errors="coerce").fillna(0.0).values)

    clim = _climatology(tree_frame, future_fill_cols)
    cfg = fe._GRAN_CONFIG[granularity]
    offset = pd.tseries.frequencies.to_offset(cfg["freq"])
    y_hist = pd.Series(pd.to_numeric(tree_frame[tcol], errors="coerce").values,
                       index=tree_frame.index).sort_index()
    extended = y_hist.copy()
    preds: List[float] = []

    for _ in range(h):
        panel = pd.DataFrame({"product": "p", "date": extended.index,
                              "y": extended.values})
        panel = fe._add_temporal_features(panel, cfg, "product", "date")
        panel, _ = fe._add_seasonal_features(panel, "product", sector)
        row = panel.iloc[[-1]].copy()
        step_month = int(pd.Timestamp(extended.index[-1] + offset).month)
        for col in future_fill_cols:  # climatology, not zero (note [7])
            if col in clim:
                row[col] = clim[col].get(step_month, clim[col]["__mean__"])
        pred = max(float(learner.predict(_encode_tree(row, feature_cols))[0]), 0.0)
        preds.append(pred)
        extended = pd.concat([extended,
                              pd.Series([pred], index=[extended.index[-1] + offset])])
    return np.array(preds)


def _make_learner(model_name: str):
    if model_name == "random_forest":
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    import xgboost as xgb
    return xgb.XGBRegressor(n_estimators=100, random_state=42, verbosity=0)


def _safe_predict(model_name: str, y: pd.Series, h: int, granularity: str,
                  tree_ctx: Optional[Dict[str, Any]] = None) -> np.ndarray:
    """Dispatches to the right runner. Raises on real failure; the caller's
    wrapper converts exceptions to failed status (note [4])."""
    season = _season_of(granularity)
    freq = _freq_of(granularity)
    if model_name == "naive":
        return _predict_naive(y, h, season)
    if model_name == "seasonal_naive":
        return _predict_seasonal_naive(y, h, season)
    if model_name == "moving_average":
        return _predict_moving_average(y, h, season, _MA_WINDOW[granularity])
    if model_name == "exponential_smoothing":
        return _predict_ses(y, h, season)
    if model_name == "sarima":
        return _predict_sarima(y, h, season)
    if model_name == "prophet":
        return _predict_prophet(y, h, season, freq)
    if model_name == "croston":
        return _predict_croston(y, h, season, freq)
    if model_name in ("random_forest", "xgboost"):
        if tree_ctx is None:
            raise ValueError("tree context missing for ML model")
        frame = tree_ctx["frame"]
        cutoff = y.index[-1]
        return _predict_tree(_make_learner(model_name),
                             frame[frame.index <= cutoff], h, granularity,
                             tree_ctx["feature_cols"], tree_ctx["future_fill_cols"],
                             tree_ctx["sector"])
    raise ValueError(f"unknown model {model_name}")


# ─── Segmentation ────────────────────────────────────────────────────────────

def segment_products(series_map: Dict[str, pd.Series], season: int
                     ) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Classifies products into high_volume / normal / low_volume / intermittent /
    short_history. Order of precedence: short_history → intermittent → volume.
    high_volume = products jointly covering the top 50% of total demand;
    low_volume = bottom 20% of per-product totals.
    """
    rows = []
    totals = {p: float(s.clip(lower=0).sum()) for p, s in series_map.items()}
    sorted_p = sorted(totals, key=totals.get, reverse=True)
    grand = sum(totals.values()) or 1.0
    cum, high_set = 0.0, set()
    for p in sorted_p:
        if cum < 0.5 * grand:
            high_set.add(p)
        cum += totals[p]
    low_cut = np.percentile(list(totals.values()), 20) if len(totals) >= 5 else -1

    segments: Dict[str, str] = {}
    for p, s in series_map.items():
        zero_ratio = float((s == 0).mean())
        if len(s) < season:
            seg = "short_history"
        elif zero_ratio > 0.35:
            seg = "intermittent"
        elif p in high_set:
            seg = "high_volume"
        elif totals[p] <= low_cut:
            seg = "low_volume"
        else:
            seg = "normal"
        segments[p] = seg
        rows.append({"product": p, "segment": seg, "n_points": len(s),
                     "total_demand": totals[p], "zero_ratio": round(zero_ratio, 3)})
    return pd.DataFrame(rows), segments


def _candidates_for(segment: str, mode: str) -> List[str]:
    """Segment- and mode-aware candidate sets (scalability rule: no expensive
    models for everyone). Availability filtering happens at execution time."""
    if segment == "short_history":
        return ["naive", "moving_average"]
    if segment == "intermittent":
        # croston stays a candidate even when statsforecast is missing — the
        # validator marks it 'skipped' so the leaderboard shows WHY (spec rule).
        return ["naive", "seasonal_naive", "moving_average", "croston"]
    if segment == "low_volume":
        return ["naive", "seasonal_naive", "moving_average"]
    base = list(_FAST)
    advanced_ok = (mode == "full") or (mode == "balanced" and segment == "high_volume")
    if advanced_ok:
        base += list(_ADVANCED)  # availability handled by the validator → 'skipped'
    return base


# ─── Walk-forward validation (note [3]) ──────────────────────────────────────

def _fold_cutoffs(n: int, season: int, vh: int, n_splits: int) -> List[int]:
    """Expanding-window fold cutoff positions; [] when history is insufficient."""
    min_train = season + 2 if n >= season + 2 + vh else max(4, n - 2 * vh)
    if n < min_train + vh:
        return []
    available = n - min_train
    splits = min(n_splits, available // vh)
    if splits < 1:
        return []
    step = max(vh, (available - vh) // max(splits, 1))
    return [min_train + i * step for i in range(splits) if min_train + i * step + vh <= n]


def _validate_product(product: str, y: pd.Series, candidates: List[str],
                      granularity: str, vh: int, n_splits: int,
                      tree_ctx: Optional[Dict[str, Any]],
                      failed_models: List[Dict[str, Any]]
                      ) -> Dict[str, Dict[str, Any]]:
    """
    Walk-forward validation of all candidates for one product. Train only on
    the past, test the next vh periods, advance, repeat. Returns per model:
    {rows: [fold rows], metrics, status, train_seconds}. Failures are recorded,
    never raised (note [4]).
    """
    season = _season_of(granularity)
    cutoffs = _fold_cutoffs(len(y), season, vh, n_splits)
    results: Dict[str, Dict[str, Any]] = {}

    for name in candidates:
        spec = _MODEL_SPECS[name]
        if spec["requires"] and not _HAS[spec["requires"]]:
            results[name] = {"rows": [], "metrics": {}, "status": "skipped",
                             "reason": "package_not_available", "train_seconds": 0.0}
            failed_models.append({"model_name": name, "product": product,
                                  "reason": "package_not_available",
                                  "status": "skipped"})
            continue
        if not cutoffs:
            results[name] = {"rows": [], "metrics": {}, "status": "insufficient_history",
                             "reason": "not enough history for walk-forward",
                             "train_seconds": 0.0}
            continue

        rows: List[Dict[str, Any]] = []
        t0 = time.perf_counter()
        fold_errors: List[str] = []
        ok_folds = 0
        # Per-fold tolerance: an early fold may lack trainable rows for ML
        # models (lags eat the start) while later, longer folds succeed.
        # The model fails only when EVERY fold fails (note [4]).
        for fold, cut in enumerate(cutoffs):
            train_y = y.iloc[:cut]
            actual = y.iloc[cut:cut + vh]
            try:
                preds = _safe_predict(name, train_y, len(actual), granularity, tree_ctx)
            except Exception as exc:
                fold_errors.append(f"fold {fold}: {type(exc).__name__}: {exc}")
                continue
            ok_folds += 1
            for step, (dt, a, pr) in enumerate(zip(actual.index, actual.values, preds), 1):
                rows.append({"product": product, "model_name": name, "fold": fold,
                             "horizon_step": step, "date": dt,
                             "actual": float(a),
                             "validation_prediction": max(float(pr), 0.0)})
        elapsed = round(time.perf_counter() - t0, 3)

        if ok_folds == 0:
            reason = fold_errors[-1] if fold_errors else "no folds executed"
            results[name] = {"rows": [], "metrics": {}, "status": "failed",
                             "reason": reason, "train_seconds": elapsed}
            failed_models.append({"model_name": name, "product": product,
                                  "reason": reason, "status": "failed"})
        else:
            m = _metrics(np.array([r["actual"] for r in rows]),
                         np.array([r["validation_prediction"] for r in rows]))
            m["n_folds"] = ok_folds
            results[name] = {"rows": rows, "metrics": m, "status": "ok",
                             "reason": "; ".join(fold_errors) if fold_errors else "",
                             "train_seconds": elapsed}
    return results


def _add_hybrid(results: Dict[str, Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    hybrid_ensemble from stored fold predictions (note [6]): top ≤3 valid models
    weighted by inverse wMAPE; blended predictions on the intersection of their
    fold rows. Returns its result entry + details, or None when <2 components.
    """
    valid = {n: r for n, r in results.items()
             if r["status"] == "ok" and np.isfinite(r["metrics"].get("wmape", np.inf))
             and r["metrics"]["wmape"] > 0}
    if len(valid) < 2:
        return None
    top = sorted(valid, key=lambda n: valid[n]["metrics"]["wmape"])[:3]
    inv = {n: 1.0 / valid[n]["metrics"]["wmape"] for n in top}
    total = sum(inv.values())
    weights = {n: round(v / total, 4) for n, v in inv.items()}

    # Blend on (fold, horizon_step) keys shared by ALL components.
    keyed = {n: {(r["fold"], r["horizon_step"]): r for r in valid[n]["rows"]} for n in top}
    common = set.intersection(*[set(k) for k in keyed.values()])
    if not common:
        return None
    rows = []
    for key in sorted(common):
        base = keyed[top[0]][key]
        blended = sum(weights[n] * keyed[n][key]["validation_prediction"] for n in top)
        rows.append({**base, "model_name": "hybrid_ensemble",
                     "validation_prediction": float(blended)})
    m = _metrics(np.array([r["actual"] for r in rows]),
                 np.array([r["validation_prediction"] for r in rows]))
    m["n_folds"] = len({r["fold"] for r in rows})
    details = {"component_models": top, "weights": weights,
               "weighting_metric": "wMAPE",
               "reason": "Weighted by historical backtesting performance"}
    return {"rows": rows, "metrics": m, "status": "ok", "reason": "",
            "train_seconds": 0.0, "hybrid_details": details}


def _select_best(results: Dict[str, Dict[str, Any]]) -> Optional[str]:
    """wMAPE → |bias| → MAE. Returns None when no model validated (note [5])."""
    ok = [(n, r["metrics"]) for n, r in results.items() if r["status"] == "ok"
          and np.isfinite(r["metrics"].get("wmape", np.inf))]
    if not ok:
        return None
    return sorted(ok, key=lambda x: (x[1]["wmape"],
                                     abs(x[1].get("forecast_bias", np.inf)),
                                     x[1].get("mae", np.inf)))[0][0]


# ─── External feature availability handling (note [7]) ───────────────────────

def _partition_features(fe_output: Dict[str, Any], tree_cols: List[str],
                        warnings_out: List[Dict[str, Any]]
                        ) -> Tuple[List[str], List[str]]:
    """
    Applies future-availability metadata: scenario_only & historical_context
    columns are EXCLUDED from training; requires_future_values are kept and
    listed for climatology fill. Emits structured warnings for each adjustment.
    Returns (allowed_feature_cols, future_fill_cols).
    """
    avail = {m["feature_name"]: m.get("future_availability", "safe_for_future")
             for m in fe_output.get("feature_metadata", [])}
    allowed, future_fill = [], []
    for col in tree_cols:
        a = avail.get(col, "safe_for_future")
        if a in ("scenario_only", "historical_context"):
            warnings_out.append({
                "level": "info", "affected_scope": "external_features",
                "affected_item": col,
                "message": (f"الميزة «{col}» ({a}) استُبعدت من توقّع الإنتاج — "
                            "لا قيم مستقبلية لها (تعود عبر محرّك السيناريوهات لاحقاً).")})
            continue
        if a == "requires_future_values":
            future_fill.append(col)
            warnings_out.append({
                "level": "info", "affected_scope": "external_features",
                "affected_item": col,
                "message": (f"الميزة «{col}» تتطلب قيماً مستقبلية — تُملأ "
                            "بمناخية الشهر التاريخية (climatology) لا بالصفر.")})
        allowed.append(col)
    return allowed, future_fill


# ─── Output assembly helpers ─────────────────────────────────────────────────

def _bounds(preds: np.ndarray, rmse_val: Optional[float]
            ) -> Tuple[np.ndarray, np.ndarray]:
    """Residual-based ±1.28·RMSE interval (note [9]); NaN when no validation."""
    if rmse_val is None or not np.isfinite(rmse_val):
        return np.full(len(preds), np.nan), np.full(len(preds), np.nan)
    return np.clip(preds - _CI_Z * rmse_val, 0, None), preds + _CI_Z * rmse_val


def _metric_cards(portfolio: Dict[str, float], mean_demand: float) -> List[Dict[str, Any]]:
    """Business-friendly metric cards with plain-Arabic explanations (UI data)
    and documented, non-universal labels (note [8])."""
    wmape, bias = portfolio.get("wmape", np.inf), portfolio.get("forecast_bias", np.inf)
    mae, rmse, mape = (portfolio.get("mae", np.inf), portfolio.get("rmse", np.inf),
                       portfolio.get("mape", np.inf))

    def fmt_pct(v):
        return f"{v:+.0%}" if np.isfinite(v) else "—"

    return [
        {"metric": "wMAPE", "value": round(wmape, 4) if np.isfinite(wmape) else None,
         "display_value": f"{wmape:.0%}" if np.isfinite(wmape) else "—",
         "label": _label_pct(wmape),
         "plain_english": f"خطأ التوقّع الموزون الإجمالي {wmape:.0%}." if np.isfinite(wmape) else "تعذّر الحساب.",
         "business_meaning": "المقياس الأنفع لمحفظة منتجات: المنتجات عالية الحجم تأخذ وزناً أكبر."},
        {"metric": "Forecast Bias", "value": round(bias, 4) if np.isfinite(bias) else None,
         "display_value": fmt_pct(bias), "label": _label_bias(bias),
         "plain_english": ("الموديل يميل لرفع التوقّع فوق الواقع." if bias > 0.05
                           else "الموديل يميل لخفض التوقّع تحت الواقع." if bias < -0.05
                           else "التوقّع متوازن لا انحياز يُذكر."),
         "business_meaning": ("الرفع الزائد يضخّم المخزون والشراء؛ "
                              "الخفض الزائد يرفع خطر النفاد وتأخّر الإنتاج.")},
        {"metric": "MAPE", "value": round(mape, 4) if np.isfinite(mape) else None,
         "display_value": f"{mape:.0%}" if np.isfinite(mape) else "—",
         "label": _label_pct(mape),
         "plain_english": f"متوسط نسبة الخطأ {mape:.0%} (يتجاهل فترات الطلب الصفري)." if np.isfinite(mape) else "تعذّر الحساب.",
         "business_meaning": "سهل الفهم لكنه غير مستقر مع الطلب الصفري/الصغير — اعتمد wMAPE للمحفظة."},
        {"metric": "MAE", "value": round(mae, 2) if np.isfinite(mae) else None,
         "display_value": f"{mae:,.1f}" if np.isfinite(mae) else "—",
         "label": _label_relative(mae, mean_demand),
         "plain_english": f"متوسط الخطأ {mae:,.1f} وحدة طلب لكل فترة." if np.isfinite(mae) else "تعذّر الحساب.",
         "business_meaning": "خطأ بوحدات الطلب نفسها — مباشر للتخطيط التشغيلي."},
        {"metric": "RMSE", "value": round(rmse, 2) if np.isfinite(rmse) else None,
         "display_value": f"{rmse:,.1f}" if np.isfinite(rmse) else "—",
         "label": _label_relative(rmse, mean_demand),
         "plain_english": f"جذر متوسط مربع الخطأ {rmse:,.1f} وحدة." if np.isfinite(rmse) else "تعذّر الحساب.",
         "business_meaning": "يعاقب الأخطاء الكبيرة بشدة — مهم حين تكون الأخطاء الكبيرة أغلى من الصغيرة."},
    ]


# ─── Main entry point ────────────────────────────────────────────────────────

def run_forecast(
    df: Optional[pd.DataFrame] = None,
    intake_result: Optional[Dict[str, Any]] = None,
    granularity: str = "monthly",
    core_mapping: Optional[Dict[str, Optional[str]]] = None,
    business_inputs: Optional[List[Dict[str, Any]]] = None,
    sector: Optional[str] = None,
    fe_output: Optional[Dict[str, Any]] = None,
    external_features: Optional[pd.DataFrame] = None,
    forecast_horizon: Optional[int] = None,
    validation_horizon: Optional[int] = None,
    horizon: Optional[int] = None,            # legacy alias for forecast_horizon
    n_eval_splits: Optional[int] = None,
    evaluation_mode: str = "balanced",
) -> Dict[str, Any]:
    """
    The Forecasting Engine entry point.

    Consumes feature_engineering output (pass fe_output from the pipeline; when
    absent it is built ONCE here — never per fold). Validates candidates per
    product via walk-forward, selects by wMAPE→|bias|→MAE, builds the final
    multi-period forecast (default 12 monthly / 26 weekly), totals, chart-ready
    data, leaderboard, and business metric cards.

    Returns the full output contract (see module DEV NOTES) plus backward-
    compatibility aliases (forecasts/comparison_table/summary_table/history/
    best_model).
    """
    # ── Configuration resolution ──
    if granularity not in fe._GRAN_CONFIG:
        raise ValueError(f"granularity must be one of {list(fe._GRAN_CONFIG)}")
    if evaluation_mode not in _MODE_FOLDS:
        raise ValueError(f"evaluation_mode must be one of {list(_MODE_FOLDS)}")
    fh = int(forecast_horizon or horizon or _DEFAULT_FORECAST_H[granularity])
    vh = int(validation_horizon or _DEFAULT_VALIDATION_H)
    n_splits = int(n_eval_splits or _MODE_FOLDS[evaluation_mode])

    if core_mapping is None and intake_result:
        core_mapping = intake_result.get("proposed_mapping")
    if business_inputs is None and intake_result:
        business_inputs = intake_result.get("business_inputs")
    if not (core_mapping and all(core_mapping.get(r) for r in ("date", "product", "qty"))):
        raise ValueError("core_mapping with date/product/qty is required")
    product_col = core_mapping["product"]
    season = _season_of(granularity)
    freq = _freq_of(granularity)

    warnings_out: List[Dict[str, Any]] = []
    failed_models: List[Dict[str, Any]] = []

    # ── Features: consume fe output; build ONCE only when not provided (note [1]) ──
    if fe_output is None:
        if df is None:
            raise ValueError("either fe_output or df must be provided")
        fe_output = fe.build_features(df, intake_result=None, granularity=granularity,
                                      sector=sector, core_mapping=core_mapping,
                                      business_inputs=business_inputs,
                                      external_features=external_features)

    base = fe_output["feature_matrices"]["baseline"]
    if base is None or len(base) == 0:
        raise ValueError("feature_engineering produced an empty panel")

    # Per-product history series from the fe panel (single source of truth).
    series_map: Dict[str, pd.Series] = {}
    for product, g in base.groupby(product_col):
        s = pd.Series(pd.to_numeric(g["y"], errors="coerce").fillna(0).values,
                      index=pd.to_datetime(g["date"])).sort_index()
        series_map[str(product)] = s
    total_series = (pd.concat(series_map.values(), axis=1).fillna(0).sum(axis=1)
                    .sort_index()) if series_map else pd.Series(dtype=float)

    # Tree matrix views + future-availability partition (note [7]).
    tree_all = fe_output["feature_matrices"]["xgboost"]
    tree_cols = [c for c in tree_all.columns if c not in (product_col, "y")]
    allowed_cols, future_fill_cols = _partition_features(fe_output, tree_cols, warnings_out)

    # ── Segmentation + per-product validation ──
    seg_df, segments = segment_products(series_map, season)
    selected: Dict[str, str] = {}
    selected_val_rows: List[Dict[str, Any]] = []
    leaderboard_rows: List[Dict[str, Any]] = []
    hybrid_per_product: Dict[str, Dict[str, Any]] = {}
    pooled: Dict[str, Dict[str, float]] = {}          # ALL-scope running aggregates
    selected_rmse: Dict[str, float] = {}

    def _pool(name: str, rows: List[Dict[str, Any]]) -> None:
        agg = pooled.setdefault(name, {"sae": 0.0, "se": 0.0, "sa": 0.0,
                                       "sse": 0.0, "n": 0, "mape_s": 0.0, "mape_n": 0})
        for r in rows:
            e = r["validation_prediction"] - r["actual"]
            agg["sae"] += abs(e); agg["se"] += e; agg["sa"] += abs(r["actual"])
            agg["sse"] += e * e; agg["n"] += 1
            if r["actual"] != 0:
                agg["mape_s"] += abs(e) / abs(r["actual"]); agg["mape_n"] += 1

    for product, y in series_map.items():
        seg = segments[product]
        candidates = _candidates_for(seg, evaluation_mode)
        tree_ctx = None
        if any(_MODEL_SPECS[c]["kind"] == "ml" for c in candidates):
            pf = tree_all[tree_all[product_col].astype(str) == product]
            tree_ctx = {"frame": pf, "feature_cols": allowed_cols,
                        "future_fill_cols": future_fill_cols, "sector": sector}

        results = _validate_product(product, y, candidates, granularity, vh,
                                    n_splits, tree_ctx, failed_models)
        hybrid = _add_hybrid(results)
        if hybrid is not None:
            results["hybrid_ensemble"] = hybrid
            hybrid_per_product[product] = hybrid["hybrid_details"]

        best = _select_best(results)
        if best is None:
            best = "naive"   # safe baseline always exists (note [4])
            warnings_out.append({"level": "warning", "affected_scope": "product",
                                 "affected_item": product,
                                 "message": f"المنتج «{product}»: تعذّر التحقّق "
                                            "(تاريخ غير كافٍ) — توقّع أساس naive."})
        selected[product] = best
        if best in results and results[best]["status"] == "ok":
            selected_val_rows.extend(results[best]["rows"])
            selected_rmse[product] = results[best]["metrics"]["rmse"]

        for name, r in results.items():
            m = r["metrics"]
            leaderboard_rows.append({
                "model_name": name, "scope": "product", "product": product,
                "mae": m.get("mae", np.nan), "rmse": m.get("rmse", np.nan),
                "mape": m.get("mape", np.nan), "wmape": m.get("wmape", np.nan),
                "forecast_bias": m.get("forecast_bias", np.nan),
                "status": r["status"], "reason_if_failed": r["reason"],
                "training_time_seconds": r["train_seconds"],
                "n_folds": m.get("n_folds", 0), "selected": name == best})
            if r["status"] == "ok":
                _pool(name, r["rows"])

    # ── ALL-scope leaderboard rows from pooled aggregates ──
    best_overall, best_overall_wmape = None, np.inf
    for name, a in pooled.items():
        if a["n"] == 0 or a["sa"] == 0:
            continue
        wmape = a["sae"] / a["sa"]
        leaderboard_rows.append({
            "model_name": name, "scope": "ALL", "product": "ALL",
            "mae": a["sae"] / a["n"], "rmse": np.sqrt(a["sse"] / a["n"]),
            "mape": (a["mape_s"] / a["mape_n"]) if a["mape_n"] else np.inf,
            "wmape": wmape, "forecast_bias": a["se"] / a["sa"],
            "status": "ok", "reason_if_failed": "",
            "training_time_seconds": np.nan, "n_folds": np.nan, "selected": False})
        if wmape < best_overall_wmape:
            best_overall, best_overall_wmape = name, wmape
    for row in leaderboard_rows:
        if row["scope"] == "ALL" and row["model_name"] == best_overall:
            row["selected"] = True
    # LSTM exposed as metadata only (note [11]).
    leaderboard_rows.append({"model_name": "lstm", "scope": "ALL", "product": "ALL",
                             "mae": np.nan, "rmse": np.nan, "mape": np.nan,
                             "wmape": np.nan, "forecast_bias": np.nan,
                             "status": "future_extension",
                             "reason_if_failed": "Requires larger datasets, tuning, "
                                                 "and additional runtime validation.",
                             "training_time_seconds": np.nan, "n_folds": np.nan,
                             "selected": False})
    leaderboard = pd.DataFrame(leaderboard_rows)

    # ── Final multi-period forecast per product (note [2]) ──
    offset = pd.tseries.frequencies.to_offset(freq)
    fc_rows: List[Dict[str, Any]] = []
    forecasts_alias: Dict[str, pd.Series] = {}
    for product, y in series_map.items():
        name = selected[product]
        tree_ctx = None
        if _MODEL_SPECS.get(name, {}).get("kind") == "ml" or name == "hybrid_ensemble":
            pf = tree_all[tree_all[product_col].astype(str) == product]
            tree_ctx = {"frame": pf, "feature_cols": allowed_cols,
                        "future_fill_cols": future_fill_cols, "sector": sector}

        def _final(nm: str) -> np.ndarray:
            return np.clip(_safe_predict(nm, y, fh, granularity, tree_ctx), 0, None)

        try:
            if name == "hybrid_ensemble":
                det = hybrid_per_product[product]
                preds = np.zeros(fh)
                for comp, w in det["weights"].items():
                    preds = preds + w * _final(comp)
            else:
                preds = _final(name)
        except Exception as exc:
            warnings_out.append({"level": "warning", "affected_scope": "product",
                                 "affected_item": product,
                                 "message": f"فشل التوقّع النهائي بـ {name} ({exc}) "
                                            "— استُخدم naive."})
            name = "naive"
            preds = _final("naive")
            selected[product] = name

        lo, hi = _bounds(preds, selected_rmse.get(product))
        start = y.index[-1] + offset
        dates = pd.date_range(start, periods=fh, freq=freq)
        forecasts_alias[product] = pd.Series(preds, index=dates, name=product)
        for step, (dt, p_, l_, h_) in enumerate(zip(dates, preds, lo, hi), 1):
            fc_rows.append({"product": product, "date": dt, "forecast": float(p_),
                            "lower_bound": float(l_) if np.isfinite(l_) else np.nan,
                            "upper_bound": float(h_) if np.isfinite(h_) else np.nan,
                            "model_used": name, "horizon_step": step,
                            "forecast_start_date": start})
    product_forecasts = pd.DataFrame(fc_rows)

    # ── Totals: bottom-up always + direct on the aggregate series (note [10]) ──
    total_forecast = (product_forecasts.groupby("date")
                      .agg(total_forecast=("forecast", "sum"),
                           total_lower_bound=("lower_bound", "sum"),
                           total_upper_bound=("upper_bound", "sum"))
                      .reset_index())
    total_forecast["horizon_step"] = range(1, len(total_forecast) + 1)

    direct_total = pd.DataFrame()
    if len(total_series) >= season + 2 + vh:
        d_candidates = list(_FAST) + (["sarima"] if _HAS["statsmodels"]
                                      and evaluation_mode != "fast" else [])
        d_failed: List[Dict[str, Any]] = []
        d_res = _validate_product("__TOTAL__", total_series, d_candidates,
                                  granularity, vh, n_splits, None, d_failed)
        d_best = _select_best(d_res) or "naive"
        try:
            d_preds = np.clip(_safe_predict(d_best, total_series, fh, granularity), 0, None)
            d_rmse = d_res.get(d_best, {}).get("metrics", {}).get("rmse")
            d_lo, d_hi = _bounds(d_preds, d_rmse)
            d_dates = pd.date_range(total_series.index[-1] + offset, periods=fh, freq=freq)
            direct_total = pd.DataFrame({"date": d_dates, "direct_total_forecast": d_preds,
                                         "lower_bound": d_lo, "upper_bound": d_hi,
                                         "model_used": d_best,
                                         "horizon_step": range(1, fh + 1)})
        except Exception:
            pass  # direct total is a bonus — bottom-up remains the contract

    # ── chart_data: history / validation / forecast, product + ALL ──
    chart_rows: List[Dict[str, Any]] = []
    for product, y in series_map.items():
        for dt, a in y.items():
            chart_rows.append({"product": product, "date": dt, "actual": float(a),
                               "validation_prediction": np.nan, "forecast": np.nan,
                               "lower_bound": np.nan, "upper_bound": np.nan,
                               "series_type": "history",
                               "model_used": selected[product]})
    for r in selected_val_rows:
        chart_rows.append({"product": r["product"], "date": r["date"],
                           "actual": r["actual"],
                           "validation_prediction": r["validation_prediction"],
                           "forecast": np.nan, "lower_bound": np.nan,
                           "upper_bound": np.nan, "series_type": "validation",
                           "model_used": r["model_name"]})
    for _, r in product_forecasts.iterrows():
        chart_rows.append({"product": r["product"], "date": r["date"],
                           "actual": np.nan, "validation_prediction": np.nan,
                           "forecast": r["forecast"], "lower_bound": r["lower_bound"],
                           "upper_bound": r["upper_bound"], "series_type": "forecast",
                           "model_used": r["model_used"]})
    for dt, a in total_series.items():
        chart_rows.append({"product": "ALL", "date": dt, "actual": float(a),
                           "validation_prediction": np.nan, "forecast": np.nan,
                           "lower_bound": np.nan, "upper_bound": np.nan,
                           "series_type": "history", "model_used": "bottom_up"})
    for _, r in total_forecast.iterrows():
        chart_rows.append({"product": "ALL", "date": r["date"], "actual": np.nan,
                           "validation_prediction": np.nan,
                           "forecast": r["total_forecast"],
                           "lower_bound": r["total_lower_bound"],
                           "upper_bound": r["total_upper_bound"],
                           "series_type": "forecast", "model_used": "bottom_up"})
    chart_data = pd.DataFrame(chart_rows)

    # ── Portfolio metrics from the SELECTED models' validation rows ──
    if selected_val_rows:
        portfolio = _metrics(np.array([r["actual"] for r in selected_val_rows]),
                             np.array([r["validation_prediction"]
                                       for r in selected_val_rows]))
    else:
        portfolio = {k: np.inf for k in ("mae", "rmse", "mape", "wmape", "forecast_bias")}
        warnings_out.append({"level": "warning", "affected_scope": "portfolio",
                             "affected_item": "ALL",
                             "message": "لا تحقّق متاحاً (تاريخ غير كافٍ) — "
                                        "مقاييس المحفظة غير محسوبة."})
    mean_demand = float(np.mean([s.mean() for s in series_map.values()])) \
        if series_map else 0.0
    bias = portfolio["forecast_bias"]
    bias_dir = ("over-forecasting" if np.isfinite(bias) and bias > 0.05
                else "under-forecasting" if np.isfinite(bias) and bias < -0.05
                else "balanced")

    forecast_start = (min(s.index[-1] for s in series_map.values()) + offset
                      if series_map else None)
    forecast_summary = {
        "forecast_horizon": fh, "validation_horizon": vh,
        "granularity": granularity, "forecast_start_date": forecast_start,
        "total_expected_demand": float(total_forecast["total_forecast"].sum())
            if len(total_forecast) else 0.0,
        "peak_forecast_period": (total_forecast.loc[
            total_forecast["total_forecast"].idxmax(), "date"]
            if len(total_forecast) else None),
        "best_overall_model": best_overall,
        "portfolio_wmape": round(portfolio["wmape"], 4)
            if np.isfinite(portfolio["wmape"]) else None,
        "forecast_bias_direction": bias_dir,
        "confidence_label": _label_pct(portfolio["wmape"]),
    }

    evaluation_summary = {
        "mode": evaluation_mode,
        "n_products_evaluated": int(sum(1 for p in series_map
                                        if any(r["product"] == p
                                               for r in selected_val_rows))),
        "n_products_forecasted": len(series_map),
        "n_folds": n_splits, "validation_horizon": vh, "forecast_horizon": fh,
        "primary_metric": "wMAPE",
        "notes": ["selection: wMAPE → |bias| → MAE",
                  "fold features sliced from a single build (static-feature "
                  "compromise documented in dev note [3])",
                  "bounds are residual-based ≈80% intervals; summed total "
                  "bounds overstate uncertainty (note [9])"],
    }

    hybrid_details = ({"weighting_metric": "wMAPE",
                       "reason": "Weighted by historical backtesting performance",
                       "per_product": hybrid_per_product}
                      if hybrid_per_product else
                      {"status": "not_applicable",
                       "reason": "fewer than 2 valid component models per product"})

    validation_predictions = pd.DataFrame(selected_val_rows)

    # Backward-compat aliases (note [12]).
    prod_rows = leaderboard[leaderboard["scope"] == "product"]
    comparison_table = prod_rows.rename(columns={"model_name": "model"})[
        ["product", "model", "mape", "rmse", "mae", "wmape", "n_folds"]].copy()
    summary_table = leaderboard[leaderboard["scope"] == "ALL"].copy()

    return {
        # New contract
        "forecast_summary": forecast_summary,
        "product_forecasts": product_forecasts,
        "total_forecast": total_forecast,
        "direct_total_forecast": direct_total,
        "chart_data": chart_data,
        "validation_predictions": validation_predictions,
        "model_leaderboard": leaderboard,
        "metric_cards": _metric_cards(portfolio, mean_demand),
        "product_segments": seg_df,
        "best_model_by_product": dict(selected),
        "best_model_overall": best_overall,
        "hybrid_details": hybrid_details,
        "evaluation_summary": evaluation_summary,
        "warnings": warnings_out,
        "failed_models": failed_models,
        "history": series_map,
        "history_total": total_series,
        "feature_engineering": {"granularity": granularity,
                                "leakage_audit": fe_output.get("leakage_audit", {}),
                                "excluded_columns": fe_output.get("excluded_columns", [])},
        # Legacy aliases
        "forecasts": forecasts_alias,
        "comparison_table": comparison_table,
        "summary_table": summary_table,
        "best_model": dict(selected),
    }
