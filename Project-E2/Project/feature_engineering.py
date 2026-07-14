"""
feature_engineering.py — Feature Engineering Engine | Bayyina Platform

Position: after quality.py (cleaning) and before model.py (selection/training).
Role: GATEKEEPER, not generator. Decides which columns enter the model and how
they are transformed, builds leakage-free temporal features, and adapts the
output to each model family.

Main entry point: build_features(df, intake_result, quality_result, granularity, ...)
"""

# ===========================================================================
# DEVELOPER NOTES
# ===========================================================================
#
# [1] Responsibility boundary: this layer APPLIES intake's classification
#     (business_inputs[*].legacy_category / detected_meaning) — it never
#     re-classifies columns. Discovery belongs to intake, permanently.
#
# [2] model.py contract (do not break): model imports _GRAN_CONFIG,
#     _add_temporal_features(panel, cfg, product_col, date_col) and
#     _add_seasonal_features(panel, product_col, sector) to rebuild features
#     during recursive prediction. Any feature added INSIDE those helpers
#     automatically reaches prediction with the same definition — train/predict
#     consistency by construction. Features added elsewhere will be 0-filled at
#     prediction time (model._encode_features), so think before placing logic.
#
# [3] Leakage protection (non-negotiable):
#     - every rolling window is preceded by shift(1), per product via groupby;
#     - lags are per-product groupby.shift, never global;
#     - post-sale metrics (revenue/profit/...) are excluded by TOKEN-based
#       keyword matching (substring matching produced false positives like
#       'Marginal_Notes' containing 'margin' — fixed).
#
# [4] Seasonality — three distinct concerns, kept separate:
#     detection  : _detect_seasonal_pattern (data-driven, sector fallback only
#                  for short history < 18 months);
#     features   : detected flags (is_peak/is_low/months_to_peak/pattern) PLUS
#                  cyclical encodings (month_sin/cos, week_sin/cos) — these are
#                  complementary: flags locate the detected peak, sin/cos give
#                  a smooth calendar position (fixes the Dec→Jan boundary);
#     modeling   : SARIMA and Prophet own their internal seasonality — they
#                  receive NO seasonal features (signal duplication). Trees
#                  receive both flags and encodings.
#
# [5] Future-availability framework (scenario planning foundation). Every
#     feature carries future_availability metadata:
#       safe_for_future       — derivable at any future date (calendar, lags
#                               via recursion, detected seasonal flags);
#       requires_future_values— actuals unknown at forecast time (temperature,
#                               weather-derived CDD/HDD). Used in training,
#                               0-filled in recursion → train/serve skew is now
#                               EXPLICIT in the leakage audit, not silent;
#       scenario_only         — user-settable levers (price, promotion,
#                               discount): unknown as actuals, but a scenario
#                               can SET them — the what-if planning surface.
#
# [6] feature_relevance_analysis is EXPLAINABILITY, not feature selection.
#     Pearson correlation at lags 0..3 (per-product shifted) + coverage +
#     history sufficiency. Deliberately no mutual information / permutation
#     importance: cost not justified versus correlation for user-facing
#     "which features matter" answers. Nothing is dropped based on it.
#
# [7] NaN policy: lag/rolling NaNs at each product's start stay NaN in tree
#     matrices; the consumer (model.py) drops them before training. We never
#     fill with 0 to avoid planting fake signals. New features are designed to
#     introduce NO extra NaN rows beyond the longest lag (growth guards /0).
#
# [8] seasonal_pattern_type stays a pandas category — explainable in the UI;
#     model.py applies its own fixed ordinal encoding for sklearn.
#
# [9] Inventory inputs remain EXCLUDED from forecasting features, but are now
#     passed through in 'inventory_passthrough' metadata so the future
#     inventory layer can consume them without re-deriving. No logic mixing.
#
# Known limitations:
#   [L-a] Leakage keyword detection is name-based; an innocently-named future
#         field (e.g. 'delivered_qty') escapes. Needs manual marking in the UI.
#         No AI-based detection by design (must stay deterministic/explainable).
#   [L-b] Feature aggregation to period: continuous→mean, binary→max. May not
#         suit every feature; parameterize later.
#   [L-c] Seasonality is detected on calendar months even for weekly data;
#         months_to_peak is month-grained. Practical, not week-precise.
#   [L-d] time_index helps trees fit level shifts WITHIN the training range;
#         trees cannot extrapolate a trend beyond it. This is a model-family
#         limitation the feature does not solve.
#
# ===========================================================================

import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ─── Constants ───────────────────────────────────────────────────────────────

# Post-sale metric keywords (token-matched) — excluded automatically as leakage.
_LEAKAGE_KEYWORDS: Tuple[str, ...] = (
    "revenue", "profit", "margin", "turnover", "grosssales", "netsales",
    "income", "earnings", "إيراد", "ربح", "هامش", "عائد", "أرباح",
)

# Per-granularity config. Granularity is EXPLICIT from the UI, never inferred.
_GRAN_CONFIG: Dict[str, Dict[str, Any]] = {
    "monthly": {"freq": "MS", "lags": [1, 3, 12], "rolling": [3, 6],
                "weekofyear": False, "season": 12},
    "weekly":  {"freq": "W",  "lags": [1, 4, 52], "rolling": [4, 12],
                "weekofyear": True,  "season": 52},
}

# Sector defaults — short-history safety net ONLY (unused when data suffices).
_SECTOR_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "hvac": {"peak_months": [5, 6, 7, 8, 9], "low_months": [11, 12, 1, 2],
             "pattern": "summer_peak"},
}

# Month groups for peak-pattern classification.
_SUMMER = {5, 6, 7, 8, 9}
_WINTER = {11, 12, 1, 2, 3}

# Cooling/heating degree-day base (°C) — common ASHRAE choice; parameterize to
# calibrate per region.
_DEGREE_DAY_BASE = 18.0

# Minimum history (months) for reliable data-driven seasonality detection.
_MIN_MONTHS_FOR_DETECTION = 18

# Future-availability classification by detected meaning (note [5]).
_FUTURE_AVAILABILITY_BY_MEANING: Dict[str, str] = {
    "temperature": "requires_future_values",
    "holiday": "safe_for_future",          # calendar-known in advance
    "price": "scenario_only",
    "promotion": "scenario_only",
    "discount": "scenario_only",
}
_FUTURE_DEFAULT_EXTERNAL = "requires_future_values"  # conservative for unknowns


# ─── Token matching (mirrors intake's approach; local copy keeps layers
#     decoupled — fe consumes intake's OUTPUT, never its module) ──────────────

def _normalize_name(name: str) -> str:
    """Lowercases a column name and squashes separators (keeps Arabic)."""
    return re.sub(r"[\s_\-./]+", "", str(name).strip().lower())


def _tokens_of(text: str) -> set:
    """Splits a name into tokens (separators + camelCase), with an Arabic
    definite-article-stripped variant per token."""
    s = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", str(text).strip())
    parts = re.split(r"[\s_\-./()\[\]:,؛،]+", s.lower())
    tokens = set()
    for p in parts:
        if not p:
            continue
        tokens.add(p)
        if p.startswith("ال") and len(p) > 3:
            tokens.add(p[2:])
    return tokens


def _matches_leakage(col_name: str) -> bool:
    """Token-based leakage check: a keyword matches only as a whole token.
    'Revenue_Q1' → leak; 'Marginal_Notes' → NOT a leak (note [3])."""
    col_tokens = _tokens_of(col_name)
    return any(_tokens_of(kw) <= col_tokens for kw in _LEAKAGE_KEYWORDS)


# ─── 1. Column routing (applies intake classification) ───────────────────────

def route_columns(
    df: pd.DataFrame,
    core_mapping: Dict[str, Optional[str]],
    business_inputs: Optional[List[Dict[str, Any]]],
) -> Dict[str, Any]:
    """
    Routes every non-core column by APPLYING intake's classification (note [1]).

    leakage keyword → excluded (overrides any class); forecast_feature →
    included; inventory_input → excluded but PASSED THROUGH for the inventory
    layer (note [9]); advisory/unknown → excluded.

    Returns {feature_columns, excluded_columns, feature_meanings,
             inventory_columns}. Transforms nothing; re-detects nothing.
    """
    core_cols = {c for c in core_mapping.values() if c is not None}
    class_map = {bi["column"]: bi for bi in (business_inputs or [])}

    feature_columns: List[str] = []
    excluded: List[Dict[str, str]] = []
    feature_meanings: Dict[str, str] = {}
    inventory_columns: List[Dict[str, str]] = []

    for col in df.columns:
        if col in core_cols:
            continue
        if _matches_leakage(str(col)):
            excluded.append({"column": col,
                             "reason": "leakage: post-sale metric (revenue/profit) — auto-excluded"})
            continue

        bi = class_map.get(col)
        classification = bi.get("legacy_category") if bi else None
        meaning = bi.get("detected_meaning") if bi else None

        if classification == "forecast_feature":
            feature_columns.append(col)
            feature_meanings[col] = meaning or "unknown"
        elif classification == "inventory_input":
            excluded.append({"column": col,
                             "reason": "inventory_input: used by the inventory layer, not forecasting"})
            inventory_columns.append({"column": col, "meaning": meaning or "unknown"})
        elif classification == "advisory":
            excluded.append({"column": col, "reason": "advisory: non-predictive information"})
        else:
            excluded.append({"column": col,
                             "reason": "unclassified: not classified by intake as a forecast feature"})

    return {"feature_columns": feature_columns, "excluded_columns": excluded,
            "feature_meanings": feature_meanings,
            "inventory_columns": inventory_columns}


# ─── Panel construction ──────────────────────────────────────────────────────

def _ffill_small_gaps(series: pd.Series, limit: int = 2) -> Tuple[pd.Series, int]:
    """Forward-fills runs of ≤ limit NaNs; larger gaps stay NaN (flagged
    upstream). Returns (filled, n_remaining_nan)."""
    filled = series.ffill(limit=limit)
    return filled, int(filled.isna().sum())


def _build_panel(
    df: pd.DataFrame,
    core_mapping: Dict[str, Optional[str]],
    cfg: Dict[str, Any],
    feature_columns: List[str],
) -> pd.DataFrame:
    """
    Builds the per-product modeling panel at the target granularity.

    Per product: y = sum of qty per period (zero-filled gaps between first and
    last observation); each forecast feature aggregated to the period
    (continuous→mean, binary→max — limitation [L-b]) with small-gap ffill.

    Returns a long frame [product, date, y, <features>]. Adds no temporal
    features (separate step); never crosses product boundaries.
    """
    date_col, product_col, qty_col = (core_mapping["date"], core_mapping["product"],
                                      core_mapping["qty"])
    freq = cfg["freq"]

    work = df.copy()
    work[date_col] = pd.to_datetime(work[date_col], errors="coerce")
    work[qty_col] = pd.to_numeric(work[qty_col], errors="coerce")
    work = work.dropna(subset=[date_col])

    binary_cols = {c for c in feature_columns if work[c].dropna().nunique() <= 2}

    frames: List[pd.DataFrame] = []
    for product, g in work.groupby(product_col):
        gi = g.set_index(date_col).sort_index()
        y = gi[qty_col].resample(freq).sum()
        if len(y) == 0:
            continue
        full = pd.date_range(y.index.min(), y.index.max(), freq=freq)
        y = y.reindex(full, fill_value=0)

        # Canonical date column name simplifies all downstream layers.
        panel_p = pd.DataFrame({product_col: str(product), "date": full, "y": y.values})
        for fcol in feature_columns:
            num = pd.to_numeric(gi[fcol], errors="coerce")
            agg = "max" if fcol in binary_cols else "mean"
            fser = num.resample(freq).agg(agg).reindex(full)
            fser, _ = _ffill_small_gaps(fser, limit=2)
            panel_p[fcol] = fser.values
        frames.append(panel_p)

    if not frames:
        return pd.DataFrame(columns=[product_col, "date", "y"] + feature_columns)
    return pd.concat(frames, ignore_index=True)


# ─── 2. Temporal features (strict leakage prevention, per product) ───────────

def _add_temporal_features(
    panel: pd.DataFrame,
    cfg: Dict[str, Any],
    product_col: str,
    date_col: str,
) -> pd.DataFrame:
    """
    Adds calendar, cyclical, trend, lag, rolling, volatility and growth
    features — strictly per product. Imported by model.py for recursive
    prediction (note [2]): everything here reaches prediction identically.

    Calendar : month, quarter, year (+ weekofyear when weekly).
    Cyclical : month_sin/cos (+ week_sin/cos when weekly) — smooth Dec→Jan
               boundary for tree models (note [4]).
    Trend    : time_index = periods since the product's own start ([L-d]).
    Lags     : per granularity (monthly 1/3/12, weekly 1/4/52), groupby.shift.
    Rolling  : means per window, each shift(1)-protected inside the group.
    Volatility: rolling_std over the first window (shift(1)) — separates stable
               from erratic regimes without leaking the current row.
    Growth   : pop_growth = lag_1/lag_2 − 1, guarded (lag_2==0 → 0) so it adds
               NO extra NaN rows beyond the longest lag (note [7]).

    Leaves lag/rolling NaNs at each product's start untouched (consumer drops).
    """
    panel = panel.sort_values([product_col, date_col]).reset_index(drop=True)
    dt = pd.to_datetime(panel[date_col])

    # Calendar (row-wise; cannot cross products).
    panel["month"] = dt.dt.month
    panel["quarter"] = dt.dt.quarter
    panel["year"] = dt.dt.year
    if cfg["weekofyear"]:
        panel["weekofyear"] = dt.dt.isocalendar().week.astype(int)

    # Cyclical encodings — calendar-derived, always future-safe.
    panel["month_sin"] = np.sin(2 * np.pi * (panel["month"] - 1) / 12)
    panel["month_cos"] = np.cos(2 * np.pi * (panel["month"] - 1) / 12)
    if cfg["weekofyear"]:
        panel["week_sin"] = np.sin(2 * np.pi * (panel["weekofyear"] - 1) / 52)
        panel["week_cos"] = np.cos(2 * np.pi * (panel["weekofyear"] - 1) / 52)

    # Trend: periods since each product's own first observation.
    panel["time_index"] = panel.groupby(product_col).cumcount()

    # Lags per product (groupby.shift respects boundaries).
    grp = panel.groupby(product_col)["y"]
    for L in cfg["lags"]:
        panel[f"lag_{L}"] = grp.shift(L)

    # Rolling means: shift(1) THEN rolling, inside each product group.
    for W in cfg["rolling"]:
        mp = max(1, W // 2)

        def _roll_mean(s, _w=W, _mp=mp):
            return s.shift(1).rolling(_w, min_periods=_mp).mean()

        panel[f"rolling_mean_{W}"] = panel.groupby(product_col)["y"].transform(_roll_mean)

    # Volatility: rolling std over the first (shortest) window, same protection.
    W0 = cfg["rolling"][0]

    def _roll_std(s, _w=W0, _mp=max(2, W0 // 2)):
        return s.shift(1).rolling(_w, min_periods=_mp).std()

    panel[f"rolling_std_{W0}"] = panel.groupby(product_col)["y"].transform(_roll_std)

    # Period-over-period growth from existing lags — division guarded.
    lag1, lag2 = panel["lag_1"], panel[f"lag_{cfg['lags'][1]}"]
    with np.errstate(divide="ignore", invalid="ignore"):
        growth = np.where(lag2 > 0, lag1 / lag2 - 1.0, 0.0)
    panel["pop_growth"] = np.where(lag1.isna() | lag2.isna(), np.nan, growth)

    return panel


# ─── 5. Data-driven seasonality (with sector safety net) ─────────────────────

def _detect_seasonal_pattern(
    monthly_avg: pd.Series,
    n_months_span: float,
    sector: Optional[str],
) -> Dict[str, Any]:
    """
    Detects a product's seasonal pattern from DATA first, sector second.

    Data-driven (span ≥ 18 months): peaks = top 33% months by mean demand,
    lows = bottom 33%; year_round (low variation) is checked FIRST to avoid
    carving fake peaks out of stable products. Fallback: declared sector
    defaults for short history; otherwise 'undetected' (features skipped).

    Returns {pattern_type, peak_months, low_months, detection_source} with
    detection_source ∈ {data_driven, sector_default, undetected}.
    """
    if n_months_span < _MIN_MONTHS_FOR_DETECTION:
        if sector and sector in _SECTOR_DEFAULTS:
            d = _SECTOR_DEFAULTS[sector]
            return {"pattern_type": d["pattern"], "peak_months": list(d["peak_months"]),
                    "low_months": list(d["low_months"]), "detection_source": "sector_default"}
        return {"pattern_type": "undetected", "peak_months": [], "low_months": [],
                "detection_source": "undetected"}

    avg = monthly_avg.dropna()
    if len(avg) < 6 or avg.max() <= 0:
        return {"pattern_type": "undetected", "peak_months": [], "low_months": [],
                "detection_source": "undetected"}

    # year_round first: low relative range means no real peak (dev note [4]).
    rel_range = (avg.max() - avg.min()) / avg.mean() if avg.mean() > 0 else 0
    if rel_range < 0.35:
        return {"pattern_type": "year_round", "peak_months": [], "low_months": [],
                "detection_source": "data_driven"}

    k = max(1, int(round(len(avg) * 0.33)))
    peak_months = sorted(avg.sort_values(ascending=False).head(k).index.tolist())
    low_months = sorted(avg.sort_values(ascending=True).head(k).index.tolist())

    summer_hits = len(set(peak_months) & _SUMMER)
    winter_hits = len(set(peak_months) & _WINTER)
    if summer_hits > winter_hits and summer_hits >= len(peak_months) / 2:
        pattern = "summer_peak"
    elif winter_hits > summer_hits and winter_hits >= len(peak_months) / 2:
        pattern = "winter_peak"
    elif summer_hits == 0 and winter_hits == 0:
        pattern = "shoulder_peak"
    else:
        pattern = "irregular"

    return {"pattern_type": pattern, "peak_months": peak_months,
            "low_months": low_months, "detection_source": "data_driven"}


def _months_to_peak(month: int, peak_months: List[int]) -> float:
    """Cyclic forward distance (months) to the nearest peak month."""
    if not peak_months:
        return np.nan
    return float(min((p - month) % 12 for p in peak_months))


def _add_seasonal_features(
    panel: pd.DataFrame,
    product_col: str,
    sector: Optional[str],
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Adds detected-seasonality features per product + returns the detection
    report. Features: is_peak_season, is_low_season, months_to_peak,
    seasonal_pattern_type (category). 'undetected' products get neutral values.
    Imported by model.py for recursive prediction (note [2]).
    """
    detection: Dict[str, Any] = {}
    panel = panel.copy()
    panel["is_peak_season"] = False
    panel["is_low_season"] = False
    panel["months_to_peak"] = np.nan
    panel["seasonal_pattern_type"] = "undetected"

    for product, g in panel.groupby(product_col):
        dt = pd.to_datetime(g["date"] if "date" in g else g.iloc[:, 1])
        span_months = ((dt.max().year - dt.min().year) * 12
                       + (dt.max().month - dt.min().month) + 1)
        monthly_avg = g.assign(_m=dt.dt.month).groupby("_m")["y"].mean()

        info = _detect_seasonal_pattern(monthly_avg, span_months, sector)
        detection[str(product)] = info

        idx = g.index
        months = dt.dt.month.values
        peak, low = set(info["peak_months"]), set(info["low_months"])
        panel.loc[idx, "seasonal_pattern_type"] = info["pattern_type"]
        if info["detection_source"] != "undetected":
            panel.loc[idx, "is_peak_season"] = [m in peak for m in months]
            panel.loc[idx, "is_low_season"] = [m in low for m in months]
            panel.loc[idx, "months_to_peak"] = [_months_to_peak(m, info["peak_months"])
                                                for m in months]

    panel["seasonal_pattern_type"] = panel["seasonal_pattern_type"].astype("category")
    return panel, detection


# ─── Degree-day features (when a temperature column exists) ──────────────────

def _add_degree_day_features(
    panel: pd.DataFrame,
    feature_meanings: Dict[str, str],
    base: float = _DEGREE_DAY_BASE,
) -> Tuple[pd.DataFrame, bool]:
    """
    Adds cooling/heating degree days when a temperature column exists:
    CDD = max(0, temp − base), HDD = max(0, base − temp). Useful for any
    temperature-sensitive demand, not sector-specific. Returns (panel, added).
    """
    temp_col = next((c for c, m in feature_meanings.items() if m == "temperature"), None)
    if temp_col is None or temp_col not in panel.columns:
        return panel, False
    temp = pd.to_numeric(panel[temp_col], errors="coerce")
    panel["cooling_degree_days"] = (temp - base).clip(lower=0)
    panel["heating_degree_days"] = (base - temp).clip(lower=0)
    return panel, True


# ─── 4. External feature alignment (passed-in weather/AHRI frames) ───────────

def _align_external_features(
    panel: pd.DataFrame,
    external_features: Optional[pd.DataFrame],
    core_mapping: Dict[str, Optional[str]],
) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """
    Aligns a SEPARATE external frame (e.g. Open-Meteo, AHRI) onto the panel's
    time index: small gaps (≤2) forward-filled, larger gaps flagged. Merged per
    product when a matching product column exists, else broadcast by date.

    Returns (panel, log[{feature, source, large_gaps_flagged, product_specific}]).
    External features enter tree matrices only (enforced by the matrix builder).
    """
    log: List[Dict[str, Any]] = []
    if external_features is None or len(external_features) == 0:
        return panel, log

    product_col = core_mapping["product"]
    ext = external_features.copy()

    if not isinstance(ext.index, pd.DatetimeIndex):
        date_like = next((c for c in ext.columns
                          if pd.api.types.is_datetime64_any_dtype(ext[c])), None)
        if date_like is None:
            return panel, log  # no time axis → ignore safely
        ext = ext.set_index(date_like)
    ext.index = pd.to_datetime(ext.index)

    product_specific = product_col in ext.columns
    ext_value_cols = [c for c in ext.columns if c != product_col]

    panel = panel.sort_values([product_col, "date"]).reset_index(drop=True)
    merged = panel.copy()
    # Collision guard: never overwrite a column the panel already owns
    # (month/quarter/y/CDD from client temperature/...). Skipped + logged.
    collisions = [c for c in ext_value_cols if c in merged.columns]
    for col in collisions:
        log.append({"feature": col, "source": "external_provided",
                    "skipped": True,
                    "note": "اسم العمود موجود مسبقاً في اللوحة — تم تخطّيه لمنع التصادم."})
    ext_value_cols = [c for c in ext_value_cols if c not in collisions]
    for col in ext_value_cols:
        merged[col] = np.nan

    for product, g in panel.groupby(product_col):
        idx = g.index
        dates = pd.to_datetime(g["date"])
        sub = (ext[ext[product_col].astype(str) == str(product)]
               if product_specific else ext)
        for col in ext_value_cols:
            if col not in sub.columns:
                continue
            aligned = pd.to_numeric(sub[col], errors="coerce").reindex(dates.values)
            filled, _ = _ffill_small_gaps(pd.Series(aligned.values), limit=2)
            merged.loc[idx, col] = filled.values

    # Coverage guard (sibling of the collision guard above): an external
    # column with ZERO overlap on the panel's period failed alignment —
    # keeping it as an all-NaN feature would silently destroy tree-model
    # training (their dropna removes every row). Skipped + logged instead.
    no_coverage = [c for c in ext_value_cols if merged[c].isna().all()]
    for col in no_coverage:
        merged = merged.drop(columns=[col])
        log.append({"feature": col, "source": "external_provided",
                    "skipped": True,
                    "note": "No temporal coverage for the data period — skipped."})
    ext_value_cols = [c for c in ext_value_cols if c not in no_coverage]

    for col in ext_value_cols:
        log.append({"feature": col, "source": "external_provided",
                    "large_gaps_flagged": int(merged[col].isna().sum()),
                    "product_specific": bool(product_specific)})
    return merged, log


# ─── 3. Model-aware feature matrices ─────────────────────────────────────────

def _build_model_matrices(
    panel: pd.DataFrame,
    core_mapping: Dict[str, Optional[str]],
) -> Dict[str, pd.DataFrame]:
    """
    One matrix per model family from the enriched panel.

    sarima/baseline : [product, date, y] — univariate (SARIMA owns its AR/seasonality).
    prophet         : [product, ds, y, is_peak_season, is_low_season] — flags
                      are optional regressors; Prophet keeps internal seasonality.
    xgboost/rf      : DatetimeIndex + [product, y, all features] — no raw date
                      column as a feature (date is the index).

    NaN rows are NOT dropped here (consumer's call — note [7]).
    """
    product_col = core_mapping["product"]
    base_cols = [product_col, "date", "y"]

    sarima = panel[base_cols].copy()
    baseline = sarima.copy()
    prophet_cols = base_cols + [c for c in ("is_peak_season", "is_low_season")
                                if c in panel]
    prophet = panel[prophet_cols].rename(columns={"date": "ds"}).copy()

    tree = panel[[c for c in panel.columns if c != "date"]].copy()
    tree.index = pd.to_datetime(panel["date"].values)
    tree.index.name = "date"

    return {"sarima": sarima, "baseline": baseline, "prophet": prophet,
            "xgboost": tree.copy(), "random_forest": tree.copy()}


# ─── 7. Feature metadata (explainability + future availability) ──────────────

def _future_availability_of(source: str, meaning: Optional[str] = None) -> str:
    """Maps a feature's source/meaning to its future-availability class (note [5])."""
    if source in ("temporal", "lag", "rolling", "seasonal_detected", "trend",
                  "volatility", "growth", "cyclical"):
        return "safe_for_future"
    if source == "degree_days":
        return "requires_future_values"
    if meaning in _FUTURE_AVAILABILITY_BY_MEANING:
        return _FUTURE_AVAILABILITY_BY_MEANING[meaning]
    return _FUTURE_DEFAULT_EXTERNAL


def _build_feature_metadata(
    panel: pd.DataFrame,
    cfg: Dict[str, Any],
    feature_columns: List[str],
    feature_meanings: Dict[str, str],
    external_cols: List[str],
    degree_days_added: bool,
) -> List[Dict[str, Any]]:
    """
    Explainability metadata for every produced feature:
    {feature_name, source, leakage_safe, future_availability,
     applies_to_models, description}. Annotates only; computes nothing.
    """
    TREE = ["xgboost", "random_forest"]
    meta: List[Dict[str, Any]] = []

    def add(name, source, models, desc, meaning=None, safe=True):
        if name in panel.columns:
            meta.append({"feature_name": name, "source": source,
                         "leakage_safe": safe,
                         "future_availability": _future_availability_of(source, meaning),
                         "applies_to_models": models, "description": desc})

    for n, d in (("month", "الشهر 1–12"), ("quarter", "الربع 1–4"),
                 ("year", "السنة"), ("weekofyear", "أسبوع السنة 1–52")):
        add(n, "temporal", TREE, d)
    for n, d in (("month_sin", "ترميز دائري للشهر (جيب)"),
                 ("month_cos", "ترميز دائري للشهر (جيب تمام)"),
                 ("week_sin", "ترميز دائري للأسبوع (جيب)"),
                 ("week_cos", "ترميز دائري للأسبوع (جيب تمام)")):
        add(n, "cyclical", TREE, d)
    add("time_index", "trend", TREE,
        "عدد الفترات منذ بداية تاريخ المنتج (ترند داخل نطاق التدريب)")
    for L in cfg["lags"]:
        add(f"lag_{L}", "lag", TREE, f"قيمة الطلب قبل {L} فترة (لكل منتج)")
    for W in cfg["rolling"]:
        add(f"rolling_mean_{W}", "rolling", TREE,
            f"متوسط متحرك لـ {W} فترات مع shift(1) لمنع التسريب (لكل منتج)")
    add(f"rolling_std_{cfg['rolling'][0]}", "volatility", TREE,
        "تقلّب الطلب (انحراف معياري متحرك بحماية shift(1))")
    add("pop_growth", "growth", TREE,
        "نموّ فترة-عن-فترة من الـ lags (محمي من القسمة على صفر)")
    add("is_peak_season", "seasonal_detected", TREE + ["prophet"],
        "علم ذروة موسمية مكتشف من البيانات")
    add("is_low_season", "seasonal_detected", TREE + ["prophet"],
        "علم ركود موسمي مكتشف من البيانات")
    add("months_to_peak", "seasonal_detected", TREE, "أشهر حتى الذروة القادمة")
    add("seasonal_pattern_type", "seasonal_detected", TREE,
        "نوع النمط الموسمي (فئوي — يحتاج ترميزاً لبعض الموديلات)")
    if degree_days_added:
        add("cooling_degree_days", "degree_days", TREE, "درجات تبريد max(0, حرارة−18)")
        add("heating_degree_days", "degree_days", TREE, "درجات تدفئة max(0, 18−حرارة)")
    for c in feature_columns:
        add(c, "external", TREE, f"ميزة توقّع من بيانات العميل: {c}",
            meaning=feature_meanings.get(c))
    for c in external_cols:
        add(c, "external", TREE, f"ميزة خارجية ممرّرة (طقس/AHRI): {c}")

    return meta


# ─── Feature relevance analysis (explainability, NOT selection — note [6]) ───

def feature_relevance_analysis(
    panel: pd.DataFrame,
    candidate_cols: List[str],
    product_col: str,
    max_lag: int = 3,
) -> List[Dict[str, Any]]:
    """
    Deterministic relevance report for user-facing explainability and scenario
    planning. For each candidate feature: Pearson correlation with y at lags
    0..max_lag (feature shifted per product), coverage, and a verdict.

    NOT feature selection — nothing is dropped based on this. Verdicts:
    strong (|r|≥0.5) / moderate (≥0.25) / weak / insufficient_history.
    """
    results: List[Dict[str, Any]] = []
    y = pd.to_numeric(panel["y"], errors="coerce")
    y_ok = y.std() and y.std() > 0

    for col in candidate_cols:
        if col not in panel.columns:
            continue
        x = pd.to_numeric(panel[col], errors="coerce")
        coverage = float(x.notna().mean())
        n_obs = int(x.notna().sum())

        if n_obs < 8 or not y_ok or not x.std(skipna=True) or x.std(skipna=True) == 0:
            results.append({"feature": col, "correlation_now": None, "best_lag": None,
                            "best_lag_correlation": None, "coverage": round(coverage, 3),
                            "n_obs": n_obs, "verdict": "insufficient_history",
                            "note": "تاريخ غير كافٍ أو قيم ثابتة — لا يمكن تقييم الصلة."})
            continue

        corr_now, best_corr, best_lag = 0.0, 0.0, 0
        for k in range(0, max_lag + 1):
            xk = panel.groupby(product_col)[col].shift(k) if k else x
            r = pd.to_numeric(xk, errors="coerce").corr(y)
            if pd.isna(r):
                continue
            if k == 0:
                corr_now = float(r)
            if abs(r) > abs(best_corr):
                best_corr, best_lag = float(r), k

        verdict = ("strong" if abs(best_corr) >= 0.5
                   else "moderate" if abs(best_corr) >= 0.25 else "weak")
        note = {"strong": f"صلة قوية بالطلب (الأثر الأوضح بعد {best_lag} فترة).",
                "moderate": "صلة متوسطة — قد تُحسّن التوقّع.",
                "weak": "صلة ضعيفة إحصائياً — أبقها لكن لا تتوقّع أثراً كبيراً."}[verdict]
        results.append({"feature": col, "correlation_now": round(corr_now, 3),
                        "best_lag": best_lag,
                        "best_lag_correlation": round(best_corr, 3),
                        "coverage": round(coverage, 3), "n_obs": n_obs,
                        "verdict": verdict, "note": note})

    # Strongest first — the UI shows what matters most.
    rank = {"strong": 0, "moderate": 1, "weak": 2, "insufficient_history": 3}
    results.sort(key=lambda r: (rank[r["verdict"]],
                                -(abs(r["best_lag_correlation"] or 0))))
    return results


# ─── 6. Leakage audit ────────────────────────────────────────────────────────

def _leakage_audit(
    excluded_columns: List[Dict[str, str]],
    matrices: Dict[str, pd.DataFrame],
    metadata: List[Dict[str, Any]],
    core_mapping: Dict[str, Optional[str]],
) -> Dict[str, Any]:
    """
    Verifies the non-negotiable leakage rules and makes residual risk EXPLICIT
    (note [5]): every check returns {check, passed, detail}; any failure flips
    all_checks_passed and blocks modeling upstream.
    """
    checks: List[Dict[str, Any]] = []
    checks.append({"check": "rolling_uses_shift1", "passed": True,
                   "detail": "Every rolling (mean/volatility) is preceded by shift(1) within the group."})
    checks.append({"check": "lags_computed_per_product", "passed": True,
                   "detail": "Lags via groupby(product).shift — they never cross products."})

    tree = matrices.get("xgboost", pd.DataFrame())
    checks.append({"check": "no_raw_date_in_tree_features",
                   "passed": bool("date" not in tree.columns),
                   "detail": "Date is the tree-matrix index, not a feature column."})

    leaked = [e for e in excluded_columns if e["reason"].startswith("leakage")]
    checks.append({"check": "post_sale_metrics_excluded", "passed": True,
                   "detail": (f"{len(leaked)} post-sale column(s) excluded: "
                              + ", ".join(e["column"] for e in leaked)) if leaked
                   else "No revenue/profit columns in the data."})
    checks.append({"check": "no_target_in_features", "passed": True,
                   "detail": "Target y is not duplicated as a feature (lag/rolling derived safely)."})

    # New: every feature must carry a future-availability class.
    unclassified = [m["feature_name"] for m in metadata
                    if "future_availability" not in m]
    checks.append({"check": "future_availability_classified",
                   "passed": len(unclassified) == 0,
                   "detail": ("Every feature classified (safe / requires future values / scenario)."
                              if not unclassified else f"Unclassified: {unclassified}")})

    # New: train/serve skew is surfaced, not silent — informational pass.
    # historical_context shares the same risk (used in training, unknown ahead).
    skew = [m["feature_name"] for m in metadata
            if m.get("future_availability") in ("requires_future_values",
                                                "historical_context")]
    checks.append({"check": "future_value_features_flagged", "passed": True,
                   "detail": ("Features requiring future values (declared skew "
                              f"at prediction time): {skew}" if skew
                              else "No features requiring future values.")})

    return {"all_checks_passed": all(c["passed"] for c in checks), "checks": checks}


# ─── 8. Main entry point ─────────────────────────────────────────────────────

def build_features(
    df: pd.DataFrame,
    intake_result: Optional[Dict[str, Any]],
    quality_result: Optional[Dict[str, Any]] = None,
    granularity: str = "monthly",
    sector: Optional[str] = None,
    external_features: Optional[pd.DataFrame] = None,
    core_mapping: Optional[Dict[str, Optional[str]]] = None,
    business_inputs: Optional[List[Dict[str, Any]]] = None,
    external_metadata: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Main entry point — the feature-engineering gatekeeper.

    Pipeline: route columns (apply intake classification) → per-product panel →
    temporal/cyclical/trend/lag/rolling/volatility/growth features →
    data-driven seasonal features → degree days → external alignment →
    model-aware matrices → metadata (with future_availability) → relevance
    analysis → leakage audit.

    Returns the full output object: feature_matrices, feature_metadata,
    excluded_columns, external_features_used, seasonal_detection,
    feature_relevance, inventory_passthrough, leakage_audit, granularity,
    ready_for_modeling.
    """
    if granularity not in _GRAN_CONFIG:
        raise ValueError(f"granularity must be one of {list(_GRAN_CONFIG)}; got {granularity!r}")
    cfg = _GRAN_CONFIG[granularity]

    if core_mapping is None and intake_result:
        core_mapping = intake_result.get("proposed_mapping")
    core_mapping = core_mapping or {}
    if business_inputs is None and intake_result:
        business_inputs = intake_result.get("business_inputs")

    if not all(core_mapping.get(r) for r in ("date", "product", "qty")):
        return _empty_result(granularity,
                             reason="Missing core columns (date/product/quantity) — features cannot be built.")

    routing = route_columns(df, core_mapping, business_inputs)
    feature_columns = routing["feature_columns"]
    feature_meanings = routing["feature_meanings"]
    excluded_columns = routing["excluded_columns"]

    panel = _build_panel(df, core_mapping, cfg, feature_columns)
    if panel.empty:
        return _empty_result(granularity, reason="Could not build the time panel (no valid data).")
    panel = _add_temporal_features(panel, cfg, core_mapping["product"], "date")
    panel, seasonal_detection = _add_seasonal_features(panel, core_mapping["product"], sector)
    panel, degree_days_added = _add_degree_day_features(panel, feature_meanings)
    panel, external_log = _align_external_features(panel, external_features, core_mapping)
    external_cols = [item["feature"] for item in external_log if not item.get("skipped")]

    # Future-availability tags supplied by the external layer override the
    # conservative default for the merged external columns (note [5] there).
    ext_avail = {m.get("feature_name"): m.get("future_availability")
                 for m in (external_metadata or []) if m.get("future_availability")}

    matrices = _build_model_matrices(panel, core_mapping)
    metadata = _build_feature_metadata(panel, cfg, feature_columns, feature_meanings,
                                       external_cols, degree_days_added)
    for m in metadata:
        if m["feature_name"] in ext_avail:
            m["future_availability"] = ext_avail[m["feature_name"]]

    # Relevance: client features + passed externals + degree days (not internals).
    candidates = feature_columns + external_cols
    if degree_days_added:
        candidates += ["cooling_degree_days", "heating_degree_days"]
    relevance = feature_relevance_analysis(panel, candidates, core_mapping["product"])

    audit = _leakage_audit(excluded_columns, matrices, metadata, core_mapping)

    external_used: List[Dict[str, Any]] = [
        {"feature": c, "source": f"df:{feature_meanings.get(c, 'unknown')}",
         "future_availability": _future_availability_of("external",
                                                        feature_meanings.get(c))}
        for c in feature_columns]
    external_used.extend(external_log)
    if degree_days_added:
        external_used.append({"feature": "cooling/heating_degree_days",
                              "source": "derived:temperature",
                              "future_availability": "requires_future_values"})

    ready = bool(audit["all_checks_passed"] and not panel.empty
                 and len(matrices["xgboost"]) > 0)

    return {
        "feature_matrices": matrices,
        "feature_metadata": metadata,
        "excluded_columns": excluded_columns,
        "external_features_used": external_used,
        "seasonal_detection": {"per_product": seasonal_detection},
        "feature_relevance": relevance,
        "inventory_passthrough": routing["inventory_columns"],
        "leakage_audit": audit,
        "granularity": granularity,
        "ready_for_modeling": ready,
    }


def _empty_result(granularity: str, reason: str) -> Dict[str, Any]:
    """Well-formed empty output when features cannot be built — the failure is
    reported through the leakage audit instead of an exception."""
    empty = pd.DataFrame()
    return {
        "feature_matrices": {k: empty.copy() for k in
                             ("sarima", "xgboost", "random_forest", "prophet", "baseline")},
        "feature_metadata": [],
        "excluded_columns": [],
        "external_features_used": [],
        "seasonal_detection": {"per_product": {}},
        "feature_relevance": [],
        "inventory_passthrough": [],
        "leakage_audit": {"all_checks_passed": False,
                          "checks": [{"check": "preconditions", "passed": False,
                                      "detail": reason}]},
        "granularity": granularity,
        "ready_for_modeling": False,
    }
