"""
external_features.py — External Features Layer | Bayyina Platform

Position: after quality.py and before feature_engineering.py.
Role: the PLATFORM provides external features internally — the user never
uploads weather/economic files. GENERIC manufacturing mode: built-in generic
calendar features in code, weather/industry indicators from internal CSVs when
present (graceful fallback when absent), deterministic recommendation, and a
ready-to-merge DataFrame plus lifecycle metadata.

Main entry point: build_external_features(start_date, end_date, granularity, ...)

This layer does NOT: build lags/rolling, detect seasonality, select features,
train models, call external APIs, or include any regional calendar logic
(no Ramadan/Eid/national holidays at this stage — see note [2]).
Time-series transformation stays in feature_engineering.py.
"""

# ===========================================================================
# DEVELOPER NOTES
# ===========================================================================
#
# [1] Registry = a plain dict, deliberately. No database, no service, no
#     plugin system. Each entry carries the feature's lifecycle metadata
#     (source, industry relevance, future availability, scenario capability,
#     leakage risk). Adding a feature = adding a dict entry.
#
# [2] GENERIC MODE (current): the platform targets generic manufacturing
#     forecasting. All regional calendar logic (Ramadan/Eid/Saudi holidays)
#     was REMOVED from generation, registry, and recommendations. Defaults:
#     country="generic", industry="manufacturing". Regional calendars can
#     return later as an opt-in registry block without touching this layer's
#     structure.
#
# [3] Calendar features here (month/quarter/year/weekofyear) duplicate what
#     feature_engineering derives inside the modeling panel — INTENTIONALLY:
#     this frame must be self-contained for standalone consumers (future
#     scenario engine, UI date grids). On merge, fe._align_external_features'
#     collision guard skips them automatically, so the modeling panel keeps a
#     single source for temporal features. Zero double-signal risk.
#
# [4] Weather = internal historical/climatology CSV
#     (weather_<city>_monthly.csv). If the requested city's file is missing we
#     fall back to the default internal file WITH a warning — silently serving
#     one city's climate for another would be dishonest. CDD/HDD derived here
#     (base 18°C) so feature_engineering needs no knowledge of this layer.
#
# [5] Future availability is explicit per feature and travels WITH the frame
#     (feature_metadata) into feature_engineering via build_features's
#     external_metadata parameter:
#       safe_for_future        — calendar (computable for any date)
#       requires_future_values — weather actuals (unknown at forecast time)
#       historical_context     — industry indicators published with a lag
#                                (AHRI, construction)
#       scenario_only          — reserved for the scenario engine's levers
#     The model must never silently depend on a feature that will not exist
#     at forecast time — this tagging feeds the leakage audit downstream.
#
# [6] Recommendation is deterministic and explainable: industry/granularity in
#     → matching registry entries + reason + availability out. No LLM.
#
# Known limitations:
#   [L-a] One weather city per run; multi-region clients get one climate.
#   [L-b] Weekly granularity upsamples monthly weather (ffill) — coarse.
#   [L-c] Internal weather file is demo climatology unless replaced with real
#         observations (see data/external/README.md).
#
# ===========================================================================

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

# Internal data directory (MVP: project files, no APIs).
DATA_DIR = Path(__file__).parent / "data" / "external"

_FREQ = {"monthly": "MS", "weekly": "W"}

# Degree-day base (°C), common ASHRAE choice — keep equal to FE's constant.
_DEGREE_DAY_BASE = 18.0

# Generic defaults (note [2]).
_DEFAULT_COUNTRY = "generic"
_DEFAULT_INDUSTRY = "manufacturing"


# ─── Feature registry (note [1]) ─────────────────────────────────────────────

_REGISTRY: Dict[str, Dict[str, Any]] = {
    # Calendar — generated in code, always available (note [3]).
    "month": dict(
        category="calendar", source="builtin_code", source_type="generated",
        country="all", city_or_region="all", industry_relevance=["all"],
        future_availability="safe_for_future", scenario_capable=False,
        leakage_risk="low", required_columns=[],
        description="رقم الشهر 1–12"),
    "quarter": dict(
        category="calendar", source="builtin_code", source_type="generated",
        country="all", city_or_region="all", industry_relevance=["all"],
        future_availability="safe_for_future", scenario_capable=False,
        leakage_risk="low", required_columns=[],
        description="رقم الربع 1–4"),
    "year": dict(
        category="calendar", source="builtin_code", source_type="generated",
        country="all", city_or_region="all", industry_relevance=["all"],
        future_availability="safe_for_future", scenario_capable=False,
        leakage_risk="low", required_columns=[],
        description="السنة"),
    "weekofyear": dict(
        category="calendar", source="builtin_code", source_type="generated",
        country="all", city_or_region="all", industry_relevance=["all"],
        future_availability="safe_for_future", scenario_capable=False,
        leakage_risk="low", required_columns=[],
        description="أسبوع السنة 1–52 (للتدرّج الأسبوعي فقط)"),
    # Weather — internal CSV; useful for any temperature-sensitive demand.
    "temperature": dict(
        category="weather", source="internal_csv", source_type="csv",
        country="all", city_or_region="default",
        industry_relevance=["all"],
        future_availability="requires_future_values", scenario_capable=True,
        leakage_risk="low", required_columns=["date", "temperature"],
        description="متوسط درجة الحرارة الشهرية (ملف داخلي)",
        filename="weather_{city}_monthly.csv"),
    "cooling_degree_days": dict(
        category="weather", source="derived", source_type="generated",
        country="all", city_or_region="default",
        industry_relevance=["all"],
        future_availability="requires_future_values", scenario_capable=True,
        leakage_risk="low", required_columns=["temperature"],
        description="درجات التبريد max(0, حرارة−18) — مشتقّة من الحرارة"),
    "heating_degree_days": dict(
        category="weather", source="derived", source_type="generated",
        country="all", city_or_region="default",
        industry_relevance=["all"],
        future_availability="requires_future_values", scenario_capable=True,
        leakage_risk="low", required_columns=["temperature"],
        description="درجات التدفئة max(0, 18−حرارة) — مشتقّة من الحرارة"),
    "humidity": dict(
        category="weather", source="internal_csv", source_type="csv",
        country="all", city_or_region="default",
        industry_relevance=["all"],
        future_availability="requires_future_values", scenario_capable=True,
        leakage_risk="low", required_columns=["date", "humidity"],
        description="متوسط الرطوبة (اختياري داخل ملف الطقس)",
        filename="weather_{city}_monthly.csv"),
    # Industry indicators — internal CSVs, loaded only when relevant.
    "ahri_shipments": dict(
        category="industry", source="internal_csv", source_type="csv",
        country="all", city_or_region="all", industry_relevance=["HVAC"],
        future_availability="historical_context", scenario_capable=False,
        leakage_risk="low", required_columns=["date", "ahri_shipments"],
        description="شحنات AHRI الصناعية — مؤشر سياق تاريخي يُنشر بتأخير",
        filename="ahri_shipments.csv"),
    "construction_activity": dict(
        category="industry", source="internal_csv", source_type="csv",
        country="all", city_or_region="all",
        industry_relevance=["manufacturing", "HVAC", "construction"],
        future_availability="historical_context", scenario_capable=False,
        leakage_risk="low", required_columns=["date", "construction_activity"],
        description="مؤشر نشاط البناء — يُفعَّل فقط إن وُجد الملف الداخلي",
        filename="construction_activity.csv"),
}


def get_registry() -> Dict[str, Dict[str, Any]]:
    """Read-only view of the registry for UI/metadata consumers."""
    return {k: dict(v) for k, v in _REGISTRY.items()}


# ─── Period grid ─────────────────────────────────────────────────────────────

def _period_grid(start_date, end_date, granularity: str) -> pd.DatetimeIndex:
    """Period index covering [start, end] at the target granularity."""
    freq = _FREQ[granularity]
    idx = pd.date_range(pd.to_datetime(start_date).normalize(),
                        pd.to_datetime(end_date).normalize(), freq=freq)
    if len(idx) == 0:  # range shorter than one period → anchor on start
        idx = pd.date_range(pd.to_datetime(start_date).normalize(), periods=1, freq=freq)
    return idx


# ─── Calendar features (generic, built-in — note [3]) ────────────────────────

def build_calendar_features(start_date, end_date, granularity: str) -> pd.DataFrame:
    """
    Generates GENERIC calendar features in code — no files, no region:
    month, quarter, year (+ weekofyear when weekly).

    These duplicate feature_engineering's panel temporals on purpose (note [3]);
    the FE merge collision guard deduplicates, and standalone consumers
    (scenario engine, UI) get a self-contained frame.
    """
    idx = _period_grid(start_date, end_date, granularity)
    out = pd.DataFrame(index=idx)
    out["month"] = idx.month
    out["quarter"] = idx.quarter
    out["year"] = idx.year
    if granularity == "weekly":
        out["weekofyear"] = idx.isocalendar().week.astype(int)
    out.index.name = "date"
    return out


# ─── Internal CSV loaders (graceful fallback — note [4]) ─────────────────────

def _load_timeseries_csv(path: Path, value_cols: List[str], granularity: str,
                         grid_index: pd.DatetimeIndex,
                         warnings_list: List[str]) -> Optional[pd.DataFrame]:
    """
    Loads a date-indexed CSV and aligns it to the period grid. Monthly data at
    weekly granularity is upsampled (ffill) with a warning [L-b].
    Returns None (never raises) when the file is missing or unreadable.
    """
    if not path.exists():
        return None
    try:
        raw = pd.read_csv(path)
        raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
        raw = raw.dropna(subset=["date"]).set_index("date").sort_index()
        cols = [c for c in value_cols if c in raw.columns]
        if not cols:
            warnings_list.append(f"{path.name}: لا يحتوي الأعمدة المتوقعة {value_cols}.")
            return None
        data = raw[cols].apply(pd.to_numeric, errors="coerce")
        freq = _FREQ[granularity]
        resampled = data.resample(freq).mean()
        if granularity == "weekly" and len(resampled) < len(grid_index) / 2:
            warnings_list.append(f"{path.name}: بيانات شهرية وُسّعت أسبوعياً (ffill) — دقة خشنة.")
            monthly = data.resample("MS").mean()
            resampled = monthly.reindex(grid_index.union(monthly.index)).ffill()
        return resampled.reindex(grid_index).ffill(limit=2)
    except Exception as exc:
        warnings_list.append(f"تعذّر قراءة {path.name} ({exc}) — تم تجاهل المصدر.")
        return None


def load_weather_features(city: Optional[str], granularity: str,
                          grid_index: pd.DatetimeIndex, data_dir: Path,
                          warnings_list: List[str]) -> Optional[pd.DataFrame]:
    """
    Loads weather_<city>_monthly.csv. When the requested city has no file,
    falls back to the default internal weather file WITH a warning (note [4]).
    Derives cooling/heating degree days. Returns None when no file exists.
    """
    requested = (city or "riyadh").strip().lower().replace(" ", "")
    path = data_dir / f"weather_{requested}_monthly.csv"
    if not path.exists():
        # Default internal climatology file (demo/support data — see README).
        fallback = data_dir / "weather_riyadh_monthly.csv"
        if fallback.exists() and requested != "riyadh":
            warnings_list.append(f"لا ملف طقس لمدينة «{city}» — استُخدم ملف الطقس "
                                 "الافتراضي الداخلي (قد لا يمثّل مناخ مدينتك).")
            path = fallback
        elif fallback.exists():
            path = fallback
    frame = _load_timeseries_csv(path, ["temperature", "humidity", "rainfall"],
                                 granularity, grid_index, warnings_list)
    if frame is None or "temperature" not in frame.columns:
        return frame
    temp = frame["temperature"]
    frame["cooling_degree_days"] = (temp - _DEGREE_DAY_BASE).clip(lower=0)
    frame["heating_degree_days"] = (_DEGREE_DAY_BASE - temp).clip(lower=0)
    return frame


def load_industry_features(industry: Optional[str], granularity: str,
                           grid_index: pd.DatetimeIndex, data_dir: Path,
                           warnings_list: List[str]) -> Optional[pd.DataFrame]:
    """
    Loads industry indicator CSVs by relevance:
      AHRI shipments        — only when industry == 'HVAC';
      construction activity — manufacturing/HVAC/construction, only if the
                              internal file exists (explicitly available).
    Returns None when nothing relevant/available.
    """
    ind = (industry or _DEFAULT_INDUSTRY).upper()
    frames: List[pd.DataFrame] = []

    if ind == "HVAC":
        ahri = _load_timeseries_csv(data_dir / "ahri_shipments.csv",
                                    ["ahri_shipments"], granularity,
                                    grid_index, warnings_list)
        if ahri is not None:
            frames.append(ahri)

    if ind in ("MANUFACTURING", "HVAC", "CONSTRUCTION"):
        cons = _load_timeseries_csv(data_dir / "construction_activity.csv",
                                    ["construction_activity"], granularity,
                                    grid_index, warnings_list)
        if cons is not None:
            frames.append(cons)

    return pd.concat(frames, axis=1) if frames else None


# ─── Deterministic recommendation (note [6]) ─────────────────────────────────

def recommend_features(country: Optional[str] = None, city: Optional[str] = None,
                       industry: Optional[str] = None, granularity: str = "monthly",
                       data_dir: Optional[Path] = None) -> List[Dict[str, Any]]:
    """
    Rule-based, explainable recommendation: registry entries matching the
    industry context, each with a reason and an availability status.
    Country plays no role in generic mode (no regional features exist).
    No LLM — same inputs always give the same answer.
    """
    data_dir = Path(data_dir) if data_dir else DATA_DIR
    ind = (industry or _DEFAULT_INDUSTRY)
    recs: List[Dict[str, Any]] = []

    for name, spec in _REGISTRY.items():
        # weekofyear only makes sense at weekly granularity.
        if name == "weekofyear" and granularity != "weekly":
            continue
        relevant = ("all" in spec["industry_relevance"]
                    or ind.upper() in [i.upper() for i in spec["industry_relevance"]])
        if not relevant:
            continue

        # Availability: builtin/derived always; CSV-backed needs the file.
        if spec["source"] in ("builtin_code", "derived"):
            available = True
        else:
            fname = spec.get("filename", "").replace(
                "{city}", (city or "riyadh").strip().lower().replace(" ", ""))
            available = (data_dir / fname).exists() or \
                (spec["category"] == "weather"
                 and (data_dir / "weather_riyadh_monthly.csv").exists())

        if spec["category"] == "calendar":
            reason = "ميزة تقويمية عامة — آمنة مستقبلاً ومفيدة عبر القطاعات."
        elif spec["category"] == "weather":
            reason = "الطلب الحسّاس للحرارة يتأثر بالمناخ — قابلة للسيناريوهات."
        else:
            reason = f"مؤشر صناعي ذو صلة بقطاع {ind} — سياق تاريخي للتفسير."

        recs.append({"feature": name, "category": spec["category"],
                     "reason": reason, "available": bool(available),
                     "future_availability": spec["future_availability"]})
    return recs


# ─── Main public entry point ─────────────────────────────────────────────────

def build_external_features(
    start_date,
    end_date,
    granularity: str = "monthly",
    country: Optional[str] = None,
    city: Optional[str] = None,
    industry: Optional[str] = None,
    include_weather: bool = True,
    include_industry: bool = True,
    data_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Builds the platform-provided external features frame for a date range.

    GENERIC mode defaults: country='generic', industry='manufacturing'.
    Calendar features are generated in code (always available); weather and
    industry features load from internal CSVs with graceful fallback — a
    missing file produces an 'unavailable_features' entry, never a crash.
    No regional calendar features are generated (note [2]).

    Returns {external_features: DataFrame (date-indexed at granularity),
             feature_metadata, recommended_features, unavailable_features,
             warnings, ready_for_feature_engineering}.
    feature_metadata is shaped for feature_engineering.build_features's
    external_metadata parameter (future-availability tagging — note [5]).
    """
    if granularity not in _FREQ:
        raise ValueError(f"granularity must be one of {list(_FREQ)}; got {granularity!r}")
    ddir = Path(data_dir) if data_dir else DATA_DIR
    country = country or _DEFAULT_COUNTRY
    industry = industry or _DEFAULT_INDUSTRY
    warnings_list: List[str] = []
    unavailable: List[Dict[str, str]] = []

    grid_index = _period_grid(start_date, end_date, granularity)
    frames: List[pd.DataFrame] = []

    # 1) Calendar — generic, built-in, zero files required.
    frames.append(build_calendar_features(start_date, end_date, granularity))

    # 2) Weather — internal CSV, graceful.
    if include_weather:
        weather = load_weather_features(city, granularity, grid_index, ddir, warnings_list)
        if weather is not None:
            frames.append(weather)
        else:
            unavailable.append({"feature": "weather",
                                "reason": "لا ملف طقس داخلياً (data/external/weather_*.csv) — "
                                          "أضِفه لتفعيل ميزات الحرارة."})

    # 3) Industry — internal CSVs by relevance, graceful.
    if include_industry:
        ind_frame = load_industry_features(industry, granularity, grid_index,
                                           ddir, warnings_list)
        if ind_frame is not None:
            frames.append(ind_frame)
        else:
            if industry.upper() == "HVAC" and not (ddir / "ahri_shipments.csv").exists():
                unavailable.append({"feature": "ahri_shipments",
                                    "reason": "لا ملف ahri_shipments.csv داخلياً — "
                                              "أضِفه لتفعيل مؤشر الشحنات."})
            if not (ddir / "construction_activity.csv").exists():
                unavailable.append({"feature": "construction_activity",
                                    "reason": "لا ملف construction_activity.csv داخلياً — "
                                              "مؤشر البناء غير مفعّل."})

    external = pd.concat(frames, axis=1)
    external.index.name = "date"

    # Metadata only for columns actually produced, straight from the registry.
    metadata = [dict(feature_name=col, **{k: v for k, v in _REGISTRY[col].items()
                                          if k != "filename"})
                for col in external.columns if col in _REGISTRY]

    recommended = recommend_features(country, city, industry, granularity, ddir)

    return {
        "external_features": external,
        "feature_metadata": metadata,
        "recommended_features": recommended,
        "unavailable_features": unavailable,
        "warnings": warnings_list,
        "ready_for_feature_engineering": bool(len(external.columns) > 0),
    }
