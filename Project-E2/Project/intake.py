"""
intake.py — Data Intake Layer | Bayyina Platform

The intake layer ONLY understands the uploaded dataset:
  - detects the required columns (date / product / demand quantity)
  - discovers the business meaning of every other column
  - derives platform capabilities and readiness from what is present
  - produces a mapping proposal a UI can review, correct, and confirm
  - remembers confirmed mappings for future uploads (mapping_memory)
  - optionally consults an LLM for ambiguous columns (llm_mapper, disabled by default)

It does NOT clean data, judge value quality, or score data quality —
that is quality.py's permanent responsibility (see boundary note below).

Public API:
  run_intake(df, llm_client=None, memory_path=None)   ← new main entry point
  apply_user_corrections(intake_result, corrections)
  confirm_mapping(intake_result, memory_path=None)
  build_intake_result(df)                              ← legacy entry (pipeline.py)
"""

# ===========================================================================
# DEVELOPER NOTES
# ===========================================================================
#
# [1] Responsibility boundary (permanent): intake = column UNDERSTANDING only.
#     All cleaning/validation/anomaly logic was REMOVED from this file — it
#     duplicated quality.py executors and risked silent divergence between two
#     cleaning paths. quality.py owns anything that touches or judges values.
#
# [2] Detection = multi-signal voting, never name alone:
#     name tokens (0.5) + dtype (0.2) + value shape (0.3). A high-confidence
#     wrong mapping silently unlocks wrong capabilities downstream, so "high"
#     confidence requires name AND structure to agree (_meaning_confidence).
#
# [3] Token-based name matching (NOT substring): "Holiday" must not match "id",
#     "Last Update" must not match "date", "Customer Name" must not win product.
#     Generic tokens (id/name/value/...) score 0.45 and can never decide alone.
#
# [4] Large files: all value-shape signals run on a deterministic sample
#     (first 1000 rows + seeded random rest, capped at _SAMPLE_CAP). Only cheap
#     vectorized aggregates touch the full column.
#
# [5] Memory before LLM: a schema fingerprint hit replays the user-confirmed
#     mapping at 0.95 confidence — most clients re-upload the same structure
#     monthly, so the LLM becomes a first-upload-only event.
#
# [6] LLM is prepared, not enabled (llm_client=None default). It only sees
#     metadata + sample values for ambiguous columns, its confidence is capped
#     at 0.70, and its proposals ALWAYS stay in needs_review. It suggests;
#     the user decides. It never overrides user or memory sources.
#
# [7] User corrections outrank everything: source="user", confidence=1.0.
#     Corrections sync BOTH the new contract and the legacy keys
#     (proposed_mapping / business_inputs) so quality.py and
#     feature_engineering.py see the corrected view. Saving to memory happens
#     only at confirm_mapping, never on mere edits.
#
# [8] Arabic strings that remain in this file are USER-FACING DATA (labels,
#     explanations, UI messages) — the product UI is Arabic. Code comments,
#     docstrings, and developer notes are English by project convention.
#
# Known limitations:
#   - Numeric Unix-timestamp date columns get at most "medium" confidence.
#   - Greedy role assignment (date→product→qty) may pick suboptimally in rare
#     contention cases; runner-ups are surfaced as alternatives for the user.
#   - Confidence labels map to fixed scores (high/med/low → .85/.62/.35) in
#     proposals — a stated simplification, not a calibration.
#
# ===========================================================================

import re
import warnings
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

import mapping_memory


# ─── Constants & registries ──────────────────────────────────────────────────

# Synonyms for the three required roles (Arabic/English). Token-matched.
_CORE_SYNONYMS: Dict[str, List[str]] = {
    "date": [
        "date", "time", "day", "month", "year", "period", "ds", "timestamp",
        "datetime", "تاريخ", "التاريخ", "يوم", "اليوم", "شهر", "الشهر",
        "سنة", "السنة", "فترة", "الفترة", "وقت", "الوقت",
    ],
    "product": [
        "product", "item", "sku", "code", "material", "part", "model", "name",
        "article", "id", "project", "منتج", "المنتج", "صنف", "الصنف", "كود",
        "الكود", "رمز", "الرمز", "مادة", "المادة", "موديل", "اسم", "مشروع",
    ],
    "qty": [
        "qty", "quantity", "amount", "volume", "units", "sold", "sales",
        "demand", "count", "value", "كمية", "الكمية", "عدد", "العدد",
        "مبيعات", "المبيعات", "طلب", "الطلب", "حجم", "وحدات",
    ],
}

# Tokens too generic to decide a role alone — capped at 0.45 (note [3]).
_GENERIC_TOKENS = {
    "id", "code", "name", "value", "amount", "count", "number", "no", "type",
    "رقم", "اسم", "كود", "عدد", "قيمة", "نوع",
}

# Sample cap for value-shape signals and profiling (note [4]).
_SAMPLE_CAP = 10_000

# Business meaning registry — single source of truth for non-core columns.
# keywords: token-matched synonyms; value_hint/temporal_hint: expected shape;
# legacy_category: forecast_feature | inventory_input | advisory (consumed by
# feature_engineering.route_columns); stockout_signal: future stockout prep.
_BUSINESS_INPUT_REGISTRY: Dict[str, Dict[str, Any]] = {
    "stock_on_hand": dict(
        label_ar="المخزون الحالي",
        keywords=["stockonhand", "onhand", "stock", "available", "availablestock",
                  "inventory", "مخزون", "رصيد", "متاح", "المتاح"],
        value_hint="non_negative", temporal_hint="time_varying",
        possible_uses=["forecast_context", "stockout_detection", "inventory_recommendation"],
        legacy_category="inventory_input", stockout_signal=True),
    "available_inventory": dict(
        label_ar="المخزون المتاح",
        keywords=["availableinventory", "freestock", "متاح", "المتاح"],
        value_hint="non_negative", temporal_hint="time_varying",
        possible_uses=["stockout_detection", "inventory_recommendation"],
        legacy_category="inventory_input", stockout_signal=True),
    "inventory_balance": dict(
        label_ar="رصيد المخزون",
        keywords=["inventorybalance", "balance", "closingstock", "رصيد", "الرصيد"],
        value_hint="any", temporal_hint="time_varying",
        possible_uses=["stockout_detection", "inventory_recommendation"],
        legacy_category="inventory_input", stockout_signal=True),
    "safety_stock": dict(
        label_ar="مخزون الأمان",
        keywords=["safetystock", "safety", "buffer", "أمان", "احتياطي"],
        value_hint="non_negative", temporal_hint="static_per_product",
        possible_uses=["inventory_recommendation", "risk_monitoring"],
        legacy_category="inventory_input", stockout_signal=False),
    "reorder_point": dict(
        label_ar="نقطة إعادة الطلب",
        keywords=["reorderpoint", "reorder", "rop", "إعادةطلب", "نقطةالطلب"],
        value_hint="non_negative", temporal_hint="static_per_product",
        possible_uses=["reorder_planning", "inventory_recommendation"],
        legacy_category="inventory_input", stockout_signal=False),
    "lead_time": dict(
        label_ar="مهلة التوريد",
        keywords=["leadtime", "lead", "deliverytime", "مهلة", "مهلةالتوريد"],
        value_hint="positive", temporal_hint="static_per_product",
        possible_uses=["reorder_planning", "inventory_optimization"],
        legacy_category="inventory_input", stockout_signal=False),
    "moq": dict(
        label_ar="الحد الأدنى لطلبية",
        keywords=["moq", "minorder", "minimumorder", "حدأدنى", "أدنىطلب"],
        value_hint="positive", temporal_hint="static_per_product",
        possible_uses=["reorder_planning", "inventory_optimization"],
        legacy_category="inventory_input", stockout_signal=False),
    "supplier": dict(
        label_ar="المورّد",
        keywords=["supplier", "vendor", "مورد", "المورد", "موزع"],
        value_hint="any", temporal_hint="static_per_product",
        possible_uses=["operational_context"],
        legacy_category="advisory", stockout_signal=False),
    "fill_rate": dict(
        label_ar="نسبة تلبية الطلب",
        keywords=["fillrate", "servicelevel", "تلبية", "نسبةالتلبية", "مستوىالخدمة"],
        value_hint="ratio_0_1", temporal_hint="time_varying",
        possible_uses=["stockout_detection", "risk_monitoring"],
        legacy_category="advisory", stockout_signal=True),
    "lost_sales": dict(
        label_ar="المبيعات المفقودة",
        keywords=["lostsales", "lost", "unmet", "shortage", "مبيعاتمفقودة", "نقص"],
        value_hint="non_negative", temporal_hint="time_varying",
        possible_uses=["stockout_detection", "demand_drivers"],
        legacy_category="inventory_input", stockout_signal=True),
    "backorders": dict(
        label_ar="الطلبات المؤجّلة",
        keywords=["backorder", "backlog", "pending", "مؤجل", "متأخر", "معلّق"],
        value_hint="non_negative", temporal_hint="time_varying",
        possible_uses=["stockout_detection", "inventory_recommendation"],
        legacy_category="inventory_input", stockout_signal=True),
    "stockout": dict(
        label_ar="مؤشر نفاد المخزون",
        keywords=["stockout", "outofstock", "oos", "نفاد", "نفادمخزون"],
        value_hint="binary", temporal_hint="time_varying",
        possible_uses=["stockout_detection"],
        legacy_category="inventory_input", stockout_signal=True),
    "price": dict(
        label_ar="السعر",
        keywords=["price", "unitprice", "cost", "سعر", "السعر", "تكلفة"],
        value_hint="positive", temporal_hint="time_varying",
        possible_uses=["forecast_improvement", "scenario_simulation", "demand_drivers"],
        legacy_category="forecast_feature", stockout_signal=False),
    "promotion": dict(
        label_ar="العروض الترويجية",
        # "discount" lives in its own meaning to avoid a scoring tie.
        keywords=["promo", "promotion", "offer", "campaign", "عرض", "ترويج", "حملة"],
        value_hint="binary", temporal_hint="time_varying",
        possible_uses=["forecast_improvement", "scenario_simulation", "demand_drivers"],
        legacy_category="forecast_feature", stockout_signal=False),
    "discount": dict(
        label_ar="الخصومات",
        keywords=["discount", "rebate", "markdown", "خصم", "تخفيض", "حسم"],
        value_hint="any", temporal_hint="time_varying",
        possible_uses=["forecast_improvement", "scenario_simulation", "demand_drivers"],
        legacy_category="forecast_feature", stockout_signal=False),
    "temperature": dict(
        label_ar="درجة الحرارة",
        keywords=["temp", "temperature", "weather", "حرارة", "طقس", "درجةالحرارة"],
        value_hint="any", temporal_hint="environmental",
        possible_uses=["forecast_improvement", "demand_drivers"],
        legacy_category="forecast_feature", stockout_signal=False),
    "holiday": dict(
        label_ar="العطلات والمواسم",
        keywords=["holiday", "season", "event", "عطلة", "موسم", "مناسبة"],
        value_hint="binary", temporal_hint="environmental",
        possible_uses=["forecast_improvement", "demand_drivers"],
        legacy_category="forecast_feature", stockout_signal=False),
    "capacity": dict(
        label_ar="الطاقة الإنتاجية",
        keywords=["capacity", "production", "throughput", "طاقة", "إنتاجية", "سعةإنتاج"],
        value_hint="positive", temporal_hint="any",
        possible_uses=["production_planning", "capacity_risk"],
        legacy_category="advisory", stockout_signal=False),
    "revenue": dict(
        label_ar="الإيراد",
        # Detected and displayed but NEVER fed to models (post-sale leakage —
        # feature_engineering excludes it by name as a second guard).
        keywords=["revenue", "turnover", "income", "grosssales", "netsales",
                  "إيراد", "الإيراد", "عائد", "دخل"],
        value_hint="any", temporal_hint="time_varying",
        possible_uses=["insight_context"],
        legacy_category="advisory", stockout_signal=False),
    "customer": dict(
        label_ar="العميل",
        keywords=["customer", "client", "buyer", "account", "عميل", "العميل", "زبون", "حساب"],
        value_hint="any", temporal_hint="any",
        possible_uses=["segmentation_context"],
        legacy_category="advisory", stockout_signal=False),
    "region": dict(
        label_ar="المنطقة/الموقع",
        keywords=["region", "location", "city", "country", "area", "branch",
                  "منطقة", "مدينة", "دولة", "فرع", "موقع"],
        value_hint="any", temporal_hint="any",
        possible_uses=["segmentation_context"],
        legacy_category="advisory", stockout_signal=False),
    "category": dict(
        label_ar="الفئة/التصنيف",
        keywords=["category", "family", "group", "segment", "line",
                  "فئة", "تصنيف", "مجموعة", "خط"],
        value_hint="any", temporal_hint="static_per_product",
        possible_uses=["segmentation_context"],
        legacy_category="advisory", stockout_signal=False),
}

# Contract-bucket routing for run_intake. Default comes from legacy_category;
# advisory meanings that still belong to a bucket are overridden explicitly.
_BUCKET_OVERRIDES: Dict[str, str] = {
    "customer": "forecast", "region": "forecast", "category": "forecast",
    "revenue": "forecast",  # forecast-related for display; never a model feature
    "supplier": "inventory", "fill_rate": "inventory", "capacity": "inventory",
}

# Platform capability requirements: which inputs unlock which module.
#   core=True → needs only date/product/qty. any_of: one suffices. all_of: all.
_CAPABILITY_REQUIREMENTS: Dict[str, Dict[str, Any]] = {
    "Demand Forecasting": dict(label_ar="توقّع الطلب", core=True, any_of=[], all_of=[],
                               domain="forecast"),
    "Forecast Improvement": dict(label_ar="تحسين دقّة التوقّع",
                                 any_of=["price", "promotion", "discount", "temperature", "holiday"],
                                 domain="forecast"),
    "Demand Drivers Analysis": dict(label_ar="تحليل محرّكات الطلب",
                                    any_of=["promotion", "price", "discount", "temperature",
                                            "holiday", "lost_sales"],
                                    domain="forecast"),
    "Scenario Simulation": dict(label_ar="محاكاة السيناريوهات",
                                any_of=["price", "promotion", "discount"], domain="scenario"),
    "Inventory Recommendations": dict(label_ar="توصيات المخزون",
                                      any_of=["stock_on_hand", "safety_stock", "lead_time",
                                              "available_inventory"],
                                      domain="inventory"),
    "Stockout Detection": dict(label_ar="كشف نفاد المخزون",
                               any_of=["stock_on_hand", "available_inventory", "inventory_balance",
                                       "fill_rate", "lost_sales", "backorders", "stockout"],
                               domain="inventory"),
    "Reorder Planning": dict(label_ar="تخطيط إعادة الطلب",
                             any_of=["lead_time", "reorder_point", "moq"], domain="inventory"),
    "Inventory Optimization": dict(label_ar="تحسين المخزون", all_of=["lead_time"],
                                   any_of=["safety_stock", "stock_on_hand"], domain="inventory"),
    "Risk Monitoring": dict(label_ar="مراقبة المخاطر",
                            any_of=["safety_stock", "capacity", "fill_rate"], domain="inventory"),
    "Production Planning": dict(label_ar="تخطيط الإنتاج", any_of=["capacity"], domain="operations"),
    "Capacity Risk Monitoring": dict(label_ar="مراقبة مخاطر الطاقة", any_of=["capacity"],
                                     domain="operations"),
}

_FORECAST_CAPS = {c for c, r in _CAPABILITY_REQUIREMENTS.items() if r["domain"] in ("forecast", "scenario")}
_INVENTORY_CAPS = {c for c, r in _CAPABILITY_REQUIREMENTS.items() if r["domain"] == "inventory"}
_OPERATIONS_CAPS = {c for c, r in _CAPABILITY_REQUIREMENTS.items() if r["domain"] == "operations"}

# New-contract required roles ↔ internal core_mapping keys.
_REQUIRED_ROLES = {"date": "date", "product": "product", "demand_quantity": "qty"}

# Confidence label → numeric score for proposals (stated simplification).
_CONF_TO_SCORE = {"high": 0.85, "medium": 0.62, "low": 0.35, "n/a": 0.20}

# Proposals below this confidence are flagged for user review.
_REVIEW_THRESHOLD = 0.75


def _bucket_of(meaning: str) -> Optional[str]:
    """Maps a registry meaning to its contract bucket: 'forecast'|'inventory'|None."""
    spec = _BUSINESS_INPUT_REGISTRY.get(meaning)
    if spec is None:
        return None
    if meaning in _BUCKET_OVERRIDES:
        return _BUCKET_OVERRIDES[meaning]
    return {"forecast_feature": "forecast", "inventory_input": "inventory"}.get(spec["legacy_category"])


# ─── Name & value signals ────────────────────────────────────────────────────

def _normalize(name: str) -> str:
    """Lowercases a column name and squashes separators (keeps Arabic)."""
    return re.sub(r"[\s_\-./]+", "", str(name).strip().lower())


def _tokens_of(text: str) -> set:
    """
    Splits a name into comparable tokens: separators + camelCase boundaries,
    lowercased, plus an Arabic-definite-article-stripped variant per token.
    'LeadTime' → {'lead','time'}; 'التاريخ' → {'التاريخ','تاريخ'}.
    """
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


def _token_match(col_name: str, keywords: List[str]) -> tuple:
    """
    Token-based keyword match (note [3]). A keyword matches only if ALL its
    tokens appear as whole tokens in the column name.
    Scores: exact token-set equality 1.0; subset 0.85; generic-only tokens 0.45.
    Returns (best_score, matched_keyword).
    """
    col_tokens = _tokens_of(col_name)
    if not col_tokens:
        return 0.0, None
    best, matched = 0.0, None
    for kw in keywords:
        kw_tokens = _tokens_of(kw)
        if not kw_tokens or not kw_tokens <= col_tokens:
            continue
        if all(t in _GENERIC_TOKENS for t in kw_tokens):
            score = 0.45
        elif kw_tokens == col_tokens:
            score = 1.0
        else:
            score = 0.85
        if score > best:
            best, matched = score, kw
            if best >= 1.0:
                break
    return best, matched


def _name_signal(col_name: str, role: str) -> float:
    """Name-match score (0–1) of a column against a required role's synonyms."""
    score, _ = _token_match(col_name, _CORE_SYNONYMS[role])
    return score


def _sampled(series: pd.Series) -> pd.Series:
    """Deterministic sample for shape signals: first 1000 + seeded random rest."""
    n = len(series)
    if n <= _SAMPLE_CAP:
        return series
    head = series.iloc[:1000]
    rest = series.iloc[1000:].sample(_SAMPLE_CAP - 1000, random_state=0)
    return pd.concat([head, rest])


def _guess_dtype(series: pd.Series) -> str:
    """Coarse logical dtype label: datetime / numeric / numeric-like / categorical."""
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    if pd.to_numeric(series, errors="coerce").notna().mean() > 0.8:
        return "numeric-like"
    return "categorical/text"


def _value_shape_signals(series: pd.Series) -> Dict[str, float]:
    """
    Role-affinity scores from a SAMPLE of the values alone:
      date_score (parseable as dates), qty_score (numeric & non-negative),
      product_score (repetition ratio — products repeat across rows).
    """
    non_null = _sampled(series).dropna()
    if len(non_null) == 0:
        return {"date_score": 0.0, "qty_score": 0.0, "product_score": 0.0}

    with warnings.catch_warnings():           # probing every column as a date
        warnings.simplefilter("ignore")        # is intentional; failures expected
        date_score = float(pd.to_datetime(non_null, errors="coerce").notna().mean())

    numeric = pd.to_numeric(non_null, errors="coerce")
    numeric_ratio = numeric.notna().mean()
    qty_score = float(numeric_ratio * (numeric.dropna() >= 0).mean()) if numeric_ratio > 0 else 0.0

    product_score = float(1.0 - non_null.nunique() / len(non_null))
    return {"date_score": date_score, "qty_score": qty_score, "product_score": product_score}


def _confidence_label(score: float) -> str:
    """Maps a 0–1 score to a human label."""
    if score >= 0.75:
        return "high"
    if score >= 0.45:
        return "medium"
    return "low"


# ─── Required-column detection ───────────────────────────────────────────────

def detect_columns(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Detects the date / product / qty columns by multi-signal voting:
    name (0.5) + dtype (0.2) + value shape (0.3) per (column, role).

    Greedy assignment (strongest score wins, one role per column); runner-ups
    become 'alternatives' for the user to arbitrate — nothing decided silently.

    Returns {mapping, details, alternatives, unresolved}.
    """
    roles = ["date", "product", "qty"]
    scores: Dict[Any, Dict[str, float]] = {}
    reasons: Dict[Any, Dict[str, List[str]]] = {}

    for col in df.columns:
        series = df[col]
        shape = _value_shape_signals(series)
        is_datetime = pd.api.types.is_datetime64_any_dtype(series)
        is_numeric = pd.api.types.is_numeric_dtype(series)
        scores[col], reasons[col] = {}, {}

        for role in roles:
            name_s = _name_signal(str(col), role)
            if role == "date":
                dtype_s, shape_s = (1.0 if is_datetime else 0.0), shape["date_score"]
            elif role == "qty":
                dtype_s, shape_s = (1.0 if is_numeric else 0.0), shape["qty_score"]
            else:  # product: text-ish dtype is a weak positive
                dtype_s = 0.5 if (not is_numeric and not is_datetime) else 0.0
                shape_s = shape["product_score"]

            scores[col][role] = 0.50 * name_s + 0.20 * dtype_s + 0.30 * shape_s

            # User-facing (Arabic) explanation of why this guess was made.
            why: List[str] = []
            if name_s >= 1.0:
                why.append("اسم العمود يطابق الدور تماماً")
            elif name_s > 0:
                why.append("اسم العمود يشبه الدور")
            if dtype_s >= 1.0:
                why.append("نوع البيانات مطابق")
            if role == "date" and shape_s > 0.8:
                why.append(f"{shape_s:.0%} من القيم تتحوّل لتاريخ")
            if role == "qty" and shape_s > 0.8:
                why.append("القيم رقمية وغير سالبة")
            if role == "product" and shape_s > 0.5:
                why.append("القيم تتكرّر عبر الصفوف (نمط منتج)")
            reasons[col][role] = why

    # Greedy claim: per role, best unclaimed column above the 0.30 floor wins.
    mapping: Dict[str, Optional[str]] = {}
    alternatives: Dict[str, List[Any]] = {}
    unresolved: List[str] = []
    claimed: set = set()
    ranked = {role: sorted(df.columns, key=lambda c: scores[c][role], reverse=True)
              for role in roles}

    for role in roles:
        winner, alts = None, []
        for col in ranked[role]:
            if scores[col][role] < 0.30:
                break
            if col in claimed:
                continue
            if winner is None:
                winner = col
            else:
                alts.append(col)
        if winner is not None:
            mapping[role] = winner
            claimed.add(winner)
            alternatives[role] = alts
        else:
            mapping[role] = None
            alternatives[role] = []
            unresolved.append(role)

    role_of_col = {v: k for k, v in mapping.items() if v is not None}
    details: Dict[Any, Any] = {}
    for col in df.columns:
        role = role_of_col.get(col, "extra")
        if role == "extra":
            details[col] = {"role": "extra", "confidence": "n/a", "score": 0.0,
                            "reason": ["ليس من الأعمدة الأساسية — يُفحص كعمود إضافي"]}
        else:
            sc = scores[col][role]
            details[col] = {"role": role, "confidence": _confidence_label(sc),
                            "score": round(float(sc), 3),
                            "reason": reasons[col][role] or ["تخمين ضعيف — يُستحسن تأكيد العميل"]}

    return {"mapping": mapping, "details": details,
            "alternatives": alternatives, "unresolved": unresolved}


# ─── Business meaning discovery ──────────────────────────────────────────────

def _profile_structure(df: pd.DataFrame, col: Any,
                       core_mapping: Dict[str, Optional[str]]) -> Dict[str, Any]:
    """
    Structural profile of one column: dtype flags, value-shape ratios, and its
    relationship to the product/time dimensions — the strongest discriminator
    between meanings (lead_time is static per product; stock varies over time;
    temperature is environmental: same value across products per date).
    Shape ratios use a sample; the groupby relations use the full frame.
    """
    date_col, product_col = core_mapping.get("date"), core_mapping.get("product")
    s = _sampled(df[col])
    non_null = s.dropna()
    n_nn = len(non_null)
    numeric = pd.to_numeric(s, errors="coerce")
    num_nn = numeric.dropna()

    profile: Dict[str, Any] = {
        "dtype": _guess_dtype(s),
        "is_numeric": pd.api.types.is_numeric_dtype(s) or (n_nn > 0 and numeric.notna().mean() > 0.8),
        "is_datetime": pd.api.types.is_datetime64_any_dtype(s),
        "is_binary": bool(non_null.nunique() == 2),
        "non_negative_ratio": float((num_nn >= 0).mean()) if len(num_nn) else 0.0,
        "zero_ratio": float((num_nn == 0).mean()) if len(num_nn) else 0.0,
        "unit_range_ratio": float(((num_nn >= 0) & (num_nn <= 1)).mean()) if len(num_nn) else 0.0,
        "constant_per_product": 0.0, "varies_over_time": 0.0, "same_across_products": 0.0,
    }

    if product_col is not None and product_col in df.columns:
        try:
            nunq = df.groupby(product_col)[col].nunique(dropna=True)
            if len(nunq):
                profile["constant_per_product"] = float((nunq <= 1).mean())
                profile["varies_over_time"] = float((nunq > 1).mean())
        except Exception:
            pass

    if date_col is not None and product_col is not None \
            and date_col in df.columns and product_col in df.columns:
        try:
            tmp = df[[date_col, product_col, col]].dropna()
            if tmp[product_col].nunique() >= 2 and tmp[date_col].nunique() >= 2:
                per_date = tmp.groupby(date_col)[col].nunique(dropna=True)
                profile["same_across_products"] = float((per_date <= 1).mean())
        except Exception:
            pass
    return profile


def _value_fit(profile: Dict[str, Any], hint: str) -> float:
    """Scores how well value patterns match a meaning's value_hint."""
    if hint == "any":
        return 0.5
    if not profile["is_numeric"] and hint in ("non_negative", "positive", "ratio_0_1"):
        return 0.0
    if hint == "non_negative":
        return profile["non_negative_ratio"]
    if hint == "positive":
        return profile["non_negative_ratio"] * (1.0 - profile["zero_ratio"])
    if hint == "ratio_0_1":
        return profile["unit_range_ratio"]
    if hint == "binary":
        return 1.0 if profile["is_binary"] else 0.0
    return 0.5


def _temporal_fit(profile: Dict[str, Any], hint: str) -> float:
    """Scores how well temporal behaviour matches a meaning's temporal_hint."""
    return {"any": 0.5,
            "static_per_product": profile["constant_per_product"],
            "time_varying": profile["varies_over_time"],
            "environmental": profile["same_across_products"]}.get(hint, 0.5)


def _meaning_confidence(name_s: float, struct_s: float) -> str:
    """'high' requires name AND structure agreement — safety rule (note [2])."""
    if (name_s >= 1.0 and struct_s >= 0.45) or (name_s >= 0.6 and struct_s >= 0.7):
        return "high"
    if name_s >= 0.6 or (name_s >= 0.3 and struct_s >= 0.6):
        return "medium"
    return "low"


def _capabilities_for_meaning(meaning: str) -> List[str]:
    """Capabilities a meaning contributes to (reads _CAPABILITY_REQUIREMENTS)."""
    return [cap for cap, req in _CAPABILITY_REQUIREMENTS.items()
            if meaning in req.get("any_of", []) or meaning in req.get("all_of", [])]


def discover_business_inputs(
    df: pd.DataFrame,
    core_mapping: Dict[str, Optional[str]],
) -> List[Dict[str, Any]]:
    """
    Infers the business meaning of every non-core column by voting
    (name 0.5 + structure 0.5) over the meaning registry.

    SAFETY: a pure structural match with zero name signal stays 'unknown' with a
    soft hint — missing a meaning is cheaper than confidently unlocking a wrong
    capability. Returns one rich dict per column (explanation strings are
    user-facing Arabic).
    """
    core_cols = {c for c in core_mapping.values() if c is not None}
    results: List[Dict[str, Any]] = []

    for col in df.columns:
        if col in core_cols:
            continue
        profile = _profile_structure(df, col, core_mapping)

        scored = []
        for meaning, spec in _BUSINESS_INPUT_REGISTRY.items():
            name_s, matched_kw = _token_match(str(col), spec["keywords"])
            struct_s = 0.5 * _value_fit(profile, spec["value_hint"]) \
                + 0.5 * _temporal_fit(profile, spec["temporal_hint"])
            scored.append((0.5 * name_s + 0.5 * struct_s, name_s, struct_s, meaning, matched_kw))
        scored.sort(reverse=True, key=lambda x: x[0])
        total, name_s, struct_s, meaning, matched_kw = scored[0]

        # No name signal or too weak overall → 'unknown' (note [2] safety rule).
        if name_s <= 0.0 or total < 0.30:
            hint = scored[0][3] if struct_s >= 0.6 else None
            explanation = (f"لم نتعرّف على معنى تجاري واضح لـ «{col}» من اسمه. "
                           + (f"شكل بياناته يُشبه «{_BUSINESS_INPUT_REGISTRY[hint]['label_ar']}» — "
                              "يُرجى التأكيد." if hint else "مُدرج كمعلومة استشارية."))
            results.append({"column": str(col), "detected_meaning": "unknown",
                            "label_ar": "غير محدّد", "confidence": "low",
                            "explanation": explanation, "guessed_dtype": profile["dtype"],
                            "possible_uses": [], "unlocks_capabilities": [],
                            "improves_forecast": False, "improves_inventory": False,
                            "supports_operations": False, "stockout_signal": False,
                            "needs_confirmation": True, "legacy_category": "advisory",
                            "alternatives": []})
            continue

        spec = _BUSINESS_INPUT_REGISTRY[meaning]
        confidence = _meaning_confidence(name_s, struct_s)
        caps = _capabilities_for_meaning(meaning)

        # User-facing explanation built from the signals that actually fired.
        why: List[str] = []
        if matched_kw:
            why.append(f"اسمه يطابق «{matched_kw}»")
        if profile["constant_per_product"] >= 0.7 and spec["temporal_hint"] == "static_per_product":
            why.append("قيمته ثابتة لكل منتج (سمة ثابتة لا طلب)")
        if profile["varies_over_time"] >= 0.5 and spec["temporal_hint"] == "time_varying":
            why.append("قيمته تتغيّر عبر الزمن")
        if profile["same_across_products"] >= 0.7 and spec["temporal_hint"] == "environmental":
            why.append("نفس القيمة لكل المنتجات في التاريخ ذاته (عامل بيئي)")
        if spec["value_hint"] == "ratio_0_1" and profile["unit_range_ratio"] >= 0.8:
            why.append("قيمه بين 0 و1 (نسبة)")
        if spec["value_hint"] == "binary" and profile["is_binary"]:
            why.append("قيمتان فقط (مؤشّر تشغيل/إيقاف)")
        explanation = (f"«{col}» ← {spec['label_ar']}: " + "؛ ".join(why) + "."
                       if why else f"«{col}» ← {spec['label_ar']}.")

        # Close runner-ups are surfaced so the user can arbitrate.
        alternatives = [{"meaning": m, "label_ar": _BUSINESS_INPUT_REGISTRY[m]["label_ar"],
                         "score": round(float(t), 3)}
                        for (t, ns, ss, m, mk) in scored[1:3] if t >= 0.30 and ns > 0]
        close_alt = bool(alternatives and (total - alternatives[0]["score"]) < 0.10)

        results.append({"column": str(col), "detected_meaning": meaning,
                        "label_ar": spec["label_ar"], "confidence": confidence,
                        "explanation": explanation, "guessed_dtype": profile["dtype"],
                        "possible_uses": spec["possible_uses"],
                        "unlocks_capabilities": caps,
                        "improves_forecast": bool(set(caps) & _FORECAST_CAPS),
                        "improves_inventory": bool(set(caps) & _INVENTORY_CAPS),
                        "supports_operations": bool(set(caps) & _OPERATIONS_CAPS),
                        "stockout_signal": bool(spec["stockout_signal"]),
                        "needs_confirmation": confidence == "low" or close_alt,
                        "legacy_category": spec["legacy_category"],
                        "alternatives": alternatives})
    return results


def detect_extra_columns(
    df: pd.DataFrame,
    core_mapping: Dict[str, Optional[str]],
    business_inputs: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """
    Legacy-schema adapter over discover_business_inputs (single source of truth).
    Pass business_inputs to avoid a redundant discovery pass.
    """
    discovered = business_inputs if business_inputs is not None \
        else discover_business_inputs(df, core_mapping)
    return [{"column": bi["column"], "guessed_dtype": bi["guessed_dtype"],
             "classification": bi["legacy_category"], "confidence": bi["confidence"],
             "reason": [bi["explanation"]]} for bi in discovered]


# ─── Dataset summary ─────────────────────────────────────────────────────────

def _infer_frequency(dates: pd.Series) -> str:
    """Infers daily/weekly/monthly/quarterly/yearly from the median date gap."""
    clean = pd.to_datetime(dates, errors="coerce").dropna().drop_duplicates().sort_values()
    if len(clean) < 2:
        return "unknown"
    median_days = clean.diff().dropna().dt.days.median()
    if median_days <= 2:
        return "daily"
    if median_days <= 10:
        return "weekly"
    if median_days <= 45:
        return "monthly"
    if median_days <= 135:
        return "quarterly"
    return "yearly"


def summarize_data(
    df: pd.DataFrame,
    core_mapping: Dict[str, Optional[str]],
    min_points_threshold: int = 12,
    business_inputs: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Display-ready dataset summary on the RAW frame (distinct from model.py's
    summarize_data which works on prepared series).

    Sections: overview (rows/products/span/frequency), per_product (point counts,
    sufficiency, top/bottom sellers), quality (gaps, zero ratio — a descriptive
    hint only; real quality judgment lives in quality.py), extra_columns.
    """
    date_col, product_col, qty_col = (core_mapping.get("date"),
                                      core_mapping.get("product"),
                                      core_mapping.get("qty"))
    summary: Dict[str, Any] = {"overview": {}, "per_product": {}, "quality": {},
                               "extra_columns": []}

    work = df.copy()  # typed working view; original untouched
    if date_col:
        work["_date"] = pd.to_datetime(work[date_col], errors="coerce")
    if qty_col:
        work["_qty"] = pd.to_numeric(work[qty_col], errors="coerce")

    overview: Dict[str, Any] = {"n_rows": int(len(df))}
    if product_col:
        overview["n_products"] = int(df[product_col].nunique())
    if date_col and work["_date"].notna().any():
        start, end = work["_date"].min(), work["_date"].max()
        overview["start_date"], overview["end_date"] = start, end
        overview["duration_months"] = int((end.year - start.year) * 12 + (end.month - start.month))
        overview["frequency"] = _infer_frequency(work["_date"])
    summary["overview"] = overview

    if product_col and date_col:
        points = work.groupby(product_col)["_date"].nunique()
        per_product: Dict[str, Any] = {
            "points_avg": float(points.mean()), "points_min": int(points.min()),
            "points_max": int(points.max()),
            "sufficient_count": int((points >= min_points_threshold).sum()),
            "sparse_count": int((points < min_points_threshold).sum()),
            "threshold": min_points_threshold,
        }
        if qty_col:
            totals = work.groupby(product_col)["_qty"].sum().sort_values(ascending=False)
            per_product["total_qty"] = float(totals.sum())
            per_product["avg_demand_per_product"] = float(totals.mean())
            per_product["top_products"] = [{"product": str(p), "total_qty": float(v)}
                                           for p, v in totals.head(5).items()]
            per_product["bottom_products"] = [{"product": str(p), "total_qty": float(v)}
                                              for p, v in totals.tail(5).items()]
        summary["per_product"] = per_product

    # Quality hints need qty too — "_qty" only exists when qty_col is mapped.
    if product_col and date_col and qty_col:
        pd_freq = {"daily": "D", "weekly": "W", "monthly": "MS",
                   "quarterly": "QS", "yearly": "YS"}.get(overview.get("frequency"))
        products_with_gaps, zero_ratios = 0, []
        if pd_freq:
            for _, g in work.dropna(subset=["_date"]).groupby(product_col):
                g_sum = g.set_index("_date")["_qty"].resample(pd_freq).sum()
                expected = len(pd.date_range(g_sum.index.min(), g_sum.index.max(), freq=pd_freq))
                observed = g.set_index("_date").resample(pd_freq).size()
                if (observed == 0).any() or expected > len(g_sum.dropna()):
                    products_with_gaps += 1
                if len(g_sum) > 0:
                    zero_ratios.append(float((g_sum == 0).mean()))
        summary["quality"] = {
            "products_with_gaps": int(products_with_gaps),
            "avg_zero_period_ratio": float(np.mean(zero_ratios)) if zero_ratios else 0.0,
            "note": "نسبة الفترات الصفرية مؤشر أوّلي على وجود منتجات متقطّعة",
        }

    summary["extra_columns"] = detect_extra_columns(df, core_mapping, business_inputs)

    # One-paragraph plain-language summary for non-technical users (Arabic UI).
    parts: List[str] = []
    if overview.get("n_products") is not None:
        parts.append(f"ملفك يحتوي {overview['n_products']} منتجاً")
    if "start_date" in overview:
        parts.append(f"يغطّي الفترة من {overview['start_date']:%Y-%m} إلى "
                     f"{overview['end_date']:%Y-%m} (حوالي {overview.get('duration_months', 0)} شهراً)")
    if summary.get("per_product"):
        parts.append(f"{summary['per_product'].get('sufficient_count', 0)} منتجاً ببيانات كافية "
                     f"للتوقّع و{summary['per_product'].get('sparse_count', 0)} ببيانات قليلة")
    summary["message_for_user"] = ("، ".join(parts) + "." if parts else
                                   "تعذّر تلخيص الملف — تأكّد من تحديد أعمدة التاريخ والمنتج والكمية.")
    return summary


# ─── Capabilities / readiness / stockout / overview ──────────────────────────

def discover_capabilities(
    business_inputs: List[Dict[str, Any]],
    core_mapping: Dict[str, Optional[str]],
) -> Dict[str, Dict[str, Any]]:
    """
    Determines which platform modules can operate given confidently-detected
    inputs (high/medium only — a weak guess must not unlock a capability).
    'Unlocked' means structurally possible, NOT guaranteed to perform well.
    """
    present = {bi["detected_meaning"] for bi in business_inputs
               if bi["confidence"] in ("high", "medium")
               and bi["detected_meaning"] not in (None, "unknown")}
    core_ok = all(core_mapping.get(r) is not None for r in ("date", "product", "qty"))

    capabilities: Dict[str, Dict[str, Any]] = {}
    for cap, req in _CAPABILITY_REQUIREMENTS.items():
        any_of, all_of = req.get("any_of", []), req.get("all_of", [])
        any_ok = (not any_of) or bool(present & set(any_of))
        all_ok = all(m in present for m in all_of)
        unlocked = core_ok if req.get("core") else bool(core_ok and any_ok and all_ok)

        available = sorted(present & (set(any_of) | set(all_of)))
        missing = sorted((set(any_of) | set(all_of)) - present) if not unlocked else []

        if req.get("core"):
            reason = ("الأعمدة الأساسية (تاريخ/منتج/كمية) متوفّرة." if unlocked
                      else "ينقص أحد الأعمدة الأساسية (تاريخ/منتج/كمية).")
        elif unlocked:
            labels = [_BUSINESS_INPUT_REGISTRY[m]["label_ar"] for m in available]
            reason = "مُتاحة لتوفّر: " + "، ".join(labels) + "."
        else:
            labels = [_BUSINESS_INPUT_REGISTRY[m]["label_ar"] for m in (set(any_of) | set(all_of))]
            reason = "تحتاج أحد هذه المدخلات: " + "، ".join(labels) + "."

        capabilities[cap] = {"label_ar": req["label_ar"], "unlocked": unlocked,
                             "domain": req["domain"], "required_any_of": any_of,
                             "required_all_of": all_of, "available_inputs": available,
                             "missing_inputs": missing, "reason": reason}
    return capabilities


def compute_readiness_scores(
    business_inputs: List[Dict[str, Any]],
    core_mapping: Dict[str, Optional[str]],
    summary: Dict[str, Any],
) -> Dict[str, Dict[str, Any]]:
    """
    PRESENCE-based readiness per platform area (0–100) with explicit reasons.
    These measure requirement completeness, NOT expected accuracy — and are
    distinct from quality.py's VALUE-quality readiness. Weights are stated
    engineering choices, not data-calibrated.
    """
    present = {bi["detected_meaning"] for bi in business_inputs
               if bi["confidence"] in ("high", "medium")
               and bi["detected_meaning"] not in (None, "unknown")}
    core_present = sum(core_mapping.get(r) is not None for r in ("date", "product", "qty"))

    # Forecast: core (50) + history sufficiency (30) + data regularity (20) + driver bonus.
    f_score, f_reasons = (core_present / 3) * 50, []
    f_reasons.append("الأعمدة الأساسية الثلاثة متوفّرة (+50)." if core_present == 3
                     else f"ينقص {3 - core_present} من الأعمدة الأساسية (الأساس ناقص).")
    per_prod = summary.get("per_product", {}) if summary else {}
    n_prod = summary.get("overview", {}).get("n_products") if summary else None
    if per_prod and n_prod:
        suff_ratio = per_prod.get("sufficient_count", 0) / max(n_prod, 1)
        f_score += suff_ratio * 30
        f_reasons.append(f"{per_prod.get('sufficient_count', 0)} من {n_prod} منتجاً "
                         f"بتاريخ كافٍ (+{suff_ratio * 30:.0f}).")
    else:
        f_reasons.append("تعذّر تقييم كفاية التاريخ (نقص في الملخّص).")
    cleanliness = max(0.0, 1.0 - (summary.get("quality", {}).get("avg_zero_period_ratio", 0.0)
                                  if summary else 0.0))
    f_score += cleanliness * 20
    f_reasons.append(f"انتظام البيانات (فترات غير صفرية) (+{cleanliness * 20:.0f}).")
    drivers = [m for m in ("price", "promotion", "discount", "temperature", "holiday")
               if m in present]
    if drivers:
        f_reasons.append("مكافأة: توفّر محرّكات طلب ("
                         + "، ".join(_BUSINESS_INPUT_REGISTRY[m]["label_ar"] for m in drivers)
                         + ") قد ترفع الدقّة.")
        f_score = min(100.0, f_score + 5)

    # Inventory: demand base (30) + stock (30) + lead time (25) + safety stock (15).
    inv_score, inv_reasons = (30.0 if core_present == 3 else 0.0), []
    inv_reasons.append("أساس الطلب متوفّر (+30)." if core_present == 3
                       else "أساس الطلب ناقص — توصيات المخزون تحتاجه.")
    for meaning, weight, label in (("stock_on_hand", 30, "المخزون الحالي"),
                                   ("lead_time", 25, "مهلة التوريد"),
                                   ("safety_stock", 15, "مخزون الأمان")):
        if meaning in present:
            inv_score += weight
            inv_reasons.append(f"{label} متوفّر (+{weight}).")
        else:
            inv_reasons.append(f"{label} غير متوفّر (−{weight}). إضافته تفتح/تقوّي توصيات المخزون.")

    # Scenario: inherits half of forecast readiness + price/promotion drivers.
    sc_score = min(f_score, 100) * 0.5
    sc_reasons = [f"يرث نصف جاهزية التوقّع (+{sc_score:.0f})."]
    for meaning, weight, label in (("price", 25, "السعر"), ("promotion", 25, "العروض الترويجية")):
        if meaning in present:
            sc_score += weight
            sc_reasons.append(f"{label} متوفّر — يتيح تغيير المتغيّرات في السيناريو (+{weight}).")
        else:
            sc_reasons.append(f"{label} غير متوفّر (−{weight}). بدونه تقلّ قوة المحاكاة.")

    return {
        "Forecast Readiness": {"label_ar": "جاهزية التوقّع",
                               "score": int(round(min(f_score, 100))), "reasons": f_reasons},
        "Inventory Readiness": {"label_ar": "جاهزية المخزون",
                                "score": int(round(min(inv_score, 100))), "reasons": inv_reasons},
        "Scenario Simulation Readiness": {"label_ar": "جاهزية محاكاة السيناريوهات",
                                          "score": int(round(min(sc_score, 100))),
                                          "reasons": sc_reasons},
    }


def detect_stockout_signals(
    df: pd.DataFrame,
    core_mapping: Dict[str, Optional[str]],
    business_inputs: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Identifies (does NOT execute) columns usable for future stockout detection.
    Pass business_inputs to avoid a redundant discovery pass.
    """
    discovered = business_inputs if business_inputs is not None \
        else discover_business_inputs(df, core_mapping)
    signals = [{"column": bi["column"], "detected_meaning": bi["detected_meaning"],
                "label_ar": bi["label_ar"], "confidence": bi["confidence"],
                "explanation": bi["explanation"]}
               for bi in discovered if bi.get("stockout_signal")]
    ready = any(s["confidence"] in ("high", "medium") for s in signals)
    if not signals:
        msg = "لم نعثر على أعمدة تدل على المخزون أو النفاد. كشف النفاد يحتاج عمود مخزون أو مبيعات مفقودة."
    elif ready:
        msg = ("وجدنا إشارات مخزون (" + "، ".join(s["label_ar"] for s in signals)
               + ") — جاهزة لتفعيل كشف نفاد المخزون لاحقاً.")
    else:
        msg = "وجدنا إشارات مخزون محتملة لكن بثقة منخفضة — يُفضّل تأكيدها."
    return {"signals": signals, "ready": ready, "message_for_user": msg}


def build_business_context_overview(
    business_inputs: List[Dict[str, Any]],
    capabilities: Dict[str, Dict[str, Any]],
    core_mapping: Dict[str, Optional[str]],
) -> Dict[str, Any]:
    """
    Executive snapshot shown BEFORE forecasting: detected inputs (incl. the
    implicit demand-history and product-catalog from core columns) + unlocked
    capabilities, with checkmark-ready display lines. Only low-confidence items
    land in needs_review — high confidence is not re-asked (UX rule).
    """
    has_demand = core_mapping.get("date") is not None and core_mapping.get("qty") is not None
    has_catalog = core_mapping.get("product") is not None
    detected_inputs = [
        {"key": "demand_history", "label_ar": "سجلّ الطلب", "present": has_demand,
         "columns": [core_mapping.get("date"), core_mapping.get("qty")],
         "confidence": "high" if has_demand else "low"},
        {"key": "product_catalog", "label_ar": "كتالوج المنتجات", "present": has_catalog,
         "columns": [core_mapping.get("product")],
         "confidence": "high" if has_catalog else "low"},
    ]
    for bi in business_inputs:
        if bi["detected_meaning"] == "unknown":
            continue
        detected_inputs.append({"key": bi["detected_meaning"], "label_ar": bi["label_ar"],
                                "present": True, "columns": [bi["column"]],
                                "confidence": bi["confidence"]})

    unlocked_capabilities = [{"capability": cap, "label_ar": info["label_ar"],
                              "unlocked": info["unlocked"], "reason": info["reason"]}
                             for cap, info in capabilities.items()]
    needs_review = [{"column": bi["column"], "label_ar": bi["label_ar"],
                     "alternatives": bi.get("alternatives", []), "explanation": bi["explanation"]}
                    for bi in business_inputs
                    if bi.get("needs_confirmation") and bi["detected_meaning"] != "unknown"]

    display_lines = ["المدخلات التجارية المكتشفة:"]
    display_lines += [f"  {'✓' if di['present'] else '○'} {di['label_ar']}" for di in detected_inputs]
    display_lines += ["", "القدرات المتاحة:"]
    display_lines += [f"  ✓ {uc['label_ar']}" for uc in unlocked_capabilities if uc["unlocked"]]

    n_inputs = sum(1 for di in detected_inputs if di["present"])
    n_caps = sum(1 for uc in unlocked_capabilities if uc["unlocked"])
    review_note = f" ({len(needs_review)} عمود يحتاج تأكيداً سريعاً)" if needs_review else ""
    message = (f"تعرّفنا على {n_inputs} نوعاً من المدخلات التجارية في ملفك، "
               f"وهي تفتح {n_caps} من قدرات المنصّة{review_note}. "
               "راجع النظرة العامة ثم ابدأ التوقّع.")

    return {"detected_inputs": detected_inputs,
            "unlocked_capabilities": unlocked_capabilities,
            "needs_review": needs_review, "display_lines": display_lines,
            "message_for_user": message}


# ─── Legacy orchestrator (kept for pipeline.py backward compatibility) ───────

def build_intake_result(
    df: pd.DataFrame,
    min_points_threshold: int = 12,
) -> Dict[str, Any]:
    """
    Legacy entry point: detection → business inputs → capabilities → summary →
    readiness → stockout signals → context overview, in one dict.

    NOTE: 'validation' and 'quality_report' keys were REMOVED — value
    validation/judgment is quality.py's job (boundary note [1]).
    ready_for_engine now simply means: all three core roles resolved.
    """
    detection = detect_columns(df)
    mapping = detection["mapping"]

    business_inputs = discover_business_inputs(df, mapping)
    capabilities = discover_capabilities(business_inputs, mapping)
    summary = summarize_data(df, mapping, min_points_threshold, business_inputs)
    readiness = compute_readiness_scores(business_inputs, mapping, summary)
    stockout_signals = detect_stockout_signals(df, mapping, business_inputs)
    business_context = build_business_context_overview(business_inputs, capabilities, mapping)

    return {
        "column_detection": detection,
        "proposed_mapping": mapping,
        "business_inputs": business_inputs,
        "extra_columns": detect_extra_columns(df, mapping, business_inputs),
        "capabilities": capabilities,
        "business_context": business_context,
        "readiness": readiness,
        "stockout_signals": stockout_signals,
        "summary": summary,
        "ready_for_engine": bool(all(mapping.get(r) is not None
                                     for r in ("date", "product", "qty"))),
    }


# ─── Smart mapping layer: profiling → rules → memory → (LLM) → contract ──────

def profile_columns(df: pd.DataFrame, n_samples: int = 5) -> Dict[str, Dict[str, Any]]:
    """
    Profiles every column from METADATA AND SAMPLES ONLY (≤ _SAMPLE_CAP rows):
    dtype, null rate, uniqueness, numeric/date parse success, sample values
    (distinct, truncated to 40 chars). This is the only LLM payload source —
    the full dataset is never sent anywhere.
    """
    profiles: Dict[str, Dict[str, Any]] = {}
    for col in df.columns:
        s = _sampled(df[col])
        non_null = s.dropna()
        n_nn = len(non_null)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            date_ok = float(pd.to_datetime(non_null, errors="coerce").notna().mean()) if n_nn else 0.0
        numeric_ok = float(pd.to_numeric(non_null, errors="coerce").notna().mean()) if n_nn else 0.0
        profiles[str(col)] = {
            "name": str(col),
            "inferred_dtype": _guess_dtype(s),
            "null_rate": round(float(df[col].isna().mean()), 4),
            "n_unique": int(non_null.nunique()),
            "unique_ratio": round(float(non_null.nunique() / n_nn), 4) if n_nn else 0.0,
            "numeric_parse_success": round(numeric_ok, 4),
            "date_parse_success": round(date_ok, 4),
            "sample_values": [str(v)[:40] for v in non_null.drop_duplicates().head(n_samples)],
        }
    return profiles


def _build_mapping_proposals(
    df: pd.DataFrame,
    legacy: Dict[str, Any],
    profiles: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    One UI-ready proposal per column from the legacy detection output:
      {column, role, bucket, confidence (0–1), source, reasons, sample_values,
       needs_review}.
    Core columns carry the detector's numeric score; meanings convert their
    confidence label via _CONF_TO_SCORE (stated simplification).
    """
    proposals: List[Dict[str, Any]] = []
    details = legacy["column_detection"]["details"]
    core_by_col = {v: k for k, v in legacy["proposed_mapping"].items() if v is not None}
    bi_by_col = {bi["column"]: bi for bi in legacy.get("business_inputs", [])}
    role_of_core = {"date": "date", "product": "product", "qty": "demand_quantity"}

    for orig in df.columns:
        col = str(orig)
        samples = profiles.get(col, {}).get("sample_values", [])

        if orig in core_by_col:  # required column (detail keys use original names)
            det = details.get(orig, {})
            conf = float(det.get("score", 0.5))
            proposals.append({"column": col, "role": role_of_core[core_by_col[orig]],
                              "bucket": "required", "confidence": round(conf, 3),
                              "source": "rule", "reasons": det.get("reason", []),
                              "sample_values": samples,
                              "needs_review": conf < _REVIEW_THRESHOLD})
            continue

        bi = bi_by_col.get(col)
        meaning = bi.get("detected_meaning") if bi else None
        if meaning and meaning != "unknown":
            conf = _CONF_TO_SCORE.get(bi.get("confidence", "low"), 0.35)
            proposals.append({"column": col, "role": meaning, "bucket": _bucket_of(meaning),
                              "confidence": round(conf, 3), "source": "rule",
                              "reasons": [bi.get("explanation", "")], "sample_values": samples,
                              "needs_review": bool(bi.get("needs_confirmation"))
                              or conf < _REVIEW_THRESHOLD})
        else:
            proposals.append({"column": col, "role": "unknown", "bucket": None,
                              "confidence": 0.2, "source": "rule",
                              "reasons": [bi.get("explanation", "لا تطابق") if bi else "عمود غير معروف"],
                              "sample_values": samples, "needs_review": True})
    return proposals


def _apply_memory_to_proposals(
    proposals: List[Dict[str, Any]],
    memory_lookup: Dict[str, Any],
    warnings_list: List[str],
) -> None:
    """
    Overlays confirmed-mapping memory onto proposals (in place).
    Full fingerprint hit → stored role at 0.95; single-column hit → 0.90
    (a past user confirmation outranks heuristics). A contradiction with the
    rule proposal keeps needs_review=True and logs a warning — never silent.
    """
    full = memory_lookup.get("full_match") or {}
    hits = memory_lookup.get("column_hits") or {}
    for p in proposals:
        stored = full.get(p["column"]) or hits.get(_normalize(p["column"]), {}).get("role")
        if not stored:
            continue
        from_full = p["column"] in full
        contradicts = p["role"] not in ("unknown", stored)
        p.update({
            "role": stored,
            "bucket": "required" if stored in _REQUIRED_ROLES else _bucket_of(stored),
            "confidence": 0.95 if from_full else 0.90,
            "source": "memory",
            "reasons": (["خريطة محفوظة من تأكيد سابق لنفس المخطّط."] if from_full
                        else ["دور مؤكَّد سابقاً لهذا العمود."]),
            "needs_review": contradicts,
        })
        if contradicts:
            warnings_list.append(
                f"العمود «{p['column']}»: الذاكرة ({stored}) تخالف القاعدة — يُستحسن المراجعة.")


def _assemble_contract(
    proposals: List[Dict[str, Any]],
    warnings_list: List[str],
) -> Dict[str, Any]:
    """
    Derives the five-bucket contract from proposals: required_mappings /
    optional_forecast_features / optional_inventory_features / ignored_columns /
    needs_review (+ warnings). Pure reshaping — no new detection.
    """
    required: Dict[str, Optional[str]] = {r: None for r in _REQUIRED_ROLES}
    forecast, inventory, ignored, review = [], [], [], []

    for p in proposals:
        entry = {"column": p["column"], "role": p["role"],
                 "confidence": p["confidence"], "source": p["source"]}
        if p["role"] in _REQUIRED_ROLES:
            required[p["role"]] = p["column"]
        elif p["bucket"] == "forecast":
            forecast.append(entry)
        elif p["bucket"] == "inventory":
            inventory.append(entry)
        elif p["role"] in ("unknown", "ignore"):
            ignored.append({"column": p["column"],
                            "reason": ("تجاهل بقرار المستخدم" if p["role"] == "ignore"
                                       else "لا معنى مكتشف بثقة")})
        if p["needs_review"]:
            review.append({"column": p["column"], "proposed_role": p["role"],
                           "confidence": p["confidence"],
                           "why": "; ".join(map(str, p["reasons"]))[:160]})

    for role, col in required.items():
        if col is None:
            warnings_list.append(f"الدور الإلزامي '{role}' بلا عمود — يجب تحديده يدوياً.")

    return {"required_mappings": required,
            "optional_forecast_features": forecast,
            "optional_inventory_features": inventory,
            "ignored_columns": ignored,
            "needs_review": review,
            "warnings": warnings_list}


def run_intake(
    df: pd.DataFrame,
    llm_client: Optional[Callable[[Dict[str, Any]], str]] = None,
    memory_path: Optional[str] = None,
    min_points_threshold: int = 12,
) -> Dict[str, Any]:
    """
    NEW main entry point — understanding only, no cleaning.

    Flow: guards → build_intake_result (full legacy payload) → profile_columns
    → memory overlay → optional LLM for ambiguous columns → five-bucket contract.

    llm_client : callable(payload dict) → JSON str. None = LLM disabled; the
                 whole flow works without it (ambiguous stays in needs_review).
    memory_path: mapping_memory storage path (None = default file).

    Returns ALL legacy keys PLUS: column_profiles, mapping_proposals,
    required_mappings, optional_forecast_features, optional_inventory_features,
    ignored_columns, needs_review, warnings, confirmed_mapping (None until
    the user confirms via confirm_mapping).
    """
    warnings_list: List[str] = []

    # Guards against degenerate inputs that previously crashed obscurely.
    if df is None or len(df.columns) == 0:
        raise ValueError("DataFrame فارغ أو بلا أعمدة — لا شيء نستقبله.")
    dup_cols = list(pd.Index(df.columns)[pd.Index(df.columns).duplicated()])
    if dup_cols:
        warnings_list.append(f"أسماء أعمدة مكرّرة: {dup_cols} — قد تُربك الكشف. وحّدها.")
    if len(df) == 0:
        warnings_list.append("الملف بلا صفوف — الكشف بالاسم فقط (لا إشارات قيم).")

    # 1) Legacy path: full backward-compat payload (token matching + sampling
    #    improvements apply automatically since it shares the same internals).
    legacy = build_intake_result(df, min_points_threshold)

    # 2) Profiles + rule-based proposals.
    profiles = profile_columns(df)
    proposals = _build_mapping_proposals(df, legacy, profiles)

    # 3) Memory BEFORE LLM (note [5]).
    memory = mapping_memory.load_memory(memory_path)
    lookup = mapping_memory.lookup([str(c) for c in df.columns], memory)
    _apply_memory_to_proposals(proposals, lookup, warnings_list)

    # 4) LLM only for still-ambiguous columns — suggests, never decides (note [6]).
    if llm_client is not None:
        import llm_mapper
        uncertain = [p for p in proposals if p["needs_review"] or p["role"] == "unknown"]
        if uncertain:
            payload = llm_mapper.build_llm_payload(profiles, proposals, uncertain, n_rows=len(df))
            suggestions = llm_mapper.suggest_mappings(payload, llm_client)
            llm_mapper.merge_suggestions(proposals, suggestions, warnings_list)

    # 5) Cross-check: the chosen qty column might be monetary (e.g. "Order Value").
    qty_col = legacy["proposed_mapping"].get("qty")
    if qty_col is not None:
        _, kw = _token_match(str(qty_col), _CORE_SYNONYMS["qty"])
        money_score, _ = _token_match(str(qty_col),
                                      _BUSINESS_INPUT_REGISTRY["revenue"]["keywords"]
                                      + _BUSINESS_INPUT_REGISTRY["price"]["keywords"])
        generic_only = kw is not None and all(t in _GENERIC_TOKENS for t in _tokens_of(kw))
        if generic_only or money_score >= 0.85:
            warnings_list.append(
                f"عمود الكمية المقترح «{qty_col}» قد يكون قيمة مالية لا كمية طلب — أكّده.")
            for p in proposals:
                if p["column"] == str(qty_col):
                    p["needs_review"] = True

    # 6) Keep the legacy proposed_mapping in sync with memory-sourced required
    #    roles, so old and new contracts never diverge.
    synced_mapping = dict(legacy["proposed_mapping"])
    for p in proposals:
        if p["source"] == "memory" and p["role"] in _REQUIRED_ROLES:
            synced_mapping[_REQUIRED_ROLES[p["role"]]] = p["column"]
    legacy = dict(legacy)
    legacy["proposed_mapping"] = synced_mapping

    result = dict(legacy)  # all legacy keys preserved as-is
    result.update({
        "column_profiles": profiles,
        "mapping_proposals": proposals,
        **_assemble_contract(proposals, warnings_list),
        "confirmed_mapping": None,
    })
    return result


def apply_user_corrections(
    intake_result: Dict[str, Any],
    corrections: Dict[str, str],
) -> Dict[str, Any]:
    """
    Applies user mapping corrections (backend function — a future UI calls it).

    corrections: {column → role}, role ∈ required roles ('date','product',
    'demand_quantity') | registry meanings | 'ignore'. Corrections get
    source='user', confidence=1.0, needs_review=False — the user outranks all.

    Syncs the legacy keys too (proposed_mapping for core roles, business_inputs
    for meanings) so quality.py / feature_engineering.py see the corrected view.
    Returns a NEW dict; persisting to memory happens only in confirm_mapping.
    """
    valid_roles = set(_REQUIRED_ROLES) | set(_BUSINESS_INPUT_REGISTRY) | {"ignore"}
    result = dict(intake_result)
    proposals = [dict(p) for p in result.get("mapping_proposals", [])]
    mapping = dict(result.get("proposed_mapping", {}))
    business_inputs = [dict(b) for b in result.get("business_inputs", [])]
    warnings_list = list(result.get("warnings", []))
    known_cols = {p["column"] for p in proposals}

    for col, role in corrections.items():
        col = str(col)
        if col not in known_cols:
            warnings_list.append(f"تصحيح متجاهَل: العمود «{col}» غير موجود.")
            continue
        if role not in valid_roles:
            warnings_list.append(f"تصحيح متجاهَل: الدور «{role}» غير معروف للعمود «{col}».")
            continue

        for p in proposals:
            if p["column"] == col:
                p.update({"role": role, "source": "user", "confidence": 1.0,
                          "needs_review": False,
                          "bucket": "required" if role in _REQUIRED_ROLES else _bucket_of(role),
                          "reasons": ["تصحيح المستخدم."]})

        if role in _REQUIRED_ROLES:
            # Sync legacy proposed_mapping; the displaced column loses its role.
            core_key = _REQUIRED_ROLES[role]
            old = mapping.get(core_key)
            if old and str(old) != col:
                for p in proposals:
                    if p["column"] == str(old):
                        p.update({"role": "unknown", "bucket": None, "confidence": 0.2,
                                  "needs_review": True,
                                  "reasons": ["أُزيح عن الدور بتصحيح المستخدم."]})
            mapping[core_key] = col
        elif role != "ignore":
            # Sync legacy business_inputs so downstream layers see the meaning.
            spec = _BUSINESS_INPUT_REGISTRY[role]
            entry_update = {"detected_meaning": role, "label_ar": spec["label_ar"],
                            "confidence": "high", "needs_confirmation": False,
                            "legacy_category": spec["legacy_category"],
                            "stockout_signal": spec["stockout_signal"],
                            "explanation": "تصحيح المستخدم."}
            found = False
            for b in business_inputs:
                if b["column"] == col:
                    b.update(entry_update)
                    found = True
            if not found:
                business_inputs.append({
                    "column": col, **entry_update, "guessed_dtype": "user",
                    "possible_uses": spec["possible_uses"],
                    "unlocks_capabilities": _capabilities_for_meaning(role),
                    "improves_forecast": False, "improves_inventory": False,
                    "supports_operations": False, "alternatives": []})

    result["mapping_proposals"] = proposals
    result["proposed_mapping"] = mapping
    result["business_inputs"] = business_inputs
    result.update(_assemble_contract(proposals, warnings_list))
    return result


def confirm_mapping(
    intake_result: Dict[str, Any],
    memory_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Finalizes the mapping: builds confirmed_mapping {column → role}, saves it to
    mapping memory (fingerprint + per-column) for future reuse, and returns the
    updated result. Fails loudly if any required role is still unmapped.
    Cleaning/quality comes AFTER this gate — that is the pipeline's job.
    """
    missing = [r for r, c in intake_result.get("required_mappings", {}).items() if c is None]
    if missing:
        raise ValueError(f"لا يمكن التأكيد — أدوار إلزامية ناقصة: {missing}")

    confirmed = {p["column"]: p["role"]
                 for p in intake_result.get("mapping_proposals", [])
                 if p["role"] != "unknown"}
    mapping_memory.save_confirmed(confirmed, memory_path)

    result = dict(intake_result)
    result["confirmed_mapping"] = confirmed
    return result
