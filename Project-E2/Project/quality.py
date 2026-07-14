"""
quality.py — Data Quality & Remediation Engine | Bayyina Platform

Position: after intake (column understanding) and before feature engineering.
Role: detect potential data issues, classify their BUSINESS impact, recommend
ONE preferred action per issue, and execute only what the user approves.

Philosophy — this is business data, not a textbook dataset:
  - Multiple records per product-period are usually transactions, not duplicates.
  - Exact duplicate rows may be valid when no transaction id exists.
  - Demand spikes may be real events (promotions, projects, recoveries).
  - Missing periods may be expected in some businesses.
  - Negative demand may be returns or corrections.
The goal is to understand risk and recommend, NOT to aggressively clean.

Public API (consumed by pipeline.py / app.py — contract preserved):
  run_quality_engine(df, intake_result, ...)
  execute_plan(df, plan, approved_actions, core_mapping, freq)
  execution_log_to_dataframe(execution_log)
"""

# ===========================================================================
# DEVELOPER NOTES
# ===========================================================================
#
# [1] Boundary with intake.py (permanent): intake evaluates input PRESENCE
#     (is there a stock column?); quality evaluates actual VALUES (are the
#     stock values plausible?). Neither re-does the other's job.
#
# [2] No industry assumptions. The previous version hardcoded HVAC peak months
#     and a 2-16 week lead-time range; both are GONE. Seasonal zeros are now
#     detected from each product's own history (a calendar month that is zero
#     across observed years is an expected dead season, not an error), and
#     lead-time outliers are relative to the column's own distribution (MAD).
#     Same protection, zero sector assumptions.
#
# [3] Three severities only: critical (blocks modeling), warning (degrades
#     results), info (worth knowing / modeling-routing fact). Intermittent
#     demand is INFO, not an error — it routes model selection, nothing more.
#
# [4] Issues carry confidence (0-1). Detection is heuristic on business data;
#     pretending certainty would be dishonest. Low confidence → the preferred
#     action downgrades to "review".
#
# [5] Duplicates are INVESTIGATED, never auto-judged:
#       exact duplicates + a transaction-id-like column duplicated too → likely
#       ETL artifact (higher confidence); no id column → could be two identical
#       real transactions (lower confidence, review).
#       product-period multiplicity with DIFFERING quantities → almost surely
#       transactions → recommend aggregation (info). Identical quantities →
#       ambiguous → review.
#     Nothing about duplicates is auto-applied anymore.
#
# [6] Outliers are advisory-only: detect, estimate impact, explain plausible
#     business causes. The cap/winsorize executor was REMOVED — no path in this
#     engine modifies a demand value automatically.
#
# [7] One preferred action per issue (recommended_action) with a rationale —
#     never a menu of five fixes. Action policies:
#       auto     → purely technical, touches no business value
#                  (trim whitespace, drop fully-empty rows, standardize dates)
#       review   → changes data, needs explicit approval
#       never    → business-record deletion; engine refuses automation
#                  (manual override still possible, loudly logged)
#       advisory → no automated fix exists; human judgment needed
#
# [8] Representation checks (new): product/segment coverage imbalance, seasonal
#     coverage gaps, stale products. Representativeness, not demographics.
#
# [9] Output-contract changes vs the previous version (verified unconsumed by
#     pipeline.py / app.py): removed keys 'business_impacts' (impact now lives
#     inside each issue), 'quality_assessment' (replaced by issues_by_category),
#     'impact_estimation' (folded into plan items), 'zero_demand_classification'.
#
# Known limitations:
#   - Seasonal-zero detection needs ≥2 observed years; with less, zeros are
#     "unclassified" and intermittency is reported at low confidence.
#   - Lead-time MAD outliers assume one consistent unit per column; mixed units
#     inside one column cannot be detected without metadata.
#
# ===========================================================================

from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ─── Constants ───────────────────────────────────────────────────────────────

_FREQ_MAP = {"daily": "D", "weekly": "W", "monthly": "MS",
             "quarterly": "QS", "yearly": "YS"}

_SEVERITY_RANK = {"critical": 0, "warning": 1, "info": 2}
_SEVERITY_PENALTY = {"critical": 40, "warning": 15, "info": 4}

# Action catalog: policy + user-facing label + expected benefit (Arabic = UI data).
_ACTIONS: Dict[str, Dict[str, str]] = {
    # auto — purely technical, no business values touched
    "trim_whitespace": dict(policy="auto", label="Trim extra whitespace",
                            benefit="Normalizes text identifiers and prevents one product splitting into several."),
    "remove_empty_rows": dict(policy="auto", label="Remove fully empty rows",
                              benefit="Rows carrying no information — removing them touches no business data."),
    "standardize_date_format": dict(policy="auto", label="Standardize date format",
                                    benefit="Ensures every valid date is read consistently."),
    # review — changes data, explicit approval required
    "remove_exact_duplicates": dict(policy="review", label="Remove exact duplicate rows",
                                    benefit="Removes likely ETL/transfer duplication — after your confirmation."),
    "aggregate_duplicate_product_date": dict(policy="review",
                                             label="Aggregate multiple (product, period) records",
                                             benefit="Combines multiple transactions into one demand signal per period."),
    "insert_missing_periods": dict(policy="review", label="Insert missing time periods (zero demand)",
                                   benefit="Makes the series continuous if absence means no sales."),
    "fill_missing_demand": dict(policy="review", label="Fill missing demand values with zero",
                                benefit="Closes series gaps (assumption: no record = no demand)."),
    "remove_invalid_dates": dict(policy="review", label="Exclude rows with invalid dates",
                                 benefit="Rows that cannot be placed on the time axis."),
    "remove_invalid_demand": dict(policy="review", label="Exclude rows with non-numeric quantity",
                                  benefit="Quantities that cannot be read as numbers enter no calculation."),
    "standardize_product_identifiers": dict(policy="review", label="Standardize product identifiers",
                                            benefit="Merges variant spellings of one product to consolidate its history."),
    # never — business-record deletion; automation refused
    "remove_negative_demand": dict(policy="never", label="Remove negative demand",
                                   benefit="May be returns/corrections — a business decision, never automatic."),
    # advisory — no automated fix; human judgment
    "review_duplicates": dict(policy="advisory", label="Review duplicates",
                              benefit="Not enough evidence to distinguish real transactions from transfer duplication."),
    "review_outliers": dict(policy="advisory", label="Review unusual values",
                            benefit="May be promotions, large projects, or real peaks — no automatic modification."),
    "collect_more_history": dict(policy="advisory", label="Provide longer history",
                                 benefit="Longer history raises forecast reliability."),
    "review_intermittent": dict(policy="advisory", label="Review intermittent demand pattern",
                                benefit="Guidance toward a suitable model (Croston) — no data modification."),
    "review_inventory_values": dict(policy="advisory", label="Check inventory values",
                                    benefit="Implausible values weaken inventory recommendations."),
    "review_lead_times": dict(policy="advisory", label="Check lead times",
                              benefit="Relatively anomalous lead times need supplier confirmation."),
    "review_coverage": dict(policy="advisory", label="Review data representation",
                            benefit="Unbalanced coverage makes forecasts weaker for under-represented segments."),
    "review_negative_demand": dict(policy="advisory", label="Review negative demand",
                                   benefit="Decide: returns to count, or errors to exclude."),
}

_POLICY_BUCKET = {"auto": "auto_actions", "review": "review_required",
                  "never": "skipped_actions", "advisory": "skipped_actions"}


# ─── Shared helpers ──────────────────────────────────────────────────────────

def _resolve_freq(intake_result: Optional[Dict[str, Any]], target_freq: Optional[str]) -> str:
    """Explicit target_freq → intake's detected frequency word → 'MS' default."""
    if target_freq:
        return target_freq
    if intake_result:
        word = intake_result.get("summary", {}).get("overview", {}).get("frequency")
        if word in _FREQ_MAP:
            return _FREQ_MAP[word]
    return "MS"


def _meaning_columns(business_inputs: Optional[List[Dict[str, Any]]]) -> Dict[str, str]:
    """{meaning → column} for confident intake detections. Consumes, never re-detects."""
    out: Dict[str, str] = {}
    for bi in (business_inputs or []):
        meaning = bi.get("detected_meaning")
        if meaning and meaning != "unknown" and bi.get("confidence") in ("high", "medium"):
            out.setdefault(meaning, bi["column"])
    return out


def _series_map(df: pd.DataFrame, m: Dict[str, Optional[str]], freq: str) -> Dict[str, pd.Series]:
    """
    READ-ONLY assessment view: per-product series resampled to freq (sum),
    zero-filled between first and last observation. Never returned as cleaned data.
    """
    date_col, product_col, qty_col = m.get("date"), m.get("product"), m.get("qty")
    if not (date_col and product_col and qty_col):
        return {}
    work = df[[date_col, product_col, qty_col]].copy()
    work[date_col] = pd.to_datetime(work[date_col], errors="coerce")
    work[qty_col] = pd.to_numeric(work[qty_col], errors="coerce")
    work = work.dropna(subset=[date_col])

    out: Dict[str, pd.Series] = {}
    for product, g in work.groupby(product_col):
        s = g.set_index(date_col)[qty_col].resample(freq).sum()
        if len(s) == 0:
            continue
        full = pd.date_range(s.index.min(), s.index.max(), freq=freq)
        out[str(product)] = s.reindex(full, fill_value=0)
    return out


def _zero_classification(series: pd.Series) -> Dict[str, Any]:
    """
    Data-driven seasonal vs structural zeros — NO sector assumptions (note [2]).

    A zero in calendar month M is SEASONAL if M is all-zero in ≥70% of the
    observed years (a recurring dead season for THIS product). Other zeros are
    STRUCTURAL (sporadic, suspicious). With <2 observed years the pattern cannot
    be established: zeros are 'unclassified' and confidence is low.
    """
    values, months, years = series.values, series.index.month, series.index.year
    n = len(series)
    zero_mask = values == 0
    n_zeros = int(zero_mask.sum())
    n_years = len(np.unique(years))

    if n_zeros == 0:
        return dict(n_zeros=0, seasonal=0, structural=0, unclassified=0,
                    structural_ratio=0.0, confidence=1.0)
    if n_years < 2:
        return dict(n_zeros=n_zeros, seasonal=0, structural=0, unclassified=n_zeros,
                    structural_ratio=0.0, confidence=0.3)

    # Fraction of observed years in which each calendar month is entirely zero.
    month_zero_frac: Dict[int, float] = {}
    for mth in np.unique(months):
        msk = months == mth
        yrs = np.unique(years[msk])
        zero_years = sum(1 for y in yrs if (values[msk & (years == y)] == 0).all())
        month_zero_frac[mth] = zero_years / len(yrs)

    seasonal = sum(1 for i in range(n)
                   if zero_mask[i] and month_zero_frac.get(months[i], 0) >= 0.7)
    structural = n_zeros - seasonal
    return dict(n_zeros=n_zeros, seasonal=seasonal, structural=structural,
                unclassified=0, structural_ratio=structural / n,
                confidence=min(1.0, 0.5 + 0.25 * n_years))


def _mad_outliers(values: pd.Series, k: float = 5.0) -> pd.Series:
    """Boolean mask of robust outliers: |x − median| > k·MAD. Empty-safe."""
    v = values.dropna()
    if len(v) < 8:
        return pd.Series(False, index=values.index)
    med = v.median()
    mad = (v - med).abs().median()
    if mad == 0:
        return pd.Series(False, index=values.index)
    return (values - med).abs() > k * mad


def _issue(issue_type: str, category: str, severity: str, confidence: float,
           description: str, affected_records: int, affected_products: List[str],
           forecast_impact: str, inventory_impact: str,
           recommended_action: str, rationale: str) -> Dict[str, Any]:
    """Standard issue object — one preferred action + rationale, never a menu."""
    return {
        "issue_type": issue_type, "category": category, "severity": severity,
        "confidence": round(float(confidence), 2), "description": description,
        "affected_records": int(affected_records),
        "affected_products": list(affected_products)[:20],
        "forecast_impact": forecast_impact, "inventory_impact": inventory_impact,
        "recommended_action": recommended_action, "rationale": rationale,
    }


# ─── 1. Structural validation ────────────────────────────────────────────────

def validate_structural(df: pd.DataFrame, m: Dict[str, Optional[str]]) -> List[Dict[str, Any]]:
    """
    Structural checks: missing required columns, invalid dates, invalid demand
    values, invalid product identifiers, fully-empty rows, exact duplicate rows.
    Exact duplicates are INVESTIGATED, not auto-judged (note [5]).
    """
    issues: List[Dict[str, Any]] = []
    date_col, product_col, qty_col = m.get("date"), m.get("product"), m.get("qty")

    # Missing required columns → critical (nothing downstream can run).
    for role, col, label in (("date", date_col, "التاريخ"), ("product", product_col, "المنتج"),
                             ("qty", qty_col, "الكمية")):
        if col is None:
            issues.append(_issue(
                f"missing_{role}_column", "structural", "critical", 1.0,
                f"لا يوجد عمود {label} — مطلوب لأي توقّع.", 0, [],
                "يمنع بناء السلسلة الزمنية كلياً.", "يمنع أي توصية مخزون.",
                "review_coverage", "عمود إلزامي غائب؛ يجب تحديده في طبقة الاستقبال."))
    if issues:
        return issues  # without core columns the remaining checks are meaningless

    parsed_dates = pd.to_datetime(df[date_col], errors="coerce")
    numeric_qty = pd.to_numeric(df[qty_col], errors="coerce")

    # Invalid dates — critical only if NOTHING parses.
    bad_dates = int(parsed_dates.isna().sum() - df[date_col].isna().sum())
    if bad_dates > 0:
        all_bad = parsed_dates.notna().sum() == 0
        issues.append(_issue(
            "invalid_dates", "structural", "critical" if all_bad else "warning", 0.95,
            f"{bad_dates} قيمة تاريخ غير قابلة للقراءة.", bad_dates, [],
            "صفوف بلا موضع زمني تسقط من السلسلة.", "لا أثر مباشر.",
            "remove_invalid_dates", "تاريخ غير صالح لا يمكن تفسيره تجارياً — الاستبعاد بعد موافقتك."))

    # Invalid/missing demand values (merged: unparseable + missing → same remedy).
    bad_qty = int(numeric_qty.isna().sum())
    if bad_qty > 0:
        all_bad = numeric_qty.notna().sum() == 0
        issues.append(_issue(
            "invalid_demand_values", "structural", "critical" if all_bad else "warning", 0.9,
            f"{bad_qty} قيمة كمية مفقودة أو غير رقمية.", bad_qty, [],
            "إشارات طلب ضائعة تُحيّز المتوسطات.", "كميات ناقصة تُضعف حساب المخزون.",
            "remove_invalid_demand", "كمية لا تُقرأ رقمياً لا تدخل أي حساب — استبعاد أو تصحيح بالمصدر."))

    # Invalid product identifiers: null/empty product values.
    bad_prod = int(df[product_col].isna().sum()
                   + (df[product_col].astype("string").str.strip() == "").sum())
    if bad_prod > 0:
        issues.append(_issue(
            "invalid_product_identifiers", "structural", "warning", 0.9,
            f"{bad_prod} صفاً بلا معرّف منتج.", bad_prod, [],
            "صفوف لا تُنسب لأي منتج تسقط من التجميع.", "لا تُحتسب في مخزون أي منتج.",
            "review_coverage", "سجلّ بلا منتج غامض تجارياً — راجع مصدره قبل أي حذف."))

    # Fully-empty rows: carry no information at all → safe-auto removal.
    empty = int(df.isna().all(axis=1).sum())
    if empty > 0:
        issues.append(_issue(
            "empty_rows", "structural", "info", 1.0,
            f"{empty} صف فارغ كلياً.", empty, [],
            "ضوضاء بلا إشارة.", "لا أثر.",
            "remove_empty_rows", "صف بلا أي قيمة ليس سجلّ عمل — حذفه آمن تقنياً."))

    # Exact duplicate rows — investigate before judging (note [5]).
    dup_mask = df.duplicated()
    n_dups = int(dup_mask.sum())
    if n_dups > 0:
        issues.append(_investigate_exact_duplicates(df, m, n_dups))

    return issues


def _investigate_exact_duplicates(df: pd.DataFrame, m: Dict[str, Optional[str]],
                                  n_dups: int) -> Dict[str, Any]:
    """
    Classifies exact-duplicate rows instead of assuming they are errors.

    Evidence considered: does a transaction-id-like column exist (near-unique
    non-core column)? If yes and it is ALSO duplicated → strong ETL-duplicate
    signal. No id column → two identical real transactions are plausible →
    review with lower confidence. High duplicate share → suspicious either way.
    """
    core_cols = {c for c in m.values() if c is not None}
    dup_share = n_dups / max(len(df), 1)

    # A near-unique non-core column behaves like a transaction identifier.
    id_like = None
    for col in df.columns:
        if col in core_cols:
            continue
        nn = df[col].dropna()
        if len(nn) and nn.nunique() / len(nn) > 0.95:
            id_like = col
            break

    if id_like is not None:
        # Identifier duplicated together with the row → not two transactions.
        verdict = ("likely_etl_duplicate", 0.85, "warning", "remove_exact_duplicates",
                   f"يوجد عمود يشبه معرّف معاملة («{id_like}») وهو مكرّر أيضاً — "
                   "تكرار النقل (ETL) هو التفسير الأرجح.")
    elif dup_share < 0.02:
        verdict = ("possible_etl_duplicate", 0.6, "info", "remove_exact_duplicates",
                   "نسبة التكرار ضئيلة وبلا معرّف معاملة — قد يكون تكرار نقل، "
                   "وقد يكون معاملتين متطابقتين حقيقيتين. القرار لك.")
    else:
        verdict = ("requires_review", 0.4, "warning", "review_duplicates",
                   "نسبة تكرار ملحوظة بلا معرّف معاملة — لا دليل كافياً للحسم؛ "
                   "راجع مصدر البيانات.")

    kind, conf, sev, action, rationale = verdict
    return _issue(
        "exact_duplicate_rows", "structural", sev, conf,
        f"{n_dups} صفاً مطابقاً تماماً لصفوف أخرى ({kind}).", n_dups, [],
        "تكرار حقيقي يضخّم الطلب المسجّل؛ معاملات حقيقية يجب أن تبقى.",
        "تضخيم الطلب يضخّم توصيات الشراء.",
        action, rationale)


# ─── 2. Time-series validation ───────────────────────────────────────────────

def validate_time_series(
    df: pd.DataFrame, m: Dict[str, Optional[str]], freq: str,
    min_history: int = 12, intermittency_threshold: float = 0.30,
) -> List[Dict[str, Any]]:
    """
    Time-series checks: insufficient history, missing periods, product-period
    multiplicity (investigated — usually transactions), intermittent demand
    (on NON-seasonal zeros only), large temporal gaps, irregular frequency.
    """
    issues: List[Dict[str, Any]] = []
    date_col, product_col, qty_col = m.get("date"), m.get("product"), m.get("qty")
    if not (date_col and product_col and qty_col):
        return issues

    series = _series_map(df, m, freq)
    if not series:
        return issues

    # Insufficient history (merged sparse+short: same remedy, two severities).
    short = {p: len(s) for p, s in series.items() if len(s) < min_history}
    if short:
        very_short = [p for p, n in short.items() if n < max(4, min_history // 2)]
        issues.append(_issue(
            "insufficient_history", "time_series",
            "warning" if very_short else "info", 0.9,
            f"{len(short)} منتجاً بتاريخ أقصر من {min_history} فترة.",
            len(short), sorted(short),
            "تاريخ قصير يمنع تعلّم الموسمية ويخفض الموثوقية.",
            "توصيات مخزون مبنية على تاريخ قصير أقل أماناً.",
            "collect_more_history",
            "لا إصلاح آلي ممكن — التوقّع سيعمل بثقة أقل حتى يتوفّر تاريخ أطول."))

    # Missing periods — counted on original dates, before zero-fill.
    work = df[[date_col, product_col]].copy()
    work[date_col] = pd.to_datetime(work[date_col], errors="coerce")
    work = work.dropna(subset=[date_col])
    missing_total, missing_products = 0, []
    for product, g in work.groupby(product_col):
        observed = g.set_index(date_col).resample(freq).size()
        if len(observed) < 2:
            continue
        gaps = int((observed == 0).sum())
        if gaps > 0:
            missing_total += gaps
            missing_products.append(str(product))
    if missing_total > 0:
        issues.append(_issue(
            "missing_periods", "time_series", "warning", 0.7,
            f"{missing_total} فترة بلا أي سجلّ عبر {len(missing_products)} منتجاً.",
            missing_total, missing_products,
            "فترات غائبة تشوّه كشف الترند والموسمية.",
            "فجوات تاريخ الطلب تُضعف حساب نقطة إعادة الطلب.",
            "insert_missing_periods",
            "إن كان الغياب يعني «لا مبيعات» فالملء بصفر صحيح؛ "
            "وإن كان نقص تسجيل فالملء يُخفي مشكلة — لذلك يحتاج موافقتك."))

    # Product-period multiplicity — investigate (note [5]).
    mult = _investigate_product_period(df, m, freq)
    if mult is not None:
        issues.append(mult)

    # Intermittency on STRUCTURAL zeros only (seasonal zeros are expected).
    intermittent, low_conf = [], False
    for product, s in series.items():
        zc = _zero_classification(s)
        if zc["confidence"] < 0.5:
            low_conf = True
        ratio = (zc["structural_ratio"] if zc["unclassified"] == 0
                 else zc["n_zeros"] / max(len(s), 1))  # short history: raw ratio, low conf
        if ratio > intermittency_threshold and len(s) >= 6:
            intermittent.append(str(product))
    if intermittent:
        issues.append(_issue(
            "intermittent_demand", "time_series", "info",
            0.4 if low_conf else 0.8,
            f"{len(intermittent)} منتجاً بطلب متقطّع (أصفار غير موسمية متكرّرة).",
            len(intermittent), intermittent,
            "يوجّه اختيار الموديل نحو طرق الطلب المتقطّع (Croston).",
            "الطلب المتقطّع يحتاج سياسة مخزون مختلفة.",
            "review_intermittent",
            "ليس خطأ بيانات — نمط عمل حقيقي يؤثّر على اختيار الموديل فقط."))

    # Large continuous gaps of non-seasonal zeros.
    gap_products = []
    for product, s in series.items():
        zc = _zero_classification(s)
        if zc["structural"] >= 4 and _longest_zero_run(s) >= 4:
            gap_products.append(str(product))
    if gap_products:
        issues.append(_issue(
            "large_temporal_gaps", "time_series", "warning", 0.6,
            f"{len(gap_products)} منتجاً بانقطاع طويل متّصل غير موسمي.",
            len(gap_products), gap_products,
            "انقطاع طويل يقطع تعلّم الاستمرارية.",
            "قد يعني توقّف منتج — توصيات الشراء له مشكوك فيها.",
            "review_coverage",
            "قد يكون توقّف إنتاج حقيقياً أو نقص بيانات — يحتاج معرفتك بالعمل."))

    # Irregular recording frequency: median gap far from the target grain.
    dates_all = pd.to_datetime(df[date_col], errors="coerce").dropna().sort_values()
    if len(dates_all) >= 10:
        median_gap = dates_all.diff().dropna().dt.days.median()
        expected = {"D": 1, "W": 7, "MS": 30, "QS": 91, "YS": 365}.get(freq, 30)
        if median_gap > expected * 3:
            issues.append(_issue(
                "irregular_frequency", "time_series", "warning", 0.6,
                f"الفاصل الوسيط بين السجلّات (~{median_gap:.0f} يوماً) أكبر بكثير من "
                f"التدرّج المطلوب.", 0, [],
                "تجميع لتدرّج أدقّ من البيانات ينتج فترات صفرية وهمية.",
                "لا أثر مباشر.",
                "review_coverage", "قد يكون التدرّج المختار أدقّ مما تسجّله بياناتك فعلاً."))

    return issues


def _longest_zero_run(series: pd.Series) -> int:
    """Length of the longest consecutive run of zeros."""
    longest = run = 0
    for v in series.values:
        run = run + 1 if v == 0 else 0
        longest = max(longest, run)
    return longest


def _investigate_product_period(df: pd.DataFrame, m: Dict[str, Optional[str]],
                                freq: str) -> Optional[Dict[str, Any]]:
    """
    Multiple records in the same product-period: transactions or duplication?

    Differing quantities inside groups → almost surely multiple transactions →
    recommend aggregation (info, high confidence). Mostly-identical rows inside
    groups → ambiguous → review (could be duplication OR identical transactions).
    """
    date_col, product_col, qty_col = m["date"], m["product"], m["qty"]
    work = df[[date_col, product_col, qty_col]].copy()
    work[date_col] = pd.to_datetime(work[date_col], errors="coerce")
    work[qty_col] = pd.to_numeric(work[qty_col], errors="coerce")
    work = work.dropna(subset=[date_col])
    work["_period"] = work[date_col].dt.to_period({"D": "D", "W": "W", "MS": "M",
                                                   "QS": "Q", "YS": "Y"}.get(freq, "M"))

    grp = work.groupby([product_col, "_period"])
    sizes = grp.size()
    multi = sizes[sizes > 1]
    if multi.empty:
        return None

    extra_records = int((multi - 1).sum())
    # Within multi-row groups: do quantities differ (transaction signature)?
    nun = grp[qty_col].nunique()
    differing = int((nun[multi.index] > 1).sum())
    differing_share = differing / len(multi)
    products = sorted({str(p) for p, _ in multi.index})

    if differing_share >= 0.5:
        sev, conf, action = "info", 0.85, "aggregate_duplicate_product_date"
        rationale = ("الكميات تختلف داخل الفترة الواحدة — السجلّات معاملات متعدّدة "
                     "على الأرجح، والتجميع هو التفسير الصحيح، ليست تكراراً.")
    else:
        sev, conf, action = "warning", 0.5, "review_duplicates"
        rationale = ("السجلّات داخل الفترة متطابقة الكمية غالباً — قد تكون تكراراً "
                     "وقد تكون معاملات متطابقة؛ لا دليل كافياً للحسم آلياً.")

    return _issue(
        "multiple_records_same_product_period", "time_series", sev, conf,
        f"{len(multi)} مجموعة (منتج، فترة) بأكثر من سجلّ ({extra_records} سجلّاً إضافياً).",
        extra_records, products,
        "بلا تجميع، تتضخّم بعض الفترات أو تتشوّش الإشارة.",
        "إشارة طلب غامضة تُربك حساب المخزون.",
        action, rationale)


# ─── 3. Business validation ──────────────────────────────────────────────────

def validate_business(
    df: pd.DataFrame, m: Dict[str, Optional[str]],
    business_inputs: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """
    Business-value plausibility: negative demand (returns?), unusual inventory
    values, relatively-unusual lead times (NO industry range — note [2]),
    demand outliers (advisory only — note [6]), suspicious flat patterns,
    inconsistent product identifier variants.
    """
    issues: List[Dict[str, Any]] = []
    product_col, qty_col = m.get("product"), m.get("qty")
    meanings = _meaning_columns(business_inputs)

    # Negative demand — possibly returns; never auto-removed.
    if qty_col and qty_col in df.columns:
        numeric = pd.to_numeric(df[qty_col], errors="coerce")
        neg = int((numeric < 0).sum())
        if neg > 0:
            issues.append(_issue(
                "negative_demand", "business", "warning", 0.5,
                f"{neg} قيمة طلب سالبة.", neg, [],
                "تشوّه المتوسطات إن كانت أخطاء؛ صحيحة إن كانت مرتجعات تُحتسب.",
                "مرتجعات حقيقية تعني مخزوناً عائداً — تجاهلها يضخّم الشراء.",
                "review_negative_demand",
                "السالب قد يكون مرتجعات مشروعة أو تصحيحات — التفسير عندك لا عندنا."))

        # Demand outliers — advisory ONLY; no removal, no capping (note [6]).
        if product_col and product_col in df.columns:
            out_count, out_products = 0, set()
            for product, g in df.groupby(product_col):
                vals = pd.to_numeric(g[qty_col], errors="coerce")
                mask = _mad_outliers(vals)
                if mask.any():
                    out_count += int(mask.sum())
                    out_products.add(str(product))
            if out_count > 0:
                share = out_count / max(len(df), 1)
                issues.append(_issue(
                    "unusual_demand_observations", "business", "info", 0.5,
                    f"{out_count} قيمة طلب بعيدة جداً عن نمط منتجها.",
                    out_count, sorted(out_products),
                    "إن كانت أحداثاً حقيقية (عروض/مشاريع) فحذفها يفقد الموديل أهم إشاراته.",
                    f"تمثّل ~{share:.1%} من السجلّات — أثرها على المتوسطات محدود غالباً.",
                    "review_outliers",
                    "الذرى قد تكون عروضاً أو مشاريع كبيرة أو تعافياً بعد نفاد — "
                    "لا نعدّلها آلياً أبداً؛ راجع سياقها التجاري."))

        # Suspicious flat pattern: one value repeated almost everywhere.
        nz = pd.to_numeric(df[qty_col], errors="coerce").dropna()
        nz = nz[nz != 0]
        if len(nz) >= 50 and (nz.value_counts(normalize=True).iloc[0] > 0.95):
            issues.append(_issue(
                "suspicious_constant_demand", "business", "info", 0.6,
                "قيمة واحدة تتكرّر في أكثر من 95% من سجلّات الطلب.", int(len(nz)), [],
                "بيانات شبه ثابتة قد تكون قيماً افتراضية لا طلباً فعلياً.",
                "توصيات مبنية على قيم افتراضية بلا معنى.",
                "review_coverage", "نمط غير معتاد في بيانات مبيعات حقيقية — تأكّد من المصدر."))

    # Inventory plausibility: negative stock (balance columns MAY be negative).
    for meaning in ("stock_on_hand", "available_inventory"):
        col = meanings.get(meaning)
        if col and col in df.columns:
            neg = int((pd.to_numeric(df[col], errors="coerce") < 0).sum())
            if neg > 0:
                issues.append(_issue(
                    "unusual_inventory_values", "business", "warning", 0.7,
                    f"عمود «{col}»: {neg} قيمة مخزون سالبة.", neg, [],
                    "لا أثر على التوقّع.", "مخزون سالب يُفسد كشف النفاد والتوصيات.",
                    "review_inventory_values",
                    "قد يكون backorder مشفّراً بالسالب أو خطأ نظام — يحتاج تأكيدك."))

    # Lead-time plausibility: relative outliers within the column itself + ≤0.
    lt_col = meanings.get("lead_time")
    if lt_col and lt_col in df.columns:
        lt = pd.to_numeric(df[lt_col], errors="coerce")
        nonpos = int((lt <= 0).sum())
        outl = int(_mad_outliers(lt).sum())
        if nonpos + outl > 0:
            issues.append(_issue(
                "unusual_lead_times", "business", "warning" if nonpos else "info", 0.6,
                f"عمود «{lt_col}»: {nonpos} قيمة ≤ صفر و{outl} قيمة شاذة نسبياً.",
                nonpos + outl, [],
                "لا أثر على التوقّع.", "مهلة غير منطقية تُفسد تخطيط إعادة الطلب.",
                "review_lead_times",
                "الشذوذ يُقاس نسبةً لقيم العمود نفسه — لا نفترض نطاقاً قطاعياً."))

    # Product identifier variants (same id after trim/case-fold).
    if product_col and product_col in df.columns:
        raw = df[product_col].dropna().astype("string")
        norm = raw.str.strip().str.lower()
        groups: Dict[str, set] = {}
        for r, nval in zip(raw, norm):
            groups.setdefault(nval, set()).add(r)
        collisions = {k: v for k, v in groups.items() if len(v) > 1}
        if collisions:
            issues.append(_issue(
                "inconsistent_product_identifiers", "business", "warning", 0.8,
                f"{len(collisions)} منتجاً مكتوباً بأكثر من صيغة (مسافات/حالة أحرف).",
                sum(len(v) for v in collisions.values()), sorted(collisions)[:20],
                "تاريخ المنتج الواحد يتجزّأ على صيغ متعددة فتضعف كل سلسلة.",
                "مخزون المنتج الواحد يتوزّع على هويات وهمية.",
                "standardize_product_identifiers",
                "صيغ تختلف بمسافة أو حالة أحرف فقط — التوحيد آمن نسبياً لكنه "
                "يلمس معرّفات، لذلك يحتاج موافقتك (AC-100 و ac-100 قد يختلفان فعلاً)."))

    return issues


# ─── 4. Representation validation (new) ──────────────────────────────────────

def validate_representation(
    df: pd.DataFrame, m: Dict[str, Optional[str]], freq: str,
    business_inputs: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """
    Representativeness checks (not demographics): product coverage imbalance,
    segment (region/customer) imbalance, seasonal coverage gaps, stale products
    whose history stops well before the dataset's end.
    """
    issues: List[Dict[str, Any]] = []
    date_col, product_col, qty_col = m.get("date"), m.get("product"), m.get("qty")
    if not (date_col and product_col and qty_col):
        return issues
    meanings = _meaning_columns(business_inputs)

    qty = pd.to_numeric(df[qty_col], errors="coerce").clip(lower=0)
    dates = pd.to_datetime(df[date_col], errors="coerce")

    # Product coverage imbalance: one product dominating total demand.
    totals = qty.groupby(df[product_col]).sum().sort_values(ascending=False)
    if len(totals) >= 3 and totals.sum() > 0:
        top_share = totals.iloc[0] / totals.sum()
        if top_share > 0.6:
            issues.append(_issue(
                "product_coverage_imbalance", "representation", "info", 0.8,
                f"منتج واحد («{totals.index[0]}») يمثّل {top_share:.0%} من إجمالي الطلب.",
                0, [str(totals.index[0])],
                "الموديلات ستتعلّم نمط المنتج المهيمن وتضعف للبقية.",
                "توصيات المنتجات الصغيرة أقل موثوقية.",
                "review_coverage", "ليس خطأً — حقيقة تمثيل تستحق وعيك عند قراءة النتائج."))

    # Segment imbalance via intake-detected region/customer columns.
    for meaning, label in (("region", "المنطقة"), ("customer", "العميل")):
        col = meanings.get(meaning)
        if col and col in df.columns:
            seg = qty.groupby(df[col]).sum().sort_values(ascending=False)
            if len(seg) >= 3 and seg.sum() > 0 and seg.iloc[0] / seg.sum() > 0.8:
                issues.append(_issue(
                    f"{meaning}_imbalance", "representation", "info", 0.7,
                    f"{label} «{seg.index[0]}» يمثّل {seg.iloc[0] / seg.sum():.0%} من الطلب.",
                    0, [],
                    f"التوقّع يعكس سلوك هذا الـ{label} أساساً — تعميمه على غيره ضعيف.",
                    "لا أثر مباشر.", "review_coverage",
                    "اختلال تمثيل لا خطأ بيانات — مهم عند تفسير النتائج."))

    # Seasonal coverage: history spans ≥ a year but some calendar months never observed.
    valid_dates = dates.dropna()
    if len(valid_dates) > 0:
        span_days = (valid_dates.max() - valid_dates.min()).days
        observed_months = set(valid_dates.dt.month.unique())
        if span_days >= 365 and len(observed_months) < 12:
            missing_months = sorted(set(range(1, 13)) - observed_months)
            issues.append(_issue(
                "seasonal_coverage_gap", "representation", "warning", 0.8,
                f"أشهر تقويمية بلا أي سجلّ إطلاقاً: {missing_months}.", 0, [],
                "الموديل لا يستطيع تعلّم موسم لم يرَه أبداً.",
                "مخزون مواسم غير المغطّاة يُخطّط بلا أساس.",
                "review_coverage", "إن كانت أشهر إغلاق فهذا طبيعي؛ وإن كان نقص تسجيل فهو فجوة حقيقية."))

        # Stale products: last record well before the dataset's end.
        period_days = {"D": 1, "W": 7, "MS": 30, "QS": 91, "YS": 365}.get(freq, 30)
        last_per_product = dates.groupby(df[product_col]).max()
        stale = last_per_product[(valid_dates.max() - last_per_product).dt.days
                                 > 3 * period_days]
        if len(stale) > 0:
            issues.append(_issue(
                "recent_history_gap", "representation", "warning", 0.7,
                f"{len(stale)} منتجاً توقّفت سجلّاته قبل نهاية البيانات بأكثر من ٣ فترات.",
                len(stale), [str(p) for p in stale.index],
                "توقّع منتج توقّفت بياناته حديثاً يمتدّ من قاعدة قديمة.",
                "قد تشتري لمنتج لم يعد يباع.",
                "review_coverage", "قد يكون المنتج أُوقف فعلاً — معلومة تجارية تملكها أنت."))

    return issues


# ─── Readiness scores (value quality — distinct from intake's presence) ──────

def compute_readiness_scores(
    issues: List[Dict[str, Any]],
    business_inputs: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    VALUE-quality readiness (0-100) per area: starts at 100, penalised by the
    issues affecting that area, scaled by severity and weighted by confidence.
    These measure requirement completeness, NOT expected accuracy.
    """
    meanings = _meaning_columns(business_inputs)

    def _score(categories: set, extra_types: set, label_ar: str) -> Dict[str, Any]:
        score, reasons = 100.0, ["الدرجة تقيس سلامة قيم البيانات لا الدقّة المتوقّعة."]
        hit = False
        for iss in issues:
            if iss["category"] in categories or iss["issue_type"] in extra_types:
                pen = _SEVERITY_PENALTY[iss["severity"]] * max(iss["confidence"], 0.3)
                score -= pen
                hit = True
                reasons.append(f"−{pen:.0f}: {iss['issue_type']} "
                               f"({iss['affected_records']} سجلّ).")
        if not hit:
            reasons.append("لا مشاكل قيمية مؤثّرة في هذا المجال.")
        return {"label_ar": label_ar, "score": int(round(max(0, min(100, score)))),
                "reasons": reasons}

    forecast = _score({"structural", "time_series", "representation"}, set(),
                      "جاهزية التوقّع")

    has_inventory = any(k in meanings for k in
                        ("stock_on_hand", "available_inventory", "lead_time", "safety_stock"))
    if has_inventory:
        inventory = _score(set(), {"unusual_inventory_values", "unusual_lead_times",
                                   "negative_demand"}, "جاهزية المخزون")
    else:
        inventory = {"label_ar": "جاهزية المخزون", "score": 0,
                     "reasons": ["لا أعمدة مخزون لتقييم قيمها (التوفّر مسؤولية intake).",
                                 "الدرجة تقيس سلامة القيم لا الدقّة."]}

    has_drivers = any(k in meanings for k in ("price", "promotion", "discount"))
    sc = forecast["score"] * 0.5 + (50 if has_drivers else 0)
    scenario = {"label_ar": "جاهزية محاكاة السيناريوهات",
                "score": int(round(min(100, sc))),
                "reasons": ["يرث نصف جاهزية قيم التوقّع.",
                            "محرّكات الطلب (سعر/عرض) متوفّرة (+50)." if has_drivers
                            else "لا محرّكات طلب — قوة المحاكاة محدودة."]}

    return {"forecast_readiness": forecast, "inventory_readiness": inventory,
            "scenario_readiness": scenario}


# ─── Remediation plan ────────────────────────────────────────────────────────

def build_remediation_plan(
    df: pd.DataFrame,
    issues: List[Dict[str, Any]],
    m: Dict[str, Optional[str]],
    freq: str,
) -> Dict[str, Any]:
    """
    Routes each issue's ONE recommended action into plan buckets by policy
    (note [7]) and embeds the impact estimate directly in each item — the user
    sees consequences before approving. Baseline technical hygiene (whitespace,
    date format) is added to auto when applicable. Pure planning, no execution.
    """
    auto, review, skipped = [], [], []
    seen: set = set()

    def _impact_estimate(action: str) -> Dict[str, int]:
        """Cheap pre-execution consequence estimate for an action."""
        est = dict(rows_affected=0, records_removed=0, values_modified=0,
                   new_periods_created=0)
        date_col, qty_col, product_col = m.get("date"), m.get("qty"), m.get("product")
        if action == "remove_empty_rows":
            n = int(df.isna().all(axis=1).sum())
            est.update(rows_affected=n, records_removed=n)
        elif action == "remove_exact_duplicates":
            n = int(df.duplicated().sum())
            est.update(rows_affected=n, records_removed=n)
        elif action == "trim_whitespace":
            n = 0
            for c in df.columns:
                if pd.api.types.is_object_dtype(df[c]):
                    s = df[c].astype("string")
                    n += int((s != s.str.strip()).sum())
            est.update(rows_affected=n, values_modified=n)
        elif action == "standardize_date_format" and date_col:
            est.update(values_modified=int(len(df)))
        elif action == "fill_missing_demand" and qty_col:
            n = int(pd.to_numeric(df[qty_col], errors="coerce").isna().sum())
            est.update(rows_affected=n, values_modified=n)
        elif action == "remove_invalid_dates" and date_col:
            parsed = pd.to_datetime(df[date_col], errors="coerce")
            n = int(parsed.isna().sum() - df[date_col].isna().sum())
            est.update(rows_affected=n, records_removed=n)
        elif action == "remove_invalid_demand" and qty_col:
            numeric = pd.to_numeric(df[qty_col], errors="coerce")
            n = int(numeric.isna().sum() - df[qty_col].isna().sum())
            est.update(rows_affected=n, records_removed=n)
        elif action == "aggregate_duplicate_product_date" and date_col and product_col:
            parsed = pd.to_datetime(df[date_col], errors="coerce")
            tmp = pd.DataFrame({"p": df[product_col], "d": parsed}).dropna()
            n = int(tmp.duplicated(subset=["p", "d"]).sum())
            est.update(rows_affected=n, records_removed=n)
        elif action == "insert_missing_periods":
            total = sum(i["affected_records"] for i in issues
                        if i["issue_type"] == "missing_periods")
            est.update(new_periods_created=total, rows_affected=total)
        elif action == "standardize_product_identifiers":
            total = sum(i["affected_records"] for i in issues
                        if i["issue_type"] == "inconsistent_product_identifiers")
            est.update(values_modified=total, rows_affected=total)
        return est

    def _add(action: str, source_issue: Optional[Dict[str, Any]]) -> None:
        if action in seen:
            return
        seen.add(action)
        spec = _ACTIONS[action]
        impact = _impact_estimate(action)
        item = {"action": action, "label": spec["label"],
                "estimated_records_affected": impact["rows_affected"],
                "impact": impact}
        if spec["policy"] == "auto":
            auto.append(item)
        elif spec["policy"] == "review":
            item["reason_for_review"] = spec["benefit"]
            if source_issue:
                item["rationale"] = source_issue["rationale"]
                item["confidence"] = source_issue["confidence"]
            review.append(item)
        else:  # never / advisory
            item["reason_skipped"] = ("Business-data modification — never applied automatically."
                                      if spec["policy"] == "never"
                                      else "No automatic fix possible — needs your business judgment.")
            skipped.append(item)

    # Baseline technical hygiene first (only when applicable).
    for base_action in ("trim_whitespace", "remove_empty_rows", "standardize_date_format"):
        if _impact_estimate(base_action)["rows_affected"] > 0 \
                or base_action == "standardize_date_format" and m.get("date"):
            _add(base_action, None)

    # Issue-driven actions (one per issue; low confidence already routed to review/advisory
    # at detection time, so no second-guessing here).
    for iss in issues:
        _add(iss["recommended_action"], iss)

    return {"auto_actions": auto, "review_required": review, "skipped_actions": skipped}


# ─── Execution + audit trail ─────────────────────────────────────────────────

def execute_plan(
    df: pd.DataFrame,
    plan: Dict[str, Any],
    approved_actions: List[str],
    core_mapping: Dict[str, Optional[str]],
    freq: str = "MS",
) -> Dict[str, Any]:
    """
    Executes ONLY user-approved actions on a copy of df, with a full audit trail.
    Approving a 'never'/advisory action still runs it if an executor exists, but
    is logged as a loud manual override — nothing happens silently.
    Returns {data, execution_log: {applied, not_applied, warnings}}.
    """
    cleaned = df.copy()
    applied, not_applied, warns = [], [], []

    planned = {}
    for bucket in ("auto_actions", "review_required", "skipped_actions"):
        for item in plan.get(bucket, []):
            planned[item["action"]] = bucket

    executors = _build_executors()
    for action, bucket in planned.items():
        if action not in approved_actions:
            not_applied.append({"action": action, "reason": f"غير مُوافَق عليه (في {bucket})."})
            continue
        if bucket == "skipped_actions":
            warns.append(f"الإجراء «{action}» مُصنَّف للتخطّي لكن المستخدم وافق صراحةً "
                         "— نُفّذ كتجاوز يدوي.")
        fn = executors.get(action)
        if fn is None:
            not_applied.append({"action": action, "reason": "إجراء استشاري — لا منفّذ آلي له."})
            continue
        try:
            cleaned, n = fn(cleaned, core_mapping, freq)
            applied.append({"action": action, "records_affected": int(n),
                            "timestamp": datetime.now().isoformat(timespec="seconds")})
        except Exception as exc:  # one failing action must not sink the rest
            not_applied.append({"action": action, "reason": f"فشل التنفيذ: {exc}"})
            warns.append(f"تعذّر تنفيذ «{action}»: {exc}")

    return {"data": cleaned,
            "execution_log": {"applied": applied, "not_applied": not_applied,
                              "warnings": warns}}


def _build_executors() -> Dict[str, Callable]:
    """action → fn(df, mapping, freq) → (new_df, records_affected). No outlier
    capping executor exists by design (note [6])."""

    def _trim(df, cm, freq):
        out, n = df.copy(), 0
        for c in out.columns:
            if pd.api.types.is_object_dtype(out[c]):
                s = out[c].astype("string")
                n += int((s != s.str.strip()).sum())
                out[c] = s.str.strip()
        return out, n

    def _empty(df, cm, freq):
        n = int(df.isna().all(axis=1).sum())
        return df.dropna(how="all"), n

    def _dates(df, cm, freq):
        out, dc = df.copy(), cm.get("date")
        if dc:
            out[dc] = pd.to_datetime(out[dc], errors="coerce")
        return out, int(len(out)) if dc else 0

    def _exact_dup(df, cm, freq):
        n = int(df.duplicated().sum())
        return df.drop_duplicates(), n

    def _fill_missing(df, cm, freq):
        out, qc = df.copy(), cm.get("qty")
        if not qc:
            return out, 0
        numeric = pd.to_numeric(out[qc], errors="coerce")
        n = int(numeric.isna().sum())
        out[qc] = numeric.fillna(0)
        return out, n

    def _aggregate(df, cm, freq):
        dc, pc, qc = cm.get("date"), cm.get("product"), cm.get("qty")
        if not (dc and pc and qc):
            return df, 0
        out = df.copy()
        out[dc] = pd.to_datetime(out[dc], errors="coerce")
        out[qc] = pd.to_numeric(out[qc], errors="coerce")
        before = len(out)
        agg = {c: ("sum" if c == qc else "first") for c in out.columns if c not in (dc, pc)}
        out = out.groupby([pc, dc], as_index=False).agg(agg)
        return out, int(before - len(out))

    def _insert_periods(df, cm, freq):
        dc, pc, qc = cm.get("date"), cm.get("product"), cm.get("qty")
        if not (dc and pc and qc):
            return df, 0
        out = df.copy()
        out[dc] = pd.to_datetime(out[dc], errors="coerce")
        out[qc] = pd.to_numeric(out[qc], errors="coerce")
        out = out.dropna(subset=[dc])
        frames, created = [], 0
        for product, g in out.groupby(pc):
            s = g.set_index(dc)[qc].resample(freq).sum()
            full = pd.date_range(s.index.min(), s.index.max(), freq=freq)
            created += len(full) - len(s)
            s = s.reindex(full, fill_value=0)
            frames.append(pd.DataFrame({pc: product, dc: s.index, qc: s.values}))
        return (pd.concat(frames, ignore_index=True) if frames else out), int(created)

    def _rm_invalid_dates(df, cm, freq):
        dc = cm.get("date")
        if not dc:
            return df, 0
        mask = pd.to_datetime(df[dc], errors="coerce").isna() & df[dc].notna()
        return df[~mask], int(mask.sum())

    def _rm_invalid_demand(df, cm, freq):
        qc = cm.get("qty")
        if not qc:
            return df, 0
        mask = pd.to_numeric(df[qc], errors="coerce").isna() & df[qc].notna()
        return df[~mask], int(mask.sum())

    def _standardize_ids(df, cm, freq):
        pc = cm.get("product")
        if not pc:
            return df, 0
        out = df.copy()
        s = out[pc].astype("string")
        new = s.str.strip()
        n = int((s != new).sum())
        out[pc] = new
        return out, n

    def _rm_negative(df, cm, freq):  # executes ONLY via explicit manual override
        qc = cm.get("qty")
        if not qc:
            return df, 0
        mask = pd.to_numeric(df[qc], errors="coerce") < 0
        return df[~mask], int(mask.sum())

    return {
        "trim_whitespace": _trim, "remove_empty_rows": _empty,
        "standardize_date_format": _dates, "remove_exact_duplicates": _exact_dup,
        "fill_missing_demand": _fill_missing,
        "aggregate_duplicate_product_date": _aggregate,
        "insert_missing_periods": _insert_periods,
        "remove_invalid_dates": _rm_invalid_dates,
        "remove_invalid_demand": _rm_invalid_demand,
        "standardize_product_identifiers": _standardize_ids,
        "remove_negative_demand": _rm_negative,
    }


def execution_log_to_dataframe(execution_log: Dict[str, Any]) -> pd.DataFrame:
    """Flattens an execution log into a tidy frame for CSV download."""
    rows: List[Dict[str, Any]] = []
    for a in execution_log.get("applied", []):
        rows.append({"status": "applied", "action": a["action"],
                     "records_affected": a.get("records_affected"),
                     "timestamp": a.get("timestamp"), "detail": ""})
    for a in execution_log.get("not_applied", []):
        rows.append({"status": "not_applied", "action": a["action"],
                     "records_affected": None, "timestamp": None,
                     "detail": a.get("reason", "")})
    for w in execution_log.get("warnings", []):
        rows.append({"status": "warning", "action": None, "records_affected": None,
                     "timestamp": None, "detail": w})
    return pd.DataFrame(rows, columns=["status", "action", "records_affected",
                                       "timestamp", "detail"])


# ─── Main entry point ────────────────────────────────────────────────────────

def run_quality_engine(
    df: pd.DataFrame,
    intake_result: Optional[Dict[str, Any]] = None,
    core_mapping: Optional[Dict[str, Optional[str]]] = None,
    business_inputs: Optional[List[Dict[str, Any]]] = None,
    target_freq: Optional[str] = None,
    min_history: int = 12,
    intermittency_threshold: float = 0.30,
    feature_engineering_min_forecast_readiness: int = 40,
) -> Dict[str, Any]:
    """
    Main entry point — assess, classify, recommend; never execute.

    Pipeline: structural → time-series → business → representation validation
    → priority sort → readiness scores → remediation plan (with embedded impact
    estimates). Execution is a separate, approval-gated step (execute_plan).

    Returns: {issues, issues_by_category, readiness_scores, remediation_plan,
    execution_log (empty), ready_for_feature_engineering, assessment_frequency,
    summary_for_user}. See note [9] for keys removed vs the previous version.
    """
    if core_mapping is None and intake_result:
        core_mapping = intake_result.get("proposed_mapping")
    core_mapping = core_mapping or {}
    if business_inputs is None and intake_result:
        business_inputs = intake_result.get("business_inputs")

    freq = _resolve_freq(intake_result, target_freq)

    issues = (validate_structural(df, core_mapping)
              + validate_time_series(df, core_mapping, freq,
                                     min_history=min_history,
                                     intermittency_threshold=intermittency_threshold)
              + validate_business(df, core_mapping, business_inputs)
              + validate_representation(df, core_mapping, freq, business_inputs))

    # Prioritize: severity first, then blast radius. The UI shows what matters most.
    issues.sort(key=lambda i: (_SEVERITY_RANK[i["severity"]], -i["affected_records"]))

    by_category: Dict[str, List[Dict[str, Any]]] = {
        "structural": [], "time_series": [], "business": [], "representation": []}
    for iss in issues:
        by_category[iss["category"]].append(iss)

    readiness = compute_readiness_scores(issues, business_inputs)
    plan = build_remediation_plan(df, issues, core_mapping, freq)

    has_critical = any(i["severity"] == "critical" for i in issues)
    ready_for_fe = bool(not has_critical
                        and readiness["forecast_readiness"]["score"]
                        >= feature_engineering_min_forecast_readiness)

    # Plain-language headline: top 3 issues only — don't drown the user.
    top = issues[:3]
    if not issues:
        summary = "لا مشاكل مؤثّرة — بياناتك جاهزة للمتابعة."
    else:
        summary = (f"وجدنا {len(issues)} ملاحظة، أهمها: "
                   + "؛ ".join(i["description"] for i in top))

    return {
        "issues": issues,
        "issues_by_category": by_category,
        "readiness_scores": readiness,
        "remediation_plan": plan,
        "execution_log": {"applied": [], "not_applied": [], "warnings": []},
        "ready_for_feature_engineering": ready_for_fe,
        "assessment_frequency": freq,
        "summary_for_user": summary,
    }
