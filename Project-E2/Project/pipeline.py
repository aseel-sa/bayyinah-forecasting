"""
pipeline.py — منسّق الطبقات | Bayyina Platform

يربط: intake → quality → feature_engineering → model.
منسّق بحت: لا منطق تجاري هنا، فقط تسلسل المراحل وتمرير المخرجات كما هي.

نقاط الدخول:
  run_pipeline(raw_dataframe, config)
  resume_after_quality_review(state, user_decisions)
"""

# ===========================================================================
## ملاحظات المطوّر واقتراحاته
# ===========================================================================
#
# [خ١] قابلية التوسّع — قائمة مراحل لا شيفرة متشعّبة:
#      المراحل بعد الجودة مُعرّفة في _POST_QUALITY_STAGES كقائمة (الاسم، الدالة).
#      إضافة طبقة لاحقة (insights/chatbot/inventory) = سطر واحد في القائمة + دالة
#      مرحلة بتوقيع موحّد _stage(ctx) — لا إعادة هيكلة. هذا يحقّق مبدأك:
#      "الإضافة إدراج لا إعادة كتابة".
#
# [خ٢] لماذا قائمة للذيل فقط لا للأنبوب كله:
#      مرحلة الجودة فيها "نقطة توقّف" بشرية (awaiting_quality_review) تكسر
#      الخطّية. فصلت ما قبل التوقّف (intake→quality) عن الذيل القابل للاستئناف
#      (feature→model→...المستقبل). الطبقات القادمة تأتي بعد model فتُلحَق بالذيل.
#      بديل: محرّك مراحل عام مع وسم "pausable" لكل مرحلة — أقوى لكنه تجريد زائد
#      لحالة توقّف واحدة. اخترت الأبسط ووثّقت نقطة التوسّع.
#
# [خ٣] ازدواج بسيط مقصود — مرحلة feature_engineering مقابل ما يبنيه model:
#      المنسّق يشغّل fe.build_features كمرحلة مستقلّة (للإخراج + بوّابة تدقيق
#      التسريب قبل النمذجة الثقيلة). model.run_forecast يبني الميزات داخلياً أيضاً
#      (walk-forward لكل طيّة + توقّع نهائي). فالميزات تُبنى مرّتين منطقياً.
#      أبقيته هكذا لأن: (أ) المنسّق يستدعي واجهات عامة فقط (لا privates)،
#      (ب) مرحلة fe تعمل كبوّابة تسريب توقف الأنبوب قبل النمذجة. تحسين مستقبلي:
#      تمرير الميزات المبنية إلى model لتفادي إعادة البناء. نبّهني لو أردته.
#
# [خ٤] حالة الاستئناف تُحفظ في _pipeline_state داخل المخرجات:
#      تحوي raw_df + config + core_mapping + business_inputs. في Streamlit تُخزَّن
#      في session_state. raw_df داخل dict مقبول (في-العملية). عند الاكتمال/الفشل
#      تُضبط None (لا حاجة للاستئناف).
#
# [خ٥] التوقّف الصاخب (halt loudly) للحالات الأربع المطلوبة:
#      لا عمود تاريخ، لا عمود هدف، فشل تدقيق التسريب، فشل كل الموديلات.
#      كلها → status="failed" + سبب صريح في errors. لا فشل صامت.
#      ملاحظة: نقص عمود المنتج أيضاً يوقف (التوقّع لكل منتج يتطلّبه) — أضفته
#      للحالات الموقفة رغم أنه غير مذكور صراحةً، لأن غيابه يُعطّل كل ما بعده.
#
# [خ٦] auto_approve: نوافق على auto_actions + review_required، ونستثني صراحةً
#      remove_negative_demand (وهو أصلاً في skipped لا review — استثناء دفاعي).
#      لا نوافق على skipped (تعديل بيانات تجارية/استشاري) إطلاقاً تلقائياً.
#
# [خ٧] لا منطق تجاري هنا: المنسّق لا يقرّر عتبات ولا يفسّر بيانات. كل قرار
#      دلالي يخصّ طبقته. اختيار الإجراءات التلقائية سياسةُ تنسيق حدّدتها المواصفة،
#      لا منطق مجال.
#
# ===========================================================================

import time
import traceback
from typing import Any, Callable, Dict, List, Optional

import pandas as pd

import intake
import quality
import external_features as ext
import feature_engineering as fe
import model
import forecast_analysis as fa
import insights as insights_layer


# ─── أدوات السياق والتسجيل ───────────────────────────────────────────────────

def _new_context(raw_df: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Creates the shared mutable pipeline context. Holds inputs, per-stage outputs,
    logs, and resume state. Does NOT run any stage.
    """
    return {
        "raw_df": raw_df,
        "config": config,
        "status": "running",          # running | complete | failed | awaiting_quality_review
        "intake_result": {},
        "quality_result": {},
        "external_result": {},
        "feature_result": {},
        "model_result": {},
        "performance_result": {},
        "insights": {},
        "core_mapping": None,
        "business_inputs": None,
        "cleaned_df": None,
        "errors": [],
        "warnings": [],
        "stage_log": [],
    }


def _halt(ctx: Dict[str, Any], stage: str, message: str) -> None:
    """Marks the pipeline failed with an explicit error (loud halt). No exception."""
    ctx["status"] = "failed"
    ctx["errors"].append({"stage": stage, "message": message})


def _run_stage(ctx: Dict[str, Any], name: str, fn: Callable[[Dict[str, Any]], None]) -> None:
    """
    Runs one stage with timing + try/except, recording status and duration in
    stage_log. A stage signals failure either by raising or by calling _halt.
    Does NOT proceed if the pipeline is already failed/awaiting.
    """
    if ctx["status"] in ("failed", "awaiting_quality_review"):
        return
    entry: Dict[str, Any] = {"stage": name, "status": "ok"}
    t0 = time.perf_counter()
    try:
        fn(ctx)
        if ctx["status"] == "failed":
            entry["status"] = "halted"
        elif ctx["status"] == "awaiting_quality_review":
            entry["status"] = "paused"
    except Exception as exc:
        entry["status"] = "error"
        entry["error"] = f"{type(exc).__name__}: {exc}"
        ctx["errors"].append({"stage": name, "message": entry["error"],
                              "trace": traceback.format_exc()})
        ctx["status"] = "failed"
    finally:
        entry["duration_s"] = round(time.perf_counter() - t0, 4)
        ctx["stage_log"].append(entry)

# المنطق: غلاف موحّد لكل مرحلة — توقيت + التقاط الأخطاء + تسجيل الحالة. الفشل
# (استثناءً أو عبر _halt) يوقف الأنبوب، والتوقّف البشري يُعلَّم "paused".


# ─── المراحل (كل مرحلة: ctx → None، تستدعي واجهات الطبقات العامة فقط) ─────────

def _stage_intake(ctx: Dict[str, Any]) -> None:
    """
    Stage 1 — intake: builds intake_result and extracts the column mapping.
    Halts loudly on missing date / target(qty) / product columns.
    Does NOT transform data — pass-through of intake's output.
    """
    ir = intake.build_intake_result(ctx["raw_df"])
    ctx["intake_result"] = ir
    cm = ir.get("proposed_mapping") or {}
    ctx["core_mapping"] = cm
    ctx["business_inputs"] = ir.get("business_inputs")

    if not cm.get("date"):
        _halt(ctx, "intake", "No date column found — a time series cannot be built.")
    if not cm.get("qty"):
        _halt(ctx, "intake", "No target (quantity) column found — nothing to forecast.")
    if not cm.get("product"):
        _halt(ctx, "intake", "No product column found — per-product forecasting requires it.")


def _stage_quality_assess(ctx: Dict[str, Any]) -> None:
    """
    Stage 2 — quality assessment: runs the quality engine to produce assessment +
    remediation plan (no execution yet). Adds a warning if data is not ready for
    feature engineering. Does NOT execute remediation (that is a separate step).
    """
    freq = fe._GRAN_CONFIG[ctx["config"]["granularity"]]["freq"]
    qr = quality.run_quality_engine(
        ctx["raw_df"], ctx["intake_result"],
        core_mapping=ctx["core_mapping"], business_inputs=ctx["business_inputs"],
        target_freq=freq,
    )
    ctx["quality_result"] = qr
    if not qr.get("ready_for_feature_engineering", True):
        ctx["warnings"].append(
            {"stage": "quality", "message": "Data is not fully ready for feature engineering "
             "(possible critical value issues) — continuing may fail at the feature stage."})


def _auto_approved_actions(plan: Dict[str, Any]) -> List[str]:
    """
    Orchestration policy for auto_approve: approve all auto + review_required actions,
    explicitly excluding remove_negative_demand (business-data modification).
    Does NOT touch skipped_actions. Pure policy, no domain logic.
    """
    actions: List[str] = [a["action"] for a in plan.get("auto_actions", [])]
    actions += [a["action"] for a in plan.get("review_required", [])
                if a["action"] != "remove_negative_demand"]
    return actions


def _stage_quality_execute(ctx: Dict[str, Any], approved_actions: List[str]) -> None:
    """
    Stage 2b — quality execution: applies the approved remediation actions via
    quality.execute_plan, producing the cleaned DataFrame and an audit trail.
    Surfaces execution warnings. Does NOT decide what to approve (caller/user does).
    """
    freq = fe._GRAN_CONFIG[ctx["config"]["granularity"]]["freq"]
    exec_res = quality.execute_plan(
        ctx["raw_df"], ctx["quality_result"]["remediation_plan"],
        approved_actions, ctx["core_mapping"], freq=freq,
    )
    ctx["cleaned_df"] = exec_res["data"]
    # نمرّر سجلّ التنفيذ كما هو ضمن quality_result (لا نلمس مخرجات الطبقة)
    ctx["quality_result"]["execution_log"] = exec_res["execution_log"]
    for w in exec_res["execution_log"].get("warnings", []):
        ctx["warnings"].append({"stage": "quality_execute", "message": w})


# sector flag → industry name understood by the external-features registry.
_SECTOR_TO_INDUSTRY = {"hvac": "HVAC"}


def _stage_external(ctx: Dict[str, Any]) -> None:
    """
    Stage 3 — external features: the PLATFORM provides external features
    (calendar in code, weather/industry from internal CSVs) for the demand
    data's date range. Non-fatal by design: any failure degrades to "no
    external features" with a warning — never halts the pipeline.
    """
    df = ctx["cleaned_df"] if ctx["cleaned_df"] is not None else ctx["raw_df"]
    cfg = ctx["config"]
    date_col = ctx["core_mapping"].get("date")
    try:
        dates = pd.to_datetime(df[date_col], errors="coerce").dropna()
        if dates.empty:
            ctx["external_result"] = {}
            return
        res = ext.build_external_features(
            start_date=dates.min(), end_date=dates.max(),
            granularity=cfg["granularity"],
            country=cfg.get("country"), city=cfg.get("city"),
            industry=_SECTOR_TO_INDUSTRY.get(cfg.get("sector") or "", cfg.get("sector")),
        )
        ctx["external_result"] = res
        for w in res.get("warnings", []):
            ctx["warnings"].append({"stage": "external_features", "message": w})
    except Exception as exc:
        ctx["external_result"] = {}
        ctx["warnings"].append({"stage": "external_features",
                                "message": f"Could not build external features ({exc}) — continued without them."})


def _external_frame(ctx: Dict[str, Any]):
    """(frame, metadata) from the external stage, or (None, None) when absent."""
    res = ctx.get("external_result") or {}
    frame = res.get("external_features")
    if frame is None or len(getattr(frame, "columns", [])) == 0:
        return None, None
    return frame, res.get("feature_metadata")


def _stage_features(ctx: Dict[str, Any]) -> None:
    """
    Stage 4 — feature_engineering: builds model-ready feature matrices on the cleaned
    data (or raw if no cleaning ran), merging platform-provided external features.
    Acts as the leakage gate: halts loudly if the leakage_audit fails.
    """
    df = ctx["cleaned_df"] if ctx["cleaned_df"] is not None else ctx["raw_df"]
    ext_frame, ext_meta = _external_frame(ctx)
    fr = fe.build_features(
        df, intake_result=None, granularity=ctx["config"]["granularity"],
        sector=ctx["config"].get("sector"),
        core_mapping=ctx["core_mapping"], business_inputs=ctx["business_inputs"],
        external_features=ext_frame, external_metadata=ext_meta,
    )
    ctx["feature_result"] = fr

    if not fr.get("leakage_audit", {}).get("all_checks_passed", False):
        _halt(ctx, "feature_engineering",
              "Leakage audit failed — loud halt before modeling.")
        return
    if not fr.get("ready_for_modeling", False):
        ctx["warnings"].append(
            {"stage": "feature_engineering",
             "message": "feature_engineering: ready_for_modeling=False — results may be weaker."})


def _stage_model(ctx: Dict[str, Any]) -> None:
    """
    Stage 4 — model: runs model.run_forecast (select_best_model via walk-forward +
    final forecast). Halts loudly if no model produced any finite score across all
    products (all models failed). Does NOT build features (model owns its internal use).
    """
    cfg = ctx["config"]
    # Features are consumed from the feature stage (single build — the old
    # double-build duplication noted in [خ٣] is resolved by passing fe_output).
    mr = model.run_forecast(
        intake_result=None, granularity=cfg["granularity"],
        core_mapping=ctx["core_mapping"], business_inputs=ctx["business_inputs"],
        sector=cfg.get("sector"),
        fe_output=ctx["feature_result"],
        forecast_horizon=cfg.get("forecast_horizon", cfg.get("horizon")),
        validation_horizon=cfg.get("validation_horizon"),
        n_eval_splits=cfg.get("n_eval_splits"),
        evaluation_mode=cfg.get("evaluation_mode", "balanced"),
    )
    ctx["model_result"] = mr

    lb = mr.get("model_leaderboard")
    import numpy as np
    ok_any = (lb is not None and len(lb) > 0
              and np.isfinite(pd.to_numeric(lb["wmape"], errors="coerce")).any())
    has_forecast = len(mr.get("product_forecasts", [])) > 0
    if not has_forecast:
        _halt(ctx, "model", "No forecast produced — every model failed, including the baseline.")
    elif not ok_any:
        ctx["warnings"].append({"stage": "model",
                                "message": "No valid validation for any model (insufficient history) — "
                                           "the forecast rests on the safe baseline."})


def _stage_forecast_performance(ctx: Dict[str, Any]) -> None:
    """
    Stage 5 — forecast performance analysis: derives the measured
    actual-vs-predicted-vs-error view from the model's validation output
    (forecast_analysis.analyze_forecast_performance — a read-only consumer).
    Non-fatal by design: any failure degrades to an empty result + warning —
    it can never alter or block the forecast itself. The what-if scenario
    engine is intentionally NOT a stage (it needs user-declared assumptions;
    the UI invokes it on the completed result).
    """
    try:
        ctx["performance_result"] = fa.analyze_forecast_performance(ctx["model_result"])
    except Exception as exc:
        ctx["performance_result"] = {}
        ctx["warnings"].append({"stage": "forecast_performance",
                                "message": f"Forecast performance analysis failed ({exc}) — "
                                           "continued without it; the forecast itself is unaffected."})



def _stage_insights(ctx: Dict[str, Any]) -> None:
    """
    Stage 6 — deterministic decision-support insights (read-only consumer of
    model + performance + quality outputs). NON-BLOCKING by design: any failure
    degrades to the standard 'failed' insights payload and a warning; it can
    never alter or halt the forecast. explainability_result is None for now
    (the contract already accepts a future explainability layer).
    """
    try:
        ctx["insights"] = insights_layer.generate_insights(
            model_result=ctx["model_result"],
            performance_result=ctx.get("performance_result") or None,
            quality_result=ctx.get("quality_result") or None,
            scenario_result=None,
            explainability_result=None,
        )
    except Exception as exc:
        ctx["insights"] = insights_layer._failed_result(f"{type(exc).__name__}: {exc}")
        ctx["warnings"].append({"stage": "insights",
                                "message": f"Insights generation failed ({exc}) — "
                                           "continued without it; the forecast is unaffected."})


# الذيل القابل للاستئناف والتوسّع. أضف مراحل المستقبل هنا (insights/inventory/...).
_POST_QUALITY_STAGES: List[tuple] = [
    ("external_features", _stage_external),
    ("feature_engineering", _stage_features),
    ("model", _stage_model),
    ("forecast_performance", _stage_forecast_performance),
    ("insights", _stage_insights),
    # ("inventory", _stage_inventory),      ← إدراج مستقبلي
]


def _run_post_quality(ctx: Dict[str, Any]) -> None:
    """Runs the extensible post-quality stage sequence, halting on first failure."""
    for name, fn in _POST_QUALITY_STAGES:
        _run_stage(ctx, name, fn)
        if ctx["status"] == "failed":
            return

# المنطق: يمرّ على مراحل الذيل بالترتيب. إضافة مرحلة = عنصر في القائمة، لا أكثر.


# ─── التجميع النهائي للمخرجات ─────────────────────────────────────────────────

def _finalize(ctx: Dict[str, Any]) -> Dict[str, Any]:
    """
    Builds the standard output object from the context. Sets status to 'complete'
    if the run reached the end without halting/pausing. Attaches resume state only
    when awaiting review. Does NOT modify any layer output.
    """
    status = ctx["status"]
    if status == "running":
        status = "complete"

    resume_state = None
    if status == "awaiting_quality_review":
        resume_state = {
            "raw_df": ctx["raw_df"],
            "config": ctx["config"],
            "core_mapping": ctx["core_mapping"],
            "business_inputs": ctx["business_inputs"],
        }

    return {
        "status": status,
        "intake_result": ctx["intake_result"],
        "quality_result": ctx["quality_result"],
        "external_result": ctx["external_result"],
        "feature_result": ctx["feature_result"],
        "model_result": ctx["model_result"],
        "performance_result": ctx["performance_result"],
        "insights": ctx["insights"],
        "errors": ctx["errors"],
        "warnings": ctx["warnings"],
        "stage_log": ctx["stage_log"],
        "_pipeline_state": resume_state,
    }


# ─── نقطة الدخول 1: run_pipeline ─────────────────────────────────────────────

def run_pipeline(raw_dataframe: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Orchestrates intake → quality → feature_engineering → model.

    config:
      granularity          : 'weekly' | 'monthly'   (required)
      sector               : str                     (optional)
      auto_approve_quality : bool                     (default False)
      horizon, n_eval_splits : optional model knobs   (defaults 3, 3)

    Flow:
      1) intake → intake_result (halt if no date/target/product column)
      2) quality → remediation plan
         • auto_approve_quality=True : approve auto + review (except negative demand),
           execute, continue.
         • False : return status='awaiting_quality_review' with the plan + resume state.
      3) feature_engineering → matrices (halt loudly if leakage_audit fails)
      4) model.run_forecast (walk-forward) → best model + forecast (halt if all fail)

    Returns the standard output object. Pure orchestration — no business logic; every
    layer output is passed through untouched.
    """
    # تحقّق إعداد أساسي (إيقاف صاخب مبكر)
    granularity = (config or {}).get("granularity")
    if granularity not in fe._GRAN_CONFIG:
        ctx = _new_context(raw_dataframe, config or {})
        _halt(ctx, "config",
              f"granularity must be one of {list(fe._GRAN_CONFIG)}; got {granularity!r}.")
        return _finalize(ctx)

    ctx = _new_context(raw_dataframe, config)

    # 1) intake
    _run_stage(ctx, "intake", _stage_intake)
    if ctx["status"] == "failed":
        return _finalize(ctx)

    # 2) quality (تقييم + خطّة)
    _run_stage(ctx, "quality", _stage_quality_assess)
    if ctx["status"] == "failed":
        return _finalize(ctx)

    # نقطة القرار: مراجعة بشرية أم موافقة تلقائية
    if not config.get("auto_approve_quality", False):
        ctx["status"] = "awaiting_quality_review"
        return _finalize(ctx)

    approved = _auto_approved_actions(ctx["quality_result"]["remediation_plan"])
    _run_stage(ctx, "quality_execute", lambda c: _stage_quality_execute(c, approved))
    if ctx["status"] == "failed":
        return _finalize(ctx)

    # 3+4) الذيل القابل للتوسّع
    _run_post_quality(ctx)
    return _finalize(ctx)

# المنطق: يشغّل intake ثم quality، ويتوقّف للمراجعة إن لزم، وإلا ينفّذ التنظيف
# المعتمد ثم مراحل الذيل (ميزات→موديل→...). كل خطوة مسجّلة، كل توقّف صاخب.


# ─── نقطة الدخول 2: resume_after_quality_review ──────────────────────────────

def resume_after_quality_review(
    state: Dict[str, Any],
    user_decisions: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Resumes a run that paused at quality review, applying the user's decisions and
    continuing from step 3 (feature_engineering) onward.

    Parameters
    ----------
    state          : the output object returned when status was 'awaiting_quality_review'
                     (must contain intake_result, quality_result, and _pipeline_state).
    user_decisions : {"approved_actions": [...]}  — actions the user approved.
                     If absent, falls back to the auto-approve policy (auto + review,
                     excluding negative demand). Explicitly approved business actions
                     (e.g. remove_negative_demand) are honored but logged as warnings
                     by execute_plan (manual override).

    Returns the standard output object. Does NOT re-run intake/quality assessment —
    it reuses the saved results and only executes remediation + the post-quality tail.
    """
    ps = state.get("_pipeline_state")
    if not ps:
        return {
            "status": "failed",
            "intake_result": state.get("intake_result", {}),
            "quality_result": state.get("quality_result", {}),
            "external_result": {}, "feature_result": {}, "model_result": {},
            "performance_result": {}, "insights": {},
            "errors": [{"stage": "resume", "message": "Resume state missing (_pipeline_state)."}],
            "warnings": [], "stage_log": [], "_pipeline_state": None,
        }

    # إعادة بناء السياق من الحالة المحفوظة (بلا إعادة تشغيل intake/quality)
    ctx = _new_context(ps["raw_df"], ps["config"])
    ctx["intake_result"] = state.get("intake_result", {})
    ctx["quality_result"] = state.get("quality_result", {})
    ctx["core_mapping"] = ps["core_mapping"]
    ctx["business_inputs"] = ps["business_inputs"]

    # قرارات المستخدم: قائمة صريحة أو السياسة التلقائية كاحتياط
    approved = user_decisions.get("approved_actions")
    if approved is None:
        approved = _auto_approved_actions(ctx["quality_result"].get("remediation_plan", {}))

    _run_stage(ctx, "quality_execute", lambda c: _stage_quality_execute(c, approved))
    if ctx["status"] == "failed":
        return _finalize(ctx)

    _run_post_quality(ctx)
    return _finalize(ctx)

# المنطق: يستأنف من تنفيذ التنظيف المعتمد ثم مراحل الذيل، مستعيداً نتائج intake/
# quality المحفوظة دون إعادة حسابها. قرارات المستخدم تُحترم (مع تحذير للتجاوزات).
