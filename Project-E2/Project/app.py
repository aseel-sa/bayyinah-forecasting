"""
app.py — Bayyina test UI | Streamlit

A simple UI to run the pipeline end-to-end on real data:
  Upload → Intake → Quality Review → Feature Engineering → Model Results
  → Forecast Performance → What-if Scenario

Run:  streamlit run app.py

Governing principle: NO business logic here — the UI only calls pipeline /
layer public entry points and renders their outputs. Every semantic decision
belongs to its layer (intake/quality/feature_engineering/model/forecast_analysis).

Extensibility: each page is a standalone function and STAGES is an ordered
list — adding a future page (insights/chatbot/inventory) = one STAGES entry
+ one page_* function. No restructuring.
"""

import io

import pandas as pd
import streamlit as st

import forecast_analysis  # what-if simulation on a completed result (public API, no logic here)
import pipeline
import quality  # for the execution-log-to-CSV helper (display formatting, no logic)

PRIMARY = "#1a4d5e"

# Page sequence (extensible: add an entry here + a page_* function).
STAGES = [
    ("upload",      "1 · Upload & Setup"),
    ("intake",      "2 · Intake Overview"),
    ("quality",     "3 · Quality Review"),
    ("features",    "4 · Feature Engineering"),
    ("results",     "5 · Model Results"),
    ("performance", "6 · Forecast Performance"),
    ("scenario",    "7 · What-if Scenario"),
    ("insights",   "8 · Insights"),
    # ("chat",     "9 · Assistant"),     ← future insertion
]
_STAGE_ORDER = [s[0] for s in STAGES]


# ─── State management ─────────────────────────────────────────────────────────

def _init_state() -> None:
    """Initialises session_state keys once. Does NOT run the pipeline."""
    st.session_state.setdefault("stage", "upload")
    st.session_state.setdefault("result", None)
    st.session_state.setdefault("upload_token", None)
    st.session_state.setdefault("completed", set())
    st.session_state.setdefault("scenario_result", None)


def _reset_state() -> None:
    """Clears pipeline state on a new upload (fresh run)."""
    st.session_state.result = None
    st.session_state.completed = set()
    st.session_state.stage = "upload"
    st.session_state.scenario_result = None


def _goto(stage: str) -> None:
    """Navigates to a page and marks the current one completed."""
    st.session_state.completed = set(st.session_state.completed) | {st.session_state.stage}
    st.session_state.stage = stage
    st.rerun()


# ─── Shared display helpers ───────────────────────────────────────────────────

def _sidebar() -> None:
    """Renders the stage tracker with checkmarks for completed pages."""
    st.sidebar.markdown(f"### <span style='color:{PRIMARY}'>Bayyina</span>",
                        unsafe_allow_html=True)
    st.sidebar.caption("Demand forecasting engine — test UI")
    st.sidebar.divider()
    current = st.session_state.stage
    done = st.session_state.completed
    for key, label in STAGES:
        if key in done:
            st.sidebar.markdown(f"✅ {label}")
        elif key == current:
            st.sidebar.markdown(f"🔵 **{label}**")
        else:
            st.sidebar.markdown(f"⚪ {label}")


def _show_errors(result: dict) -> bool:
    """
    If the pipeline failed, shows the failed stage + clean error summary (no raw
    stack traces). Returns True if a failure was shown.
    """
    if not result or result.get("status") != "failed":
        return False
    st.error("Pipeline halted — processing could not be completed.")
    for err in result.get("errors", []):
        st.markdown(f"**Stage:** `{err.get('stage', '?')}`")
        st.markdown(f"**Reason:** {err.get('message', 'unspecified')}")
    # Failed-stage hint from the log (no trace)
    for s in result.get("stage_log", []):
        if s.get("status") in ("error", "halted"):
            st.caption(f"Last stage: {s['stage']} ({s['status']}, {s.get('duration_s')}s)")
    if st.button("🔄 Start over"):
        _reset_state()
        st.rerun()
    return True


def _readiness_row(scores: dict) -> None:
    """Renders readiness scores as metric columns. scores: {key→{score, ...}}."""
    if not scores:
        return
    cols = st.columns(len(scores))
    for col, (key, val) in zip(cols, scores.items()):
        label = key.replace("_", " ").title()
        col.metric(label, f"{val.get('score', 0)}%")


# ─── Page 1: Upload & Setup ───────────────────────────────────────────────────

def page_upload() -> None:
    """Upload & configure page: CSV upload, granularity, sector, run button."""
    st.header("Upload Data & Setup")
    st.write("Upload a sales CSV (date, product, quantity, plus any extra columns).")

    file = st.file_uploader("CSV file", type=["csv"])
    c1, c2 = st.columns(2)
    granularity = c1.selectbox("Time granularity", ["monthly", "weekly"],
                               format_func=lambda x: {"monthly": "Monthly", "weekly": "Weekly"}[x])
    sector = c2.selectbox("Sector (safety net for short history)", ["none", "hvac"])

    # New upload detected → reset state
    if file is not None:
        token = f"{file.name}:{file.size}"
        if token != st.session_state.upload_token:
            st.session_state.upload_token = token
            _reset_state()

    if file is not None and st.button("▶️ Run pipeline", type="primary"):
        try:
            df = pd.read_csv(file)
        except Exception as exc:
            st.error(f"Could not read the CSV: {exc}")
            return
        config = {
            "granularity": granularity,
            "sector": None if sector == "none" else sector,
            "auto_approve_quality": False,   # pause at quality for human review
        }
        with st.spinner("Running intake and quality assessment…"):
            result = pipeline.run_pipeline(df, config)
        st.session_state.result = result
        if result.get("status") == "failed":
            st.rerun()
        else:
            _goto("intake")

    if st.session_state.result and _show_errors(st.session_state.result):
        return


# ─── Page 2: Intake Overview ──────────────────────────────────────────────────

def page_intake() -> None:
    """Intake overview: detected columns, business inputs, capabilities, readiness."""
    result = st.session_state.result
    if _show_errors(result):
        return
    ir = result["intake_result"]
    st.header("Intake Overview")

    mapping = ir.get("proposed_mapping", {})
    c1, c2, c3 = st.columns(3)
    c1.metric("Date column", mapping.get("date") or "—")
    c2.metric("Product column", mapping.get("product") or "—")
    c3.metric("Demand column", mapping.get("qty") or "—")

    st.subheader("Detected business inputs")
    bis = ir.get("business_inputs", []) or []
    shown = [b for b in bis if b.get("detected_meaning") != "unknown"]
    if shown:
        st.dataframe(pd.DataFrame([{
            "Column": b["column"],
            "Meaning": b["detected_meaning"].replace("_", " "),
            "Confidence": b["confidence"],
        } for b in shown]), use_container_width=True, hide_index=True)
    else:
        st.caption("No additional columns detected.")

    st.subheader("Unlocked capabilities")
    caps = ir.get("capabilities", {})
    unlocked = [c.replace("_", " ").title() for c, info in caps.items() if info.get("unlocked")]
    if unlocked:
        st.markdown("  ".join(f"✅ {u}" for u in unlocked))
    else:
        st.caption("No capabilities unlocked.")

    st.subheader("Readiness scores (column availability)")
    _readiness_row(ir.get("readiness", {}))

    st.divider()
    if st.button("Continue to quality review →", type="primary"):
        _goto("quality")


# ─── Page 3: Quality Review (Human-in-the-Loop) ───────────────────────────────

def page_quality() -> None:
    """Quality review: 3 tabs (auto / review-required / skipped) + approve & continue."""
    result = st.session_state.result
    if _show_errors(result):
        return
    qr = result["quality_result"]
    plan = qr.get("remediation_plan", {})
    st.header("Quality Review")
    st.caption("Review the detected issues and approve remediation actions. "
               "Nothing is applied without your approval.")

    st.subheader("Readiness scores (value quality)")
    _readiness_row(qr.get("readiness_scores", {}))
    st.divider()

    tab_auto, tab_review, tab_skip = st.tabs(
        ["✅ Automatic (safe)", "⚠️ Needs approval", "⛔ Skipped"])

    with tab_auto:
        auto = plan.get("auto_actions", [])
        st.caption("Safe corrections applied automatically (for your information).")
        if auto:
            st.dataframe(pd.DataFrame([{
                "Action": a.get("label", a["action"]),
                "Records affected": a.get("estimated_records_affected", 0),
            } for a in auto]), use_container_width=True, hide_index=True)
        else:
            st.caption("No automatic actions.")

    approved_review = []
    with tab_review:
        review = plan.get("review_required", [])
        st.caption("Each action changes values — approve or reject it.")
        if review:
            for i, a in enumerate(review):
                with st.container(border=True):
                    st.markdown(f"**{a.get('label', a['action'])}**")
                    st.caption(a.get("reason_for_review", ""))
                    st.caption(f"Records affected (estimate): {a.get('estimated_records_affected', 0)}")
                    if st.toggle("Approve", value=True, key=f"rev_{i}"):
                        approved_review.append(a["action"])
        else:
            st.caption("No actions need approval.")

    with tab_skip:
        skipped = plan.get("skipped_actions", [])
        st.caption("Actions never applied automatically (business-data changes or advisory).")
        if skipped:
            st.dataframe(pd.DataFrame([{
                "Action": a.get("label", a["action"]),
                "Reason": a.get("reason_skipped", ""),
            } for a in skipped]), use_container_width=True, hide_index=True)
        else:
            st.caption("No skipped actions.")

    st.divider()
    if st.button("✔️ Approve & continue", type="primary"):
        auto_actions = [a["action"] for a in plan.get("auto_actions", [])]
        approved = auto_actions + approved_review
        with st.spinner("Executing cleaning, building features, training models… "
                        "this may take a minute."):
            new_result = pipeline.resume_after_quality_review(
                result, {"approved_actions": approved})
        st.session_state.result = new_result
        if new_result.get("status") == "failed":
            st.rerun()
        else:
            _goto("features")


# ─── Page 4: Feature Engineering Summary ──────────────────────────────────────

def page_features() -> None:
    """Feature engineering summary: metadata by source, excluded, seasonal, leakage."""
    result = st.session_state.result
    if _show_errors(result):
        return
    fr = result["feature_result"]
    st.header("Feature Engineering Summary")

    audit = fr.get("leakage_audit", {})
    if audit.get("all_checks_passed"):
        st.success("✅ Leakage audit: all checks passed.")
    else:
        st.error("❌ Leakage audit: a check failed.")
    with st.expander("Leakage check details"):
        for c in audit.get("checks", []):
            mark = "✅" if c.get("passed") else "❌"
            st.markdown(f"{mark} `{c['check']}` — {c.get('detail', '')}")

    st.subheader("Features by source")
    meta = fr.get("feature_metadata", [])
    if meta:
        by_source: dict = {}
        for m in meta:
            by_source.setdefault(m["source"], []).append(m["feature_name"])
        cols = st.columns(len(by_source))
        for col, (src, feats) in zip(cols, by_source.items()):
            col.markdown(f"**{src}**")
            for f in feats:
                col.caption(f"• {f}")
    else:
        st.caption("No features.")

    st.subheader("Excluded columns")
    excluded = fr.get("excluded_columns", [])
    if excluded:
        st.dataframe(pd.DataFrame(excluded).rename(
            columns={"column": "Column", "reason": "Reason"}),
            use_container_width=True, hide_index=True)
    else:
        st.caption("No excluded columns.")

    st.subheader("Seasonal detection per product")
    seasonal = fr.get("seasonal_detection", {}).get("per_product", {})
    if seasonal:
        st.dataframe(pd.DataFrame([{
            "Product": p, "Pattern": s["pattern_type"],
            "Peak months": ", ".join(map(str, s.get("peak_months", []))) or "—",
            "Source": s["detection_source"],
        } for p, s in seasonal.items()]), use_container_width=True, hide_index=True)
    else:
        st.caption("No seasonal detection.")

    st.divider()
    if st.button("Continue to modeling →", type="primary"):
        _goto("results")


# ─── Page 5: Model Results ────────────────────────────────────────────────────

def page_results() -> None:
    """Model results: comparison table, forecast plot, per-product selector, downloads."""
    result = st.session_state.result
    if _show_errors(result):
        return
    mr = result["model_result"]
    st.header("Model Results")

    best = mr.get("best_model", {})
    comparison = mr.get("comparison_table", pd.DataFrame())
    forecasts = mr.get("forecasts", {})
    history = mr.get("history", {})
    products = list(forecasts.keys())

    if not products:
        st.warning("No forecasts available.")
        return

    product = st.selectbox("Choose a product", products,
                           format_func=lambda p: f"{p}  (best: {best.get(p, '—')})")

    # Comparison table for the selected product, best model highlighted
    st.subheader("Model comparison")
    if not comparison.empty:
        sub = comparison[comparison["product"].astype(str) == str(product)].copy()
        show = sub[["model", "mape", "rmse", "mae", "n_folds"]].rename(columns={
            "model": "Model", "mape": "MAPE", "rmse": "RMSE", "mae": "MAE",
            "n_folds": "Folds"})

        best_family = best.get(product)

        def _hl(row):
            ok = row["Model"] == best_family
            return [f"background-color: {PRIMARY}; color: white" if ok else "" for _ in row]

        st.dataframe(show.style.apply(_hl, axis=1).format(
            {"MAPE": "{:.1%}", "RMSE": "{:.1f}", "MAE": "{:.1f}"}, na_rep="∞"),
            use_container_width=True, hide_index=True)
        st.caption(f"Best model for this product: **{best_family}**")

    # Forecast plot: actual history + future horizon
    st.subheader("Forecast")
    chart_df = pd.DataFrame()
    if product in history:
        chart_df = pd.DataFrame({"actual": history[product]})
    if product in forecasts:
        fc = pd.DataFrame({"forecast": forecasts[product]})
        chart_df = chart_df.join(fc, how="outer") if not chart_df.empty else fc
    if not chart_df.empty:
        st.line_chart(chart_df)
    st.caption("The actual line is historical demand; the forecast line is the future horizon.")

    # Downloads
    st.divider()
    st.subheader("Downloads")
    c1, c2 = st.columns(2)

    fc_rows = []
    for p, s in forecasts.items():
        for d, v in s.items():
            fc_rows.append({"product": p, "date": d, "forecast": float(v)})
    fc_csv = pd.DataFrame(fc_rows).to_csv(index=False).encode("utf-8")
    c1.download_button("⬇️ Download forecasts (CSV)", fc_csv,
                       "bayyina_forecasts.csv", "text/csv")

    exec_log = result.get("quality_result", {}).get("execution_log", {})
    log_csv = quality.execution_log_to_dataframe(exec_log).to_csv(index=False).encode("utf-8")
    c2.download_button("⬇️ Download execution log (CSV)", log_csv,
                       "bayyina_execution_log.csv", "text/csv")

    st.divider()
    if st.button("Continue to forecast performance →", type="primary"):
        _goto("performance")


# ─── Page 6: Forecast Performance (MEASURED — actual vs predicted) ────────────

def page_performance() -> None:
    """Forecast performance (MEASURED): actual vs predicted vs error from the
    walk-forward validation — displays performance_result as-is (no logic)."""
    result = st.session_state.result
    if _show_errors(result):
        return
    perf = result.get("performance_result") or {}
    st.header("Forecast Performance Analysis")
    st.caption("Historical measurement from walk-forward validation: actual vs "
               "predicted and the error per period. This is **measured** accuracy — "
               "not assumption simulation (that lives on the What-if page).")

    pp = perf.get("per_period")
    if pp is None or len(pp) == 0:
        for w in perf.get("warnings", []):
            st.info(w.get("message", ""))
        st.caption("No validation data available to display performance (insufficient history).")
    else:
        summary = perf.get("summary", {})

        cards = summary.get("metric_cards", [])
        if cards:
            st.subheader("Accuracy metrics (from the model layer)")
            cols = st.columns(len(cards))
            for col, card in zip(cols, cards):
                col.metric(card["metric"], card.get("display_value", "—"))
                col.caption(card.get("label", ""))

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Over-forecast periods", summary.get("n_over_forecast", 0))
        c2.metric("Under-forecast periods", summary.get("n_under_forecast", 0))
        c3.metric("Exact matches", summary.get("n_exact", 0))
        c4.metric("Overall confidence", summary.get("confidence_label") or "—")

        st.subheader("Total: actual vs predicted (validation periods)")
        tot = perf["per_period_total"].set_index("date")
        st.line_chart(tot[["actual", "predicted"]])
        st.bar_chart(tot[["error"]])
        st.caption("Error = predicted − actual (positive = over-forecast, negative = under-forecast).")

        st.subheader("By product")
        prods = sorted(pp["product"].astype(str).unique())
        prod = st.selectbox("Choose a product", prods, key="perf_product")
        sub = pp[pp["product"].astype(str) == prod].sort_values("date").set_index("date")
        st.line_chart(sub[["actual", "predicted"]])
        st.bar_chart(sub[["error"]])

        with st.expander("Full detail table"):
            st.dataframe(pp.rename(columns={
                "product": "Product", "date": "Date", "model_name": "Model",
                "actual": "Actual", "predicted": "Predicted", "error": "Error",
                "error_pct": "Error %", "direction": "Direction"}),
                use_container_width=True, hide_index=True)
        st.download_button("⬇️ Download performance analysis (CSV)",
                           pp.to_csv(index=False).encode("utf-8"),
                           "bayyina_forecast_performance.csv", "text/csv")

    st.divider()
    if st.button("Continue to what-if simulation →", type="primary"):
        _goto("scenario")


# ─── Page 7: What-if Scenario (ASSUMPTION-based) ──────────────────────────────

def page_scenario() -> None:
    """What-if simulation (ASSUMPTION): collects declared assumptions, calls
    forecast_analysis.run_scenario on the completed result, displays its
    contract. The baseline is never modified (the layer guarantees it)."""
    result = st.session_state.result
    if _show_errors(result):
        return
    mr = result["model_result"]
    st.header("What-if Scenario Simulation")
    st.warning("⚠️ Results on this page are **hypothetical**: an arithmetic "
               "adjustment of the baseline forecast under assumptions you declare — "
               "not a confirmed prediction, and the baseline forecast is never changed. "
               "This differs from the Forecast Performance page, which shows measured accuracy.")

    pf = mr.get("product_forecasts")
    if pf is None or len(pf) == 0:
        st.info("No baseline forecast available to build a scenario.")
        return

    c1, c2, c3 = st.columns(3)
    demand = c1.slider("Demand change % (+ increase / − decrease)", -50, 100, 0, 5)
    strength = c2.slider("Seasonal strength (1.0 = unchanged)", 0.0, 2.5, 1.0, 0.1)
    shift = c3.slider("Peak shift (periods)", -6, 6, 0, 1)

    if st.button("▶️ Run scenario", type="primary"):
        st.session_state.scenario_result = forecast_analysis.run_scenario(
            mr, {"demand_change_pct": demand,
                 "seasonal_strength_factor": strength,
                 "peak_shift_periods": shift})

    res = st.session_state.scenario_result
    if not res:
        st.caption("Set the assumptions above, then run the scenario.")
        return

    # Layer warnings: the mandatory disclaimer prominently, the rest by level
    for w in res.get("warnings", []):
        if w.get("affected_item") == "disclaimer":
            st.warning(f"⚠️ {w.get('message', '')}")
        elif w.get("level") == "warning":
            st.info(w.get("message", ""))
        else:
            st.caption(f"ℹ️ {w.get('message', '')}")

    imp = res["impact_summary"]
    delta_pct = imp.get("total_demand_delta_pct")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Baseline total", f"{imp['total_demand_baseline']:,.0f}")
    c2.metric("Scenario total", f"{imp['total_demand_scenario']:,.0f}",
              delta=f"{delta_pct:+.1f}%" if delta_pct is not None else None)
    pk_b, pk_s = imp.get("peak_period_baseline"), imp.get("peak_period_scenario")
    c3.metric("Baseline peak", pk_b.strftime("%Y-%m") if pk_b is not None else "—")
    c4.metric("Scenario peak", pk_s.strftime("%Y-%m") if pk_s is not None else "—")

    st.subheader("Baseline vs scenario (total demand)")
    comp = res["scenario_comparison"].set_index("date")
    st.line_chart(comp[["baseline_total", "scenario_total"]].rename(
        columns={"baseline_total": "baseline", "scenario_total": "scenario"}))

    st.subheader("Most affected products")
    affected = imp.get("most_affected_products", [])
    if affected:
        st.dataframe(pd.DataFrame(affected).rename(columns={
            "product": "Product", "baseline_total": "Baseline total",
            "scenario_total": "Scenario total", "delta": "Delta",
            "delta_pct": "Delta %"}), use_container_width=True, hide_index=True)

    with st.expander("Declared assumptions (as applied)"):
        st.json(res.get("scenario_assumptions", {}))

    st.divider()
    c1, c2 = st.columns(2)
    c1.download_button("⬇️ Download scenario comparison (CSV)",
                       res["scenario_comparison"].to_csv(index=False).encode("utf-8"),
                       "bayyina_scenario_comparison.csv", "text/csv")
    c2.download_button("⬇️ Download scenario forecast (CSV)",
                       res["scenario_forecast"].to_csv(index=False).encode("utf-8"),
                       "bayyina_scenario_forecast.csv", "text/csv")


def page_insights() -> None:
    """Decision insights — DISPLAY ONLY. All logic lives in insights.py; this
    page just renders the returned structure (no business logic in app.py)."""
    result = st.session_state.result
    if _show_errors(result):
        return
    ins = result.get("insights") or {}
    st.header("Decision Insights")
    if not ins or ins.get("status") == "failed":
        st.info(ins.get("executive_summary",
                        "Insights are not available for this run; the forecast itself is unaffected."))
        return
    if ins.get("status") == "partial":
        st.caption("Partial insights — some validation signals were limited.")

    st.subheader("Executive Summary")
    st.write(ins.get("executive_summary", ""))

    conf = ins.get("model_confidence") or {}
    if conf.get("level"):
        st.metric("Model confidence", conf.get("level"))
        for r in conf.get("reasons", []):
            st.caption("• " + str(r))

    findings = ins.get("key_findings") or []
    if findings:
        st.subheader("Key Findings")
        _sev = {"high": "🔴", "medium": "🟠", "low": "🟡"}
        for f in findings:
            st.markdown(f"{_sev.get(f.get('severity'), '•')} **{f.get('title','')}** — "
                        f"{f.get('description','')}")

    hr = ins.get("high_risk_products") or []
    if hr:
        st.subheader("Products to Review")
        st.dataframe(pd.DataFrame([{
            "Product": r.get("product"), "Risk": r.get("risk_level"),
            "Drivers": "; ".join(r.get("risk_drivers", [])),
            "Suggested action": r.get("recommended_action"),
        } for r in hr]), use_container_width=True, hide_index=True)

    actions = ins.get("recommended_actions") or []
    if actions:
        st.subheader("Recommended Actions")
        _pri = {"high": "🔴", "medium": "🟠", "low": "🟢"}
        for a in actions:
            st.markdown(f"{_pri.get(a.get('priority'), '•')} **{a.get('action','')}**")
            st.caption(a.get("reason", ""))

    lims = ins.get("data_limitations") or []
    if lims:
        with st.expander("Data limitations & notes"):
            for l in lims:
                st.caption("• " + str(l))
        st.caption("Technical explainability status: "
                   + str((ins.get("explainability") or {}).get("status", "not_available")))


# ─── Main routing ─────────────────────────────────────────────────────────────

_PAGES = {
    "upload": page_upload, "intake": page_intake, "quality": page_quality,
    "features": page_features, "results": page_results,
    "performance": page_performance, "scenario": page_scenario,
    "insights": page_insights,
}


def main() -> None:
    """Entry point: config, sidebar tracker, route to the current page function."""
    st.set_page_config(page_title="Bayyina — Demand Forecasting", layout="wide")
    # Light primary-color touch for headers (minimal CSS)
    st.markdown(f"<style>h1,h2,h3{{color:{PRIMARY};}}</style>", unsafe_allow_html=True)
    _init_state()
    _sidebar()
    _PAGES.get(st.session_state.stage, page_upload)()


if __name__ == "__main__":
    main()
