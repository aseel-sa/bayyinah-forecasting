# Bayyina — Next Platform Roadmap

**Type:** technical + product roadmap for the next platform capabilities.
**Rule observed:** no production code was modified to produce this document — inspection and planning only.
**Companion document:** `TEAM_EXECUTION_PLAN.md` (team-facing version).

---

## 1. Current Platform Status

| Layer | File | What it enables today |
|---|---|---|
| Intake | `intake.py` (+`mapping_memory.py`, `llm_mapper.py`) | Understands any uploaded schema (AR/EN), maps date/product/demand with confidence + reasons, discovers 22 business-input meanings, remembers confirmed mappings across uploads, optional (disabled) LLM assist for ambiguous columns |
| Quality | `quality.py` | Business-aware risk detection in 4 categories (structural / time-series / business / representation), 3 severities + confidence, duplicate *investigation* (transactions vs ETL), approval-gated remediation with audit log, value-quality readiness scores |
| External Features | `external_features.py` | Platform-provided context (user uploads nothing): generic calendar, weather climatology CSV (+CDD/HDD), AHRI when industry=HVAC, lifecycle metadata incl. `future_availability` and `scenario_capable` |
| Feature Engineering | `feature_engineering.py` | Leakage-audited model-ready matrices per model family; temporal/lag/rolling/cyclical/trend/volatility features; data-driven seasonality; feature relevance analysis; `inventory_passthrough` |
| Forecasting Engine | `model.py` | Walk-forward validation (separate validation/forecast horizons, defaults 3 and 12/26), 9 model candidates + hybrid ensemble, graceful skip/fail handling, product + total + direct-total forecasts, `chart_data` (history/validation/forecast), metric cards, leaderboard, fast/balanced/full modes (100 products ≈ 0.9s in fast) |
| Streamlit UI | `app.py` | Working 5-page flow (upload → intake → quality review → features → results), human-in-the-loop quality approval. **Results page still consumes only legacy aliases** — the new model contract (chart_data, metric_cards, leaderboard, totals, warnings) is invisible to users |

Orchestration: `pipeline.py` — stages are a pluggable list; adding a layer is one entry + one function.

## 2. Requirements Coverage Matrix

| Requirement | Status | Notes |
|---|---|---|
| Upload historical datasets | **Done** | CSV upload + state reset |
| Preview data | **Partially Done** | Mapping/business-inputs shown; no raw-rows preview table |
| Validate dataset structure | **Done** | Intake roles + quality structural checks |
| Preprocess / quality checks | **Done** | 4-category assessment + approval-gated remediation + audit log |
| Generate future forecasts | **Done — Needs UI Exposure** | 12/26-period defaults, bounds, totals; UI shows only the basic product line |
| Compare forecasting models | **Done — Needs UI Exposure** | Full leaderboard (status, wMAPE, bias, selected) exists; UI shows a reduced legacy table |
| Evaluate forecasting accuracy | **Done — Exposed (2026-06-13)** | Walk-forward, 5 metrics; metric cards + actual-vs-predicted-vs-error view now rendered on the Forecast Performance page (`forecast_analysis.py` §1) |
| Visualize historical + forecast trends | **Partially Done** | History+forecast line on results page; validation view on performance page; total chart/bounds/start marker on page 5 still pending |
| Detect anomalies / unusual patterns | **Done — Needs UI Exposure** | Quality `unusual_demand_observations` (advisory, never auto-modified); issues list not rendered in UI |
| Generate AI-assisted insights | **Not Started** | §5 — note: the forecast-performance view originally scoped into insights now lives in `forecast_analysis.py` §1 |
| Simulate forecasting scenarios | **Done — phases A/B (2026-06-13)** | `forecast_analysis.py` §2 + What-if page: demand ±% (global/per-product), seasonal strength, peak shift, baseline-vs-scenario comparison, mandatory disclaimer. Phase C (feature-value/weather scenarios) pending |
| Executive-friendly summaries | **Not Started** | Seed exists: `forecast_summary` + `metric_cards` + plain-language strings throughout |

**Reading of the matrix (updated 2026-06-13):** the backend covers ~11/12 requirements; the visible product covers ~8/12. Remaining big wins: results-page rebuild on the new contract, then insights + executive summary.

## 3. Next Capability: Scenario Simulation / What-if Analysis

**What it means here:** the user asks "what if demand grows 15%? what if summer is hotter? what if the peak shifts a month?" and the platform returns an adjusted forecast next to the baseline with a business-impact delta — without silently pretending the scenario is a prediction.

**Why it matters for manufacturing:** capacity, procurement, and inventory decisions are made against ranges, not single lines. A planner who can stress-test demand before committing a production plan avoids both idle capacity and missed orders.

**Build first (phase order within the layer):**

- **A. Demand adjustment** — global or per-product ±X% (optionally ramping over the horizon). Pure post-forecast arithmetic. Trivial, safe, immediately demo-able.
- **B. Seasonal scenario** — amplify/dampen seasonal deviation around the forecast's mean level (multiply deviation-from-level by a factor), or shift the peak ±k periods (roll the seasonal component). Post-forecast arithmetic on the baseline curve.
- **C. Weather/temperature scenario** — "+2°C summer": only valid when temperature/CDD/HDD features exist AND the selected model actually consumed them (check `features_used` / relevance). Two tiers: (1) cheap — elasticity approximation using `feature_relevance_analysis` correlation; (2) accurate — re-run the tree-model prediction with scenario values injected through the existing `requires_future_values` fill hook (the climatology-fill mechanism in `model._predict_tree` was designed to accept scenario values).
- **D. External feature assumptions** — generalize C: any `requires_future_values`/`scenario_only` feature can be given a controlled future assumption; the layer records the assumption explicitly.
- **E. Operational scenarios (future phase — do NOT build yet):** supply disruption, capacity constraint, lead-time increase, inventory shortage. These need the inventory layer and cost inputs first.

**Do NOT build yet:** Monte-Carlo simulation, price-elasticity estimation from data, multi-scenario optimization, operational scenarios (E).

**Architecture:** ~~`scenario_engine.py`~~ **IMPLEMENTED 2026-06-13 as `forecast_analysis.py` Section 2** (owner decided to co-locate it with Forecast Performance Analysis in one file, sealed sections). Phases A and B are live and tested; C and D remain future work (C needs a public re-predict hook in model.py).

Responsibilities: take the baseline `model` output, apply declared assumptions, produce an adjusted forecast, compare, summarize impact. **Never retrain** for A/B/D-simple; re-predict (not retrain) only for C-accurate. Never mutate the baseline result. *(All three invariants are enforced by tests.)*

```python
run_scenario(model_result, scenario: dict, fe_output=None) -> {
    "baseline_forecast":  DataFrame,   # echo of product/total baseline
    "scenario_forecast":  DataFrame,   # same schema as product_forecasts/total
    "scenario_comparison": DataFrame,  # per date: baseline, scenario, delta, delta_pct
    "impact_summary": {                # business language
        "total_demand_delta", "total_demand_delta_pct",
        "peak_period_baseline", "peak_period_scenario",
        "most_affected_products": [...],
    },
    "scenario_assumptions": {...},     # exactly what the user declared
    "warnings": [...],                 # incl. mandatory "scenario ≠ prediction" label
}
```

**Simple-adjustment vs feature-rerun split:** A, B, D-as-declared-multipliers = post-forecast adjustments (no model involvement). C-accurate and any feature-value scenario = re-predict through the model layer with injected future feature values (plumbing already anticipated). The contract is identical either way; `scenario_assumptions.method` records which path ran.

## 4. Next Capability: Executive Summary

**File:** `executive_summary.py`. Consumes `model_result` + `quality_result` (+ `scenario_result` when present). Pure structured-data producer — no Streamlit dependency, no LLM dependency (it *prepares* an LLM payload; it does not call one).

Content (all derivable from existing outputs): forecast horizon and start; total expected demand; growth/decline vs trailing history (compare forecast total to the same-length trailing actual total); peak period; highest-risk products (volatile segments + wide bounds + low per-product validation accuracy); model confidence (`confidence_label` + portfolio wMAPE); accuracy summary in plain language (reuse metric cards); bias explanation tied to inventory consequence; operational implications; recommended actions (rule-based, e.g. "bias is −7% → review safety stock for top products").

```python
build_executive_summary(model_result, quality_result, scenario_result=None) -> {
    "headline": "...",                    # one sentence, numbers included
    "key_takeaways": [...],               # 3–5 bullets
    "forecast_highlights": [...],         # total, growth, peak, top products
    "risks": [...],                       # from quality issues + segments + bias
    "recommended_actions": [...],         # rule-based, conservative
    "confidence_explanation": "...",      # why the label is what it is
    "llm_prompt_payload": {...},          # structured numbers for optional narration
}
```

## 5. Next Capability: AI Insight Generation

**File:** `insights.py`. Two strictly separated tiers:

- **Deterministic insights (build first):** rule-based findings computed from existing outputs — demand trend direction (history slope + forecast level), peak month, model bias tendency, high-volatility products (`rolling_std`/segments), high-forecast-risk products (wide bounds / poor validation), total-demand growth, quality-risk caveats ("3 warnings may reduce confidence"), external-feature influence (from `feature_relevance` verdicts). Each insight: `{insight_type, severity, message, evidence (numbers), affected_items}`. Same inputs → same insights, testable, no API cost.
- **LLM narrative (optional, later):** a readability layer over the deterministic insights — never a source of facts.

```python
generate_insights(model_result, quality_result, fe_output=None, llm_client=None) -> {
    "deterministic_insights": [...],
    "llm_summary": "..." | None,
    "llm_prompt_payload": {...},
    "warnings": [...],
}
```

**Grounding rule (non-negotiable):** the LLM receives only structured platform outputs; it must not introduce any number not present in the payload; output failing a numeric-grounding check is discarded in favor of the deterministic text.

## 6. Next Capability: Forecast Explanation

Not a new engine — an assembly view over existing metadata, surfaced in UI + executive summary:

- **Why this model:** from the leaderboard — "seasonal_naive won for product X with wMAPE 14% vs random_forest 19%" (data already in `model_leaderboard.selected`).
- **Which metrics matter:** metric-card business_meaning fields (wMAPE for portfolios; bias for inventory direction).
- **What bias means:** existing card language — over-forecast = capital sitting in stock; under-forecast = stockout exposure.
- **What drives total demand:** top products by forecast share (`product_forecasts` aggregation).
- **External features used:** `features_used`-style info + `feature_relevance` verdicts ("temperature: strong relation, effect peaks after 1 period").
- **Confidence:** `confidence_label` + validation depth (n_folds) + history length caveats.
- **Cautions:** intermittent/short-history segments, skipped advanced models, climatology-not-live-weather, scenario-≠-prediction.

Target audience is non-technical: every item is one plain sentence with one number.

## 7. Next Capability: Inventory / Operational Risk Recommendations

**File:** `operational_risk.py` (preferred name — honest about scope). **Explicitly not replenishment optimization** — no EOQ, no safety-stock solver, no reorder points yet.

First useful outputs (all computable from existing data):
- under-forecast risk (negative bias + tight inventory if columns exist),
- over-forecast risk (positive bias → capital lock),
- demand-spike risk (volatility segment + seasonal peak proximity),
- high-volatility products list,
- products needing review (poor validation accuracy / short history / intermittent),
- possible stockout exposure ONLY when inventory columns exist (`inventory_passthrough` from feature engineering provides them),
- total-demand planning implication (capacity vs peak period).

Output: list of risk cards `{risk_type, severity, products, evidence, suggested_action}` + an honest `scope_note` that this is risk flagging, not optimization.

## 8. Streamlit Upgrade Plan

Principle: stop polishing incrementally; expose the contracts that already exist. Page map and priority:

| Priority | Page | What changes |
|---|---|---|
| **P0** | C. Forecast Dashboard | Rebuild results page on `chart_data`: total-demand chart ("ALL"), per-product chart with **history/validation/forecast as separate lines**, forecast-start marker, bounds band, horizon selector (re-run with `forecast_horizon`), top-products table from `product_forecasts` |
| **P0** | D. Model Comparison | `model_leaderboard` table (product + ALL scopes, skipped/failed rendered calmly with reasons), `metric_cards` as cards, one-line metric explanations from card fields |
| **P1** | B. Data Quality | Render `issues` list (severity icons, rationale, impacts) and `summary_for_user` — today only the plan buckets show |
| **P1** | A. Upload & Mapping | Raw-rows preview; mapping-correction dropdowns wired to existing `apply_user_corrections` / `confirm_mapping` (backend ready and tested) |
| **P2** | F. Executive Summary page | After §4 ships |
| ~~P2~~ **DONE 2026-06-13** | E. Scenario page | Shipped as page 7 "What-if Scenario": sliders (demand ±%, seasonal strength, peak shift) + baseline-vs-scenario chart + impact summary + prominent disclaimer |
| **DONE 2026-06-13** (unplanned addition) | G. Forecast Performance page | Page 6: actual vs predicted vs error per period (totals + per product), echoed metric cards, over/under counts — the "demand, prediction, and error" view |

The existing page-function + `STAGES` list architecture supports adding pages as one list entry + one function each — E and G were added exactly this way. The app UI is now fully English (2026-06-13).

## 9. Hugging Face / LLM Plan

**Where LLMs help (in order of value):** executive-summary narration (§4 payload), insight narratives (§5), metric/quality-warning explanation on demand, later a chatbot over forecast results (answers grounded in `model_result` keys only).

**Where LLMs are forbidden:** core forecasting, inventing insights or numbers, overriding model selection or validation results, silently changing data.

**Safe-LLM rule (verbatim, enforce in code):**
> The LLM summarizes structured forecast outputs. The LLM does not invent values. The LLM does not override model results.

Implementation guidance: same injection pattern as `llm_mapper.py` (callable client, disabled by default, strict output validation, graceful degradation to deterministic text). A numeric-grounding check (every number in the narrative must appear in the payload) gates display. Hugging Face specifically: a hosted small instruct model is sufficient for narration; nothing in the platform requires fine-tuning. Do not add the dependency before Phase 4.

## 10. Recommended Build Order

- **Phase 1 — UI exposure (highest ROI):** Forecast Dashboard + Model Comparison on the new contract (total forecast, chart_data lines, leaderboard, calm warnings). *Exit:* a stakeholder sees the full forecasting story without reading code. **Partially done 2026-06-13:** metric cards + validation view live on the performance page; the results-page rebuild (chart_data lines, leaderboard, total chart) remains.
- **Phase 2 — explanation layers:** `insights.py` (deterministic; performance view no longer in its scope) + `executive_summary.py` + Executive Summary page.
- ~~Phase 3 — scenarios~~ **DONE 2026-06-13 (out of order, by owner approval):** demand ±% and seasonal scenarios + baseline-vs-scenario UI, as `forecast_analysis.py` §2. Weather/feature tier (phase C) deferred.
- **Phase 4 — risk + narration:** `operational_risk.py`; optional LLM narration for summary/insights.
- **Phase 5 — stretch:** chatbot over results; weather-scenario accurate tier; Hugging Face integration if time allows.

Parallel/continuous: ~~install statsmodels/prophet/statsforecast and validate~~ **DONE 2026-06-13** (all 9 model families verified end-to-end); ~~seed an automated test suite~~ **SEEDED 2026-06-13** (`test_forecast_analysis.py`, 32 tests) — extend to the five original layers.

## 11. Risks and Tradeoffs (honest)

- The UI currently hides most backend capability — until Phase 1, demos under-represent the platform.
- Scenario simulation can mislead if presented as prediction — every scenario output must carry an explicit "assumption, not forecast" label (built into the contract's warnings).
- LLM summaries must be grounded; an ungrounded narrative is worse than no narrative.
- Inventory/stockout outputs depend on clients actually providing inventory columns; degrade to demand-side risks otherwise.
- Weather climatology is internal demo data, not live weather; weather scenarios inherit this limitation.
- SARIMA/Prophet/Croston are skipped in the current environment (packages not installed) — they have not been exercised end-to-end here.
- Fast mode's documented static-feature compromise (seasonal flags derived from full history) trades purity for 1000-product scale.
- Executive summaries must not overclaim certainty: confidence labels are engineering thresholds, and residual-based bounds are approximations (summed total bounds overstate).

## 12. Final Recommended Architecture

```
intake.py               # schema understanding, mapping memory, optional LLM assist
quality.py              # value-risk assessment + approval-gated remediation
external_features.py    # platform-provided context + lifecycle metadata
feature_engineering.py  # leakage-audited model-ready features + relevance (+ coverage guard)
model.py                # Forecasting Engine: validation, selection, forecasts, totals
forecast_analysis.py    # SHIPPED 2026-06-13 — §1 measured performance view; §2 what-if
                        #   scenarios (was planned as scenario_engine.py; owner chose one
                        #   file, two sealed sections)
insights.py             # NEW — deterministic findings (+ optional LLM narrative)
executive_summary.py    # NEW — structured executive output (+ LLM payload)
operational_risk.py     # NEW — risk flags (not replenishment optimization)
app.py                  # Streamlit (English): pages over contracts; zero business logic
pipeline.py             # orchestration; forecast_performance joined _POST_QUALITY_STAGES
test_forecast_analysis.py  # SHIPPED 2026-06-13 — first automated test suite (32 tests)
```

Each new file follows the established platform rules: single responsibility, documented output contract, English dev notes, graceful degradation, and no silent decisions. (User-facing strings are now English — owner decision 2026-06-13; the original Arabic-UI rule is retired.)

---

## 13. Update Log

- **2026-06-13:** Scenario simulation (phases A/B) and forecast-performance analysis shipped in `forecast_analysis.py` with 32 tests; pipeline gained the non-fatal `forecast_performance` stage + `performance_result` key; app gained pages 6–7 and was fully translated to English; all model packages installed and every model family verified end-to-end; FE coverage guard fixed the all-NaN external-column bug for data outside the weather CSV's 2018–2027 window; demo datasets added (`data/sample_demand_data.csv` synthetic, `data/australian_wine_demand.csv` real ABS wine data 1980–1994). Details: `TECHNICAL_HANDOFF.md` §13.
