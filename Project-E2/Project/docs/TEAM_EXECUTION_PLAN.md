# Bayyina — Team Execution Plan

*Practical plan: what to build next, in what order, and why. Companion to `NEXT_PLATFORM_ROADMAP.md` (full technical detail).*

---

## 1. Where We Are Now

The backend foundation is built and tested: five layers (intake → quality → external features → feature engineering → forecasting engine) connected by a pipeline with documented contracts. The forecasting engine already produces 12-month product forecasts, a total-demand forecast, model comparison with honest backtesting, business-language metric cards, and graceful handling of failures.

**The gap is visibility, not capability:** the Streamlit results page still shows the old minimal outputs. Most of what the platform can do is invisible in a demo.

## 2. What We Should NOT Do Next

- ❌ **Don't keep polishing Streamlit randomly** — upgrade it against the new output contract, page by page, by priority.
- ❌ **Don't add Hugging Face / LLM features before forecast results are clearly displayed** — narration over invisible numbers helps nobody.
- ❌ **Don't add LSTM for appearance** — its slot is reserved in the leaderboard as "future extension"; adding it now adds runtime risk, not value.
- ❌ **Don't build scenario simulation before the baseline forecast is visible** — a what-if comparison needs a clearly displayed "what-is" first.
- ❌ **Don't let any LLM generate claims that aren't in the platform outputs** — the rule is: LLM summarizes structured outputs, never invents values, never overrides models.

## 3. What We Should Build Next (priority order)

**A. Forecast Results UI (first — highest return)**
- Total demand chart, product chart with history / validation / forecast as separate lines, forecast-start marker
- Model leaderboard (skipped/failed models shown calmly with reasons)
- Metric cards (wMAPE, bias, MAE… already written in business language)
- Warnings panel

**B. Executive Summary Layer** (`executive_summary.py`)
- Structured summary first (headline, takeaways, risks, recommended actions) — all numbers already exist in the model output
- LLM-narrated version optional, later

**C. Deterministic Insights** (`insights.py`)
- Trend direction, peak month, bias tendency, high-volatility products, top products, quality caveats
- Rule-based, testable, zero API cost — LLM narrative comes after

**D. Scenario Simulation** — ✅ **DONE 2026-06-13** (shipped early, with owner approval, as `forecast_analysis.py` §2 — not a separate `scenario_engine.py`)
- Demand ±X% and seasonal strength/shift (pure post-forecast math) — live and tested
- Baseline vs scenario chart + impact summary — live (app page 7)
- Every scenario output labeled "assumption, not prediction" — enforced by contract + test
- Bonus shipped with it: **Forecast Performance page** (actual vs predicted vs error per validation period — app page 6)

**E. Operational Risk / Inventory hints** (`operational_risk.py`)
- Only after forecasts and scenarios are visible
- Risk flags (stockout exposure, over/under-forecast risk) — not full inventory optimization

## 4. Team Task Breakdown

| Role | Owns | First deliverable |
|---|---|---|
| **Streamlit/UI** | Pages over the new contracts | Forecast Dashboard on `chart_data` + leaderboard + metric cards |
| **Backend integration** | Pipeline wiring, environment | ✅ DONE 2026-06-13: all model packages installed, every family verified end-to-end; `forecast_performance` wired into the stage list (non-fatal) |
| **Insights / executive summary** | `insights.py`, `executive_summary.py` | Deterministic insights list + structured summary consumed by a new page |
| **Scenario simulation** | ~~`scenario_engine.py`~~ `forecast_analysis.py` §2 | ✅ DONE 2026-06-13: demand ±% / seasonal strength / peak shift with baseline-vs-scenario comparison |
| **Testing / demo** | Test suite + demo script | ✅ SEEDED 2026-06-13: `test_forecast_analysis.py` (32 tests). Next: extend contract-freezing tests to the five original layers; rehearse the demo narrative below |

Dependencies: UI work (A) blocks nothing and unblocks everyone's demos — start it immediately. B and C can proceed in parallel with A. D starts after A ships. E starts last.

## 5. Demo Narrative

> "Bayyina helps manufacturers upload forecasting data, understand data quality, enrich the forecast with external context, compare multiple models, forecast both product-level and total demand, explain model performance in business language, and simulate what-if scenarios for planning decisions. Watch: I upload a messy sales file — the platform tells me what it understood and asks only where it's unsure. It reviews the data like an analyst and recommends fixes I approve. Then it races several models in honest backtests — here a simple seasonal model beat the ML model, and the platform says so. I get a 12-month forecast per product, a total demand line for capacity planning, and a scorecard in plain language: weighted error 17%, slightly under-forecasting — that means stockout risk. Then I ask: what if demand grows 15%? — and it shows me the impact next to the baseline."

## 6. Success Criteria

The platform is demo-ready when *(status as of 2026-06-13)*:

- [x] A user can upload data and reach a forecast without touching code
- [x] The quality report appears with issues and recommendations *(plan buckets; full issues list still pending)*
- [ ] The forecast chart clearly separates history, validation, and future *(history+forecast on page 5; validation view lives on page 6 — single combined chart still pending)*
- [ ] The total demand forecast appears *(computed; not yet charted on page 5)*
- [ ] The forecast horizon is configurable from the UI
- [ ] Model comparison (leaderboard) appears, including skipped/failed models calmly *(page 5 still shows the legacy reduced table)*
- [x] Metrics are explained in business language (metric cards — shown on the Forecast Performance page)
- [x] The scenario page compares baseline vs scenario with an impact summary
- [ ] An executive summary is generated
- [ ] Every team member can explain why each layer exists in one sentence

## 7. Final Recommendation

**Do not rebuild the app. Expose what's already built.** The backend contracts are designed, tested, and waiting — the fastest path to a convincing platform is making them visible (Forecast Dashboard + leaderboard + metric cards), then adding the explanation layers (insights, executive summary), then scenarios. Everything else — LLM narration, chatbot, deep learning — comes after the forecasting story is visible and trusted.

---

## 8. Update Log

- **2026-06-13:** Item D (scenario simulation) shipped early with owner approval, together with a Forecast Performance page, both in `forecast_analysis.py` (one file, two sealed sections) — 32 automated tests, non-fatal pipeline stage, two new app pages. All model packages installed; SARIMA/Prophet/Croston/XGBoost verified end-to-end. App fully translated to English. FE coverage-guard bug fix (all-NaN external columns no longer kill tree models on historical data). Demo datasets added: `data/sample_demand_data.csv` (synthetic) and `data/australian_wine_demand.csv` (real, 1980–1994). Remaining top priority is unchanged: rebuild the results page (A) on the new contract, then B (executive summary) and C (insights — performance view no longer in its scope). Full detail: `TECHNICAL_HANDOFF.md` §13.
