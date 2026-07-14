# Bayyina — Team Report

*A plain-language summary of what was built, why, and what comes next.*

---

## 1. What We Built

Bayyina is a platform foundation for AI-assisted demand forecasting in manufacturing. A planner uploads raw sales data — any column names, Arabic or English — and the platform understands the file, checks it for risks, enriches it with external context, builds forecasting features safely, compares several forecasting models honestly, and delivers product-level and total-demand forecasts with accuracy explained in business language.

It is not a single model or a dashboard. It is seven cooperating layers, each with one job and a clear contract, so the platform can grow (inventory advice, executive summaries, insights) without rewrites — the newest layer (forecast performance + what-if scenarios) plugged in exactly this way, with zero changes to the forecasting core.

## 2. Why It Matters

Forecast quality drives money decisions:

- **Inventory planning** — over-forecasting fills warehouses with capital that sits still.
- **Production planning** — under-forecasting causes missed orders and rushed production.
- **Procurement** — lead-time purchasing depends on knowing demand months ahead.
- **Supply-chain risk** — a total-demand view exposes capacity crunches before they happen.
- **Cost control** — both overproduction and underproduction are expensive; bias direction tells you which one you're paying for.
- **Executive decisions** — leadership needs one trustworthy number for total demand, not forty spreadsheets.

## 3. Platform Workflow

```
Upload data → Understand columns → Check quality → Add external context
→ Build forecasting features → Compare models → Generate forecast
→ Explain results → Support decisions
```

A human stays in the loop at the two moments that matter: confirming what the columns mean, and approving any change to the data.

## 4. Key Engineering Decisions

| Decision | Why |
|---|---|
| **Separate layers with hard boundaries** | Each layer can be improved or replaced alone; bugs stay local; new layers (scenario, inventory) plug in without rewrites. |
| **The model layer never creates its own features** | One source of truth for features means training and prediction can never silently diverge — a classic forecasting bug class, eliminated by design. |
| **Walk-forward validation, never random splits** | Time series must be tested the way reality works: train on the past, predict the future. Random splits leak the future and flatter the models. |
| **Compare many models per product** | No single model wins everywhere. Simple baselines compete too — and when a simple model wins, it wins. |
| **Hybrid ensemble** | Blending the top models (weighted by proven backtest accuracy) often beats any single one — added safely, never forced to win. |
| **Total demand forecast** | Executives plan capacity and procurement on the total, not on 200 individual products. |
| **Business-friendly metric cards** | "wMAPE 17% — good. Slight under-forecasting — stockout risk." A planner acts on that; they don't act on a raw RMSE. |
| **Generic manufacturing by default** | Regional calendar logic (Ramadan/Saudi holidays) was removed for now so the platform serves any manufacturer; it can return later as an opt-in extension. |

## 5. Why This Is AI Engineering, Not Prompt Engineering

The team designed and built:

- **Data contracts** between every layer — documented inputs and outputs, with backward-compatibility aliases.
- **Leakage prevention** as an enforced, audited rule (7 automated checks that can halt the pipeline) — not a hope.
- **A validation workflow** that simulates real forecasting honestly (walk-forward backtesting).
- **Modular architecture** — six layers, single responsibilities, an orchestrator that treats stages as a pluggable list.
- **Model comparison and selection** by defensible criteria (weighted error first, bias second).
- **Failure handling** — a missing package or a crashing model is recorded and skipped; the user always gets a forecast.
- **Scalable execution modes** — fast mode handled 100 products in under a second in testing; designed for 1,000+.
- **Explainable outputs** — every mapping has a confidence and reason; every data issue has a rationale; every metric has a plain-language card.

## 6. Current Status

**Done (phase-one capabilities):**
- Intake — smart column mapping with memory of confirmed mappings and an optional (disabled) LLM assist
- Quality — business-aware risk detection with approval-gated remediation
- External Features — platform-provided calendar/weather/industry context
- Feature Engineering — leakage-audited, model-ready feature matrices
- Forecasting Engine — multi-model, multi-horizon, product + total forecasts

**Done (added 2026-06-13):**
- Forecast Performance Analysis — actual vs predicted vs error per validation period, with over/under-forecast indication (its own app page)
- What-if Scenario Simulation — demand ±%, seasonal strength, peak shift; baseline-vs-scenario comparison with impact summary, always labeled as assumption (its own app page)
- All 9 model families running end-to-end (statistical + ML packages installed and verified)
- First automated test suite (32 tests)
- English UI
- Two demo datasets, including real wine-industry sales data (1980–1994)

**Built but needs UI exposure (on the results page):**
- chart_data (history / validation / forecast lines in one chart)
- model_leaderboard
- total_forecast
- warnings and failed_models (shown calmly, not alarmingly)

**Next:**
- Streamlit results-page upgrade
- Executive summary generation
- Insight generation
- Operational risk flags

## 7. Demo Story

> "Bayyina helps manufacturers move from spreadsheet forecasting to an AI-assisted workflow. Watch: I upload a messy sales file — mixed column names, extra columns, some bad rows. The platform reads it and tells me what it understood: this is the date, this is the product, this is demand — and it asks me only about the columns it's unsure of. Then it reviews the data like an analyst: it found duplicate rows but recognized they're probably separate transactions, not errors — it recommends, I approve. It adds calendar and climate context on its own; I upload nothing extra. Then it races several forecasting models on honest backtests and shows me the leaderboard — for this product a simple seasonal model actually beat the machine-learning model, and the platform is honest about that. Finally: a 12-month forecast per product, a total demand line for capacity planning, and a plain-language scorecard — 'weighted error 17%, slightly under-forecasting, which means stockout risk.' That's a decision an operations manager can act on."

## 8. Risks To Mention Honestly

- This is a **platform foundation** — strong architecture, phase-one features; not yet production-deployed.
- ~~Advanced models skipped when packages aren't installed~~ **Resolved 2026-06-13:** all model packages are installed and every family (SARIMA, Prophet, Croston, XGBoost, Random Forest) has been exercised end-to-end on real data. Note: SARIMA needs roughly 40+ months of history to validate — on shorter datasets it honestly reports "history too short".
- Weather data is currently **internal demo climatology** covering 2018–2027, not live weather — data outside that window simply runs without weather features (guarded, logged).
- Deep learning (LSTM) is **intentionally future work** — the platform reserves its slot but does not pretend to have it.
- The results page **does not yet display** the full new forecast outputs (combined chart, leaderboard, total chart) — metric cards and the validation view are now visible on the Forecast Performance page.
- Scenario results are **assumptions, not predictions** — every scenario output carries that label by contract, and the baseline forecast is never altered.
- Accuracy labels ("good", "needs attention") are practical engineering thresholds, **not universal forecasting truth**.

## 9. Update Log

**2026-06-13 — two new capabilities, all models live, English UI, real data:**
- Shipped `forecast_analysis.py` with two clearly separated capabilities: **Forecast Performance Analysis** (measured — how accurate has the model been?) and **What-if Scenario Simulation** (hypothetical — what happens if demand changes?). Each has its own app page; the UI never blurs measured error with assumed impact.
- Wrote the platform's **first automated test suite** (32 tests, written before the implementation), including proof that a crashing analysis stage cannot corrupt the forecast and that scenarios never modify the baseline.
- Installed the statistical/ML packages and **verified all 9 model families end-to-end** — on the real dataset SARIMA actually wins one product, and the leaderboard shows it.
- **Fixed a latent platform bug:** external features with zero date coverage (e.g., the 2018–2027 weather file vs 1980s data) used to silently destroy tree-model training; they are now skipped with a logged reason.
- **Translated the app to English** (owner decision), including the layer messages the app renders.
- Added two demo datasets: a seeded synthetic one (`data/sample_demand_data.csv`) and a **real** one — Australian monthly wine sales 1980–1994 (`data/australian_wine_demand.csv`), which produces honest, imperfect numbers (portfolio weighted error ≈ 13%, a near-even split of over- and under-forecast periods).

## 10. Final Team Summary

Bayyina's value is its architecture. Because every layer has one responsibility and a documented contract, the platform scales in three directions at once: **more data** (fast mode processed 100 products in under a second, with execution modes designed for 1,000+), **more capability** (scenario simulation, inventory recommendations, and executive summaries plug into contracts that already anticipate them — scenario metadata, inventory passthrough, and summary objects exist today), and **more intelligence** (the LLM-assist and deep-learning slots are designed in, disabled, and waiting). The forecasting core follows production-oriented design principles — honest backtesting, leakage audits, graceful failure, explainable selection — which means the work ahead is about extending the platform, not repairing it. That is exactly the position a phase-one foundation should be in.
