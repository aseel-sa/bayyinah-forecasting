# Bayyina — Technical Handoff Report

**Audience:** the next AI model / engineer continuing this work.
**Status:** platform foundation with phase-one capabilities **+ the first two analysis capabilities (forecast performance, what-if scenarios) implemented, tested, and integrated** — see the Update Log (2026-06-13) at the end of this document.
**Last verified environment (2026-06-13):** Python 3.13.2, pandas 2.2.3, scikit-learn 1.8.0; **statsmodels, prophet, statsforecast, xgboost ALL installed and their model paths verified end-to-end** (pytest still absent — tests use stdlib `unittest`).

---

## 1. Executive Technical Summary

Bayyina is an AI-powered demand forecasting platform for manufacturers. A user uploads messy operational sales data (any column names, Arabic or English, any granularity); the platform understands the schema, assesses data risk, enriches the data with platform-provided external context, builds leakage-safe features, validates multiple forecasting models with walk-forward backtesting, and produces product-level and total-demand forecasts with business-language accuracy explanations.

It is built as a **layered AI engineering platform**, not a model script: each layer has a single responsibility, an explicit output contract, and a hard boundary. Layers communicate only through documented dict/DataFrame contracts so any layer can be improved or replaced without rewriting its neighbors. Every heuristic decision in the system (column mapping, quality severity, model selection) carries a confidence/reason and is surfaced to the user — nothing is decided silently.

## 2. Current Architecture

```
Intake → Quality → External Features → Feature Engineering → Model → Forecast Performance → Streamlit UI
                                                                   ↘ (UI-triggered) What-if Scenario
                                                                    → (next: Insights / Executive Summary / Operational Risk)
```

Orchestrated by `pipeline.py` (`run_pipeline` / `resume_after_quality_review`). Stages after Quality live in `_POST_QUALITY_STAGES`, a plain list — adding a future layer is one list entry + one stage function. The first such addition exists: the non-fatal `forecast_performance` stage (consumes `model_result`, can never halt the pipeline or alter the forecast). Scenario simulation is deliberately NOT a stage — it needs user-declared assumptions, so the UI invokes it on the completed result.

| Layer | Owns | Explicitly does NOT |
|---|---|---|
| **Intake** (`intake.py`) | Column *understanding*: role mapping, business-meaning discovery, capabilities/readiness (presence-based), mapping proposals, user corrections, mapping memory | Clean data, judge value quality, train anything |
| **Quality** (`quality.py`) | *Value* risk: structural / time-series / business / representation issues, severity+confidence, remediation plan, approval-gated execution, audit log | Re-detect columns, aggressive auto-deletes, forecasting |
| **External Features** (`external_features.py`) | Platform-provided external context: generic calendar, weather climatology CSV, industry indicators; lifecycle metadata (future availability, scenario capability) | Lags/rolling, seasonality detection, API calls, user-uploaded feature files |
| **Feature Engineering** (`feature_engineering.py`) | Gatekeeper: column routing, panel, temporal/lag/rolling/cyclical/trend/volatility features, data-driven seasonality, external alignment, feature metadata, relevance analysis, leakage audit | Model training, feature selection |
| **Model** (`model.py`) | Forecasting Engine: validation, selection, multi-horizon forecast, totals, hybrid, metric cards, leaderboard, failure handling | Creating training features, UI, LLM insights |
| **Forecast Analysis** (`forecast_analysis.py`) | TWO sealed sections: (1) measured forecast performance (actual vs predicted vs error from validation rows); (2) what-if scenario simulation (assumption-based post-forecast adjustment, baseline never mutated, mandatory disclaimer) | Forecasting, retraining, recomputing metrics the model layer already provides |
| **UI** (`app.py`) | 7-page Streamlit flow (English) with human-in-the-loop quality review | Business logic (calls pipeline / layer public entry points only) |

**The cardinal boundary rule:** intake evaluates input *presence* (is there a stock column?); quality evaluates *values* (are the stock values plausible?); feature engineering *applies* intake's classification (never re-classifies); model *consumes* feature matrices (never builds training features).

## 3. File-by-File Summary

### `intake.py` (~1,414 lines)
- **Purpose:** understand the uploaded dataset; produce a reviewable mapping proposal.
- **Public:** `run_intake(df, llm_client=None, memory_path=None)` (new main entry), `apply_user_corrections(result, corrections)`, `confirm_mapping(result, memory_path)`, `build_intake_result(df)` (legacy entry used by pipeline), `profile_columns(df)`, `detect_columns(df)`, `discover_business_inputs(df, mapping)`.
- **Inputs:** raw DataFrame (never reads disk). **Outputs:** mapping + confidence + reasons, business inputs (meaning/confidence/legacy_category), capabilities, presence-readiness, five-bucket contract (required / forecast features / inventory features / ignored / needs_review), column profiles, proposals.
- **Recent changes:** token-based name matching (fixed substring false positives: `Holiday`→`id`, `Last Update`→`date`, `Customer Name`→product, `Order Value`→qty); deterministic sampling cap `_SAMPLE_CAP=10_000` for value-shape signals; new meanings (revenue/customer/region/category/discount/stockout); the cleaning trio + `validate_data`/`quality_report` were **deleted** (duplicated quality.py — boundary enforcement); slimmed 2,941→1,414 lines with English notes; redundant discovery passes deduplicated (1M-row `run_intake`: 25.4s → 6.8s).
- **Design notes:** multi-signal voting (name 0.5 / dtype 0.2 / value-shape 0.3); "high" confidence requires name AND structure agreement (a high-confidence wrong mapping silently unlocks wrong capabilities downstream); memory-before-LLM (schema fingerprint replays confirmed mappings at 0.95).
- **Risks:** greedy role assignment (date→product→qty) can be suboptimal under contention (alternatives surfaced to user); numeric Unix-timestamp dates cap at medium confidence; confidence-label→score mapping (.85/.62/.35) is a stated simplification.

### `mapping_memory.py` (~100 lines)
- JSON store of user-confirmed mappings: schema fingerprint → full mapping; per-column normalized-name → role+count. Corrupt/missing file degrades to empty (never blocks intake). Saved **only** on `confirm_mapping`. Single-tenant file; upgrade path: SQLite + tenant scoping.

### `llm_mapper.py` (~154 lines)
- **Prepared but disabled** (`llm_client=None` default). Metadata-only payload (profiles + rule guesses + resolved mappings; never raw data). Strict validation: unknown columns/roles silently rejected; confidence hard-capped at 0.70; suggestions ALWAYS stay `needs_review`; never overrides user or memory sources. Client failure → empty list, flow degrades to manual review. No API dependency — inject any `callable(payload)->json_str`.

### `quality.py` (~1,143 lines)
- **Purpose:** business-aware data-risk engine. Philosophy: this is business data — duplicates may be transactions, spikes may be promotions, negative demand may be returns. Understand risk, recommend, never aggressively clean.
- **Public:** `run_quality_engine(df, intake_result, core_mapping=, business_inputs=, target_freq=, ...)`, `execute_plan(df, plan, approved_actions, core_mapping, freq)`, `execution_log_to_dataframe(log)`.
- **Recent changes:** complete philosophy refactor — 4 validation categories, 3 severities (critical/warning/info) + numeric confidence per issue, duplicate *investigation* (txn-id evidence → ETL vs transactions vs review), outliers advisory-only (the cap/winsorize executor was **removed entirely**), **all HVAC-specific constants removed** (peak months, 2–16-week lead-time range) and replaced with data-driven equivalents (recurring-month zero detection; MAD relative outliers), new representation category.
- **Risks:** seasonal-zero classification needs ≥2 observed years (below that, conservative + low confidence); mixed lead-time units within one column undetectable; severity penalties are engineering choices.

### `external_features.py` (~430 lines)
- **Purpose:** platform-provided external features (user never uploads feature files). Generic-manufacturing mode.
- **Public:** `build_external_features(start_date, end_date, granularity, country=None, city=None, industry=None, ...)`, `recommend_features(...)`, `build_calendar_features(...)`, `get_registry()`.
- **Recent changes:** all Saudi/Ramadan/Eid logic **deleted** (was implemented, then removed per generic-platform decision; recoverable from session history as an opt-in registry block later). Defaults `country="generic"`, `industry="manufacturing"`.
- **Risks:** one weather city per run; weekly granularity upsamples monthly weather (ffill); the bundled weather CSV is placeholder climatology (see data README).

### `feature_engineering.py` (~899 lines)
- **Purpose:** gatekeeper between cleaned data and models — features earn their place.
- **Public:** `build_features(df, intake_result, quality_result=None, granularity, sector=None, external_features=None, core_mapping=None, business_inputs=None, external_metadata=None)`, `feature_relevance_analysis(...)`, `route_columns(...)`. Internal helpers `_GRAN_CONFIG`, `_add_temporal_features`, `_add_seasonal_features` are intentionally imported by model.py (one feature definition for train AND recursive predict).
- **Recent changes:** token-based leakage matching (fixed `Marginal_Notes` false positive); future-availability framework (`safe_for_future` / `requires_future_values` / `scenario_only` / historical via external metadata) on every feature; `feature_relevance_analysis` (Pearson at lags 0–3, coverage, verdict — explainability, NOT selection); new features: `month_sin/cos`, `week_sin/cos`, `time_index`, `rolling_std`, `pop_growth` (guarded /0; designed to add no NaN rows beyond the longest lag); `inventory_passthrough`; merge collision guard (external columns never overwrite panel-owned columns — skipped + logged); `external_metadata` parameter consumed for availability tagging.
- **Risks:** name-based leakage detection misses innocently-named future fields (`delivered_qty`) — needs manual UI marking; `time_index` does not give trees extrapolation ability (model-family limit); relevance is pooled linear correlation (per-product nonlinear effects can read "weak").

### `model.py` (~999 lines) — the Forecasting Engine
- **Purpose:** model execution/validation/selection/forecast/output formatting. See §8 for full detail.
- **Public:** `run_forecast(df=None, fe_output=None, core_mapping=..., granularity, forecast_horizon=None, validation_horizon=None, evaluation_mode="fast|balanced|full", ...)` plus helpers `segment_products`, `_metrics`.
- **Recent changes:** complete refactor (see §8). Notable bug fixed: per-fold tolerance — ML models previously failed entirely because the *first* (shortest) fold lacked trainable rows after lag-NaN; now a model fails only if **all** folds fail.
- **Risks:** static-feature compromise in fold slicing (§8/§10); residual-based CIs are approximations.

### `pipeline.py` (~430 lines)
- Pure orchestration; stage list `_POST_QUALITY_STAGES = [external_features, feature_engineering, model, forecast_performance]`; per-stage timing + try/except; loud halts (no date/qty/product column, leakage-audit failure, no forecast produced); human pause at quality (`awaiting_quality_review` + `resume_after_quality_review(state, user_decisions)`); auto-approve policy excludes `remove_negative_demand`. Model stage now passes `fe_output=ctx["feature_result"]` — the historical double feature build (old dev note [خ٣]) is resolved. The `forecast_performance` stage is non-fatal by design (failure → empty `performance_result` + warning, forecast untouched). Output object gained the additive key `performance_result`; all pre-existing keys unchanged. User-visible halt/warning messages are English.
- **Risk:** `_pipeline_state` carries `raw_df` in-memory (fine for Streamlit session_state; NOT JSON-serializable for cross-session resume).

### `forecast_analysis.py` (~388 lines) — NEW (2026-06-13)
- **Two capabilities in one file by owner decision**, sealed sections with `_perf_*` / `_scen_*` prefixes, no cross-calls, either deletable without breaking the other. Imports only numpy/pandas (no project modules — circular imports impossible).
- **Section 1 `analyze_forecast_performance(model_result)`** (measured): per-period `actual/predicted/error/error_pct/direction` table derived 1:1 from `validation_predictions` (error = predicted − actual, same sign convention as `model._metrics`; `error_pct` NaN when actual=0 — never inf), pooled per-date totals, summary that ECHOES `metric_cards`/`portfolio_wmape`/`confidence_label` verbatim (zero recomputation). Degrades to an empty contract + warning without validation history. `analysis_type: "forecast_performance_measured"`.
- **Section 2 `run_scenario(model_result, scenario, fe_output=None)`** (assumption): demand ±% (global + per-product overrides), seasonal strength factor, peak shift — pure post-forecast arithmetic per product (`level + roll(factor·dev, shift)`, then multiplier, then clip ≥0; order is contractual). Baseline defensively copied, never mutated; bounds inherited via parallel shift (labeled approximations); every output carries a mandatory assumption-not-prediction disclaimer warning (`affected_item="disclaimer"`). Invalid inputs fall back to neutral defaults loudly. `fe_output` reserved for the future feature-rerun tier (weather scenarios via `model._climatology`/`_predict_tree` — still requires exposing a public re-predict hook in model.py, deferred). `analysis_type: "scenario_simulation_assumption"`.

### `test_forecast_analysis.py` (~347 lines) — NEW (2026-06-13): the first automated test suite
- 32 stdlib-`unittest` tests, written BEFORE the implementation: exact arithmetic on a hand-built fixture, integration against a real pipeline run, input-immutability for both sections, identity/linearity/ordering invariants for scenarios, disclaimer always present, graceful degradation, and 4 pipeline-integration tests including failure isolation (a mocked crash in the analysis stage → pipeline completes, forecast intact, warning emitted). Run: `python -m unittest test_forecast_analysis`.

### `app.py` (~482 lines)
- 7-page Streamlit flow, **fully English** (Upload → Intake → Quality review w/ approve-reject toggles → Features → Results → **Forecast Performance** → **What-if Scenario**), sidebar stage tracker, state reset on new upload, clean failure display, CSV downloads. Page 6 renders `performance_result` (actual-vs-predicted line + error bars, totals and per-product, echoed metric cards, over/under counts). Page 7 collects three sliders (demand ±%, seasonal strength, peak shift), calls `forecast_analysis.run_scenario`, renders the disclaimer prominently, impact metrics, baseline-vs-scenario chart, most-affected products, declared assumptions. The two pages explicitly distinguish measured accuracy from assumption simulation. **Results page (5) still consumes only legacy aliases** (`forecasts`/`comparison_table`/`history`) — still the highest-priority UI task (§11). No mapping-correction dropdowns yet (backend `apply_user_corrections`/`confirm_mapping` ready).

### `data/`
- `external/README.md` — file formats + the four platform-mode statements. `external/weather_riyadh_monthly.csv` — **placeholder climatology**: monthly temperature normals repeated 2018–2027 (smooth identical seasonal curve every year, not observations). **Coverage caveat:** data outside 2018–2027 gets zero weather coverage — handled by the FE coverage guard (see Update Log).
- `sample_demand_data.csv` — NEW: seeded synthetic demo set (5 products × 42 months, planted summer/winter peaks, growth trend, stable, intermittent).
- `australian_wine_demand.csv` — NEW: REAL data (Australian Bureau of Statistics monthly wine sales 1980–1994 via GitHub), 6 products × 180 months, prepared wide→long with values untouched (2 missing quantities intentionally kept for the quality engine). Realistic accuracy: portfolio wMAPE ≈ 13% in full mode; SARIMA validates and wins for Fortified.

### `.streamlit/config.toml`
- Theme primary color `#1a4d5e` (petrol blue).

## 4. Intake Layer Details

- **Column mapping:** per (column, role) score = 0.5·name + 0.2·dtype + 0.3·value-shape. Name matching is **token-based** (separators + camelCase split + Arabic definite-article stripping); generic tokens (`id`, `name`, `value`, …) cap at 0.45 and cannot decide alone. Value-shape signals (date-parse rate, numeric non-negativity, repetition ratio) computed on a deterministic sample (first 1,000 + seeded random, ≤10k). Greedy assignment with a 0.30 floor; runner-ups become user-facing alternatives.
- **Roles detected:** required — `date`, `product` (sku/item/project), `demand_quantity` (internal key `qty`). A cross-check flags a qty column that token-matches money keywords (`Order Value`) for review.
- **Business inputs:** 22-meaning registry (stock_on_hand, lead_time, safety_stock, reorder_point, moq, supplier, fill_rate, lost_sales, backorders, stockout, price, promotion, discount, temperature, holiday, capacity, revenue, customer, region, category, …) scored by name (0.5) + structure (0.5: value-hint fit × temporal-hint fit — static-per-product vs time-varying vs environmental). Pure structural matches with zero name signal stay `unknown` (missing a meaning is cheaper than confidently unlocking a wrong capability).
- **LLM treatment:** optional reviewer for ambiguous columns only; suggestions capped at 0.70 confidence and always `needs_review`; never outranks user or memory; full flow works with it disabled.
- **Why intake doesn't clean/judge:** two cleaning paths diverge silently (this duplication existed and was removed); quality.py owns anything touching or judging values, permanently.

## 5. Quality Layer Details

- **Structural:** missing core columns (critical), unparseable dates / non-numeric demand (critical only when *nothing* parses), invalid product ids, empty rows, exact duplicates — **investigated**: a duplicated near-unique txn-id column → likely ETL artifact (conf 0.85, removal recommended via review); no id + tiny rate → conf 0.6; large rate → `review_duplicates` (conf 0.4). Duplicate removal is never auto.
- **Time-series:** insufficient history (merged sparse+short), missing periods (counted pre-zero-fill; "may be expected" caveat), product-period multiplicity (differing quantities within groups → transactions → aggregate, **info** severity; identical → review), intermittency computed on **non-seasonal zeros only** (a calendar month all-zero in ≥70% of observed years is an expected dead season — fully data-driven, no sector assumption), long non-seasonal gaps, irregular recording frequency vs chosen granularity.
- **Business:** negative demand (warning — returns are legitimate; deletion is `never`-policy, manual override loudly logged), negative stock, lead-time anomalies relative to the column's own distribution (MAD), suspicious constant demand (>95% one value), product-identifier variants (trim auto; case-merge review — `AC-100` vs `ac-100` may genuinely differ).
- **Representation (new):** product dominance (>60% of demand), region/customer segment imbalance (via intake meanings), calendar months never observed despite ≥1y span, stale products (last record >3 periods before dataset end).
- **Remediation plan:** one preferred action per issue routed by policy — `auto` (whitespace/empty rows/date format only), `review` (anything changing values; impact estimate embedded per item), `never` (business-record deletion), `advisory` (human judgment). Execution is approval-gated (`execute_plan`) with full audit log (applied/not_applied/warnings → CSV).
- **Readiness:** value-quality scores (distinct from intake's presence scores): start at 100, subtract severity-penalty × confidence per relevant issue; reasons always attached; first reason states "this measures value completeness, not accuracy".
- **User-facing warnings:** the sorted `issues` list (severity → blast radius), `summary_for_user` headline (top 3), and each issue's description/rationale/forecast_impact/inventory_impact — all Arabic UI strings.

## 6. External Features Layer Details

- **Why it exists:** external features have a lifecycle (source, coverage, future availability, scenario capability, update cadence) that is metadata management, not time-series math — keeping it in feature_engineering would bloat the gatekeeper.
- **User never uploads feature files:** the platform generates calendar features in code and loads internal CSVs (`data/external/`). Missing file → `unavailable_features` entry + warning, never a crash; calendar works with zero files.
- **Provided:** calendar `month/quarter/year(/weekofyear)` (these duplicate FE panel temporals **intentionally** for standalone consumers; FE's merge collision-guard deduplicates them, verified single `month` column in tree matrices); weather `temperature/humidity(+derived CDD/HDD, base 18°C)`; industry `ahri_shipments` (HVAC only), `construction_activity` (only if file exists).
- **Generic by default:** `country="generic"`, `industry="manufacturing"`. Saudi/Ramadan features were fully implemented then **removed** per the generic-platform decision (deleted, not commented; recoverable from session history as a future opt-in registry block).
- **Climatology:** the weather CSV is historical/normals data aligned to the period grid (monthly→weekly upsampled via ffill with warning). Unknown city falls back to the default file **with a warning** (silent substitution would be dishonest).
- **Metadata returned:** registry fields per produced column (category, source, future_availability, scenario_capable, leakage_risk, industry_relevance, description) — shaped for FE's `external_metadata` parameter. **Future availability classes:** `safe_for_future` (calendar), `requires_future_values` (weather), `historical_context` (AHRI/construction — published with a lag), `scenario_only` (reserved for scenario-engine levers).

## 7. Feature Engineering Layer Details

- **Matrices per model family:** `sarima`/`baseline` = `[product, date, y]` univariate; `prophet` = `[product, ds, y, is_peak_season, is_low_season]` (flags as optional regressors; Prophet keeps internal seasonality); `xgboost`/`random_forest` = DatetimeIndex + product + y + all features (raw date never a feature column).
- **Temporal:** month/quarter/year(/weekofyear), cyclical `month_sin/cos` (+weekly), `time_index` (per-product cumcount).
- **Leakage-safe lags/rolling:** lags via `groupby(product).shift` (never crosses products); every rolling (mean + std) is `shift(1)`-then-roll **inside** the group; `pop_growth` derived from existing lags with /0 guard. NaN rows at series starts stay NaN (consumer drops; never zero-filled to avoid fake signals).
- **Seasonality — three separated concerns:** detection (top/bottom 33% monthly means, `year_round` low-variation check first, sector default only as a <18-month fallback, else `undetected`); features (detected flags + cyclical encodings — complementary, trees only); modeling (SARIMA/Prophet own internal seasonality and receive NO seasonal features — signal duplication avoided).
- **External alignment:** date-join (per product if a product column exists, else broadcast), ffill ≤2 gaps, larger gaps flagged; collision guard skips panel-owned names.
- **Feature metadata:** every feature → `{feature_name, source, leakage_safe, future_availability, applies_to_models, description}` — feeds explainability and model-layer availability handling.
- **Leakage audit:** 7 checks (shift1, per-product lags, no raw date in trees, post-sale exclusions via token match, no target duplication, all features availability-classified, requires-future features explicitly flagged). Any failure flips `all_checks_passed` → pipeline halts before modeling.
- **Safe for future:** calendar/cyclical/trend/lag/rolling/seasonal-detected (lags update recursively). Not safe: weather actuals (climatology-filled downstream), scenario/historical columns (excluded from training downstream).
- **Remaining risks:** see §10.

## 8. Model Layer Details

- **Why refactored:** the previous version conflated horizons (validation hardcoded 1, final default 3), silently fell back to mean when packages were missing (models appeared to have run), zero-filled future externals, returned visualization-hostile dicts, and rebuilt features per fold per product per model (impossible at scale).
- **No internal feature creation:** consumes `fe_output` (pipeline passes the feature stage's result — single build). Recursive prediction reuses `fe._add_temporal_features`/`_add_seasonal_features`, so train and predict share one feature definition. When called directly without `fe_output`, it builds features ONCE, never per fold.
- **Two horizons:** `validation_horizon` (default 3) = steps per walk-forward fold; `forecast_horizon` (default **12 monthly / 26 weekly**) = the user-facing forecast. Both overridable (`config.forecast_horizon` / `validation_horizon`; legacy `horizon` accepted as alias).
- **Walk-forward:** expanding window; `min_train = season+2` (degraded for short series); fold cutoffs spaced across available test region; train strictly on past, test next `vh` periods. Random splits are never used because they leak future patterns into training and overstate accuracy on autocorrelated series. Insufficient history → `insufficient_history` status, safe naive forecast, explicit warning.
- **Per-fold tolerance (bug fixed):** a model fails only when ALL folds fail — early short folds legitimately lack trainable rows for ML models (lags consume the start).
- **Models:** fast (dependency-free): `naive`, `seasonal_naive` (repeats last cycle — real multi-step shape), `moving_average`, `exponential_smoothing` (manual SES). Advanced: `sarima` (statsmodels), `prophet`, `croston` (statsforecast), `random_forest` (sklearn), `xgboost`. Missing package → status `skipped`, reason `package_not_available` (verified live: sarima/prophet/croston show as skipped in this environment); runtime error → `failed` + message; pipeline never crashes; naive guarantees a usable forecast. `lstm` exposed as leaderboard metadata only (`future_extension`).
- **Selection:** wMAPE → |forecast_bias| → MAE. Never MAPE-only. Simple models win when they score better (observed: seasonal_naive beating ML on one product).
- **hybrid_ensemble:** top ≤3 valid models weighted by inverse wMAPE; validation predictions blended from components' **stored** fold rows (no retraining); skipped under 2 components; appears in the leaderboard as a normal candidate and is never forced to win. `hybrid_details` exposes components + weights per product.
- **Forecast outputs:** per product, the selected model retrains on full history and forecasts `fh` steps (trees recursively). `chart_data` separates `series_type = history | validation | forecast` for products and `"ALL"`, with `forecast_start_date` and bounds. Totals: `bottom_up_total` always; `direct_total_forecast` (candidates validated on the aggregated series) also implemented.
- **Metric cards:** wMAPE / Forecast Bias / MAPE / MAE / RMSE with plain-Arabic explanations and labels (`excellent/good/needs_attention/poor`). Thresholds documented in code (wMAPE ≤10/20/35%; |bias| ≤5/15/30%; MAE/RMSE relative to mean demand ≤15/30/50%) and explicitly NOT claimed universal. wMAPE is the portfolio metric (volume-weighted, zero-tolerant); bias direction (over/under/balanced at ±5%) ties directly to inventory risk language.
- **Scale:** `evaluation_mode`: `fast` (fast models only, 2 folds — measured **100 products in 0.9s**, ≈9s projected for 1,000), `balanced` (advanced models for high-volume products only, fast for the long tail), `full` (everything, may be slow). Segmentation (`high_volume` = top-50%-of-demand contributors, `normal`, `low_volume` = bottom 20%, `intermittent` = zero-ratio>0.35, `short_history` = under one season) selects candidates.
- **External future values:** `scenario_only`/`historical_context` features are excluded from training (warned); `requires_future_values` kept in training and future steps filled with **per-product calendar-month climatology** — never zero (warned, explicit).
- **Accepted trade-off (be aware):** fold evaluation slices ROWS of a single full-history feature build. Lag/rolling values are backward-looking (slice-safe), but detected-seasonality flags and panel-level values derive from full history — a static-feature compromise. The pure alternative (rebuild features per fold) was measured and is infeasible at 1,000+ products. Documented in model dev-note [3] and `evaluation_summary.notes`.

## 9. Output Contracts

**Intake →** `proposed_mapping`, `business_inputs` (column/meaning/confidence/legacy_category/stockout_signal), `capabilities`, `readiness` (presence), `business_context`, `summary`, five-bucket contract (`required_mappings`, `optional_forecast_features`, `optional_inventory_features`, `ignored_columns`, `needs_review`), `column_profiles`, `mapping_proposals`, `confirmed_mapping`, `ready_for_engine`.

**Quality →** `issues` (+`issues_by_category`), `readiness_scores` (value-quality), `remediation_plan` (`auto_actions`/`review_required`/`skipped_actions`, impact embedded), `execution_log`, `ready_for_feature_engineering`, `summary_for_user`.

**External Features →** `external_features` (date-indexed frame), `feature_metadata` (registry-shaped), `recommended_features`, `unavailable_features`, `warnings`, `ready_for_feature_engineering`.

**Feature Engineering →** `feature_matrices` (5 family views), `feature_metadata` (incl. `future_availability`), `excluded_columns`, `external_features_used`, `seasonal_detection`, `feature_relevance`, `inventory_passthrough`, `leakage_audit`, `ready_for_modeling`.

**Model →** full contract with consumer mapping:

| Key | Primary consumer |
|---|---|
| `forecast_summary` | Streamlit header cards + Executive Summary layer |
| `product_forecasts` (long DF: product/date/forecast/bounds/model_used/horizon_step/forecast_start_date) | Streamlit charts, downloads, Inventory layer |
| `total_forecast`, `direct_total_forecast` | Streamlit total chart, Executive Summary |
| `chart_data` (product or "ALL" / series_type history\|validation\|forecast / bounds) | Streamlit — THE chart contract |
| `validation_predictions` (selected models' fold rows) | Streamlit validation overlay, diagnostics |
| `model_leaderboard` (scope product/ALL, all metrics, status, selected) | Streamlit comparison, diagnostics |
| `metric_cards` | Streamlit business cards, Insights |
| `product_segments`, `best_model_by_product`, `best_model_overall`, `hybrid_details` | Streamlit + Insights |
| `evaluation_summary`, `warnings`, `failed_models` | Diagnostics (display non-scarily) |
| `history`, `history_total` | Charts |
| `feature_engineering` (audit echo) | Diagnostics |
| Legacy aliases: `forecasts`, `comparison_table`, `summary_table`, `best_model` | Current app.py (until upgraded) |

## 10. Known Risks and Technical Debt (honest list)

1. ~~statsmodels / prophet / statsforecast are not installed~~ **RESOLVED 2026-06-13:** all model packages installed; SARIMA/Prophet/Croston/XGBoost validated end-to-end on real data (wine dataset, full mode). Remaining caveat: SARIMA needs ≥28 training points (its own guard), so datasets under ~40 months never validate it — folds max out below the guard. The "SARIMA ~12% on AHRI" claim from the original brief remains unreproduced (no AHRI dataset).
2. **Static-feature walk-forward compromise** (model §8): seasonal flags derived from full history leak mildly into folds. Accepted for 1000×-scale; revisit if accuracy claims become contractual.
3. **Weather is placeholder climatology** (identical normals each year, Riyadh only) — fine as a smooth seasonal regressor, not real weather. `requires_future_values` handling (climatology fill) is reasonable but unvalidated against real data.
4. **app.py results page (page 5) does not yet consume the new contract** — chart_data/leaderboard/total/warnings still invisible there. Partially mitigated 2026-06-13: metric_cards and the validation view are now exposed on the Forecast Performance page (6).
5. **Metric thresholds and quality severity penalties are engineering labels**, not calibrated truth.
6. **Hybrid quality = backtest quality**: few folds → noisy weights. It is never forced as winner, which bounds the damage.
7. **Bottom-up vs direct totals can disagree**; no reconciliation logic exists (documented as future work).
8. **CI bounds are residual-based approximations** (±1.28·RMSE); summed product bounds overstate total uncertainty (no diversification effect).
9. **Leakage keyword detection is name-based** — innocently-named future fields escape; needs a manual UI marking flow.
10. **Mapping memory is a single JSON file** (no tenancy, no concurrency).
11. ~~No automated test suite~~ **SEEDED 2026-06-13:** `test_forecast_analysis.py` (32 tests) covers the analysis module + pipeline integration. The five original layers still lack their own contract-freezing tests — still a high-leverage investment.
12. **`_pipeline_state` holds raw_df in memory** — resume works within a session only.
13. Balanced mode runs ML in walk-forward for high-volume products (~11s for 2 products here); `full` mode on large portfolios will be slow — UI should warn.

## 11. What To Do Next

**High priority**
1. Upgrade Streamlit results page (page 5) to the new model contract: `chart_data` (history/validation/forecast as separate lines + forecast-start marker + bounds band), total-demand chart ("ALL"), `model_leaderboard` (with skipped/failed shown calmly), warnings panel. (metric_cards + validation view already exposed on page 6.)
2. Intake mapping-correction UI (dropdowns) wired to `apply_user_corrections`/`confirm_mapping` — backend is ready and tested.
3. ~~Install statsmodels/prophet/statsforecast and validate~~ **DONE 2026-06-13** (see Update Log).
4. Extend the automated test suite to the five original layers (`test_forecast_analysis.py` is the seed and the pattern: fixtures from a real pipeline run + golden-digest regression gate).

**Medium priority**
5. ~~Scenario simulation layer~~ **DONE 2026-06-13** for phases A/B (demand ±%, seasonal strength/shift) as `forecast_analysis.py` Section 2. Remaining: phase C (weather/feature-value scenarios) — needs a public re-predict hook in model.py over the existing climatology plumbing.
6. Executive summary generation (consume `forecast_summary` + `metric_cards` + `total_forecast`; approved design exists — see roadmap).
7. Insights generation (`insights.py`, deterministic findings only — the performance view originally planned inside it now lives in `forecast_analysis.py` Section 1).
8. Inventory recommendations (consume `inventory_passthrough` + forecasts).
9. Better future-external strategy (real weather feed with broader coverage years; reconcile bottom-up vs direct totals).

**Future**
9. LLM-based insight narration; chatbot assistant; LSTM/TFT (leaderboard slot reserved); API weather/economic integrations; regional calendars as opt-in registry blocks (Saudi implementation recoverable from history).

## 12. How To Explain This Project In A Demo

> A planner uploads a messy sales export — any column names, any language. Bayyina **understands** the columns (and asks only when unsure), **checks** the data like a careful analyst — telling apart real business behavior (returns, transactions, seasonal zeros) from genuine data problems — and recommends fixes it never applies without approval. It **enriches** the data with platform-provided context like calendar structure and climate, then builds forecasting features under a strict leakage audit. It **races** multiple forecasting models per product in honest walk-forward backtests — simple baselines included, and they win when they deserve to — then produces a 12-month product-level forecast, a total-demand forecast for capacity planning, and a chart that separates what happened, what the model would have predicted, and what comes next. Finally it explains accuracy the way a planner thinks: "weighted error 17%, slight under-forecasting — that means stockout risk."

---

## 13. Update Log

### 2026-06-13 — Forecast Performance + What-if Scenarios shipped; all models live; English UI; real dataset

**New capability (approved architecture → implemented → tested → integrated):**
- `forecast_analysis.py` created: Forecast Performance Analysis (measured) + What-if Scenario Simulation (assumption) as two sealed sections in one file (owner decision). Read-only consumers of `model_result`; no project imports; both degrade gracefully; scenario outputs always carry the assumption-not-prediction disclaimer and never mutate the baseline.
- `test_forecast_analysis.py` created: 32 tests (tests-first), including baseline-immutability, scenario identity/linearity/ordering invariants, and pipeline failure-isolation. This is the project's first automated test suite.
- `pipeline.py`: additive integration only — `forecast_performance` appended to `_POST_QUALITY_STAGES` (non-fatal: failure → warning, forecast untouched) + new `performance_result` output key. Verified by golden-digest comparison: pre/post pipeline outputs byte-identical except the new stage-log entry.
- `app.py`: two new pages via the designed `STAGES` extension point — 6 · Forecast Performance (the actual-vs-predicted-vs-error view) and 7 · What-if Scenario (sliders → `run_scenario` → baseline-vs-scenario comparison). Scenario state resets on new upload.

**Environment (resolves old risk #1):** xgboost, statsmodels, prophet, statsforecast (+ openpyxl) installed. All 9 model families verified validating `ok` end-to-end (full mode). Local Python is 3.13.2; pytest absent → stdlib `unittest`.

**Bug fixed (pre-existing, latent):** `feature_engineering._align_external_features` now skips+logs external columns with ZERO temporal coverage instead of merging them as all-NaN. Previously, any dataset outside the weather CSV's 2018–2027 window got all-NaN temperature columns, whose dropna deleted every tree-model training row ("fewer than 5 trainable rows") — random_forest/xgboost failed on all historical data. The guard mirrors the function's existing collision-guard pattern.

**UI language:** app fully translated to English, plus the layer strings the app actually renders: quality `_ACTIONS` labels/benefits + skip reasons, FE leakage-audit details + routing exclusion reasons, pipeline halt/warning messages, forecast_analysis warnings. NOT translated (not rendered by the app): intake internals, model metric-card prose (`plain_english`/`business_meaning`), quality issue descriptions/rationales, execution-log content — an open task if those ever surface in the UI.

**Datasets:** `data/sample_demand_data.csv` (synthetic, seeded, clean) and `data/australian_wine_demand.csv` (real ABS data 1980–1994, 6 products × 180 months, melted wide→long, values untouched incl. 2 missing quantities). The real set produces honest numbers: wMAPE ≈ 13% ("good"), 34 over- / 34 under-forecast validation periods, SARIMA winning one product.

**Contracts:** no existing key renamed, removed, or changed; `performance_result` (pipeline) is the only addition. Forecasting, selection, and validation behavior untouched (the FE coverage guard changes which *external* columns enter matrices for out-of-coverage data — previously those columns were all-NaN and only destroyed training).

---
*Originally written from direct inspection of the repository on the handoff date; §13 logs subsequent changes. Update this log with every change set.*
