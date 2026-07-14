"""
test_insights.py — lightweight tests for the deterministic insights layer.

Run: python test_insights.py   (no test framework required)
Covers the 7 behaviours required by the spec.
"""

import json

import numpy as np
import pandas as pd

import insights


# ─── Fixtures ────────────────────────────────────────────────────────────────

def _minimal_model_result() -> dict:
    """A small but realistic model.run_forecast-shaped output."""
    return {
        "forecast_summary": {
            "forecast_horizon": 3, "validation_horizon": 1, "granularity": "monthly",
            "forecast_start_date": pd.Timestamp("2024-01-01"),
            "total_expected_demand": 600.0, "peak_forecast_period": pd.Timestamp("2024-02-01"),
            "best_overall_model": "seasonal_naive", "portfolio_wmape": 0.18,
            "forecast_bias_direction": "under-forecasting", "confidence_label": "good",
        },
        "product_forecasts": pd.DataFrame([
            {"product": "A", "date": pd.Timestamp("2024-01-01"), "forecast": 100.0},
            {"product": "A", "date": pd.Timestamp("2024-02-01"), "forecast": 110.0},
            {"product": "A", "date": pd.Timestamp("2024-03-01"), "forecast": 120.0},
            {"product": "B", "date": pd.Timestamp("2024-01-01"), "forecast": 20.0},
            {"product": "B", "date": pd.Timestamp("2024-02-01"), "forecast": 18.0},
            {"product": "B", "date": pd.Timestamp("2024-03-01"), "forecast": 22.0},
        ]),
        "total_forecast": pd.DataFrame([
            {"date": pd.Timestamp("2024-01-01"), "total_forecast": 120.0},
            {"date": pd.Timestamp("2024-02-01"), "total_forecast": 128.0},
            {"date": pd.Timestamp("2024-03-01"), "total_forecast": 142.0},
        ]),
        "model_leaderboard": pd.DataFrame([
            {"model_name": "seasonal_naive", "scope": "product", "product": "A",
             "wmape": 0.12, "mae": 8.0, "forecast_bias": -0.05, "status": "ok", "selected": True},
            {"model_name": "naive", "scope": "product", "product": "B",
             "wmape": 0.55, "mae": 9.0, "forecast_bias": 0.40, "status": "ok", "selected": True},
        ]),
        "product_segments": pd.DataFrame([
            {"product": "A", "segment": "high_volume", "n_points": 24, "total_demand": 2400, "zero_ratio": 0.0},
            {"product": "B", "segment": "intermittent", "n_points": 24, "total_demand": 240, "zero_ratio": 0.4},
        ]),
        "best_model_by_product": {"A": "seasonal_naive", "B": "naive"},
        "evaluation_summary": {"mode": "balanced", "n_products_evaluated": 2,
                               "n_products_forecasted": 2, "n_folds": 2, "primary_metric": "wMAPE"},
        "metric_cards": [{"metric": "wMAPE", "value": 0.18, "display_value": "18%",
                          "label": "good", "plain_english": "...", "business_meaning": "..."}],
        "warnings": [],
        "failed_models": [],
        "history": {
            "A": pd.Series([90, 95, 100], index=pd.date_range("2023-10-01", periods=3, freq="MS")),
            "B": pd.Series([19, 21, 20], index=pd.date_range("2023-10-01", periods=3, freq="MS")),
        },
    }


def _quality_with_warnings() -> dict:
    return {"issues": [
        {"issue_type": "missing_periods", "category": "time_series", "severity": "warning",
         "description": "12 missing periods across products."},
        {"issue_type": "negative_demand", "category": "business", "severity": "warning",
         "description": "5 negative demand values."},
        {"issue_type": "invalid_dates", "category": "structural", "severity": "critical",
         "description": "30 unparseable dates."},
    ]}


# ─── Tests ───────────────────────────────────────────────────────────────────

REQUIRED_KEYS = {"status", "executive_summary", "key_findings", "top_growth_products",
                 "high_risk_products", "model_confidence", "forecast_quality_notes",
                 "data_limitations", "recommended_actions", "explainability", "chatbot_context"}


def test_required_keys_minimal():
    out = insights.generate_insights(_minimal_model_result())
    assert REQUIRED_KEYS <= set(out), f"missing: {REQUIRED_KEYS - set(out)}"
    assert out["status"] in ("success", "partial")
    print("✓ 1. required keys present with minimal model_result")


def test_missing_optional_inputs_no_crash():
    # Only model_result; all optional inputs None.
    out = insights.generate_insights(_minimal_model_result(), None, None, None, None)
    assert REQUIRED_KEYS <= set(out)
    # Also: empty model_result → partial, not crash.
    out2 = insights.generate_insights({})
    assert out2["status"] == "partial" and REQUIRED_KEYS <= set(out2)
    print("✓ 2. missing optional inputs (and empty model_result) do not crash")


def test_quality_warnings_reduce_confidence():
    clean = insights.generate_insights(_minimal_model_result())
    noisy = insights.generate_insights(_minimal_model_result(), quality_result=_quality_with_warnings())
    order = {"High": 3, "Medium": 2, "Low": 1}
    assert order[noisy["model_confidence"]["level"]] <= order[clean["model_confidence"]["level"]]
    assert noisy["model_confidence"]["level"] in ("Medium", "Low")
    print(f"✓ 3. quality warnings reduce confidence "
          f"({clean['model_confidence']['level']} → {noisy['model_confidence']['level']})")


def test_high_risk_products_returned():
    out = insights.generate_insights(_minimal_model_result())
    risks = {r["product"]: r for r in out["high_risk_products"]}
    assert "B" in risks, "product B (wMAPE 55%, bias +40%) should be high-risk"
    assert risks["B"]["risk_level"] == "high"
    assert any("error" in d for d in risks["B"]["risk_drivers"])
    # Scenario impact should also surface a product as risky.
    scen = {"impact_summary": {"total_demand_delta_pct": 0.2, "most_affected_products": ["A"]}}
    out2 = insights.generate_insights(_minimal_model_result(), scenario_result=scen)
    assert any(r["product"] == "A" for r in out2["high_risk_products"])
    print("✓ 4. high-risk products returned from error/bias and scenario impact")


def test_explainability_placeholder():
    out = insights.generate_insights(_minimal_model_result(), explainability_result=None)
    ex = out["explainability"]
    assert ex["status"] == "not_available" and ex["method"] is None
    assert ex["top_drivers"] == [] and len(ex["notes"]) >= 2
    # Provided explainability passes through.
    out2 = insights.generate_insights(_minimal_model_result(),
                                      explainability_result={"status": "available",
                                                             "method": "shap",
                                                             "top_drivers": [{"feature": "lag_1"}]})
    assert out2["explainability"]["status"] == "available"
    assert out2["explainability"]["method"] == "shap"
    print("✓ 5. explainability placeholder when None; passthrough when provided")


def test_json_serializable():
    out = insights.generate_insights(_minimal_model_result(),
                                     quality_result=_quality_with_warnings())
    s = json.dumps(out)  # raises if any non-serializable value leaked
    assert isinstance(s, str) and len(s) > 0
    # No pandas Timestamps / numpy types leaked into chatbot_context.
    json.dumps(out["chatbot_context"])
    print("✓ 6. output is JSON serializable")


def test_failure_does_not_break_integration():
    # A genuinely broken model_result (None) → 'partial' valid payload, no raise.
    out = insights.generate_insights(None)  # type: ignore
    assert out["status"] in ("partial", "failed")
    assert REQUIRED_KEYS <= set(out)
    # _failed_result shape is valid and serializable (pipeline uses it).
    fr = insights._failed_result("boom")
    assert fr["status"] == "failed" and REQUIRED_KEYS <= set(fr)
    json.dumps(insights._json_safe(fr))
    print("✓ 7. failure returns a valid non-blocking payload")


if __name__ == "__main__":
    for fn in [test_required_keys_minimal, test_missing_optional_inputs_no_crash,
               test_quality_warnings_reduce_confidence, test_high_risk_products_returned,
               test_explainability_placeholder, test_json_serializable,
               test_failure_does_not_break_integration]:
        fn()
    print("\nALL INSIGHTS TESTS PASSED")
