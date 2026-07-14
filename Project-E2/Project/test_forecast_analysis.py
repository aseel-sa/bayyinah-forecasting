"""
test_forecast_analysis.py — اختبارات وحدة تحليل التوقّعات | Bayyina

Covers BOTH capabilities of forecast_analysis.py independently:
  Section 1 — Forecast Performance Analysis (measured: actual vs predicted vs error)
  Section 2 — What-if / Scenario Simulation (assumption-based adjustments)

Test design rules (approved):
  * Each section's tests import ONLY its own entry point — deleting one section
    must leave the other section's tests green (removability requirement).
  * Spec-based math: expected values are recomputed in the test from the
    documented contract, never copied from the implementation.
  * Two fixture tiers: a hand-built minimal model_result for exact arithmetic,
    and a real end-to-end pipeline run for integration/traceability.
  * stdlib unittest only — no new dependencies.

Run:  python -m unittest test_forecast_analysis -v
"""

import copy
import unittest

import numpy as np
import pandas as pd

import pipeline
from forecast_analysis import analyze_forecast_performance, run_scenario


# ─── Fixtures ─────────────────────────────────────────────────────────────────

def _fixture_df() -> pd.DataFrame:
    """36 months x 3 products, seeded: P1 summer-peak high volume,
    P2 stable mid volume, P3 intermittent low volume. (Same recipe as the
    golden regression snapshot.)"""
    rng = np.random.RandomState(7)
    dates = pd.date_range("2022-01-01", periods=36, freq="MS")
    rows = []
    for dt in dates:
        summer = 1.0 if dt.month in (6, 7, 8) else 0.0
        rows.append({"Date": dt, "Product": "P1",
                     "Quantity": 1000 + 400 * summer + rng.randint(-50, 51)})
        rows.append({"Date": dt, "Product": "P2",
                     "Quantity": 500 + rng.randint(-30, 31)})
        q3 = 0 if rng.rand() < 0.4 else 40 + rng.randint(0, 20)
        rows.append({"Date": dt, "Product": "P3", "Quantity": q3})
    return pd.DataFrame(rows)


MODEL_RESULT = None     # populated once by setUpModule (real pipeline output)
PIPELINE_RESULT = None  # the full pipeline output (for integration tests)


def setUpModule():
    global MODEL_RESULT, PIPELINE_RESULT
    res = pipeline.run_pipeline(_fixture_df(), {"granularity": "monthly",
                                                "auto_approve_quality": True})
    assert res["status"] == "complete", f"fixture pipeline failed: {res['errors']}"
    PIPELINE_RESULT = res
    MODEL_RESULT = res["model_result"]


def _tiny_perf_model_result() -> dict:
    """Hand-built minimal model_result for exact Section-1 arithmetic,
    including a zero-actual row (error_pct must be NaN there)."""
    vp = pd.DataFrame([
        {"product": "P1", "model_name": "naive", "fold": 0, "horizon_step": 1,
         "date": pd.Timestamp("2025-01-01"), "actual": 100.0,
         "validation_prediction": 110.0},   # over,  error +10, pct +10
        {"product": "P1", "model_name": "naive", "fold": 0, "horizon_step": 2,
         "date": pd.Timestamp("2025-02-01"), "actual": 200.0,
         "validation_prediction": 180.0},   # under, error -20, pct -10
        {"product": "P2", "model_name": "naive", "fold": 0, "horizon_step": 1,
         "date": pd.Timestamp("2025-01-01"), "actual": 0.0,
         "validation_prediction": 5.0},     # over,  error +5,  pct NaN
        {"product": "P2", "model_name": "naive", "fold": 0, "horizon_step": 2,
         "date": pd.Timestamp("2025-02-01"), "actual": 50.0,
         "validation_prediction": 50.0},    # exact, error 0,   pct 0
    ])
    return {
        "validation_predictions": vp,
        "metric_cards": [{"metric": "wMAPE", "value": 0.1,
                          "display_value": "10%", "label": "excellent",
                          "plain_english": "x", "business_meaning": "y"}],
        "forecast_summary": {"portfolio_wmape": 0.1,
                             "forecast_bias_direction": "balanced",
                             "confidence_label": "excellent"},
        "evaluation_summary": {"mode": "balanced"},
    }


# ═══ Section 1 — Forecast Performance Analysis ════════════════════════════════

class TestForecastPerformanceExact(unittest.TestCase):
    """Exact arithmetic on the hand-built fixture."""

    def setUp(self):
        self.mr = _tiny_perf_model_result()
        self.out = analyze_forecast_performance(self.mr)

    def test_analysis_type_discriminator(self):
        self.assertEqual(self.out["analysis_type"], "forecast_performance_measured")

    def test_per_period_row_math(self):
        pp = self.out["per_period"].set_index(["product", "date"]).sort_index()
        r = pp.loc[("P1", pd.Timestamp("2025-01-01"))]
        self.assertAlmostEqual(float(r["error"]), 10.0)
        self.assertAlmostEqual(float(r["error_pct"]), 10.0)
        self.assertEqual(r["direction"], "over")
        r = pp.loc[("P1", pd.Timestamp("2025-02-01"))]
        self.assertAlmostEqual(float(r["error"]), -20.0)
        self.assertAlmostEqual(float(r["error_pct"]), -10.0)
        self.assertEqual(r["direction"], "under")

    def test_error_pct_nan_when_actual_zero(self):
        pp = self.out["per_period"].set_index(["product", "date"]).sort_index()
        r = pp.loc[("P2", pd.Timestamp("2025-01-01"))]
        self.assertTrue(np.isnan(float(r["error_pct"])))
        self.assertEqual(r["direction"], "over")   # direction still defined

    def test_exact_direction(self):
        pp = self.out["per_period"].set_index(["product", "date"]).sort_index()
        r = pp.loc[("P2", pd.Timestamp("2025-02-01"))]
        self.assertAlmostEqual(float(r["error"]), 0.0)
        self.assertEqual(r["direction"], "exact")

    def test_direction_counts(self):
        s = self.out["summary"]
        self.assertEqual(s["n_over_forecast"], 2)
        self.assertEqual(s["n_under_forecast"], 1)
        self.assertEqual(s["n_exact"], 1)
        self.assertEqual(s["n_validation_rows"], 4)
        self.assertEqual(s["n_products_evaluated"], 2)
        self.assertEqual(s["n_periods_evaluated"], 2)

    def test_total_aggregation(self):
        tot = self.out["per_period_total"].set_index("date").sort_index()
        d1 = tot.loc[pd.Timestamp("2025-01-01")]
        self.assertAlmostEqual(float(d1["actual"]), 100.0)
        self.assertAlmostEqual(float(d1["predicted"]), 115.0)
        self.assertAlmostEqual(float(d1["error"]), 15.0)
        self.assertEqual(d1["direction"], "over")
        self.assertEqual(int(d1["n_products"]), 2)

    def test_summary_echoes_not_recomputes(self):
        s = self.out["summary"]
        self.assertEqual(s["portfolio_wmape"], 0.1)
        self.assertEqual(s["forecast_bias_direction"], "balanced")
        self.assertEqual(s["confidence_label"], "excellent")
        self.assertEqual(s["metric_cards"], self.mr["metric_cards"])
        self.assertIsNot(s["metric_cards"], self.mr["metric_cards"])  # copy, not ref


class TestForecastPerformanceIntegration(unittest.TestCase):
    """Against the real pipeline output."""

    def test_traceability_one_row_per_validation_row(self):
        out = analyze_forecast_performance(MODEL_RESULT)
        vp = MODEL_RESULT["validation_predictions"]
        self.assertEqual(len(out["per_period"]), len(vp))
        # Every error is exactly predicted - actual from the source rows.
        merged = out["per_period"]
        self.assertTrue(np.allclose(merged["error"],
                                    merged["predicted"] - merged["actual"]))

    def test_input_not_mutated(self):
        vp_before = MODEL_RESULT["validation_predictions"].copy(deep=True)
        cards_before = copy.deepcopy(MODEL_RESULT["metric_cards"])
        analyze_forecast_performance(MODEL_RESULT)
        pd.testing.assert_frame_equal(MODEL_RESULT["validation_predictions"], vp_before)
        self.assertEqual(MODEL_RESULT["metric_cards"], cards_before)

    def test_degrades_without_validation_history(self):
        mr = {"validation_predictions": pd.DataFrame(),
              "metric_cards": [], "forecast_summary": {}}
        out = analyze_forecast_performance(mr)
        self.assertEqual(out["analysis_type"], "forecast_performance_measured")
        self.assertEqual(len(out["per_period"]), 0)
        self.assertEqual(len(out["per_period_total"]), 0)
        self.assertTrue(any(w["level"] == "warning" for w in out["warnings"]))

    def test_degrades_when_key_missing_entirely(self):
        out = analyze_forecast_performance({})
        self.assertEqual(len(out["per_period"]), 0)
        self.assertTrue(out["warnings"])


# ═══ Section 2 — What-if / Scenario Simulation ════════════════════════════════

def _expected_transform(baseline: pd.DataFrame, product: str,
                        factor: float = 1.0, shift: int = 0,
                        mult: float = 1.0) -> np.ndarray:
    """Spec-based expected scenario values for one product:
    level + roll(factor * (f - level), shift), then * mult, then clip >= 0.
    Order matters and is part of the documented contract."""
    g = baseline[baseline["product"] == product].sort_values("horizon_step")
    f = g["forecast"].to_numpy(dtype=float)
    level = f.mean()
    dev = np.roll(factor * (f - level), shift)
    return np.clip(mult * (level + dev), 0, None)


class TestScenarioSimulation(unittest.TestCase):

    def setUp(self):
        self.baseline = MODEL_RESULT["product_forecasts"].copy(deep=True)

    def _scen(self, scenario):
        return run_scenario(MODEL_RESULT, scenario)

    def test_analysis_type_discriminator(self):
        out = self._scen({})
        self.assertEqual(out["analysis_type"], "scenario_simulation_assumption")

    def test_identity_scenario_equals_baseline(self):
        out = self._scen({})
        self.assertTrue(np.allclose(out["scenario_forecast"]["forecast"],
                                    out["baseline_forecast"]["forecast"]))
        comp = out["scenario_comparison"]
        self.assertTrue(np.allclose(comp["delta"], 0.0))

    def test_global_increase_10_pct(self):
        out = self._scen({"demand_change_pct": 10})
        comp = out["scenario_comparison"]
        self.assertTrue(np.allclose(comp["scenario_total"],
                                    comp["baseline_total"] * 1.10))
        valid = comp["baseline_total"] != 0
        self.assertTrue(np.allclose(comp.loc[valid, "delta_pct"], 10.0))

    def test_global_decrease_15_pct(self):
        out = self._scen({"demand_change_pct": -15})
        comp = out["scenario_comparison"]
        self.assertTrue(np.allclose(comp["scenario_total"],
                                    comp["baseline_total"] * 0.85))
        self.assertTrue((out["scenario_forecast"]["forecast"] >= 0).all())

    def test_per_product_override(self):
        out = self._scen({"demand_change_pct": 10,
                          "per_product_change_pct": {"P1": 20}})
        sf = out["scenario_forecast"]
        for prod, mult in (("P1", 1.20), ("P2", 1.10), ("P3", 1.10)):
            exp = _expected_transform(self.baseline, prod, mult=mult)
            got = (sf[sf["product"] == prod].sort_values("horizon_step")
                   ["forecast"].to_numpy(dtype=float))
            self.assertTrue(np.allclose(got, exp), f"product {prod}")

    def test_seasonal_strength_amplification(self):
        out = self._scen({"seasonal_strength_factor": 1.5})
        sf = out["scenario_forecast"]
        for prod in ("P1", "P2", "P3"):
            exp = _expected_transform(self.baseline, prod, factor=1.5)
            got = (sf[sf["product"] == prod].sort_values("horizon_step")
                   ["forecast"].to_numpy(dtype=float))
            self.assertTrue(np.allclose(got, exp), f"product {prod}")
        # Amplification preserves each product's mean level (mean(dev)=0),
        # so totals stay ~equal when nothing was clipped at zero.
        if (sf["forecast"] > 0).all():
            self.assertAlmostEqual(float(sf["forecast"].sum()),
                                   float(self.baseline["forecast"].sum()),
                                   places=6)
        # P1 is seasonal: variation must increase.
        b1 = self.baseline[self.baseline["product"] == "P1"]["forecast"]
        s1 = sf[sf["product"] == "P1"]["forecast"]
        self.assertGreater(float(s1.std()), float(b1.std()) * 1.01)

    def test_peak_shift(self):
        out = self._scen({"peak_shift_periods": 2})
        sf = out["scenario_forecast"]
        for prod in ("P1", "P2", "P3"):
            exp = _expected_transform(self.baseline, prod, shift=2)
            got = (sf[sf["product"] == prod].sort_values("horizon_step")
                   ["forecast"].to_numpy(dtype=float))
            self.assertTrue(np.allclose(got, exp), f"product {prod}")

    def test_combined_order_seasonal_then_multiplier(self):
        out = self._scen({"demand_change_pct": 10,
                          "seasonal_strength_factor": 1.5,
                          "peak_shift_periods": 1})
        sf = out["scenario_forecast"]
        for prod in ("P1", "P2", "P3"):
            exp = _expected_transform(self.baseline, prod,
                                      factor=1.5, shift=1, mult=1.10)
            got = (sf[sf["product"] == prod].sort_values("horizon_step")
                   ["forecast"].to_numpy(dtype=float))
            self.assertTrue(np.allclose(got, exp), f"product {prod}")

    def test_baseline_never_mutated(self):
        before = MODEL_RESULT["product_forecasts"].copy(deep=True)
        tot_before = MODEL_RESULT["total_forecast"].copy(deep=True)
        self._scen({"demand_change_pct": 50, "seasonal_strength_factor": 2.0,
                    "peak_shift_periods": 3})
        pd.testing.assert_frame_equal(MODEL_RESULT["product_forecasts"], before)
        pd.testing.assert_frame_equal(MODEL_RESULT["total_forecast"], tot_before)

    def test_baseline_echo_is_a_copy(self):
        out = self._scen({})
        pd.testing.assert_frame_equal(out["baseline_forecast"],
                                      MODEL_RESULT["product_forecasts"])
        self.assertIsNot(out["baseline_forecast"], MODEL_RESULT["product_forecasts"])

    def test_mandatory_assumption_warning_always_present(self):
        for scenario in ({}, {"demand_change_pct": 10}):
            out = self._scen(scenario)
            self.assertTrue(
                any(w["affected_scope"] == "scenario"
                    and w["affected_item"] == "disclaimer" for w in out["warnings"]),
                "assumption-not-prediction disclaimer missing")

    def test_bounds_inherited_parallel_shift(self):
        out = self._scen({"demand_change_pct": 10})
        sf = out["scenario_forecast"].sort_values(["product", "horizon_step"])
        bf = out["baseline_forecast"].sort_values(["product", "horizon_step"])
        fin = (np.isfinite(bf["lower_bound"].to_numpy(dtype=float))
               & (sf["lower_bound"].to_numpy(dtype=float) > 0))
        if fin.any():
            width_b = (bf["upper_bound"] - bf["lower_bound"]).to_numpy(dtype=float)[fin]
            width_s = (sf["upper_bound"] - sf["lower_bound"]).to_numpy(dtype=float)[fin]
            self.assertTrue(np.allclose(width_s, width_b),
                            "bounds must shift in parallel (inherited width)")

    def test_impact_summary_math(self):
        out = self._scen({"demand_change_pct": 10})
        imp = out["impact_summary"]
        base_sum = float(self.baseline["forecast"].sum())
        self.assertAlmostEqual(imp["total_demand_baseline"], base_sum, places=4)
        self.assertAlmostEqual(imp["total_demand_delta"], base_sum * 0.10, places=4)
        self.assertAlmostEqual(imp["total_demand_delta_pct"], 10.0, places=6)
        self.assertIsNotNone(imp["peak_period_baseline"])
        affected = imp["most_affected_products"]
        self.assertLessEqual(len(affected), 5)
        deltas = [abs(a["delta"]) for a in affected]
        self.assertEqual(deltas, sorted(deltas, reverse=True))

    def test_assumptions_echo_and_method(self):
        out = self._scen({"name": "demo", "demand_change_pct": 10})
        a = out["scenario_assumptions"]
        self.assertEqual(a["method"], "post_forecast_adjustment")
        self.assertEqual(a["name"], "demo")
        self.assertEqual(a["demand_change_pct"], 10)
        self.assertEqual(a["seasonal_strength_factor"], 1.0)   # resolved default
        self.assertEqual(a["peak_shift_periods"], 0)
        self.assertIn("application_order", a)

    def test_unknown_product_override_warns_and_is_ignored(self):
        out = self._scen({"per_product_change_pct": {"NOPE": 50}})
        self.assertTrue(any(w["affected_item"] == "NOPE" for w in out["warnings"]))
        self.assertTrue(np.allclose(out["scenario_forecast"]["forecast"],
                                    out["baseline_forecast"]["forecast"]))

    def test_invalid_inputs_fall_back_to_defaults_with_warning(self):
        out = self._scen({"seasonal_strength_factor": -2,
                          "demand_change_pct": "abc",
                          "bogus_key": 1})
        self.assertTrue(np.allclose(out["scenario_forecast"]["forecast"],
                                    out["baseline_forecast"]["forecast"]))
        self.assertGreaterEqual(
            len([w for w in out["warnings"] if w["level"] == "warning"]), 1)

    def test_degrades_on_empty_model_result(self):
        out = run_scenario({}, {"demand_change_pct": 10})
        self.assertEqual(out["analysis_type"], "scenario_simulation_assumption")
        self.assertEqual(len(out["scenario_forecast"]), 0)
        self.assertTrue(out["warnings"])


# ═══ Pipeline integration (the forecast_performance tail stage) ═══════════════

class TestPipelineIntegration(unittest.TestCase):
    """The non-fatal forecast_performance stage added to _POST_QUALITY_STAGES."""

    def test_performance_result_in_pipeline_output(self):
        perf = PIPELINE_RESULT.get("performance_result")
        self.assertIsNotNone(perf)
        self.assertEqual(perf["analysis_type"], "forecast_performance_measured")
        self.assertEqual(len(perf["per_period"]),
                         len(MODEL_RESULT["validation_predictions"]))

    def test_stage_logged_and_pipeline_complete(self):
        stages = {s["stage"]: s["status"] for s in PIPELINE_RESULT["stage_log"]}
        self.assertEqual(stages.get("forecast_performance"), "ok")
        self.assertEqual(PIPELINE_RESULT["status"], "complete")

    def test_existing_contract_keys_preserved(self):
        for key in ("intake_result", "quality_result", "external_result",
                    "feature_result", "model_result", "errors", "warnings",
                    "stage_log", "_pipeline_state"):
            self.assertIn(key, PIPELINE_RESULT)
        for key in ("forecast_summary", "product_forecasts", "total_forecast",
                    "chart_data", "validation_predictions", "model_leaderboard",
                    "metric_cards", "history", "forecasts", "comparison_table",
                    "best_model"):
            self.assertIn(key, MODEL_RESULT)

    def test_stage_failure_does_not_corrupt_forecast(self):
        """A crashing analysis stage must degrade to a warning — the pipeline
        completes and model_result is untouched (non-fatal requirement)."""
        from unittest.mock import patch
        with patch.object(pipeline.fa, "analyze_forecast_performance",
                          side_effect=RuntimeError("boom")):
            res = pipeline.run_pipeline(_fixture_df(),
                                        {"granularity": "monthly",
                                         "auto_approve_quality": True,
                                         "evaluation_mode": "fast"})
        self.assertEqual(res["status"], "complete")
        self.assertEqual(res["performance_result"], {})
        self.assertTrue(any(w["stage"] == "forecast_performance"
                            for w in res["warnings"]))
        self.assertGreater(len(res["model_result"]["product_forecasts"]), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
