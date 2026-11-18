"""Pytest suite for Chase Auto EV Business Enablement & Governance ML Engine."""
import json
import pickle
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from main import generate_synthetic_ev_portfolio, train_and_evaluate  # noqa: E402
from subvention_optimizer import optimize_subvention_allocation  # noqa: E402


def test_ev_portfolio_mae_and_r2():
    df, res = train_and_evaluate()
    assert len(df) == 5000
    assert set(["Tesla", "Rivian", "Jaguar Land Rover"]).issubset(set(df["oem_partner"].unique()))
    assert res["regressor_mae"] < 600.0
    assert abs(res["regressor_mae"] - 532.78) < 1.0
    assert res["regressor_r2"] > 0.99
    assert 0.60 < res["classifier_auc"] < 0.85


def test_subvention_optimizer_budget_conservation():
    budget = 10_000_000
    df = optimize_subvention_allocation(total_budget_usd=budget, oem="Tesla")
    assert len(df) == 4
    assert abs(df["allocated_subvention_usd"].sum() - budget) < 1.0


def test_generated_datasets_models_and_charts():
    assert (ROOT / "data" / "synthetic_ev_portfolio.csv").is_file()
    assert (ROOT / "data" / "cross_functional_initiatives.csv").is_file()
    assert (ROOT / "data" / "oem_subvention_matrix.csv").is_file()

    init_df = pd.read_csv(ROOT / "data" / "cross_functional_initiatives.csv")
    assert len(init_df) == 1200
    assert "predicted_slip_prob" in init_df.columns
    assert "watermelon_flag" in init_df.columns

    metrics_path = ROOT / "reports" / "ml_metrics_summary.json"
    assert metrics_path.is_file()
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert metrics["governance"]["best_model_name"] in (
        "LogisticRegression", "RandomForestClassifier", "GradientBoostingClassifier"
    )
    assert metrics["governance"]["benchmarks"][metrics["governance"]["best_model_name"]]["roc_auc"] > 0.80

    model_path = ROOT / "models" / "best_model.pkl"
    assert model_path.is_file()
    with open(model_path, "rb") as f:
        bundle = pickle.load(f)
    assert "initiative_slippage_model" in bundle
    assert "ev_residual_regressor" in bundle

    for fig_name in [
        "1_feature_importance.png",
        "2_model_comparison.png",
        "3_data_distributions.png",
        "4_scenario_simulation.png",
    ]:
        fp = ROOT / "reports" / "figures" / fig_name
        assert fp.is_file() and fp.stat().st_size > 10_000


def test_dashboard_and_devtools_screenshot():
    dash = ROOT / "dashboard" / "index.html"
    assert dash.is_file()
    html = dash.read_text(encoding="utf-8")
    assert "devtools-status" in html
    assert "Chase Auto" in html

    shot = ROOT / "reports" / "figures" / "5_dashboard_devtools_screenshot.png"
    assert shot.is_file() and shot.stat().st_size > 20_000
