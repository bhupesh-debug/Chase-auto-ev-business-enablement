"""
Strategy-to-Execution, Stakeholder Governance & EV Partnership ML Engine
Project: chase-auto-ev-business-enablement (JPMC VP - Business Enablement Manager)
Author: Bhupesh Chandra Dimri
"""

import json
import os
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    mean_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from main import train_and_evaluate  # noqa: E402
from subvention_optimizer import optimize_subvention_allocation  # noqa: E402

DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

OEMS = ["Tesla", "Rivian", "Jaguar Land Rover", "Subaru", "Aston Martin", "Maserati", "McLaren"]
FUNCTIONS = [
    "Risk", "Legal", "Compliance", "Controls",
    "Technology", "Product", "Operations", "Finance", "Marketing",
]
PILLARS = [
    "EV Promo Launch",
    "DTC Credit API Integration",
    "Residual Risk Corridor",
    "CIB/Private Bank Cross-Sell",
    "Dealer Floorplan Enablement",
    "QBR Governance Automation",
]


def generate_cross_functional_initiatives(n_samples: int = 1200, seed: int = 42) -> pd.DataFrame:
    """Simulate 1,200 cross-functional JPMC Chase Auto EV milestones with stakeholder friction drivers."""
    rng = np.random.default_rng(seed)
    oem_p = [0.28, 0.24, 0.16, 0.14, 0.06, 0.06, 0.06]
    fn_queue_base = {
        "Risk": 9.5, "Legal": 10.5, "Compliance": 11.0, "Controls": 9.0,
        "Technology": 8.5, "Product": 5.5, "Operations": 6.0, "Finance": 6.5, "Marketing": 4.5,
    }
    rows = []
    for i in range(n_samples):
        oem = rng.choice(OEMS, p=oem_p)
        fn = rng.choice(FUNCTIONS)
        pillar = rng.choice(PILLARS)
        deps = int(rng.integers(1, 9))
        gates = int(rng.integers(1, 6))
        queue_days = float(np.clip(rng.normal(fn_queue_base[fn], 3.2), 1.0, 24.0))
        tech_util = float(np.clip(rng.normal(86.0 if fn == "Technology" else 78.0, 12.0), 50.0, 115.0))
        blockers = int(rng.poisson(1.1 if deps >= 5 else 0.5))
        blockers = min(blockers, 5)
        align_score = float(np.clip(rng.normal(76.0 - 3.5 * blockers - 0.18 * max(0.0, tech_util - 85.0), 9.0), 35.0, 99.0))
        days_to_due = int(rng.integers(4, 61))

        # True structural probability of milestone slippage (independent of self-reported RAG)
        z = (
            -4.2
            + 0.42 * deps
            + 0.35 * gates
            + 0.18 * queue_days
            + 0.065 * max(0.0, tech_util - 80.0)
            + 0.85 * blockers
            - 0.055 * align_score
            + (0.55 if oem in ("Rivian", "Tesla") and pillar == "DTC Credit API Integration" else 0.0)
            + rng.normal(0.0, 0.45)
        )
        prob_slip = float(1.0 / (1.0 + np.exp(-z)))
        slipped = int(rng.binomial(1, prob_slip))
        delay_days = int(max(0, round(prob_slip * 18.0 + rng.normal(0.0, 2.2)))) if slipped else 0

        # Teams frequently over-report Green ("watermelon" reporting) even when blockers/utilization loom
        if prob_slip < 0.35:
            rag = "Green"
        elif prob_slip < 0.65:
            rag = str(rng.choice(["Green", "Amber"], p=[0.42, 0.58]))
        else:
            rag = str(rng.choice(["Green", "Amber", "Red"], p=[0.28, 0.44, 0.28]))

        rows.append({
            "initiative_id": f"INIT-{1000 + i}",
            "oem_partner": str(oem),
            "workstream_function": str(fn),
            "strategic_pillar": str(pillar),
            "dependency_count": deps,
            "approval_gates_count": gates,
            "review_queue_days": round(queue_days, 1),
            "tech_capacity_utilization_pct": round(tech_util, 1),
            "open_raid_blockers": blockers,
            "stakeholder_alignment_score": round(align_score, 1),
            "days_to_target_deadline": days_to_due,
            "self_reported_rag": rag,
            "milestone_slipped": slipped,
            "delay_days": delay_days,
        })
    return pd.DataFrame(rows)


def run_initiative_models(df_init: pd.DataFrame):
    """Benchmark Baseline Logistic Regression vs Random Forest vs Gradient Boosting on slippage risk."""
    num_cols = [
        "dependency_count",
        "approval_gates_count",
        "review_queue_days",
        "tech_capacity_utilization_pct",
        "open_raid_blockers",
        "stakeholder_alignment_score",
        "days_to_target_deadline",
    ]
    X_cat = pd.get_dummies(
        df_init[["oem_partner", "workstream_function", "strategic_pillar"]],
        drop_first=False,
        dtype=float,
    )
    X = pd.concat([df_init[num_cols].astype(float), X_cat], axis=1)
    feature_names = list(X.columns)
    y = df_init["milestone_slipped"].values

    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, df_init.index.values, test_size=0.2, stratify=y, random_state=42
    )

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    models = {
        "LogisticRegression": (LogisticRegression(max_iter=1000, random_state=42), True),
        "RandomForestClassifier": (RandomForestClassifier(n_estimators=150, max_depth=6, random_state=42), False),
        "GradientBoostingClassifier": (GradientBoostingClassifier(n_estimators=150, max_depth=3, random_state=42), False),
    }

    bench = {}
    fitted = {}
    probs_test = {}
    for name, (clf, scaled) in models.items():
        Xtr = X_train_sc if scaled else X_train
        Xte = X_test_sc if scaled else X_test
        clf.fit(Xtr, y_train)
        prob = clf.predict_proba(Xte)[:, 1]
        pred = (prob >= 0.5).astype(int)
        bench[name] = {
            "accuracy": round(float(accuracy_score(y_test, pred)), 4),
            "precision": round(float(precision_score(y_test, pred, zero_division=0)), 4),
            "recall": round(float(recall_score(y_test, pred, zero_division=0)), 4),
            "f1": round(float(f1_score(y_test, pred, zero_division=0)), 4),
            "roc_auc": round(float(roc_auc_score(y_test, prob)), 4),
            "pr_auc": round(float(average_precision_score(y_test, prob)), 4),
        }
        fitted[name] = clf
        probs_test[name] = prob

    best_name = max(bench, key=lambda k: bench[k]["roc_auc"])
    best_clf = fitted[best_name]
    all_probs = best_clf.predict_proba(X_sc := scaler.transform(X) if best_name == "LogisticRegression" else X)[:, 1]

    df_scored = df_init.copy()
    df_scored["predicted_slip_prob"] = np.round(all_probs, 4)
    df_scored["watermelon_flag"] = (
        (df_scored["self_reported_rag"] == "Green") & (df_scored["predicted_slip_prob"] >= 0.70)
    )

    if hasattr(best_clf, "feature_importances_"):
        importances = best_clf.feature_importances_
    else:
        importances = np.abs(best_clf.coef_[0])
        importances = importances / importances.sum()

    top_features = sorted(zip(feature_names, importances.tolist()), key=lambda x: x[1], reverse=True)[:10]

    return {
        "benchmarks": bench,
        "best_model_name": best_name,
        "best_model": best_clf,
        "scaler": scaler,
        "feature_names": feature_names,
        "top_features": top_features,
        "df_scored": df_scored,
        "test_indices": idx_test,
        "test_probs": probs_test,
    }


def run_ev_residual_benchmarks(df_ev: pd.DataFrame):
    """Benchmark Baseline vs RandomForest vs GradientBoosting on EV portfolio data."""
    df = df_ev.copy()
    df["oem_tesla"] = (df["oem_partner"] == "Tesla").astype(int)
    df["oem_rivian"] = (df["oem_partner"] == "Rivian").astype(int)
    df["oem_jlr"] = (df["oem_partner"] == "Jaguar Land Rover").astype(int)
    df["battery_energy_density_proxy"] = df["battery_kwh"] / (df["msrp"] / 1000)

    features = [
        "msrp", "battery_kwh", "lease_term_months", "annual_mileage",
        "contracted_rv_pct", "battery_soh", "ev_market_sentiment_index",
        "oem_subvention_bps", "dealer_tier", "lessee_fico",
        "oem_tesla", "oem_rivian", "oem_jlr", "battery_energy_density_proxy",
    ]
    X = df[features]
    y_rv = df["realized_rv_dollar"]
    y_ret = df["returned_to_bank"]

    Xtr, Xte, ytr_rv, yte_rv = train_test_split(X, y_rv, test_size=0.2, random_state=42)
    Xtr_c, Xte_c, ytr_c, yte_c = train_test_split(X, y_ret, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    Xtr_sc = scaler.fit_transform(Xtr)
    Xte_sc = scaler.transform(Xte)

    # Regression benchmarks (Realized RV $)
    reg_models = {
        "LinearRegression": (LinearRegression(), True),
        "RandomForestRegressor": (RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42), False),
        "GradientBoostingRegressor": (GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=42), False),
    }
    reg_bench = {}
    best_reg = None
    for name, (m, sc) in reg_models.items():
        m.fit(Xtr_sc if sc else Xtr, ytr_rv)
        pred = m.predict(Xte_sc if sc else Xte)
        reg_bench[name] = {
            "mae": round(float(mean_absolute_error(yte_rv, pred)), 2),
            "r2": round(float(r2_score(yte_rv, pred)), 4),
        }
        if name == "GradientBoostingRegressor":
            best_reg = m

    # Classification benchmarks (Lease Return probability)
    clf_models = {
        "LogisticRegression": (LogisticRegression(max_iter=1000, random_state=42), True),
        "RandomForestClassifier": (RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42), False),
        "GradientBoostingClassifier": (GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=42), False),
    }
    clf_bench = {}
    best_clf = None
    for name, (m, sc) in clf_models.items():
        m.fit(Xtr_sc if sc else Xtr_c, ytr_c)
        prob = m.predict_proba(Xte_sc if sc else Xte_c)[:, 1]
        clf_bench[name] = {
            "roc_auc": round(float(roc_auc_score(yte_c, prob)), 4),
            "accuracy": round(float(accuracy_score(yte_c, (prob >= 0.5).astype(int))), 4),
        }
        if name == "RandomForestClassifier":
            best_clf = m

    top_ev_features = sorted(zip(features, best_clf.feature_importances_.tolist()), key=lambda x: x[1], reverse=True)[:10]

    # Scenario simulations on returned EVs under Passive vs 50/50 Risk-Sharing Corridor (beyond 5% MSRP band)
    returned_df = df[df["returned_to_bank"] == 1].copy()
    scenarios = {}
    for label, idx_shift in [("Best Case (+10 Index)", +10.0), ("Base Case (Current)", 0.0), ("Worst Case (-10 Index)", -10.0)]:
        adj_factor = np.clip((returned_df["ev_market_sentiment_index"] + idx_shift) / returned_df["ev_market_sentiment_index"], 0.78, 1.22)
        stressed_rv = np.minimum(returned_df["msrp"] * 0.72, returned_df["realized_rv_dollar"] * adj_factor)
        gross_loss = np.maximum(0.0, returned_df["contracted_rv_dollar"] - stressed_rv)
        corridor_threshold = 0.05 * returned_df["contracted_rv_dollar"]
        excess_loss = np.maximum(0.0, gross_loss - corridor_threshold)
        bank_net_50_50 = gross_loss - 0.50 * excess_loss
        scenarios[label] = {
            "passive_exposure_mm": round(float(gross_loss.sum() / 1e6), 2),
            "corridor_50_50_mm": round(float(bank_net_50_50.sum() / 1e6), 2),
            "savings_mm": round(float((gross_loss.sum() - bank_net_50_50.sum()) / 1e6), 2),
        }

    return {
        "reg_benchmarks": reg_bench,
        "clf_benchmarks": clf_bench,
        "top_features": top_ev_features,
        "scenarios": scenarios,
        "best_reg": best_reg,
        "best_clf": best_clf,
    }


def generate_charts(ev_df: pd.DataFrame, init_res: dict, ev_res: dict):
    """Generate 4 institutional-grade matplotlib charts for README, Deck, and Playbook."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams["font.sans-serif"] = "DejaVu Sans"
    plt.rcParams["axes.edgecolor"] = "#BDC3C7"
    plt.rcParams["axes.linewidth"] = 0.8

    navy, teal, gold, red = "#0B2545", "#117A8B", "#B8860B", "#B03A2E"

    # 1. Feature Importance Comparison (2 panels)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2), dpi=180)
    f_ev = ev_res["top_features"][:8][::-1]
    ax1.barh([x[0] for x in f_ev], [x[1] * 100 for x in f_ev], color=navy)
    ax1.set_title("EV Lease-End Return Drivers (%)", fontsize=11, fontweight="bold", color=navy)
    ax1.set_xlabel("Relative Importance (%)")
    ax1.grid(axis="x", alpha=0.25)

    f_init = init_res["top_features"][:8][::-1]
    ax2.barh([x[0] for x in f_init], [x[1] * 100 for x in f_init], color=teal)
    ax2.set_title("Cross-Functional Milestone Slippage Drivers (%)", fontsize=11, fontweight="bold", color=navy)
    ax2.set_xlabel("Relative Importance (%)")
    ax2.grid(axis="x", alpha=0.25)

    fig.suptitle("Top ML Risk Signals for the Chase Auto EV Brands General Manager", fontsize=13, fontweight="bold", color=navy)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "1_feature_importance.png", bbox_inches="tight")
    plt.close(fig)

    # 2. Model Performance Comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.8), dpi=180)
    m_names = ["Logistic\nBaseline", "Random\nForest", "Gradient\nBoosting"]
    aucs = [
        init_res["benchmarks"]["LogisticRegression"]["roc_auc"],
        init_res["benchmarks"]["RandomForestClassifier"]["roc_auc"],
        init_res["benchmarks"]["GradientBoostingClassifier"]["roc_auc"],
    ]
    bars1 = ax1.bar(m_names, aucs, color=["#7F8C8D", teal, navy], width=0.55)
    ax1.set_ylim(0.5, 1.0)
    ax1.set_title("Initiative Slippage Classifier (ROC-AUC)", fontsize=11, fontweight="bold", color=navy)
    ax1.set_ylabel("Out-of-Sample ROC-AUC")
    ax1.grid(axis="y", alpha=0.25)
    for b, v in zip(bars1, aucs):
        ax1.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.4f}", ha="center", fontsize=9, fontweight="bold")

    r_names = ["Linear\nBaseline", "Random\nForest", "Gradient\nBoosting"]
    maes = [
        ev_res["reg_benchmarks"]["LinearRegression"]["mae"],
        ev_res["reg_benchmarks"]["RandomForestRegressor"]["mae"],
        ev_res["reg_benchmarks"]["GradientBoostingRegressor"]["mae"],
    ]
    bars2 = ax2.bar(r_names, maes, color=["#7F8C8D", teal, navy], width=0.55)
    ax2.set_title("Realized EV Residual Valuation Error (MAE $)", fontsize=11, fontweight="bold", color=navy)
    ax2.set_ylabel("Mean Absolute Error ($ / vehicle)")
    ax2.grid(axis="y", alpha=0.25)
    for b, v in zip(bars2, maes):
        ax2.text(b.get_x() + b.get_width() / 2, v + max(maes) * 0.02, f"${v:,.0f}", ha="center", fontsize=9, fontweight="bold")

    fig.suptitle("Baseline vs. Intermediate vs. Advanced Model Benchmarking [SYN]", fontsize=13, fontweight="bold", color=navy)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "2_model_comparison.png", bbox_inches="tight")
    plt.close(fig)

    # 3. Data Distributions (Residual Variance by OEM + Review Latency by Stakeholder Function)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.0), dpi=180)
    oems_ev = ["Tesla", "Rivian", "Jaguar Land Rover"]
    data_box = [ev_df[ev_df["oem_partner"] == o]["rv_variance_dollar"].values for o in oems_ev]
    bp = ax1.boxplot(data_box, tick_labels=oems_ev, patch_artist=True)
    for patch, col in zip(bp["boxes"], [navy, teal, gold]):
        patch.set_facecolor(col)
        patch.set_alpha(0.75)
    ax1.axhline(0, color=red, linestyle="--", linewidth=1)
    ax1.set_title("Contracted vs. Realized RV Exposure ($ / unit)", fontsize=11, fontweight="bold", color=navy)
    ax1.set_ylabel("Positive = Underwater Loss Risk ($)")
    ax1.grid(axis="y", alpha=0.25)

    scored = init_res["df_scored"]
    fn_means = scored.groupby("workstream_function")["review_queue_days"].mean().sort_values(ascending=True)
    colors_fn = [red if v > 8.5 else teal for v in fn_means.values]
    ax2.barh(fn_means.index, fn_means.values, color=colors_fn)
    ax2.set_title("Avg Governance Review Queue Latency by Team (Days)", fontsize=11, fontweight="bold", color=navy)
    ax2.set_xlabel("Mean Sign-Off Latency (Days)")
    ax2.grid(axis="x", alpha=0.25)

    fig.suptitle("Portfolio Risk & Matrix Bottleneck Telemetry [SYN]", fontsize=13, fontweight="bold", color=navy)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "3_data_distributions.png", bbox_inches="tight")
    plt.close(fig)

    # 4. Scenario Simulation (Passive vs 50/50 Risk-Sharing Corridor)
    fig, ax = plt.subplots(figsize=(10.5, 4.8), dpi=180)
    scens = list(ev_res["scenarios"].keys())
    passive = [ev_res["scenarios"][k]["passive_exposure_mm"] for k in scens]
    corridor = [ev_res["scenarios"][k]["corridor_50_50_mm"] for k in scens]
    x = np.arange(len(scens))
    w = 0.34
    b1 = ax.bar(x - w / 2, passive, w, label="Option A: Passive Acceptance ($M)", color=red, alpha=0.85)
    b2 = ax.bar(x + w / 2, corridor, w, label="Option C: 50/50 Risk-Sharing Corridor ($M)", color=navy)
    ax.set_xticks(x)
    ax.set_xticklabels(scens, fontsize=10, fontweight="bold")
    ax.set_ylabel("Net Chase Auto Residual Loss Exposure ($M)")
    ax.set_title("QBR Scenario Simulation: Residual Exposure Under OEM Risk-Sharing Corridor [SYN]", fontsize=12, fontweight="bold", color=navy)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    for b in list(b1) + list(b2):
        h = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, h + 0.5, f"${h:.2f}M", ha="center", fontsize=8.8, fontweight="bold")
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "4_scenario_simulation.png", bbox_inches="tight")
    plt.close(fig)


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Run core EV lease model from src/main.py
    ev_df, ev_summary = train_and_evaluate()
    ev_df.to_csv(DATA_DIR / "synthetic_ev_portfolio.csv", index=False)

    # 2. Generate and persist subvention allocation table
    subv_df = optimize_subvention_allocation(total_budget_usd=10_000_000, oem="Tesla")
    subv_df.to_csv(DATA_DIR / "oem_subvention_matrix.csv", index=False)

    # 3. Generate cross-functional governance dataset & train models
    init_df = generate_cross_functional_initiatives(n_samples=1200, seed=42)
    init_res = run_initiative_models(init_df)
    scored_df = init_res["df_scored"]
    scored_df.to_csv(DATA_DIR / "cross_functional_initiatives.csv", index=False)

    # 4. Run EV model benchmarks & scenarios
    ev_res = run_ev_residual_benchmarks(ev_df)

    # 5. Generate 4 PNG visuals
    generate_charts(ev_df, init_res, ev_res)

    # 6. Persist trained model bundle & metrics JSON
    model_bundle = {
        "initiative_slippage_model": init_res["best_model"],
        "initiative_scaler": init_res["scaler"],
        "initiative_features": init_res["feature_names"],
        "ev_residual_regressor": ev_res["best_reg"],
        "ev_return_classifier": ev_res["best_clf"],
    }
    with open(MODELS_DIR / "best_model.pkl", "wb") as f:
        pickle.dump(model_bundle, f)

    watermelon_df = scored_df[scored_df["watermelon_flag"]].sort_values("predicted_slip_prob", ascending=False)
    sample_alerts = scored_df.sort_values("predicted_slip_prob", ascending=False).head(40)[
        [
            "initiative_id", "oem_partner", "workstream_function", "strategic_pillar",
            "dependency_count", "review_queue_days", "tech_capacity_utilization_pct",
            "self_reported_rag", "predicted_slip_prob", "watermelon_flag",
        ]
    ].to_dict(orient="records")

    summary_payload = {
        "ev_portfolio": {
            "total_units": int(ev_summary["total_units"]),
            "portfolio_msrp_mm": round(float(ev_summary["portfolio_msrp_mm"]), 2),
            "contracted_rv_mm": round(float(ev_summary["contracted_rv_mm"]), 2),
            "return_rate_pct": round(float(ev_summary["return_rate_pct"]), 1),
            "net_rv_impairment_exposure_mm": round(float(ev_summary["net_rv_impairment_exposure_mm"]), 2),
            "classifier_auc": round(float(ev_summary["classifier_auc"]), 4),
            "regressor_mae": round(float(ev_summary["regressor_mae"]), 2),
            "regressor_r2": round(float(ev_summary["regressor_r2"]), 4),
            "reg_benchmarks": ev_res["reg_benchmarks"],
            "clf_benchmarks": ev_res["clf_benchmarks"],
            "scenarios": ev_res["scenarios"],
        },
        "governance": {
            "total_initiatives": int(len(scored_df)),
            "overall_slippage_rate_pct": round(float(scored_df["milestone_slipped"].mean() * 100), 1),
            "watermelon_count": int(len(watermelon_df)),
            "best_model_name": init_res["best_model_name"],
            "benchmarks": init_res["benchmarks"],
            "top_features": [[k, round(float(v), 4)] for k, v in init_res["top_features"]],
            "sample_alerts": sample_alerts,
        },
        "subvention_tiers": subv_df.to_dict(orient="records"),
    }

    with open(REPORTS_DIR / "ml_metrics_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2)

    # Markdown model report
    best_gov = init_res["benchmarks"][init_res["best_model_name"]]
    report_md = f"""# Chase Auto EV Brands — ML Model & Strategy-to-Execution Verification Report

- **EV Portfolio Size (`[SYN]`):** `{ev_summary['total_units']:,}` leases (`${ev_summary['portfolio_msrp_mm']:.2f}M` MSRP, `${ev_summary['contracted_rv_mm']:.2f}M` Contracted RV)
- **EV Lease-End Return Rate & Exposure:** `{ev_summary['return_rate_pct']:.1f}%` returns | `${ev_summary['net_rv_impairment_exposure_mm']:.2f}M` impairment exposure
- **EV Residual Value Regressor (`GradientBoostingRegressor`):** **`${ev_summary['regressor_mae']:,.2f}` MAE** | **`R^2 = {ev_summary['regressor_r2']:.4f}`** (Baseline Linear Regression MAE: `${ev_res['reg_benchmarks']['LinearRegression']['mae']:,.2f}`)
- **Cross-Functional Initiative Slippage Classifier (`{init_res['best_model_name']}`):** ROC-AUC **`{best_gov['roc_auc']:.4f}`** | PR-AUC `{best_gov['pr_auc']:.4f}` | Accuracy `{best_gov['accuracy']*100:.1f}%`
- **Watermelon Initiatives Flagged (`self_reported_rag == Green` & `pred_prob >= 0.70`):** **`{len(watermelon_df)}`** out of `{len(scored_df)}` tracked milestones
"""
    (REPORTS_DIR / "model_report.md").write_text(report_md, encoding="utf-8")

    print("EXECUTION_GOVERNANCE_ML OK")
    print(json.dumps({
        "ev_mae": summary_payload["ev_portfolio"]["regressor_mae"],
        "ev_r2": summary_payload["ev_portfolio"]["regressor_r2"],
        "ev_exposure_mm": summary_payload["ev_portfolio"]["net_rv_impairment_exposure_mm"],
        "gov_best_model": summary_payload["governance"]["best_model_name"],
        "gov_benchmarks": summary_payload["governance"]["benchmarks"],
        "watermelon_count": summary_payload["governance"]["watermelon_count"],
        "scenarios": summary_payload["ev_portfolio"]["scenarios"],
    }, indent=2))


if __name__ == "__main__":
    main()
