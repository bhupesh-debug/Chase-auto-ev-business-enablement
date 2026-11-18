# Chase Auto EV Captive Finance & Strategy-to-Execution ML Suite (`Chase-auto-ev-business-enablement`)

**Author**: Bhupesh Chandra Dimri  
**Target Organization**: JPMorgan Chase & Co. — Consumer & Community Banking (CCB) / Chase Auto  
**Target Role**: Vice President — Business Enablement Manager (Electric Vehicles Brands) | Job ID: `210805542`  
**Location**: 277 Park Ave, New York, NY  

---

## 1. Executive Summary & Problem Overview
Chase Auto operates as the largest bank-owned private label captive finance provider in the United States, managing an **$80B+ asset portfolio** across **75% of U.S. franchised dealers** and serving as the captive partner to seven auto manufacturers: **Subaru, Jaguar Land Rover, Aston Martin, Maserati, McLaren, Rivian, and Tesla** `[JD]`.

In executing the EV Brands VP Business Enablement mandate (GM advisor, cross-functional accountability across 9+ internal teams, executive preparedness, and QBR negotiation support), leadership faces three analytical bottlenecks:
1. **Cross-Functional Strategy-to-Execution Slippage ("Watermelon" Reporting)**: Multi-workstream initiatives across Risk, Legal, Compliance, Controls, Technology, Product, Operations, Finance, and Marketing often remain self-reported `Green` until approval queues or Tech capacity bottlenecks trigger late-stage misses.
2. **EV Residual Value (RV) Volatility & Battery Depreciation**: Rapid OEM price cuts and battery State-of-Health (`SOH`) declines drive lease-end return spikes and residual impairment when auction values fall below contracted residuals.
3. **OEM Subvention & Dealer Floorplan Allocation**: Balancing rate buy-down pools across dealer tiers while protecting bank economics via OEM risk-sharing corridors.

**`Chase-Auto-EV-EnablementAI`** delivers a verified 3-model Python ML pipeline (`src/execution_governance_ml.py` + `src/main.py`), an offline **Executive Command Center SPA** audited with headless Chromium DevTools (`dashboard/index.html`), 4 visual intelligence figures (`reports/figures/`), a 10-slide presentation deck (`slides/project_deck.pptx`), and automated `pytest` checks (`tests/test_ml_pipeline.py`). All simulation inputs are synthetic (`[SYN]`).

---

## 2. Verified Model Benchmarks & Business Impact (`[SYN]`)

| Model / Analysis | Best Model | Benchmark Performance | Business Decision Enabled |
|---|---|---|---|
| **Problem 1: Cross-Functional Milestone Slippage** | `LogisticRegression` (vs. RF `0.8610`, GBC `0.8569`) | **ROC-AUC `0.8705`** (`89.6%` Acc; GBC PR-AUC `0.6523`) | Flags **18 "Watermelon" milestones** (self-reported `Green` with `>=70%` ML slippage probability) out of `1,200` tracked milestones for GM pre-wiring |
| **Problem 2A: Realized EV Residual Value** | `GradientBoostingRegressor` (vs. Linear `$921.52`, RF `$1,312.11`) | **`$532.78 MAE`** (**$R^2 = 0.9954$**) | Forecasts lease-end wholesale auction value across `5,000` simulated EV leases (`$401.65M` MSRP) |
| **Problem 2B: Lease-End Return Propensity** | `LogisticRegression` / `RandomForestClassifier` | `0.6832` LogReg AUC (`0.6440` RF AUC) | Identifies `$19.29M` net impairment exposure across `56.2%` projected lease returns (`$225.12M` contracted RV) |
| **Problem 3: QBR Subvention & Risk Corridor** | Elasticity & Scenario Simulation | Base savings **`$6.98M`** (`$19.73M -> $12.74M`); Worst **`$12.36M`** | Supports GM push for a 50/50 OEM risk-sharing corridor beyond 5% variance + `$10M` Tier 1–4 subvention split (`+19.25%` Tier-1 lift) |

---

## 3. Repository Architecture
```text
chase-auto-ev-business-enablement/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── synthetic_ev_portfolio.csv          # 5,000 simulated EV leases (Tesla, Rivian, JLR)
│   ├── cross_functional_initiatives.csv    # 1,200 scored milestones (7 OEMs x 9 JPMC functions)
│   └── oem_subvention_matrix.csv           # Dealer Tier 1-4 elasticity allocation schedule
├── src/
│   ├── main.py                             # Core EV lease return & RV valuation engine ($532.78 MAE)
│   ├── execution_governance_ml.py          # 3-model benchmarking, watermelon detection & 4 PNG charts
│   ├── data_pipeline.py                    # OEM battery telemetry ingestion handler
│   ├── subvention_optimizer.py             # Multi-tier dealer subvention optimizer
│   └── executive_qbr_exporter.py           # Automated GM QBR markdown exporter
├── models/
│   └── best_model.pkl                      # Persisted sklearn model & scaler bundle
├── dashboard/
│   └── index.html                          # Offline Chrome DevTools-verified GM Command Center SPA
├── reports/
│   ├── executive_brief_jpmc_chase.md       # 1-page Executive Decision Memorandum
│   ├── model_report.md                     # Quantitative model benchmark report
│   ├── ml_metrics_summary.json             # Single-source verified metrics JSON
│   └── figures/
│       ├── 1_feature_importance.png        # Top EV return & initiative slippage drivers
│       ├── 2_model_comparison.png          # Baseline vs. RF vs. Gradient Boosting comparison
│       ├── 3_data_distributions.png        # OEM RV variance & JPMC function queue latency
│       ├── 4_scenario_simulation.png       # Passive vs. 50/50 QBR Risk-Sharing Corridor
│       └── 5_dashboard_devtools_screenshot.png # Headless Chromium DevTools render
├── slides/
│   └── project_deck.pptx                   # 10-slide executive presentation
└── tests/
    └── test_ml_pipeline.py                 # Automated pytest suite
```

---

## 4. How to Run & Verify
```bash
# Clone the repository
git clone https://github.com/bhupesh-debug/Chase-auto-ev-business-enablement.git
cd Chase-auto-ev-business-enablement

# Install dependencies
pip install -r requirements.txt

# 1. Run core EV valuation & return engine ($532.78 MAE)
python src/main.py

# 2. Run expanded Strategy-to-Execution governance ML suite, benchmarks & 4 charts
python src/execution_governance_ml.py

# 3. Run automated verification test suite
pytest tests/ -v

# 4. Open interactive GM Command Center locally
# Open dashboard/index.html in Chrome/Edge (DevTools telemetry runs automatically)
```
