# Chase Auto EV Captive Finance: Executive Business Enablement & Residual Risk Engine (`Chase-Auto-EV-EnablementAI`)

**Author**: Bhupesh Chandra Dimri  
**Target Organization**: JPMorgan Chase & Co. — Consumer & Community Banking (CCB) / Chase Auto  
**Target Role**: Vice President — Business Enablement Manager (Electric Vehicles Brands) | Job ID: `210805542`  
**Location**: 277 Park Ave, New York, NY  

---

## 1. Executive Summary & Problem Overview
Chase Auto operates as the largest bank-owned private label captive finance provider in the United States, managing an **$80B+ asset portfolio** and serving as the exclusive captive partner to 7 global OEMs, including electric mobility leaders **Tesla, Rivian, and Jaguar Land Rover**. 

In high-volume EV leasing, the General Manager (GM) and EV leadership face two critical strategic headwinds:
1. **Residual Value (RV) Volatility & Battery Depreciation**: EV battery health degradation (State of Health / SOH), charging network fragmentation, and aggressive OEM retail price cuts trigger dramatic swings in secondary market wholesale valuations. When wholesale values plunge below contracted lease-end residual values, lessees return vehicles en masse to Chase Auto, creating tens of millions in credit/residual impairment risk.
2. **OEM Subvention vs. Dealer Floorplan Multi-Tier Optimization**: Balancing OEM interest-rate buy-downs (subvention bps) across 75% of U.S. franchised auto dealers while preserving Chase CCB risk and profitability thresholds.

**`Chase-Auto-EV-EnablementAI`** is an executive-grade dual-tier machine learning engine designed for the **VP Business Enablement Manager**. It models 5,000 simulated EV lease contracts ($401.65M portfolio volume), predicting lease-end return probability and forecasting realized wholesale residual values within a $532 MAE margin, empowering the GM to proactively hedge residual loss exposures and optimize OEM incentive allocations during Quarterly Business Reviews (QBRs).

---

## 2. Key Results & Quantitative Impact
- **Portfolio Scale Modeled**: 5,000 active EV lease contracts across Tesla (Model 3, Y, Cybertruck), Rivian (R1T, R1S), and JLR ($401.65M Portfolio MSRP; $225.12M Contracted Residual Value).
- **Residual Value Prediction Accuracy**: Gradient Boosting Regressor achieved **$532.78 MAE** and **$R^2 = 0.9954$** across complex multi-OEM depreciation curves.
- **Lease-End Return Risk Classification**: Random Forest model isolated the primary drivers of vehicle return:
  1. *EV Wholesale Market Sentiment Index* (29.34% importance)
  2. *Contracted RV Percentage* (14.64%)
  3. *Battery State of Health (SOH %)* (14.15%)
  4. *Lessee FICO Score* (10.76%)
  5. *Original Vehicle MSRP* (8.75%)
- **Identified Residual Loss Avoidance**: Highlighted **$19.29M in net impairment exposure** on returned units, providing the GM with empirical data to negotiate risk-sharing reserves in OEM captive contracts.

---

## 3. Project Directory Architecture
```
chase-auto-ev-business-enablement/
├── README.md                           # Comprehensive documentation and executive summary
├── requirements.txt                    # Python dependencies (pandas, scikit-learn, numpy)
├── src/
│   ├── main.py                         # Portfolio simulation, ML training & evaluation
│   ├── data_pipeline.py                # OEM feed ingestion & battery telemetry handler
│   ├── subvention_optimizer.py         # Multi-tier dealer subvention solver
│   └── executive_qbr_exporter.py       # Automated GM QBR deck & memo generator
├── data/
│   ├── synthetic_ev_portfolio.csv      # 5,000 simulated EV contracts
│   └── oem_subvention_matrix.csv       # Rate buy-down schedules
└── reports/
    └── executive_brief_jpmc_chase.md   # Executive 1-page decision memo
```

---

## 4. How to Run
```bash
# Clone the repository
git clone https://github.com/bhupesh-debug/chase-auto-ev-business-enablement.git
cd chase-auto-ev-business-enablement

# Install dependencies
pip install -r requirements.txt

# Execute the simulation and ML engine
python src/main.py
```
