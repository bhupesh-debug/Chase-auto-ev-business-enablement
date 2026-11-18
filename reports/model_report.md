# Chase Auto EV Brands — ML Model & Strategy-to-Execution Verification Report

- **EV Portfolio Size (`[SYN]`):** `5,000` leases (`$401.65M` MSRP, `$225.12M` Contracted RV)
- **EV Lease-End Return Rate & Exposure:** `56.2%` returns | `$19.29M` impairment exposure
- **EV Residual Value Regressor (`GradientBoostingRegressor`):** **`$532.78` MAE** | **`R^2 = 0.9954`** (Baseline Linear Regression MAE: `$921.52`)
- **Cross-Functional Initiative Slippage Classifier (`LogisticRegression`):** ROC-AUC **`0.8705`** | PR-AUC `0.6141` | Accuracy `89.6%`
- **Watermelon Initiatives Flagged (`self_reported_rag == Green` & `pred_prob >= 0.70`):** **`18`** out of `1200` tracked milestones
