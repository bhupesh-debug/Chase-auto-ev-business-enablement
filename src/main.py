"""
Chase Auto EV Captive Finance - Business Enablement & Residual Risk Analytics Engine
Project: chase-auto-ev-business-enablement
Author: Bhupesh Chandra Dimri
"""

import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, mean_absolute_error, r2_score

np.random.seed(42)

def generate_synthetic_ev_portfolio(n_samples=5000):
    oems = ['Tesla', 'Rivian', 'Jaguar Land Rover']
    oem_weights = [0.60, 0.30, 0.10]
    
    selected_oems = np.random.choice(oems, size=n_samples, p=oem_weights)
    
    data = []
    for i, oem in enumerate(selected_oems):
        if oem == 'Tesla':
            model = np.random.choice(['Model 3', 'Model Y', 'Cybertruck'], p=[0.4, 0.5, 0.1])
            msrp = np.random.uniform(42000, 95000) if model != 'Cybertruck' else np.random.uniform(80000, 115000)
            battery_kwh = np.random.choice([60, 75, 82, 123])
        elif oem == 'Rivian':
            model = np.random.choice(['R1T', 'R1S'], p=[0.45, 0.55])
            msrp = np.random.uniform(74000, 108000)
            battery_kwh = np.random.choice([105, 135, 149])
        else: # JLR
            model = np.random.choice(['I-PACE', 'Range Rover EV'], p=[0.6, 0.4])
            msrp = np.random.uniform(72000, 135000)
            battery_kwh = np.random.choice([90, 105])
            
        lease_term = np.random.choice([24, 36, 48], p=[0.15, 0.70, 0.15])
        annual_mileage = np.random.choice([10000, 12000, 15000], p=[0.3, 0.5, 0.2])
        contracted_rv_pct = np.clip(np.random.normal(0.56 if lease_term == 36 else (0.65 if lease_term == 24 else 0.46), 0.04), 0.38, 0.75)
        contracted_rv_dollar = msrp * contracted_rv_pct
        
        # Battery Health & Tech Obsolescence factors
        battery_soh = np.clip(np.random.normal(0.92, 0.04), 0.78, 0.99)
        supercharger_speed_factor = 1.0 if oem == 'Tesla' else np.random.uniform(0.85, 0.98)
        ev_market_sentiment = np.random.normal(95.0, 8.0) # Manheim EV sub-index (base 100)
        
        # OEM Subvention (bps interest rate buy-down paid by OEM to Chase Auto)
        oem_subvention_bps = np.random.choice([150, 250, 350, 450], p=[0.25, 0.40, 0.25, 0.10])
        dealer_tier = np.random.choice([1, 2, 3, 4], p=[0.20, 0.45, 0.25, 0.10])
        lessee_fico = np.random.randint(660, 840)
        
        # True market value at lease end (Realized RV)
        # Driven by battery SOH, market sentiment, mileage, and OEM brand retention
        depreciation_factor = (0.50 if lease_term == 36 else (0.61 if lease_term == 24 else 0.40))
        realized_rv_pct = depreciation_factor * (battery_soh / 0.92) * (ev_market_sentiment / 100.0) + (0.03 if oem == 'Tesla' else 0.0)
        realized_rv_pct = np.clip(realized_rv_pct, 0.30, 0.72)
        realized_rv_dollar = msrp * realized_rv_pct
        
        # RV Loss per unit when returned: Contracted RV - Realized Market RV
        rv_variance_dollar = contracted_rv_dollar - realized_rv_dollar
        
        # Return probability: if market value < contracted RV (lessee is underwater), they return to Chase Auto
        underwater_gap = (contracted_rv_dollar - realized_rv_dollar) / msrp
        return_logit = 1.2 * underwater_gap * 10.0 + (0.3 if lessee_fico > 760 else -0.2) - 0.5
        prob_return = 1.0 / (1.0 + np.exp(-return_logit))
        returned = int(np.random.binomial(1, prob_return))
        
        data.append({
            'contract_id': f'CA-EV-{100000+i}',
            'oem_partner': oem,
            'model': model,
            'msrp': round(msrp, 2),
            'battery_kwh': battery_kwh,
            'lease_term_months': lease_term,
            'annual_mileage': annual_mileage,
            'contracted_rv_pct': round(contracted_rv_pct, 4),
            'contracted_rv_dollar': round(contracted_rv_dollar, 2),
            'battery_soh': round(battery_soh, 4),
            'ev_market_sentiment_index': round(ev_market_sentiment, 2),
            'oem_subvention_bps': oem_subvention_bps,
            'dealer_tier': dealer_tier,
            'lessee_fico': lessee_fico,
            'realized_rv_dollar': round(realized_rv_dollar, 2),
            'rv_variance_dollar': round(rv_variance_dollar, 2),
            'returned_to_bank': returned
        })
        
    return pd.DataFrame(data)

def train_and_evaluate():
    df = generate_synthetic_ev_portfolio(5000)
    
    # Feature Engineering
    df['oem_tesla'] = (df['oem_partner'] == 'Tesla').astype(int)
    df['oem_rivian'] = (df['oem_partner'] == 'Rivian').astype(int)
    df['oem_jlr'] = (df['oem_partner'] == 'Jaguar Land Rover').astype(int)
    df['battery_energy_density_proxy'] = df['battery_kwh'] / (df['msrp'] / 1000)
    df['underwater_rv_gap'] = df['contracted_rv_dollar'] - (df['msrp'] * 0.50)
    
    features = [
        'msrp', 'battery_kwh', 'lease_term_months', 'annual_mileage',
        'contracted_rv_pct', 'battery_soh', 'ev_market_sentiment_index',
        'oem_subvention_bps', 'dealer_tier', 'lessee_fico',
        'oem_tesla', 'oem_rivian', 'oem_jlr', 'battery_energy_density_proxy'
    ]
    
    # 1. Classification: Return Probability
    X = df[features]
    y_return = df['returned_to_bank']
    X_train, X_test, y_train, y_test = train_test_split(X, y_return, test_size=0.2, random_state=42)
    
    clf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
    clf.fit(X_train, y_train)
    y_pred_proba = clf.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred_proba)
    
    # 2. Regression: Realized Residual Value
    y_rv = df['realized_rv_dollar']
    X_train_rv, X_test_rv, y_train_rv, y_test_rv = train_test_split(X, y_rv, test_size=0.2, random_state=42)
    reg = GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=42)
    reg.fit(X_train_rv, y_train_rv)
    y_pred_rv = reg.predict(X_test_rv)
    mae = mean_absolute_error(y_test_rv, y_pred_rv)
    r2 = r2_score(y_test_rv, y_pred_rv)
    
    # Portfolio Impact Analysis
    total_portfolio_msrp = df['msrp'].sum()
    total_contracted_rv = df['contracted_rv_dollar'].sum()
    total_expected_return_units = df['returned_to_bank'].sum()
    total_rv_variance_on_returns = df[df['returned_to_bank'] == 1]['rv_variance_dollar'].sum()
    
    results = {
        'total_units': len(df),
        'portfolio_msrp_mm': total_portfolio_msrp / 1e6,
        'contracted_rv_mm': total_contracted_rv / 1e6,
        'return_rate_pct': (total_expected_return_units / len(df)) * 100,
        'net_rv_impairment_exposure_mm': total_rv_variance_on_returns / 1e6,
        'classifier_auc': auc,
        'regressor_mae': mae,
        'regressor_r2': r2,
        'top_features_clf': sorted(zip(features, clf.feature_importances_), key=lambda x: x[1], reverse=True)[:5]
    }
    
    return df, results

if __name__ == '__main__':
    df, res = train_and_evaluate()
    print("=== CHASE AUTO EV CAPTIVE PORTFOLIO RESULTS ===")
    print(f"Total Contracts: {res['total_units']:,}")
    print(f"Total Portfolio MSRP: ${res['portfolio_msrp_mm']:.2f}M")
    print(f"Contracted Residual Value: ${res['contracted_rv_mm']:.2f}M")
    print(f"Lease-End Return Rate: {res['return_rate_pct']:.1f}%")
    print(f"Net RV Impairment Exposure on Returns: ${res['net_rv_impairment_exposure_mm']:.2f}M")
    print(f"Return Classifier AUC-ROC: {res['classifier_auc']:.4f}")
    print(f"Residual Value Regressor R2: {res['regressor_r2']:.4f} (MAE: ${res['regressor_mae']:,.2f})")
    print("\nTop Predictors of Lease Return:")
    for feat, imp in res['top_features_clf']:
        print(f" - {feat}: {imp*100:.2f}%")
