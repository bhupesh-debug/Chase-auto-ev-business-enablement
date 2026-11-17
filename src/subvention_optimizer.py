"""
Subvention Optimizer Module
Calculates optimal rate buy-down allocations across dealer tiers to maximize Chase Auto originations.
"""

import numpy as np
import pandas as pd

def optimize_subvention_allocation(total_budget_usd=10000000, oem='Tesla'):
    """
    Optimizes OEM subvention dollar allocation across Tier 1-4 dealers.
    """
    dealer_tiers = [1, 2, 3, 4]
    base_volumes = [1200, 2500, 1500, 800] # annual unit volume baseline
    elasticity = [1.8, 1.4, 1.1, 0.8]      # volume elasticity to rate buy-downs
    
    weights = np.array(base_volumes) * np.array(elasticity)
    normalized_weights = weights / weights.sum()
    
    tier_allocation = total_budget_usd * normalized_weights
    subvention_per_unit = tier_allocation / np.array(base_volumes)
    
    results = []
    for tier, vol, alloc, per_unit in zip(dealer_tiers, base_volumes, tier_allocation, subvention_per_unit):
        results.append({
            'dealer_tier': tier,
            'baseline_volume': vol,
            'allocated_subvention_usd': round(alloc, 2),
            'subvention_per_unit_usd': round(per_unit, 2),
            'projected_volume_lift_pct': round((per_unit / 1000) * 8.5, 2)
        })
    return pd.DataFrame(results)

if __name__ == '__main__':
    df = optimize_subvention_allocation()
    print("OEM Subvention Allocation Optimization:")
    print(df)
