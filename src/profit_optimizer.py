"""
Profit Optimization Module
Finds optimal threshold based on business metrics
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from src.config import CLV, INTERVENTION_COST, FALSE_POSITIVE_COST

class ProfitOptimizer:
    """
    Optimize threshold for maximum profit
    """
    
    def __init__(self, clv: float = CLV, 
                 intervention_cost: float = INTERVENTION_COST,
                 fp_cost: float = FALSE_POSITIVE_COST):
        """
        Args:
            clv: Customer Lifetime Value
            intervention_cost: Cost to retain a customer
            fp_cost: Cost of unnecessary intervention
        """
        self.clv = clv
        self.intervention_cost = intervention_cost
        self.fp_cost = fp_cost
    
    def calculate_profit(self, y_true: np.ndarray, 
                        y_pred: np.ndarray) -> float:
        """
        Calculate total profit based on predictions
        
        Profit Logic:
        - True Positive (TP): Save churner = CLV - intervention_cost
        - False Positive (FP): Wrong intervention = -fp_cost
        - True Negative (TN): No action needed = 0
        - False Negative (FN): Lost customer = -CLV
        """
        TP = np.sum((y_true == 1) & (y_pred == 1))
        FP = np.sum((y_true == 0) & (y_pred == 1))
        TN = np.sum((y_true == 0) & (y_pred == 0))
        FN = np.sum((y_true == 1) & (y_pred == 0))
        
        profit = (
            TP * (self.clv - self.intervention_cost) +  # Saved customers
            FP * (-self.fp_cost) +                       # Wasted interventions
            TN * 0 +                                      # Correct non-interventions
            FN * (-self.clv)                              # Lost customers
        )
        
        return profit
    
    def find_optimal_threshold(self, y_true: np.ndarray,
                               y_pred_proba: np.ndarray,
                               thresholds: np.ndarray = None) -> Tuple[float, float]:
        """
        Find threshold that maximizes profit
        """
        if thresholds is None:
            thresholds = np.arange(0.1, 0.9, 0.01)
        
        profits = []
        
        for threshold in thresholds:
            y_pred = (y_pred_proba >= threshold).astype(int)
            profit = self.calculate_profit(y_true, y_pred)
            profits.append(profit)
        
        optimal_idx = np.argmax(profits)
        optimal_threshold = thresholds[optimal_idx]
        max_profit = profits[optimal_idx]
        
        return optimal_threshold, max_profit
    
    def plot_profit_curve(self, y_true: np.ndarray,
                         y_pred_proba: np.ndarray,
                         save_path: str = None):
        """
        Plot profit vs threshold
        """
        thresholds = np.arange(0.1, 0.9, 0.01)
        profits = []
        
        for threshold in thresholds:
            y_pred = (y_pred_proba >= threshold).astype(int)
            profit = self.calculate_profit(y_true, y_pred)
            profits.append(profit)
        
        optimal_threshold, max_profit = self.find_optimal_threshold(y_true, y_pred_proba)
        
        plt.figure(figsize=(12, 6))
        plt.plot(thresholds, profits, linewidth=2, color='darkblue')
        plt.axvline(optimal_threshold, color='red', linestyle='--', 
                   label=f'Optimal Threshold = {optimal_threshold:.3f}')
        plt.axhline(max_profit, color='green', linestyle='--', alpha=0.5,
                   label=f'Max Profit = ${max_profit:,.0f}')
        
        plt.xlabel('Classification Threshold', fontsize=12)
        plt.ylabel('Total Profit ($)', fontsize=12)
        plt.title('Profit Optimization Curve', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Profit curve saved to {save_path}")
        
        plt.tight_layout()
        plt.show()
        
        return optimal_threshold, max_profit
    
    def print_profit_analysis(self, y_true: np.ndarray,
                             y_pred_proba: np.ndarray,
                             threshold: float):
        """
        Print detailed profit analysis
        """
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        TP = np.sum((y_true == 1) & (y_pred == 1))
        FP = np.sum((y_true == 0) & (y_pred == 1))
        TN = np.sum((y_true == 0) & (y_pred == 0))
        FN = np.sum((y_true == 1) & (y_pred == 0))
        
        total_profit = self.calculate_profit(y_true, y_pred)
        
        print("\n" + "="*60)
        print("💰 PROFIT ANALYSIS")
        print("="*60)
        print(f"Threshold: {threshold:.3f}")
        print(f"\nConfusion Matrix:")
        print(f"  True Positives (Saved):        {TP:>6} → ${TP * (self.clv - self.intervention_cost):>12,.0f}")
        print(f"  False Positives (Wasted):      {FP:>6} → ${FP * (-self.fp_cost):>12,.0f}")
        print(f"  True Negatives (Correct):      {TN:>6} → ${0:>12,.0f}")
        print(f"  False Negatives (Lost):        {FN:>6} → ${FN * (-self.clv):>12,.0f}")
        print(f"\nTotal Profit: ${total_profit:,.0f}")
        print(f"Intervention Rate: {(TP + FP) / len(y_true):.1%}")
        print(f"Customers Saved: {TP} / {np.sum(y_true == 1)} ({TP / np.sum(y_true == 1):.1%})")
        print("="*60 + "\n")

# Main execution
if __name__ == "__main__":
    from src.config import PROCESSED_DATA_DIR
    
    print("🚀 Starting profit optimization...")
    
    # Load OOF predictions
    oof_df = pd.read_csv(PROCESSED_DATA_DIR / "oof_predictions.csv")
    
    # Create ensemble
    ensemble_pred = (oof_df['lgbm_oof'] * 0.4 + 
                    oof_df['catboost_oof'] * 0.35 + 
                    oof_df['xgb_oof'] * 0.25)
    
    y_true = oof_df['target'].values
    
    # Initialize optimizer
    optimizer = ProfitOptimizer()
    
    # Find optimal threshold
    optimal_threshold, max_profit = optimizer.plot_profit_curve(
        y_true, ensemble_pred,
        save_path='experiments/profit_curve.png'
    )
    
    print(f"\n✅ Optimal Threshold: {optimal_threshold:.3f}")
    print(f"💰 Maximum Profit: ${max_profit:,.0f}")
    
    # Print detailed analysis
    optimizer.print_profit_analysis(y_true, ensemble_pred, optimal_threshold)
    
    # Compare with default 0.5 threshold
    print("\n📊 Comparison with Default Threshold (0.5):")
    optimizer.print_profit_analysis(y_true, ensemble_pred, 0.5) 