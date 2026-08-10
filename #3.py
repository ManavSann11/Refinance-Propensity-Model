"""
Refinance Propensity Model - Day 3: Strategy Comparison
Compares three outreach strategies to identify the most profitable approach.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# File paths for input and output data
INPUT_CSV = "data/day2_scored_portfolio.csv"  # Input from Day 2
OUTPUT_RESULTS_CSV = "data/day3_strategy_results.csv"
OUTPUT_TOP_TARGETS_CSV = "data/day3_top_expected_profit_targets.csv"
REPORT_PATH = "reports/day3_summary.txt"
PLOT_PATH = "reports/day3_strategy_comparison.png"

# Configuration
SEED = 42
TOP_N_VALUES = [25, 50, 100, 200, 500, 1000]  # Outreach sizes to test

def evaluate_strategy(sorted_df: pd.DataFrame, top_n: int) -> dict:
    """
    Evaluate the business value of contacting the top_n borrowers in a ranked list.
    
    Args:
        sorted_df: DataFrame sorted by a strategy's ranking (e.g., by expected profit)
        top_n: Number of top-ranked borrowers to evaluate
    
    Returns:
        Dictionary containing profit metrics and borrower characteristics
    """
    subset = sorted_df.head(top_n).copy()
    
    total_expected_profit = subset["expected_profit"].sum()
    avg_expected_profit = subset["expected_profit"].mean()
    median_expected_profit = subset["expected_profit"].median()
    positive_profit_pct = (subset["expected_profit"] > 0).mean()
    avg_probability = subset["predicted_refi_probability"].mean()
    avg_monthly_savings = subset["monthly_savings"].mean()
    avg_balance = subset["current_balance"].mean()
    
    return {
        "top_n": top_n,
        "total_expected_profit": total_expected_profit,
        "avg_expected_profit": avg_expected_profit,
        "median_expected_profit": median_expected_profit,
        "positive_profit_pct": positive_profit_pct,
        "avg_predicted_refi_probability": avg_probability,
        "avg_monthly_savings": avg_monthly_savings,
        "avg_current_balance": avg_balance,
    }

def run_strategy_comparison(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compare three outreach strategies:
    1. Random — baseline benchmark
    2. Monthly savings heuristic — simple business rule
    3. Expected profit ranking — model-driven approach
    
    Args:
        df: Scored portfolio from Day 2
    
    Returns:
        DataFrame with strategy results for each outreach size
    """
    rng = np.random.default_rng(SEED)
    
    # Create rankings for each strategy
    random_df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
    savings_df = df.sort_values("monthly_savings", ascending=False).reset_index(drop=True)
    expected_profit_df = df.sort_values("expected_profit", ascending=False).reset_index(drop=True)
    
    results = []
    for n in TOP_N_VALUES:
        # Evaluate each strategy at this outreach size
        random_result = evaluate_strategy(random_df, n)
        random_result["strategy"] = "random"
        results.append(random_result)
        
        savings_result = evaluate_strategy(savings_df, n)
        savings_result["strategy"] = "monthly_savings"
        results.append(savings_result)
        
        expected_profit_result = evaluate_strategy(expected_profit_df, n)
        expected_profit_result["strategy"] = "expected_profit"
        results.append(expected_profit_result)
    
    results_df = pd.DataFrame(results)
    
    # Ensure strategies are ordered consistently for plotting
    strategy_order = ["random", "monthly_savings", "expected_profit"]
    results_df["strategy"] = pd.Categorical(results_df["strategy"], categories=strategy_order, ordered=True)
    results_df = results_df.sort_values(["top_n", "strategy"]).reset_index(drop=True)
    
    return results_df

def build_summary(df: pd.DataFrame, results_df: pd.DataFrame) -> str:
    """
    Build a readable text summary for Day 3.
    """
    # Find best strategy for each outreach size
    best_by_n = (
        results_df.sort_values(["top_n", "total_expected_profit"], ascending=[True, False])
        .groupby("top_n")
        .head(1)
        .reset_index(drop=True)
    )
    
    # Focus on largest outreach size for detailed comparison
    largest_n = max(TOP_N_VALUES)
    largest_subset = results_df[results_df["top_n"] == largest_n].sort_values("total_expected_profit", ascending=False)
    
    # Top 10 targets overall
    top_targets = df.sort_values("expected_profit", ascending=False).head(10)
    
    lines = []
    lines.append("=== DAY 3 SUMMARY ===")
    lines.append("Strategy comparison for refinance outreach targeting")
    lines.append("")
    lines.append("OVERVIEW")
    lines.append(f"  Loans evaluated: {len(df):,}")
    lines.append(f"  Outreach sizes tested: {TOP_N_VALUES}")
    lines.append("")
    lines.append("BEST STRATEGY BY OUTREACH SIZE")
    for _, row in best_by_n.iterrows():
        lines.append(
            f"  Top {int(row['top_n']):>4}: {row['strategy']:<16} | "
            f"Total Expected Profit = ${row['total_expected_profit']:,.2f} | "
            f"Avg Profit = ${row['avg_expected_profit']:,.2f}"
        )
    lines.append("")
    lines.append(f"DETAIL AT TOP {largest_n}")
    for _, row in largest_subset.iterrows():
        lines.append(
            f"  {row['strategy']:<16} | "
            f"Total Expected Profit = ${row['total_expected_profit']:,.2f} | "
            f"Avg Profit = ${row['avg_expected_profit']:,.2f} | "
            f"% Positive = {row['positive_profit_pct'] * 100:,.1f}%"
        )
    lines.append("")
    lines.append("INTERPRETATION")
    lines.append(
        "  This analysis compares naive outreach strategies against a model-driven expected value strategy. "
        "If expected_profit dominates, it shows that combining refinance propensity with loan economics leads "
        "to better targeting decisions than using simple heuristics like savings alone."
    )
    lines.append("")
    lines.append("TOP 10 EXPECTED-PROFIT TARGETS")
    lines.append(
        top_targets[
            [
                "loan_id",
                "predicted_refi_probability",
                "current_balance",
                "monthly_savings",
                "expected_profit",
                "fico",
                "ltv_current",
                "days_delinquent",
            ]
        ].to_string(index=False)
    )
    return "\n".join(lines)

def save_strategy_plot(results_df: pd.DataFrame, output_path: str) -> None:
    """
    Plot total expected profit vs outreach size for each strategy.
    """
    plt.figure(figsize=(9, 6))
    for strategy in results_df["strategy"].cat.categories:
        subset = results_df[results_df["strategy"] == strategy]
        plt.plot(
            subset["top_n"],
            subset["total_expected_profit"],
            marker="o",
            label=strategy
        )
    plt.xlabel("Number of Borrowers Contacted")
    plt.ylabel("Total Expected Profit")
    plt.title("Day 3 Strategy Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

def main():
    """Run the Day 3 strategy comparison pipeline."""
    os.makedirs("data", exist_ok=True)
    os.makedirs("reports", exist_ok=True)
    
    # Load scored portfolio from Day 2
    df = pd.read_csv(INPUT_CSV)
    
    # Validate required columns exist
    required_cols = [
        "loan_id",
        "monthly_savings",
        "predicted_refi_probability",
        "expected_profit",
        "current_balance",
        "fico",
        "ltv_current",
        "days_delinquent",
    ]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in {INPUT_CSV}: {missing_cols}")
    
    # Run comparison
    results_df = run_strategy_comparison(df)
    
    # Save outputs
    results_df.to_csv(OUTPUT_RESULTS_CSV, index=False)
    top_expected_profit_targets = df.sort_values("expected_profit", ascending=False).head(100)
    top_expected_profit_targets.to_csv(OUTPUT_TOP_TARGETS_CSV, index=False)
    save_strategy_plot(results_df, PLOT_PATH)
    summary = build_summary(df, results_df)
    
    with open(REPORT_PATH, "w") as f:
        f.write(summary)
    
    # Print results to console
    print(summary)
    print("")
    print("FULL STRATEGY RESULTS TABLE")
    print(results_df.to_string(index=False))
    print("")
    print(f"Saved: {OUTPUT_RESULTS_CSV}")
    print(f"Saved: {OUTPUT_TOP_TARGETS_CSV}")
    print(f"Saved: {REPORT_PATH}")
    print(f"Saved: {PLOT_PATH}")

if __name__ == "__main__":
    main()
