import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
INPUT_CSV = "data/day1_portfolio.csv"
OUTPUT_RESULTS_CSV = "data/day4_scenario_results.csv"
REPORT_PATH = "reports/day4_summary.txt"
PLOT_PATH = "reports/day4_scenario_comparison.png"
OUTREACH_COST = 150.0
REVENUE_RATE = 0.01
TOP_N_VALUES = [25, 50, 100, 200, 500, 1000]
SCENARIOS = [
    {"name": "4.5%", "market_rate": 0.045},
    {"name": "5.5%", "market_rate": 0.055},
    {"name": "6.5%", "market_rate": 0.065},
    {"name": "7.0%", "market_rate": 0.070},
]
def monthly_payment(principal: float, annual_rate: float, term_months: int) -> float:
    if principal <= 0:
        return 0.0
    if annual_rate <= 0:
        return principal / max(term_months, 1)
    r = annual_rate / 12.0
    n = max(int(term_months), 1)
    return (r * principal) / (1 - (1 + r) ** (-n))
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def apply_rate_scenario(df: pd.DataFrame, current_market_rate: float) -> pd.DataFrame:
    """
    Starting from Day 1 portfolio, recompute refinance economics under a new market rate.
    """
    out = df.copy()
    out["scenario_market_rate"] = current_market_rate
    out["rate_diff"] = out["original_rate"] - current_market_rate
    out["pmt_refi_scenario"] = [
        monthly_payment(bal, current_market_rate, int(rt if rt > 0 else 1))
        for bal, rt in zip(out["current_balance"], out["remaining_term_months"])
    ]
    out["monthly_savings_scenario"] = out["pmt_original"] - out["pmt_refi_scenario"]
    out["closing_cost_scenario"] = 0.02 * out["current_balance"]
    out["break_even_months_scenario"] = np.where(
        out["monthly_savings_scenario"] > 0,
        out["closing_cost_scenario"] / out["monthly_savings_scenario"],
        np.inf
    )
    out["in_the_money_scenario"] = (
            (out["rate_diff"] > 0) &
            (out["monthly_savings_scenario"] > 0)
    ).astype(int)
    out["econ_rational_scenario"] = (
            (out["monthly_savings_scenario"] > 0) &
            (out["break_even_months_scenario"] < out["remaining_term_months"]) &
            (out["ltv_current"] <= 0.95) &
            (out["fico"] >= 620) &
            (out["days_delinquent"] == 0)
    ).astype(int)
    return out
def predict_refi_probability(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reuse the Day 2 intuition, but directly define a scenario-based refinance probability.
    This keeps Day 4 focused on how rate changes alter the opportunity set.
    """
    out = df.copy()
    savings = out["monthly_savings_scenario"].clip(lower=-500, upper=2000)
    break_even = out["break_even_months_scenario"].replace(np.inf, 999).clip(0, 999)
    ltv = out["ltv_current"].clip(0, 2)
    fico = out["fico"].clip(300, 850)
    delinquent = (out["days_delinquent"] > 0).astype(int)
    rate_diff = out["rate_diff"].clip(-0.05, 0.05)
    loan_age_years = (out["months_elapsed"] / 12.0).clip(0, 40)
    score = (
            -4.0
            + 0.006 * savings
            - 0.010 * break_even
            - 1.75 * ltv
            + 0.0045 * (fico - 680)
            - 2.5 * delinquent
            + 10.0 * rate_diff
            + 0.04 * loan_age_years
            + 1.25 * out["econ_rational_scenario"]
    )
    out["predicted_refi_probability_scenario"] = sigmoid(score).clip(0.0001, 0.95)
    return out
def score_expected_profit(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["expected_revenue_if_refi_scenario"] = REVENUE_RATE * out["current_balance"]
    out["expected_profit_scenario"] = (
            out["predicted_refi_probability_scenario"] * out["expected_revenue_if_refi_scenario"]
            - OUTREACH_COST
    )
    out.loc[out["monthly_savings_scenario"] <= 0, "expected_profit_scenario"] = -OUTREACH_COST
    return out
def evaluate_top_n(sorted_df: pd.DataFrame, top_n: int) -> dict:
    subset = sorted_df.head(top_n).copy()
    return {
        "top_n": top_n,
        "total_expected_profit": subset["expected_profit_scenario"].sum(),
        "avg_expected_profit": subset["expected_profit_scenario"].mean(),
        "positive_profit_pct": (subset["expected_profit_scenario"] > 0).mean(),
        "avg_probability": subset["predicted_refi_probability_scenario"].mean(),
        "avg_monthly_savings": subset["monthly_savings_scenario"].mean(),
    }
def run_single_scenario(day1_df: pd.DataFrame, scenario_name: str, market_rate: float):
    scenario_df = apply_rate_scenario(day1_df, market_rate)
    scenario_df = predict_refi_probability(scenario_df)
    scenario_df = score_expected_profit(scenario_df)
    ranked_df = scenario_df.sort_values("expected_profit_scenario", ascending=False).reset_index(drop=True)
    scenario_results = []
    for n in TOP_N_VALUES:
        row = evaluate_top_n(ranked_df, n)
        row["scenario"] = scenario_name
        row["market_rate"] = market_rate
        scenario_results.append(row)
    portfolio_summary = {
        "scenario": scenario_name,
        "market_rate": market_rate,
        "loans": len(scenario_df),
        "pct_in_the_money": scenario_df["in_the_money_scenario"].mean(),
        "pct_econ_rational": scenario_df["econ_rational_scenario"].mean(),
        "mean_probability": scenario_df["predicted_refi_probability_scenario"].mean(),
        "mean_expected_profit": scenario_df["expected_profit_scenario"].mean(),
        "median_expected_profit": scenario_df["expected_profit_scenario"].median(),
        "pct_positive_expected_profit": (scenario_df["expected_profit_scenario"] > 0).mean(),
        "best_top_n": None,
        "best_total_expected_profit": None,
    }
    results_df = pd.DataFrame(scenario_results)
    best_row = results_df.sort_values("total_expected_profit", ascending=False).iloc[0]
    portfolio_summary["best_top_n"] = int(best_row["top_n"])
    portfolio_summary["best_total_expected_profit"] = float(best_row["total_expected_profit"])
    return scenario_df, results_df, portfolio_summary
def build_summary(portfolio_summaries_df: pd.DataFrame, topn_results_df: pd.DataFrame) -> str:
    lines = []
    lines.append("=== DAY 4 SUMMARY ===")
    lines.append("Rate scenario analysis for refinance outreach")
    lines.append("")
    lines.append("PORTFOLIO-LEVEL SCENARIO SUMMARY")
    for _, row in portfolio_summaries_df.sort_values("market_rate").iterrows():
        lines.append(
            f"Scenario {row['scenario']} (market rate {row['market_rate']*100:.1f}%): "
            f"In-the-money = {row['pct_in_the_money']*100:.1f}%, "
            f"Econ rational = {row['pct_econ_rational']*100:.1f}%, "
            f"% Positive EV = {row['pct_positive_expected_profit']*100:.1f}%, "
            f"Best outreach size = Top {int(row['best_top_n'])}, "
            f"Best total expected profit = ${row['best_total_expected_profit']:,.2f}"
        )
    lines.append("")
    lines.append("BEST OUTREACH DECISION BY SCENARIO")
    best_rows = (
        topn_results_df.sort_values(["scenario", "total_expected_profit"], ascending=[True, False])
        .groupby("scenario")
        .head(1)
        .reset_index(drop=True)
    )
    for _, row in best_rows.iterrows():
        lines.append(
            f"  {row['scenario']}: Top {int(row['top_n'])} borrowers "
            f"→ Total Expected Profit = ${row['total_expected_profit']:,.2f}, "
            f"Avg Profit = ${row['avg_expected_profit']:,.2f}, "
            f"Avg Monthly Savings = ${row['avg_monthly_savings']:,.2f}"
        )
    lines.append("")
    lines.append("INTERPRETATION")
    lines.append(
        "  This analysis shows how the refinance opportunity set expands or contracts as mortgage rates move. "
        "Lower market rates increase monthly savings, shorten break-even periods, and expand the set of borrowers "
        "who are economically rational to target. Higher market rates compress the opportunity set and make broad "
        "outreach less profitable."
    )
    return "\n".join(lines)
def save_plot(topn_results_df: pd.DataFrame, output_path: str):
    plt.figure(figsize=(10, 6))
    for scenario in sorted(topn_results_df["scenario"].unique()):
        subset = topn_results_df[topn_results_df["scenario"] == scenario].sort_values("top_n")
        plt.plot(
            subset["top_n"],
            subset["total_expected_profit"],
            marker="o",
            label=scenario
        )
    plt.xlabel("Number of Borrowers Contacted")
    plt.ylabel("Total Expected Profit")
    plt.title("Day 4: Profit vs Outreach Size Across Rate Scenarios")
    plt.legend(title="Market Rate")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
def main():
    os.makedirs("data", exist_ok=True)
    os.makedirs("reports", exist_ok=True)
    day1_df = pd.read_csv(INPUT_CSV)
    required_cols = [
        "loan_id",
        "original_rate",
        "current_balance",
        "ltv_current",
        "fico",
        "days_delinquent",
        "months_elapsed",
        "remaining_term_months",
        "pmt_original",
    ]
    missing = [c for c in required_cols if c not in day1_df.columns]
    if missing:
        raise ValueError(f"Missing required columns from Day 1 file: {missing}")
    all_topn_results = []
    all_portfolio_summaries = []
    for scenario in SCENARIOS:
        scenario_df, results_df, portfolio_summary = run_single_scenario(
            day1_df,
            scenario_name=scenario["name"],
            market_rate=scenario["market_rate"]
        )
        all_topn_results.append(results_df)
        all_portfolio_summaries.append(portfolio_summary)
    topn_results_df = pd.concat(all_topn_results, ignore_index=True)
    portfolio_summaries_df = pd.DataFrame(all_portfolio_summaries)
    topn_results_df.to_csv(OUTPUT_RESULTS_CSV, index=False)
    save_plot(topn_results_df, PLOT_PATH)
    summary = build_summary(portfolio_summaries_df, topn_results_df)
    with open(REPORT_PATH, "w") as f:
        f.write(summary)
    print(summary)
    print("")
    print("FULL SCENARIO RESULTS")
    print(topn_results_df.sort_values(["market_rate", "top_n"]).to_string(index=False))
    print("")
    print(f"Saved: {OUTPUT_RESULTS_CSV}")
    print(f"Saved: {REPORT_PATH}")
    print(f"Saved: {PLOT_PATH}")
if __name__ == "__main__":
    main()
