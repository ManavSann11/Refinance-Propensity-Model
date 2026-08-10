"""
Refinance Propensity Model - Day 1: Portfolio Generation
Generates a synthetic mortgage portfolio and computes initial refinance economics.
"""

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from datetime import datetime

# Configuration parameters
N_LOANS = 8000                          # Number of loans in synthetic portfolio
SEED = 42                               # Random seed for reproducibility
CURRENT_MARKET_RATE = 0.065             # Current market rate (6.5%) for refinance comparison
CLOSING_COST_RATE = 0.02                # Closing costs as percentage of loan balance
MIN_FICO = 620                          # Minimum FICO score for a "rational" refinance
MAX_LTV = 0.95                          # Maximum LTV for a "rational" refinance
OUT_DATA = "data"                       # Output directory for CSV files
OUT_REPORTS = "reports"                 # Output directory for text/plot reports

def monthly_payment(P: float, annual_rate: float, n_months: int) -> float:
    """
    Calculate the fixed monthly payment for a loan.
    
    Args:
        P: Principal balance
        annual_rate: Annual interest rate (e.g., 0.065 for 6.5%)
        n_months: Loan term in months
    
    Returns:
        Monthly payment amount
    """
    if P <= 0:
        return 0.0
    if annual_rate <= 0:
        return P / n_months
    r = annual_rate / 12.0
    # Standard mortgage payment formula: P * r * (1+r)^n / ((1+r)^n - 1)
    return (r * P) / (1 - (1 + r) ** (-n_months))

def remaining_balance(P: float, annual_rate: float, n_months: int, k_months: int) -> float:
    """
    Calculate remaining balance after k months of payments.
    
    Args:
        P: Original principal
        annual_rate: Annual interest rate
        n_months: Original loan term in months
        k_months: Months elapsed since origination
    
    Returns:
        Remaining balance (clipped to >= 0)
    """
    k = int(np.clip(k_months, 0, n_months))
    if P <= 0:
        return 0.0
    if annual_rate <= 0:
        return max(0.0, P * (1 - k / n_months))
    r = annual_rate / 12.0
    M = monthly_payment(P, annual_rate, n_months)
    pow_ = (1 + r) ** k
    bal = P * pow_ - M * (pow_ - 1) / r
    return float(max(0.0, bal))

def months_between(d0: pd.Timestamp, d1: pd.Timestamp) -> int:
    """
    Calculate the number of months between two dates (rounded down).
    """
    if d1 <= d0:
        return 0
    return int((d1 - d0).days // 30)

def generate_portfolio(n_loans=N_LOANS, seed=SEED) -> pd.DataFrame:
    """
    Generate a synthetic mortgage portfolio with realistic borrower characteristics.
    
    Key assumptions:
    - Region distribution: Northeast (20%), Midwest (22%), South (35%), West (23%)
    - Origination dates: 2016-2024
    - Loan terms: 15 years (25%) or 30 years (75%)
    - Original balances: lognormal distribution with mean ~$300k
    - Interest rates: vary by origination year (lower in 2020-2021)
    - FICO scores: normal distribution centered at 720
    - LTV: normal distribution centered at 80%
    - Delinquency: 8% of loans have some delinquency history
    """
    rng = np.random.default_rng(seed)
    
    # Assign regions
    regions = np.array(["Northeast", "Midwest", "South", "West"])
    region = rng.choice(regions, size=n_loans, p=[0.20, 0.22, 0.35, 0.23])
    
    # Origination dates spread across 2016-2024
    start = np.datetime64("2016-01-01")
    end = np.datetime64("2024-12-31")
    origination_date = start + (end - start) * rng.random(n_loans)
    origination_date = pd.to_datetime(origination_date)
    
    # Loan terms: 15-year (25%) or 30-year (75%)
    term_years = rng.choice([15, 30], size=n_loans, p=[0.25, 0.75])
    term_months = term_years * 12
    
    # Original balances: lognormal to avoid negative values, clip to realistic range
    original_balance = rng.lognormal(mean=np.log(300_000), sigma=0.55, size=n_loans)
    original_balance = np.clip(original_balance, 80_000, 1_500_000)
    
    # Interest rates vary by origination year (lower rates in 2020-2021 COVID period)
    year = origination_date.year.to_numpy()
    base_rate = np.where(
        year <= 2019, rng.normal(0.045, 0.008, size=n_loans),
        np.where(year <= 2021, rng.normal(0.030, 0.006, size=n_loans),
                 rng.normal(0.055, 0.010, size=n_loans))
    )
    original_rate = np.clip(base_rate, 0.020, 0.090)
    
    # FICO scores: normal distribution, clipped to realistic range
    fico = rng.normal(720, 55, size=n_loans)
    fico = np.clip(fico, 580, 850).round().astype(int)
    
    # Original LTV: normal distribution, clipped to avoid extremes
    orig_ltv = np.clip(rng.normal(0.80, 0.10, size=n_loans), 0.50, 0.98)
    property_value_orig = original_balance / orig_ltv
    
    # Delinquency status: 8% have some past delinquency (30+ days late)
    days_delinquent = rng.choice([0, 30, 60, 90, 120], size=n_loans,
                                 p=[0.92, 0.04, 0.02, 0.015, 0.005])
    
    return pd.DataFrame({
        "loan_id": np.arange(1, n_loans + 1),
        "region": region,
        "origination_date": origination_date,
        "term_years": term_years,
        "term_months": term_months,
        "original_balance": original_balance,
        "original_rate": original_rate,
        "fico": fico,
        "days_delinquent": days_delinquent,
        "property_value_orig": property_value_orig,
    })

def add_current_snapshot(df: pd.DataFrame, seed=SEED+1) -> pd.DataFrame:
    """
    Add current loan status: remaining balance, months elapsed, home value, LTV.
    
    Simulates home value appreciation using a geometric Brownian motion model
    with region-specific drift rates.
    """
    rng = np.random.default_rng(seed)
    today = pd.Timestamp(datetime.now())
    
    # Calculate months since origination
    months_elapsed = df["origination_date"].apply(lambda d: months_between(d, today)).astype(int)
    remaining_term_months = np.maximum(0, df["term_months"] - months_elapsed)
    
    # Current balances using amortization formula
    balances = np.array([
        remaining_balance(P, r, int(n), int(k))
        for P, r, n, k in zip(df["original_balance"], df["original_rate"], df["term_months"], months_elapsed)
    ])
    
    # Home value simulation using geometric Brownian motion
    # Drift varies by region (West has highest appreciation)
    drift = df["region"].map({
        "Northeast": 0.045,
        "Midwest": 0.035,
        "South": 0.040,
        "West": 0.055
    }).astype(float).to_numpy()
    
    years_age = (months_elapsed / 12.0).to_numpy()
    sigma = 0.10  # volatility of home prices
    Z = rng.normal(0, 1, size=len(df))
    t = np.clip(years_age, 0.05, 20.0)
    log_mult = (drift - 0.5 * sigma**2) * t + sigma * np.sqrt(t) * Z
    mult = np.exp(log_mult)
    
    current_home_value = df["property_value_orig"].to_numpy() * mult
    current_home_value = np.clip(current_home_value, 50_000, 5_000_000)
    
    # Add new columns to DataFrame
    out = df.copy()
    out["months_elapsed"] = months_elapsed
    out["remaining_term_months"] = remaining_term_months
    out["current_balance"] = balances
    out["current_home_value"] = current_home_value
    out["ltv_current"] = out["current_balance"] / out["current_home_value"]
    
    return out

def add_refi_economics(df: pd.DataFrame, current_market_rate=CURRENT_MARKET_RATE) -> pd.DataFrame:
    """
    Compute refinance economics for each loan:
    - Original payment vs. refinanced payment
    - Monthly savings
    - Break-even period
    - Economic rationality flags
    """
    out = df.copy()
    
    # Current monthly payments
    out["pmt_original"] = [
        monthly_payment(P, r, int(n))
        for P, r, n in zip(out["original_balance"], out["original_rate"], out["term_months"])
    ]
    
    # Potential new payment at current market rate
    out["pmt_refi"] = [
        monthly_payment(bal, current_market_rate, int(rt if rt > 0 else 1))
        for bal, rt in zip(out["current_balance"], out["remaining_term_months"])
    ]
    
    out["rate_diff"] = out["original_rate"] - current_market_rate
    out["monthly_savings"] = out["pmt_original"] - out["pmt_refi"]
    out["closing_cost"] = CLOSING_COST_RATE * out["current_balance"]
    
    # Break-even: months to recoup closing costs through monthly savings
    out["break_even_months"] = np.where(
        out["monthly_savings"] > 0,
        out["closing_cost"] / out["monthly_savings"],
        np.inf  # Never breaks even if no savings
    )
    
    # Flags for analysis
    out["underwater"] = (out["ltv_current"] > 1.0).astype(int)
    out["in_the_money"] = ((out["rate_diff"] > 0) & (out["monthly_savings"] > 0)).astype(int)
    
    # Economic rationality: a loan is "rational" to refinance if:
    # 1. Monthly savings > 0
    # 2. Break-even period < remaining loan term
    # 3. LTV <= 95%
    # 4. FICO >= 620
    # 5. No current delinquency
    out["econ_rational"] = (
            (out["monthly_savings"] > 0) &
            (out["break_even_months"] < out["remaining_term_months"]) &
            (out["ltv_current"] <= MAX_LTV) &
            (out["fico"] >= MIN_FICO) &
            (out["days_delinquent"] == 0)
    ).astype(int)
    
    return out

def write_summary(df: pd.DataFrame, current_market_rate=CURRENT_MARKET_RATE) -> str:
    """Generate a text summary of the Day 1 portfolio."""
    econ = df[df["econ_rational"] == 1]
    
    lines = []
    lines.append("=== DAY 1 SUMMARY ===")
    lines.append(f"Loans: {len(df):,}")
    lines.append(f"Market rate scenario: {current_market_rate*100:.2f}%")
    lines.append("")
    lines.append("Portfolio:")
    lines.append(f"  Total current balance: ${df['current_balance'].sum():,.0f}")
    lines.append(f"  Median LTV: {df['ltv_current'].median():.3f}")
    lines.append(f"  % Underwater (LTV>1): {df['underwater'].mean()*100:.1f}%")
    lines.append("")
    lines.append("Refi economics:")
    lines.append(f"  % In-the-money: {df['in_the_money'].mean()*100:.1f}%")
    lines.append(f"  % Economically rational (incl FICO/LTV/current): {df['econ_rational'].mean()*100:.1f}%")
    
    if len(econ) > 0:
        lines.append(f"  Median monthly savings (econ): ${econ['monthly_savings'].median():.0f}")
        lines.append(f"  Median break-even months (econ): {econ['break_even_months'].median():.1f}")
    else:
        lines.append("  Econ-rational set is empty under this rate scenario; try lowering CURRENT_MARKET_RATE.")
    
    return "\n".join(lines)

def main():
    """Run the Day 1 portfolio generation pipeline."""
    os.makedirs(OUT_DATA, exist_ok=True)
    os.makedirs(OUT_REPORTS, exist_ok=True)
    
    # Generate portfolio and add economics
    df = generate_portfolio()
    df = add_current_snapshot(df)
    df = add_refi_economics(df)
    
    # Save portfolio
    csv_path = os.path.join(OUT_DATA, "day1_portfolio.csv")
    df.to_csv(csv_path, index=False)
    
    # Generate and save summary
    summary = write_summary(df)
    summary_path = os.path.join(OUT_REPORTS, "day1_summary.txt")
    with open(summary_path, "w") as f:
        f.write(summary)
    print(summary)
    
    # Create LTV distribution plot
    fig_path = os.path.join(OUT_REPORTS, "day1_distributions.png")
    plt.figure(figsize=(10,6))
    plt.hist(df["ltv_current"].replace([np.inf, -np.inf], np.nan).dropna(), bins=40)
    plt.title("Current LTV Distribution")
    plt.xlabel("LTV")
    plt.ylabel("Count")
    plt.savefig(fig_path, dpi=200)
    plt.close()
    
    print(f"\nSaved: {csv_path}")
    print(f"Saved: {summary_path}")
    print(f"Saved: {fig_path}")

if __name__ == "__main__":
    main()
