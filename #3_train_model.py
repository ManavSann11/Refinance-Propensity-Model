import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
INPUT_CSV = "data/day1_portfolio.csv"
OUTPUT_CSV = "data/day2_scored_portfolio.csv"
TOP_TARGETS_CSV = "data/top_refi_targets.csv"
REPORT_PATH = "reports/day2_summary.txt"
ROC_PLOT_PATH = "reports/day2_roc_curve.png"
SEED = 42
OUTREACH_COST = 150.0
REVENUE_RATE = 0.01   # lender earns ~1% of current balance if refi closes
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def create_true_refi_probability(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a hidden behavioral probability for whether a borrower actually refinances.
    This is synthetic ground truth used to train the Day 2 model.

    The logic is intentionally plausible:
    - higher monthly savings -> more likely to refi
    - shorter break-even -> more likely
    - lower LTV -> more likely
    - higher FICO -> more likely
    - delinquency -> much less likely
    - older loans / positive rate incentive -> somewhat more likely
    """
    out = df.copy()
    savings = out["monthly_savings"].clip(lower=-500, upper=1500)
    break_even = out["break_even_months"].replace(np.inf, 999).clip(0, 999)
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
    )
    score += 1.25 * out["econ_rational"]
    true_prob = sigmoid(score).clip(0.001, 0.95)
    out["true_refi_probability"] = true_prob
    return out
def sample_refi_outcomes(df: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    """
    Sample actual refinance outcomes from the synthetic true probability.
    """
    rng = np.random.default_rng(seed)
    out = df.copy()
    out["did_refi"] = rng.binomial(1, out["true_refi_probability"])
    return out
def train_refi_model(df: pd.DataFrame):
    """
    Train logistic regression on simulated refinance outcomes.
    """
    feature_cols = [
        "monthly_savings",
        "ltv_current",
        "fico",
        "days_delinquent",
        "months_elapsed",
        "current_balance",
    ]
    model_df = df.copy()
    model_df["break_even_months"] = model_df["break_even_months"].replace(np.inf, 999)
    X = model_df[feature_cols]
    y = model_df["did_refi"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.25,
        random_state=SEED,
        stratify=y
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    model = LogisticRegression(
        random_state=SEED,
        max_iter=2000,
    )
    model.fit(X_train_scaled, y_train)
    y_proba_test = model.predict_proba(X_test_scaled)[:, 1]
    y_pred_test = (y_proba_test >= 0.05).astype(int)
    auc = roc_auc_score(y_test, y_proba_test)
    cm = confusion_matrix(y_test, y_pred_test)
    return {
        "model": model,
        "scaler": scaler,
        "feature_cols": feature_cols,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "y_proba_test": y_proba_test,
        "y_pred_test": y_pred_test,
        "auc": auc,
        "confusion_matrix": cm,
    }
def score_portfolio(df: pd.DataFrame, model_info: dict) -> pd.DataFrame:
    """
    Predict refinance propensity for all loans and compute expected lender profit.
    """
    out = df.copy()
    X_all = out[model_info["feature_cols"]].copy()
    X_all_scaled = model_info["scaler"].transform(X_all)
    out["predicted_refi_probability"] = model_info["model"].predict_proba(X_all_scaled)[:, 1]
    out["expected_revenue_if_refi"] = REVENUE_RATE * out["current_balance"]
    out["expected_profit"] = (
            out["predicted_refi_probability"] * out["expected_revenue_if_refi"]
            - OUTREACH_COST
    )
    out.loc[out["monthly_savings"] <= 0, "expected_profit"] = -OUTREACH_COST
    return out
def make_coefficient_table(model_info: dict) -> pd.DataFrame:
    """
    Return a simple coefficient table for interpretation.
    Note: because features are standardized, coefficients are comparable in magnitude.
    """
    return pd.DataFrame({
        "feature": model_info["feature_cols"],
        "coefficient": model_info["model"].coef_[0]
    }).sort_values("coefficient", ascending=False)
def save_roc_plot(model_info: dict, path: str):
    fpr, tpr, _ = roc_curve(model_info["y_test"], model_info["y_proba_test"])
    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, label=f"AUC = {model_info['auc']:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Day 2 ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
def build_summary(scored_df: pd.DataFrame, model_info: dict, coef_df: pd.DataFrame) -> str:
    top_10 = scored_df.sort_values("expected_profit", ascending=False).head(10)
    positive_profit = (scored_df["expected_profit"] > 0).mean()
    lines = []
    lines.append("=== DAY 2 SUMMARY ===")
    lines.append(f"Loans scored: {len(scored_df):,}")
    lines.append("")
    lines.append("MODEL PERFORMANCE")
    lines.append(f"  Test AUC: {model_info['auc']:.3f}")
    lines.append(f"  Confusion Matrix:")
    lines.append(f"    {model_info['confusion_matrix']}")
    lines.append("")
    lines.append("PORTFOLIO PROPENSITY")
    lines.append(f"  Mean predicted refinance probability: {scored_df['predicted_refi_probability'].mean():.4f}")
    lines.append(f"  Actual simulated refinance rate: {scored_df['did_refi'].mean():.4f}")
    lines.append("")
    lines.append("ECONOMICS")
    lines.append(f"  Mean expected profit per loan: ${scored_df['expected_profit'].mean():.2f}")
    lines.append(f"  Median expected profit per loan: ${scored_df['expected_profit'].median():.2f}")
    lines.append(f"  % of loans with positive expected profit: {positive_profit * 100:.1f}%")
    lines.append("")
    lines.append("TOP POSITIVE COEFFICIENTS")
    for _, row in coef_df.head(5).iterrows():
        lines.append(f"  {row['feature']}: {row['coefficient']:.4f}")
    lines.append("")
    lines.append("MOST NEGATIVE COEFFICIENTS")
    for _, row in coef_df.tail(5).iterrows():
        lines.append(f"  {row['feature']}: {row['coefficient']:.4f}")
    lines.append("")
    lines.append("TOP 10 EXPECTED-PROFIT TARGETS")
    lines.append(
        top_10[
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
def main():
    os.makedirs("data", exist_ok=True)
    os.makedirs("reports", exist_ok=True)
    df = pd.read_csv(INPUT_CSV)
    df = create_true_refi_probability(df)
    df = sample_refi_outcomes(df)
    model_info = train_refi_model(df)
    scored_df = score_portfolio(df, model_info)
    coef_df = make_coefficient_table(model_info)
    scored_df.to_csv(OUTPUT_CSV, index=False)
    top_targets = scored_df.sort_values("expected_profit", ascending=False).head(100)
    top_targets.to_csv(TOP_TARGETS_CSV, index=False)
    save_roc_plot(model_info, ROC_PLOT_PATH)
    summary = build_summary(scored_df, model_info, coef_df)
    with open(REPORT_PATH, "w") as f:
        f.write(summary)
    print(summary)
    print("")
    print(f"Saved: {OUTPUT_CSV}")
    print(f"Saved: {TOP_TARGETS_CSV}")
    print(f"Saved: {REPORT_PATH}")
    print(f"Saved: {ROC_PLOT_PATH}")
if __name__ == "__main__":
    main()
