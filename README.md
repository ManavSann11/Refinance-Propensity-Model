# Refinance-Propensity-Model
A machine learning model that predicts which mortgage borrowers are most likely to refinance and optimizes outreach targeting to maximize expected profit. 

## Problem Statement
A mortgage lender wants to identify borrowers who are most likely to refinance and should be targeted for outreach. The challenge is to balance conversion probability against outreach cost while considering borrower economics. 

## Data Generation
This project begins by generating a synthetic portfolio of 8,000 loans with realistic borrower characteristics:
- Region distribution (Northeast, Midwest, South, West)
- Origination dates (2016-2024)
- Loan terms (15 or 30 years)
- Original balances (lognormal distribution)
- Interest rates (varying by year)
- FICO scores (mean 720, std 55)
- Original LTV (mean 0.80, std 0.10)
- Delinquency status

## Feature Engineering
Key features are engineered to capture borrower refinance economics:
- **Monthly savings** - difference between current payment and refinanced payment
- **Break-even period** - months to recover closing costs
- **Current LTV** - loan balance divided by current home value
- **FICO score** - credit quality signal
- **Deliquency status** - risk signal
- **Loan age** - months since origination

## Modeling Approach
A logistic regression model is trained to predict refinance probability using the engineered features. The model is trained on a simulated behavioral outcome that incorporates:
- Monthly savings
- Break-even period
- LTV
- FICO score
- Delinquency status
- Economic rationality

## Strategy Comparison
Three outreach strategies are compared to identify the most profitable approach:

| Strategy | Description |
| :--- | :--- |
| **Random** | Select borrowers at random |
| **Monthly Savings Heuristic** | Select borrowers with highest monthly savings |
| **Expected Profit** | Select borrowers with highest predicted probability × expected revenue − outreach cost |

The expected profit strategy outperformed the others by 40%.

## Tech Stack
- Python
- Pandas, NumPy
- Scikit-learn (Logistic Regression, StandardScaler)
- Matplotlib

## Repository Structure
refinance-propensity-model/
├── README.md
├── refinance_model.py
├── requirements.txt
└── .gitignore

## How to Run
1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run `python refinance_model.py`
4. Outputs: portfolio data, model performance, strategy comparison

## Results
- The expected profit strategy improved total expected profit by 40% compared to the savings heuristic
- The model provides interpretable coefficients and clear targeting recommendations


## Author
Manav Sannappanavar
NYU | Mathematics and Data Science
