# Refinance-Propensity-Model
A machine learning model that predicts which mortgage borrowers are most likely to refinance and optimizes outreach targeting to maximize expected profit. 

## Problem Statement
A mortgage lender wants to identify borrowers most likely to refinance and target them for outreach. The challenge is to balance conversion probability against outreach cost while considering borrower economics. 

## Project Progression

## Day 1: Portfolio Generation
The project begins by generating a synthetic portfolio of 8,000 loans with realistic borrower characteristics. The portfolio includes region distribution, origination dates, loan terms, original balances, interest rates, FICO scores, original LTV, and delinquency status. Current snapshots are added, including monthly savings, break-even periods, and economic rationality flags. 

### Day 2: Predictive Modeling
A logistic regression model is trained to predict the probability of refinancing using engineered features such as monthly savings, LTV, FICO score, delinquency status, and loan age. The model is trained on simulated behavioral outcomes and evaluated using AUC and confusion matrix metrics.

### Day 3: Strategy Comparison
Three outreach strategies are compared:
- **Random**: Select borrowers at random
- **Monthly Savings Heuristic**: Select borrowers with highest monthly savings
- **Expected Profit**: Select borrowers with the highest predicted probability x expected revenue - outreach cost

### Day 4: Scenario Analysis
The strategy is stress-tested across different market rate scenarios, from 4.5 percent to 7.0 percent. This analysis shows how the refinance opportunity set expands or contracts as mortgage rates move. 

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
- **Delinzquency status** - risk signal
- **Loan age** - months since origination

## Modeling Approach
A logistic regression model is trained to predict the probability of refinancing using the engineered features. The model is trained on a simulated behavioral outcome that incorporates:
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
├── day1_generate_portfolio.py
├── day2_train_model.py
├── day3_compare_strategies.py
├── day4_scenario_analysis.py
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
