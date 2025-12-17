#!/usr/bin/env python3
"""Generate synthetic credit data for testing and development.

This script creates realistic synthetic credit data that can be used
to test the Fair Credit Score Prediction pipeline without needing
real customer data.

Usage:
    python scripts/generate_sample_data.py
    python scripts/generate_sample_data.py --num-records 5000
    python scripts/generate_sample_data.py --output data/my_data.csv
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def generate_credit_data(
    num_records: int = 1000,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic credit data.

    Args:
        num_records: Number of records to generate.
        seed: Random seed for reproducibility.

    Returns:
        DataFrame with synthetic credit data.
    """
    np.random.seed(seed)

    # Generate base features
    income = np.random.lognormal(mean=10.8, sigma=0.5, size=num_records).astype(int)
    income = np.clip(income, 20000, 500000)  # Realistic income range

    # Debt correlates somewhat with income
    debt_ratio = np.random.beta(2, 5, num_records)  # Skewed toward lower debt
    debt = (income * debt_ratio * 0.8).astype(int)

    # Loan amount correlates with income
    loan_ratio = np.random.uniform(0.1, 0.8, num_records)
    loan_amount = (income * loan_ratio).astype(int)
    loan_amount = np.clip(loan_amount, 5000, 500000)

    # Other features
    loan_term = np.random.choice([12, 24, 36, 48, 60, 72, 84], num_records)
    num_credit_cards = np.random.poisson(3, num_records)
    num_credit_cards = np.clip(num_credit_cards, 0, 15)

    # Credit score - influenced by debt ratio
    base_score = np.random.normal(680, 80, num_records)
    debt_penalty = debt_ratio * 100
    credit_score = (base_score - debt_penalty + np.random.normal(0, 30, num_records)).astype(int)
    credit_score = np.clip(credit_score, 300, 850)

    # Categorical features
    gender = np.random.choice(['Male', 'Female'], num_records, p=[0.52, 0.48])

    education = np.random.choice(
        ['High School', "Bachelor's", "Master's", 'PhD'],
        num_records,
        p=[0.35, 0.40, 0.20, 0.05]
    )

    payment_history = np.random.choice(
        ['Excellent', 'Good', 'Fair', 'Poor'],
        num_records,
        p=[0.25, 0.40, 0.25, 0.10]
    )

    employment_status = np.random.choice(
        ['Employed', 'Self-Employed', 'Unemployed'],
        num_records,
        p=[0.75, 0.18, 0.07]
    )

    residence_type = np.random.choice(
        ['Owned', 'Rented', 'Mortgage'],
        num_records,
        p=[0.30, 0.40, 0.30]
    )

    marital_status = np.random.choice(
        ['Single', 'Married', 'Divorced'],
        num_records,
        p=[0.35, 0.50, 0.15]
    )

    # Generate creditworthiness based on features (simulating real patterns)
    # Higher income, lower debt ratio, better payment history -> more likely creditworthy
    creditworthy_prob = (
        0.3 +  # Base probability
        0.2 * (income > 60000) +
        0.15 * (debt_ratio < 0.3) +
        0.15 * (credit_score > 700) +
        0.1 * (np.isin(payment_history, ['Excellent', 'Good'])) +
        0.1 * (np.isin(employment_status, ['Employed', 'Self-Employed']))
    )
    creditworthy_prob = np.clip(creditworthy_prob, 0.1, 0.95)
    creditworthiness = (np.random.random(num_records) < creditworthy_prob).astype(int)

    # Create DataFrame
    data = pd.DataFrame({
        'Income': income,
        'Debt': debt,
        'Loan_Amount': loan_amount,
        'Loan_Term': loan_term,
        'Num_Credit_Cards': num_credit_cards,
        'Credit_Score': credit_score,
        'Gender': gender,
        'Education': education,
        'Payment_History': payment_history,
        'Employment_Status': employment_status,
        'Residence_Type': residence_type,
        'Marital_Status': marital_status,
        'Creditworthiness': creditworthiness,
    })

    return data


def print_data_summary(data: pd.DataFrame) -> None:
    """Print summary statistics for the generated data."""
    print("\n" + "=" * 60)
    print("GENERATED DATA SUMMARY")
    print("=" * 60)

    print(f"\nTotal records: {len(data)}")
    print(f"Creditworthy: {data['Creditworthiness'].sum()} ({data['Creditworthiness'].mean()*100:.1f}%)")
    print(f"Not Creditworthy: {(1-data['Creditworthiness']).sum()} ({(1-data['Creditworthiness'].mean())*100:.1f}%)")

    print("\nNumeric Features:")
    print(f"  Income: ${data['Income'].mean():,.0f} (avg), ${data['Income'].min():,} - ${data['Income'].max():,}")
    print(f"  Debt: ${data['Debt'].mean():,.0f} (avg)")
    print(f"  Loan Amount: ${data['Loan_Amount'].mean():,.0f} (avg)")
    print(f"  Credit Score: {data['Credit_Score'].mean():.0f} (avg), {data['Credit_Score'].min()} - {data['Credit_Score'].max()}")

    print("\nCategorical Distribution:")
    print(f"  Gender: {dict(data['Gender'].value_counts())}")
    print(f"  Education: {dict(data['Education'].value_counts())}")
    print(f"  Employment: {dict(data['Employment_Status'].value_counts())}")
    print(f"  Residence: {dict(data['Residence_Type'].value_counts())}")

    # Calculate expected CDI distribution
    cdi = (
        (data['Education'] == 'High School').astype(int) +
        (data['Residence_Type'] == 'Rented').astype(int) +
        (data['Marital_Status'] == 'Single').astype(int) +
        (data['Gender'] == 'Female').astype(int)
    )
    disadvantaged = (cdi >= 2).sum()
    print(f"\nExpected CDI Distribution:")
    print(f"  CDI >= 2 (Disadvantaged): {disadvantaged} ({disadvantaged/len(data)*100:.1f}%)")
    print(f"  CDI < 2 (Privileged): {len(data)-disadvantaged} ({(len(data)-disadvantaged)/len(data)*100:.1f}%)")

    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic credit data for testing"
    )
    parser.add_argument(
        "--num-records", "-n",
        type=int,
        default=1000,
        help="Number of records to generate (default: 1000)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="data/sample_credit_data.csv",
        help="Output file path (default: data/sample_credit_data.csv)"
    )
    parser.add_argument(
        "--seed", "-s",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress summary output"
    )

    args = parser.parse_args()

    # Create output directory if needed
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Generate data
    print(f"Generating {args.num_records} synthetic credit records...")
    data = generate_credit_data(num_records=args.num_records, seed=args.seed)

    # Save to CSV
    data.to_csv(output_path, index=False)
    print(f"✓ Saved to {output_path}")

    # Print summary
    if not args.quiet:
        print_data_summary(data)

    print("Done! You can now run:")
    print(f"  python -m src.main run --data-path {output_path}")


if __name__ == "__main__":
    main()
