"""Generate a synthetic bank churn-style dataset for model training."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


SEED = 42
TRAIN_ROWS = 20_000
TEST_ROWS = 5_000

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "training" / "data"
TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"

TRAIN_COLUMNS = [
    "id",
    "CustomerId",
    "Surname",
    "CreditScore",
    "Geography",
    "Gender",
    "Age",
    "Tenure",
    "Balance",
    "NumOfProducts",
    "HasCrCard",
    "IsActiveMember",
    "EstimatedSalary",
    "Exited",
]
TEST_COLUMNS = [column for column in TRAIN_COLUMNS if column != "Exited"]
SURNAMES = [
    "Smith",
    "Johnson",
    "Williams",
    "Brown",
    "Jones",
    "Miller",
    "Davis",
    "Garcia",
    "Wilson",
    "Taylor",
]


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


def generate_customers(n_rows: int, rng: np.random.Generator, start_id: int) -> pd.DataFrame:
    """Create customers with realistic but intentionally synthetic churn signal."""
    row_ids = np.arange(n_rows)
    customer_ids = np.arange(start_id, start_id + n_rows)
    surnames = rng.choice(SURNAMES, size=n_rows)
    credit_score = np.clip(rng.normal(650, 95, n_rows).round(), 300, 900).astype(int)
    geography = rng.choice(["France", "Germany", "Spain"], size=n_rows, p=[0.52, 0.25, 0.23])
    gender = rng.choice(["Female", "Male"], size=n_rows, p=[0.46, 0.54])
    age = np.clip(rng.normal(41, 11, n_rows).round(), 18, 92).astype(int)
    tenure = rng.integers(0, 11, size=n_rows)
    num_products = rng.choice([1, 2, 3, 4], size=n_rows, p=[0.48, 0.38, 0.11, 0.03])
    has_cr_card = rng.binomial(1, 0.71, n_rows)
    is_active = rng.binomial(1, 0.51, n_rows)
    estimated_salary = np.clip(rng.lognormal(mean=11.25, sigma=0.45, size=n_rows), 10_000, 240_000)

    balance_zero = rng.binomial(1, 0.36, n_rows).astype(bool)
    balance = np.where(
        balance_zero,
        0.0,
        np.clip(rng.normal(118_000, 45_000, n_rows), 5_000, 260_000),
    )

    logit = (
        -2.35
        + 0.045 * (age - 40)
        + 0.80 * (1 - is_active)
        + 0.42 * (geography == "Germany")
        + 0.12 * (geography == "Spain")
        + 0.45 * (gender == "Female")
        + 0.006 * np.maximum(620 - credit_score, 0)
        - 0.45 * (num_products == 2)
        + 0.35 * (num_products == 3)
        + 0.95 * (num_products == 4)
        + 0.0000032 * np.maximum(balance - 120_000, 0)
        + 0.02 * (tenure < 2)
        + rng.normal(0, 0.55, n_rows)
    )
    exited = rng.binomial(1, sigmoid(logit))

    data = pd.DataFrame(
        {
            "id": row_ids,
            "CustomerId": customer_ids,
            "Surname": surnames,
            "CreditScore": credit_score,
            "Geography": geography,
            "Gender": gender,
            "Age": age,
            "Tenure": tenure,
            "Balance": balance.round(2),
            "NumOfProducts": num_products,
            "HasCrCard": has_cr_card,
            "IsActiveMember": is_active,
            "EstimatedSalary": estimated_salary.round(2),
            "Exited": exited,
        }
    )
    return data[TRAIN_COLUMNS]


def report_existing_data() -> bool:
    if not TRAIN_PATH.exists():
        return False

    train = pd.read_csv(TRAIN_PATH)
    print("Existing training data found. Skipping synthetic data generation.")
    print(f"Existing train.csv shape: {train.shape}")
    if "Exited" in train.columns:
        print(f"Existing train.csv churn rate: {train['Exited'].mean():.4f}")
    if TEST_PATH.exists():
        test = pd.read_csv(TEST_PATH)
        print(f"Existing test.csv shape: {test.shape}")
    return True


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if report_existing_data():
        return

    rng = np.random.default_rng(SEED)

    train = generate_customers(TRAIN_ROWS, rng, start_id=15_600_000)
    test = generate_customers(TEST_ROWS, rng, start_id=16_600_000)

    train.to_csv(TRAIN_PATH, index=False)
    wrote_test = False
    if TEST_PATH.exists():
        print(f"Existing test.csv found. Leaving it unchanged at {TEST_PATH}.")
    else:
        test[TEST_COLUMNS].to_csv(TEST_PATH, index=False)
        wrote_test = True

    print(f"Wrote {TRAIN_PATH} with shape {train.shape}; churn rate={train['Exited'].mean():.4f}")
    if wrote_test:
        print(f"Wrote {TEST_PATH} with shape {test[TEST_COLUMNS].shape}")


if __name__ == "__main__":
    main()
