import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


def load_data(path: str = "data/heart.csv") -> pd.DataFrame:
    """
    Loads and cleans your heart dataset.

    - Uses 'num' as original disease severity label (0..4)
    - Converts it to a binary 'target' (0 = no disease, 1 = disease)
    - Drops ID/metadata columns: id, dataset, num
    - One-hot encodes categorical variables so all features are numeric
    """

    print(f"Trying to load dataset from: {os.path.abspath(path)}")

    if not os.path.exists(path):
        raise FileNotFoundError(f"Could not find {path}. Make sure data/heart.csv exists.")

    # Read CSV, treating '?' as NaN just in case
    df = pd.read_csv(path, na_values=["?"])
    print("Columns in dataset:", df.columns.tolist())

    if "num" not in df.columns:
        raise ValueError("Expected a 'num' column in heart.csv (disease severity).")

    # Convert num (0..4) -> binary target
    df["target"] = (df["num"] > 0).astype(int)

    # Drop non-feature columns
    drop_cols = ["id", "dataset", "num"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # Separate features and target
    y = df["target"].copy()
    X = df.drop(columns=["target"])

    # Identify categorical vs numeric
    cat_cols = X.select_dtypes(include=["object", "bool"]).columns.tolist()
    num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

    print("\nDetected categorical columns:", cat_cols)
    print("Detected numeric columns:", num_cols)

    # One-hot encode categorical columns so everything becomes numeric
    X_encoded = pd.get_dummies(X, columns=cat_cols, drop_first=True)

    # Impute any remaining NaNs in numeric columns with column mean
    X_encoded = X_encoded.fillna(X_encoded.mean())

    # Final dataframe: all numeric features + binary target
    final_df = X_encoded.copy()
    final_df["target"] = y.values

    print("\nAfter encoding, columns:", final_df.columns.tolist())
    print("Target value counts:\n", final_df["target"].value_counts(), "\n")

    return final_df


def simulate_potential_outcomes(df: pd.DataFrame, outcome_col: str = "target") -> pd.DataFrame:
    """
    Simulate a randomized clinical trial using potential outcomes.
    All feature columns are numeric at this point.
    """
    # Features only
    X = df.drop(columns=[outcome_col])

    # Impute any remaining NaNs (safety)
    X = X.fillna(X.mean())

    n, d = X.shape
    print(f"Simulating trial with {n} patients and {d} numeric features.")

    # Standardize with protection against zero std
    means = X.mean()
    stds = X.std(ddof=0).replace(0, 1.0)  # avoid division by zero

    X_std = (X - means) / stds

    # Hidden coefficients
    beta = np.random.normal(loc=0.0, scale=0.5, size=d)

    base_logit = X_std.values @ beta
    delta = -0.7  # protective treatment effect

    def sigmoid(z):
        return 1.0 / (1.0 + np.exp(-z))

    p0 = sigmoid(base_logit)
    p1 = sigmoid(base_logit + delta)

    # Safety: replace NaNs and clip probabilities into (0,1)
    p0 = np.nan_to_num(p0, nan=0.5)
    p1 = np.nan_to_num(p1, nan=0.5)

    p0 = np.clip(p0, 1e-6, 1 - 1e-6)
    p1 = np.clip(p1, 1e-6, 1 - 1e-6)

    # Sample potential outcomes
    Y0 = np.random.binomial(1, p0)
    Y1 = np.random.binomial(1, p1)

    # Random treatment assignment
    treatment = np.random.binomial(1, 0.5, size=n)

    # Observed outcome
    Y_obs = treatment * Y1 + (1 - treatment) * Y0

    sim_df = X.copy()
    sim_df["treatment"] = treatment
    sim_df["outcome"] = Y_obs

    print("Simulation complete. Shape:", sim_df.shape)
    return sim_df


def main():
    print("Current working directory:", os.getcwd())

    # 1. Load & encode dataset
    df = load_data()

    # 2. Simulate blinded clinical trial
    sim_df = simulate_potential_outcomes(df)

    # 3. Split into train/val vs test (patient-level)
    train_val_df, test_df = train_test_split(
        sim_df,
        test_size=0.2,
        random_state=RANDOM_SEED,
        stratify=sim_df["outcome"],
    )

    # 4. Save to disk
    train_path = "trial_train_val.csv"
    test_path = "trial_test_blinded.csv"

    train_val_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print("\nSaved files:")
    print(f"  Train/Val: {train_path}  shape={train_val_df.shape}")
    print(f"  Test (blinded): {test_path}  shape={test_df.shape}")


if __name__ == "__main__":
    main()
