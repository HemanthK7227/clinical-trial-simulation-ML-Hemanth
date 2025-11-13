import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    brier_score_loss,
    confusion_matrix,
    classification_report,
)

RANDOM_SEED = 42


def load_data():
    """
    Load the simulated trial train/val data and blinded test data.
    'outcome' is the trial endpoint; all other columns are features.
    """
    train_val = pd.read_csv("trial_train_val.csv")
    test = pd.read_csv("trial_test_blinded.csv")

    X_train_val = train_val.drop(columns=["outcome"])
    y_train_val = train_val["outcome"]

    X_test = test.drop(columns=["outcome"])
    y_test = test["outcome"]

    return X_train_val, y_train_val, X_test, y_test


def build_pipeline(X):
    """
    Build a scikit-learn Pipeline:
      - Standardizes all numeric columns
      - Fits a Logistic Regression classifier
    All columns are numeric/bool because we one-hot encoded categoricals earlier.
    """
    numeric_cols = X.columns.tolist()
    print("Numeric feature columns:", numeric_cols)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
        ],
        remainder="drop",
    )

    clf = LogisticRegression(
        max_iter=1000,
        random_state=RANDOM_SEED,
    )

    pipe = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("clf", clf),
        ]
    )

    return pipe


def main():
    # 1. Load data
    X_train_val, y_train_val, X_test, y_test = load_data()

    # 2. Build pipeline
    pipe = build_pipeline(X_train_val)

    # 3. Hyperparameter grid for Logistic Regression
    param_grid = {
        "clf__C": [0.1, 1.0, 10.0],
        "clf__penalty": ["l2"],
        "clf__solver": ["lbfgs"],
    }

    # 4. Grid search with cross-validation on train_val set
    search = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        cv=5,
        scoring="roc_auc",
        n_jobs=-1,
        refit=True,
    )

    print("\nFitting Logistic Regression with cross-validation...")
    search.fit(X_train_val, y_train_val)

    print("Best params:", search.best_params_)
    print("Best CV ROC-AUC:", search.best_score_)

    best_model = search.best_estimator_

    # 5. Final blinded evaluation on held-out test set
    y_proba = best_model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    test_auc = roc_auc_score(y_test, y_proba)
    test_brier = brier_score_loss(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, digits=3)

    print("\n=== Final Blinded Test Evaluation (Logistic Regression) ===")
    print("Test ROC-AUC:", test_auc)
    print("Test Brier score:", test_brier)
    print("\nConfusion matrix:\n", cm)
    print("\nClassification report:\n", report)


if __name__ == "__main__":
    main()
