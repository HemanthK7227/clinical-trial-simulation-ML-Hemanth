import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
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
    'outcome' is the endpoint.
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
    Build a pipeline with:
      - StandardScaler on all numeric features
      - MLPClassifier as the model
    """
    numeric_cols = X.columns.tolist()
    print("Numeric feature columns:", numeric_cols)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
        ],
        remainder="drop",
    )

    mlp = MLPClassifier(
        max_iter=300,
        random_state=RANDOM_SEED,
    )

    pipe = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("clf", mlp),
        ]
    )

    return pipe


def main():
    X_train_val, y_train_val, X_test, y_test = load_data()

    pipe = build_pipeline(X_train_val)

    param_grid = {
        "clf__hidden_layer_sizes": [(64,), (64, 32), (128, 64)],
        "clf__alpha": [1e-4, 1e-3, 1e-2],
        "clf__learning_rate_init": [1e-3, 5e-4],
    }

    search = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        cv=3,               # 3-fold for speed
        scoring="roc_auc",
        n_jobs=-1,
        refit=True,
    )

    print("\nFitting MLPClassifier with cross-validation...")
    search.fit(X_train_val, y_train_val)

    print("Best params:", search.best_params_)
    print("Best CV ROC-AUC:", search.best_score_)

    best_model = search.best_estimator_

    y_proba = best_model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    test_auc = roc_auc_score(y_test, y_proba)
    test_brier = brier_score_loss(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, digits=3)

    print("\n=== Final Blinded Test Evaluation (MLP) ===")
    print("Test ROC-AUC:", test_auc)
    print("Test Brier score:", test_brier)
    print("\nConfusion matrix:\n", cm)
    print("\nClassification report:\n", report)


if __name__ == "__main__":
    main()
