"""
product_model.py  --  Product Recommendation Model (Member 4)

What this script does
---------------------
1. Loads data/product_model_data.csv (prepared by Member 1)
2. Cleans it (removes exact duplicate rows) and engineers features
3. Compares three models with customer-grouped cross-validation:
       Logistic Regression, Random Forest, XGBoost (falls back to
       Gradient Boosting if xgboost is not installed)
4. Evaluates the models on a held-out test set (accuracy, weighted F1, log loss)
   and compares them with a "most frequent class" baseline
5. Saves the best model to models/product_model.pkl
6. Saves tables and plots to outputs/product_model/
7. Exposes predict_product(customer_id) for the command-line app (app.py)

How to run (from the repo root)
-------------------------------
    python src/product_model.py                     # train + evaluate + save
    python src/product_model.py --predict A151      # predict for one customer
    python src/product_model.py --list-customers    # show valid customer IDs

In app.py:
    from src.product_model import predict_product
    result = predict_product("A151")
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")  # lets plots be saved without a display (works in Colab / servers)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
    log_loss,
)
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

try:
    from xgboost import XGBClassifier

    HAS_XGB = True
except ImportError:  # keeps the script working if xgboost is not installed
    HAS_XGB = False

# Paths and settings  (change these if your folder names are different) 
ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "product_model_data.csv"
MODEL_PATH = ROOT / "models" / "product_model.pkl"
OUT_DIR = ROOT / "outputs" / "product_model"

TARGET = "product_category"
ID_COL = "customer_id"
RANDOM_STATE = 42
N_SPLITS = 5          # cross-validation folds
TEST_SIZE = 0.2       # share of CUSTOMERS held out for the final test
# engagement_level and interest_level are just binned versions of the two
# scores, so they are dropped to avoid redundant features.
DROP_REDUNDANT = ["engagement_level", "interest_level"]

NUMERIC = ["platform_count", "engagement_score", "purchase_interest_score"]
CATEGORICAL = ["review_sentiment"]

# 1. Load, clean, feature engineering
def load_and_clean(path=DATA_PATH):
    """Load the CSV and remove exact duplicate rows."""
    df = pd.read_csv(path)
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"[data] loaded {before} rows, removed {before - len(df)} exact duplicates "
          f"-> {len(df)} rows, {df[ID_COL].nunique()} unique customers")
    return df


def add_platform_columns(df, platforms=None):
    """
    'social_media_platform' holds strings like 'TikTok, Twitter'.
    Turn it into one 0/1 column per platform (platform_TikTok, ...).
    At prediction time, pass the platform list saved at training time.
    """
    df = df.copy()
    dummies = df["social_media_platform"].str.get_dummies(sep=", ")
    dummies.columns = ["platform_" + c.strip() for c in dummies.columns]
    if platforms is not None:  # force the same columns as training
        dummies = dummies.reindex(columns=platforms, fill_value=0)
    return pd.concat([df, dummies], axis=1)


def build_features(df, platforms=None):
    """Return X (features only). Platform columns are created here."""
    df = add_platform_columns(df, platforms)
    platform_cols = [c for c in df.columns if c.startswith("platform_") and c != "platform_count"]
    X = df[NUMERIC + CATEGORICAL + platform_cols]
    return X, platform_cols


def make_preprocessor(platform_cols):
    return ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
            ("num", StandardScaler(), NUMERIC),
            ("plat", "passthrough", platform_cols),
        ]
    )


# 2. Models
def get_models(n_classes):
    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=3, random_state=RANDOM_STATE
        ),
    }
    if HAS_XGB:
        models["XGBoost"] = XGBClassifier(
            n_estimators=150, max_depth=3, learning_rate=0.05,
            objective="multi:softprob", num_class=n_classes,
            eval_metric="mlogloss", random_state=RANDOM_STATE, verbosity=0,
        )
    else:
        print("[warn] xgboost not installed -> using Gradient Boosting instead "
              "(run: pip install xgboost)")
        models["Gradient Boosting"] = GradientBoostingClassifier(
            n_estimators=100, max_depth=2, learning_rate=0.05, random_state=RANDOM_STATE
        )
    return models


def full_proba(pipe, X, n_classes):
    """predict_proba with a column for every class (even if a fold missed one)."""
    p = pipe.predict_proba(X)
    out = np.zeros((len(X), n_classes))
    out[:, pipe.named_steps["model"].classes_] = p
    return out


def score(y_true, proba, n_classes):
    """Accuracy, weighted F1 and log loss from predicted probabilities."""
    pred = proba.argmax(axis=1)
    return {
        "accuracy": accuracy_score(y_true, pred),
        "f1_weighted": f1_score(y_true, pred, average="weighted", zero_division=0),
        "log_loss": log_loss(y_true, proba, labels=list(range(n_classes))),
    }

# 3. Train + evaluate + save
def train():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    df = load_and_clean()
    X, platform_cols = build_features(df)
    le = LabelEncoder()
    y = le.fit_transform(df[TARGET])
    groups = df[ID_COL]
    n_classes = len(le.classes_)

    print(f"[data] classes: {list(le.classes_)}")
    print(f"[data] features: {list(X.columns)}")

    # hold out whole customers for the final test (no customer in both sets)
    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    train_idx, test_idx = next(gss.split(X, y, groups))
    X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
    y_tr, y_te = y[train_idx], y[test_idx]
    g_tr = groups.iloc[train_idx]
    print(f"[split] train: {len(X_tr)} rows / {g_tr.nunique()} customers | "
          f"test: {len(X_te)} rows / {groups.iloc[test_idx].nunique()} customers")

    models = get_models(n_classes)
    baseline = DummyClassifier(strategy="prior")  # always predicts the common class

    rows = []
    fitted = {}
    cv = GroupKFold(n_splits=N_SPLITS)

    for name, clf in {"Baseline (most frequent)": baseline, **models}.items():
        pipe = Pipeline([("prep", make_preprocessor(platform_cols)), ("model", clf)])

        # grouped cross-validation on the training part
        cv_scores = []
        for tr, va in cv.split(X_tr, y_tr, g_tr):
            pipe.fit(X_tr.iloc[tr], y_tr[tr])
            cv_scores.append(score(y_tr[va], full_proba(pipe, X_tr.iloc[va], n_classes), n_classes))
        cv_df = pd.DataFrame(cv_scores)

        # final fit on the whole training part, score on the held-out test
        pipe.fit(X_tr, y_tr)
        test_proba = full_proba(pipe, X_te, n_classes)
        test = score(y_te, test_proba, n_classes)
        train_loss = log_loss(y_tr, full_proba(pipe, X_tr, n_classes), labels=list(range(n_classes)))

        rows.append({
            "model": name,
            "cv_accuracy": cv_df["accuracy"].mean(),
            "cv_f1_weighted": cv_df["f1_weighted"].mean(),
            "cv_log_loss": cv_df["log_loss"].mean(),
            "test_accuracy": test["accuracy"],
            "test_f1_weighted": test["f1_weighted"],
            "test_log_loss": test["log_loss"],
            "train_log_loss": train_loss,
        })
        fitted[name] = (pipe, test_proba)

    results = pd.DataFrame(rows).set_index("model").round(4)
    results.to_csv(OUT_DIR / "model_comparison.csv")
    print("\n=== Model comparison (CV = grouped 5-fold on train; test = held-out customers) ===")
    print(results.to_string())

    # pick the best real model by CV weighted F1 (ties broken by lower CV log loss)
    candidates = results.drop(index="Baseline (most frequent)")
    best_name = candidates.sort_values(["cv_f1_weighted", "cv_log_loss"],
                                       ascending=[False, True]).index[0]
    best_pipe, best_test_proba = fitted[best_name]
    print(f"\n[best] {best_name} (highest CV weighted F1)")

    best_pred = best_test_proba.argmax(axis=1)
    report = classification_report(y_te, best_pred, labels=list(range(n_classes)),
                                   target_names=le.classes_, zero_division=0)
    print("\nClassification report on held-out test set:\n" + report)
    (OUT_DIR / "classification_report.txt").write_text(report)

    # plots
    plot_model_comparison(results)
    plot_confusion(y_te, best_pred, le.classes_, best_name)
    plot_feature_importance(best_pipe, best_name)

    # refit best model on ALL data, then save it for the command-line app
    final_pipe = Pipeline([("prep", make_preprocessor(platform_cols)),
                           ("model", get_models(n_classes)[best_name])])
    final_pipe.fit(X, y)
    joblib.dump({
        "pipeline": final_pipe,
        "label_encoder": le,
        "platform_cols": platform_cols,
        "model_name": best_name,
        "metrics": results.loc[best_name].to_dict(),
    }, MODEL_PATH)
    (OUT_DIR / "metrics.json").write_text(json.dumps(
        {"best_model": best_name, "results": results.to_dict(orient="index")}, indent=2))
    print(f"\n[saved] model   -> {MODEL_PATH}")
    print(f"[saved] outputs -> {OUT_DIR}")
    return results


# 4. Plots
def plot_model_comparison(results):
    cols = ["cv_accuracy", "cv_f1_weighted", "test_accuracy", "test_f1_weighted"]
    ax = results[cols].plot(kind="bar", figsize=(9, 5))
    ax.set_title("Model comparison (higher is better)")
    ax.set_ylabel("Score")
    ax.set_xlabel("")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "model_comparison.png", dpi=150)
    plt.close()


def plot_confusion(y_true, y_pred, class_names, model_name):
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, labels=list(range(len(class_names))),
        display_labels=class_names, ax=ax, cmap="Blues", xticks_rotation=45,
    )
    ax.set_title(f"Confusion matrix - {model_name} (held-out test)")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "confusion_matrix.png", dpi=150)
    plt.close()


def plot_feature_importance(pipe, model_name):
    model = pipe.named_steps["model"]
    names = pipe.named_steps["prep"].get_feature_names_out()
    if hasattr(model, "feature_importances_"):
        values = model.feature_importances_
    elif hasattr(model, "coef_"):
        values = np.abs(model.coef_).mean(axis=0)
    else:
        return
    imp = pd.Series(values, index=[n.split("__")[-1] for n in names]).sort_values().tail(12)
    imp.plot(kind="barh", figsize=(7, 5))
    plt.title(f"Top features - {model_name}")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "feature_importance.png", dpi=150)
    plt.close()


# 5. Prediction functions (used by app.py)
_ARTIFACT = None


def _load_artifact():
    global _ARTIFACT
    if _ARTIFACT is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model not found at {MODEL_PATH}. Run: python src/product_model.py")
        _ARTIFACT = joblib.load(MODEL_PATH)
    return _ARTIFACT


def list_customers():
    """All customer IDs that exist in the dataset."""
    return sorted(pd.read_csv(DATA_PATH)[ID_COL].unique())


def predict_from_row(row_df):
    """Predict from a one-row DataFrame that has the raw dataset columns."""
    art = _load_artifact()
    X, _ = build_features(row_df, platforms=art["platform_cols"])
    proba = art["pipeline"].predict_proba(X)[0]
    le = art["label_encoder"]
    order = np.argsort(proba)[::-1]
    return {
        "product": le.classes_[order[0]],
        "confidence": float(proba[order[0]]),
        "top3": [(le.classes_[i], float(proba[i])) for i in order[:3]],
        "model": art["model_name"],
    }


def predict_product(customer_id):
    """
    Look up a customer's social-media profile and predict the product category.
    Returns a dict: {"customer_id", "product", "confidence", "top3", "model"}
    Raises ValueError if the customer ID is not in the dataset.
    """
    df = pd.read_csv(DATA_PATH).drop_duplicates()
    rows = df[df[ID_COL] == customer_id]
    if rows.empty:
        raise ValueError(f"Unknown customer_id: {customer_id}")
    # A customer's features are identical on every row, so take the first.
    result = predict_from_row(rows.head(1))
    result["customer_id"] = customer_id
    return result


# 6. Command line
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Product recommendation model")
    parser.add_argument("--predict", metavar="CUSTOMER_ID", help="predict for one customer")
    parser.add_argument("--list-customers", action="store_true", help="print valid customer IDs")
    args = parser.parse_args()

    if args.list_customers:
        print(", ".join(list_customers()))
    elif args.predict:
        try:
            r = predict_product(args.predict)
        except (ValueError, FileNotFoundError) as e:
            sys.exit(f"Error: {e}")
        print(f"Customer {r['customer_id']} -> {r['product']} "
              f"(confidence {r['confidence']:.0%}, model: {r['model']})")
        for label, p in r["top3"]:
            print(f"   {label:<12} {p:.1%}")
    else:
        train()