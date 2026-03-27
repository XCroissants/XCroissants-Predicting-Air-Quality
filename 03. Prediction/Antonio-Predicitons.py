"""
ML Pipeline: County-based train/test split + LightGBM with Optuna tuning
-------------------------------------------------------------------------
Assumptions:
  - df is a pandas DataFrame with one row per observation
  - 'county_id' column identifies the county each observation belongs to
  - 'target' column is the continuous outcome you want to predict
  - All other columns are features

Dependencies:
    pip install lightgbm optuna scikit-learn pandas numpy
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
import optuna
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold


# ── 1. County-based train/test split ─────────────────────────────────────────

def county_train_test_split(
    df: pd.DataFrame,
    county_col: str = "county_id",
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split a dataset into train and test by county, not by row.

    All observations from a given county land entirely in train or entirely
    in test — no county straddles the boundary. This gives an honest estimate
    of how well the model generalises to counties it has never seen.

    Args:
        df:           Full dataset with a county identifier column.
        county_col:   Name of the column containing county IDs.
        test_size:    Fraction of counties to hold out for testing (default 0.2).
        random_state: Random seed for reproducibility.

    Returns:
        (train_df, test_df) — two DataFrames, split by county.
    """
    rng = np.random.default_rng(random_state)

    counties = df[county_col].unique()
    n_test = max(1, int(len(counties) * test_size))

    test_counties = rng.choice(counties, size=n_test, replace=False)
    train_counties = np.setdiff1d(counties, test_counties)

    train_df = df[df[county_col].isin(train_counties)].copy()
    test_df  = df[df[county_col].isin(test_counties)].copy()

    print(f"Total counties : {len(counties)}")
    print(f"Train counties : {len(train_counties)}  ({len(train_df):,} rows)")
    print(f"Test counties  : {len(test_counties)}   ({len(test_df):,} rows)")

    return train_df, test_df


# ── 2. LightGBM — Optuna tuning + final fit ───────────────────────────────────

def tune_and_fit_lgbm(
    train_df: pd.DataFrame,
    target_col: str = "target",
    county_col: str = "county_id",
    n_trials: int = 50,
    cv_folds: int = 5,
    random_state: int = 42,
) -> tuple[lgb.LGBMRegressor, dict, list[str]]:
    """
    Tune LightGBM hyperparameters with Optuna (county-aware CV), then refit
    on the full training set using the best parameters.

    Cross-validation folds are constructed at the county level so that no
    county appears in both the fold's train and validation splits — keeping
    the same generalisation guarantee as the outer train/test split.

    Args:
        train_df:   Training data (output of county_train_test_split).
        target_col: Name of the target column.
        county_col: Name of the county identifier column.
        n_trials:   Number of Optuna trials (default 50; raise for better tuning).
        cv_folds:   Number of cross-validation folds (default 5).
        random_state: Random seed.

    Returns:
        model       — fitted LGBMRegressor on full training data
        best_params — dict of best hyperparameters found by Optuna
        feature_cols — list of feature column names used
    """
    feature_cols = [c for c in train_df.columns if c not in [target_col, county_col]]
    X_train = train_df[feature_cols].values
    y_train = train_df[target_col].values
    counties_train = train_df[county_col].values

    # Build county-level CV folds
    unique_counties = np.unique(counties_train)
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    county_folds = list(kf.split(unique_counties))   # splits over counties

    def get_row_indices(county_subset):
        """Return row indices in train_df whose county is in county_subset."""
        mask = np.isin(counties_train, county_subset)
        return np.where(mask)[0]

    # ── Optuna objective ──────────────────────────────────────────────────────
    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators"    : 2000,          # controlled by early stopping
            "learning_rate"   : trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "num_leaves"      : trial.suggest_int("num_leaves", 20, 300),
            "max_depth"       : trial.suggest_int("max_depth", 3, 10),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
            "subsample"       : trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha"       : trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda"      : trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "random_state"    : random_state,
            "n_jobs"          : -1,
            "verbose"         : -1,
        }

        fold_scores = []
        for train_county_idx, val_county_idx in county_folds:
            train_counties_fold = unique_counties[train_county_idx]
            val_counties_fold   = unique_counties[val_county_idx]

            tr_idx  = get_row_indices(train_counties_fold)
            val_idx = get_row_indices(val_counties_fold)

            X_tr,  y_tr  = X_train[tr_idx],  y_train[tr_idx]
            X_val, y_val = X_train[val_idx], y_train[val_idx]

            model = lgb.LGBMRegressor(**params)
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                callbacks=[
                    lgb.early_stopping(50, verbose=False),
                    lgb.log_evaluation(-1),
                ],
            )
            preds = model.predict(X_val, num_iteration=model.best_iteration_)
            fold_scores.append(mean_squared_error(y_val, preds, squared=False))  # RMSE

        return np.mean(fold_scores)

    # ── Run Optuna ────────────────────────────────────────────────────────────
    print(f"\nRunning Optuna tuning ({n_trials} trials, {cv_folds}-fold county CV)...")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=random_state))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best_params = study.best_params
    best_params.update({"n_estimators": 2000, "random_state": random_state, "n_jobs": -1, "verbose": -1})
    print(f"\nBest CV RMSE : {study.best_value:.4f}")
    print(f"Best params  : {best_params}")

    # ── Refit on full training data ───────────────────────────────────────────
    # Use a 10% internal validation split just to drive early stopping;
    # this split is random since we only need it for the stopping criterion,
    # not for evaluation.
    val_cut = int(len(X_train) * 0.9)
    X_tr_full, X_val_es = X_train[:val_cut], X_train[val_cut:]
    y_tr_full, y_val_es = y_train[:val_cut], y_train[val_cut:]

    final_model = lgb.LGBMRegressor(**best_params)
    final_model.fit(
        X_tr_full, y_tr_full,
        eval_set=[(X_val_es, y_val_es)],
        callbacks=[
            lgb.early_stopping(50, verbose=False),
            lgb.log_evaluation(100),
        ],
    )
    print(f"\nFinal model trained with {final_model.best_iteration_} trees.")

    return final_model, best_params, feature_cols


# ── 3. Evaluate on held-out test set ─────────────────────────────────────────

def evaluate(model, test_df: pd.DataFrame, feature_cols: list[str], target_col: str = "target"):
    """Print RMSE and R² on the held-out test counties."""
    X_test = test_df[feature_cols].values
    y_test = test_df[target_col].values

    preds = model.predict(X_test, num_iteration=model.best_iteration_)
    rmse  = mean_squared_error(y_test, preds, squared=False)
    r2    = r2_score(y_test, preds)

    print(f"\nTest set performance (unseen counties):")
    print(f"  RMSE : {rmse:.4f}")
    print(f"  R²   : {r2:.4f}")
    return preds


# ── 4. Example usage ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Replace this block with your actual data loading
    rng = np.random.default_rng(0)
    n_obs, n_features, n_counties = 500_000, 140, 1287

    county_ids = rng.integers(0, n_counties, size=n_obs)
    X_demo = rng.standard_normal((n_obs, n_features))
    y_demo = X_demo[:, :5].sum(axis=1) + rng.standard_normal(n_obs) * 0.5

    df = pd.DataFrame(X_demo, columns=[f"f{i}" for i in range(n_features)])
    df["county_id"] = county_ids
    df["target"]    = y_demo

    # 1. Split by county
    train_df, test_df = county_train_test_split(df, county_col="county_id", test_size=0.2)

    # 2. Tune and fit LightGBM
    model, best_params, feature_cols = tune_and_fit_lgbm(
        train_df,
        target_col="target",
        county_col="county_id",
        n_trials=50,
        cv_folds=5,
    )

    # 3. Evaluate on held-out test counties
    test_preds = evaluate(model, test_df, feature_cols, target_col="target")
