import numpy as np
import pandas as pd
import lightgbm as lgb
import optuna
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold


train = pd.read_csv("/users/eleves-b/2025/antonio.raphael/Documents/XCroissants-Predicting-Air-Quality/03. Prediction/00.A Train-Test-Communes/Train-Communes.csv")
test = pd.read_csv("/users/eleves-b/2025/antonio.raphael/Documents/XCroissants-Predicting-Air-Quality/03. Prediction/00.A Train-Test-Communes/Test-Communes.csv")
data = pd.read_csv("/users/eleves-b/2025/antonio.raphael/Downloads/AirQualityData_Imputed_Feature_Engineered.csv")

train_df = data[data['ninsee'].isin(train['Commune'])]
test_df = data[data['ninsee'].isin(test['Commune'])]

train_df_no2 = train_df.drop(columns = ['pm10', 'o3'])
test_df_no2 = test_df.drop(columns = ['pm10', 'o3'])

# ── 2. LightGBM — Optuna tuning + final fit ───────────────────────────────────

def tune_and_fit_lgbm(
    train_df: pd.DataFrame,
    target_col: str = "target",
    commune_col: str = "ninsee",
    n_trials: int = 40,
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
    feature_cols = [c for c in train_df.columns if c not in [target_col, commune_col, "Date"]]
    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    communes_train = train_df[commune_col]

    # Build county-level CV folds
    unique_communes = np.unique(communes_train)
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    commune_folds = list(kf.split(unique_communes))   # splits over counties

    def get_row_indices(communes_subset):
        """Return row indices in train_df whose county is in county_subset."""
        mask = np.isin(communes_train, communes_subset)
        return np.where(mask)[0]

    # ── Optuna objective ──────────────────────────────────────────────────────
    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators"    : 500,          # controlled by early stopping
            "learning_rate"   : trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "num_leaves"      : trial.suggest_int("num_leaves", 20, 64),
            "max_depth"       : trial.suggest_int("max_depth", 3, 10),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
            "subsample"       : trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha"       : trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda"      : trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "random_state"    : random_state,
            "n_jobs"          : 4,
            "verbose"         : -1,
        }

        fold_scores = []
        for train_communes_idx, val_communes_idx in commune_folds:
            train_communes_fold = unique_communes[train_communes_idx]
            val_communes_fold   = unique_communes[val_communes_idx]

            tr_idx  = get_row_indices(train_communes_fold)
            val_idx = get_row_indices(val_communes_fold)

            X_tr,  y_tr  = X_train.iloc[tr_idx],  y_train.iloc[tr_idx]
            X_val, y_val = X_train.iloc[val_idx], y_train.iloc[val_idx]

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
            fold_scores.append(mean_squared_error(y_val, preds) ** 0.5)

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
    X_test = test_df[feature_cols]
    y_test = test_df[target_col]

    preds = model.predict(X_test, num_iteration=model.best_iteration_)
    rmse = mean_squared_error(y_test, preds) ** 0.5
    r2    = r2_score(y_test, preds)

    print(f"\nTest set performance (unseen counties):")
    print(f"  RMSE : {rmse:.4f}")
    print(f"  R²   : {r2:.4f}")
    return preds

if __name__ == "__main__":
    # 2. Tune and fit LightGBM
    model, best_params, feature_cols = tune_and_fit_lgbm(
        train_df_no2,
        target_col="no2",
        commune_col="ninsee",
        n_trials=40,
        cv_folds=5,
    )

    # 3. Evaluate on held-out test counties
    test_preds = evaluate(model, test_df_no2, feature_cols, target_col="no2")

print(test_preds)

import os, json
OUTPUT_DIR = "/users/eleves-b/2025/antonio.raphael/Documents/XCroissants-Predicting-Air-Quality/03. Prediction/LightGBM-Predictions/Outputs-no2"
os.makedirs(OUTPUT_DIR, exist_ok=True)

y_true = test_df_no2["no2"].values
y_pred = model.predict(test_df_no2[feature_cols], num_iteration=model.best_iteration_)

pd.DataFrame({
    "ninsee"         : test_df_no2["ninsee"].values,
    "actual_no2"    : y_true,
    "predicted_no2" : y_pred,
    "residual"       : y_true - y_pred,
    "abs_error"      : np.abs(y_true - y_pred),
}).to_csv(os.path.join(OUTPUT_DIR, "predictions_vs_actuals.csv"), index=False)

model.booster_.save_model(os.path.join(OUTPUT_DIR, "lgbm_no2_model.txt"))
json.dump(best_params, open(os.path.join(OUTPUT_DIR, "best_params.json"), "w"), indent=2)
json.dump(feature_cols, open(os.path.join(OUTPUT_DIR, "feature_cols.json"), "w"), indent=2)

print(f"All outputs saved to {OUTPUT_DIR}")