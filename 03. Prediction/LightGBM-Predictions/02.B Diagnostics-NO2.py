#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 13:20:37 2026

@author: antonioraphael
"""
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score
import lightgbm as lgb

results_df = pd.read_csv("/Users/antonioraphael/Documents/PROJECT-CLONES/XCroissants-Predicting-Air-Quality/03. Prediction/LightGBM-Predictions/Outputs-no2/predictions_vs_actuals.csv")
model = lgb.Booster(model_file="/Users/antonioraphael/Documents/PROJECT-CLONES/XCroissants-Predicting-Air-Quality/03. Prediction/LightGBM-Predictions/Outputs-no2/lgbm_no2_model.txt")
feature_cols = json.load(open("/Users/antonioraphael/Documents/PROJECT-CLONES/XCroissants-Predicting-Air-Quality/03. Prediction/LightGBM-Predictions/Outputs-no2/feature_cols.json"))
best_params  = json.load(open("/Users/antonioraphael/Documents/PROJECT-CLONES/XCroissants-Predicting-Air-Quality/03. Prediction/LightGBM-Predictions/Outputs-no2/best_params.json"))

test = pd.read_csv("/Users/antonioraphael/Documents/PROJECT-CLONES/XCroissants-Predicting-Air-Quality/03. Prediction/00.A Train-Test-Communes/Test-Communes.csv")
data = pd.read_csv("/Users/antonioraphael/Documents/PROJECT-CLONES/Data-Storage/AirQualityData/PredictionData/AirQualityData_Imputed_Feature_Engineered.csv")

test_df = data[data['ninsee'].isin(test['Commune'])]
test_df_no2 = test_df.drop(columns = ['o3', 'pm10'])


# ── Plot styling ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor"  : "white",
    "axes.facecolor"    : "white",
    "axes.grid"         : True,
    "grid.alpha"        : 0.3,
    "axes.spines.top"   : False,
    "axes.spines.right" : False,
    "font.size"         : 11,
})


def plot_all_diagnostics(model, test_df, feature_cols, target_col="no2", save_path="diagnostics.png"):

    y_true = test_df[target_col].values
    y_pred = model.predict(test_df[feature_cols], num_iteration=model.best_iteration)
    residuals = y_true - y_pred

    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    r2   = r2_score(y_true, y_pred)
    mae  = np.mean(np.abs(residuals))

    fig = plt.figure(figsize=(18, 12))
    fig.suptitle(
        f"LightGBM no2 — Test set diagnostics (unseen communes)\n"
        f"RMSE: {rmse:.4f}  |  R²: {r2:.4f}  |  MAE: {mae:.4f}",
        fontsize=14, fontweight="bold", y=0.98
    )
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

    # ── 1. Predicted vs Actual ────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.scatter(y_true, y_pred, alpha=0.3, s=8, color="#1a6bb0", rasterized=True)
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax1.plot(lims, lims, "r--", linewidth=1.5, label="Perfect prediction")
    ax1.set_xlabel("Actual no2")
    ax1.set_ylabel("Predicted no2")
    ax1.set_title("Predicted vs Actual")
    ax1.legend(fontsize=9)

    # ── 2. Residuals vs Predicted ─────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.scatter(y_pred, residuals, alpha=0.3, s=8, color="#e07b39", rasterized=True)
    ax2.axhline(0, color="red", linewidth=1.5, linestyle="--")
    ax2.set_xlabel("Predicted no2")
    ax2.set_ylabel("Residual (actual − predicted)")
    ax2.set_title("Residuals vs Predicted")
    # Add a smoothed trend line to spot systematic bias
    sort_idx = np.argsort(y_pred)
    window = max(1, len(y_pred) // 50)
    smooth = pd.Series(residuals[sort_idx]).rolling(window, center=True).mean()
    ax2.plot(y_pred[sort_idx], smooth, color="darkred", linewidth=1.5, label="Trend")
    ax2.legend(fontsize=9)

    # ── 3. Residual distribution ──────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    sns.histplot(residuals, bins=60, kde=True, ax=ax3, color="#5b8a3c", alpha=0.6)
    ax3.axvline(0, color="red", linewidth=1.5, linestyle="--")
    ax3.axvline(residuals.mean(), color="orange", linewidth=1.5,
                linestyle="--", label=f"Mean: {residuals.mean():.3f}")
    ax3.set_xlabel("Residual")
    ax3.set_ylabel("Count")
    ax3.set_title("Residual distribution")
    ax3.legend(fontsize=9)

    # ── 4. Feature importance (top 20) ───────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 0])
    importance = pd.Series(
        model.feature_importance(importance_type='split'),
        index=feature_cols
    ).sort_values(ascending=False).head(20)
    importance.sort_values().plot(kind="barh", ax=ax4, color="#1a6bb0", alpha=0.8)
    ax4.set_xlabel("Importance (split count)")
    ax4.set_title("Top 20 features by importance")
    ax4.tick_params(axis="y", labelsize=8)

    # ── 5. Mean absolute error by commune (top 20 worst) ─────────────────────
    ax5 = fig.add_subplot(gs[1, 1])
    commune_col = "ninsee"
    commune_mae = (
        test_df.assign(abs_error=np.abs(residuals))
        .groupby(commune_col)["abs_error"]
        .mean()
        .sort_values(ascending=False)
        .head(20)
    )
    commune_mae.sort_values().plot(kind="barh", ax=ax5, color="#e07b39", alpha=0.8)
    ax5.set_xlabel("Mean absolute error")
    ax5.set_title("Top 20 worst communes (MAE)")
    ax5.tick_params(axis="y", labelsize=8)

    # ── 6. Cumulative error distribution (CDF) ────────────────────────────────
    ax6 = fig.add_subplot(gs[1, 2])
    abs_errors = np.abs(residuals)
    sorted_errors = np.sort(abs_errors)
    cdf = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
    ax6.plot(sorted_errors, cdf, color="#5b8a3c", linewidth=2)
    # Mark the 50th, 90th, 95th percentiles
    for pct, col in [(0.5, "orange"), (0.9, "red"), (0.95, "darkred")]:
        val = np.quantile(abs_errors, pct)
        ax6.axvline(val, color=col, linestyle="--", linewidth=1.2,
                    label=f"p{int(pct*100)}: {val:.2f}")
    ax6.set_xlabel("Absolute error")
    ax6.set_ylabel("Cumulative proportion of predictions")
    ax6.set_title("Error CDF")
    ax6.legend(fontsize=9)

    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved to {save_path}")
    plt.show()
    return fig


# ── Usage ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    plot_all_diagnostics(
        model=model,
        test_df=test_df_no2,
        feature_cols=feature_cols,
        target_col="no2",
        save_path="/Users/antonioraphael/Documents/PROJECT-CLONES/XCroissants-Predicting-Air-Quality/03. Prediction/LightGBM-Predictions/Outputs-no2/no2_diagnostics.png",
    )
