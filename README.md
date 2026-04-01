# Predicting Air Quality

The final datasets can be found in the following Dropbox folder (too big for the repository) : https://www.dropbox.com/scl/fo/ez551qcvze3xgtkde9h0f/ADH3IJ7ZKtFiRHHCYmL2TT8?rlkey=vvdkm4m2123ehazl1m17dgyrw&st=pxy1egbk&dl=0 

This repository contains the code written for an Introduction to Machine Learning Course
final project completed by Louise Gatty, Antonio Raphael, and Anne Thebaud.

# Project Overview

This project aims to apply machine learning methods to predict daily air pollution 
levels in Ile-de-France communes. The authors used historic air pollution measurements
for PM10, O3, and NO2 published daily for all of the communes in Ile-de-France for the
year 2017. Predictions utilize three different prediction methods: OLS, Elastic net,
and LightGBM trained over a diverse feature space containing meteorological data
socio-economic, energy production and consumption, and terrain feature data.

# Directory Tree

```text
.
├── 00. Data csvs
│   ├── 1. initial raw data
│   │   ├── GES.xlsx
│   │   ├── PIB.xlsx
│   │   ├── demographie.xlsx
│   │   ├── eco.xlsx
│   │   ├── emploi.xlsx
│   │   ├── logement.xlsx
│   │   ├── mobilites.xlsx
│   │   ├── ressources.xlsx
│   │   ├── revenu.xlsx
│   │   └── transport.xlsx
│   ├── 2. data processed individually
│   │   ├── Commune_Energy_Data.csv
│   │   ├── Commune_Land_Data.csv
│   │   ├── Commune_Transport_Data.csv
│   │   ├── Energy_Names.csv
│   │   ├── Energy_Names.numbers
│   │   ├── SocioEcon_Names.csv
│   │   │   ├── Sheet 1-SocioEcon_Names.csv
│   │   │   └── Sheet 2-Table 1.csv
│   │   ├── energy_indicators.csv
│   │   ├── socio_eco_indicators.csv
│   │   ├── transport.xlsx
│   │   └── weather_2017_idf.csv
│   └── 3. Intermediary Data Sets
│       └── Imputed_Normalised_KNN_data.csv
├── 01. Code Data Cleaning
│   ├── 01.A Commune-Land-Type.Rmd
│   ├── 01.B Commune-Energy-Consumption.Rmd
│   ├── 01.C Commune-Transport-Data.Rmd
│   ├── 01.D Cleaning_energy.Rmd
│   ├── 01.E Cleaning_socio_eco.Rmd
│   ├── 01.F Pulling-Weather-Data
│   │   ├── API-Script-Batched.py
│   │   └── Commune-Centroids.Rmd
│   ├── 02.A Merging-Data-Sets.Rmd
│   ├── 02.B Feature Engineering.Rmd
│   └── 03.A Imputing-Nan-values-code.ipynb
├── 02. Data-Exploration
│   ├── Child-Markdowns
│   │   └── Summary-Statistics.Rmd
│   ├── Data-Distributions-2.Rmd
│   ├── Data-Distributions-2.html
│   ├── Data-Distributions.Rmd
│   ├── Data-Distributions.html
│   └── Imputation_KNN.ipynb
├── 03. Prediction
│   ├── 00. Train-Test-Communes.Rmd
│   ├── 00.A Train-Test-Communes
│   │   ├── Test-Communes.csv
│   │   └── Train-Communes.csv
│   ├── Elastic-Net-Predictions
│   │   ├── Elastic-Net-Predictions.ipynb
│   │   ├── Outputs-NO2
│   │   │   ├── elastic_net_best_params.json
│   │   │   ├── elastic_net_coeffs.txt
│   │   │   ├── feature_cols.json
│   │   │   ├── no2_diagnostics.png
│   │   │   └── predictions_vs_actuals.csv
│   │   ├── Outputs-O3
│   │   │   ├── elastic_net_best_params.json
│   │   │   ├── elastic_net_coeffs.txt
│   │   │   ├── feature_cols.json
│   │   │   ├── o3_diagnostics.png
│   │   │   └── predictions_vs_actuals.csv
│   │   └── Outputs-PM10
│   │       ├── elastic_net_best_params.json
│   │       ├── elastic_net_coeffs.txt
│   │       ├── feature_cols.json
│   │       ├── pm10_diagnostics.png
│   │       └── predictions_vs_actuals.csv
│   ├── LightGBM-Predictions
│   │   ├── 01.A 03-Predictions.py
│   │   ├── 01.B NO2-Predictions.py
│   │   ├── 01.C PM10-Predicitons.py
│   │   ├── 01.D NO2-Predictions-No-Lags.py
│   │   ├── 01.E PM-10-Predictions-No-Lags.py
│   │   ├── 01.F O3-Predictions-No-Lags.py
│   │   ├── 02.A Diagnostics-O3.py
│   │   ├── 02.B Diagnostics-NO2.py
│   │   ├── 02.C Diagnostics-PM10.py
│   │   ├── 02.D Diagnostics-NO2-No-Lags.py
│   │   ├── 02.E Diagnostics-PM10-No-Lags.py
│   │   ├── 02.F Diagnostics-O3-No-Lags.py
│   │   ├── Outputs-no2
│   │   │   ├── best_params.json
│   │   │   ├── feature_cols.json
│   │   │   ├── lgbm_no2_model.txt
│   │   │   ├── no2_diagnostics.png
│   │   │   └── predictions_vs_actuals.csv
│   │   ├── Outputs-no2-no-lag
│   │   │   ├── best_params.json
│   │   │   ├── feature_cols.json
│   │   │   ├── lgbm_no2_model.txt
│   │   │   ├── no2_diagnostics.png
│   │   │   └── predictions_vs_actuals.csv
│   │   ├── Outputs-o3
│   │   │   ├── best_params.json
│   │   │   ├── feature_cols.json
│   │   │   ├── lgbm_o3_model.txt
│   │   │   ├── o3_diagnostics.png
│   │   │   └── predictions_vs_actuals.csv
│   │   ├── Outputs-o3-no-lag
│   │   │   ├── best_params.json
│   │   │   ├── feature_cols.json
│   │   │   ├── lgbm_o3_model.txt
│   │   │   ├── o3_diagnostics.png
│   │   │   └── predictions_vs_actuals.csv
│   │   ├── Outputs-pm10
│   │   │   ├── best_params.json
│   │   │   ├── feature_cols.json
│   │   │   ├── lgbm_pm10_model.txt
│   │   │   ├── pm10_diagnostics.png
│   │   │   └── predictions_vs_actuals.csv
│   │   └── Outputs-pm10-No-Lags
│   │       ├── best_params.json
│   │       ├── feature_cols.json
│   │       ├── lgbm_pm10_model.txt
│   │       ├── pm10_diagnostics.png
│   │       └── predictions_vs_actuals.csv
│   └── OLS-Predictions
│       ├── OLS-Predictions.ipynb
│       ├── Outputs-NO2
│       │   ├── feature_cols.json
│       │   ├── linear_model_coeffs.txt
│       │   ├── no2_diagnostics.png
│       │   └── predictions_vs_actuals.csv
│       ├── Outputs-O3
│       │   ├── feature_cols.json
│       │   ├── linear_model_coeffs.txt
│       │   ├── o3_diagnostics.png
│       │   └── predictions_vs_actuals.csv
│       └── Outputs-PM10
│           ├── feature_cols.json
│           ├── linear_model_coeffs.txt
│           ├── pm10_diagnostics.png
│           └── predictions_vs_actuals.csv
├── 04. Report-Content
│   ├── Graphs
│   │   ├── no2_diagnostics-no-lag.png
│   │   ├── no2_diagnostics.png
│   │   ├── o3_diagnostics-no-lag.png
│   │   ├── o3_diagnostics.png
│   │   ├── pm10_diagnostics-no-lag.png
│   │   └── pm10_diagnostics.png
│   └── Tables
│       ├── Outcome-Summary-Statistics.Rmd
│       └── Outcome-Summary-Statistics.csv
├── README.md
└── XCroissants-Predicting-Air-Quality.Rproj

```

# Sub-Directory Descriptions and Objectives

## 00. Data csvs

Data storage, some of the files are raw data sets, some are processed data sets,
some are intermediary data sets that are saved between stages of data processing.
There is no code in this sub-directory

## 01. Code Data Cleaning

This sub-directory contains all of the data pre-processing including sourcing 
formatting individual data sets, merging the data sets, feature engineering,
and imputation. Scripts in 01.A to 01.F source the individual data sets which
eventually are all compiled into the final data set used for prediction. Files
01.A through 01.E clean data that are time invariant (i.e. there is one
observation per commune), and 01.F pulls historic meteorologic data from an API,
which gives the data that varies on a day by day basis.

02.A Merging-Data-Sets combines all of the outcome variables and feature data into
one unified data set, and performs some initial data cleaning. More detail is given
in the final report.

 02.B Conducts the feature engineering. Generally speaking, time-invariant
 variables are transformed via Yeo-Johnson transformation, then standardized mean
 0 and standard deviation 1. Time varying variables (the weather data) are smoothed
 using Holt-Winters Exponential Smoothly. 03.A Imputes missing values via 
 KNN imputation. Both are discussed in further detail in the report.
 
 ## 02. Data-Exploration
 
 This visualizes the feature space via distribution and something that 
 approximates a scatterplot to visualize the relationship between individual
 features and the outcomes variables.
 
 ## 03. Prediction
 
 ### 00. Train-Test-Communes
 
 Given the temporal nature of the data, care needed to be taken in how the train 
 and test splits were chosen to avoid data leakage. The decision was made to sample
 individual communes rather than individual observations or sampling periods of time.
 A more detailed explanation is given in the report of the reasoning behind this
 choice, and this script samples the communes with set randomization for
 reproducibility.
 
 ### LightGBM-Predictions
 
 Contains all of the code, outputs, and diagnostics for the LightGBM models. 
 The process for training the models and the results of the evaluation are
 discussed at length in the report
 
 ### Elastic-Net-Prediction
 
 ### OLS-Predictions
 
 ## 04. Report-Content
 
 No code in this sub-directory. Contains some graphs and tables that are included
 in the report.
 
 ## 05. Final-Report
 
 The final PDF report written detailing the project, the decisions made, the
 modeling, and the performance of the models on unseen test data. This is the final
 deliverable submitted for this assignment.
 