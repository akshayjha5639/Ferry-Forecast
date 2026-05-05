📌 Overview
This project builds a short-term demand forecasting system for Toronto Island Park ferry operations using over a decade of historical ticket transaction data (May 2015 – December 2025). The goal is to shift ferry operations from reactive scheduling to predictive intelligence — enabling operators to anticipate passenger demand 15 minutes to 2 hours in advance.
A fully interactive Streamlit dashboard has been deployed for real-time forecast access.
🔗 Live App: https://ferry-forecast-jebktrxr9arffoh6ltsbiy.streamlit.app/

🎯 Problem Statement
Despite holding 10+ years of high-resolution ticket data, Toronto Island ferry operators have no mechanism to forecast demand ahead of time. This leads to:

Terminal congestion noticed only after it forms
Inefficient ferry dispatch and staffing
Poor passenger experience during peak periods


📊 Dataset
AttributeDetailsPeriodMay 2015 — December 2025Frequency15-minute intervalsRecords~350,000+Features (engineered)35TargetsSales Count, Redemption CountMissing IntervalsNone

🔧 Methodology
Feature Engineering

Lag features — 15 min, 30 min, 1 hr, 2 hr, 24 hr, 48 hr, 7-day lookbacks
Rolling statistics — mean and std over 1hr, 2hr, 24hr windows
Cyclical encodings — sine/cosine for hour, day of week, month
Binary flags — is_weekend, is_peak_hour, quarter

Models Evaluated
TierModelsBaselinesNaïve, Moving Average, Linear RegressionMachine LearningRandom Forest, Gradient Boosting, XGBoostTime-SeriesFacebook Prophet
Train/Test Split

80% training / 20% test — strict temporal split, no random shuffling


📈 Results
RankModelMAERMSENotes🥇Gradient Boosting17.11863.005Best — deployed🥈Random Forest17.34762.952Strong alternative🥉XGBoost17.49063.706Close third4Linear Regression18.04963.311Best baseline5Moving Average19.52169.836—6Naïve Forecast21.45086.374Benchmark❌Prophet67.700116.801Not suitable
Gradient Boosting achieved a 20.2% MAE improvement over the Naïve baseline.

Prophet underperformed significantly due to its smooth additive decomposition being incompatible with the sharp, zero-inflated demand patterns of recreational ferry services.


💻 Streamlit Dashboard Features

🤖 Model selector — Switch between GB, RF, XGBoost
⏱️ Horizon selector — 15 min to 2 hours ahead
🎯 Target selector — Sales or Redemptions
📅 Date picker — Inspect any date in 2015–2025
📊 KPI cards — Daily totals, peak count, peak hour
📈 Forecast chart — Interactive Plotly with confidence bands
🔍 Actuals vs Predicted — Day-level evaluation + residuals
📋 Model comparison — MAE/RMSE bar charts for all models


📁 Repository Structure
ferry-forecast-app/
│
├── app.py                        ← Streamlit dashboard
├── requirements.txt              ← Python dependencies
├── README.md                     ← This file
│
├── models/
│   ├── gb_model.pkl              ← Gradient Boosting (best model)
│   ├── rf_model.pkl              ← Random Forest
│   └── xgb_model.pkl            ← XGBoost
│
├── data/
│   ├── feature_cols.pkl          ← Feature column names
│   └── ferry_data.csv            ← Cleaned dataset
│
├── notebooks/
│   └── ferry_forecasting.ipynb   ← Full analysis notebook (Colab)
│
└── research/
    └── Ferry_Forecast_Research_Paper.docx   ← Full research paper

📄 Research Paper
The complete research paper is available in this repository under /research/Ferry_Forecast_Research_Paper.docx.
Title: Short-Term Demand Forecasting for Toronto Island Ferry Operations — A Comparative Study of Statistical and Machine Learning Approaches
Abstract: This paper presents a comprehensive study of short-term demand forecasting for Toronto Island Park ferry operations using a decade of historical ticket transaction data. Seven forecasting approaches are evaluated across three tiers: naive baselines, machine learning models, and classical time-series models. Results demonstrate that Gradient Boosting achieves the lowest MAE (17.118), outperforming naive forecasting by 20.2% and confirming that supervised ML with engineered lag features is better suited for high-frequency transit demand forecasting than classical statistical decomposition.
📥 https://www.academia.edu/166257135/Ferry_Forecast_Research_Paper
