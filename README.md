Weather Temperature Forecasting System

Status: Deployed
Type: Machine Learning / Time Series
Division: AI / ML Team

📄 Project Information
Field	Details
Project Name	Weather Temperature Forecasting System
Document ID	DOC-ML-2024-001
Prepared By	AI/ML Development Team
Date	31 March 2026
Classification	Internal — Company Confidential


1. Problem Overview

This project implements a complete weather temperature forecasting pipeline using classical time series techniques.

The system is trained on a Kaggle daily weather dataset (2013–2024) containing ~3,500 records of meteorological data.

Objective

To accurately forecast future daily temperatures using historical patterns and statistical modeling.

Model Details
🔹 ARIMA (Univariate Baseline)
Uses only temperature (temp)
Establishes baseline performance
Auto-tuned using AIC-based auto_arima
Includes:
Stationarity testing (ADF Test)
ACF / PACF analysis
Residual diagnostics
🔹 Seasonal Handling
Captures weekly seasonality (m = 7)
Improves prediction accuracy

Dataset Summary
Attribute	Value
Source	Kaggle — Daily Weather Dataset
Date Range	April 2013 – December 2024
Total Records	~3,559
Target Variable	temp (°C)
Train Split	80% (~2,847)
Test Split	20% (~712)

Tech Stack
Category	Tool / Library	Purpose
Language	Python	Core development
ML Library	statsmodels	ARIMA modeling
Auto Tuning	pmdarima	auto_arima
Data	pandas / numpy	Data processing
Visualization	matplotlib / seaborn	Plots & analysis
ML Utilities	scikit-learn	Metrics
Model Saving	joblib	Serialization
Web App	Streamlit	UI
Notebook	Google Colab	Training
Deployment	Streamlit Cloud / Hugging Face	Hosting
Version Control	Git / GitHub	Code management

3. Deployment

The trained model is deployed using a Streamlit web application.

⚙️ Features
Adjustable forecast horizon (7–90 days)
Interactive visualization
95% confidence intervals
Downloadable forecast data (CSV)

Live App

👉 https://arimamodel-eeuqnlkzyv6zjk9j45aeji.streamlit.app/

Model Evaluation

Performance on test dataset (20%):

Metric	Value
MAE	3.45 °C
RMSE	4.59 °C
MAPE	15.65%




