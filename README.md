# DAT803 Exam Project – Natural Gas Futures Time Series Analysis
Time Series Analysis Performed by Kjetil Strandos – May 2026

This repository contains my exam project for **DAT803 Data Engineering: From Theory to Practice**.

The project analyzes **Natural Gas futures** as a financial time series. The main goal is to build a complete and reproducible data engineering and time series pipeline: from raw CSV data to cleaning, feature engineering, exploratory analysis, stationarity testing, ARIMA/SARIMA modelling, forecast evaluation and presentation in a Streamlit app.

\---

## Deliverable

The final notebook export/report is submitted separately through Wiseflow as part of the DAT803 exam delivery.

This repository contains the notebook, Streamlit app and supporting project files.

\---

## Project Overview

The project focuses on Natural Gas futures as the main dataset.

A secondary S\&P 500 futures dataset is used only as supporting market context in the exploratory analysis. It is used for comparison, static correlation and rolling correlation, but it is **not used as an input variable in the forecasting model**.

The project is intentionally built around a classical and explainable time series workflow. ARIMA/SARIMA models are used as systematic baseline models, not as trading-ready forecasting systems.

\---

## Main Goals

The main goals of the project are to:

* Load and clean financial time series data from CSV files
* Standardize and prepare the data for analysis
* Engineer relevant time series features
* Explore price behaviour, returns, volatility and market context
* Test stationarity using ADF, ACF and PACF
* Run systematic ARIMA/SARIMA grid search
* Forecast transformed target series
* Reintegrate forecasts back to interpretable scale
* Evaluate model performance using error metrics
* Present the full workflow in a Streamlit app

\---

## Datasets

### Main dataset

* **Natural Gas Futures**
* Used for cleaning, feature engineering, EDA, stationarity analysis, modelling, forecasting and evaluation

### Supporting dataset

* **S\&P 500 Futures**
* Used only for exploratory comparison and broader market context
* Not used as a forecasting input variable

### Market events

A separate market event dataset is used for contextual annotations in EDA plots. These events are used only for interpretation and visualization, not as model features.

\---

## Project Pipeline

Raw CSV data  
→ Data cleaning and standardization  
→ Feature engineering  
→ Exploratory Data Analysis  
→ Stationarity analysis  
→ ARIMA/SARIMA grid search  
→ Forecasting  
→ Forecast reintegration  
→ Evaluation  
→ Streamlit presentation app

\---

## Feature Engineering

The project creates several time series features, including:

* Log returns
* First difference of close price
* Rolling mean
* Rolling standard deviation
* Lag features
* First difference of rolling volatility
* VWAP
* VWAP distance
* Calendar-based features

The main modelling targets are:

* `Close\\\_diff\\\_1`
* `Log\\\_returns`
* `Std\\\_diff\\\_1`

These transformed series are used instead of modelling the raw close price directly.

\---

## Exploratory Data Analysis

The EDA investigates:

* Natural Gas price development
* Log returns
* Volatility clustering
* Rolling volatility
* VWAP behaviour
* Calendar return statistics
* Comparison with S\&P 500 futures
* Static and rolling correlation between NG and ES
* Market event context

The purpose of the EDA is to understand the structure of the time series before modelling.

\---

## Stationarity Analysis

The project uses:

* ADF test
* ACF plots
* PACF plots

The raw close price shows strong persistence and is not ideal as a direct modelling target. Transformed series such as log returns, first differences and volatility changes are more suitable for ARIMA/SARIMA modelling.

\---

## Modelling

The modelling section uses ARIMA/SARIMA as classical time series baseline models.

The grid search was performed in the notebook and saved to CSV. The Streamlit app loads the saved grid search results and presents the best model candidates by selected metrics.

Model selection is based on:

* RMSE
* MAE
* AIC
* BIC
* Model simplicity
* Stationarity and ACF/PACF interpretation

\---

## Forecasting and Evaluation

Forecasts are generated for three market periods:

* Complete dataset
* High volatility period
* Low volatility period

The forecasts are reintegrated back to the original scale to make them easier to interpret.

For example:

* `Close\\\_diff\\\_1` is reintegrated back to close price
* `Log\\\_returns` is reintegrated back to close price
* `Std\\\_diff\\\_1` is reintegrated back to rolling volatility

Evaluation is based on:

* MSE
* RMSE
* MAE

The results show that ARIMA/SARIMA provides a systematic baseline, but forecast performance is limited on noisy financial market data.

\---

## Streamlit App

The project includes a Streamlit app that presents the most important parts of the notebook workflow in an interactive format.

The app contains the following pages:

* Overview
* EDA
* Stationarity
* Modelling
* Results
* Conclusion

The app is meant as a presentation layer for the notebook work. It does not replace the notebook, but makes the main findings and workflow easier to inspect.

\---

## How to Run the App Locally

From the `exam\\\_app` folder, run:

`streamlit run natgas\\\_app.py`

Required Python packages are listed in:

`requirements.txt`

\---

## Project Structure

DAT803-exam/  
├── exam\_app/  
│   ├── data/  
│   │   ├── Raw data/  
│   │   ├── Natural Gas Futures 010400-082319.csv  
│   │   ├── SP500 Futures 010400-010306.csv  
│   │   ├── SP500 Futures 010406-082319.csv  
│   │   ├── arima\_grid\_search\_results\_7ee13318.csv  
│    │   └── market\_events\_wikipedia.csv  
│   ├── pages/  
│   │   ├── 01\_Overview.py  
│   │   ├── 02\_EDA.py  
│   │   ├── 03\_Stationarity.py  
│   │   ├── 04\_Modelling.py  
│   │   ├── 05\_Results.py  
│    │    └── 06\_Conclusion.py  
│   ├── src/  
│   │   ├── **init**.py  
│    │    └── natgas\_utils.py  
│   ├── natgas\_app.py  
│    └── requirements.txt  
├── finance\_time\_series.ipynb  
├── notebook\_utils.py  
├── README.md  
└── .gitignore

\---

## Main Findings

The project shows that:

* The raw Natural Gas close price changes level over time and is not ideal as a direct ARIMA/SARIMA target
* Log returns are more stable around zero, but still noisy
* Rolling volatility shows clustering and regime-like behaviour
* Natural Gas and S\&P 500 futures do not have a stable relationship over time
* S\&P 500 futures provide useful market context, but are not used as model input
* ARIMA/SARIMA models provide a systematic baseline, but produce limited forecast performance on noisy financial data
* Reintegrated forecasts are often relatively flat, suggesting limited stable structure for the model to project forward

\---

## Limitations

Important limitations:

* The project uses classical univariate ARIMA/SARIMA models
* External variables such as weather, storage data, macro data and energy fundamentals are not included
* Market events are used only for EDA context, not as model features
* Financial time series are noisy and difficult to forecast
* Forecast errors can accumulate when transformed predictions are reintegrated
* The models should be interpreted as baseline models, not trading-ready systems

\---

## Possible Future Work

Possible extensions include:

* More advanced volatility modelling
* More systematic regime detection
* Walk-forward validation
* External explanatory variables such as weather, storage data or macro data
* Machine learning or deep learning models
* Directional prediction instead of raw price forecasting
* Backtesting whether volatility regimes or correlation shifts contain useful trading or risk-management information

\---

## Final Conclusion

The strongest result of this project is the complete, explainable and reproducible pipeline from raw financial data to modelling and forecast evaluation.

The project shows both the usefulness and the limitations of classical ARIMA/SARIMA models on noisy financial market data.

