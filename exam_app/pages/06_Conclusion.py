# ============================================================
# 06_Conclusion.py
# Conclusion Page - Natural Gas Futures Streamlit App
# DAT803 Exam Project
# ============================================================

import streamlit as st


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Conclusion - Natural Gas Futures",
    page_icon="✅",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title("Conclusion")
st.caption("DAT803 Exam Project - Natural Gas Futures Time Series Analysis")


# ============================================================
# SHORT SUMMARY
# ============================================================

st.markdown(
    """
    This project demonstrates a complete data engineering and time series analysis pipeline
    using **Natural Gas futures** as the main dataset.

    The main goal was not to prove perfect predictability, but to build a structured and reproducible workflow
    from raw financial data to feature engineering, exploratory analysis, modelling, forecasting and evaluation.
    """
)


# ============================================================
# WHAT WAS DONE
# ============================================================

st.header("What Was Done")

col1, col2 = st.columns(2)

with col1:
    st.markdown(
        """
        ### Data Pipeline

        - Loaded raw financial CSV data
        - Cleaned and standardized NG and ES futures data
        - Converted dates, prices and volume values
        - Created reusable utility functions
        - Built a Streamlit presentation app
        """
    )

with col2:
    st.markdown(
        """
        ### Time Series Analysis

        - Created log returns and differenced series
        - Calculated rolling volatility
        - Created lag features
        - Calculated VWAP and VWAP difference
        - Tested stationarity using ADF, ACF and PACF
        - Ran ARIMA/SARIMA grid search
        - Evaluated reintegrated forecasts
        """
    )


# ============================================================
# MAIN FINDINGS
# ============================================================

st.header("Main Findings")

st.markdown(
    """
    The analysis showed that:

    - The raw Natural Gas close price changes level over time and is not ideal as a direct modelling target.
    - Log returns are more stable around zero, but still very noisy.
    - Rolling volatility shows clear clustering and regime-like behaviour.
    - NG and ES do not have a stable relationship over time.
    - ES is useful as market context, but was not used as an input variable in the forecasting model.
    - ARIMA/SARIMA provides a systematic baseline, but forecast performance is limited on noisy financial data.
    """
)


# ============================================================
# MODEL INTERPRETATION
# ============================================================

st.header("Model Interpretation")

st.markdown(
    """
    The ARIMA/SARIMA models were useful for testing a classical forecasting workflow.

    However, the forecast plots show that the models often produce relatively flat forecasts after reintegration.
    This suggests that the transformed financial series contain limited stable structure for the model to project forward.

    This is especially important for financial data, where noise, shocks and regime changes make forecasting difficult.
    """
)

st.info(
    """
    Weak forecast performance from a simple ARIMA/SARIMA baseline does not mean that financial forecasting is impossible.

    A key takeaway is that raw price prediction is difficult. More realistic forecasting approaches often use transformed targets or richer feature sets, such as returns, volatility measures, regime indicators, directional probabilities, technical indicators, macroeconomic data, fundamental data or textual data.

    This project is intentionally limited to classical univariate time series models as an explainable baseline.
    """
)


# ============================================================
# LIMITATIONS
# ============================================================

st.header("Limitations")

st.warning(
    """
    The models should be interpreted as baseline models, not as trading-ready forecasting systems.
    """
)

st.markdown(
    """
    Important limitations:

    - The project uses classical univariate ARIMA/SARIMA models.
    - External variables such as weather, storage data, macro data and energy fundamentals are not included.
    - Market events are used only as EDA context, not as model features.
    - Financial returns are noisy and difficult to forecast.
    - Forecast errors can accumulate when transformed predictions are reintegrated back to the original scale.
    """
)


# ============================================================
# POSSIBLE FUTURE WORK
# ============================================================

st.header("Possible Future Work")

st.markdown(
    """
    Possible extensions include:

    - More advanced volatility modelling
    - More systematic regime detection
    - Testing external explanatory variables such as weather, storage data, macro data or energy fundamentals
    - Comparing ARIMA/SARIMA with machine learning or deep learning models
    - Testing transformed targets such as direction, risk-adjusted returns or volatility regimes
    - Using walk-forward validation for more realistic out-of-sample testing
    - Backtesting whether volatility regimes, extreme rolling standard deviation values or rolling correlation shifts contain useful trading or risk-management information
    """
)


# ============================================================
# FINAL MESSAGE
# ============================================================

st.success(
    """
    Final conclusion:

    The strongest result of this project is the complete, explainable and reproducible pipeline
    from raw financial data to modelling and forecast evaluation.
    """
)


st.info(
    """
    Main message:

    I built a full time series pipeline, tested classical forecasting models systematically,
    and showed both the usefulness and the limitations of ARIMA/SARIMA on noisy financial market data.
    """
)

