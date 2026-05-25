# ============================================================
# 01_Overview.py
# Overview Page - Natural Gas Futures Streamlit App
# DAT803 Exam Project
# ============================================================

import sys
from pathlib import Path
import streamlit as st


# ============================================================
# 1. PATH SETUP
# ============================================================

APP_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = APP_DIR / "data"

if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import src.natgas_utils as ngu


# ============================================================
# 2. PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Overview - Natural Gas Futures",
    page_icon="🏠",
    layout="wide"
)


# ============================================================
# 3. LOAD DATA
# ============================================================

@st.cache_data
def load_data():
    ng_df, es_df, ng_title, es_title = ngu.load_project_datasets(DATA_DIR)
    grid_results_df, grid_results_file = ngu.load_latest_grid_search_results(DATA_DIR)

    return ng_df, es_df, ng_title, es_title, grid_results_df, grid_results_file


try:
    ng_df, es_df, ng_title, es_title, grid_results_df, grid_results_file = load_data()
except Exception as e:
    st.error(f"Could not load project data: {e}")
    st.stop()


# ============================================================
# 4. TITLE
# ============================================================

st.title("Project Overview")
st.caption("Natural Gas Futures - DAT803 Exam Project")


# ============================================================
# 5. SHORT INTRODUCTION
# ============================================================

st.markdown(
    """
    This project analyses **Natural Gas futures** as a financial time series.

    The goal is to build a complete data engineering and time series pipeline from raw CSV files to cleaning, feature engineering, EDA, stationarity testing, modelling, forecasting and evaluation.
    """
)


# ============================================================
# 6. PROJECT SCOPE
# ============================================================

st.subheader("Project Scope")

col1, col2 = st.columns(2)

with col1:
    st.markdown(
        f"""
        ### Main dataset: {ng_title}

        {ng_title} are the main dataset.

        Used for:

        - cleaning
        - feature engineering
        - EDA
        - stationarity analysis
        - modelling
        - prediction and evaluation
        """
    )

with col2:
    st.markdown(
        f"""
        ### Supporting dataset: {es_title}

        {es_title} are used only as market context.

        Used for:

        - comparison
        - static correlation
        - rolling correlation
        - broader market reference

        **ES is not used as an input variable in the prediction model.**
        """
    )


# ============================================================
# 7. PIPELINE
# ============================================================

st.subheader("Pipeline")

st.markdown(
    """
    ```text
    Raw financial CSV data
    → cleaning and standardization
    → feature engineering
    → exploratory data analysis
    → stationarity analysis
    → ARIMA/SARIMA grid search
    → forecasting
    → evaluation
    → interpretation
    ```
    """
)


# ============================================================
# 8. DATASET SUMMARY
# ============================================================

st.subheader("Dataset Summary")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("NG rows", f"{len(ng_df):,}")

with col2:
    st.metric("NG columns", len(ng_df.columns))

with col3:
    st.metric("Start date", str(ng_df.index.min().date()))

with col4:
    st.metric("End date", str(ng_df.index.max().date()))


col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("ES rows", f"{len(es_df):,}")

with col2:
    if grid_results_df is not None:
        st.metric("Grid search rows", f"{len(grid_results_df):,}")
    else:
        st.metric("Grid search rows", "N/A")

with col3:
    pass

with col4:
    pass


# ============================================================
# 9. DATA PREVIEW
# ============================================================

st.subheader("Cleaned NG Data Preview")

preview_cols = [
    col for col in ["Open", "High", "Low", "Close", "Volume", "Change %"]
    if col in ng_df.columns
]

st.dataframe(ng_df[preview_cols].head())


with st.expander("Show engineered features"):
    feature_cols = [
        col for col in [
            "Log_returns",
            "Close_diff_1",
            "Rolling_mean_14",
            "Rolling_std_14",
            "Lag1",
            "Lag2",
            "Lag5",
            "Lag10",
            "Std_diff_1",
            "VWAP",
            "VWAP_diff",
            "VWAP_regime"
        ]
        if col in ng_df.columns
    ]

    st.dataframe(ng_df[feature_cols].head(20))


# ============================================================
# 10. KEY MESSAGE
# ============================================================

st.info(
    """
    Key message:

    This project is mainly about building a clean, reproducible time series pipeline.
    The model is important, but the full process from raw data to evaluated forecast is the main result.
    """
)