# ============================================================
# natgas_app.py
# Main Landing Page - Natural Gas Futures Streamlit App
# DAT803 Exam Project
# ============================================================

import sys
from pathlib import Path

import streamlit as st


# ============================================================
# 1. PATH SETUP
# ============================================================

APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"

if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import src.natgas_utils as ngu


# ============================================================
# 2. PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="DAT803 - Natural Gas Futures",
    page_icon="📈",
    layout="wide"
)


# ============================================================
# 3. LOAD BASIC STATUS
# ============================================================

@st.cache_data
def load_app_status():
    """
    Load only enough data to show that the app is connected to the project files.
    """

    ng_df, es_df, ng_title, es_title = ngu.load_project_datasets(DATA_DIR)
    grid_results_df, grid_results_file = ngu.load_latest_grid_search_results(DATA_DIR)

    return ng_df, es_df, ng_title, es_title, grid_results_df, grid_results_file


try:
    ng_df, es_df, ng_title, es_title, grid_results_df, grid_results_file = load_app_status()
    data_loaded = True

except Exception as e:
    ng_df = None
    es_df = None
    ng_title = "Natural Gas Futures"
    es_title = "S&P 500 Futures"
    grid_results_df = None
    grid_results_file = None
    data_loaded = False
    load_error = e


# ============================================================
# 4. TITLE
# ============================================================

st.title("Natural Gas Futures - Time Series Analysis")
st.caption("DAT803 Data Engineering: From Theory to Practice")
st.caption("Project by Kjetil Strandos")


st.markdown(
    """
    This Streamlit app presents the exam project for **DAT803 Data Engineering**.

    The project analyses **Natural Gas futures** as the main financial time series.
    **S&P 500 futures** are used only as a supporting dataset for exploratory comparison.

    The app is designed as a presentation layer for the notebook work.
    It shows the most important parts of the pipeline in a structured and interactive way.
    """
)


# ============================================================
# 5. PROJECT FLOW
# ============================================================

st.subheader("Project Flow")

st.markdown(
    """
    ```text
    Raw CSV data
    → Data cleaning
    → Feature engineering
    → Exploratory Data Analysis
    → Stationarity Analysis
    → ARIMA/SARIMA Modelling
    → Forecast Evaluation
    → Conclusion
    ```
    """
)


# ============================================================
# 6. PAGE GUIDE
# ============================================================

st.subheader("Project Pages")

col1, col2 = st.columns(2)

with col1:
    st.markdown(
        """
        ### Suggested review order

        1. **Overview**  
           Project scope, datasets and pipeline.

        2. **EDA**  
           Price development, returns, volatility, VWAP, correlation and market context.

        3. **Stationarity**  
           ADF test, ACF and PACF.

        4. **Modelling**  
           ARIMA/SARIMA setup and grid search results.
        """
    )

with col2:
    st.markdown(
        """
        ### Final pages

        5. **Results**  
           Forecast plots and evaluation metrics.

        6. **Conclusion**  
           Main findings, limitations and possible future improvements.
        """
    )


# ============================================================
# 7. DATA STATUS
# ============================================================

st.subheader("Data Status")

if data_loaded:

    col1, col2, col3 = st.columns(3)

    with col1:
        st.success("NG data loaded")
        st.metric("NG rows", f"{len(ng_df):,}")

    with col2:
        st.success("ES data loaded")
        st.metric("ES rows", f"{len(es_df):,}")

    with col3:
        if grid_results_df is not None:
            st.success("Grid search loaded")
            st.metric("Grid rows", f"{len(grid_results_df):,}")

            if grid_results_file is not None:
                st.caption(f"Grid file: `{grid_results_file.name}`")
        else:
            st.warning("Grid search file missing")

else:
    st.error("Project data could not be loaded.")
    st.write(load_error)


# ============================================================
# 8. FINAL NOTE
# ============================================================

st.info(
    """
    Use the pages in the sidebar to move through the project.

    The goal of this app is not to show every line of code, but to explain the most important choices, findings and limitations in the time series pipeline.
    """
)