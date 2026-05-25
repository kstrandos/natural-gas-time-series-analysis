# ============================================================
# 03_Stationarity.py
# Stationarity Page - Natural Gas Futures Streamlit App
# DAT803 Exam Project
# ============================================================

import sys
from pathlib import Path
import warnings
import streamlit as st
import pandas as pd
from statsmodels.tsa.stattools import adfuller


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
    page_title="Stationarity - Natural Gas Futures",
    page_icon="📉",
    layout="wide"
)


# ============================================================
# 3. LOAD DATA
# ============================================================

@st.cache_data
def load_data():
    ng_df, es_df, ng_title, es_title = ngu.load_project_datasets(DATA_DIR)
    return ng_df, es_df, ng_title, es_title


try:
    ng_df, es_df, ng_title, es_title = load_data()
except Exception as e:
    st.error(f"Could not load project datasets: {e}")
    st.stop()


# ============================================================
# 4. HELPER FUNCTIONS
# ============================================================

def run_adf_test(series: pd.Series) -> dict:
    """
    Run Augmented Dickey-Fuller test and return key results.
    """

    clean_series = series.dropna()

    if len(clean_series) < 20:
        raise ValueError("Series is too short for ADF test.")

    adf_result = adfuller(clean_series)

    return {
        "ADF Statistic": adf_result[0],
        "p-value": adf_result[1],
        "Used Lags": adf_result[2],
        "Observations": adf_result[3],
        "1% Critical Value": adf_result[4]["1%"],
        "5% Critical Value": adf_result[4]["5%"],
        "10% Critical Value": adf_result[4]["10%"]
    }


def create_adf_summary(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """
    Create a compact ADF summary table for selected series.

    The table is kept presentation-friendly:
    - no Used Lags
    - no Observations
    - no overly strong "Stationary" label

    ADF result is stated as whether the unit-root null hypothesis is rejected.
    """

    rows = []

    for col in columns:
        series = df[col].dropna()

        if len(series) < 20:
            continue

        result = run_adf_test(series)

        p_value = result["p-value"]

        rows.append({
            "Series": col,
            "ADF Statistic": result["ADF Statistic"],
            "p-value": p_value,
            "ADF result": (
                "Reject unit root"
                if p_value < 0.05
                else "Cannot reject unit root"
            )
        })

    return pd.DataFrame(rows)


# ============================================================
# 5. PAGE TITLE
# ============================================================

st.title("Stationarity Analysis")
st.caption("ADF Test, ACF and PACF")

st.markdown(
    """
    This page checks whether selected Natural Gas series are stationary.

    Stationarity matters because ARIMA/SARIMA models are designed for time series
    where the statistical properties are relatively stable over time.
    """
)


# ============================================================
# 6. CONTROLS
# ============================================================

st.sidebar.subheader("Stationarity Controls")

selected_period = st.sidebar.selectbox(
    "Select predefined period",
    [
        "Complete",
        "High volatility",
        "Low volatility"
    ],
    index=0
)

selected_group = st.sidebar.selectbox(
    "Select series for comparison",
    [
        "Before differencing",
        "Model targets"
    ],
    index=0
)

selected_lags = st.sidebar.slider(
    "ACF/PACF lags",
    min_value=10,
    max_value=40,
    value=20,
    step=5
)

use_custom_date_range = st.sidebar.checkbox(
    "Use custom date range",
    value=False
)


# ============================================================
# 7. FILTER DATA
# ============================================================

filtered_df = ngu.filter_by_period(ng_df, selected_period)

if use_custom_date_range:
    min_date = ng_df.index.min().date()
    max_date = ng_df.index.max().date()

    selected_date_range = st.sidebar.slider(
        "Select analysis period",
        min_value=min_date,
        max_value=max_date,
        value=(
            filtered_df.index.min().date(),
            filtered_df.index.max().date()
        )
    )

    start_date, end_date = selected_date_range

    filtered_df = ng_df.loc[
        (ng_df.index.date >= start_date) &
        (ng_df.index.date <= end_date)
    ].copy()


# ============================================================
# 8. SELECT SERIES COMPARISON
# ============================================================

if selected_group == "Model targets":
    selected_columns = [
        "Close_diff_1",
        "Log_returns",
        "Std_diff_1"
    ]

    group_description = """
    This comparison shows the transformed series used as modelling candidates.

    - `Close_diff_1`: first difference of close price
    - `Log_returns`: daily log returns
    - `Std_diff_1`: first difference of rolling volatility
    """

else:
    selected_columns = [
        "Close",
        "Log_returns",
        "Rolling_std_14"
    ]

    group_description = """
    This comparison shows the raw price level before differencing, together with two transformed reference series.
    
    - `Close`: raw price level
    - `Log_returns`: daily log returns
    - `Rolling_std_14`: rolling volatility level
    """


selected_columns = [
    col for col in selected_columns
    if col in filtered_df.columns
]

if not selected_columns:
    st.error("None of the selected columns exist in the dataframe.")
    st.stop()


# ============================================================
# 9. SELECTED PERIOD SUMMARY
# ============================================================

st.subheader("Selected Period Summary")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Rows", f"{len(filtered_df):,}")

with col2:
    st.metric("Start", str(filtered_df.index.min().date()))

with col3:
    st.metric("End", str(filtered_df.index.max().date()))

with col4:
    st.metric("Series comparison", selected_group)


st.markdown(group_description)


# ============================================================
# 10. ADF SUMMARY
# ============================================================

st.subheader("ADF Test Summary")

adf_summary = create_adf_summary(
    filtered_df,
    selected_columns
)

if adf_summary.empty:
    st.warning("ADF summary could not be created for the selected series.")
else:
    st.dataframe(
        adf_summary.round({
            "ADF Statistic": 4,
            "p-value": 6
        }),
        use_container_width=True,
        hide_index=True
    )

st.markdown(
    """
    **Interpretation**

    The ADF test is a formal test for a unit root.

    A p-value below 0.05 means that the test rejects the unit-root null hypothesis.
    This is useful, but it does not automatically mean that the series is easy to forecast or ideal for direct modelling.

    The ADF result should therefore be interpreted together with the ACF and PACF plots.
    """
)


# ============================================================
# 11. PACF / ACF GRID
# ============================================================

st.subheader("PACF and ACF Comparison")

warnings.simplefilter("ignore")

min_length = min(
    filtered_df[col].dropna().shape[0]
    for col in selected_columns
)

if min_length < selected_lags + 5:
    st.warning("The selected period has too few observations for the selected number of lags.")
    st.stop()


fig = ngu.plot_acf_pacf_grid(
    filtered_df,
    columns=selected_columns,
    lags=selected_lags,
    title=f"PACF and ACF Comparison - {selected_group}"
)

st.pyplot(fig)


# ============================================================
# 12. FINAL INTERPRETATION
# ============================================================

st.subheader("Interpretation Guide")

if selected_group == "Model targets":

    st.markdown(
        """
        The model target group focuses on transformed series used as modelling candidates.

        Compared with the raw close price, these transformed series show much weaker and more short-lived autocorrelation. This supports the idea that they are more suitable for ARIMA/SARIMA modelling than the original price level.

        In the ACF and PACF plots, most lag values are close to zero, especially for `Log_returns` and `Close_diff_1`. This suggests limited linear predictability, which is common in financial return data.

        `Std_diff_1` shows some stronger isolated lag effects. This may indicate that volatility changes contain more structure than price returns, but it should still be interpreted carefully.

        The ACF and PACF plots are used as support for model selection, but the final model choice is based on grid search and forecast evaluation.
        """
    )

elif selected_group == "Before differencing":

    st.markdown(
        """
        The before-differencing group helps explain why transformation is useful.

        - The raw close price often shows strong persistence and long-lasting autocorrelation. This is visible when the ACF remains high across many lags.

        - Log returns are usually more centered around zero and show much weaker autocorrelation.

        - Rolling volatility can still show persistence and clustering, which suggests that volatility behaves differently from returns and may contain regime-like structure.

        This supports the decision to avoid modelling the raw close price directly and instead focus on transformed modelling targets.
        """
    )

st.info(
    """
    ADF, ACF and PACF should be interpreted together.

    - ADF gives a formal unit-root test.
    - ACF shows autocorrelation across lags.
    - PACF helps identify direct lag relationships.

    These tools support model selection, but they do not guarantee strong forecast performance.
    """
)