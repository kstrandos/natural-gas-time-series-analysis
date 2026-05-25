# ============================================================
# 04_Modelling.py
# Modelling Page - Natural Gas Futures Streamlit App
# DAT803 Exam Project
# ============================================================

import sys
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np


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
    page_title="Modelling - Natural Gas Futures",
    page_icon="🧮",
    layout="wide"
)


# ============================================================
# 3. LOAD DATA
# ============================================================

@st.cache_data
def load_grid_results():
    """
    Load the newest ARIMA/SARIMA grid search result file.
    """

    grid_results_df, grid_results_file = ngu.load_latest_grid_search_results(DATA_DIR)

    return grid_results_df, grid_results_file


try:
    grid_results_df, grid_results_file = load_grid_results()
except Exception as e:
    st.error(f"Could not load grid search results: {e}")
    st.stop()


if grid_results_df is None or grid_results_file is None:
    st.warning("No grid search result file was found.")
    st.stop()


# ============================================================
# 4. HELPER FUNCTIONS
# ============================================================

def find_first_existing_column(df: pd.DataFrame, candidates: list[str]):
    """
    Return the first column name that exists in df.
    """

    for col in candidates:
        if col in df.columns:
            return col

    return None


def ordered_existing_values(values, preferred_order):
    """
    Return values in preferred order first, then any remaining values alphabetically.
    """

    values = [str(v) for v in values if pd.notna(v)]

    ordered = [value for value in preferred_order if value in values]
    remaining = sorted([value for value in values if value not in ordered])

    return ordered + remaining


def format_metric(value, decimals=6):
    """
    Format numeric metric values safely.
    """

    if isinstance(value, (int, float, np.number)):
        return f"{value:.{decimals}f}"

    return str(value)


def get_display_columns(df: pd.DataFrame, required_cols: list[str], metric_cols: list[str]):
    """
    Select a compact set of columns for presentation.
    """

    display_cols = []

    for col in required_cols + metric_cols:
        if col is not None and col in df.columns and col not in display_cols:
            display_cols.append(col)

    return display_cols


# ============================================================
# 5. DETECT IMPORTANT COLUMNS
# ============================================================

target_col = find_first_existing_column(
    grid_results_df,
    ["target", "target_col", "target_series", "series", "Target"]
)

regime_col = find_first_existing_column(
    grid_results_df,
    ["regime", "period", "segment", "dataset", "Regime", "Period"]
)

order_col = find_first_existing_column(
    grid_results_df,
    ["order", "Order", "arima_order"]
)

seasonal_order_col = find_first_existing_column(
    grid_results_df,
    ["seasonal_order", "Seasonal Order", "sarima_order"]
)

metric_candidates = [
    "rmse", "RMSE",
    "mae", "MAE",
    "mape", "MAPE",
    "mse", "MSE",
    "aic", "AIC",
    "bic", "BIC"
]

available_metric_cols = [
    col for col in metric_candidates
    if col in grid_results_df.columns
]

if target_col is None:
    st.error("Could not detect target column in grid search results.")
    st.write("Available columns:", grid_results_df.columns.tolist())
    st.stop()

if regime_col is None:
    st.error("Could not detect regime/period column in grid search results.")
    st.write("Available columns:", grid_results_df.columns.tolist())
    st.stop()

if order_col is None:
    st.error("Could not detect ARIMA order column in grid search results.")
    st.write("Available columns:", grid_results_df.columns.tolist())
    st.stop()

if not available_metric_cols:
    st.error("Could not detect any known metric columns in grid search results.")
    st.write("Available columns:", grid_results_df.columns.tolist())
    st.stop()


# ============================================================
# 6. PAGE TITLE
# ============================================================

st.title("Modelling")
st.caption("ARIMA/SARIMA Grid Search Results")

st.markdown(
    """
    This page presents the saved ARIMA/SARIMA grid search results.

    The full grid search was performed in the notebook.  
    Streamlit is used here only to present and compare the stored results.
    """
)


# ============================================================
# 7. MODELLING APPROACH
# ============================================================

st.subheader("Modelling Approach")

col1, col2 = st.columns(2)

with col1:
    st.markdown(
        """
        ### Why ARIMA/SARIMA?

        ARIMA/SARIMA models provide a classical time series baseline.

        They are useful for testing whether selected transformed NG series contain
        autocorrelation structure that can be modelled systematically.
        """
    )

with col2:
    st.markdown(
        """
        ### Modelling targets

        The modelling focuses on transformed series rather than raw price level:

        - `Close_diff_1`
        - `Log_returns`
        - `Std_diff_1`

        This follows from the stationarity and ACF/PACF analysis.
        """
    )


st.markdown(
    """
    The workflow is:

    ```text
    Select stationary/transformed target
    → split into train/test period
    → run grid search in notebook
    → save results to CSV
    → present best candidates in Streamlit
    ```
    """
)


# ============================================================
# 8. GRID SEARCH SUMMARY
# ============================================================

st.subheader("Grid Search Summary")

target_values = ordered_existing_values(
    grid_results_df[target_col].dropna().unique(),
    ["Close_diff_1", "Log_returns", "Std_diff_1"]
)

regime_values = ordered_existing_values(
    grid_results_df[regime_col].dropna().unique(),
    ["Complete", "High volatility", "Low volatility"]
)

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Tested models", f"{len(grid_results_df):,}")

with col2:
    st.metric("Target series", len(target_values))

with col3:
    st.metric("Market periods", len(regime_values))


with st.expander("Show raw grid search table"):
    st.dataframe(
        grid_results_df,
        use_container_width=True
    )


# ============================================================
# 9. MODEL SELECTION CONTROLS
# ============================================================

st.sidebar.subheader("Model Selection")

selected_target = st.sidebar.selectbox(
    "Target series",
    target_values,
    index=0
)

selected_regime = st.sidebar.selectbox(
    "Market period",
    regime_values,
    index=0
)

preferred_metric_order = [
    metric for metric in [
        "rmse", "RMSE",
        "mae", "MAE",
        "aic", "AIC",
        "bic", "BIC",
        "mape", "MAPE",
        "mse", "MSE"
    ]
    if metric in available_metric_cols
]

selected_metric = st.sidebar.selectbox(
    "Ranking metric",
    preferred_metric_order,
    index=0
)


# ============================================================
# 10. FILTER AND SORT RESULTS
# ============================================================

filtered_df = grid_results_df[
    (grid_results_df[target_col].astype(str) == selected_target) &
    (grid_results_df[regime_col].astype(str) == selected_regime)
].copy()

if filtered_df.empty:
    st.warning("No grid search rows match the selected target and period.")
    st.stop()

# All supported criteria here are lower-is-better
filtered_df = filtered_df.sort_values(
    by=selected_metric,
    ascending=True
)

best_row = filtered_df.iloc[0]


# ============================================================
# 11. BEST MODEL SUMMARY
# ============================================================

st.subheader("Best Candidate")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Target", selected_target)

with col2:
    st.metric("Period", selected_regime)

with col3:
    st.metric("Order", str(best_row[order_col]))

with col4:
    if seasonal_order_col is not None:
        st.metric("Seasonal Order", str(best_row[seasonal_order_col]))
    else:
        st.metric("Seasonal Order", "N/A")


metric_cols_for_display = [
    col for col in ["rmse", "RMSE", "mae", "MAE", "mape", "MAPE", "mse", "MSE", "aic", "AIC", "bic", "BIC"]
    if col in filtered_df.columns
]

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        selected_metric,
        format_metric(best_row[selected_metric])
    )

with col2:
    aic_col = "aic" if "aic" in filtered_df.columns else "AIC" if "AIC" in filtered_df.columns else None
    if aic_col:
        st.metric("AIC", format_metric(best_row[aic_col], decimals=2))

with col3:
    bic_col = "bic" if "bic" in filtered_df.columns else "BIC" if "BIC" in filtered_df.columns else None
    if bic_col:
        st.metric("BIC", format_metric(best_row[bic_col], decimals=2))

with col4:
    mae_col = "mae" if "mae" in filtered_df.columns else "MAE" if "MAE" in filtered_df.columns else None
    if mae_col:
        st.metric("MAE", format_metric(best_row[mae_col]))


# ============================================================
# 12. TOP MODEL CANDIDATES
# ============================================================

st.subheader("Top 3 Model Candidates")

display_cols = get_display_columns(
    filtered_df,
    required_cols=[
        target_col,
        regime_col,
        order_col,
        seasonal_order_col
    ],
    metric_cols=metric_cols_for_display
)

top_n = 3

st.dataframe(
    filtered_df[display_cols].head(top_n),
    use_container_width=True,
    hide_index=True
)


# ============================================================
# 13. INTERPRETATION
# ============================================================

st.subheader("Interpretation")

st.markdown(
    """
    The table shows the best model candidates for the selected target series and market period.

    The selected ranking metric is used to sort the grid search results.  
    For the metrics used here, lower values are better.

    Model selection should not be based on one number alone.  
    The selected candidate should also be interpreted together with:

    - stationarity results
    - ACF/PACF behaviour
    - forecast plots
    - error metrics
    - model simplicity
    """
)

st.info(
    """
    Main message:

    The grid search shows that model selection was systematic rather than arbitrary.
    The next step is to evaluate the selected model through forecast plots and error metrics.
    """
)