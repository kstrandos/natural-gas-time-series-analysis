# ============================================================
# 05_Results.py
# Forecast Results Page - Natural Gas Futures Streamlit App
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
    page_title="Results - Natural Gas Futures",
    page_icon="📈",
    layout="wide"
)


# ============================================================
# 3. LOAD DATA
# ============================================================

@st.cache_data
def load_data():
    """
    Load project datasets and latest grid search results.
    """

    ng_df, _, _, _ = ngu.load_project_datasets(DATA_DIR)
    grid_results_df, _ = ngu.load_latest_grid_search_results(DATA_DIR)

    return ng_df, grid_results_df


@st.cache_data(show_spinner=False)
def cached_forecast_results(ng_df, grid_results_df, selected_target):
    return ngu.get_reintegrated_forecasts_for_target(
        ng_df=ng_df,
        all_results_df=grid_results_df,
        target_col=selected_target
    )
    

try:
    ng_df, grid_results_df = load_data()
except Exception as e:
    st.error(f"Could not load data: {e}")
    st.stop()

if grid_results_df is None:
    st.error("No grid search results found.")
    st.stop()


# ============================================================
# 4. PAGE TITLE
# ============================================================

st.title("Forecast Results")
st.caption("Reintegrated Forecasts Across Market Periods")

st.markdown(
    """
    This page presents the final forecast results.

    For each selected target series, the best grid search model is fitted and evaluated across:

    - Complete dataset
    - High volatility period
    - Low volatility period

    The forecasts are reintegrated back to the original scale to make the results easier to interpret.
    """
)


# ============================================================
# 5. SIDEBAR CONTROLS
# ============================================================

st.sidebar.subheader("Forecast Selection")

target_options = [
    target for target in ["Close_diff_1", "Log_returns", "Std_diff_1"]
    if target in grid_results_df["target"].astype(str).unique()
]

selected_target = st.sidebar.selectbox(
    "Target series",
    target_options,
    index=0
)

forecast_view = st.sidebar.radio(
    "Forecast view",
    [
        "Full train/test view",
        "Test period zoom"
    ],
    index=0
)


# ============================================================
# 6. SELECTED SETUP
# ============================================================

st.subheader("Selected Forecast Setup")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Target series", selected_target)

with col2:
    st.metric("Forecast view", forecast_view)

with col3:
    st.metric("Market periods", "Complete / High / Low")


st.caption("For each market period, the best model is selected from the saved grid search results by RMSE, then AIC.")

# ============================================================
# 7. RUN FORECAST
# ============================================================


with st.spinner("Fitting selected models and generating reintegrated forecast plots..."):

    try:
        results, summary_df = cached_forecast_results(
            ng_df,
            grid_results_df,
            selected_target
        )

    except Exception as e:
        st.error(f"Could not generate forecast results: {e}")
        st.stop()


# ============================================================
# 8. FORECAST PLOT
# ============================================================

st.subheader("Forecast Plot")

zoom = forecast_view == "Test period zoom"

fig = ngu.plot_reintegrated_forecasts_grid(
    results=results,
    target_col=selected_target,
    zoom=zoom
)

st.pyplot(fig)

if selected_target in ["Close_diff_1", "Log_returns"]:
    st.caption(
        """
        Note: The `Close_diff_1` and `Log_returns` charts may look very similar because both forecasts are reintegrated back to the original close-price scale.

        The grey train line and black actual test line therefore show the same historical close prices in both views. The only model-driven difference is the red forecast line.

        Since both targets describe short-term price changes, and log returns are often close to ordinary percentage changes for small daily movements, the reintegrated forecasts may also look very similar when the ARIMA/SARIMA model finds only weak structure.
        """
    )

# ============================================================
# 9. RESULT TABLE
# ============================================================

st.subheader("Evaluation Summary")

summary_display = summary_df.copy()

summary_display = summary_display.rename(columns={
    "regime": "Period",
    "target": "Target",
    "order": "Order",
    "seasonal_order": "Seasonal order",
    "reintegrated_mse": "MSE",
    "reintegrated_rmse": "RMSE",
    "reintegrated_mae": "MAE"
})

for col in ["MSE", "RMSE", "MAE"]:
    if col in summary_display.columns:
        summary_display[col] = summary_display[col].round(6)

for col in ["Order", "Seasonal order"]:
    if col in summary_display.columns:
        summary_display[col] = summary_display[col].astype(str)
        
st.dataframe(
    summary_display,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# 10. INTERPRETATION
# ============================================================

st.subheader("Interpretation")

if selected_target == "Close_diff_1":

    st.markdown(
        """
        The model forecasts first differences in the close price and then reintegrates the forecast back to the close-price level.

        This makes the forecast easier to inspect visually, but it also means that errors can accumulate over the forecast horizon.

        A relatively flat forecast line suggests that the model does not find enough stable structure in the differenced price series to project a strong price path forward.
        """
    )

elif selected_target == "Log_returns":

    st.markdown(
        """
        The model forecasts daily log returns and then reintegrates them back to the close-price level using cumulative log returns.

        Log returns are more stationary than raw prices, but they are also noisy and often contain limited linear predictability.

        If the forecast appears relatively flat after reintegration, this suggests that the model mainly estimates an average expected return path rather than capturing short-term price movements.
        """
    )

elif selected_target == "Std_diff_1":

    st.markdown(
        """
        The model forecasts changes in rolling volatility and then reintegrates the forecast back to the rolling volatility level.

        The results suggest that the model can estimate a baseline volatility level, but it does not capture sudden volatility spikes well.

        The relatively low error values should be interpreted carefully because rolling volatility is measured on a much smaller scale than the close price.
        """
    )


st.info(
    """
    Main message:

    The forecasts provide a systematic ARIMA/SARIMA baseline, but they do not prove strong predictability.
    The main result is the complete workflow from grid search to reintegrated forecast evaluation.
    """
)