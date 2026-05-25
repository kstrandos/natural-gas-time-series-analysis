# ============================================================
# 02_EDA.py
# EDA Page - Natural Gas Futures Streamlit App
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
    page_title="EDA - Natural Gas Futures",
    page_icon="📊",
    layout="wide"
)


# ============================================================
# 3. LOAD DATA
# ============================================================

@st.cache_data
def load_data():
    """
    Load NG and ES datasets using the same project utility function
    as the main app.
    """

    ng_df, es_df, ng_title, es_title = ngu.load_project_datasets(DATA_DIR)

    return ng_df, es_df, ng_title, es_title


try:
    ng_df, es_df, ng_title, es_title = load_data()
except Exception as e:
    st.error(f"Could not load project datasets: {e}")
    st.stop()


# ============================================================
# 4. PAGE TITLE
# ============================================================

st.title("Exploratory Data Analysis")
st.caption("Natural Gas Futures - DAT803 Exam Project")

st.markdown(
    """
    This page presents the main exploratory analysis of the Natural Gas futures dataset.

    The goal is to understand price behaviour, returns, volatility, market context and possible regime differences before modelling.
    """
)


# ============================================================
# 5. CONTROLS
# ============================================================

st.sidebar.subheader("EDA Controls")

selected_period = st.sidebar.selectbox(
    "Select predefined period",
    [
        "Complete",
        "High volatility",
        "Low volatility"
    ],
    index=0
)

use_custom_date_range = st.sidebar.checkbox(
    "Use custom date range",
    value=False
)

show_data_table = st.sidebar.checkbox(
    "Show filtered NG data table",
    value=False
)


# ============================================================
# 6. FILTER DATA
# ============================================================

filtered_ng_df = ngu.filter_by_period(ng_df, selected_period)

if use_custom_date_range:
    min_date = ng_df.index.min().date()
    max_date = ng_df.index.max().date()

    selected_date_range = st.sidebar.slider(
        "Select analysis period",
        min_value=min_date,
        max_value=max_date,
        value=(
            filtered_ng_df.index.min().date(),
            filtered_ng_df.index.max().date()
        )
    )

    start_date, end_date = selected_date_range

    filtered_ng_df = ng_df.loc[
        (ng_df.index.date >= start_date) &
        (ng_df.index.date <= end_date)
    ].copy()

if filtered_ng_df.empty:
    st.warning("No data available for the selected period.")
    st.stop()


# ============================================================
# 7. PERIOD SUMMARY
# ============================================================

st.subheader("Selected Period Summary")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric("Rows", f"{len(filtered_ng_df):,}")

with col2:
    st.metric("Start", str(filtered_ng_df.index.min().date()))

with col3:
    st.metric("End", str(filtered_ng_df.index.max().date()))

with col4:
    if "Close" in filtered_ng_df.columns:
        st.metric("Mean Close", f"{filtered_ng_df['Close'].mean():.2f}")

with col5:
    if "Log_returns" in filtered_ng_df.columns:
        st.metric("Return Std", f"{filtered_ng_df['Log_returns'].std():.4f}")


if show_data_table:
    with st.expander("Filtered NG dataset", expanded=True):
        st.dataframe(filtered_ng_df, use_container_width=True)


# ============================================================
# 8. EDA SECTION SELECTOR
# ============================================================

eda_section = st.radio(
    "EDA section",
    [
        "Main Overview",
        "NG vs ES & Market Events",
        "Additional Exploration"
    ],
    horizontal=True,
    label_visibility="collapsed",
    key="eda_section"
)


# ============================================================
# SECTION 1: MAIN OVERVIEW
# ============================================================

if eda_section == "Main Overview":

    st.subheader("Price, Returns, Volatility and Volume")

    st.pyplot(
        ngu.plot_price_returns_volatility(
            filtered_ng_df,
            title=f"{ng_title} - Overview"
        )
    )

    st.markdown(
        """
        **Interpretation**

        The raw close price shows large structural changes over time.
        This suggests that the original price level is not ideal for direct ARIMA modelling.

        Log returns are more centered around zero, but still noisy.

        Rolling standard deviation shows volatility clustering, where high-volatility periods tend to occur in clusters.
        This supports the idea that the dataset contains different volatility regimes.
        """
    )


# ============================================================
# SECTION 2: NG VS ES & MARKET EVENTS
# ============================================================

elif eda_section == "NG vs ES & Market Events":

    st.subheader("NG vs ES with Market Events")

    st.markdown(
        """
        This section compares Natural Gas futures with S&P 500 futures.

        Both price series are normalized to 100 at the start of the overlapping period.
        This makes it easier to compare relative development, since NG and ES trade at very different price levels.

        Market events are shown as contextual annotations only.  
        They are not used as model features.
        """
    )

    if es_df is None:
        st.warning("ES dataset could not be loaded.")

    else:
        filtered_es_df = es_df.loc[
            (es_df.index >= filtered_ng_df.index.min()) &
            (es_df.index <= filtered_ng_df.index.max())
        ].copy()

        if filtered_es_df.empty:
            st.warning("No ES data available for the selected NG period.")

        else:
            try:
                events = ngu.load_market_events(DATA_DIR)

                events_project = ngu.prepare_project_events(
                    events,
                    start_date=filtered_ng_df.index.min(),
                    end_date=filtered_ng_df.index.max()
                )

                fig = ngu.plot_normalized_prices_with_market_events_plotly(
                    ng_df=filtered_ng_df,
                    es_df=filtered_es_df,
                    events_project=events_project
                )
                
                st.plotly_chart(fig, use_container_width=True)

                
                st.markdown(
                    """
                    **Interpretation**

                    The normalized price chart shows that NG and ES do not behave the same way over time.

                    ES provides useful broader market context, but Natural Gas is also affected by commodity-specific factors such as supply, demand, storage, weather and energy market structure.
                    """
                )

                with st.expander("Show market events used in plot"):
                    show_cols = [
                        col for col in [
                            "event_date",
                            "Name",
                            "Short_Name",
                            "Country",
                            "label_offset_days",
                            "Notes"
                        ]
                        if col in events_project.columns
                    ]

                    st.dataframe(events_project[show_cols])

                # --------------------------------------------------------
                # Rolling correlation
                # --------------------------------------------------------

                st.subheader("Rolling Correlation")

                st.markdown(
                    """
                    The rolling correlation shows whether the relationship between NG and ES is stable or changes over time.
                    """
                )

                rolling_window = st.sidebar.slider(
                    "Rolling correlation window",
                    min_value=5,
                    max_value=90,
                    value=60,
                    step=5,
                )

                corr_df = ngu.create_ng_es_correlation_df(
                    ng_df=filtered_ng_df,
                    es_df=filtered_es_df,
                    window=rolling_window
                )

                if len(corr_df.dropna()) < rolling_window:
                    st.info("Not enough overlapping observations to calculate rolling correlation.")

                else:
                    fig_corr = ngu.plot_rolling_correlation_with_market_events(
                        corr_df=corr_df,
                        events_project=events_project,
                        window=rolling_window
                    )

                    st.pyplot(fig_corr)

                    static_corr = corr_df["NG_Log_returns"].corr(
                        corr_df["ES_Log_returns"]
                    )

                    mean_rolling_corr = corr_df[f"Rolling_corr_{rolling_window}"].mean()

                    col1, col2 = st.columns(2)

                    with col1:
                        st.metric(
                            "Static correlation",
                            f"{static_corr:.4f}"
                        )

                    with col2:
                        st.metric(
                            f"Mean {rolling_window}-day rolling correlation",
                            f"{mean_rolling_corr:.4f}"
                        )

                    st.markdown(
                        """
                        **Interpretation**

                        The correlation is not stable over time.

                        This supports the decision to use ES only as an EDA comparison and market-context dataset,
                        not as an input variable in the univariate ARIMA/SARIMA forecasting model.

                        The event lines provide context, not proof of causality.
                        A possible future extension would be to test whether changing correlation regimes contain useful trading or forecasting information.
                        """
                    )

            except Exception as e:
                st.error(f"Could not create NG vs ES market event analysis: {e}")
            

# ============================================================
# SECTION 3: ADDITIONAL EXPLORATION
# ============================================================

elif eda_section == "Additional Exploration":

    st.subheader("Additional Exploration")

    st.markdown(
        """
        This section contains supporting EDA elements.

        These are useful for deeper interpretation, but the main presentation focus is on:
        price development, volatility, market events and the relationship between NG and ES.
        """
    )

    # --------------------------------------------------------
    # VWAP ANALYSIS
    # --------------------------------------------------------

    st.header("VWAP Analysis")

    st.markdown(
        """
        VWAP is used as a volume-adjusted reference price.

        In this project it is used as an exploratory feature, not as a direct ARIMA/SARIMA model input.
        """
    )

    if "VWAP" in filtered_ng_df.columns and "Volume" in filtered_ng_df.columns:

        st.pyplot(
            ngu.plot_price_vwap_volume(
                filtered_ng_df.dropna(subset=["VWAP"]),
                title=f"{ng_title} - Close, VWAP and Volume - {selected_period}"
            )
        )

        col1, col2, col3 = st.columns(3)

        with col1:
            above_vwap = (filtered_ng_df["VWAP_diff"] > 0).mean() * 100
            st.metric("Above VWAP", f"{above_vwap:.1f}%")

        with col2:
            below_vwap = (filtered_ng_df["VWAP_diff"] < 0).mean() * 100
            st.metric("Below VWAP", f"{below_vwap:.1f}%")

        with col3:
            mean_vwap_diff = filtered_ng_df["VWAP_diff"].mean()
            st.metric("Avg. distance from VWAP", f"{mean_vwap_diff:.4f}")

        st.markdown(
            """
            **Interpretation**
        
            VWAP is often used in trading as an important volume-adjusted reference level.
        
            In this chart, price often reacts around VWAP. When price is above VWAP, VWAP may act as a support-like reference area. When price has established itself below VWAP, the same level may instead act more like resistance.
        
            The metric `Avg. distance from VWAP` shows the average distance between the close price and VWAP. A positive value means that price was, on average, above VWAP during the selected period. A negative value means that price was, on average, below VWAP.
        
            In this project, VWAP is used as an exploratory reference level, not as a proven trading signal. A possible future extension would be to test anchored VWAP from market events, major highs/lows, week starts, month starts or other meaningful reference points.
            """
        )

    else:
        st.warning("VWAP or Volume column is missing.")

    st.divider()

    # --------------------------------------------------------
    # CALENDAR STATISTICS
    # --------------------------------------------------------

    st.header("Calendar Return Statistics")

    st.markdown(
        """
        These tables summarize log returns by year, month and weekday.
    
        They are used as descriptive EDA only, not as direct model inputs.
        """
    )
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Yearly")
        st.dataframe(
            ngu.get_yearly_stats(filtered_ng_df).round(6),
            use_container_width=True,
            hide_index=True
        )
    
    with col2:
        st.subheader("Monthly")
        st.dataframe(
            ngu.get_monthly_stats(filtered_ng_df).round(6),
            use_container_width=True,
            hide_index=True
        )
    
    with col3:
        st.subheader("Weekday")
        st.dataframe(
            ngu.get_weekday_stats(filtered_ng_df).round(6),
            use_container_width=True,
            hide_index=True
        )

    
    st.markdown(
        """
        **Interpretation**
    
        Calendar statistics are used as descriptive EDA only. Any apparent differences across years, months or weekdays should be interpreted carefully and not treated as stable predictive signals without further testing.
        """
    )



