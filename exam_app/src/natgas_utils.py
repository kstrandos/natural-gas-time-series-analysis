# ============================================================
# natgas_utils.py
# Natural Gas Streamlit App Utilities
# DAT803 Exam Project
# ============================================================

from pathlib import Path
import ast
import io

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import textwrap

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


# ============================================================
# 1. DATA LOADING AND CLEANING
# ============================================================

def convert_volume_to_float(value):
    """
    Convert Investing.com-style volume strings to floats.

    Examples:
    - '30.15K' -> 30150
    - '2.4M'   -> 2400000
    - '-'      -> np.nan
    """

    if pd.isna(value):
        return np.nan

    if isinstance(value, (int, float)):
        return value

    value = str(value).strip()

    if value in ["-", "", "nan", "NaN", "None"]:
        return np.nan

    multiplier = 1

    if value.endswith("K"):
        multiplier = 1_000
        value = value.replace("K", "")
    elif value.endswith("M"):
        multiplier = 1_000_000
        value = value.replace("M", "")
    elif value.endswith("B"):
        multiplier = 1_000_000_000
        value = value.replace("B", "")

    try:
        return float(value.replace(",", "")) * multiplier
    except ValueError:
        return np.nan


def load_quoted_csv(filepath: Path) -> pd.DataFrame:
    """
    Load a CSV file where each line may be wrapped in extra quotes.

    This is used for the older ES dataset where the CSV structure differs
    from the regular Investing.com files.
    """

    clean_lines = []

    with open(filepath, "r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()

            if line.startswith('"') and line.endswith('"'):
                line = line[1:-1]

            line = line.replace('""', '"')

            clean_lines.append(line)

    clean_text = "\n".join(clean_lines)

    df = pd.read_csv(
        io.StringIO(clean_text),
        parse_dates=["Date"]
    )

    return df


def load_ng_raw(data_dir: Path) -> pd.DataFrame:
    """
    Load the raw Natural Gas futures dataset used in the notebook.
    """

    ng_file = data_dir / "Natural Gas Futures 010400-082319.csv"

    ng_df = pd.read_csv(
        ng_file,
        parse_dates=["Date"]
    )

    return ng_df


def load_es_raw(data_dir: Path) -> pd.DataFrame:
    """
    Load and combine the two ES futures datasets used in the notebook.

    ES consists of:
    - SP500 Futures 010406-082319.csv
    - SP500 Futures 010400-010306.csv
    """

    es_file1 = data_dir / "SP500 Futures 010406-082319.csv"
    es_file2 = data_dir / "SP500 Futures 010400-010306.csv"

    es_part1 = pd.read_csv(
        es_file1,
        parse_dates=["Date"]
    )

    es_part2 = load_quoted_csv(es_file2)

    es_df = pd.concat(
        [es_part1, es_part2],
        ignore_index=True
    )

    return es_df


def get_dataset_titles(data_dir: Path) -> tuple[str, str]:
    """
    Get readable dataset titles based on the original file names.
    """

    ng_file = data_dir / "Natural Gas Futures 010400-082319.csv"
    es_file = data_dir / "SP500 Futures 010406-082319.csv"

    ng_title = ng_file.stem.split(" 0")[0]
    es_title = es_file.stem.split(" 0")[0]

    return ng_title, es_title


def clean_ng_data(ng_df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean Natural Gas futures data using the same logic as the notebook,
    but standardize Vol. to Volume for easier Streamlit plotting.
    """

    ng_df = ng_df.copy()

    ng_df = ng_df.sort_values("Date").set_index("Date")

    if "Price" in ng_df.columns and "Close" not in ng_df.columns:
        ng_df = ng_df.rename(columns={"Price": "Close"})

    if "Vol." in ng_df.columns and "Volume" not in ng_df.columns:
        ng_df = ng_df.rename(columns={"Vol.": "Volume"})

    for col in ["Open", "High", "Low", "Close"]:
        if col in ng_df.columns:
            ng_df[col] = (
                ng_df[col]
                .astype(str)
                .str.replace(",", "", regex=False)
            )
            ng_df[col] = pd.to_numeric(ng_df[col], errors="coerce")

    if "Volume" in ng_df.columns:
        ng_df["Volume"] = ng_df["Volume"].apply(convert_volume_to_float)
        ng_df["Volume"] = ng_df["Volume"].ffill()

    ng_df = ng_df.sort_index()

    return ng_df


def clean_es_data(es_df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean ES futures data using the same logic as the notebook,
    but standardize Vol. to Volume for easier Streamlit plotting.
    """

    es_df = es_df.copy()

    if "Price" in es_df.columns and "Close" not in es_df.columns:
        es_df = es_df.rename(columns={"Price": "Close"})

    if "Vol." in es_df.columns and "Volume" not in es_df.columns:
        es_df = es_df.rename(columns={"Vol.": "Volume"})

    for col in ["Open", "High", "Low", "Close"]:
        if col in es_df.columns:
            es_df[col] = (
                es_df[col]
                .astype(str)
                .str.replace(",", "", regex=False)
            )
            es_df[col] = pd.to_numeric(es_df[col], errors="coerce")

    if "Volume" in es_df.columns:
        es_df["Volume"] = es_df["Volume"].apply(convert_volume_to_float)
        es_df["Volume"] = es_df["Volume"].ffill()

    es_df = es_df.drop_duplicates(subset=["Date"])
    es_df = es_df.sort_values("Date").set_index("Date")
    es_df = es_df.sort_index()

    return es_df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add the main engineered features used in the project.

    Features:
    - Log_returns
    - Close_diff_1
    - Rolling_mean_14
    - Rolling_std_14
    - Lag1, Lag2, Lag5, Lag10
    - Std_diff_1
    - VWAP
    - VWAP_diff
    - VWAP_regime
    - calendar features
    """

    df = df.copy()

    if "Close" in df.columns:
        df["Log_returns"] = np.log(df["Close"] / df["Close"].shift(1))
        df["Close_diff_1"] = df["Close"].diff(1)

    if "Log_returns" in df.columns:
        df["Rolling_mean_14"] = df["Log_returns"].rolling(14).mean()
        df["Rolling_std_14"] = df["Log_returns"].rolling(14).std()

        df["Lag1"] = df["Log_returns"].shift(1)
        df["Lag2"] = df["Log_returns"].shift(2)
        df["Lag5"] = df["Log_returns"].shift(5)
        df["Lag10"] = df["Log_returns"].shift(10)

    if "Rolling_std_14" in df.columns:
        df["Std_diff_1"] = df["Rolling_std_14"].diff(1)

    if "Close" in df.columns and "Volume" in df.columns:
        valid_volume = df["Volume"].replace(0, np.nan)

        df["VWAP"] = (
            (df["Close"] * valid_volume).cumsum()
            / valid_volume.cumsum()
        )

        df["VWAP_diff"] = df["Close"] - df["VWAP"]

        df["VWAP_regime"] = np.where(
            df["VWAP_diff"] > 0,
            "Above VWAP",
            "Below VWAP"
        )

    df["Year"] = df.index.year
    df["Month"] = df.index.month
    df["Day"] = df.index.day
    df["Day of Week"] = df.index.day_name().str[:3]
    df["Month Name"] = df.index.month_name().str[:3]
    df["is_sunday"] = (df["Day of Week"] == "Sun").astype(int)

    return df


def load_project_datasets(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, str, str]:
    """
    Load, clean and feature-engineer the project datasets using notebook logic.

    Returns:
    - ng_df
    - es_df
    - ng_title
    - es_title
    """

    ng_raw = load_ng_raw(data_dir)
    es_raw = load_es_raw(data_dir)

    ng_df = clean_ng_data(ng_raw)
    es_df = clean_es_data(es_raw)

    ng_df = add_features(ng_df)
    es_df = add_features(es_df)

    ng_title, es_title = get_dataset_titles(data_dir)

    return ng_df, es_df, ng_title, es_title


# ============================================================
# 2. GRID SEARCH LOADING
# ============================================================

def find_latest_grid_search_file(data_dir: Path) -> Path | None:
    """
    Find the newest ARIMA grid search result file in the data folder.

    Expected filename pattern:
    arima_grid_search_results_*.csv
    """

    matching_files = list(data_dir.glob("arima_grid_search_results_*.csv"))

    if not matching_files:
        return None

    latest_file = max(matching_files, key=lambda file: file.stat().st_mtime)

    return latest_file


def load_latest_grid_search_results(data_dir: Path) -> tuple[pd.DataFrame | None, Path | None]:
    """
    Load the newest ARIMA grid search results file from the data folder.
    """

    latest_file = find_latest_grid_search_file(data_dir)

    if latest_file is None:
        return None, None

    df = pd.read_csv(latest_file)

    return df, latest_file


# ============================================================
# 3. PERIOD FILTERING
# ============================================================


def filter_by_period(df: pd.DataFrame, period_name: str) -> pd.DataFrame:
    """
    Filter a dataframe using the predefined regime names.
    """

    if period_name == "Complete":
        return df.copy()

    if period_name == "High volatility":
        return df.loc["2000":"2010"].copy()

    if period_name == "Low volatility":
        return df.loc["2011":"2019"].copy()

    return df.copy()


# ============================================================
# 4. EDA TABLE HELPERS
# ============================================================

def get_yearly_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Yearly summary of log returns.

    Returns a presentation-ready table with Year as text.
    This prevents Streamlit from displaying years as 2,000 instead of 2000.
    """

    stats = (
        df.groupby("Year")["Log_returns"]
        .agg(["mean", "std", "min", "max", "count"])
        .reset_index()
        .sort_values("Year")
    )

    stats["Year"] = stats["Year"].astype(str)

    return stats


def get_monthly_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Monthly summary of log returns.

    Returns a presentation-ready table ordered from Jan to Dec.
    """

    month_order = [
        "Jan", "Feb", "Mar", "Apr",
        "May", "Jun", "Jul", "Aug",
        "Sep", "Oct", "Nov", "Dec"
    ]

    stats = (
        df.groupby("Month Name")["Log_returns"]
        .agg(["mean", "std", "min", "max", "count"])
        .reindex(month_order)
        .reset_index()
    )

    return stats


def get_weekday_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Weekday summary of log returns.

    Returns a presentation-ready table ordered by weekday.
    """

    weekday_order = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sun"]

    stats = (
        df.groupby("Day of Week")["Log_returns"]
        .agg(["mean", "std", "min", "max", "count"])
    )

    stats = (
        stats
        .reindex([day for day in weekday_order if day in stats.index])
        .reset_index()
    )

    return stats


# ============================================================
# 5. PLOTTING HELPERS
# ============================================================


def plot_price_returns_volatility(df: pd.DataFrame, title: str = "NG Price, Returns and Volatility"):
    """
    Plot Close, Log_returns, Rolling_std_14 and Volume in one stacked figure.
    """

    rows = 4 if "Volume" in df.columns else 3

    fig, axes = plt.subplots(
        rows,
        1,
        figsize=(12, 8),
        sharex=True
    )

    axes[0].plot(df.index, df["Close"], linewidth=0.8)
    axes[0].set_ylabel("Close")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(df.index, df["Log_returns"], linewidth=0.8)
    axes[1].set_ylabel("Log returns")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(df.index, df["Rolling_std_14"], linewidth=0.8)
    axes[2].set_ylabel("Rolling std 14")
    axes[2].grid(True, alpha=0.3)

    if rows == 4:
        axes[3].plot(df.index, df["Volume"], linewidth=0.8)
        axes[3].set_ylabel("Volume")
        axes[3].grid(True, alpha=0.3)

    fig.suptitle(title)
    fig.tight_layout()

    return fig


def plot_price_vwap_volume(df: pd.DataFrame, title: str = "NG Price with VWAP and Volume"):
    """
    Plot Close and VWAP with Volume below.
    """

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(12, 7),
        linewidth=0.8,
        sharex=True
    )

    axes[0].plot(df.index, df["Close"], label="Close", linewidth=0.8)
    axes[0].plot(df.index, df["VWAP"], label="VWAP", linewidth=0.8)
    axes[0].set_ylabel("USD")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(df.index, df["Volume"], label="Volume", linewidth=0.6)
    axes[1].set_ylabel("Volume")
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(title)
    fig.tight_layout()

    return fig


# ============================================================
# 6. MARKET EVENTS
# ============================================================


def load_market_events(data_dir: Path) -> pd.DataFrame:
    """
    Load cached market events from local CSV.

    Expected file:
    data/market_events_wikipedia.csv

    The file is created in the notebook from Wikipedia:
    List of stock market crashes and bear markets.

    Events are used only for EDA context, not as model features.
    """

    events_file = data_dir / "market_events_wikipedia.csv"

    if not events_file.exists():
        raise FileNotFoundError(
            f"Market events file not found: {events_file}"
        )

    events_raw = pd.read_csv(events_file)

    required_cols = ["Name", "Date", "Country", "Notes"]

    missing_cols = [
        col for col in required_cols
        if col not in events_raw.columns
    ]

    if missing_cols:
        raise ValueError(
            f"Market events file is missing columns: {missing_cols}"
        )

    events = events_raw[required_cols].copy()

    events["date_text"] = events["Date"].astype(str)

    try:
        events["event_date"] = pd.to_datetime(
            events["date_text"],
            errors="coerce",
            dayfirst=True,
            format="mixed"
        )
    except TypeError:
        events["event_date"] = pd.to_datetime(
            events["date_text"],
            errors="coerce",
            dayfirst=True
        )

    # Manual correction for date ranges that pandas cannot parse automatically
    events.loc[
        events["Name"].str.contains("2008 financial crisis", case=False, na=False),
        "event_date"
    ] = pd.to_datetime("2008-09-16")

    events = events.dropna(subset=["event_date"])

    return events


def prepare_project_events(
    events: pd.DataFrame,
    start_date,
    end_date
) -> pd.DataFrame:
    """
    Filter and prepare market events for the project period.

    Adds:
    - Short_Name
    - label_offset_days
    """

    events_project = events[
        (events["event_date"] >= pd.to_datetime(start_date)) &
        (events["event_date"] <= pd.to_datetime(end_date))
    ].copy()

    events_project = events_project.sort_values("event_date")

    # Short labels for better plot readability
    events_project["Short_Name"] = events_project["Name"].replace({
        "Economic effects of the September 11 attacks": "9/11",
        "Stock market downturn of 2002": "2002 downturn",
        "Chinese stock bubble of 2007": "China 2007",
        "2009 Dubai debt standstill": "Dubai 2009",
        "2010 flash crash": "Flash crash",
        "August 2011 stock markets fall": "Aug 2011 fall",
        "European sovereign debt crisis": "Euro area crisis",
        "2015–16 Chinese stock market crash": "China 2015–16",
        "2015–2016 stock market selloff": "2015–16 selloff",
        "2018 cryptocurrency crash": "Crypto 2018"
    })

    events_project.loc[
        events_project["Name"].str.contains("2008 financial crisis", case=False, na=False),
        "Short_Name"
    ] = "Finance cr. 2008"

    # Manual label offsets for readability
    events_project["label_offset_days"] = 0

    event_offsets = {
        "Dubai 2009": -50,
        "Flash crash": -50,
        "Euro area crisis": 50,
        "China 2015–16": -60,
        "2015–16 selloff": 60
    }

    for event_name, offset in event_offsets.items():
        events_project.loc[
            events_project["Short_Name"] == event_name,
            "label_offset_days"
        ] = offset

    return events_project


def plot_normalized_prices(
    ng_df: pd.DataFrame,
    es_df: pd.DataFrame,
    events_project: pd.DataFrame | None = None,
    show_events: bool = False,
    title: str | None = None
):
    """
    Plot normalized NG and ES prices.

    Prices are normalized to 100 at the start of the overlapping period.

    Optional:
    - show market event lines if events_project is provided and show_events=True
    """

    # Use only overlapping date range
    common_start = max(ng_df.index.min(), es_df.index.min())
    common_end = min(ng_df.index.max(), es_df.index.max())

    ng_plot = ng_df.loc[common_start:common_end].copy()
    es_plot = es_df.loc[common_start:common_end].copy()

    ng_plot = ng_plot.dropna(subset=["Close"])
    es_plot = es_plot.dropna(subset=["Close"])

    if ng_plot.empty or es_plot.empty:
        raise ValueError("NG or ES data is empty after aligning date range.")

    # Normalize price
    ng_norm = ng_plot["Close"] / ng_plot["Close"].iloc[0] * 100
    es_norm = es_plot["Close"] / es_plot["Close"].iloc[0] * 100

    if title is None:
        if show_events:
            title = "Natural Gas & S&P 500 Futures\nNormalized Price Chart with Market Events"
        else:
            title = "Natural Gas vs S&P 500 Futures\nNormalized Price Comparison"

    fig, ax1 = plt.subplots(figsize=(12, 7.5 if show_events else 5))

    ax1.plot(
        ng_norm.index,
        ng_norm,
        linewidth=0.8,
        label="NG Normalized"
    )

    ax1.plot(
        es_norm.index,
        es_norm,
        linewidth=0.8,
        label="ES Normalized"
    )

    # Optional event lines and labels
    if show_events and events_project is not None:

        for _, row in events_project.iterrows():

            event_date = row["event_date"]
            event_name = row["Short_Name"]
            x_offset = pd.Timedelta(days=row["label_offset_days"])

            if event_date < common_start or event_date > common_end:
                continue

            ax1.axvline(
                x=event_date,
                linestyle="--",
                color="red",
                alpha=0.15
            )

            ax1.text(
                event_date + x_offset,
                -0.12,
                event_name,
                rotation=90,
                fontsize=9,
                ha="center",
                va="top",
                transform=ax1.get_xaxis_transform(),
                clip_on=False
            )

    ax1.set_title(title)
    ax1.set_ylabel("Normalized price (start = 100)")

    ax1.legend(
        loc="lower left",
        bbox_to_anchor=(0.01, 1.02),
        frameon=True
    )

    ax1.grid(True, alpha=0.3)

    if show_events:
        fig.subplots_adjust(top=0.82, bottom=0.35)
    else:
        fig.tight_layout()

    return fig


def plot_normalized_prices_with_market_events(
    ng_df: pd.DataFrame,
    es_df: pd.DataFrame,
    events_project: pd.DataFrame,
    title: str = "Natural Gas & S&P 500 Futures\nNormalized Price Chart with Market Events"
):
    """
    Backward-compatible wrapper for the market event plot.
    """

    return plot_normalized_prices(
        ng_df=ng_df,
        es_df=es_df,
        events_project=events_project,
        show_events=True,
        title=title
    )


def create_ng_es_correlation_df(
    ng_df: pd.DataFrame,
    es_df: pd.DataFrame,
    window: int = 60
) -> pd.DataFrame:
    """
    Create a dataframe with NG and ES log returns and rolling correlation.

    The function aligns NG and ES on common dates.
    """

    corr_df = pd.DataFrame({
        "NG_Log_returns": ng_df["Log_returns"],
        "ES_Log_returns": es_df["Log_returns"]
    }).dropna()

    corr_df[f"Rolling_corr_{window}"] = (
        corr_df["NG_Log_returns"]
        .rolling(window)
        .corr(corr_df["ES_Log_returns"])
    )

    return corr_df


def plot_rolling_correlation_with_market_events(
    corr_df: pd.DataFrame,
    events_project: pd.DataFrame,
    window: int = 60,
    title: str | None = None
):
    """
    Plot rolling correlation between NG and ES log returns with market events.
    """

    corr_col = f"Rolling_corr_{window}"

    if corr_col not in corr_df.columns:
        raise ValueError(f"Column {corr_col} not found in correlation dataframe.")

    if title is None:
        title = f"{window}-Day Rolling Correlation: NG vs ES Log Returns with Market Events"

    fig, ax1 = plt.subplots(figsize=(12, 5))

    ax1.plot(
        corr_df.index,
        corr_df[corr_col],
        label=f"{window}-day rolling correlation",
        linewidth=0.8
    )

    ax1.axhline(
        y=0,
        color="black",
        linewidth=1
    )

    # Add event lines and labels
    for _, row in events_project.iterrows():

        event_date = row["event_date"]
        event_name = row["Short_Name"]
        x_offset = pd.Timedelta(days=row["label_offset_days"])

        if event_date < corr_df.index.min() or event_date > corr_df.index.max():
            continue

        ax1.axvline(
            x=event_date,
            linestyle="--",
            color = "red",
            alpha=0.2
        )

        ax1.text(
            event_date + x_offset,
            -0.12,
            event_name,
            rotation=90,
            fontsize=9,
            ha="center",
            va="top",
            transform=ax1.get_xaxis_transform(),
            clip_on=False
        )

    ax1.set_title(title)
    ax1.set_ylabel("Correlation")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    fig.subplots_adjust(bottom=0.32)
    fig.tight_layout()

    return fig


def plot_normalized_prices_with_market_events_plotly(
    ng_df: pd.DataFrame,
    es_df: pd.DataFrame,
    events_project: pd.DataFrame,
    title: str = "Natural Gas & S&P 500 Futures - Normalized Price Chart with Market Events"
):
    """
    Minimal and explicit Plotly version.
    First goal: make NG and ES normalized price lines match the Matplotlib chart.
    """

    common_start = max(ng_df.index.min(), es_df.index.min())
    common_end = min(ng_df.index.max(), es_df.index.max())

    ng_close = ng_df.loc[common_start:common_end, "Close"].astype(float)
    es_close = es_df.loc[common_start:common_end, "Close"].astype(float)

    plot_df = pd.concat(
        [
            ng_close.rename("NG_Close"),
            es_close.rename("ES_Close")
        ],
        axis=1,
        join="inner"
    ).dropna()

    plot_df = plot_df.sort_index()

    plot_df["NG_Normalized"] = (
        plot_df["NG_Close"] / plot_df["NG_Close"].iloc[0] * 100
    )

    plot_df["ES_Normalized"] = (
        plot_df["ES_Close"] / plot_df["ES_Close"].iloc[0] * 100
    )

    # Force plain Python values for Plotly
    x_values = plot_df.index.strftime("%Y-%m-%d").tolist()
    ng_values = plot_df["NG_Normalized"].astype(float).tolist()
    es_values = plot_df["ES_Normalized"].astype(float).tolist()

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=x_values,
            y=ng_values,
            mode="lines",
            name="NG Normalized",
            line=dict(width=1.5, color="#1f77b4"),
            hovertemplate=(
                "Date: %{x}<br>"
                "NG normalized: %{y:.2f}"
                "<extra></extra>"
            )
        )
    )

    fig.add_trace(
        go.Scatter(
            x=x_values,
            y=es_values,
            mode="lines",
            name="ES Normalized",
            line=dict(width=1.5, color="#ff7f0e"),
            hovertemplate=(
                "Date: %{x}<br>"
                "ES normalized: %{y:.2f}"
                "<extra></extra>"
            )
        )
    )


    # ------------------------------------------------------------
    # Market event lines, labels and hover markers
    # ------------------------------------------------------------
    
    event_line_color = "rgba(255, 80, 80, 0.25)"
    
    label_y_paper = -0.12
    
    for _, row in events_project.iterrows():
    
        event_date = pd.to_datetime(row["event_date"])
    
        if event_date < pd.to_datetime("2000-01-01") or event_date > pd.to_datetime("2020-01-01"):
            continue
    
        event_name = row["Short_Name"]
        full_name = row["Name"] if "Name" in row and pd.notna(row["Name"]) else event_name
        country = row["Country"] if "Country" in row and pd.notna(row["Country"]) else ""
        notes = row["Notes"] if "Notes" in row and pd.notna(row["Notes"]) else "No notes available"
        
        if isinstance(notes, str):
            notes = "<br>".join(textwrap.wrap(notes, width=150))
        else:
            notes = "No notes available"
            
        x_offset = (
            pd.Timedelta(days=int(row["label_offset_days"]))
            if "label_offset_days" in row
            else pd.Timedelta(days=0)
        )
    
        label_date = event_date + x_offset
    
        # Vertical dashed line inside the plot area
        fig.add_vline(
            x=event_date,
            line_width=1,
            line_dash="dash",
            line_color=event_line_color
        )
        hover_notes = notes

        annotation_hover_text = (
            "<span style='display:block; text-align:left; width:520px;'>"
            f"<b>{full_name}</b><br><br>"
            f"<b>Date:</b> {event_date.strftime('%Y-%m-%d')}<br>"
            f"<b>Country:</b> {country}<br><br>"
            f"<b>Notes:</b><br>"
            f"{hover_notes}"
            "</span>"
        )
        
        fig.add_annotation(
            x=label_date,
            y=label_y_paper,
            yref="paper",
            text=event_name,
            showarrow=False,
            textangle=-90,
            font=dict(
                size=15,
                color="black"
            ),
            xanchor="center",
            yanchor="top",
            hovertext=annotation_hover_text,
            captureevents=True,
            hoverlabel=dict(
                bgcolor="white",
                bordercolor="black",
                font=dict(
                    size=14,
                    family="Arial",
                    color="black"
                )
            )
        )
    
    fig.update_layout(
        title=dict(
            text=title,
            x=0.5,
            xanchor="center",
            font=dict(
                color="black",
                size=20,
                family="Arial"
            )
        ),
        template="plotly_white",
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(color="black", size=15),
        # xaxis_title="Date",
        yaxis_title="Normalized price (start = 100)",
        height=730,
        margin=dict(l=80, r=40, t=80, b=220),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            font=dict(color="black", size=15)
        ),
        hoverlabel=dict(
            bgcolor="white",
            bordercolor="black",
            font=dict(
                size=14,
                family="Arial",
                color="black"
            )
        ),
    )

    fig.update_yaxes(
        range=[
            0,
            max(max(ng_values), max(es_values)) * 1.1
        ],
        tickmode="array",
        tickvals=[
            100, 200, 300, 400, 500, 600, 700
        ],
        ticktext=[
            "100", "200", "300", "400", "500", "600", "700"
        ],
        showgrid=True,
        gridcolor="rgba(0,0,0,0.12)",
        showline=True,
        linewidth=1,
        linecolor="black",
        mirror=True,
        ticks="outside",
        tickfont=dict(color="black", size=17),
        title_font=dict(color="black", size=19),
        zeroline=False
    )
    
    fig.update_xaxes(
        range=[
            common_start.strftime("%Y-%m-%d"),
            common_end.strftime("%Y-%m-%d")
        ],
        showgrid=True,
        gridcolor="rgba(0,0,0,0.12)",
        showline=True,
        zeroline=False,
        linewidth=1,
        linecolor="black",
        mirror=True,
        ticks="outside",
        tickfont=dict(color="black", size=17),
        title_font=dict(color="black", size=19)
    )

    return fig


# ============================================================
# 7. STATIONARITY PLOTS
# ============================================================

def plot_acf_pacf_grid(
    df: pd.DataFrame,
    columns: list[str],
    lags: int = 20,
    title: str = "PACF and ACF Comparison"
):
    """
    Plot PACF and ACF for multiple time series in a grid.

    Layout:
    - left column: PACF
    - right column: ACF
    - one row per selected series

    This follows the notebook presentation style.
    """

    n_rows = len(columns)

    fig, axes = plt.subplots(
        n_rows,
        2,
        figsize=(14, 3.2 * n_rows)
    )

    if n_rows == 1:
        axes = np.array([axes])

    for row_idx, col in enumerate(columns):

        series = df[col].dropna()

        # PACF
        plot_pacf(
            series,
            lags=lags,
            ax=axes[row_idx, 0],
            method="ywm"
        )

        axes[row_idx, 0].set_title(
            f"{col} – Partial Autocorrelation Function (PACF)"
        )

        axes[row_idx, 0].grid(True, alpha=0.3)

        # ACF
        plot_acf(
            series,
            lags=lags,
            ax=axes[row_idx, 1]
        )

        axes[row_idx, 1].set_title(
            f"{col} – Autocorrelation Function (ACF)"
        )

        axes[row_idx, 1].grid(True, alpha=0.3)

    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    return fig


# ============================================================
# 8. FORECAST / RESULTS HELPERS
# ============================================================

def get_regime_dataframe(ng_df: pd.DataFrame, regime_name: str) -> pd.DataFrame:
    """
    Return the dataframe belonging to one modelling regime.
    """

    if regime_name == "Complete":
        return ng_df.copy()

    if regime_name == "High volatility":
        return ng_df.loc["2000":"2010"].copy()

    if regime_name == "Low volatility":
        return ng_df.loc["2011":"2019"].copy()

    raise ValueError(f"Unknown regime: {regime_name}")


def prepare_train_test_for_target(
    ng_df: pd.DataFrame,
    target_col: str,
    regime_name: str
):
    """
    Prepare chronological train/test split using notebook logic.

    Complete:
    - 80/20 split

    High volatility / Low volatility:
    - 90/10 split
    """

    regime_df = get_regime_dataframe(ng_df, regime_name)

    series = regime_df[target_col].dropna()

    if regime_name == "Complete":
        split_idx = int(len(series) * 0.8)
    else:
        split_idx = int(len(series) * 0.9)

    train = series.iloc[:split_idx]
    test = series.iloc[split_idx:]

    return regime_df, train, test


def get_best_grid_model_by_metric(
    all_results_df: pd.DataFrame,
    target_col: str,
    regime_name: str,
    metric: str = "rmse"
):
    """
    Get best grid search model for selected target/regime/metric.

    Lower values are treated as better for all supported metrics.
    """

    filtered = all_results_df[
        (all_results_df["target"].astype(str) == str(target_col)) &
        (all_results_df["regime"].astype(str) == str(regime_name))
    ].copy()

    if filtered.empty:
        raise ValueError(f"No grid results found for {target_col} / {regime_name}")

    sort_cols = [metric]

    if metric != "aic" and "aic" in filtered.columns:
        sort_cols.append("aic")

    best_row = filtered.sort_values(sort_cols, ascending=True).iloc[0]

    best_order = best_row["order"]
    best_seasonal = best_row["seasonal_order"]

    if isinstance(best_order, str):
        best_order = ast.literal_eval(best_order)

    if isinstance(best_seasonal, str):
        best_seasonal = ast.literal_eval(best_seasonal)

    return best_order, best_seasonal, best_row


def fit_best_model_forecast(
    train: pd.Series,
    test: pd.Series,
    best_order: tuple,
    best_seasonal: tuple
) -> pd.Series:
    """
    Fit selected SARIMAX model and forecast the test period.

    This follows the notebook logic, but returns a clean pandas Series.
    """

    from statsmodels.tsa.statespace.sarimax import SARIMAX

    model_grid = SARIMAX(
        train,
        order=best_order,
        seasonal_order=best_seasonal,
        enforce_stationarity=False,
        enforce_invertibility=False
    )

    fit_grid = model_grid.fit(disp=False)

    pred_grid = fit_grid.forecast(steps=len(test))

    pred_grid = pd.Series(
        np.asarray(pred_grid),
        index=test.index,
        name="Grid Search Forecast"
    )

    return pred_grid, fit_grid


def reintegrate_forecast(
    original_df: pd.DataFrame,
    train: pd.Series,
    test: pd.Series,
    pred_grid: pd.Series,
    target_col: str
) -> dict:
    """
    Reintegrate forecast back to original scale.

    - Close_diff_1 -> Close level
    - Log_returns  -> Close level
    - Std_diff_1   -> Rolling_std_14 level
    """

    from sklearn.metrics import mean_squared_error, mean_absolute_error

    if target_col == "Close_diff_1":
        actual_train_original = original_df.loc[train.index, "Close"]
        actual_test_original = original_df.loc[test.index, "Close"]

        last_train_value = actual_train_original.iloc[-1]
        pred_original = last_train_value + pred_grid.cumsum()

        y_label = "Close"

    elif target_col == "Log_returns":
        actual_train_original = original_df.loc[train.index, "Close"]
        actual_test_original = original_df.loc[test.index, "Close"]

        last_train_value = actual_train_original.iloc[-1]
        pred_original = last_train_value * np.exp(pred_grid.cumsum())

        y_label = "Close"

    elif target_col == "Std_diff_1":
        actual_train_original = original_df.loc[train.index, "Rolling_std_14"]
        actual_test_original = original_df.loc[test.index, "Rolling_std_14"]

        last_train_value = actual_train_original.iloc[-1]
        pred_original = last_train_value + pred_grid.cumsum()

        y_label = "Rolling_std_14"

    else:
        raise ValueError("Unsupported target_col for reintegration.")

    pred_original = pd.Series(
        np.asarray(pred_original),
        index=test.index,
        name=f"Predicted {y_label}"
    )

    mse = mean_squared_error(actual_test_original, pred_original)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actual_test_original, pred_original)

    return {
        "actual_train": actual_train_original,
        "actual_test": actual_test_original,
        "forecast": pred_original,
        "y_label": y_label,
        "mse": mse,
        "rmse": rmse,
        "mae": mae
    }


def get_reintegrated_forecast_result(
    ng_df: pd.DataFrame,
    all_results_df: pd.DataFrame,
    target_col: str,
    regime_name: str,
    metric: str = "rmse"
) -> dict:
    """
    Complete notebook-style forecast pipeline for one target/regime:

    - prepare train/test split
    - find best grid search model
    - fit selected model
    - forecast test period
    - reintegrate forecast to original scale
    - return result dictionary
    """

    original_df, train, test = prepare_train_test_for_target(
        ng_df=ng_df,
        target_col=target_col,
        regime_name=regime_name
    )

    best_order, best_seasonal, best_row = get_best_grid_model_by_metric(
        all_results_df=all_results_df,
        target_col=target_col,
        regime_name=regime_name,
        metric=metric
    )

    pred_grid, fit_grid = fit_best_model_forecast(
        train=train,
        test=test,
        best_order=best_order,
        best_seasonal=best_seasonal
    )

    reintegrated = reintegrate_forecast(
        original_df=original_df,
        train=train,
        test=test,
        pred_grid=pred_grid,
        target_col=target_col
    )

    return {
        "regime": regime_name,
        "target": target_col,
        "order": best_order,
        "seasonal_order": best_seasonal,
        "best_row": best_row,
        "train": train,
        "test": test,
        "raw_forecast": pred_grid,
        "fitted_aic": fit_grid.aic,
        "fitted_bic": fit_grid.bic,
        **reintegrated
    }


def get_reintegrated_forecasts_for_target(
    ng_df: pd.DataFrame,
    all_results_df: pd.DataFrame,
    target_col: str
) -> tuple[list[dict], pd.DataFrame]:
    """
    Create reintegrated forecast results for all three regimes.

    This follows the notebook logic:
    - Complete
    - High volatility
    - Low volatility

    Best model is selected by RMSE, then AIC.
    """

    regimes = [
        "Complete",
        "High volatility",
        "Low volatility"
    ]

    results = []

    for regime_name in regimes:
        result = get_reintegrated_forecast_result(
            ng_df=ng_df,
            all_results_df=all_results_df,
            target_col=target_col,
            regime_name=regime_name,
            metric="rmse"
        )

        results.append(result)

    summary_df = pd.DataFrame([
        {
            "regime": r["regime"],
            "target": r["target"],
            "order": r["order"],
            "seasonal_order": r["seasonal_order"],
            "reintegrated_mse": r["mse"],
            "reintegrated_rmse": r["rmse"],
            "reintegrated_mae": r["mae"]
        }
        for r in results
    ])

    return results, summary_df


def plot_reintegrated_forecasts_grid(
    results: list[dict],
    target_col: str,
    zoom: bool = False
):
    """
    Plot reintegrated forecasts for all regimes in one figure.

    This follows the notebook presentation style.

    If zoom=False:
    - show train, actual test and forecast

    If zoom=True:
    - show actual test and forecast only
    """

    if zoom:
        fig, axes = plt.subplots(
            3,
            1,
            figsize=(14, 10),
            sharex=False
        )

        fig_title = f"Test Period Zoom - {target_col}"

    else:
        fig, axes = plt.subplots(
            3,
            1,
            figsize=(14, 12),
            sharex=False
        )

        fig_title = f"Reintegrated Forecasts - {target_col}"

    for ax, result in zip(axes, results):

        if not zoom:
            ax.plot(
                result["actual_train"].index,
                result["actual_train"],
                label="Train",
                color="gray"
            )

        ax.plot(
            result["actual_test"].index,
            result["actual_test"],
            label="Actual test",
            color="black"
        )

        ax.plot(
            result["forecast"].index,
            result["forecast"],
            label="Forecast",
            linestyle="--",
            color="tab:red"
        )

        if zoom:
            ax.set_title(
                f"Test Zoom - {result['regime']} | "
                f"{result['y_label']} from {target_col} | "
                f"RMSE={result['rmse']:.5f}, MAE={result['mae']:.5f}"
            )

        else:
            ax.set_title(
                f"{result['regime']} | "
                f"{result['y_label']} from {target_col} | "
                f"order={result['order']} | "
                f"RMSE={result['rmse']:.5f}, MAE={result['mae']:.5f}"
            )

        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle(fig_title, fontsize=14)
    fig.tight_layout()

    return fig