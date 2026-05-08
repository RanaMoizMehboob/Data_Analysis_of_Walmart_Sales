import base64
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
import streamlit as st
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.seasonal import seasonal_decompose

warnings.filterwarnings("ignore")
sns.set_theme(style="darkgrid")


# -----------------------------------------------------------------------------
# Page configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Walmart Sales Analytics Dashboard by Rana Moiz Mehboob",
    page_icon="📊",
    layout="wide",
)


# -----------------------------------------------------------------------------
# Data loading and preprocessing
# -----------------------------------------------------------------------------
@st.cache_data
def load_data(data_path: str = "Walmart.csv") -> pd.DataFrame:
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"The dataset file '{data_path}' was not found. Please place Walmart.csv in the project folder."
        )

    df = pd.read_csv(data_path)
    df.columns = [col.strip().replace(" ", "_") for col in df.columns]
    df = df.drop_duplicates().reset_index(drop=True)

    df["Date"] = pd.to_datetime(df["Date"], format="%d-%m-%Y", errors="coerce")
    df["Weekly_Sales"] = pd.to_numeric(df["Weekly_Sales"], errors="coerce")
    df["Temperature"] = pd.to_numeric(df["Temperature"], errors="coerce")
    df["Fuel_Price"] = pd.to_numeric(df["Fuel_Price"], errors="coerce")
    df["CPI"] = pd.to_numeric(df["CPI"], errors="coerce")
    df["Unemployment"] = pd.to_numeric(df["Unemployment"], errors="coerce")
    df["Holiday_Flag"] = df["Holiday_Flag"].astype(int)
    df["Store"] = df["Store"].astype(int)

    df = df.dropna(subset=["Date", "Weekly_Sales"])
    df = df[df["Weekly_Sales"] >= 0]

    return df


@st.cache_data
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    engineered = df.copy()
    engineered["Year"] = engineered["Date"].dt.year
    engineered["Month"] = engineered["Date"].dt.month_name()
    engineered["Month_Num"] = engineered["Date"].dt.month
    engineered["Week"] = engineered["Date"].dt.isocalendar().week
    engineered["DayOfWeek"] = engineered["Date"].dt.day_name()
    engineered["Quarter"] = engineered["Date"].dt.quarter
    engineered["Sales_MA_4"] = (
        engineered.sort_values("Date").groupby("Store")["Weekly_Sales"].rolling(4).mean().reset_index(level=0, drop=True)
    )
    engineered["Estimated_Profit"] = engineered["Weekly_Sales"] * 0.10
    engineered["Profit_Margin"] = 0.10

    engineered["Performance_Segment"] = pd.qcut(
        engineered["Weekly_Sales"],
        q=4,
        labels=["Low Sales", "Moderate Sales", "Strong Sales", "Top Sales"],
    )

    def region_mapper(store_id: int) -> str:
        if store_id <= 10:
            return "North"
        if store_id <= 20:
            return "South"
        if store_id <= 30:
            return "East"
        if store_id <= 40:
            return "West"
        return "Central"

    engineered["Region"] = engineered["Store"].apply(region_mapper)
    engineered["Season"] = engineered["Month_Num"].apply(
        lambda x: "Winter" if x in [12, 1, 2]
        else "Spring" if x in [3, 4, 5]
        else "Summer" if x in [6, 7, 8]
        else "Fall"
    )
    engineered["Sales_Category"] = engineered["Performance_Segment"]

    return engineered


def compute_kpis(df: pd.DataFrame) -> dict:
    total_revenue = df["Weekly_Sales"].sum()
    total_profit = df["Estimated_Profit"].sum()
    total_records = len(df)
    average_order_value = df["Weekly_Sales"].mean()
    best_category = df["Sales_Category"].value_counts().idxmax()
    top_region = df["Region"].value_counts().idxmax()
    first_year_sales = df[df["Year"] == df["Year"].min()]["Weekly_Sales"].mean()
    last_year_sales = df[df["Year"] == df["Year"].max()]["Weekly_Sales"].mean()
    growth_percentage = ((last_year_sales - first_year_sales) / first_year_sales) * 100 if first_year_sales else 0

    return {
        "total_revenue": total_revenue,
        "total_profit": total_profit,
        "total_records": total_records,
        "average_order_value": average_order_value,
        "best_category": best_category,
        "top_region": top_region,
        "growth_percentage": growth_percentage,
        "profit_margin": 10,
    }


# -----------------------------------------------------------------------------
# Charting helper functions
# -----------------------------------------------------------------------------

def line_chart_sales_trend(df: pd.DataFrame):
    trend = df.groupby("Date")["Weekly_Sales"].sum().reset_index()
    fig = px.line(
        trend,
        x="Date",
        y="Weekly_Sales",
        title="Total Weekly Sales Trend",
        labels={"Weekly_Sales": "Weekly Sales", "Date": "Date"},
        template="plotly_dark",
    )
    fig.update_layout(margin=dict(l=40, r=20, t=60, b=40))
    return fig


def bar_chart_region_sales(df: pd.DataFrame):
    region_sales = df.groupby("Region")["Weekly_Sales"].sum().sort_values(ascending=False).reset_index()
    fig = px.bar(
        region_sales,
        x="Region",
        y="Weekly_Sales",
        title="Sales by Region",
        color="Region",
        template="plotly_dark",
    )
    return fig


def pie_chart_performance_segment(df: pd.DataFrame):
    category_sales = df.groupby("Sales_Category")["Weekly_Sales"].sum().reset_index()
    fig = px.pie(
        category_sales,
        names="Sales_Category",
        values="Weekly_Sales",
        title="Sales Share by Performance Segment",
        hole=0.4,
        template="plotly_dark",
    )
    return fig


def heatmap_sales_by_weekday_month(df: pd.DataFrame):
    pivot = df.pivot_table(
        values="Weekly_Sales",
        index="DayOfWeek",
        columns="Month",
        aggfunc="mean",
    ).reindex(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(pivot, annot=True, fmt=".0f", cmap="viridis", ax=ax)
    ax.set_title("Average Weekly Sales by Weekday and Month")
    ax.set_xlabel("Month")
    ax.set_ylabel("Day of Week")
    plt.tight_layout()
    return fig


def scatter_fuel_vs_sales(df: pd.DataFrame):
    fig = px.scatter(
        df,
        x="Fuel_Price",
        y="Weekly_Sales",
        color="Holiday_Flag",
        title="Fuel Price vs Weekly Sales",
        labels={"Fuel_Price": "Fuel Price", "Weekly_Sales": "Weekly Sales", "Holiday_Flag": "Holiday"},
        template="plotly_dark",
        hover_data=["Store", "Region"],
    )
    return fig


def box_plot_seasonal_sales(df: pd.DataFrame):
    fig = px.box(
        df,
        x="Season",
        y="Weekly_Sales",
        color="Season",
        title="Seasonal Weekly Sales Distribution",
        template="plotly_dark",
    )
    return fig


def area_chart_monthly_sales(df: pd.DataFrame):
    monthly = (
        df.groupby(["Year", "Month_Num", "Month"])["Weekly_Sales"]
        .sum()
        .reset_index()
        .sort_values(["Year", "Month_Num"])
    )
    fig = px.area(
        monthly,
        x="Month",
        y="Weekly_Sales",
        color="Year",
        line_group="Year",
        title="Monthly Sales Trend by Year",
        template="plotly_dark",
        category_orders={"Month": ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]},
    )
    return fig


def decomposition_chart(df: pd.DataFrame):
    series = df.groupby("Date")["Weekly_Sales"].sum().reset_index()
    series = series.sort_values("Date").drop_duplicates("Date").set_index("Date").asfreq("W-MON")
    series["Weekly_Sales"] = series["Weekly_Sales"].ffill().interpolate()

    if len(series.dropna()) < 104:
        return None

    try:
        decomposition = seasonal_decompose(series["Weekly_Sales"].dropna(), model="additive", period=52)
    except Exception:
        return None

    fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(12, 10), sharex=True)
    decomposition.observed.plot(ax=axes[0], color="#1f77b4")
    axes[0].set_title("Observed Weekly Sales")
    decomposition.trend.plot(ax=axes[1], color="#ff7f0e")
    axes[1].set_title("Trend")
    decomposition.seasonal.plot(ax=axes[2], color="#2ca02c")
    axes[2].set_title("Seasonality")
    decomposition.resid.plot(ax=axes[3], color="#d62728")
    axes[3].set_title("Residual")
    plt.tight_layout()
    return fig


def correlation_heatmap(df: pd.DataFrame):
    numeric_cols = ["Weekly_Sales", "Temperature", "Fuel_Price", "CPI", "Unemployment", "Estimated_Profit"]
    corr = df[numeric_cols].corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax, fmt=".2f")
    ax.set_title("Correlation Matrix")
    plt.tight_layout()
    return fig


def histogram_sales(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(x=df["Weekly_Sales"], bins=30, kde=True, color="#1f77b4", ax=ax)
    ax.set_title("Weekly Sales Distribution")
    ax.set_xlabel("Weekly Sales")
    ax.set_ylabel("Count")
    plt.tight_layout()
    return fig


def forecast_sales(df: pd.DataFrame, periods: int = 12) -> pd.DataFrame:
    series = df.groupby("Date")["Weekly_Sales"].sum().reset_index()
    series = series.sort_values("Date").reset_index(drop=True)
    series["TimeIndex"] = np.arange(len(series))

    model = LinearRegression()
    model.fit(series[["TimeIndex"]], series["Weekly_Sales"])
    future_index = np.arange(len(series), len(series) + periods)
    forecast = model.predict(future_index.reshape(-1, 1))

    future_dates = pd.date_range(start=series["Date"].max() + pd.Timedelta(days=7), periods=periods, freq="W-MON")
    forecast_df = pd.DataFrame(
        {"Date": future_dates, "Forecasted_Sales": forecast, "TimeIndex": future_index}
    )
    return forecast_df


def download_button(df: pd.DataFrame, filename: str = "walmart_processed.csv") -> None:
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f"<a href='data:file/csv;base64,{b64}' download='{filename}'>Download processed data as CSV</a>"
    st.markdown(href, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Dashboard layout and user interface
# -----------------------------------------------------------------------------

def main():
    st.title("Walmart Sales Analytics Dashboard by Rana Moiz Mehboob")
    st.write("Professional sales analytics, forecasting, and interactive reporting for Walmart weekly store performance.")

    try:
        df = load_data()
    except FileNotFoundError as error:
        st.error(str(error))
        return

    df = engineer_features(df)

    with st.sidebar:
        st.header("Filters")
        min_date = df["Date"].min()
        max_date = df["Date"].max()
        date_range = st.date_input(
            "Select date range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
        )

        region_options = st.multiselect(
            "Select regions",
            options=df["Region"].unique(),
            default=df["Region"].unique().tolist(),
        )

        category_options = st.multiselect(
            "Select performance segments",
            options=df["Sales_Category"].unique(),
            default=df["Sales_Category"].unique().tolist(),
        )

        holiday_filter = st.selectbox(
            "Holiday filter",
            options=["All", "Holiday Weeks", "Non-Holiday Weeks"],
        )

        st.markdown("---")
        st.write("Developed by Rana Moiz Mehboob using Python, Streamlit, Plotly, and Data Analytics")

    if len(date_range) != 2:
        st.warning("Please select a valid date range.")
        return

    start_date, end_date = date_range
    filtered_df = df[(df["Date"] >= pd.to_datetime(start_date)) & (df["Date"] <= pd.to_datetime(end_date))]
    filtered_df = filtered_df[filtered_df["Region"].isin(region_options)]
    filtered_df = filtered_df[filtered_df["Sales_Category"].isin(category_options)]

    if holiday_filter == "Holiday Weeks":
        filtered_df = filtered_df[filtered_df["Holiday_Flag"] == 1]
    elif holiday_filter == "Non-Holiday Weeks":
        filtered_df = filtered_df[filtered_df["Holiday_Flag"] == 0]

    if filtered_df.empty:
        st.warning("No rows match the selected filters. Adjust the filters to see the analytics.")
        return

    kpis = compute_kpis(filtered_df)

    kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
    kpi_col1.metric("Total Revenue", f"${kpis['total_revenue']:,.0f}")
    kpi_col2.metric("Total Profit", f"${kpis['total_profit']:,.0f}")
    kpi_col3.metric("Total Records", f"{kpis['total_records']:,}")
    kpi_col4.metric("Avg Weekly Sales", f"${kpis['average_order_value']:,.0f}")

    kpi_col5, kpi_col6, kpi_col7, kpi_col8 = st.columns(4)
    kpi_col5.metric("Best Segment", kpis["best_category"])
    kpi_col6.metric("Top Region", kpis["top_region"])
    kpi_col7.metric("Growth (%)", f"{kpis['growth_percentage']:.2f}%")
    kpi_col8.metric("Profit Margin", f"{kpis['profit_margin']}%")

    st.markdown("---")
    st.subheader("Business Insights")

    with st.expander("View filtered dataset summary"):
        st.write(filtered_df.head(10))
        st.write(filtered_df.describe(include="all"))

    st.subheader("Sales Trends and Performance")
    st.plotly_chart(line_chart_sales_trend(filtered_df), use_container_width=True)

    col1, col2 = st.columns(2)
    col1.plotly_chart(bar_chart_region_sales(filtered_df), use_container_width=True)
    col2.plotly_chart(pie_chart_performance_segment(filtered_df), use_container_width=True)

    col3, col4 = st.columns(2)
    col3.plotly_chart(scatter_fuel_vs_sales(filtered_df), use_container_width=True)
    col4.plotly_chart(box_plot_seasonal_sales(filtered_df), use_container_width=True)

    st.plotly_chart(area_chart_monthly_sales(filtered_df), use_container_width=True)
    st.pyplot(heatmap_sales_by_weekday_month(filtered_df))
    st.pyplot(correlation_heatmap(filtered_df))
    st.pyplot(histogram_sales(filtered_df))
    fig = decomposition_chart(filtered_df)
    if fig is not None:
        st.pyplot(fig)
    else:
        st.warning("Please select a larger date range for Trend Analysis")

    st.markdown("---")
    st.subheader("Advanced Statistical Overview")
    stats_df = pd.DataFrame(
        {
            "Metric": ["Mean", "Median", "Mode", "Std Dev", "Variance"],
            "Weekly Sales": [
                filtered_df["Weekly_Sales"].mean(),
                filtered_df["Weekly_Sales"].median(),
                float(filtered_df["Weekly_Sales"].mode().iloc[0]),
                filtered_df["Weekly_Sales"].std(),
                filtered_df["Weekly_Sales"].var(),
            ],
        }
    )
    st.table(stats_df.style.format({"Weekly Sales": "${:,.2f}"}))

    st.markdown("---")
    st.subheader("Sales Forecasting")
    forecast_df = forecast_sales(filtered_df)
    forecast_fig = px.line(
        forecast_df,
        x="Date",
        y="Forecasted_Sales",
        title="12-Week Sales Forecast",
        template="plotly_dark",
        labels={"Forecasted_Sales": "Forecasted Weekly Sales"},
    )
    st.plotly_chart(forecast_fig, use_container_width=True)
    st.write(forecast_df)

    st.markdown("---")
    download_button(filtered_df)
    st.markdown("---")
    st.write("**Project developed by Rana Moiz Mehboob using Python, Streamlit, Plotly, and Data Analytics.**")


if __name__ == "__main__":
    main()
