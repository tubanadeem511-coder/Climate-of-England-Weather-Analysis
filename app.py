from pathlib import Path
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Climate of England Dashboard", page_icon="🌦️", layout="wide")
sns.set_theme(style="whitegrid")

CSV_PATH = Path(__file__).parent / "Climate of England _ Professional Reforms.csv"

@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    data = pd.read_csv(path)
    data["Date"] = pd.to_datetime(data["Formatted Date"], format="%m/%d/%Y", errors="coerce")
    data["Timestamp"] = pd.to_datetime(
        data["Formatted Date"] + " " + data["Time"],
        format="%m/%d/%Y %H:%M:%S.%f",
        errors="coerce"
    )
    data["Precip Type"] = data["Precip Type"].fillna("Unknown")
    data = data.dropna(subset=["Timestamp"]).drop_duplicates().copy()
    data = data.sort_values("Timestamp")
    return data

st.title("🌦️ Climate of England — Weather Dashboard")
st.write(
    "Explore hourly weather observations from England. Use the sidebar to filter the date range "
    "and inspect temperature, humidity, wind, weather conditions and correlations."
)

if not CSV_PATH.exists():
    st.error(
        "CSV file not found. Put 'Climate of England _ Professional Reforms.csv' "
        "in the same folder as app.py."
    )
    st.stop()

data = load_data(str(CSV_PATH))
min_date = data["Date"].min().date()
max_date = data["Date"].max().date()

st.sidebar.header("Filters")
st.sidebar.info("Location: England (the CSV does not contain a country/place column).")
selected_range = st.sidebar.date_input(
    "Date range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

if isinstance(selected_range, (tuple, list)) and len(selected_range) == 2:
    start_date, end_date = selected_range
elif isinstance(selected_range, (tuple, list)) and len(selected_range) == 1:
    start_date = end_date = selected_range[0]
else:
    start_date = end_date = selected_range

filtered = data[
    (data["Date"].dt.date >= start_date) &
    (data["Date"].dt.date <= end_date)
].copy()

if filtered.empty:
    st.warning("No observations match the selected date range.")
    st.stop()

# Four headline metrics
avg_temp = filtered["Temperature (C)"].mean()
max_temp = filtered["Temperature (C)"].max()
avg_humidity = filtered["Humidity"].mean()
number_days = filtered["Date"].dt.date.nunique()

m1, m2, m3, m4 = st.columns(4)
m1.metric("Average temperature", f"{avg_temp:.1f} °C")
m2.metric("Highest temperature", f"{max_temp:.1f} °C")
m3.metric("Average humidity", f"{avg_humidity:.0%}")
m4.metric("Number of days", f"{number_days:,}")

st.caption(
    f"Showing {len(filtered):,} hourly records from {start_date:%d %b %Y} "
    f"to {end_date:%d %b %Y}. Exact duplicate rows are excluded; source data is not modified."
)

# Graph 1 — Univariate temperature distribution
st.subheader("1. Temperature distribution")
fig, ax = plt.subplots(figsize=(9, 4))
ax.hist(filtered["Temperature (C)"].dropna(), bins=35)
ax.set_title("Hourly temperature distribution")
ax.set_xlabel("Temperature (°C)")
ax.set_ylabel("Number of records")
st.pyplot(fig)
plt.close(fig)
st.write(
    f"Typical temperature: mean **{filtered['Temperature (C)'].mean():.2f} °C**, "
    f"median **{filtered['Temperature (C)'].median():.2f} °C**."
)

# Graph 2 — Bivariate scatter
st.subheader("2. Temperature vs humidity")
fig, ax = plt.subplots(figsize=(9, 4))
sns.scatterplot(data=filtered, x="Temperature (C)", y="Humidity", alpha=0.25, s=12, ax=ax)
ax.set_title("Temperature vs humidity")
ax.set_xlabel("Temperature (°C)")
ax.set_ylabel("Humidity (fraction)")
st.pyplot(fig)
plt.close(fig)
st.write(
    f"Pearson correlation: **{filtered['Temperature (C)'].corr(filtered['Humidity']):.3f}**. "
    "Correlation describes linear association, not causation."
)

# Graph 3 — Line chart over time
st.subheader("3. Temperature over time")
daily = filtered.set_index("Timestamp")["Temperature (C)"].resample("D").mean()
fig, ax = plt.subplots(figsize=(12, 4))
daily.plot(ax=ax)
ax.set_title("Daily average temperature over time")
ax.set_xlabel("Date")
ax.set_ylabel("Average temperature (°C)")
st.pyplot(fig)
plt.close(fig)

# Graph 4 — Correlation heatmap
st.subheader("4. Weather correlation heatmap")
numeric_cols = [
    "Temperature (C)", "Apparent Temperature (C)", "Humidity",
    "Wind Speed (km/h)", "Visibility (km)", "Pressure (millibars)"
]
fig, ax = plt.subplots(figsize=(9, 6))
sns.heatmap(filtered[numeric_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
ax.set_title("Correlation between numeric weather variables")
st.pyplot(fig)
plt.close(fig)

# Graph 5 — Weather summary frequencies
st.subheader("5. Most frequent weather conditions")
condition_counts = filtered["Summary"].value_counts().head(12).sort_values()
fig, ax = plt.subplots(figsize=(9, 5))
condition_counts.plot(kind="barh", ax=ax)
ax.set_title("Most frequent weather summaries")
ax.set_xlabel("Number of hourly records")
ax.set_ylabel("Weather condition")
st.pyplot(fig)
plt.close(fig)

# Graph 6 — monthly average temperature (meaningful substitute for place comparison)
st.subheader("6. Average temperature by month")
month_avg = filtered.groupby(filtered["Timestamp"].dt.month)["Temperature (C)"].mean()
fig, ax = plt.subplots(figsize=(9, 4))
month_avg.plot(kind="bar", ax=ax)
ax.set_title("Average temperature by calendar month")
ax.set_xlabel("Month number")
ax.set_ylabel("Average temperature (°C)")
st.pyplot(fig)
plt.close(fig)

st.subheader("Main findings")
st.markdown(
    f"""
    - Average temperature in the selected period is **{avg_temp:.2f} °C**.
    - The highest recorded temperature in the selected period is **{max_temp:.2f} °C**.
    - Average humidity is **{avg_humidity:.1%}**.
    - The most common weather summary is **{filtered['Summary'].mode().iloc[0]}**.
    - Temperature and humidity have a Pearson correlation of **{filtered['Temperature (C)'].corr(filtered['Humidity']):.2f}**.
    - This file contains no place/country column and no numeric rainfall amount, so it cannot support place comparisons or rainfall-volume charts.
    """
)

show_raw = st.checkbox("Show raw data table")
if show_raw:
    st.subheader("Filtered raw data")
    st.dataframe(filtered, use_container_width=True)

st.caption("Source CSV is read-only. The dashboard does not edit or overwrite it.")
