import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
from datetime import timedelta

# ── Page Config ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Toronto Island Ferry Forecast",
    page_icon="⛴️",
    layout="wide"
)

# ── Load Assets ────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    return {
        "Gradient Boosting": joblib.load("models/gb_model.pkl"),
        "Random Forest":     joblib.load("models/rf_model.pkl"),
        "XGBoost":           joblib.load("models/xgb_model.pkl"),
    }

@st.cache_data
def load_data():
    df = pd.read_csv("data/ferry_data.csv", parse_dates=["timestamp"])
    df["date"] = pd.to_datetime(df["date"])
    return df

@st.cache_data
def load_features():
    return joblib.load("data/feature_cols.pkl")

models      = load_models()
df          = load_data()
feature_cols= load_features()

# ── Sidebar ────────────────────────────────────────────────────────────
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/4/4b/Toronto_Islands_ferry_Trillium.jpg/320px-Toronto_Islands_ferry_Trillium.jpg", use_column_width=True)
st.sidebar.title("⛴️ Ferry Forecast")
st.sidebar.markdown("---")

selected_model = st.sidebar.selectbox(
    "🤖 Select Forecast Model",
    list(models.keys())
)

horizon_label = st.sidebar.selectbox(
    "⏱️ Forecast Horizon",
    ["15 minutes", "30 minutes", "1 hour", "2 hours"]
)
horizon_map = {"15 minutes": 1, "30 minutes": 2, "1 hour": 4, "2 hours": 8}
horizon_steps = horizon_map[horizon_label]

target = st.sidebar.radio(
    "🎯 Prediction Target",
    ["Sales", "Redemptions"]
)
target_col = "sales" if target == "Sales" else "redemptions"

st.sidebar.markdown("---")
min_date = df["timestamp"].min().date()
max_date = df["timestamp"].max().date()
selected_date = st.sidebar.date_input(
    "📅 Select Date to Inspect",
    value=max_date,
    min_value=min_date,
    max_value=max_date
)

st.sidebar.markdown("---")
st.sidebar.markdown("**📊 Model Performance**")
perf = {
    "Gradient Boosting": {"MAE": 17.118, "RMSE": 63.005},
    "Random Forest":     {"MAE": 17.347, "RMSE": 62.952},
    "XGBoost":           {"MAE": 17.490, "RMSE": 63.706},
}
st.sidebar.metric("MAE",  perf[selected_model]["MAE"])
st.sidebar.metric("RMSE", perf[selected_model]["RMSE"])

# ── Header ─────────────────────────────────────────────────────────────
st.title("⛴️ Toronto Island Ferry — Demand Forecast Dashboard")
st.markdown(f"**Model:** `{selected_model}`  |  **Horizon:** `{horizon_label}`  |  **Target:** `{target}`")
st.markdown("---")

# ── KPI Cards ──────────────────────────────────────────────────────────
day_data = df[df["timestamp"].dt.date == selected_date]

if len(day_data) == 0:
    st.warning("No data available for selected date. Please choose another date.")
    st.stop()

col1, col2, col3, col4 = st.columns(4)
col1.metric("🎟️ Total Sales Today",       int(day_data["sales"].sum()))
col2.metric("🚢 Total Boardings Today",   int(day_data["redemptions"].sum()))
col3.metric("📈 Peak Sales (15-min)",     int(day_data["sales"].max()))
col4.metric("⏰ Peak Hour",               f"{day_data.loc[day_data['sales'].idxmax(), 'timestamp'].strftime('%H:%M')}")

st.markdown("---")

# ── Generate Forecast ──────────────────────────────────────────────────
model = models[selected_model]

# Get last known row to build features for future prediction
last_row   = df[df["timestamp"].dt.date <= selected_date].iloc[-1]
future_rows = []

for step in range(1, horizon_steps + 1):
    future_time = last_row["timestamp"] + timedelta(minutes=15 * step)
    row = {}
    row["hour"]        = future_time.hour
    row["day_of_week"] = future_time.dayofweek
    row["month"]       = future_time.month
    row["is_weekend"]  = int(future_time.dayofweek in [5, 6])
    row["is_peak_hour"]= int(future_time.hour in range(10, 18))
    row["quarter"]     = future_time.quarter
    row["hour_sin"]    = np.sin(2 * np.pi * future_time.hour / 24)
    row["hour_cos"]    = np.cos(2 * np.pi * future_time.hour / 24)
    row["dow_sin"]     = np.sin(2 * np.pi * future_time.dayofweek / 7)
    row["dow_cos"]     = np.cos(2 * np.pi * future_time.dayofweek / 7)
    row["month_sin"]   = np.sin(2 * np.pi * future_time.month / 12)
    row["month_cos"]   = np.cos(2 * np.pi * future_time.month / 12)

    # Lag features from last known data
    for lag in [1, 2, 4, 8, 96, 192, 672]:
        src_idx = len(df) - lag
        row[f"sales_lag_{lag}"]       = df["sales"].iloc[src_idx] if src_idx >= 0 else 0
        row[f"redemptions_lag_{lag}"] = df["redemptions"].iloc[src_idx] if src_idx >= 0 else 0

    # Rolling features
    for window in [4, 8, 96]:
        row[f"sales_rollmean_{window}"]  = df["sales"].iloc[-window:].mean()
        row[f"sales_rollstd_{window}"]   = df["sales"].iloc[-window:].std()
        row[f"redeem_rollmean_{window}"] = df["redemptions"].iloc[-window:].mean()

    future_rows.append(row)

future_df   = pd.DataFrame(future_rows)[feature_cols]
future_pred = np.clip(model.predict(future_df), 0, None)

future_times = [last_row["timestamp"] + timedelta(minutes=15 * s) for s in range(1, horizon_steps + 1)]

# Confidence band (±1 MAE as simple uncertainty estimate)
mae_val = perf[selected_model]["MAE"]
pred_upper = future_pred + mae_val
pred_lower = np.clip(future_pred - mae_val, 0, None)

# ── Tabs ───────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📈 Forecast View", "🔍 Actuals vs Predicted", "📊 Model Comparison"])

# ── Tab 1: Forecast ────────────────────────────────────────────────────
with tab1:
    st.subheader(f"🔮 {horizon_label} Ahead Forecast — {target}")

    # Show last 48 actuals + forecast
    recent = day_data.tail(48)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=recent["timestamp"], y=recent[target_col],
        name="Actual", line=dict(color="#1f77b4", width=2)
    ))
    fig.add_trace(go.Scatter(
        x=future_times, y=future_pred,
        name="Forecast", line=dict(color="orange", width=2, dash="dash"),
        mode="lines+markers"
    ))
    fig.add_trace(go.Scatter(
        x=future_times + future_times[::-1],
        y=list(pred_upper) + list(pred_lower[::-1]),
        fill="toself", fillcolor="rgba(255,165,0,0.15)",
        line=dict(color="rgba(255,255,255,0)"),
        name="Confidence Band"
    ))

    fig.update_layout(
        title=f"{selected_model} — {target} Forecast for next {horizon_label}",
        xaxis_title="Time",
        yaxis_title=f"{target} Count",
        hovermode="x unified",
        height=450,
        legend=dict(orientation="h", yanchor="bottom", y=1.02)
    )
    st.plotly_chart(fig, use_container_width=True)

    # Forecast table
    st.markdown("**📋 Forecast Values**")
    forecast_table = pd.DataFrame({
        "Timestamp":       [t.strftime("%Y-%m-%d %H:%M") for t in future_times],
        "Predicted":       future_pred.round(1),
        "Lower Bound":     pred_lower.round(1),
        "Upper Bound":     pred_upper.round(1),
    })
    st.dataframe(forecast_table, use_container_width=True)

# ── Tab 2: Actuals vs Predicted ────────────────────────────────────────
with tab2:
    st.subheader("🔍 Actuals vs Predicted — Selected Day")

    if len(day_data) >= 10:
        X_day   = day_data[feature_cols].dropna()
        y_day   = day_data.loc[X_day.index, target_col]
        ts_day  = day_data.loc[X_day.index, "timestamp"]
        y_hat   = np.clip(model.predict(X_day), 0, None)

        residuals = y_day.values - y_hat
        day_mae   = np.mean(np.abs(residuals))
        day_rmse  = np.sqrt(np.mean(residuals**2))

        m1, m2, m3 = st.columns(3)
        m1.metric("Day MAE",       round(day_mae, 2))
        m2.metric("Day RMSE",      round(day_rmse, 2))
        m3.metric("Max Residual",  round(np.max(np.abs(residuals)), 2))

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=ts_day, y=y_day,
            name="Actual", line=dict(color="#1f77b4", width=2)
        ))
        fig2.add_trace(go.Scatter(
            x=ts_day, y=y_hat,
            name="Predicted", line=dict(color="orange", width=2, dash="dash")
        ))
        fig2.update_layout(
            title=f"Actual vs Predicted {target} — {selected_date}",
            xaxis_title="Time", yaxis_title=f"{target} Count",
            hovermode="x unified", height=400
        )
        st.plotly_chart(fig2, use_container_width=True)

        # Residual plot
        fig3 = go.Figure()
        fig3.add_trace(go.Bar(
            x=ts_day, y=residuals,
            marker_color=["red" if r < 0 else "green" for r in residuals],
            name="Residual"
        ))
        fig3.add_hline(y=0, line_dash="dash", line_color="black")
        fig3.update_layout(
            title="Residuals (Actual − Predicted)",
            xaxis_title="Time", yaxis_title="Residual",
            height=300
        )
        st.plotly_chart(fig3, use_container_width=True)
    else:
        st.info("Not enough data for selected date to show actuals vs predicted.")

# ── Tab 3: Model Comparison ────────────────────────────────────────────
with tab3:
    st.subheader("📊 Model Comparison — All Models")

    comp_data = {
        "Model":    ["Gradient Boosting", "Random Forest", "XGBoost",
                     "Linear Regression", "Moving Average", "Naïve"],
        "MAE":      [17.118, 17.347, 17.490, 18.049, 19.521, 21.450],
        "RMSE":     [63.005, 62.952, 63.706, 63.311, 69.836, 86.374],
        "MAPE (%)": [116.32, 120.19, 119.13, 136.31, 115.89, 128.17],
        "Type":     ["ML","ML","ML","Baseline","Baseline","Baseline"]
    }
    comp_df = pd.DataFrame(comp_data)

    fig4 = go.Figure()
    colors = ["steelblue" if t == "ML" else "lightcoral" for t in comp_df["Type"]]
    fig4.add_trace(go.Bar(
        x=comp_df["Model"], y=comp_df["MAE"],
        marker_color=colors, name="MAE",
        text=comp_df["MAE"], textposition="outside"
    ))
    fig4.update_layout(
        title="MAE by Model (lower is better)",
        yaxis_title="MAE", height=400,
        xaxis_tickangle=-20
    )
    st.plotly_chart(fig4, use_container_width=True)

    fig5 = go.Figure()
    fig5.add_trace(go.Bar(
        x=comp_df["Model"], y=comp_df["RMSE"],
        marker_color=colors, name="RMSE",
        text=comp_df["RMSE"], textposition="outside"
    ))
    fig5.update_layout(
        title="RMSE by Model (lower is better)",
        yaxis_title="RMSE", height=400,
        xaxis_tickangle=-20
    )
    st.plotly_chart(fig5, use_container_width=True)

    st.markdown("**📋 Full Metrics Table**")
    st.dataframe(comp_df, use_container_width=True)

# ── Footer ─────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("*Toronto Island Ferry Demand Forecasting — Capstone Project*")