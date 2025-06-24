# streamlit_sarimax_app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, r2_score

# --- Load and preprocess data ---
@st.cache_data
def load_and_prepare_data():
    df = pd.read_csv("merged_data.csv")  # Replace with your path
    df = df.sort_values(['Krankenkasse', 'Jahr', 'Quartal'])
    
    # Create datetime index
    df['date'] = pd.to_datetime(df['Jahr'].astype(str) + 'Q' + df['Quartal'].astype(str))
    df.set_index('date', inplace=True)

    # Compute competitor average Zusatzbeitrag
    df['competitor_contrib'] = df.groupby(['Jahr', 'Quartal'])['Zusatzbeitrag'].transform(
        lambda x: (x.sum() - x) / (len(x) - 1)
    )

    return df

df = load_and_prepare_data()

# --- Select top 10 funds by Mitglieder ---
top_funds = df.groupby("Krankenkasse")["Mitglieder"].mean().nlargest(10).index.tolist()

# --- Streamlit UI ---
st.title("Zusatzbeitrag Forecasting with SARIMAX")
selected_fund = st.selectbox("Select a Krankenkasse (Top 10 by members):", top_funds)

# --- Filter and prepare data for selected fund ---
fund_df = df[df["Krankenkasse"] == selected_fund].copy()

if len(fund_df) < 12:
    st.warning("Not enough data points for SARIMAX.")
    st.stop()

# --- Train-test split (last 8 quarters = test) ---
train = fund_df.iloc[:-8]
test = fund_df.iloc[-8:]

# --- Fit SARIMAX (with seasonal component and competitor contrib as exog) ---
order = (1, 1, 1)
seasonal_order = (1, 1, 1, 4)  # quarterly seasonality

try:    
    model = SARIMAX(
        train["Zusatzbeitrag"],
        exog=train[["competitor_contrib"]],
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    results = model.fit(disp=False)

    # Forecast with exogenous future competitor contrib
    pred = results.get_forecast(
        steps=8,
        exog=test[["competitor_contrib"]]
    )
    forecast = pred.predicted_mean
    conf_int = pred.conf_int()

    # Evaluate
    mae = mean_absolute_error(test["Zusatzbeitrag"], forecast)
    r2 = r2_score(test["Zusatzbeitrag"], forecast)

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(train.index, train["Zusatzbeitrag"], label="Train", color="blue")
    ax.plot(test.index, test["Zusatzbeitrag"], label="Actual", color="green")
    ax.plot(test.index, forecast, label="Forecast", color="orange")
    print(conf_int)
    ax.fill_between(test.index, conf_int.iloc[:, 0], conf_int.iloc[:, 1], color="orange", alpha=0.2)
    ax.set_title(f"SARIMAX Forecast for {selected_fund}")
    ax.set_ylabel("Zusatzbeitrag")
    ax.legend()

    st.pyplot(fig)

    st.markdown(f"**MAE:** {mae:.4f} &nbsp;&nbsp;&nbsp; **R²:** {r2:.4f}")

except Exception as e:
    st.error(f"Model fitting failed: {e}")
