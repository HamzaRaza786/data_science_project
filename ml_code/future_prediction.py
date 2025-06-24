# streamlit_forecast_models.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from pandas.tseries.offsets import QuarterEnd
from sklearn.preprocessing import StandardScaler

# --- Load and preprocess data ---
@st.cache_data
def load_and_prepare_data():
    df = pd.read_csv("merged_data.csv")  # Replace with your path
    df = df.sort_values(['Krankenkasse', 'Jahr', 'Quartal'])
    df['date'] = pd.to_datetime(df['Jahr'].astype(str) + 'Q' + df['Quartal'].astype(str)) + QuarterEnd(0)
    df.set_index('date', inplace=True)
    df['competitor_contrib'] = df.groupby(['Jahr', 'Quartal'])['Zusatzbeitrag'].transform(
        lambda x: (x.sum() - x) / (len(x) - 1)
    )
    return df

df = load_and_prepare_data()
top_funds = df.groupby("Krankenkasse")["Mitglieder"].mean().nlargest(10).index.tolist()

# --- Streamlit UI ---
st.title("Zusatzbeitrag Forecasting: ARIMA / SARIMA / Random Forest")
selected_fund = st.selectbox("Select Krankenkasse (Top 10):", top_funds)
model_type = st.selectbox("Select Model:", ["ARIMA", "SARIMA", "Random Forest"])

col1, col2 = st.columns(2)
with col1:
    future_year = st.number_input("Forecast until year", min_value=2025, max_value=2035, value=2026)
with col2:
    future_quarter = st.selectbox("Quarter", [1, 2, 3, 4])

fund_df = df[df["Krankenkasse"] == selected_fund].copy()
if len(fund_df) < 12:
    st.warning("Not enough data points for modeling.")
    st.stop()

train = fund_df.iloc[:-8]
test = fund_df.iloc[-8:]

last_date = fund_df.index.max()
target_date = pd.to_datetime(f"{future_year}Q{future_quarter}") + QuarterEnd(0)
steps_ahead = ((target_date.year - last_date.year) * 4 + (target_date.quarter - last_date.quarter))

if steps_ahead <= 0:
    st.warning("Future date must be after latest data point.")
    st.stop()

future_dates = pd.date_range(start=last_date + QuarterEnd(1), periods=steps_ahead, freq='Q')
conf_int = None  # For ARIMA/SARIMA, will be set later
# --- Modeling ---
try:
    if model_type == "ARIMA":
        model = ARIMA(train["Zusatzbeitrag"], order=(1, 1, 1))
        results = model.fit()
        pred = results.get_forecast(steps=8)
        conf_int = pred.conf_int()
        forecast = results.forecast(steps=8)
        test_mae = mean_absolute_error(test["Zusatzbeitrag"], forecast)
        test_r2 = r2_score(test["Zusatzbeitrag"], forecast)
        future_forecast = results.forecast(steps=steps_ahead)
    elif model_type == "SARIMA":
        model = SARIMAX(
            train["Zusatzbeitrag"],
            exog=train[["competitor_contrib"]],
            order=(1, 1, 1),
            seasonal_order=(1, 1, 1, 4),
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        results = model.fit(disp=False)
        pred = results.get_forecast(steps=8, exog=test[["competitor_contrib"]])
        forecast = pred.predicted_mean
        conf_int = pred.conf_int()
        test_mae = mean_absolute_error(test["Zusatzbeitrag"], forecast)
        test_r2 = r2_score(test["Zusatzbeitrag"], forecast)
        future_exog = pd.DataFrame({"competitor_contrib": [test["competitor_contrib"].iloc[-1]] * steps_ahead})
        future_forecast = results.get_forecast(steps=steps_ahead, exog=future_exog).predicted_mean
    else:  # Random Forest
        # Create features
        rf_df = fund_df.copy()
        rf_df['quarter_id'] = np.arange(len(rf_df))
        rf_df['Zusatzbeitrag_prev'] = rf_df['Zusatzbeitrag'].shift(1)
        features = ['competitor_contrib', 'Zusatzbeitrag_prev', 'Jahr', 'Quartal']
        X = rf_df[features]
        y = rf_df['Zusatzbeitrag']
        X_train, y_train = X.iloc[:-8], y.iloc[:-8]
        X_test, y_test = X.iloc[-8:], y.iloc[-8:]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train_scaled, y_train)
        forecast = model.predict(X_test_scaled)
        test_mae = mean_absolute_error(y_test, forecast)
        test_r2 = r2_score(y_test, forecast)
        # --- Future Forecasting ---
        future_steps = steps_ahead
        last_known = rf_df.iloc[-1]
        future_rows = []

        last_quarter_id = last_known['quarter_id']
        last_year = int(last_known['Jahr'])
        last_quarter = int(last_known['Quartal'])
        last_zb = last_known['Zusatzbeitrag']
        last_comp = last_known['competitor_contrib']

        for i in range(1, future_steps + 1):
            # Increment quarter/year
            next_qid = last_quarter_id + i
            next_quarter = (last_quarter + i - 1) % 4 + 1
            next_year = last_year + ((last_quarter + i - 1) // 4)

            # Create feature row
            row = {
                # "quarter_id": next_qid,
                "competitor_contrib": last_comp,  # could be updated later
                "Zusatzbeitrag_prev": last_zb,
                "Jahr": next_year,
                "Quartal": next_quarter
            }

            row_df = pd.DataFrame([row])
            row_scaled = scaler.transform(row_df)
            next_pred = model.predict(row_scaled)[0]
            row["Zusatzbeitrag"] = next_pred
            future_rows.append(row)
            last_zb = next_pred  # Update last Zusatzbeitrag for next iteration

        # Convert to DataFrame
        future_forecast = pd.DataFrame(future_rows)
        future_forecast = future_forecast.set_index(pd.date_range(start=last_date + QuarterEnd(1), periods=steps_ahead, freq='Q'))['Zusatzbeitrag']
    # --- Plot ---
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(train.index, train["Zusatzbeitrag"], label="Train", color="blue")
    ax.plot(test.index, test["Zusatzbeitrag"], label="Actual", color="green")
    ax.plot(test.index, forecast, label="Forecast", color="orange")
    ax.plot(future_dates, future_forecast, label="Future Forecast", color="purple")
    ax.set_title(f"Forecasting Zusatzbeitrag for {selected_fund} using {model_type}")
    ax.set_ylabel("Zusatzbeitrag")
    ax.legend()
    if model_type in ["SARIMA", "ARIMA"]:
        ax.fill_between(test.index, conf_int.iloc[:, 0], conf_int.iloc[:, 1], color="orange", alpha=0.2)
    st.pyplot(fig)

    st.markdown(f"**Test MAE:** {test_mae:.4f} &nbsp;&nbsp;&nbsp; **R²:** {test_r2:.4f}")

except Exception as e:
    st.error(f"Error: {e}")
