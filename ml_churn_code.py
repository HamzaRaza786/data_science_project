# forecast_and_churn.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, r2_score, pairwise_distances
from sklearn.ensemble import HistGradientBoostingRegressor, GradientBoostingRegressor, RandomForestRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.model_selection import train_test_split
from pandas.tseries.offsets import QuarterEnd

# --- DATA LOADING AND PREPROCESSING ---

@st.cache_data
def load_and_prepare_data() -> pd.DataFrame:
    df = pd.read_csv('./ml_code/merged_data.csv')  # Replace with your path
    df = df.sort_values(['Krankenkasse', 'Jahr', 'Quartal'])
    df['date'] = pd.to_datetime(df['Jahr'].astype(str) + 'Q' + df['Quartal'].astype(str))
    df.set_index('date', inplace=True)
    df = filter_funds_with_2025_q1(df)
    df['competitor_contrib'] = compute_competitor_contrib_knn(df)
    print("competitor_contrib computed", df['competitor_contrib'])
    df = add_additional_features(df)
    return df

def filter_funds_with_2025_q1(df: pd.DataFrame) -> pd.DataFrame:
    required_quarter = pd.to_datetime("2025Q1")
    valid_funds = df[df.index == required_quarter]['Krankenkasse'].unique()
    return df[df['Krankenkasse'].isin(valid_funds)].copy()

def compute_competitor_contrib_knn(df: pd.DataFrame) -> pd.Series:
    df = df.copy()
    contrib_values = {}

    df_reset = df.reset_index()  # Keep original index in a column
    df_reset['_orig_index'] = df.index  # Save original index before drop=True

    for (jahr, quartal), group in df_reset.groupby(['Jahr', 'Quartal']):
        group = group.reset_index(drop=True)  # This index is temporary
        features = group[['Mitglieder', 'Zusatzbeitrag', 'Jahr', 'Quartal']].to_numpy()

        # Compute pairwise Euclidean distances
        dists = pairwise_distances(features, metric='euclidean')

        for i, row in group.iterrows():
            dist_to_others = dists[i].copy()
            dist_to_others[i] = np.inf  # ignore self

            top_k_indices = np.argsort(dist_to_others)[:10]
            mean_contrib = group.loc[top_k_indices, 'Zusatzbeitrag'].mean()

            orig_idx = row['_orig_index']
            contrib_values[orig_idx] = mean_contrib

            if row['Krankenkasse'] == 'techniker-krankenkasse (tk)':
                print(top_k_indices, row['Mitglieder'], row['Zusatzbeitrag'], mean_contrib)

    # Return as Series with proper index matching df
    contrib_series = pd.Series(contrib_values, index=df.index)
    
    # Fill any missing values with overall mean or some fallback
    contrib_series = contrib_series.fillna(df['Zusatzbeitrag'].mean())

    return contrib_series

def add_additional_features(df: pd.DataFrame) -> pd.DataFrame:
    df['Zusatzbeitrag_prev'] = df.groupby('Krankenkasse')['Zusatzbeitrag'].shift(1)
    df['Zusatz_diff'] = df['Zusatzbeitrag'] - df['competitor_contrib']
    df['is_start_of_year'] = (df['Quartal'] == 1).astype(int)
    return df



# --- MAIN STREAMLIT PAGE FUNCTION ---

def run_page():
    st.title("Zusatzbeitrag Forecasting & Churn Analysis")

    # Load the dataset (cached)
    df = load_and_prepare_data()


    df_sorted = df.sort_values(by=['Mitglieder', 'Marktanteil Mitglieder'], ascending=False)

    # Select the top 20 unique funds based on sorted order
    top_funds = df_sorted['Krankenkasse'].unique()[:20]
    top_funds_selected = st.checkbox("Show only top 20 Krankenkasse", value=True)
    # --- User Input Section ---
    col1, col2 = st.columns(2)
    with col1:
        future_year = st.number_input("Forecast until year", min_value=2025, max_value=2035, value=2026)
    with col2:
        future_quarter = st.selectbox("Quarter", [1, 2, 3, 4])
    selected_fund = None
    if top_funds_selected:
        # If top funds checkbox is selected, show only top 20 funds
        selected_fund = st.selectbox("Select Krankenkasse :", top_funds)
    else:
        # If not selected, show all unique funds
        selected_fund = st.selectbox("Select Krankenkasse:", df['Krankenkasse'].unique())

    fund_df = df[df["Krankenkasse"] == selected_fund].copy()

    if len(fund_df) < 12:
        st.warning("Not enough data points for modeling.")
        st.stop()

    # --- Forecasting Section ---
    st.subheader(f"Zusatzbeitrag Forecast for {selected_fund}")

    # Calculate steps ahead to forecast
    last_date = fund_df.index.max()
    target_date = pd.to_datetime(f"{future_year}Q{future_quarter}") + QuarterEnd(0)
    steps_ahead = ((target_date.year - last_date.year) * 4 + (target_date.quarter - last_date.quarter))

    if steps_ahead <= 0:
        st.warning("Future date must be after latest data point.")
        st.stop()

    future_dates = pd.date_range(start=last_date + QuarterEnd(1), periods=steps_ahead, freq='Q')

    # Train-test split for validation
    train = fund_df.iloc[:-8]
    test = fund_df.iloc[-8:]

    # Train SARIMAX model
    model = SARIMAX(
        train["Zusatzbeitrag"],
        exog=train[["competitor_contrib"]],
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, 4),
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    results = model.fit(disp=False)

    # Forecast and evaluate
    pred = results.get_forecast(steps=8, exog=test[["competitor_contrib"]])
    forecast = pred.predicted_mean
    conf_int = pred.conf_int()

    mae = mean_absolute_error(test["Zusatzbeitrag"], forecast)
    r2 = r2_score(test["Zusatzbeitrag"], forecast)

    # Forecast into the future
    future_exog = pd.DataFrame({"competitor_contrib": [test["competitor_contrib"].iloc[-1]] * steps_ahead})
    future_forecast = results.get_forecast(steps=steps_ahead, exog=future_exog).predicted_mean

    # Plot forecast and confidence intervals
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(train.index, train["Zusatzbeitrag"], label="Train", color="blue")
    ax.plot(test.index, test["Zusatzbeitrag"], label="Actual", color="green")
    ax.plot(test.index, forecast, label="Forecast", color="orange")
    ax.plot(future_dates, future_forecast, label="Future Forecast", color="purple")
    ax.fill_between(test.index, conf_int.iloc[:, 0], conf_int.iloc[:, 1], color="orange", alpha=0.2)
    ax.set_title(f"SARIMAX Forecast for {selected_fund}")
    ax.set_ylabel("Zusatzbeitrag")
    ax.legend()
    st.pyplot(fig)
    st.markdown(f"**MAE:** {mae:.4f} &nbsp;&nbsp;&nbsp; **R²:** {r2:.4f}")

    # --- Churn Modeling Section ---
    st.subheader(f"Predicted Relative Churn for {selected_fund}")
    selected_model_name = st.selectbox("Model Type", ["HistGradientBoosting", "LGBMRegressor", "GradientBoosting", "Ridge", "DecisionTree", "RandomForest", "SVR", "MLPRegressor", "KernelRidge", "XGBRegressor", "CatBoostRegressor"])

    # Prepare data for churn prediction
    churn_df = df.dropna(subset=['Zusatzbeitrag', 'competitor_contrib', 'churn_rel'])
    fund_churn_df = churn_df[churn_df['Krankenkasse'] == selected_fund].copy()

    if fund_churn_df.empty:
        st.warning("No churn data available for the selected fund.")
    else:
        # Train-test split
        features = ['Zusatzbeitrag', 'competitor_contrib', 'Zusatzbeitrag_prev', 'Zusatz_diff']
        X = churn_df[features]
        y = churn_df['churn_rel']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Choose model based on user selection
        models = {
            "HistGradientBoosting": HistGradientBoostingRegressor(
                learning_rate=0.05,
                max_depth=7,
                max_iter=100,
                max_bins=100,  # Optimal bin count
                early_stopping=True,  # Automatic iteration control
                l2_regularization=0.1  # Prevent overfitting

            ),
            "LGBMRegressor": LGBMRegressor(),
            "GradientBoosting": GradientBoostingRegressor(
                n_estimators=200, max_depth=5, learning_rate=0.1
            ),
            "Ridge": Ridge(max_iter=10000),
            "DecisionTree": DecisionTreeRegressor(),
            "RandomForest": RandomForestRegressor(),
            "SVR": SVR(),
            "MLPRegressor": MLPRegressor(max_iter=500),
            "KernelRidge": KernelRidge(),
            "XGBRegressor": XGBRegressor(verbosity=0, n_estimators=1000, random_state=42),
            "LGBMRegressor": LGBMRegressor(),
            "CatBoostRegressor": CatBoostRegressor(verbose=0)

        }

        model = models[selected_model_name]
        model.fit(X_train, y_train)

        # Evaluate model
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        st.markdown(f"**MAE (Overall):** {mae:.4f} &nbsp;&nbsp;&nbsp; **R² (Overall):** {r2:.4f}")

        # Predict on the selected fund
        fund_churn_df['date'] = pd.to_datetime(fund_churn_df['Jahr'].astype(str) + 'Q' + fund_churn_df['Quartal'].astype(str))
        fund_churn_df[f'predicted_churn_{selected_model_name}'] = model.predict(fund_churn_df[features])

        # Plot predicted vs actual churn
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(fund_churn_df['date'], fund_churn_df['churn_rel'], label='Actual Churn', color='black', linestyle='--', marker='o')
        ax.plot(fund_churn_df['date'], fund_churn_df[f'predicted_churn_{selected_model_name}'], label='Predicted Churn', color='blue')
        ax.set_title(f"Predicted vs Actual Relative Churn for {selected_fund}")
        ax.set_ylabel("Relative Churn")
        ax.legend()
        st.pyplot(fig)

        # Display results in table
        columns_to_show = ['date', 'Zusatzbeitrag', 'competitor_contrib', 'churn_rel', f'predicted_churn_{selected_model_name}']
        st.subheader(f"Churn Predictions for {selected_fund} using {selected_model_name}")
        st.dataframe(fund_churn_df[columns_to_show])
