import streamlit as st
import pandas as pd
import altair as alt
from ml_churn_code import run_page

st.set_page_config(page_title="GKV Churn Dashboard", layout="wide")

# --- DATENLADE-FUNKTION ---
@st.cache_data
def load_data():
    return pd.read_csv("ml_code/merged_data.csv")

@st.cache_data
def load_causal_data():
    return pd.read_csv("data/causal_effects.csv")

df = load_data()

# --- SIDEBAR NAVIGATION ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", [
    "Start",
    "Causal Effects",
    "Predictive Model",
    "Single Fund Prediction"
])
st.sidebar.markdown("---")
st.sidebar.write("GKV Churn Dashboard")

# --- START SEITE ---
if page == "Start":
    st.title("GKV Churn Dashboard")
    st.markdown("""
    Analyze the impact of additional contributions on member churn.  
    **Navigation:**  
    - **Causal Effects**: Causal Analysis with Difference-in-Differences  
    - **Predictive Model**: Churn Prediction  
    - **Single Fund Prediction**: Individual Fund Forecast  
    """)

    # 1) Avg. Additional Contribution Rate by Quarter
    st.subheader("Average Additional Contribution Rate by Quarter")
    contrib = (
        df.groupby(["Jahr", "Quartal"])["Zusatzbeitrag"]
          .mean()
          .reset_index()
    )
    contrib["Period"] = contrib["Jahr"].astype(str) + " Q" + contrib["Quartal"].astype(str)
    contrib_series = contrib.set_index("Period")["Zusatzbeitrag"]
    st.line_chart(contrib_series)

    # 2) Total Membership by Quarter
    st.subheader("Total Membership by Quarter")
    members = (
        df.groupby(["Jahr", "Quartal"])["Mitglieder"]
          .sum()
          .reset_index()
    )
    members["Period"] = members["Jahr"].astype(str) + " Q" + members["Quartal"].astype(str)
    members_series = members.set_index("Period")["Mitglieder"]
    st.line_chart(members_series)

    # 3) Average Churn Rate by Quarter
    st.subheader("Average Churn Rate by Quarter")
    churn = (
        df.groupby(["Jahr", "Quartal"])["churn_rel"]
          .mean()
          .reset_index()
    )
    churn["Period"] = churn["Jahr"].astype(str) + " Q" + churn["Quartal"].astype(str)
    churn_series = churn.set_index("Period")["churn_rel"]
    st.line_chart(churn_series)

    # 4) Churn Rate Distribution for Latest Quarter
    st.subheader("Churn Rate Distribution (Latest Quarter)")
    latest = (
        df.sort_values(["Jahr", "Quartal"])
          .groupby("Krankenkasse")
          .last()
          .reset_index()
    )
    bar = alt.Chart(latest).mark_bar().encode(
        x=alt.X("Krankenkasse:N", title="Fund"),
        y=alt.Y("churn_rel:Q", title="Churn Rate"),
        tooltip=["Krankenkasse", alt.Tooltip("churn_rel", format=".2%")]
    )
    st.altair_chart(bar, use_container_width=True)

# --- CAUSAL EFFECTS SEITE ---
elif page == "Causal Effects":
    st.title("Causal Effects (Causal Analysis)")
    st.markdown("""
    This section shows estimated treatment effects of additional contribution
    changes on churn for different funds and time periods.
    """)
    causal_df = load_causal_data()

    # KPI cards
    avg_treatment_effect = causal_df["predicted_effect_cf"].mean()
    avg_churn_rate = causal_df["churn_rel"].mean()
    c1, c2 = st.columns(2)
    c1.metric("Avg Treatment Effect", f"{avg_treatment_effect:.4f}")
    c2.metric("Avg Churn Rate", f"{avg_churn_rate:.2%}")

    st.markdown("---")
    st.subheader("Average Causal Effect per Fund")
    avg_effects_per_fund = (
        causal_df.groupby("Krankenkasse")["predicted_effect_cf"]
                 .mean()
                 .reset_index()
    )
    st.dataframe(avg_effects_per_fund)

    st.markdown("---")
    st.subheader("Causal Effects over Time")
    funds = causal_df["Krankenkasse"].unique().tolist()
    selected_funds = st.multiselect("Select funds to visualize", funds, default=funds[:2])
    if selected_funds:
        plot_df = causal_df[causal_df["Krankenkasse"].isin(selected_funds)]
        pivot_causal = plot_df.pivot_table(
            index=["Jahr", "Quartal"],
            columns="Krankenkasse",
            values="predicted_effect_cf"
        )
        pivot_causal.index = pivot_causal.index.map(lambda t: f"{t[0]} Q{t[1]}")
        st.line_chart(pivot_causal)
    else:
        st.info("Select at least one fund to compare.")

    st.markdown("---")
    st.subheader("Actual vs Predicted Churn Rate")
    selected_fund = st.selectbox("Select fund", funds)
    if selected_fund:
        compare_df = causal_df[causal_df["Krankenkasse"] == selected_fund].copy()
        compare_df["Period"] = compare_df.apply(
            lambda r: f"{int(r['Jahr'])} Q{int(r['Quartal'])}", axis=1
        )
        compare_df = compare_df.sort_values(["Jahr", "Quartal"])
        chart = alt.Chart(compare_df).transform_fold(
            ["churn_rel", "predicted_effect_cf"],
            as_=["variable", "value"]
        ).mark_line().encode(
            x="Period:N",
            y="value:Q",
            color="variable:N"
        )
        st.altair_chart(chart, use_container_width=True)
    else:
        st.info("Select a fund to visualize.")

# --- PREDICTIVE MODEL SEITE ---
elif page == "Predictive Model":
    run_page()

# --- EINZEL-VORHERSAGE SEITE ---
elif page == "Single Fund Prediction":
    st.title("Single Fund Prediction")
    st.info("Enter the values for a single fund to receive a churn prediction.")

    with st.form(key="fund_input_form"):
        beitrag = st.number_input("Additional Contribution (%)", min_value=0.0, max_value=5.0, step=0.1)
        jahr = st.selectbox("Year", [2021, 2022, 2023, 2024])
        risiko = st.number_input("Morbidity Risk Factor", min_value=0.0, max_value=5.0, step=0.01)
        mitglieder = st.number_input("Number of Members", min_value=1)
        submit_btn = st.form_submit_button(label="Predict")

    if submit_btn:
        # TODO: Load real predictive model and compute prediction
        churn_risk = beitrag * 0.01 + risiko * 0.02
        st.success(f"Churn Risk (Dummy): {churn_risk:.2%}")
