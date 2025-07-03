import os
import pickle
import streamlit as st
import pandas as pd
import altair as alt
from ml_churn_code import run_page

# --- Streamlit Page Config ---
st.set_page_config(page_title="GKV Churn Dashboard", layout="wide")

# --- DATENLADE-FUNKTIONEN ---
@st.cache_data
def load_data():
    path = "ml_code/merged_data.csv"
    if not os.path.exists(path):
        st.error(f"EDA file not found: {path}")
        return pd.DataFrame()
    return pd.read_csv(path)

@st.cache_data
def load_causal_data():
    path = "data/causal_effects.csv"
    if not os.path.exists(path):
        st.error(f"Causal effects file not found: {path}")
        return pd.DataFrame()
    return pd.read_csv(path)

@st.cache_resource
def load_predictive_model():
    path = "ml_code/ml_churn_model.pkl"
    if not os.path.exists(path):
        st.error(f"Model file not found: {path}")
        return None, None
    with open(path, "rb") as f:
        dv, model = pickle.load(f)
    return dv, model

df           = load_data()
causal_df    = load_causal_data()
dv, model    = load_predictive_model()

# --- SIDEBAR NAVIGATION ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", [
    "Start",
    "Causal Effects",
    "Predictive Model",
    "Single Fund Prediction"
])
st.sidebar.markdown("---")
st.sidebar.write("© 2025 Your Group | GKV Churn")

# --- START SEITE ---
if page == "Start":
    st.title("GKV Churn Dashboard")
    st.markdown("""
        Explore trends in additional contributions, membership and churn before diving into causal or predictive analyses.
        **Navigation:**  
        - **Causal Effects**: Causal Analysis (DiD)  
        - **Predictive Model**: Churn Prediction  
        - **Single Fund Prediction**: Individual Fund Forecast  
    """)

    # 1) Average additional contribution by quarter
    st.subheader("Average Additional Contribution Rate by Quarter")
    contrib = df.groupby(["Jahr","Quartal"])["Zusatzbeitrag"].mean().reset_index()
    contrib["Period"] = contrib["Jahr"].astype(str) + " Q" + contrib["Quartal"].astype(str)
    st.line_chart(contrib.set_index("Period")["Zusatzbeitrag"])

    # 2) Total membership by quarter
    st.subheader("Total Membership by Quarter")
    members = df.groupby(["Jahr","Quartal"])["Mitglieder"].sum().reset_index()
    members["Period"] = members["Jahr"].astype(str) + " Q" + members["Quartal"].astype(str)
    st.line_chart(members.set_index("Period")["Mitglieder"])

    # 3) Average churn rate by quarter
    st.subheader("Average Churn Rate by Quarter")
    churn = df.groupby(["Jahr","Quartal"])["churn_rel"].mean().reset_index()
    churn["Period"] = churn["Jahr"].astype(str) + " Q" + churn["Quartal"].astype(str)
    st.line_chart(churn.set_index("Period")["churn_rel"])

    # 4) Churn rate distribution (latest quarter)
    st.subheader("Churn Rate Distribution (Latest Quarter)")
    latest = (
        df.sort_values(["Jahr","Quartal"])
          .groupby("Krankenkasse").last().reset_index()
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
    st.markdown("Estimated treatment effects of additional contribution changes on churn, across funds and time periods.")

    # KPI-Cards
    avg_te = causal_df["predicted_effect_cf"].mean()
    avg_cr = causal_df["churn_rel"].mean()
    c1, c2 = st.columns(2)
    c1.metric("Avg Treatment Effect", f"{avg_te:.4f}")
    c2.metric("Avg Churn Rate", f"{avg_cr:.2%}")

    st.markdown("---")
    st.subheader("Average Causal Effect per Fund")
    avg_effects = causal_df.groupby("Krankenkasse")["predicted_effect_cf"].mean().reset_index()
    st.dataframe(avg_effects)

    st.markdown("---")
    st.subheader("Causal Effects over Time")
    funds = causal_df["Krankenkasse"].unique().tolist()
    sel = st.multiselect("Select funds to visualize", funds, default=funds[:2])
    if sel:
        df_sel = causal_df[causal_df["Krankenkasse"].isin(sel)]
        pivot = df_sel.pivot_table(
            index=["Jahr","Quartal"],
            columns="Krankenkasse",
            values="predicted_effect_cf"
        )
        pivot.index = pivot.index.map(lambda t: f"{t[0]} Q{t[1]}")
        st.line_chart(pivot)
    else:
        st.info("Select at least one fund to compare.")

    st.markdown("---")
    st.subheader("Actual vs Predicted Churn Rate")
    fund = st.selectbox("Select fund", funds)
    if fund:
        cmp = causal_df[causal_df["Krankenkasse"]==fund].copy()
        cmp["Period"] = cmp.apply(lambda r: f"{int(r['Jahr'])} Q{int(r['Quartal'])}", axis=1)
        cmp = cmp.sort_values(["Jahr","Quartal"])
        chart = alt.Chart(cmp).transform_fold(
            ["churn_rel","predicted_effect_cf"], as_=["variable","value"]
        ).mark_line().encode(
            x="Period:N", y="value:Q", color="variable:N"
        )
        st.altair_chart(chart, use_container_width=True)
    else:
        st.info("Select a fund to visualize.")

# --- PREDICTIVE MODEL SEITE ---
elif page == "Predictive Model":
    run_page()

# --- SINGLE FUND PREDICTION SEITE ---
elif page == "Single Fund Prediction":
    st.title("Single Fund Prediction")
    st.info("Enter the values for a single fund to receive a churn prediction.")

    with st.form(key="fund_input_form"):
        beitrag   = st.number_input("Additional Contribution (%)", min_value=0.0, max_value=5.0, step=0.1)
        jahr      = st.selectbox("Year", [2021, 2022, 2023, 2024])
        risiko    = st.number_input("Morbidity Risk Factor", min_value=0.0, max_value=5.0, step=0.01)
        mitglieder= st.number_input("Number of Members", min_value=1)
        submit    = st.form_submit_button(label="Predict")

    if submit:
        if dv is None or model is None:
            st.error("Predictive model not loaded.")
        else:
            input_dict = {
                "Zusatzbeitrag": float(beitrag),
                "Jahr":          jahr,
                "Risikofaktor":  float(risiko),
                "Mitglieder":    int(mitglieder)
            }
            X = dv.transform([input_dict])
            y_pred = model.predict_proba(X)[0,1]
            st.success(f"Churn Risk: {y_pred:.2%}")
