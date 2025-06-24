import streamlit as st
import pandas as pd
import numpy as np
from ml_churn_code import run_page

st.set_page_config(page_title="GKV Churn Dashboard", layout="wide")

# --- DATENLADE-FUNKTION ---
@st.cache_data
def load_data():
    # TODO: Hier später mit echten Daten ersetzen:
    # return pd.read_csv("data/merged_gkv.csv")
    return pd.DataFrame({
        'Krankenkasse': ['AOK', 'TK', 'BKK', 'AOK', 'TK', 'BKK'],
        'Jahr':         [2023,   2023,  2023, 2024, 2024,  2024],
        'Quartal':      [1,      2,     3,    1,    2,     3],
        'Mitglieder':   [150000, 120000, 80000, 152000, 123000, 82000],
        'Zusatzbeitrag':[1.3,    1.4,   1.2,   1.5,   1.4,    1.3],
        'churn_rel':    [0.002,  0.001, 0.004, 0.0025,0.0012, 0.0041]
    })

df = load_data()

# --- SIDEBAR NAVIGATION ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", [
    "Start",
    "Causal Effects",
    "Predictive Model",
    # "Batch Upload",
    "Single Fund Prediction"
])
st.sidebar.markdown("---")
st.sidebar.write("GKV Churn")

# --- START SEITE ---
if page == "Start":
    st.title("GKV Churn Dashboard")
    st.markdown("""
    Analyze the impact of additional contributions on member churn. 
    **Navigation:**  
    - **Causal Effects**: Causal Analysis with Difference-in-Differences  
    - **Predictive Model**: Churn Prediction
    - **Single Fund Prediction**: Einzel-Vorhersage  
    """)
    st.dataframe(df)

# --- CAUSAL EFFECTS SEITE ---
elif page == "Causal Effects":
    st.title("Causal Effects (Causal Analysis)")
    st.info("Charts and tables for the DiD analysis appear here.")

    # --- KPI-CARDS ---
    avg_treatment_effect = np.nan  # TODO: echten Treatment-Effekt einfügen
    avg_churn_rate       = df['churn_rel'].mean()
    model_accuracy       = np.nan  # TODO: echte Modell-Accuracy
    c1, c2, c3 = st.columns(3)
    c1.metric("Avg Treatment Effect", f"{avg_treatment_effect}")
    c2.metric("Avg Churn Rate", f"{avg_churn_rate:.2%}")
    c3.metric("Model Accuracy", f"{model_accuracy}")

    st.markdown("---")

    # --- Vergleichsansicht ---
    st.subheader("Compare Churn Rate Across Funds")
    funds   = df['Krankenkasse'].unique().tolist()
    compare = st.multiselect("Select funds to compare", funds, default=funds[:2])
    if compare:
        comp_df = df[df['Krankenkasse'].isin(compare)]
        chart_data = comp_df.pivot_table(
            index=['Jahr','Quartal'],
            columns='Krankenkasse',
            values='churn_rel'
        )
        st.line_chart(chart_data)
    else:
        st.info("Select at least two funds to compare.")

    st.markdown("---")

    # --- Wettbewerber-Beiträge ---
    st.subheader("Competitor Contribution Rates")
    contrib_data = df.pivot_table(
    index=['Jahr','Quartal'],
    columns='Krankenkasse',
    values='Zusatzbeitrag'
)
    contrib_data.index = contrib_data.index.map(lambda t: f"{t[0]} Q{t[1]}")
    st.line_chart(contrib_data)

# --- PREDICTIVE MODEL SEITE ---
elif page == "Predictive Model":
    run_page()
# --- EINZEL-EINGABE SEITE ---
elif page == "Single Fund Prediction":
    st.title("Single Fund Prediction")
    st.info("Enter the values for a single fund and receive a churn prediction.")

    with st.form(key='user_input_form'):
        beitrag    = st.number_input('Zusatzbeitrag (%)',   min_value=0.0, max_value=5.0, step=0.1)
        jahr       = st.selectbox('Jahr',                 [2021,2022,2023,2024])
        risiko     = st.number_input('Morbiditätsfaktor',  min_value=0.0, max_value=5.0, step=0.01)
        mitglieder = st.number_input('Mitgliederzahl',     min_value=1)
        submit_btn = st.form_submit_button(label='Predict')

    if submit_btn:
        # TODO: Hier echtes Modell laden/Ergebnis berechnen
        churn_risk = beitrag * 0.01 + risiko * 0.02
        st.success(f"Churn-Risiko (Dummy): {churn_risk:.2%}")