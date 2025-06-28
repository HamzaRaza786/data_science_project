import streamlit as st
import pandas as pd
import altair as alt
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
        'Zusatzbeitrag': [1.3,    1.4,   1.2,   1.5,   1.4,    1.3],
        'churn_rel':    [0.002,  0.001, 0.004, 0.0025, 0.0012, 0.0041]
    })


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

    st.markdown("""
    This section shows estimated treatment effects of Zusatzbeitrag changes on churn 
    across different Krankenkassen and time periods, based on our causal model.
    """)

    # Mock causal effects
    causal_df = load_causal_data()

    # KPI cards
    avg_treatment_effect = causal_df['predicted_effect_cf'].mean()
    avg_churn_rate = causal_df['churn_rel'].mean()
    c1, c2 = st.columns(2)
    c1.metric("Avg Treatment Effect", f"{avg_treatment_effect:.4f}")
    c2.metric("Avg Churn Rate", f"{avg_churn_rate:.2%}")

    st.markdown("---")

    # Table of average treatment effects per fund
    st.subheader("Average Causal Effect per Fund")
    avg_effects_per_fund = causal_df.groupby(
        "Krankenkasse")['predicted_effect_cf'].mean().reset_index()
    st.dataframe(avg_effects_per_fund)

    st.markdown("---")

    # Comparison over time
    st.subheader("Causal Effects over Time")
    funds = causal_df['Krankenkasse'].unique().tolist()
    selected_funds = st.multiselect(
        "Select funds to visualize", funds, default=funds[:2])
    if selected_funds:
        plot_df = causal_df[causal_df['Krankenkasse'].isin(selected_funds)]
        pivot_causal = plot_df.pivot_table(
            index=['Jahr', 'Quartal'],
            columns='Krankenkasse',
            values='predicted_effect_cf'
        )
        pivot_causal.index = pivot_causal.index.map(
            lambda t: f"{t[0]} Q{t[1]}")
        st.line_chart(pivot_causal)
    else:
        st.info("Select at least one fund to compare.")

    st.markdown("---")

    st.subheader("Actual vs Predicted Churn Rate")

    funds = causal_df['Krankenkasse'].unique().tolist()
    selected_fund = st.selectbox("Select funds", funds)

    if selected_fund:
        compare_df = causal_df[causal_df['Krankenkasse']
                               == selected_fund].copy()
        compare_df['time'] = compare_df.apply(
            lambda r: f"{int(r['Jahr'])} Q{int(r['Quartal'])}", axis=1)
        compare_df = compare_df.sort_values(['Jahr', 'Quartal'])

        chart = alt.Chart(compare_df).transform_fold(
            ['churn_rel', 'predicted_effect_cf'],
            as_=['variable', 'value']
        ).mark_line().encode(
            x='time:N',
            y='value:Q',
            color='variable:N',
            strokeDash='Krankenkasse:N'
        )

        st.altair_chart(chart)
    else:
        st.info("Select at least one fund to visualize.")

# --- PREDICTIVE MODEL SEITE ---
elif page == "Predictive Model":
    run_page()
# --- EINZEL-EINGABE SEITE ---
elif page == "Single Fund Prediction":
    st.title("Single Fund Prediction")
    st.info("Enter the values for a single fund and receive a churn prediction.")

    with st.form(key='user_input_form'):
        beitrag = st.number_input(
            'Zusatzbeitrag (%)',   min_value=0.0, max_value=5.0, step=0.1)
        jahr = st.selectbox('Jahr',                 [2021, 2022, 2023, 2024])
        risiko = st.number_input('Morbiditätsfaktor',
                                 min_value=0.0, max_value=5.0, step=0.01)
        mitglieder = st.number_input('Mitgliederzahl',     min_value=1)
        submit_btn = st.form_submit_button(label='Predict')

    if submit_btn:
        # TODO: Hier echtes Modell laden/Ergebnis berechnen
        churn_risk = beitrag * 0.01 + risiko * 0.02
        st.success(f"Churn-Risiko (Dummy): {churn_risk:.2%}")
