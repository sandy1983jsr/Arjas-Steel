import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from blend_model import train_predict_model, predict_coke_properties
from optimizer import optimize_blend
from data_loader import load_coal_db, load_blend_db
from config import ORANGE, DEFAULT_CONSTRAINTS

st.set_page_config(page_title="🟧 Coal Blend Optimization", layout="wide")

st.title("🟧 Coal Blend Optimization Dashboard")

st.sidebar.header("Step 1: Upload Coal Property Database (.csv)")
coal_file = st.sidebar.file_uploader("Coal Properties CSV", type=["csv"])
st.sidebar.header("Step 2: Upload Historical Blend/Coke Data (.csv)")
blend_file = st.sidebar.file_uploader("Blend Lab Data CSV", type=["csv"])

if not (coal_file and blend_file):
    st.warning("Please upload both the coal property and blend lab CSVs to proceed.")
    st.stop()

coal_df = load_coal_db(coal_file)
blend_df = load_blend_db(blend_file, coal_df)

st.success("Files uploaded! Model ready to train.")

st.header("Train Prediction Model")
model, blend_columns, coke_columns = train_predict_model(blend_df)
st.success("Model trained. Ready for optimization and simulation.")

st.header("Set Quality Constraints and What-If Inputs")
with st.form("constraints_form"):
    min_csr = st.number_input("Min CSR", value=DEFAULT_CONSTRAINTS['CSR'])
    max_cri = st.number_input("Max CRI", value=DEFAULT_CONSTRAINTS['CRI'])
    max_ash = st.number_input("Max Ash (%)", value=DEFAULT_CONSTRAINTS['Ash'])
    min_yield = st.number_input("Min Coke Yield (%)", value=DEFAULT_CONSTRAINTS['Yield'])
    blend_cost = coal_df.set_index('Coal')['Cost per ton (INR)'].to_dict()
    avail_limits = {}
    st.write("Set max availability for each coal (leave blank for unlimited):")
    for c in coal_df['Coal']:
        avail_limits[c] = st.number_input(f"Max {c} (%)", min_value=0.0, max_value=100.0, value=100.0)
    submitted = st.form_submit_button("Run Optimization")

if submitted:
    constraints = {
        'CSR': min_csr, 'CRI': max_cri, 'Ash': max_ash, 'Yield': min_yield
    }
    limits = {k: v/100 for k,v in avail_limits.items()}
    result = optimize_blend(model, coal_df, blend_columns, coke_columns, constraints, limits)
    st.markdown(f"<h3 style='color:{ORANGE}'>Optimal Blend Recommendation</h3>", unsafe_allow_html=True)
    blend_df_result = pd.DataFrame({'Coal': list(result['blend'].keys()), 'Proportion (%)': [v*100 for v in result['blend'].values()]})
    st.dataframe(blend_df_result, use_container_width=True)
    st.markdown(f"**Blend Cost:** ₹ {result['cost']:.2f} / ton")
    pred_df = pd.DataFrame({'Parameter': coke_columns, 'Predicted Value': [result['quality'][k] for k in coke_columns]})
    quality_chart = alt.Chart(pred_df).mark_bar(color=ORANGE).encode(
        x=alt.X('Parameter', title='Coke Quality Parameter'),
        y=alt.Y('Predicted Value', title='Value'),
        tooltip=['Parameter', 'Predicted Value']
    ).properties(title="Predicted Coke Quality")
    st.altair_chart(quality_chart, use_container_width=True)

    st.header("Sensitivity What-If Simulator")
    coal_select = st.selectbox("Select coal to simulate cost/availability change", coal_df['Coal'])
    new_cost = st.number_input(f"New Cost for {coal_select} (INR/ton)", value=blend_cost[coal_select])
    new_limit = st.number_input(f"New Max Availability for {coal_select} (%)", value=avail_limits[coal_select])
    if st.button("Run What-If"):
        coal_df_sim = coal_df.copy()
        coal_df_sim.loc[coal_df_sim['Coal']==coal_select, 'Cost per ton (INR)'] = new_cost
        sim_limits = limits.copy()
        sim_limits[coal_select] = new_limit/100
        sim_result = optimize_blend(model, coal_df_sim, blend_columns, coke_columns, constraints, sim_limits)
        st.markdown(f"<h4 style='color:{ORANGE}'>Simulated Blend & Quality</h4>", unsafe_allow_html=True)
        sim_blend_df = pd.DataFrame({'Coal': list(sim_result['blend'].keys()), 'Proportion (%)': [v*100 for v in sim_result['blend'].values()]})
        st.dataframe(sim_blend_df, use_container_width=True)
        st.markdown(f"**Simulated Blend Cost:** ₹ {sim_result['cost']:.2f} / ton")
        sim_pred_df = pd.DataFrame({'Parameter': coke_columns, 'Predicted Value': [sim_result['quality'][k] for k in coke_columns]})
        sim_chart = alt.Chart(sim_pred_df).mark_bar(color=ORANGE).encode(
            x=alt.X('Parameter', title='Coke Quality Parameter'),
            y=alt.Y('Predicted Value', title='Value'),
            tooltip=['Parameter', 'Predicted Value']
        ).properties(title="Simulated Coke Quality")
        st.altair_chart(sim_chart, use_container_width=True)

st.header("Historical Model Performance")
for k in coke_columns:
    perf_chart = alt.Chart(blend_df).mark_circle(color=ORANGE).encode(
        x=alt.X('blend_id:N', title='Blend Sample'),
        y=alt.Y(k, title=f'{k} (actual)'),
        tooltip=['blend_id', k]
    ).properties(
        width=600,
        height=300,
        title=f"Actual {k} in Historical Blends"
    )
    st.altair_chart(perf_chart, use_container_width=True)
