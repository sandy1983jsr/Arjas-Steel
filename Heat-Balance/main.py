import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from heat_balance import *
from data_loader import *
from config import THRESHOLDS, DASHBOARD_COLORS

st.set_page_config(page_title="Hot Blast Stove Heat Balance", layout="wide")

st.title("🔥 Hot Blast Stove Heat Balance Dashboard")

# --- File Upload UI ---
st.sidebar.header("Step 1: Upload Your Cycle Data (.csv)")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, parse_dates=['timestamp'])
    st.success("Data uploaded successfully!")
else:
    st.warning("Please upload a CSV file to continue.")
    st.stop()

st.sidebar.header("What-If Analysis & Thresholds")
st.sidebar.markdown("Set thresholds to trigger alerts:")

eff_low = st.sidebar.slider("Efficiency Alert Threshold (%)", 40, 90, int(THRESHOLDS["efficiency_low"]*100))/100
shell_high = st.sidebar.slider("Shell Temp Alert (°C)", 80, 200, int(THRESHOLDS["shell_temp_high"]))
cycle_long = st.sidebar.slider("Long Cycle Duration (min)", 60, 300, int(THRESHOLDS["cycle_duration_long"]))

st.sidebar.header("What-If Inputs")
stove_select = st.sidebar.selectbox("Stove ID", sorted(df['stove_id'].unique()))
latest = get_latest_cycle(df, stove_select)

# What-if user inputs
st.markdown("## 🟧 Projections (Orange) – Current Cycle / What-If")
with st.form("whatif_form"):
    st.write("Modify values for what-if analysis:")
    m_fuel = st.number_input("Fuel Flow Rate (Nm³/hr)", value=float(latest["m_fuel"]), min_value=0.0)
    cv_fuel = st.number_input("Fuel Calorific Value (MJ/Nm³)", value=float(latest["cv_fuel"]), min_value=0.0)
    eta_comb = st.number_input("Combustion Efficiency", value=float(latest["eta_combustion"]), min_value=0.0, max_value=1.0)
    m_air = st.number_input("Blast Air Flow Rate (kg/hr)", value=float(latest["m_air"]), min_value=0.0)
    t_hot_blast = st.number_input("Hot Blast Temp (°C)", value=float(latest["t_hot_blast"]), min_value=0.0)
    t_ambient = st.number_input("Ambient Temp (°C)", value=float(latest["t_ambient"]), min_value=0.0)
    m_flue = st.number_input("Flue Gas Flow (kg/hr)", value=float(latest["m_flue"]), min_value=0.0)
    t_flue = st.number_input("Flue Gas Temp (°C)", value=float(latest["t_flue"]), min_value=0.0)
    t_ref = st.number_input("Flue Gas Ref Temp (°C)", value=float(latest["t_ref"]), min_value=0.0)
    k = st.number_input("Shell Conductivity (W/mK)", value=float(latest["k"]), min_value=0.0)
    A = st.number_input("Shell Area (m²)", value=float(latest["A"]), min_value=0.0)
    t_internal = st.number_input("Internal Temp (°C)", value=float(latest["t_internal"]), min_value=0.0)
    t_surface = st.number_input("Shell Surface Temp (°C)", value=float(latest["t_surface"]), min_value=0.0)
    d = st.number_input("Shell Thickness (m)", value=float(latest["d"]), min_value=0.01)
    eps = st.number_input("Shell Emissivity", value=float(latest["eps"]), min_value=0.0, max_value=1.0)
    sigma = float(latest["sigma"])
    m_air_comb = st.number_input("Combustion Air Flow (kg/hr)", value=float(latest["m_air_comb"]), min_value=0.0)
    t_air_comb = st.number_input("Combustion Air Temp (°C)", value=float(latest["t_air_comb"]), min_value=0.0)
    cp_air = float(latest["cp_air"])
    cp_flue = float(latest["cp_flue"])
    submitted = st.form_submit_button("Run What-If")

if submitted:
    Q_fuel, Q_air_comb = calculate_heat_input(m_fuel, cv_fuel, eta_comb, m_air_comb, cp_air, t_air_comb, t_ambient)
    Q_blast = calculate_heat_output(m_air, cp_air, t_hot_blast, t_ambient)
    Q_flue = calculate_flue_loss(m_flue, cp_flue, t_flue, t_ref)
    Q_shell = calculate_shell_loss(k, A, t_internal, t_surface, d, eps, sigma, t_ambient)
    eta_stove = calculate_efficiency(Q_blast, Q_fuel, Q_air_comb, Q_flue, Q_shell)
    color = DASHBOARD_COLORS["projection"]

    st.markdown(f"<h3 style='color:{color}'>Stove Efficiency: {eta_stove*100:.1f}%</h3>", unsafe_allow_html=True)

    result_df = pd.DataFrame({
        'Metric': ['Heat Input', 'Useful Heat', 'Flue Loss', 'Shell Loss', 'Efficiency'],
        'Value': [Q_fuel + Q_air_comb, Q_blast, Q_flue, Q_shell, eta_stove*100]
    })

    bar_chart = alt.Chart(result_df).mark_bar(color=color).encode(
        x=alt.X('Metric', title='Parameter'),
        y=alt.Y('Value', title='Value (kJ or %)'),
        tooltip=['Metric', 'Value']
    ).properties(
        width=600,
        height=300,
        title="What-If Results"
    )
    st.altair_chart(bar_chart, use_container_width=True)

    if eta_stove < eff_low:
        st.markdown(f"<span style='color:{DASHBOARD_COLORS['alert']}'>⚠️ Efficiency Below Threshold!</span>", unsafe_allow_html=True)
    if t_surface > shell_high:
        st.markdown(f"<span style='color:{DASHBOARD_COLORS['alert']}'>⚠️ Shell Overheat!</span>", unsafe_allow_html=True)
else:
    st.info("Run what-if to see updated projections.")

# Historical Dashboard (Grey)
st.markdown("## 🩶 Historical Cycles (Grey) – Trend & Breakdown")
hist = get_historical_data(df, stove_select)
Q_fuel_hist, Q_air_comb_hist, Q_blast_hist, Q_flue_hist, Q_shell_hist, eta_hist = [], [], [], [], [], []
for _, row in hist.iterrows():
    Qf, Qac = calculate_heat_input(row["m_fuel"], row["cv_fuel"], row["eta_combustion"], row["m_air_comb"], row["cp_air"], row["t_air_comb"], row["t_ambient"])
    Qb = calculate_heat_output(row["m_air"], row["cp_air"], row["t_hot_blast"], row["t_ambient"])
    Qfl = calculate_flue_loss(row["m_flue"], row["cp_flue"], row["t_flue"], row["t_ref"])
    Qsh = calculate_shell_loss(row["k"], row["A"], row["t_internal"], row["t_surface"], row["d"], row["eps"], row["sigma"], row["t_ambient"])
    et = calculate_efficiency(Qb, Qf, Qac, Qfl, Qsh)
    Q_fuel_hist.append(Qf)
    Q_air_comb_hist.append(Qac)
    Q_blast_hist.append(Qb)
    Q_flue_hist.append(Qfl)
    Q_shell_hist.append(Qsh)
    eta_hist.append(et)

hist["Efficiency"] = eta_hist
hist["Q_fuel"] = Q_fuel_hist
hist["Q_air_comb"] = Q_air_comb_hist
hist["Q_blast"] = Q_blast_hist
hist["Q_flue"] = Q_flue_hist
hist["Q_shell"] = Q_shell_hist

color_hist = DASHBOARD_COLORS["historical"]

eff_chart = alt.Chart(hist).mark_line(color=color_hist).encode(
    x=alt.X('timestamp:T', title='Cycle Timestamp'),
    y=alt.Y('Efficiency:Q', title='Stove Efficiency (fraction)'),
    tooltip=['timestamp', 'Efficiency']
).properties(
    width=800,
    height=350,
    title="Historical Stove Efficiency per Cycle"
)
st.altair_chart(eff_chart, use_container_width=True)

st.dataframe(hist[["timestamp", "Efficiency", "Q_blast", "Q_fuel", "Q_flue", "Q_shell"]])

# KPI Panel
st.markdown("## 📊 KPIs & Alerts")
st.metric("Latest Efficiency (%)", f"{hist['Efficiency'].iloc[-1]*100:.1f}")
st.metric("Fuel Consumption (Nm³)", f"{hist['m_fuel'].iloc[-1]:.1f}")
st.metric("Hot Blast Output per Fuel Unit", f"{hist['Q_blast'].iloc[-1]/hist['Q_fuel'].iloc[-1]:.2f}")
st.metric("Shell Temp (°C)", f"{hist['t_surface'].iloc[-1]:.1f}")

if hist["Efficiency"].iloc[-1] < eff_low:
    st.markdown(f"<span style='color:{DASHBOARD_COLORS['alert']}'>⚠️ Latest Cycle Efficiency Below Threshold!</span>", unsafe_allow_html=True)
if hist["t_surface"].iloc[-1] > shell_high:
    st.markdown(f"<span style='color:{DASHBOARD_COLORS['alert']}'>⚠️ Shell Overheat Alert!</span>", unsafe_allow_html=True)

# Condition Index
st.markdown("## 🟠 Condition Index (Chequered Brick Health)")
cond_index = get_condition_index(hist["Efficiency"].tolist())
st.metric("Condition Index (Rolling Mean)", f"{cond_index*100:.1f}")
