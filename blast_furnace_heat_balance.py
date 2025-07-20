import streamlit as st
import pandas as pd
import numpy as np
import time
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# --- Constants ---
STEFAN_BOLTZMANN = 5.67e-8  # W/m2.K4
areas = {'belly': 50, 'bosh': 40, 'hearth': 30}  # Example areas
emissivity = 0.85

specific_heats = {'CO': 1.04, 'CO2': 0.84, 'N2': 1.04, 'H2': 14.30, 'CH4': 2.22}
molecular_weights = {'CO': 28, 'CO2': 44, 'N2': 28, 'H2': 2, 'CH4': 16}
T_ref = 25 + 273.15

# --- Functions ---
def calc_conduction_heat_loss(k, A, T_inside, T_outside, d):
    return k * A * (T_inside - T_outside) / d

def calc_radiation_heat_loss(epsilon, A, T_shell, T_ambient):
    return epsilon * STEFAN_BOLTZMANN * A * (T_shell**4 - T_ambient**4)

def calc_shell_heat_loss(row):
    total_cond = 0
    total_rad = 0
    for zone in areas:
        T_shell = row[f'shell_temp_{zone}'] + 273.15
        T_ambient = row['ambient_temp'] + 273.15
        k = row[f'thermal_cond_{zone}']
        d = row[f'wall_thickness_{zone}']
        A = areas[zone]
        T_inside = T_shell + 150  # Adjust as per plant data
        cond = calc_conduction_heat_loss(k, A, T_inside, T_shell, d)
        rad = calc_radiation_heat_loss(emissivity, A, T_shell, T_ambient)
        total_cond += cond
        total_rad += rad
    return total_cond + total_rad

def calc_gas_heat_loss(row):
    T_gas = row['off_gas_temp'] + 273.15
    total_heat = 0
    total_flow = row['off_gas_flow']
    for gas, cp in specific_heats.items():
        frac = row[gas] / 100
        flow_gas = total_flow * frac
        mol_gas = flow_gas / 22.4
        mass_gas = mol_gas * molecular_weights[gas] / 1000
        heat = mass_gas * cp * (T_gas - T_ref)
        total_heat += heat
    return total_heat / 3600

def process_data(df):
    df['shell_heat_loss_kW'] = df.apply(calc_shell_heat_loss, axis=1) / 1000
    df['off_gas_heat_loss_kW'] = df.apply(calc_gas_heat_loss, axis=1)
    df['heat_loss_total_kW'] = df['shell_heat_loss_kW'] + df['off_gas_heat_loss_kW']
    df['heat_loss_per_ton_coke'] = df['heat_loss_total_kW'] / (df['coke_rate'] + 1e-6)
    df['heat_loss_coke_ratio'] = df['heat_loss_total_kW'] / df['coke_rate']
    return df

def ml_prediction(df):
    features = [
        'shell_heat_loss_kW', 'off_gas_heat_loss_kW',
        'fuel_injection_rate', 'hot_metal_temp', 'hot_metal_qty'
    ]
    X = df[features]
    y = df['coke_rate']
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X, y)
    df['coke_rate_pred'] = rf.predict(X)
    df['refractory_condition_index'] = 1 - (df['shell_heat_loss_kW'] - df['shell_heat_loss_kW'].min()) / (df['shell_heat_loss_kW'].max() - df['shell_heat_loss_kW'].min())
    return df

# --- STREAMLIT APP ---
st.set_page_config(page_title="Blast Furnace Heat Balance Dashboard", layout="wide")
st.title("Blast Furnace Heat Balance - Arjas Steel")

# --- Data Upload or Real-Time Simulation ---
data_source = st.sidebar.selectbox("Choose Data Source", ["Upload CSV", "Simulate Real-Time Data"])

if data_source == "Upload CSV":
    uploaded_file = st.file_uploader("Upload blast furnace data (.csv)", type="csv")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        df = process_data(df)
        df = ml_prediction(df)
else:
    # Simulate real-time data (random or from a preset file)
    st.sidebar.write("Simulating real-time data... (Demo purpose)")
    df = pd.read_csv('blast_furnace_data.csv')  # You can replace with OPC/PLC integration
    df = process_data(df)
    df = ml_prediction(df)
    # For demo, just show last N rows, simulating stream
    df = df.tail(100)

# --- Dashboard ---
if 'df' in locals():
    tab1, tab2, tab3, tab4 = st.tabs(["Heat Loss Trend", "Coke Rate", "Heat Loss/Coke Ratio", "Refractory Condition"])
    
    with tab1:
        st.subheader("Heat Loss Trend")
        st.line_chart(df[['shell_heat_loss_kW', 'off_gas_heat_loss_kW']])
    
    with tab2:
        st.subheader("Coke Rate: Actual vs Predicted")
        st.line_chart(df[['coke_rate', 'coke_rate_pred']])
    
    with tab3:
        st.subheader("Heat Loss / Coke Rate Ratio")
        st.line_chart(df['heat_loss_coke_ratio'])
    
    with tab4:
        st.subheader("Refractory Condition Index")
        st.line_chart(df['refractory_condition_index'])
    
    st.markdown("### Data Table (Last 10 rows)")
    st.dataframe(df.tail(10))

    st.success("Dashboard generated successfully!")
else:
    st.info("Please upload a CSV file or select simulation to view dashboard.")

# --- Real-Time PLC/DCS Integration ---
# To integrate real-time industrial data, replace the data loading logic above with code that pulls from OPC-UA, MQTT, or your plant's DCS/PLC historian.
# Example for OPC-UA (requires 'opcua' python package):
# from opcua import Client
# client = Client("opc.tcp://your_plc_ip:port")
# client.connect()
# data = client.get_node("ns=2;s=Channel1.Device1.Tag1").get_value()
