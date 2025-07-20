import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.linear_model import LinearRegression

STEFAN_BOLTZMANN = 5.67e-8  # W/m2.K4

# ---------- DATA INGESTION AND CALCULATION ----------

def read_data(file):
    df = pd.read_csv(file, parse_dates=['Timestamp'])
    df.sort_values('Timestamp', inplace=True)
    return df

def calculate_shell_heat_loss(df, zones, conductivity_map):
    results = []
    for zone in zones:
        temp_col = f"{zone}_Temp"
        thick_col = f"{zone}_Thickness"
        area_col = f"{zone}_Area"
        emiss_col = f"{zone}_Emissivity"
        # Conduction
        Q_cond = conductivity_map[zone] * df[area_col] * (df[temp_col] - df['Ambient_Temp']) / df[thick_col]
        # Radiation
        Q_rad = df[emiss_col] * STEFAN_BOLTZMANN * df[area_col] * (
            np.power(df[temp_col]+273.15, 4) - np.power(df['Ambient_Temp']+273.15, 4)
        )
        results.append(Q_cond + Q_rad)
    df['Shell_Heat_Loss_kW'] = np.sum(results, axis=0) / 1000  # W to kW
    return df

def calculate_offgas_heat_loss(df, gas_cp_map):
    Q_gas = 0
    for gas in gas_cp_map:
        flow_col = f"{gas}_Flow"  # Nm3/hr
        temp_col = "Offgas_Temp"  # °C
        cp = gas_cp_map[gas]      # kJ/Nm3.K
        Q_gas += df[flow_col] * cp * (df[temp_col] - 25) / 3600  # kW
    df['Offgas_Heat_Loss_kW'] = Q_gas
    return df

def build_heat_balance(df):
    df['Total_Heat_Loss_kW'] = df['Shell_Heat_Loss_kW'] + df['Offgas_Heat_Loss_kW']
    return df

def correlate_coke_heat(df):
    X = df[['Total_Heat_Loss_kW']]
    y = df['Coke_Rate']
    model = LinearRegression()
    model.fit(X, y)
    df['Predicted_Coke_Rate'] = model.predict(X)
    return model, df

def refractory_condition_index(df):
    df['HeatLoss_CokeRate_Ratio'] = df['Total_Heat_Loss_kW'] / df['Coke_Rate']
    idx = (df['HeatLoss_CokeRate_Ratio'] - df['HeatLoss_CokeRate_Ratio'].min()) / \
          (df['HeatLoss_CokeRate_Ratio'].max() - df['HeatLoss_CokeRate_Ratio'].min())
    df['Refractory_Index'] = idx
    return df

def linear_forecast(df, col, periods=24):
    x = np.arange(len(df))
    y = df[col].values
    coef = np.polyfit(x, y, 1)
    fut_x = np.arange(len(df), len(df) + periods)
    fut_y = coef[0] * fut_x + coef[1]
    fut_time = pd.date_range(df['Timestamp'].iloc[-1], periods=periods+1, freq='H')[1:]
    forecast_df = pd.DataFrame({'Timestamp': fut_time, col: fut_y})
    return forecast_df

# ---------- WHAT-IF AND OPTIMIZATION ----------

def simulate_scenario(shell_temp, thickness, coke_rate, fuel_rate, offgas_temp, co_flow):
    shell_area = 100
    refractory_conductivity = 2
    shell_heat_loss = refractory_conductivity * shell_area * (shell_temp - 30) / thickness / 1000
    offgas_heat_loss = co_flow * 1.04 * (offgas_temp - 30) / 3600
    total_heat_loss = shell_heat_loss + offgas_heat_loss
    heatloss_cokerate_ratio = total_heat_loss / coke_rate
    refractory_index = (heatloss_cokerate_ratio - 0.8) / (1.5 - 0.8)  # normalize (example)
    return shell_heat_loss, offgas_heat_loss, total_heat_loss, refractory_index

def optimize_scenario():
    best = None
    best_val = float('inf')
    for shell_temp in range(280, 320, 2):
        for thickness in np.arange(0.25, 0.40, 0.01):
            shell_heat_loss, _, total_heat_loss, _ = simulate_scenario(
                shell_temp, thickness, 470, 24, 225, 375
            )
            if total_heat_loss < best_val:
                best = (shell_temp, thickness, shell_heat_loss, total_heat_loss)
                best_val = total_heat_loss
    return best

# ---------- PLOTTING ----------

def plot_dashboard(df, forecast_df, col, title, ylabel):
    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(df['Timestamp'], df[col], color='grey', label=f'{title} (Historical)')
    ax.plot(forecast_df['Timestamp'], forecast_df[col], color='orange', label=f'{title} (Forecast)')
    ax.set_xlabel('Time')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid()
    st.pyplot(fig)

# ---------- MAIN STREAMLIT APP ----------

st.title("Blast Furnace Heat Balance & Predictive Dashboard - Arjas Steel")

st.markdown("""
**Upload CSV with columns like:**  
Timestamp, Ambient_Temp, [Zone]_Temp, [Zone]_Thickness, [Zone]_Area, [Zone]_Emissivity (e.g. Belly, Bosh, Hearth),  
Coke_Rate, Fuel_Injection, Offgas_Temp, CO_Flow, CO2_Flow, CH4_Flow, H2_Flow, N2_Flow (Nm3/hr)
""")

uploaded_file = st.file_uploader("Upload CSV Data", type=['csv'])

# --- Dynamic thresholds sidebar ---
st.sidebar.header("Alert Thresholds")
refractory_threshold = st.sidebar.slider("Refractory Index Alert Threshold", 0.0, 1.0, 0.85, 0.01)
heat_loss_threshold = st.sidebar.number_input("Total Heat Loss Threshold (kW)", min_value=0.0, value=175.0)
coke_rate_threshold = st.sidebar.number_input("Coke Rate Deviation Threshold (kg/thm)", min_value=0.0, value=20.0)

# --- What-If Analysis ---
st.header("What-If Analysis")
shell_temp = st.slider("Shell Temperature (°C)", 260, 350, 300)
thickness = st.slider("Refractory Thickness (m)", 0.20, 0.40, 0.32)
coke_rate = st.slider("Coke Rate (kg/thm)", 400, 550, 470)
fuel_rate = st.slider("Fuel Injection Rate (Nm³/hr)", 10, 40, 24)
offgas_temp = st.slider("Offgas Temperature (°C)", 180, 280, 225)
co_flow = st.slider("CO Flow (Nm³/hr)", 300, 500, 375)

shell_heat_loss, offgas_heat_loss, total_heat_loss, refractory_index = simulate_scenario(
    shell_temp, thickness, coke_rate, fuel_rate, offgas_temp, co_flow
)

st.write(f"**Shell Heat Loss:** {shell_heat_loss:.2f} kW")
st.write(f"**Offgas Heat Loss:** {offgas_heat_loss:.2f} kW")
st.write(f"**Total Heat Loss:** {total_heat_loss:.2f} kW")
st.write(f"**Refractory Condition Index:** {refractory_index:.2f}")

# --- Alerts based on thresholds ---
alert_msgs = []
if refractory_index > refractory_threshold:
    alert_msgs.append(f"⚠️ Refractory Index {refractory_index:.2f} > threshold {refractory_threshold}")
if total_heat_loss > heat_loss_threshold:
    alert_msgs.append(f"⚠️ Total Heat Loss {total_heat_loss:.2f} kW > threshold {heat_loss_threshold}")
if abs(coke_rate - 470) > coke_rate_threshold:
    alert_msgs.append(f"⚠️ Coke Rate deviation {abs(coke_rate-470):.2f} > threshold {coke_rate_threshold}")

if alert_msgs:
    for msg in alert_msgs:
        st.error(msg)
else:
    st.success("All parameters within thresholds.")

# --- Optimization scenario ---
if st.button("Find Optimal Scenario (Minimize Heat Loss)"):
    shell_temp_opt, thickness_opt, shell_heat_loss_opt, total_heat_loss_opt = optimize_scenario()
    st.write(f"**Optimal Shell Temp:** {shell_temp_opt} °C")
    st.write(f"**Optimal Thickness:** {thickness_opt:.2f} m")
    st.write(f"**Optimal Shell Heat Loss:** {shell_heat_loss_opt:.2f} kW")
    st.write(f"**Optimal Total Heat Loss:** {total_heat_loss_opt:.2f} kW")

# --- CSV Data Analysis and Dashboards ---
if uploaded_file:
    zones = ['Belly', 'Bosh', 'Hearth']
    conductivity_map = {"Belly":2.0, "Bosh":2.5, "Hearth":3.0}
    gas_cp_map = {"CO":1.04, "CO2":0.84, "CH4":2.22, "H2":3.42, "N2":1.04}
    
    df = read_data(uploaded_file)
    df = calculate_shell_heat_loss(df, zones, conductivity_map)
    df = calculate_offgas_heat_loss(df, gas_cp_map)
    df = build_heat_balance(df)
    _, df = correlate_coke_heat(df)
    df = refractory_condition_index(df)
    
    # Forecasts
    shell_forecast = linear_forecast(df, 'Shell_Heat_Loss_kW', periods=24)
    offgas_forecast = linear_forecast(df, 'Offgas_Heat_Loss_kW', periods=24)
    coke_forecast = linear_forecast(df, 'Coke_Rate', periods=24)
    index_forecast = linear_forecast(df, 'Refractory_Index', periods=24)
    
    st.header("Shell Heat Loss Dashboard")
    plot_dashboard(df, shell_forecast, 'Shell_Heat_Loss_kW', "Shell Heat Loss", "kW")
    st.header("Off-Gas Heat Loss Dashboard")
    plot_dashboard(df, offgas_forecast, 'Offgas_Heat_Loss_kW', "Off-Gas Heat Loss", "kW")
    st.header("Coke Rate Dashboard")
    plot_dashboard(df, coke_forecast, 'Coke_Rate', "Coke Rate", "kg/thm")
    st.header("Refractory Condition Index")
    plot_dashboard(df, index_forecast, 'Refractory_Index', "Refractory Condition Index", "Index (0-1)")
    
    st.subheader("Alerts & Thresholds (Historical Data)")
    alert_df = df[df['Refractory_Index'] > refractory_threshold]
    if not alert_df.empty:
        st.error(f"⚠️ High Refractory Condition Index: {len(alert_df)} recent entries above threshold! Recommend inspection.")
        st.dataframe(alert_df[['Timestamp','Refractory_Index','Shell_Heat_Loss_kW','Coke_Rate']])
    else:
        st.success("No critical refractory alerts in recent data.")
    
    st.subheader("Projected Data Table (Next 24 Hours)")
    st.dataframe(shell_forecast.rename(columns={'Timestamp':'Timestamp','Shell_Heat_Loss_kW':'Shell_Heat_Loss_kW'}))
else:
    st.info("Please upload CSV file to proceed.")

st.markdown("""
**Notes:**  
- All alert thresholds are user-settable in the sidebar.  
- Use sliders to conduct what-if analysis and see instant effect on key metrics and alerts.  
- Click 'Find Optimal Scenario' to identify settings minimizing heat loss (can be extended with more optimization logic).
- Expand zones, gases, or optimization routine as needed.
""")
