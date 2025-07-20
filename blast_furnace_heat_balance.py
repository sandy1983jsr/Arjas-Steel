import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from prophet import Prophet

# ----------- CONSTANTS -----------
STEFAN_BOLTZMANN = 5.67e-8  # W/m2.K4

# ----------- FUNCTIONS ------------

def read_data(file):
    df = pd.read_csv(file, parse_dates=['Timestamp'])
    df.sort_values('Timestamp', inplace=True)
    return df

def calculate_shell_heat_loss(df, zones, conductivity_map):
    """
    Calculate shell heat loss for each zone (conduction + radiation).
    zones: list of zone names
    conductivity_map: dict with zone: conductivity
    """
    results = []
    for zone in zones:
        temp_col = f"{zone}_Temp"
        thick_col = f"{zone}_Thickness"
        area_col = f"{zone}_Area"
        emiss_col = f"{zone}_Emissivity"
        # conduction
        Q_cond = conductivity_map[zone] * df[area_col] * (df[temp_col] - df['Ambient_Temp']) / df[thick_col]
        # radiation
        Q_rad = df[emiss_col] * STEFAN_BOLTZMANN * df[area_col] * (
            np.power(df[temp_col]+273.15, 4) - np.power(df['Ambient_Temp']+273.15, 4)
        )
        results.append(Q_cond + Q_rad)
    df['Shell_Heat_Loss_kW'] = np.sum(results, axis=0) / 1000  # W to kW
    return df

def calculate_offgas_heat_loss(df, gas_cp_map):
    """
    Calculate sensible heat loss in off-gas (sum across components).
    """
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
    """
    Composite indicator: high heat loss/coke rate ratio = possible refractory degradation.
    """
    df['HeatLoss_CokeRate_Ratio'] = df['Total_Heat_Loss_kW'] / df['Coke_Rate']
    # simple normalization
    idx = (df['HeatLoss_CokeRate_Ratio'] - df['HeatLoss_CokeRate_Ratio'].min()) / \
          (df['HeatLoss_CokeRate_Ratio'].max() - df['HeatLoss_CokeRate_Ratio'].min())
    df['Refractory_Index'] = idx
    return df

def prophet_forecast(df, col, periods=24):
    """
    Forecast future values using Prophet.
    """
    fdf = df[['Timestamp', col]].rename(columns={'Timestamp':'ds', col:'y'})
    m = Prophet()
    m.fit(fdf)
    future = m.make_future_dataframe(periods=periods, freq='H')
    forecast = m.predict(future)
    return forecast[['ds', 'yhat']].tail(periods)

def plot_dashboard(df, forecast_df, col, title, ylabel):
    fig, ax = plt.subplots(figsize=(10,5))
    # Historical
    ax.plot(df['Timestamp'], df[col], color='grey', label=f'{title} (Historical)')
    # Projected
    ax.plot(forecast_df['ds'], forecast_df['yhat'], color='orange', label=f'{title} (Forecast)')
    ax.set_xlabel('Time')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid()
    st.pyplot(fig)

# ----------- STREAMLIT UI ------------

st.title("Blast Furnace Heat Balance & Predictive Dashboard - Arjas Steel")
st.markdown("""
**Upload CSV with columns like:**
- Timestamp, Ambient_Temp
- [Zone]_Temp, [Zone]_Thickness, [Zone]_Area, [Zone]_Emissivity (for each zone, e.g. belly, bosh, hearth)
- Coke_Rate, Fuel_Injection, Offgas_Temp, CO_Flow, CO2_Flow, CH4_Flow, H2_Flow, N2_Flow (Nm3/hr)
""")

uploaded_file = st.file_uploader("Upload CSV Data", type=['csv'])
if uploaded_file:
    # Zone segmentation and parameters
    zones = ['Belly', 'Bosh', 'Hearth']  # customize as per your data
    conductivity_map = {"Belly":2.0, "Bosh":2.5, "Hearth":3.0}  # W/mK
    gas_cp_map = {"CO":1.04, "CO2":0.84, "CH4":2.22, "H2":3.42, "N2":1.04}  # kJ/Nm3.K
    
    df = read_data(uploaded_file)
    df = calculate_shell_heat_loss(df, zones, conductivity_map)
    df = calculate_offgas_heat_loss(df, gas_cp_map)
    df = build_heat_balance(df)
    _, df = correlate_coke_heat(df)
    df = refractory_condition_index(df)
    
    # Forecasts
    shell_forecast = prophet_forecast(df, 'Shell_Heat_Loss_kW', periods=24)
    offgas_forecast = prophet_forecast(df, 'Offgas_Heat_Loss_kW', periods=24)
    coke_forecast = prophet_forecast(df, 'Coke_Rate', periods=24)
    index_forecast = prophet_forecast(df, 'Refractory_Index', periods=24)
    
    # Dashboards
    st.header("Shell Heat Loss Dashboard")
    plot_dashboard(df, shell_forecast, 'Shell_Heat_Loss_kW', "Shell Heat Loss", "kW")
    st.header("Off-Gas Heat Loss Dashboard")
    plot_dashboard(df, offgas_forecast, 'Offgas_Heat_Loss_kW', "Off-Gas Heat Loss", "kW")
    st.header("Coke Rate Dashboard")
    plot_dashboard(df, coke_forecast, 'Coke_Rate', "Coke Rate", "kg/thm")
    st.header("Refractory Condition Index")
    plot_dashboard(df, index_forecast, 'Refractory_Index', "Refractory Condition Index", "Index (0-1)")
    
    st.subheader("Alerts & Thresholds")
    alert_df = df[df['Refractory_Index'] > 0.85]
    if not alert_df.empty:
        st.error(f"⚠️ High Refractory Condition Index: {len(alert_df)} recent entries above threshold! Recommend inspection.")
        st.dataframe(alert_df[['Timestamp','Refractory_Index','Shell_Heat_Loss_kW','Coke_Rate']])
    else:
        st.success("No critical refractory alerts in recent data.")
    
    st.subheader("Projected Data Table (Next 24 Hours)")
    st.dataframe(shell_forecast.rename(columns={'ds':'Timestamp','yhat':'Shell_Heat_Loss_kW'}))
else:
    st.info("Please upload CSV file to proceed.")

st.markdown("""
**Note:** For real-time DCS/OPC UA integration, replace the CSV section with appropriate data feed logic.  
**ML models (Random Forest, LSTM) can be plugged in for more advanced predictions as data volume increases.**
""")
