import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.linear_model import LinearRegression

# ---------------------------
# FUNCTIONS
# ---------------------------

def read_data(file):
    """
    Reads CSV data uploaded by user.
    Expected columns: ['Timestamp', 'Shell_Temp', 'Offgas_Temp', 'Coke_Rate', 'Refractory_Thickness']
    """
    df = pd.read_csv(file, parse_dates=['Timestamp'])
    df.sort_values('Timestamp', inplace=True)
    return df

def calculate_heat_loss(df, shell_area=100, offgas_flow=1000, shell_emissivity=0.8):
    """
    Calculates heat loss from shell and offgas.
    - shell_area: m2 (default typical value, adjust as per plant)
    - offgas_flow: Nm3/hr (default, adjust as per plant)
    - shell_emissivity: Emissivity factor (default, typical 0.8)
    """
    # Constants
    STEEL_THERMAL_CONDUCTIVITY = 43  # W/mK, typical steel
    REFRACTORY_THERMAL_CONDUCTIVITY = 2  # W/mK, typical refractory
    GAS_SPECIFIC_HEAT = 1.02  # kJ/Nm3.K

    # Heat loss from shell (simplified as Q = k*A*ΔT/thickness)
    df['Shell_Heat_Loss_kW'] = (
        REFRACTORY_THERMAL_CONDUCTIVITY * shell_area *
        (df['Shell_Temp'] - 35) / df['Refractory_Thickness']
    ) / 1000  # W to kW

    # Heat loss via offgas
    df['Offgas_Heat_Loss_kW'] = (
        offgas_flow * GAS_SPECIFIC_HEAT * (df['Offgas_Temp'] - 35) / 3600  # per hour to per second
    )

    # Total Heat Loss
    df['Total_Heat_Loss_kW'] = df['Shell_Heat_Loss_kW'] + df['Offgas_Heat_Loss_kW']

    return df

def correlate_heat_loss_with_coke_rate(df):
    """
    Fits a regression model between heat loss and coke rate.
    Returns: Regression model, projected coke rate (future)
    """
    X = df[['Total_Heat_Loss_kW']]
    y = df['Coke_Rate']
    model = LinearRegression()
    model.fit(X, y)
    df['Predicted_Coke_Rate'] = model.predict(X)
    return model, df

def project_future(df, periods=10):
    """
    Projects heat loss and coke rate for next 'periods' time steps.
    Uses linear trend based on last N points.
    """
    # Linear trend on heat loss
    last = df.tail(15)
    time_numeric = np.arange(len(last))
    trend = np.polyfit(time_numeric, last['Total_Heat_Loss_kW'], 1)
    future_heat_loss = [last['Total_Heat_Loss_kW'].values[-1] + trend[0]*(i+1) for i in range(periods)]
    # Use model to predict coke rate
    model = LinearRegression()
    model.fit(df[['Total_Heat_Loss_kW']], df['Coke_Rate'])
    future_coke_rate = model.predict(np.array(future_heat_loss).reshape(-1,1))
    future_time = pd.date_range(df['Timestamp'].iloc[-1], periods=periods+1, freq='H')[1:]
    future_df = pd.DataFrame({
        'Timestamp': future_time,
        'Total_Heat_Loss_kW': future_heat_loss,
        'Predicted_Coke_Rate': future_coke_rate
    })
    return future_df

def plot_heat_loss_dashboard(df, future_df):
    """
    Plots heat loss graph, historical (grey) and projected (orange).
    """
    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(df['Timestamp'], df['Total_Heat_Loss_kW'], color='grey', label='Heat Loss (Historical)')
    ax.plot(future_df['Timestamp'], future_df['Total_Heat_Loss_kW'], color='orange', label='Heat Loss (Projected)')
    ax.set_xlabel('Time')
    ax.set_ylabel('Heat Loss (kW)')
    ax.legend()
    ax.set_title('Blast Furnace Heat Loss')
    ax.grid()
    st.pyplot(fig)

def plot_coke_rate_dashboard(df, future_df):
    """
    Plots coke rate graph, historical (grey) and projected (orange).
    """
    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(df['Timestamp'], df['Coke_Rate'], color='grey', label='Coke Rate (Historical)')
    ax.plot(future_df['Timestamp'], future_df['Predicted_Coke_Rate'], color='orange', label='Coke Rate (Projected)')
    ax.set_xlabel('Time')
    ax.set_ylabel('Coke Rate (kg/thm)')
    ax.legend()
    ax.set_title('Blast Furnace Coke Rate')
    ax.grid()
    st.pyplot(fig)

# ---------------------------
# STREAMLIT APP
# ---------------------------

st.title('Blast Furnace Heat Balance Dashboard - Arjas Steel')

st.markdown("""
Upload your CSV file containing the following columns:
- Timestamp (date/time)
- Shell_Temp (C)
- Offgas_Temp (C)
- Coke_Rate (kg/thm)
- Refractory_Thickness (m)
""")

uploaded_file = st.file_uploader("Upload CSV data", type=['csv'])

if uploaded_file:
    df = read_data(uploaded_file)
    df = calculate_heat_loss(df)
    _, df = correlate_heat_loss_with_coke_rate(df)
    future_df = project_future(df, periods=24)
    
    st.header("Heat Loss Dashboard")
    plot_heat_loss_dashboard(df, future_df)
    
    st.header("Coke Rate Dashboard")
    plot_coke_rate_dashboard(df, future_df)

    st.subheader('Projected Data Table')
    st.dataframe(future_df)
else:
    st.info("Please upload CSV file to proceed.")

st.markdown("""
**Note:** For DCS connectivity, implement a data ingestion layer that fetches data from your DCS historian and formats it as per the required columns. This code can easily be adapted for such real-time input.
""")
