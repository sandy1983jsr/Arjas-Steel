import pandas as pd

def load_cycle_data(csv_path):
    """
    Load stove cycle data from CSV.
    Expected columns:
    - cycle_id, stove_id, timestamp, m_fuel, cv_fuel, eta_combustion,
      m_air, cp_air, t_hot_blast, t_ambient,
      m_flue, cp_flue, t_flue, t_ref,
      k, A, t_internal, t_surface, d, eps, sigma,
      m_air_comb, t_air_comb
    """
    df = pd.read_csv(csv_path, parse_dates=['timestamp'])
    return df

def get_historical_data(df, stove_id=None):
    if stove_id is not None:
        return df[df['stove_id'] == stove_id]
    return df

def get_latest_cycle(df, stove_id):
    stove_cycles = df[df['stove_id'] == stove_id]
    return stove_cycles.sort_values('timestamp').iloc[-1]
