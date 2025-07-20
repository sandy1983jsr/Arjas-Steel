import numpy as np

def calculate_heat_input(m_fuel, cv_fuel, eta_combustion, m_air_comb=None, cp_air=1.005, t_air_comb=None, t_ambient=None):
    """Calculate total heat input (fuel + combustion air) in kJ."""
    Q_fuel = m_fuel * cv_fuel * eta_combustion  # Fuel heat input
    Q_air_comb = 0
    if m_air_comb is not None and t_air_comb is not None and t_ambient is not None:
        Q_air_comb = m_air_comb * cp_air * (t_air_comb - t_ambient)
    return Q_fuel, Q_air_comb

def calculate_heat_output(m_air, cp_air, t_hot_blast, t_ambient):
    """Calculate useful heat delivered to hot blast in kJ."""
    Q_blast = m_air * cp_air * (t_hot_blast - t_ambient)
    return Q_blast

def calculate_flue_loss(m_flue, cp_flue, t_flue, t_ref):
    """Calculate heat loss via flue gases in kJ."""
    Q_flue = m_flue * cp_flue * (t_flue - t_ref)
    return Q_flue

def calculate_shell_loss(k, A, t_internal, t_surface, d, eps, sigma, t_ambient):
    """Calculate heat loss via shell (conduction + radiation)."""
    Q_shell_cond = k * A * (t_internal - t_surface) / d
    Q_shell_rad = eps * sigma * A * (np.power(t_surface, 4) - np.power(t_ambient, 4))
    return Q_shell_cond + Q_shell_rad

def calculate_efficiency(Q_blast, Q_fuel, Q_air_comb, Q_flue=None, Q_shell=None):
    """Calculate efficiency, optionally including loss terms."""
    if Q_flue is not None and Q_shell is not None:
        eta_stove = 1 - (Q_flue + Q_shell) / (Q_fuel + Q_air_comb)
    else:
        eta_stove = Q_blast / (Q_fuel + Q_air_comb)
    return eta_stove

def get_condition_index(efficiency_list, window=20):
    """Trend index for chequered brick health: rolling mean of efficiency."""
    if len(efficiency_list) < window:
        return np.mean(efficiency_list)
    return np.mean(efficiency_list[-window:])
