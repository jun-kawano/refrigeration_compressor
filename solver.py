import numpy as np
import CoolProp.CoolProp as CP
from scipy.integrate import solve_ivp
from config import (freq, T_suc, fluid, P_c, R_gas, T_cil, fator_esc_reverso, custom_rtol, custom_atol)


def simular_condicao(P_suc_target, nome_condicao, delta_P_max, num_ciclos, compressor, solver_method='RK23'):
    print(f"\n--- Iniciando simulação da {nome_condicao} (solve_ivp) ---")
    t_stop = num_ciclos * (1 / freq)

    # inlet Enthalpy
    h_suc_in = CP.PropsSI('H', 'T', T_suc, 'P', P_suc_target, fluid)

    # initial Conditions
    T0 = T_suc
    vol0 = compressor.volume(0.0)
    m0 = vol0 * CP.PropsSI('D', 'T', T_suc, 'P', P_suc_target, fluid)

    # Y = [T, m, y_suc, v_suc, y_des, v_des]
    Y0 = [T0, m0, 0.0, 0.0, 0.0, 0.0]

    # ---------------------------------------------------------
    # ODE function
    # ---------------------------------------------------------
    def compressor_odes(t, Y):
        T_i, m_i, y_suc_i, v_suc_i, y_des_i, v_des_i = Y

        # state protections to avoid problems with over stepping
        T_i = max(T_i, 150.0)
        m_i = max(m_i, 1e-10)

        V_i = compressor.volume(t)
        rho_i = m_i / V_i

        # thermodynamic props (with fallback to ideal gas law)
        try:
            P_i = CP.PropsSI('P', 'T', T_i, 'D', rho_i, fluid)
            if np.isnan(P_i) or P_i < 0:
                P_i = rho_i * R_gas * T_i
        except ValueError:
            P_i = rho_i * R_gas * T_i

        Cv_i = CP.PropsSI('CVMASS', 'T', T_i, 'D', rho_i, fluid)
        dP_dT_i = CP.PropsSI('d(P)/d(T)|D', 'T', T_i, 'D', rho_i, fluid)
        h_i = CP.PropsSI('H', 'T', T_i, 'D', rho_i, fluid)
        k_i = CP.PropsSI('CP0MASS', 'T', T_i, 'D', rho_i, fluid) / Cv_i

        # --- SUCTION VALVE DYNAMICS ---
        delta_P_suc = P_suc_target - P_i
        dv_dt_suc, F_gas_suc, a_ef_s = compressor.suction_valve.get_acceleration(delta_P_suc, y_suc_i, v_suc_i)
        a_ee_s = compressor.suction_valve.Aee(y_suc_i)

        # --- DISCHARGE VALVE DYNAMICS ---
        delta_P_des = P_i - P_c
        dv_dt_des, F_gas_des, a_ef_d = compressor.discharge_valve.get_acceleration(delta_P_des, y_des_i, v_des_i)
        a_ee_d = compressor.discharge_valve.Aee(y_des_i)

        # --- MASS BALANCE ---
        dm_dt_suc = compressor.suction_valve.m_dot_valve(P_suc_target, P_i, T_suc, k_i, a_ee_s, R_gas,
                                                         fator_esc_reverso)
        dm_dt_des = compressor.discharge_valve.m_dot_valve(P_i, P_c, T_i, k_i, a_ee_d, R_gas, fator_esc_reverso)
        dm_dt = dm_dt_suc - dm_dt_des

        # --- ENERGY BALANCE ---
        # Central difference for dV/dt (smooth and accurate)
        dt_vol = 1e-6
        dV_dt = (compressor.volume(t + dt_vol) - compressor.volume(max(0, t - dt_vol))) / (2 * dt_vol)

        H_convec = compressor.h_coef(T_i, rho_i, fluid)
        A_convec = compressor.area_convec(t)
        Q_dot = H_convec * A_convec * (T_cil - T_i)
        H_flux = (dm_dt_suc * h_suc_in) - (dm_dt_des * h_i)

        numerador = Q_dot + H_flux - (h_i * dm_dt) - (T_i * dP_dT_i * (dV_dt - (dm_dt / rho_i)))
        dT_dt = numerador / (m_i * Cv_i)

        # Return [dT/dt, dm/dt, dy_suc/dt, dv_suc/dt, dy_des/dt, dv_des/dt]
        return [dT_dt, dm_dt, v_suc_i, dv_dt_suc, v_des_i, dv_dt_des]


    sol = solve_ivp(
        fun=compressor_odes,
        t_span=(0, t_stop),
        y0=Y0,
        method=solver_method,
        rtol=custom_rtol,
        atol=custom_atol
    )

    # ==========================================
    # VERIFICACAO DE CONVERGENCIA
    # ==========================================
    if sol.success:
        print(f"[{nome_condicao}] Simulação {solver_method} concluída com sucesso! ({sol.message})")
    else:
        print(f"[{nome_condicao}] FALHA NA SIMULAÇÃO! Motivo: {sol.message}")
    # ==========================================

    print(f"[{nome_condicao}] Processando dados de saída...")


    # ---------------------------------------------------------
    # Post-Processing
    # ---------------------------------------------------------
    # solve_ivp returns the primary states. We must calculate the
    # dependent algebraic variables (P, h, m_dot) post-integration.

    t_out = sol.t
    T_out = sol.y[0]
    m_out = sol.y[1]
    y_suc_out = sol.y[2]
    v_suc_out = sol.y[3]
    y_des_out = sol.y[4]
    v_des_out = sol.y[5]

    # Pre-allocate output arrays
    P_out = np.zeros_like(t_out)
    V_out = np.zeros_like(t_out)
    h_out = np.zeros_like(t_out)
    m_suc_out = np.zeros_like(t_out)
    m_des_out = np.zeros_like(t_out)

    for i, t_val in enumerate(t_out):
        T_i = T_out[i]
        m_i = m_out[i]
        y_s_i = y_suc_out[i]
        y_d_i = y_des_out[i]

        V_i = compressor.volume(t_val)
        rho_i = m_i / V_i
        V_out[i] = V_i

        try:
            P_i = CP.PropsSI('P', 'T', T_i, 'D', rho_i, fluid)
            h_i = CP.PropsSI('H', 'T', T_i, 'D', rho_i, fluid)
            k_i = CP.PropsSI('CP0MASS', 'T', T_i, 'D', rho_i, fluid) / CP.PropsSI('CVMASS', 'T', T_i, 'D', rho_i, fluid)
        except ValueError:
            P_i = rho_i * R_gas * T_i
            h_i = 0  # Fallback
            k_i = 1.4  # Fallback

        P_out[i] = P_i
        h_out[i] = h_i

        a_ee_s = compressor.suction_valve.Aee(y_s_i)
        a_ee_d = compressor.discharge_valve.Aee(y_d_i)

        m_suc_out[i] = compressor.suction_valve.m_dot_valve(P_suc_target, P_i, T_suc, k_i, a_ee_s, R_gas,
                                                            fator_esc_reverso)
        m_des_out[i] = compressor.discharge_valve.m_dot_valve(P_i, P_c, T_i, k_i, a_ee_d, R_gas, fator_esc_reverso)

    return {
        'tempo': t_out,
        'P': P_out,
        'T': T_out,
        'V': V_out,
        'h': h_out,
        'm': m_out,
        'y_suc': y_suc_out,
        'y_des': y_des_out,
        'v_suc': v_suc_out,
        'v_des': v_des_out,
        'm_suc': m_suc_out,
        'm_des': m_des_out
    }