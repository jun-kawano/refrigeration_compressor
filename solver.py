import numpy as np
import CoolProp.CoolProp as CP
from config import *
from compressor_model import h_coef, Aef_suc, Aef_des, Aee_suc, Aee_des, m_dot_valve

def simular_condicao(P_suc_target, nome_condicao, delta_P_max, num_ciclos, compressor):
    print(f"\n--- Iniciando simulação da {nome_condicao} (Passo Adaptativo) ---")
    t_stop = num_ciclos * (1 / freq)
    i = 0
    h_suc_in = CP.PropsSI('H', 'T', T_suc, 'P', P_suc_target, fluid)
    T_cil_local = 65 + 273.15
    t_ciclo = 1.0 / freq

    # Limites do Passo de Tempo
    dt_max = t_ciclo / 180
    dt_min = dt_max / 128
    dt = dt_max

    # Inicializacao das variaveis
    sim_time = [0.0]
    T = [T_suc]
    P = [P_suc_target]
    vol = [compressor.volume(0.0)]
    m = [vol[0] * CP.PropsSI('D', 'T', T_suc, 'P', P_suc_target, fluid)]
    h = [CP.PropsSI('H', 'T', T_suc, 'P', P_suc_target, fluid)]
    s = [CP.PropsSI('S', 'T', T_suc, 'P', P_suc_target, fluid)]

    y_v_suc, v_v_suc = [0.0], [0.0]
    y_v_des, v_v_des = [0.0], [0.0]
    mdot_suc, mdot_des = [0.0], [0.0]

    F_gas_s_list, F_gas_d_list = [0.0], [0.0]
    A_ef_s_list, A_ef_d_list = [Aef_suc(0.0)], [Aef_des(0.0)]
    A_ee_s_list, A_ee_d_list = [Aee_suc(0.0)], [Aee_des(0.0)]

    t = 0.0
    ciclo_atual = 0

    # Loop do Solver
    while t < t_stop:
        T_i = T[-1]
        P_i = P[-1]
        m_i = m[-1]
        V_i = vol[-1]

        rho_i = m_i / V_i

        Cv_i = CP.PropsSI('CVMASS', 'T', T_i, 'D', rho_i, fluid)
        dP_dT_i = CP.PropsSI('d(P)/d(T)|D', 'T', T_i, 'D', rho_i, fluid)
        h_i = h[-1]
        k_i = CP.PropsSI('CP0MASS', 'T', T_i, 'D', rho_i, fluid) / Cv_i

        # --- LOGICA DAS VALVULAS ---
        y_suc_i, v_suc_i = y_v_suc[-1], v_v_suc[-1]
        y_des_i, v_des_i = y_v_des[-1], v_v_des[-1]

        a_ef_s = Aef_suc(y_suc_i)
        a_ef_d = Aef_des(y_des_i)
        a_ee_s = Aee_suc(y_suc_i)
        a_ee_d = Aee_des(y_des_i)

        # Dinamica Succao
        delta_P_suc = P_suc_target - P_i
        F_gas_suc = delta_P_suc * a_ef_s
        dv_dt_suc = (F_gas_suc - (k_eq_s * y_suc_i)) / m_eq_s

        if y_suc_i <= 0.0 and dv_dt_suc < 0:
            dv_dt_suc = 0.0
        elif y_suc_i >= y_max_s and dv_dt_suc > 0:
            dv_dt_suc = 0.0

        # Dinamica Descarga
        delta_P_des = P_i - P_c
        F_gas_des = delta_P_des * a_ef_d
        dv_dt_des = (F_gas_des - (k_eq_d * y_des_i)) / m_eq_d

        if y_des_i <= 0.0 and dv_dt_des < 0:
            dv_dt_des = 0.0
        elif y_des_i >= y_max_d and dv_dt_des > 0:
            dv_dt_des = 0.0

        # balanco de massa
        dm_dt_suc = m_dot_valve(P_suc_target, P_i, T_suc, k_i, a_ee_s)
        dm_dt_des = m_dot_valve(P_i, P_c, T_i, k_i, a_ee_d)

        dm_dt = dm_dt_suc - dm_dt_des

        V_prox_calc = compressor.volume(t + dt)
        dV_dt = (V_prox_calc - V_i) / dt

        # repassando a classe do compressor para a troca termica
        H_convec = h_coef(T_i, rho_i, compressor)
        A_convec = compressor.area_convec(t)
        Q_dot = H_convec * A_convec * (T_cil_local - T_i)
        H_flux = (dm_dt_suc * h_suc_in) - (dm_dt_des * h_i)
        # balanco de Energia
        numerador = Q_dot + H_flux - (h_i * dm_dt) - (T_i * dP_dT_i * (dV_dt - (dm_dt / rho_i)))
        dT_dt = numerador / (m_i * Cv_i)

        # calculos das proximas variaveis
        T_prox = T_i + dT_dt * dt
        T_prox = max(T_prox, 150.0)  # T min do coolprop

        m_prox = m_i + dm_dt * dt
        m_prox = max(m_prox, 1e-10)  # evitar problemas de inf

        V_prox = V_prox_calc
        rho_prox = m_prox / V_prox

        v_suc_prox = v_suc_i + dv_dt_suc * dt
        v_des_prox = v_des_i + dv_dt_des * dt

        y_suc_prox = y_suc_i + v_suc_prox * dt + (dv_dt_suc * dt * dt / 2)
        y_des_prox = y_des_i + v_des_prox * dt + (dv_dt_des * dt * dt / 2)

        # limites (bounce)
        if y_suc_prox < 0.0:
            y_suc_prox, v_suc_prox = 0.0, 0.0
        elif y_suc_prox > y_max_s:
            y_suc_prox, v_suc_prox = y_max_s, 0.0

        if y_des_prox < 0.0:
            y_des_prox, v_des_prox = 0.0, 0.0
        elif y_des_prox > y_max_d:
            y_des_prox, v_des_prox = y_max_d, 0.0

        try:
            P_prox = CP.PropsSI('P', 'T', T_prox, 'D', rho_prox, fluid)
            if np.isnan(P_prox) or P_prox < 0:
                w = 0.2
                P_prox = (1-w)*(rho_prox * R_gas * T_prox) + w*P_i
        except Exception:
            w = 0.2
            P_prox = (1-w)*(rho_prox * R_gas * T_prox) + w*P_i

        delta_P = abs(P_prox - P_i)

        # adaptacao de passo
        if delta_P <= delta_P_max or dt <= dt_min:
            t += dt
            i += 1
            sim_time.append(t)
            T.append(T_prox)
            P.append(P_prox)
            m.append(m_prox)
            vol.append(V_prox)
            h.append(CP.PropsSI('H', 'T', T_prox, 'D', rho_prox, fluid))
            s.append(CP.PropsSI('S', 'T', T_prox, 'D', rho_prox, fluid))

            y_v_suc.append(y_suc_prox)
            v_v_suc.append(v_suc_prox)
            y_v_des.append(y_des_prox)
            v_v_des.append(v_des_prox)

            mdot_suc.append(dm_dt_suc)
            mdot_des.append(dm_dt_des)

            F_gas_s_list.append(F_gas_suc)
            F_gas_d_list.append(F_gas_des)
            A_ef_s_list.append(a_ef_s)
            A_ef_d_list.append(a_ef_d)
            A_ee_s_list.append(a_ee_s)
            A_ee_d_list.append(a_ee_d)

            dt = min(dt * 1.1, dt_max)

            if t >= (ciclo_atual + 1) * t_ciclo:
                ciclo_atual += 1
                print(f"[{nome_condicao}] Ciclo {ciclo_atual}/{num_ciclos} finalizado. Passos totais: {len(sim_time)}")
        else:
            dt = dt / 2.0

    print(f"[{nome_condicao}] Simulação concluida!")
    return {
        'tempo': np.array(sim_time),
        'P': np.array(P),
        'T': np.array(T),
        'V': np.array(vol),
        'h': np.array(h),
        'm': np.array(m),
        'y_suc': np.array(y_v_suc),
        'y_des': np.array(y_v_des),
        'v_suc': np.array(v_v_suc),
        'v_des': np.array(v_v_des),
        'm_suc': np.array(mdot_suc),
        'm_des': np.array(mdot_des)
    }