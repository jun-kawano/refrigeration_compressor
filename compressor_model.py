
import numpy as np
import CoolProp.CoolProp as CP


# ==============================================================================
# MODELOS DE VALVULAS
# ==============================================================================

class Valve:
    def __init__(self, valve_type, m_eq, k_eq, y_max):
        self.valve_type = valve_type
        self.m_eq = m_eq
        self.k_eq = k_eq
        self.y_max = y_max
        if self.valve_type == 'suction':
            self.aef_coeffs_low = [5.13E-05, -4.73E-02, 1.30E+02]
            self.aef_coeffs_high = [4.42E-05, 2.39E-02, -3.28E+01, 1.84E+04, -5.51E+06, 6.69E+08]
            self.aef_const = 5.13E-05
            self.aee_coeffs = [0.0, 2.55E-02, -9.19E+00, 1.94E+03, -2.00E+05]
        elif self.valve_type == 'discharge':
            self.aef_coeffs = [3.32E-05, -1.38E-01, 2.87E+02, -1.62E+05]
            self.aee_coeffs = [0.0, 3.12E-02, -2.43E+01, 5.30E+03]
        else:
            raise ValueError("valve_type must be either 'suction' or 'discharge'")

    def Aef(self, y):
        #
        y_safe = max(0.0, min(y, self.y_max))

        if self.valve_type == 'suction':
            if y_safe <= 0.3e-3:
                c1, c2, c3 = self.aef_coeffs_low
                return c1 + (c2 * y_safe) + (c3 * y_safe * y_safe)
            elif y_safe <= 3e-3:
                c1, c2, c3, c4, c5, c6 = self.aef_coeffs_high
                return c1 + (c2 * y_safe) + (c3 * y_safe * y_safe) + (c4 * y_safe ** 3) + (c5 * y_safe ** 4) + (
                            c6 * y_safe ** 5)
            else:
                return self.aef_const
        elif self.valve_type == 'discharge':
            c1, c2, c3, c4 = self.aef_coeffs
            return c1 + (c2 * y_safe) + (c3 * y_safe * y_safe) + (c4 * y_safe ** 3)

    def Aee(self, y):
        y_safe = max(0.0, min(y, self.y_max))

        if self.valve_type == 'suction':
            c1, c2, c3, c4, c5 = self.aee_coeffs
            return c1 + (c2 * y_safe) + (c3 * y_safe * y_safe) + (c4 * y_safe ** 3) + (c5 * y_safe ** 4)
        elif self.valve_type == 'discharge':
            c1, c2, c3, c4 = self.aee_coeffs
            return c1 + (c2 * y_safe) + (c3 * y_safe * y_safe) + (c4 * y_safe ** 3)

    def m_dot_valve(self, P_up, P_down, T_up, k, area, R_gas, fator_esc_reverso):
        if area <= 0.0:
            return 0.0
        if P_down > P_up:
            P_in, P_out = P_down, P_up
            sinal = -1.0
            area = fator_esc_reverso * area
        else:
            P_in, P_out = P_up, P_down
            sinal = 1.0

        rs = P_out / P_in
        r_choke = (2 / (k + 1)) ** (k / (k - 1))
        if rs < r_choke:
            rs = r_choke

        term1 = 2 * k / (R_gas * T_up * (k - 1))
        term2 = rs ** (2 / k) - rs ** ((k + 1) / k)

        return sinal * area * P_in * np.sqrt(term1 * term2)

    # def get_acceleration(self, delta_P, y):
    #     a_ef = self.Aef(y)
    #     F_gas = delta_P * a_ef
    #     dv_dt = (F_gas - (self.k_eq * y)) / self.m_eq
    #
    #     if y <= 0.0 and dv_dt < 0:
    #         dv_dt = 0.0
    #     elif y >= self.y_max and dv_dt > 0:
    #         dv_dt = 0.0
    #     return dv_dt, F_gas, a_ef
    def get_acceleration(self, delta_P, y, v):
        a_ef = self.Aef(y)
        F_gas = delta_P * a_ef
        dv_dt = (F_gas - (self.k_eq * y)) / self.m_eq

        # ==========================================
        # penalty method for dealing with backstop
        k_bump = 5e5  # Stiff virtual spring
        c_bump = 20.0  # Virtual damper to stop bouncing

        if y < 0.0:
            dv_dt += (-k_bump * y - c_bump * v) / self.m_eq
        elif y > self.y_max:
            dv_dt += (-k_bump * (y - self.y_max) - c_bump * v) / self.m_eq

        return dv_dt, F_gas, a_ef

# ==============================================================================
# MODELOS DE COMPRESSORES
# ==============================================================================S

class ReciprocatingCompressor:
    """classe com a geometria e cinematica de um compressor alternativo de pistao"""

    def __init__(self, Dp, r_manivela, l_biela, l_pmls, dm, Vm, freq,
                 m_eq_s, k_eq_s, y_max_s,
                 m_eq_d, k_eq_d, y_max_d):
        self.Dp = Dp
        self.r_manivela = r_manivela
        self.l_biela = l_biela
        self.l_pmls = l_pmls
        self.dm = dm
        self.Vm = Vm
        self.freq = freq
        self.L = 2.0 * r_manivela
        self.sp_med = 2.0 * (2.0 * r_manivela) * freq
        self.A_piston = 0.25 * np.pi * Dp * Dp
        self.V_swept = self.A_piston * self.L
        self.suction_valve = Valve('suction', m_eq_s, k_eq_s, y_max_s)
        self.discharge_valve = Valve('discharge', m_eq_d, k_eq_d, y_max_d)

    def crank_angle(self, t):
        return np.remainder(2 * np.pi * self.freq * t, 4 * np.pi)

    def piston_height(self, t):
        ca = self.crank_angle(t)
        return self.l_pmls - (-self.r_manivela * np.cos(ca) + np.sqrt(
            self.l_biela ** 2 - (self.r_manivela * np.sin(ca) - self.dm) ** 2))

    def volume(self, t):
        """retorna o volume instantaneo da camara"""
        return self.Vm + self.A_piston * self.piston_height(t)

    def area_convec(self, t):
        """retorna a area instantanea de troca termica convectiva"""
        return np.pi * self.Dp * self.piston_height(t) + (2 * self.A_piston)

    def char_length(self):
        """comprimento caracteristico (diametro do cilindro)"""
        return self.Dp

    def h_coef(self, Ti, rho_i, fluid):
        A_coef = 0.7
        mu_G = CP.PropsSI('V', 'T', Ti, 'D', rho_i, fluid)
        k_term = CP.PropsSI('L', 'T', Ti, 'D', rho_i, fluid)
        v_ref = self.sp_med
        L_char = self.char_length()
        Re = rho_i * v_ref * L_char / mu_G
        return 3 * (k_term * A_coef * (Re ** A_coef)) / L_char