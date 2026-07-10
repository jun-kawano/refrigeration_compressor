import numpy as np
import CoolProp.CoolProp as CP
from config import fluid, R_gas, fator_esc_reverso


# ==============================================================================
# MODELOS DE COMPRESSORES
# ==============================================================================

class ReciprocatingCompressor:
    """classe com a geometria e cinematica de um compressor alternativo de pistao"""

    def __init__(self, Dp, r_manivela, l_biela, l_pmls, dm, Vm, freq, sp_med):
        self.Dp = Dp
        self.r_manivela = r_manivela
        self.l_biela = l_biela
        self.l_pmls = l_pmls
        self.dm = dm
        self.Vm = Vm
        self.freq = freq
        self.sp_med = sp_med
        self.A_piston = 0.25 * np.pi * Dp ** 2

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

    def reference_velocity(self):
        """Velocidade de referencia para o de Re"""
        return self.sp_med

    def char_length(self):
        """comprimento caracteristico (diametro do cilindro)"""
        return self.Dp


# ==============================================================================
# FUNCOES TERMODINAMICAS E DE VALVULA ADAPTADAS
# ==============================================================================

def h_coef(Ti, rho_i, compressor):
    A_coef = 0.7
    mu_G = CP.PropsSI('V', 'T', Ti, 'D', rho_i, fluid)
    k_term = CP.PropsSI('L', 'T', Ti, 'D', rho_i, fluid)
    v_ref = compressor.reference_velocity()
    L_char = compressor.char_length()
    Re = rho_i * v_ref * L_char / mu_G
    return 3 * (k_term * A_coef * (Re ** A_coef)) / L_char


def Aef_suc(y):
    if y <= 0.3e-3:
        c1, c2, c3 = 5.13E-05, -4.73E-02, 1.30E+02
        Aef = c1 + (c2 * y) + (c3 * y * y)
    elif y <= 3e-3:
        c1, c2, c3 = 4.42E-05, 2.39E-02, -3.28E+01
        c4, c5, c6 = 1.84E+04, -5.51E+06, 6.69E+08
        Aef = c1 + (c2 * y) + (c3 * y * y) + (c4 * y ** 3) + (c5 * y ** 4) + (c6 * y ** 5)
    else:
        Aef = 5.13E-05
    return Aef


def Aef_des(y):
    c1, c2, c3, c4 = 3.32E-05, -1.38E-01, 2.87E+02, -1.62E+05
    Aef = c1 + (c2 * y) + (c3 * y * y) + (c4 * y ** 3)
    return Aef


def Aee_suc(y):
    c1, c2, c3, c4, c5 = 0.0, 2.55E-02, -9.19E+00, 1.94E+03, -2.00E+05
    Aee = c1 + (c2 * y) + (c3 * y * y) + (c4 * y ** 3) + (c5 * y ** 4)
    return Aee


def Aee_des(y):
    c1, c2, c3, c4 = 0.0, 3.12E-02, -2.43E+01, 5.30E+03
    Aee = c1 + (c2 * y) + (c3 * y * y) + (c4 * y ** 3)
    return Aee


def m_dot_valve(P_up, P_down, T_up, k, area):
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