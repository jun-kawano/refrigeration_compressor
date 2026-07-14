import CoolProp.CoolProp as CP
import numpy as np
from compressor_model import ReciprocatingCompressor

# ==============================================================================
# PARAMETROS DA SIMULACAO
# ==============================================================================

num_ciclos = 15
delta_P_max = 1000
solver_method = 'RK23'
# ==============================================================================
# PARAMETROS GERAIS E TERMODINAMICOS
# ==============================================================================

# condicoes operacionais
fluid = 'R600a'
T_eA = -23.3 + 273.15
T_eB = -10.0 + 273.15
T_c = 54.4 + 273.15

T_cil = 65 + 273.15
T_suc = 32.2 + 273.15

P_eA = CP.PropsSI('P','T',T_eA,'Q', 0, fluid)
P_eB = CP.PropsSI('P','T',T_eB,'Q', 0, fluid)
P_c = CP.PropsSI('P','T',T_c,'Q', 0, fluid)

# geometria do cilindro
Dp = 22.5e-3        # Diametro do pistao [m]
r_manivela = 7.5e-3 # Excentricidade (raio da manivela) [m]
l_biela = 38.5e-3   # Comprimento da biela [m]
l_pmls = l_biela + r_manivela  # Comprimento do pms [m]
dm = 2e-3           # Reversibilidade entre eixo e cilindro [m]
Do_s = 8e-2         # Diametro do orificio de succao [m]
Do_d = 6.5e-3       # Diametro do orificio de descarga [m]
N = 3600            # RPM
Vm = 120e-9         # Volume morto [m³]
freq = N / 60.0



# Constante do gas
R_u = CP.PropsSI('gas_constant', fluid)
R_gas = R_u / CP.PropsSI('M', fluid)

# ==============================================================================
# PARAMETROS DINAMICOS DAS VALVULAS
# ==============================================================================
fator_esc_reverso = 0.2

omega_s = 394 * 2 * np.pi
omega_d = 528 * 2 * np.pi

# m_eq_s = 0.1318e-3   # massa equivalente (kg)
# k_eq_s = omega_s**2 * m_eq_s  # rigidez equivalente (N/m)
#
# m_eq_d = 0.259e-3    # massa equivalente (kg)
# k_eq_d = omega_d**2 * m_eq_d  # rigidez equivalente (N/m)

# Succao
m_eq_s = 4e-4   # Massa equivalente (kg) #4e-3
k_eq_s = (215.5)  # Rigidez equivalente (N/m) #215.5
y_max_s = 3e-3 # Altura do batente (m)#  3e-3

# Descarga
m_eq_d = 3e-4   # Massa equivalente (kg) #3e-3
k_eq_d = (660)  # Rigidez equivalente (N/m) # (660.0) #FEA = 1667
y_max_d = 8e-4 # Altura do batente (m) #8e-4

# y_max_s = 3e-3       # altura do batente (m)
# y_max_d = 8e-4       # altura do batente (m)
delta_t_s = 2 / omega_s
delta_t_d = 2 / omega_s


compressor_ativo = ReciprocatingCompressor(
    Dp, r_manivela, l_biela, l_pmls,
    dm, Vm, freq,
    m_eq_s, k_eq_s, y_max_s,
    m_eq_d, k_eq_d, y_max_d
)
