import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path
import CoolProp.CoolProp as CP
import config


# ==============================================================================
# 1 - GRAFICO: DIAGRAMA P-V
# ==============================================================================
def plot_diagrama_pv(res, titulo, freq, num_ciclos, P_e, P_c, out_dir, cor_p='blue'):
    safe_titulo = titulo.replace(' ', '_')
    t_start_last = (num_ciclos - 1) * (1.0 / freq)
    idx = np.where(res['tempo'] >= t_start_last)[0]

    fig1, ax1 = plt.subplots(figsize=(7, 5))

    ax1.plot(res['V'][idx] * 1e6, res['P'][idx] / 1e5, color=cor_p, lw=2, label=f'{titulo} (Último Ciclo)')
    ax1.axhline(P_e / 1e5, ls=':', color='blue', label='Pressao de Evaporacao')
    ax1.axhline(P_c / 1e5, ls=':', color='red', label='Pressao de Condensacao')

    ax1.set_ylim(0, 12)
    ax1.set_title(f"Diagrama P-V - Regime Permanente - {titulo}", fontsize=12)
    ax1.set_xlabel('Volume (cm³)')
    ax1.set_ylabel('Pressao (bar)')
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend()

    fig1.tight_layout()
    fig1.savefig(out_dir / f"Diagrama_PV_{safe_titulo}.png", format='png', bbox_inches='tight')
    plt.close(fig1)


# ==============================================================================
# 2 - GRAFICO: DINAMICA DO CICLO (P e V vs Tempo/CA)
# ==============================================================================
def plot_dinamica_ciclo(res, titulo, freq, num_ciclos, out_dir, cor_p='blue'):
    safe_titulo = titulo.replace(' ', '_')
    t_start_last = (num_ciclos - 1) * (1.0 / freq)
    idx = np.where(res['tempo'] >= t_start_last)[0]
    t_plot_ms = (res['tempo'][idx] - t_start_last) * 1000

    fig2, ax_p = plt.subplots(figsize=(8, 5))

    ax_p.plot(t_plot_ms, res['P'][idx] / 1e5, color=cor_p, lw=2, label='Pressao')
    ax_p.set_ylabel('Pressao (bar)', color=cor_p, fontsize=11)
    ax_p.tick_params(axis='y', labelcolor=cor_p, direction='in')

    ax_v = ax_p.twinx()
    ax_v.plot(t_plot_ms, res['V'][idx] * 1e6, color='green', lw=2, ls='--', label='Volume')
    ax_v.set_ylabel('Volume (cm³)', color='green', fontsize=11)
    ax_v.tick_params(axis='y', labelcolor='green', direction='in')

    def t2ca(x): return x * (freq * 360 / 1000)

    def ca2t(x): return x / (freq * 360 / 1000)

    secax2 = ax_p.secondary_xaxis('top', functions=(t2ca, ca2t))
    secax2.set_xlabel('Angulo de Manivela (°CA)', fontsize=10, labelpad=10)
    secax2.set_xticks([0, 90, 180, 270, 360])

    ax_p.set_xlabel('Tempo no Ciclo (ms)', fontsize=11)
    ax_p.set_title(f"Dinamica do Último Ciclo - {titulo}", fontsize=12, pad=30)
    ax_p.grid(True, linestyle='--', alpha=0.3)

    h1, l1 = ax_p.get_legend_handles_labels()
    h2, l2 = ax_v.get_legend_handles_labels()
    ax_p.legend(h1 + h2, l1 + l2, loc='center left', fontsize=9, framealpha=1.0)

    fig2.tight_layout()
    fig2.savefig(out_dir / f"Dinamica_Ciclo_{safe_titulo}.png", format='png', bbox_inches='tight')
    plt.close(fig2)


# ==============================================================================
# 3 - DESLOCAMENTO DAS VALVULAS
# ==============================================================================
def plot_deslocamento_valvulas(res, titulo, freq, num_ciclos, y_max_s, y_max_d, out_dir):
    safe_titulo = titulo.replace(' ', '_')
    t_ciclo = 1.0 / freq
    t_start_last = (num_ciclos - 1) * t_ciclo
    idx = np.where(res['tempo'] >= t_start_last)[0]
    t_plot_ms = (res['tempo'][idx] - t_start_last) * 1000

    fig3, ax_valv = plt.subplots(figsize=(10, 5))

    t_max_ms = t_ciclo * 1000
    y_suc_mm = res['y_suc'][idx] * 1000
    y_des_mm = res['y_des'][idx] * 1000

    ax_valv.xaxis.set_major_locator(ticker.MultipleLocator(1.0))
    ax_valv.xaxis.set_minor_locator(ticker.MultipleLocator(0.2))
    ax_valv.set_xlim(0, t_max_ms)

    ax_valv.plot(t_plot_ms, y_suc_mm, color='purple', lw=2, label='Valvula de Succao')
    ax_valv.plot(t_plot_ms, y_des_mm, color='red', lw=2, label='Valvula de Descarga')

    ax_valv.set_xlabel('Tempo no Ciclo (ms)', fontsize=11)
    ax_valv.set_ylabel('Elevacao das Valvulas (mm)', fontsize=11)

    ax_valv.axhline(y_max_s * 1000, color='purple', linestyle=':', alpha=0.4)
    ax_valv.axhline(y_max_d * 1000, color='red', linestyle=':', alpha=0.4)

    def t2ca(x): return x * (freq * 360 / 1000)

    def ca2t(x): return x / (freq * 360 / 1000)

    secax3 = ax_valv.secondary_xaxis('top', functions=(t2ca, ca2t))
    secax3.set_xlabel('Angulo de Manivela (°CA)')
    secax3.xaxis.set_major_locator(ticker.MultipleLocator(45))

    ax_valv_v = ax_valv.twinx()
    ax_valv_v.plot(t_plot_ms, res['V'][idx] * 1e6, color='green', lw=1.5, ls='--', alpha=0.7, label='Volume')
    ax_valv_v.set_ylabel('Volume (cm³)', fontsize=11, color='green')
    ax_valv_v.tick_params(axis='y', labelcolor='green')

    ax_valv.set_title(f"Dinamica das Valvulas - {titulo}", fontsize=13, pad=25)
    ax_valv.grid(True, which='major', linestyle='-', alpha=0.5)
    ax_valv.grid(True, which='minor', linestyle=':', alpha=0.2)

    lines, labels = ax_valv.get_legend_handles_labels()
    lines_v, labels_v = ax_valv_v.get_legend_handles_labels()
    ax_valv.legend(lines + lines_v, labels + labels_v, loc='upper right', frameon=True, fontsize=9)

    fig3.tight_layout()
    fig3.savefig(out_dir / f"Deslocamento_Valvulas_{safe_titulo}.png", format='png', bbox_inches='tight')
    plt.close(fig3)


# ==============================================================================
# 4 - VAZOES MASSICAS
# ==============================================================================
def plot_vazoes_massicas(res, titulo, freq, num_ciclos, out_dir):
    safe_titulo = titulo.replace(' ', '_')
    t_start_last = (num_ciclos - 1) * (1.0 / freq)
    idx = np.where(res['tempo'] >= t_start_last)[0]

    fig4, ax_vaz = plt.subplots(figsize=(9, 5))

    t_norm = res['tempo'][idx] - t_start_last
    angulo_ca = t_norm * freq * 360.0

    m_suc_gs = res['m_suc'][idx] * 1000.0
    m_des_gs = res['m_des'][idx] * 1000.0

    ax_vaz.scatter(angulo_ca, m_suc_gs, s=0.1, color='purple', lw=2, label='Vazao de Succao')
    ax_vaz.scatter(angulo_ca, m_des_gs, s=0.1, color='red', lw=2, label='Vazao de Descarga')
    ax_vaz.axhline(0, color='black', linestyle='-', linewidth=1.5, alpha=0.5)

    ax_vaz.set_xlabel('Angulo de Manivela (°CA)', fontsize=11)
    ax_vaz.set_ylabel('Vazao Massica (g/s)', fontsize=11)
    ax_vaz.set_title(f"Vazoes Massicas vs Angulo de Manivela - {titulo}", fontsize=12, pad=15)

    ax_vaz.set_xlim(0, 360)
    ax_vaz.set_xticks(np.arange(0, 361, 45))

    ax_vaz.grid(True, linestyle='--', alpha=0.6)
    ax_vaz.legend(loc='upper right', fontsize=10)

    fig4.tight_layout()
    fig4.savefig(out_dir / f"Vazoes_Massicas_{safe_titulo}.png", format='png', bbox_inches='tight')
    plt.close(fig4)

# ==============================================================================
# CALCULO DE METRICAS DE DESEMPENHO
# ==============================================================================
def calculate_mass_flow(res, freq, num_ciclos, compressor):
    t_start_last = (num_ciclos - 1) * (1.0 / compressor.freq)
    idx = np.where(res['tempo'] >= t_start_last)[0]
    t_cycle = res['tempo'][idx] - t_start_last
    m_suc_cycle = res['m_suc'][idx]
    mass_flow = np.trapezoid(m_suc_cycle, t_cycle) * freq
    return mass_flow

def calculate_cycle_work(res, num_ciclos, compressor):
    t_start_last = (num_ciclos - 1) * (1.0 / compressor.freq)
    idx = np.where(res['tempo'] >= t_start_last)[0]
    p_cycle = res['P'][idx]
    v_cycle = res['V'][idx]
    work_cycle = np.trapezoid(p_cycle, v_cycle)
    return work_cycle

def calculate_indicated_power(res, freq, num_ciclos, compressor):
    t_start_last = (num_ciclos - 1) * (1.0 / compressor.freq)
    idx = np.where(res['tempo'] >= t_start_last)[0]
    p_cycle = res['P'][idx]
    v_cycle = res['V'][idx]
    work_cycle = np.trapezoid(p_cycle, v_cycle)
    indicated_power = -work_cycle * compressor.freq
    return indicated_power

def calculate_volumetric_efficiency(mass_flow, P_e, compressor):
    rho_evap = CP.PropsSI('D', 'P', P_e,'Q', 1.0,  config.fluid)
    V_sw = compressor.V_swept
    volumetric_efficiency = mass_flow / (rho_evap * V_sw * freq)
    return volumetric_efficiency

def calculate_isentropic_efficiency(mass_flow, indicated_power, P_e, P_c, T_suc):
    h_suc = CP.PropsSI('H', 'P', P_e,'T', T_suc,  config.fluid)
    s_suc = CP.PropsSI('S', 'P', P_e,'T', T_suc,  config.fluid)
    h_des_isen = CP.PropsSI('H', 'P', P_c, 'S', s_suc, config.fluid)
    isentropic_power = mass_flow * (h_des_isen - h_suc)
    isentropic_efficiency = isentropic_power / indicated_power
    return isentropic_efficiency

def calculate_cop(mass_flow, indicated_power, P_e, P_c, T_suc):
    h_cond = CP.PropsSI('H', 'P', P_c,'Q', 0.0,  config.fluid)
    h_evap = CP.PropsSI('H', 'P', P_e,'T', T_suc,  config.fluid)
    delta_h = h_evap - h_cond
    q_evap = mass_flow * delta_h
    cop = q_evap / indicated_power
    return cop

def sim_summary(res, titulo, freq, num_ciclos, P_e, P_c, T_suc, compressor):
    mass_flow = calculate_mass_flow(res, freq, num_ciclos, compressor)
    work_cycle = calculate_cycle_work(res, num_ciclos,compressor)
    indicated_power = calculate_indicated_power(res, freq, num_ciclos,compressor)
    vol_eff = calculate_volumetric_efficiency(mass_flow, P_e, compressor)
    isen_eff = calculate_isentropic_efficiency(mass_flow, indicated_power, P_e, P_c, T_suc)
    cop = calculate_cop(mass_flow, indicated_power, P_e, P_c, T_suc)

    print(f"\n--- Resumo da Simulação: {titulo} ---")
    print(f"  Vazão mássica: {mass_flow*3600:.2f} kg/h")
    print(f"  Trabalho por ciclo: {work_cycle:.2f} J")
    print(f"  Potência indicada: {indicated_power:.2f} W")
    print(f"  Eficiência volumétrica: {vol_eff:.2%}")
    print(f"  Eficiência isentrópica: {isen_eff:.2%}")
    print(f"  COP: {cop:.2f}")

# ==============================================================================
# EXECUCAO PRINCIPAL (Leitura dos CSVs e geracao dos plots)
# ==============================================================================
if __name__ == '__main__':
    from config import freq, P_eA, P_eB, P_c, T_eA, T_eB, y_max_s, y_max_d, T_suc, num_ciclos

    

    out_dir = Path('outputs')

    csv_A = out_dir / "Resultados_Condicao_A.csv"
    csv_B = out_dir / "Resultados_Condicao_B.csv"

    def csv_to_dict(caminho_csv):
        df = pd.read_csv(caminho_csv)
        return {coluna: df[coluna].values for coluna in df.columns}

    if csv_A.exists():
        print(f"Lendo e gerando graficos para: {csv_A}")
        res_A = csv_to_dict(csv_A)
        plot_diagrama_pv(res_A, "Condicao A", freq, num_ciclos, P_eA, P_c, out_dir, cor_p='blue')
        plot_dinamica_ciclo(res_A, "Condicao A", freq, num_ciclos, out_dir, cor_p='blue')
        plot_deslocamento_valvulas(res_A, "Condicao A", freq, num_ciclos, y_max_s, y_max_d, out_dir)
        plot_vazoes_massicas(res_A, "Condicao A", freq, num_ciclos, out_dir)
        sim_summary(res_A, "Condicao A", freq, num_ciclos, P_eA, P_c, T_suc, config.compressor_ativo)
    else:
        print(f"Aviso: Arquivo {csv_A} nao encontrado.")

    if csv_B.exists():
        print(f"Lendo e gerando graficos para: {csv_B}")
        res_B = csv_to_dict(csv_B)
        plot_diagrama_pv(res_B, "Condicao B", freq, num_ciclos, P_eB, P_c, out_dir, cor_p='darkorange')
        plot_dinamica_ciclo(res_B, "Condicao B", freq, num_ciclos, out_dir, cor_p='darkorange')
        plot_deslocamento_valvulas(res_B, "Condicao B", freq, num_ciclos, y_max_s, y_max_d, out_dir)
        plot_vazoes_massicas(res_B, "Condicao B", freq, num_ciclos, out_dir)
        sim_summary(res_B, "Condicao B", freq, num_ciclos, P_eB, P_c, T_suc, config.compressor_ativo)
    else:
        print(f"Aviso: Arquivo {csv_B} nao encontrado.")

    print("Processo de geracao de graficos concluído!")
