import time
import concurrent.futures
import pandas as pd
from config import *
from compressor_model import ReciprocatingCompressor
from solver import simular_condicao
from pathlib import Path

if __name__ == '__main__':
    num_ciclos = 15
    delta_P_max = 1000
    out_dir = Path('outputs')
    out_dir.mkdir(parents=True, exist_ok=True)

    # ==============================================================================
    # ESCOLHA E CONFIGURACAO DO MODELO DO COMPRESSOR
    # ==============================================================================
    compressor_ativo = ReciprocatingCompressor(
        Dp, r_manivela, l_biela, l_pmls, dm, Vm, freq, sp_med
    )

    print(f"Usando modelo geométrico: {compressor_ativo.__class__.__name__}")

    # ==============================================================================
    # EXECUCAO PARALELA
    # ==============================================================================
    start_time = time.time()
    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as executor:
        future_A = executor.submit(simular_condicao, P_eA, "Condição A", delta_P_max, num_ciclos, compressor_ativo)
        future_B = executor.submit(simular_condicao, P_eB, "Condição B", delta_P_max, num_ciclos, compressor_ativo)

        res_A = future_A.result()
        res_B = future_B.result()

    elapsed = time.time() - start_time
    print(f"\nSimulações concluídas em {elapsed:.2f}s.")

    # ==============================================================================
    # SALVAR DADOS PARA CSV
    # ==============================================================================
    print("Salvando resultados em CSV...")
    df_A = pd.DataFrame(res_A)
    df_B = pd.DataFrame(res_B)

    df_A.to_csv(out_dir / "Resultados_Condicao_A.csv", index=False)
    df_B.to_csv(out_dir / "Resultados_Condicao_B.csv", index=False)

    print("Simulações finalizadas com sucesso! Execute plotter.py para gerar os gráficos.")