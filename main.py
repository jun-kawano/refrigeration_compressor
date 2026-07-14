import time
import concurrent.futures
import pandas as pd
import config
from solver import simular_condicao
from pathlib import Path

if __name__ == '__main__':
    out_dir = Path('outputs')
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Usando modelo geométrico: {config.compressor_ativo.__class__.__name__} e método: {config.solver_method}")

    # ==============================================================================
    # EXECUCAO PARALELA
    # ==============================================================================
    start_time = time.time()
    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as executor:
        future_A = executor.submit(
            simular_condicao, config.P_eA, "Condição A", config.delta_P_max, config.num_ciclos, config.compressor_ativo, config.solver_method
        )
        future_B = executor.submit(
            simular_condicao, config.P_eB, "Condição B", config.delta_P_max, config.num_ciclos, config.compressor_ativo, config.solver_method
        )

        res_A = future_A.result()
        res_B = future_B.result()

    # res_A = simular_condicao(config.P_eA, "Condição A", config.delta_P_max, config.num_ciclos, config.compressor_ativo, config.solver_method)

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
