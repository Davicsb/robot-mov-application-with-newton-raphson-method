import numpy as np
from typing import Callable, List, Optional, Tuple


def jacobiana_numerica(
    F: Callable[[np.ndarray], np.ndarray],
    x: np.ndarray,
    h: float = 1e-7,
) -> np.ndarray:
    """Calcula a matriz Jacobiana de F(x) numericamente por diferenças finitas centrais."""
    n = len(x)
    J = np.zeros((n, n), dtype=float)

    for j in range(n):
        x_mais = x.copy()
        x_menos = x.copy()
        x_mais[j] += h
        x_menos[j] -= h
        J[:, j] = (F(x_mais) - F(x_menos)) / (2 * h)

    return J


def metodo_newton_sistemas(
    F: Callable[[np.ndarray], np.ndarray],
    x0: np.ndarray,
    J: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    tol: float = 1e-6,
    max_iter: int = 100,
    exibir_tabela: bool = True,
) -> Tuple[Optional[np.ndarray], int, bool, List[np.ndarray]]:
    """
    Resolve um sistema de equações não lineares F(x) = 0 pelo Método de Newton.

    Parâmetros:
        F: Função vetorial que recebe um array x e retorna o array F(x).
        x0: Vetor com o chute inicial.
        J: Função que retorna a matriz Jacobiana J(x). Se None, calcula numericamente.
        tol: Tolerância para a norma do passo e de F(x).
        max_iter: Número máximo de iterações.
        exibir_tabela: Se True, imprime o progresso no terminal.

    Retorna:
        (solucao, iteracoes, convergiu, historico)
    """
    x = np.array(x0, dtype=float)
    historico: List[np.ndarray] = [x.copy()]

    if exibir_tabela:
        print(f"{'Iter':<6} | {'||F(x_k)||':<14} | {'||Delta x||':<14} | {'x_k'}")
        print("-" * 75)

    for k in range(1, max_iter + 1):
        Fx = np.array(F(x), dtype=float)
        Jx = np.array(J(x), dtype=float) if J is not None else jacobiana_numerica(F, x)

        try:
            # Resolve J(x) * delta_x = -F(x)
            delta_x = np.linalg.solve(Jx, -Fx)
        except np.linalg.LinAlgError:
            if exibir_tabela:
                print(f"\n[Erro] Matriz Jacobiana singular na iteração {k}. Método interrompido.")
            return None, k, False, historico

        x_novo = x + delta_x
        historico.append(x_novo.copy())

        norma_passo = float(np.linalg.norm(delta_x, ord=np.inf))
        norma_F = float(np.linalg.norm(F(x_novo), ord=np.inf))

        if exibir_tabela:
            x_formatado = np.array2string(x_novo, precision=6, suppress_small=True)
            print(f"{k:<6} | {norma_F:<14.6e} | {norma_passo:<14.6e} | {x_formatado}")

        if norma_passo < tol or norma_F < tol:
            return x_novo, k, True, historico

        x = x_novo

    if exibir_tabela:
        print("\n[Aviso] Número máximo de iterações atingido sem convergência.")
    return x, max_iter, False, historico



# ÁREA DE ENTRADA 
if __name__ == "__main__":
    # Sistema de exemplo:
    #   x² + y² - 4 = 0
    #   e^x + y - 1 = 0
    def F(v: np.ndarray) -> np.ndarray:
        x, y = v
        return np.array([
            x**2 + y**2 - 4,
            np.exp(x) + y - 1,
        ])

    J = None  # Deixe como None para usar a Jacobiana numérica
    chute_inicial = [1.0, -1.0]
    tolerancia = 1e-8
    max_iteracoes = 50

    solucao, iteracoes, convergiu, _ = metodo_newton_sistemas(
        F=F,
        x0=np.array(chute_inicial),
        J=J,
        tol=tolerancia,
        max_iter=max_iteracoes,
        exibir_tabela=True,
    )

    print("\nRESULTADO FINAL")
    if convergiu and solucao is not None:
        print(f"Solução encontrada : {np.array2string(solucao, precision=8)}")
        print(f"F(solução)         : {np.array2string(F(solucao), precision=2)}")
        print(f"Iterações          : {iteracoes}")
    else:
        print("Não foi possível encontrar a solução com os parâmetros fornecidos.")