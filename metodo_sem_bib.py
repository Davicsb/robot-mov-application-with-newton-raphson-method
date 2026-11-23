import math
from typing import Callable, List, Optional, Tuple

Vetor = List[float]
Matriz = List[List[float]]


def norma_infinito(v: Vetor) -> float:
    """Retorna o maior valor absoluto dos elementos do vetor."""
    return max(abs(valor) for valor in v)


def formatar_vetor(v: Vetor, precisao: int = 6) -> str:
    """Formata um vetor como string (substitui np.array2string)."""
    elementos = ", ".join(f"{valor:.{precisao}f}" for valor in v)
    return f"[{elementos}]"


def resolver_sistema_linear(A: Matriz, b: Vetor) -> Vetor:
    """Resolve A * x = b por Eliminação de Gauss com pivoteamento parcial."""
    n = len(b)
    M = [linha[:] for linha in A]
    rhs = b[:]

    # 1. Eliminação progressiva
    for i in range(n):
        pivo_linha = max(range(i, n), key=lambda r: abs(M[r][i]))
        if abs(M[pivo_linha][i]) < 1e-12:
            raise ValueError("Matriz singular.")

        M[i], M[pivo_linha] = M[pivo_linha], M[i]
        rhs[i], rhs[pivo_linha] = rhs[pivo_linha], rhs[i]

        for j in range(i + 1, n):
            fator = M[j][i] / M[i][i]
            for k in range(i, n):
                M[j][k] -= fator * M[i][k]
            rhs[j] -= fator * rhs[i]

    # 2. Substituição regressiva
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        soma = sum(M[i][j] * x[j] for j in range(i + 1, n))
        x[i] = (rhs[i] - soma) / M[i][i]

    return x


def jacobiana_numerica(
    F: Callable[[Vetor], Vetor],
    x: Vetor,
    h: float = 1e-7,
) -> Matriz:
    """Calcula a matriz Jacobiana de F(x) numericamente por diferenças finitas centrais."""
    n = len(x)
    J: Matriz = [[0.0] * n for _ in range(n)]

    for j in range(n):
        x_mais = x[:]
        x_menos = x[:]
        x_mais[j] += h
        x_menos[j] -= h

        f_mais = F(x_mais)
        f_menos = F(x_menos)

        for i in range(n):
            J[i][j] = (f_mais[i] - f_menos[i]) / (2.0 * h)

    return J


def metodo_newton_sistemas(
    F: Callable[[Vetor], Vetor],
    x0: Vetor,
    J: Optional[Callable[[Vetor], Matriz]] = None,
    tol: float = 1e-6,
    max_iter: int = 100,
    exibir_tabela: bool = True,
) -> Tuple[Optional[Vetor], int, bool, List[Vetor]]:
    """
    Resolve um sistema de equações não lineares F(x) = 0 pelo Método de Newton.

    Parâmetros:
        F: Função vetorial que recebe uma lista x e retorna a lista F(x).
        x0: Lista com o chute inicial.
        J: Função que retorna a matriz Jacobiana J(x). Se None, calcula numericamente.
        tol: Tolerância para a norma do passo e de F(x).
        max_iter: Número máximo de iterações.
        exibir_tabela: Se True, imprime o progresso no terminal.

    Retorna:
        (solucao, iteracoes, convergiu, historico)
    """
    x: Vetor = [float(v) for v in x0]
    historico: List[Vetor] = [x[:]]

    if exibir_tabela:
        print(f"{'Iter':<6} | {'||F(x_k)||':<14} | {'||Delta x||':<14} | {'x_k'}")

    for k in range(1, max_iter + 1):
        Fx = [float(v) for v in F(x)]
        Jx = J(x) if J is not None else jacobiana_numerica(F, x)
        menos_Fx = [-v for v in Fx]

        try:
            # Resolve J(x) * delta_x = -F(x)
            delta_x = resolver_sistema_linear(Jx, menos_Fx)
        except ValueError:
            if exibir_tabela:
                print(f"\n[Erro] Matriz Jacobiana singular na iteração {k}. Método interrompido.")
            return None, k, False, historico

        x_novo = [xi + di for xi, di in zip(x, delta_x)]
        historico.append(x_novo[:])

        norma_passo = norma_infinito(delta_x)
        norma_F = norma_infinito(F(x_novo))

        if exibir_tabela:
            x_formatado = formatar_vetor(x_novo, precisao=6)
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
    def F(v: Vetor) -> Vetor:
        x, y = v
        return [
            x**2 + y**2 - 4,
            math.exp(x) + y - 1,
        ]

    J = None  # Deixe como None para usar a Jacobiana numérica
    chute_inicial = [1.0, -1.0]
    tolerancia = 1e-8
    max_iteracoes = 50

    solucao, iteracoes, convergiu, _ = metodo_newton_sistemas(
        F=F,
        x0=chute_inicial,
        J=J,
        tol=tolerancia,
        max_iter=max_iteracoes,
        exibir_tabela=True,
    )

    print("\nRESULTADO FINAL")
    if convergiu and solucao is not None:
        print(f"Solução encontrada : {formatar_vetor(solucao, precisao=8)}")
        print(f"F(solução)         : {formatar_vetor(F(solucao), precisao=2)}")
        print(f"Iterações          : {iteracoes}")
    else:
        print("Não foi possível encontrar a solução com os parâmetros fornecidos.")