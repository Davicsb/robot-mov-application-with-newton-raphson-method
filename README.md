# Método de Newton e Cinemática Inversa de Manipulador Planar 2R

Implementação modular em Python do **Método de Newton-Raphson** para resolução de equações e sistemas de equações não lineares, aplicada a uma simulação interativa de **cinemática inversa** de um manipulador robótico planar de 2 graus de liberdade (2R).

---

## Estrutura do Repositório

```text
.
├── metodo_newton.py                 # Solver escalar f(x) = 0 (com derivada numérica opcional)
├── metodo_newton_sistemas.py        # Solver multivariável F(x) = 0 com Jacobiana analítica ou numérica
└── simulacao_manipulador_planar.py  # Simulação interativa 2R usando o solver de sistemas
```

---

## Pré-requisitos

* Python 3.9+
* `numpy` (manipulação de vetores/matrizes e resolução de sistemas lineares)
* `matplotlib` (visualização e interatividade da simulação)

Instale as dependências com:

```bash
pip install numpy matplotlib
```

---

## Como Executar

### 1. Equação Não Linear Escalar
Resolve $f(x) = 0$ usando apenas a biblioteca padrão (`math`):

```bash
python metodo_newton.py
```

### 2. Sistema de Equações Não Lineares (Standalone)
Resolve sistemas $F(x) = 0$ ($F: \mathbb{R}^n \to \mathbb{R}^n$) via linearização pela matriz Jacobiana $J(x_k)\Delta x_k = -F(x_k)$ e atualização $x_{k+1} = x_k + \Delta x_k$:

```bash
python metodo_newton_sistemas.py
```

> O arquivo usa `if __name__ == "__main__":`, permitindo ser executado diretamente no terminal ou importado por outros scripts sem disparar o exemplo padrão.

### 3. Simulação Interativa do Manipulador Planar 2R
Abre uma janela gráfica do Matplotlib com o espaço de trabalho (*workspace*) do manipulador:

```bash
python simulacao_manipulador_planar.py
```

* **Interação:** Clique em qualquer ponto dentro dos limites do círculo para calcular a cinemática inversa em tempo real.
* **Warm Start:** A pose atual das juntas é reutilizada como chute inicial do próximo clique, acelerando a convergência.

---

## Visão Geral Técnica

### Solver Multivariável (`metodo_newton_sistemas.py`)
* **Jacobiana Numérica (`jacobiana_numerica`):** Quando `J=None`, aproxima as derivadas parciais automaticamente por diferenças finitas centrais com erro de truncamento $O(h^2)$ e passo padrão $h = 10^{-7}$.
* **Estabilidade Numérica:** Resolve $J(x)\Delta x = -F(x)$ com `np.linalg.solve` (decomposição LU) em vez de inverter a matriz explicitamente, capturando `np.linalg.LinAlgError` caso a Jacobiana seja singular.
* **Critérios de Parada:** Interrompe quando $\|\Delta x\|_\infty < \text{tol}$ ou $\|F(x_{\text{novo}})\|_\infty < \text{tol}$, retornando `(solucao, iteracoes, convergiu, historico)`.

### Cinemática Inversa (`simulacao_manipulador_planar.py`)
Resolve o sistema não linear da cinemática direta do manipulador 2R:

$$L_1 \cos(\theta_1) + L_2 \cos(\theta_1 + \theta_2) - x_{\text{alvo}} = 0$$
$$L_1 \sin(\theta_1) + L_2 \sin(\theta_1 + \theta_2) - y_{\text{alvo}} = 0$$

---

## Pontos de Atenção

* **Chute inicial (`x0`):** O Método de Newton tem convergência quadrática local, mas pode divergir se o chute inicial estiver longe da raiz ou próximo a singularidades.
* **Singularidades:** Configurações com o braço totalmente esticado ($\theta_2 = 0^\circ$) ou dobrado ($\theta_2 = 180^\circ$) tornam a Jacobiana singular. Para lidar com essas regiões sem travar, o solver pode ser estendido com *Damped Least Squares* (Levenberg-Marquardt).
* **Desempenho:** Para sistemas grandes ou aplicações em tempo real, prefira fornecer a Jacobiana analítica `J` em vez da aproximação numérica (que exige $2n$ avaliações extras de $F$ por iteração).