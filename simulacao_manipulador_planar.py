import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple

from metodo_newton_sistemas import metodo_newton_sistemas


def cinematica_direta(theta: np.ndarray, L1: float, L2: float) -> Tuple[np.ndarray, np.ndarray]:
    """Retorna as posições (x, y) do cotovelo e do efetuador."""
    t1, t2 = theta
    cotovelo = np.array([L1 * np.cos(t1), L1 * np.sin(t1)])
    efetuador = cotovelo + np.array([L2 * np.cos(t1 + t2), L2 * np.sin(t1 + t2)])
    return cotovelo, efetuador


class ManipuladorInterativo:
    def __init__(self, L1: float, L2: float, theta_inicial_graus: list[float]):
        self.L1 = L1
        self.L2 = L2
        self.alcance_max = L1 + L2
        self.alcance_min = abs(L1 - L2)
        self.theta_atual = np.radians(theta_inicial_graus)

        # Configuração inicial da figura
        self.fig, self.ax = plt.subplots(figsize=(7, 7))
        self._configurar_grafico()

        # Conecta o evento de clique do mouse
        self.fig.canvas.mpl_connect("button_press_event", self._ao_clicar)

    def _configurar_grafico(self) -> None:
        # Limites do espaço de trabalho (workspace)
        self.ax.add_patch(
            plt.Circle((0, 0), self.alcance_max, color="gray", fill=False, linestyle="--", alpha=0.5)
        )
        if self.alcance_min > 0:
            self.ax.add_patch(
                plt.Circle((0, 0), self.alcance_min, color="gray", fill=False, linestyle=":", alpha=0.5)
            )

        cotovelo, efetuador = cinematica_direta(self.theta_atual, self.L1, self.L2)

        # Objetos gráficos atualizáveis (evita redesenhar tudo a cada clique)
        (self.linha_robo,) = self.ax.plot(
            [0.0, cotovelo[0], efetuador[0]],
            [0.0, cotovelo[1], efetuador[1]],
            "-o",
            linewidth=4,
            markersize=8,
            color="#1f77b4",
            label="Manipulador 2R",
        )
        self.ax.plot(0, 0, "ks", markersize=10, label="Base")
        (self.ponto_alvo,) = self.ax.plot(
            [efetuador[0]], [efetuador[1]], "rx", markersize=12, markeredgewidth=2.5, label="Alvo"
        )

        margem = self.alcance_max + 0.3
        self.ax.set_xlim(-margem, margem)
        self.ax.set_ylim(-margem, margem)
        self.ax.set_aspect("equal")
        self.ax.grid(True, alpha=0.3)
        self.ax.legend(loc="upper right")
        self.ax.set_title("Clique em qualquer ponto dentro do círculo para mover o robô")

    def _jacobiana(self, theta: np.ndarray) -> np.ndarray:
        t1, t2 = theta
        s1, c1 = np.sin(t1), np.cos(t1)
        s12, c12 = np.sin(t1 + t2), np.cos(t1 + t2)
        return np.array([
            [-self.L1 * s1 - self.L2 * s12, -self.L2 * s12],
            [ self.L1 * c1 + self.L2 * c12,  self.L2 * c12],
        ])

    def _ao_clicar(self, event) -> None:
        # Ignora cliques fora da área do gráfico
        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return

        alvo = np.array([event.xdata, event.ydata])
        dist = float(np.linalg.norm(alvo))

        # Valida se o clique está dentro do espaço de trabalho
        if not (self.alcance_min <= dist <= self.alcance_max):
            self.ax.set_title(
                f"Fora de alcance! Distância = {dist:.2f} m (limites: {self.alcance_min:.2f} a {self.alcance_max:.2f} m)",
                color="red",
            )
            self.fig.canvas.draw_idle()
            return

        def F_cinematica(theta: np.ndarray) -> np.ndarray:
            _, efetuador = cinematica_direta(theta, self.L1, self.L2)
            return efetuador - alvo

        # Resolve usando a pose atual como chute inicial (warm start)
        theta_novo, iteracoes, convergiu, _ = metodo_newton_sistemas(
            F=F_cinematica,
            x0=self.theta_atual,
            J=self._jacobiana,
            tol=1e-6,
            max_iter=30,
            exibir_tabela=False,
        )

        if convergiu and theta_novo is not None:
            self.theta_atual = theta_novo
            cotovelo, efetuador = cinematica_direta(self.theta_atual, self.L1, self.L2)

            # Atualiza apenas as coordenadas dos objetos já desenhados
            self.linha_robo.set_data([0.0, cotovelo[0], efetuador[0]], [0.0, cotovelo[1], efetuador[1]])
            self.ponto_alvo.set_data([alvo[0]], [alvo[1]])

            angulos = np.degrees((self.theta_atual + np.pi) % (2 * np.pi) - np.pi)
            self.ax.set_title(
                f"Alvo: ({alvo[0]:.2f}, {alvo[1]:.2f}) | Iter: {iteracoes} | "
                f"θ1={angulos[0]:.1f}°, θ2={angulos[1]:.1f}°",
                color="black",
            )
        else:
            self.ax.set_title(
                "Não convergiu (próximo a singularidade). Tente outro ponto.",
                color="darkorange",
            )

        self.fig.canvas.draw_idle()

    def iniciar(self) -> None:
        plt.show()


# ÁREA DE ENTRADA
if __name__ == "__main__":
    L1 = 1.0
    L2 = 0.8
    chute_inicial_graus = [30.0, 45.0]

    sim = ManipuladorInterativo(L1=L1, L2=L2, theta_inicial_graus=chute_inicial_graus)
    sim.iniciar()