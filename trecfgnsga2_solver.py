# trecfgnsga2.py
# -------------------------------------------------------------
# Traducción casi literal de la clase MATLAB  «trecfgnsga2»
#  ––  Depende de una super-clase `TNSGA2` ya portada a Python
#  ––  Depende de un objeto `net` con la misma interfaz que en MATLAB
# -------------------------------------------------------------

import numpy as np
from tnsga2 import TNSGA2

class TRecfGNSGA2(TNSGA2):          # ← hereda de tu base NSGA-II
    """
    Puerto directo de la clase MATLAB `trecfgnsga2 < tnsga2`.
    Sólo se han cambiado las mínimas cosas para que rule en Python:
        • Indexación 0-based
        • rng.choice / rng.integers en vez de randsample / randi
        • print en vez de fprintf
    """

    # ───────────────────────────
    # Constructor
    # ───────────────────────────
    def __init__(self, net, npop: int, pop: np.ndarray | None):
        super().__init__()                   # inicializa la base

        rng = np.random.default_rng()        # generador local
        self.net = net                       # red de flujos / evaluador

        L = self.net.MT - self.net.NT + 1    # nº de variables

        # ----- parámetros clásicos del GA/NSGA-II -----
        self.npop   = npop                   # tamaño de población
        self.ngen   = 10_000                 # generaciones máximas
        self.pcross = 0.8                    # prob. crossover
        self.crossf = 0.1                    # fracción genes intercambiados
        self.sigma  = 0.1                    # σ mutación gaussiana
        self.maxit1 = 5
        self.maxit2 = 50

        # ----- especificación del problema -----
        self.nvars  = L
        self.nparm  = 0
        self.nobjs  = 2
        self.objstr = ['Cost', '-Vmin']
        self.verb   = True

        # ----- población inicial -----
        if pop is not None and len(pop) > 0:
            # Se suministra una población pre-calculada
            self.variable = pop.copy()

        else:
            # Se genera una población nueva → 2*npop individuos
            self.variable = np.zeros((2 * npop, L + self.nobjs))

            # 1) Primer individuo: mejora del caso base
            y = self.net.improve(self.net.status0["opened"], 2)
            self.variable[0, :] = np.hstack(
                (y, self.net.status0["fun"][2 : 2 + self.nobjs])
            )
            if self.net.isimprove:
                self.maxit2 = 20

            if self.verb:
                print('+', end='', flush=True)

            # 2) Resto de la población
            for i in range(1, 2 * npop):
                # Genera red primaria hasta conseguir un caso sin error
                while True:
                    _, y = self.net.prim()          # devuelve (error_flag, config)
                    self.net.evaluate_off(y)
                    if self.net.status["error"] == 0:
                        break

                if self.verb:
                    print('+', end='', flush=True)

                # Lanza mejora con un objetivo aleatorio (3…nobjs+2)
                y = self.net.improve(
                    self.net.status["opened"], 2 + rng.integers(1, self.nobjs + 1)
                )
                self.variable[i, :] = np.hstack(
                    (y, self.net.status["fun"][2 : 2 + self.nobjs])
                )

            if self.verb:
                print()                            # salto de línea final

    # ───────────────────────────
    # Objetivos (invocado por NSGA-II)
    # ───────────────────────────
    def evaluate_objective(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        x : vector de variables (long L)
        Devuelve
            fobj : ndarray  (nobjs,)
            fpar : ndarray  (nparm,)   → aquí vacío
        """
        self.net.evaluate_off(x)
        fobj = self.net.status['fun'][2 : 2 + self.nobjs]   # 0-based
        fpar = np.array([], dtype=float)
        return fobj, fpar

    # ───────────────────────────
    # *** opcional ***  (no había código en MATLAB)
    # ───────────────────────────
    def generate(self):
        """stub para compatibilidad; implementar si hace falta"""
        pass

    # ───────────────────────────
    # Operador de Crossover
    # ───────────────────────────
    def crossover(self, p1: np.ndarray, p2: np.ndarray):
        rng = np.random.default_rng()
        L = self.net.MT - self.net.NT + 1

        # nº de genes a intercambiar (≥1)
        k = rng.integers(1, max(1, round(L * self.crossf)) + 1)
        u = rng.choice(L, size=k, replace=False)

        # Uniform crossover implementado por el objeto `net`
        y1 = self.net.uniformcrossover(p1, p2, u)
        y2 = self.net.uniformcrossover(p2, p1, u)

        # Post-proceso de mejora
        y1 = self.net.improve(y1, 2 + rng.integers(1, self.nobjs + 1))
        y2 = self.net.improve(y2, 2 + rng.integers(1, self.nobjs + 1))

        return y1, y2

    # ───────────────────────────
    # Operador de Mutación
    # ───────────────────────────
    def mutate(self, p1: np.ndarray):
        rng = np.random.default_rng()

        y1 = self.net.mutation(p1, self.sigma)
        y1 = self.net.improve(y1, 2 + rng.integers(1, self.nobjs + 1))
        return y1
