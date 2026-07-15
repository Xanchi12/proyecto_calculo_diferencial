# -*- coding: utf-8 -*-
"""
Módulo: integracion_numerica.py
=====================================================================
Implementa varios métodos de integración numérica "desde cero" (sin
depender de una caja negra) para que el estudiante pueda justificar su
elección y analizar el error de aproximación, tal como pide la
"Parte 5: Cálculo del área":

    - Regla del Trapecio (compuesta)
    - Regla de Simpson 1/3 (compuesta)
    - Regla de Simpson 3/8
    - Cuadratura de Gauss-Legendre (orden configurable)
    - Integración de Romberg (extrapolación de Richardson sobre
      Trapecio)
    - Monte Carlo (método estocástico, útil para comparar convergencia)

Además incluye un valor de referencia de alta precisión mediante
`scipy.integrate.quad` (cuadratura adaptativa de Gauss-Kronrod), que
se usa como "verdad" para estimar el error real de cada método y
compararlos.

Fundamento del análisis de error
---------------------------------
- Trapecio: error global O(h²), proporcional a f''(ξ).
- Simpson 1/3: error global O(h⁴), proporcional a f⁽⁴⁾(ξ) — mucho más
  preciso para funciones suaves con el mismo número de nodos.
- Gauss-Legendre de n puntos: exacto para polinomios de grado ≤ 2n-1;
  converge muy rápido en funciones suaves con pocos nodos.
- Romberg: mejora Trapecio con extrapolación de Richardson, alcanzando
  convergencia de orden superior sin evaluar derivadas.
- Monte Carlo: error decrece como O(1/√N) — mucho más lento, pero
  robusto ante integrandos muy irregulares o de alta dimensión (aquí
  se incluye únicamente con fines comparativos/educativos).
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy import integrate as sci_integrate


@dataclass
class ResultadoIntegracion:
    metodo: str
    valor: float
    n_evaluaciones: int
    error_estimado: float | None = None  # frente a referencia de alta precisión

    def __str__(self):
        base = f"{self.metodo:<28s} valor = {self.valor: .10f}   (n = {self.n_evaluaciones})"
        if self.error_estimado is not None:
            base += f"   |error| ≈ {self.error_estimado:.3e}"
        return base


class IntegradorNumerico:
    """
    Agrupa distintos métodos de integración numérica sobre una función
    escalar `h(x)` (típicamente |f(x) - g(x)|).

    Parameters
    ----------
    h : callable
        Función vectorizada h(x_array) -> valores. Debe poder evaluarse
        en arreglos NumPy (usar `funcion_numerica()` de
        `FuncionMatematica`, o una combinación de dos de ellas).
    """

    def __init__(self, h):
        self.h = h

    def _h_escalar(self, x: float) -> float:
        return float(self.h(np.array([x]))[0])

    # ------------------------------------------------------------------
    def trapecio(self, a: float, b: float, n: int = 1000) -> ResultadoIntegracion:
        """Regla del trapecio compuesta con n subintervalos."""
        xs = np.linspace(a, b, n + 1)
        ys = self.h(xs)
        ys = np.nan_to_num(ys, nan=0.0)
        hstep = (b - a) / n
        valor = hstep * (ys[0] / 2 + ys[-1] / 2 + np.sum(ys[1:-1]))
        return ResultadoIntegracion("Trapecio compuesto", valor, n + 1)

    def simpson_1_3(self, a: float, b: float, n: int = 1000) -> ResultadoIntegracion:
        """Regla de Simpson 1/3 compuesta. n debe ser par (se ajusta
        automáticamente sumando 1 si es impar)."""
        if n % 2 != 0:
            n += 1
        xs = np.linspace(a, b, n + 1)
        ys = self.h(xs)
        ys = np.nan_to_num(ys, nan=0.0)
        hstep = (b - a) / n
        suma_impares = np.sum(ys[1:-1:2])   # coeficientes 4
        suma_pares = np.sum(ys[2:-1:2])     # coeficientes 2
        valor = (hstep / 3) * (ys[0] + ys[-1] +
                               4 * suma_impares + 2 * suma_pares)
        return ResultadoIntegracion("Simpson 1/3 compuesto", valor, n + 1)

    def simpson_3_8(self, a: float, b: float, n: int = 999) -> ResultadoIntegracion:
        """Regla de Simpson 3/8 compuesta. n debe ser múltiplo de 3."""
        n = n - (n % 3) if n % 3 != 0 else n
        n = max(n, 3)
        xs = np.linspace(a, b, n + 1)
        ys = self.h(xs)
        ys = np.nan_to_num(ys, nan=0.0)
        hstep = (b - a) / n
        suma = ys[0] + ys[-1]
        for i in range(1, n):
            coef = 2 if i % 3 == 0 else 3
            suma += coef * ys[i]
        valor = (3 * hstep / 8) * suma
        return ResultadoIntegracion("Simpson 3/8 compuesto", valor, n + 1)

    def gauss_legendre(self, a: float, b: float, n: int = 20) -> ResultadoIntegracion:
        """Cuadratura de Gauss-Legendre de n nodos, exacta para
        polinomios de grado <= 2n - 1."""
        nodos, pesos = leggauss(n)
        # Cambio de variable de [-1, 1] a [a, b]
        xs = 0.5 * (b - a) * nodos + 0.5 * (b + a)
        ys = self.h(xs)
        ys = np.nan_to_num(ys, nan=0.0)
        valor = 0.5 * (b - a) * np.sum(pesos * ys)
        return ResultadoIntegracion("Gauss-Legendre", valor, n)

    def romberg(self, a: float, b: float, niveles: int = 6) -> ResultadoIntegracion:
        """
        Integración de Romberg: extrapolación de Richardson aplicada a
        aproximaciones sucesivas de la regla del trapecio.
        """
        R = np.zeros((niveles, niveles))
        n_evals = 0
        for i in range(niveles):
            n = 2 ** i
            resultado_trap = self.trapecio(a, b, n)
            R[i, 0] = resultado_trap.valor
            n_evals += resultado_trap.n_evaluaciones
            for k in range(1, i + 1):
                R[i, k] = R[i, k - 1] + \
                    (R[i, k - 1] - R[i - 1, k - 1]) / (4 ** k - 1)
        return ResultadoIntegracion("Romberg", R[niveles - 1, niveles - 1], n_evals)

    def monte_carlo(self, a: float, b: float, n: int = 200_000,
                    semilla: int = 42) -> ResultadoIntegracion:
        """Integración por Monte Carlo simple (muestreo uniforme)."""
        rng = np.random.default_rng(semilla)
        xs = rng.uniform(a, b, n)
        ys = self.h(xs)
        ys = np.nan_to_num(ys, nan=0.0)
        valor = (b - a) * np.mean(ys)
        return ResultadoIntegracion("Monte Carlo", valor, n)

    # ------------------------------------------------------------------
    def referencia_alta_precision(self, a: float, b: float,
                                  puntos_criticos=None) -> ResultadoIntegracion:
        """
        Calcula un valor de referencia de muy alta precisión con
        cuadratura adaptativa de Gauss-Kronrod (scipy.integrate.quad),
        indicando explícitamente los puntos donde el integrando puede
        no ser suave (intersecciones, discontinuidades) para que el
        algoritmo adaptativo refine allí.
        """
        puntos_criticos = puntos_criticos or []
        valor, error_abs = sci_integrate.quad(
            self._h_escalar, a, b, points=sorted(puntos_criticos),
            limit=400, full_output=0
        )
        return ResultadoIntegracion(
            "Referencia (Gauss-Kronrod adaptativa)", valor, -1,
            error_estimado=error_abs
        )

    # ------------------------------------------------------------------
    def comparar_metodos(self, a: float, b: float, puntos_criticos=None,
                         n_trapecio=2000, n_simpson=2000, n_gauss=30):
        """
        Ejecuta todos los métodos y calcula el error de cada uno frente
        a la referencia de alta precisión.

        Returns
        -------
        (ResultadoIntegracion referencia, list[ResultadoIntegracion] métodos)
        """
        referencia = self.referencia_alta_precision(a, b, puntos_criticos)

        metodos = [
            self.trapecio(a, b, n_trapecio),
            self.simpson_1_3(a, b, n_simpson),
            self.simpson_3_8(a, b, n_simpson if n_simpson %
                             3 == 0 else n_simpson + (3 - n_simpson % 3)),
            self.gauss_legendre(a, b, n_gauss),
            self.romberg(a, b, niveles=7),
            self.monte_carlo(a, b, n=300_000),
        ]
        for m in metodos:
            m.error_estimado = abs(m.valor - referencia.valor)

        return referencia, metodos

    def analisis_convergencia(self, a: float, b: float, metodo: str = "simpson",
                              valores_n=(4, 8, 16, 32, 64, 128, 256, 512, 1024)):
        """
        Analiza cómo decrece el error del método elegido a medida que
        aumenta el número de subintervalos n, útil para justificar el
        orden de convergencia empírico observado (Parte 5: "Analizar
        el error de aproximación").

        Returns
        -------
        list[tuple[int, float, float]]
            Lista de (n, valor_aproximado, error_absoluto).
        """
        referencia = self.referencia_alta_precision(a, b).valor
        metodo = metodo.lower()
        func_metodo = {
            "trapecio": self.trapecio,
            "simpson": self.simpson_1_3,
        }.get(metodo)
        if func_metodo is None:
            raise ValueError("metodo debe ser 'trapecio' o 'simpson'")

        resultados = []
        for n in valores_n:
            r = func_metodo(a, b, n)
            error = abs(r.valor - referencia)
            resultados.append((n, r.valor, error))
        return resultados
