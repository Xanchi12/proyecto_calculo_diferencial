# -*- coding: utf-8 -*-
"""
Módulo: intersecciones.py
=====================================================================
Implementa la "Parte 3: Análisis de intersecciones" de la actividad:

    - ¿Las funciones f(x) y g(x) se intersectan en [a, b]?
    - ¿En qué puntos ocurre la intersección?
    - ¿Cómo afectan estos puntos al cálculo del área?

Estrategia
----------
1. Se intenta resolver simbólicamente f(x) - g(x) = 0 con
   `sympy.solveset` restringido al intervalo [a, b].
2. Si SymPy no logra una solución cerrada (expresión trascendente
   complicada), se recurre a un método numérico robusto: se evalúa
   h(x) = f(x) - g(x) sobre una malla fina y se detectan cambios de
   signo, refinando cada raíz con el método de Brent
   (`scipy.optimize.brentq`), que garantiza convergencia si h cambia
   de signo en el subintervalo.
3. Los extremos del intervalo [a, b] siempre se incluyen como "puntos
   de corte" del particionado, de modo que el área entre curvas pueda
   calcularse tramo a tramo aunque no exista intersección real.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import sympy as sp
from scipy.optimize import brentq

from funciones import FuncionMatematica


@dataclass
class PuntoInterseccion:
    x: float
    y: float
    metodo: str  # "simbólico" o "numérico (Brent)"

    def __str__(self):
        return f"({self.x:.6g}, {self.y:.6g})  [{self.metodo}]"


class BuscadorIntersecciones:
    """Encuentra los puntos donde f(x) = g(x) dentro de un intervalo."""

    def __init__(self, f: FuncionMatematica, g: FuncionMatematica,
                 puntos_malla: int = 20000):
        self.f = f
        self.g = g
        self.puntos_malla = puntos_malla

    # ------------------------------------------------------------------
    def _raices_simbolicas(self, a: float, b: float):
        x = self.f.x
        diferencia = self.f.diferencia_con(self.g)
        raices = []
        try:
            soluciones = sp.solveset(
                sp.Eq(diferencia, 0), x, domain=sp.Interval(a, b)
            )
            if soluciones.is_FiniteSet:
                for s in soluciones:
                    if s.is_real:
                        raices.append(float(s))
        except (NotImplementedError, TypeError):
            pass
        return raices

    def _raices_numericas(self, a: float, b: float, ya_encontradas):
        f_num = self.f.funcion_numerica()
        g_num = self.g.funcion_numerica()

        def h(t):
            return f_num(np.array([t]))[0] - g_num(np.array([t]))[0]

        xs = np.linspace(a, b, self.puntos_malla)
        hs = np.array([h(v) if np.isfinite(v) else np.nan for v in xs])

        raices = []
        for i in range(len(xs) - 1):
            y0, y1 = hs[i], hs[i + 1]
            if not (np.isfinite(y0) and np.isfinite(y1)):
                continue
            if y0 == 0:
                raices.append(xs[i])
                continue
            if y0 * y1 < 0:
                try:
                    raiz = brentq(h, xs[i], xs[i + 1], xtol=1e-12, maxiter=200)
                    # Evitar duplicar raíces ya encontradas simbólicamente
                    if not any(abs(raiz - r) < 1e-6 for r in ya_encontradas):
                        raices.append(raiz)
                except (ValueError, RuntimeError):
                    continue
        return raices

    # ------------------------------------------------------------------
    def encontrar(self, a: float, b: float):
        """
        Busca todas las intersecciones de f y g en [a, b].

        Returns
        -------
        list[PuntoInterseccion]
            Ordenada por valor de x, sin duplicados.
        """
        if a >= b:
            raise ValueError("Se requiere a < b.")

        simbolicas = self._raices_simbolicas(a, b)
        numericas = self._raices_numericas(a, b, simbolicas)

        puntos = []
        f_num = self.f.funcion_numerica()

        for r in simbolicas:
            y = f_num(np.array([r]))[0]
            puntos.append(PuntoInterseccion(r, y, "simbólico"))
        for r in numericas:
            y = f_num(np.array([r]))[0]
            puntos.append(PuntoInterseccion(r, y, "numérico (Brent)"))

        puntos.sort(key=lambda p: p.x)
        return _deduplicar_puntos(puntos)

    def particion_por_intersecciones(self, a: float, b: float):
        """
        Devuelve la lista ordenada de "puntos de corte" del intervalo
        [a, b]: los extremos a y b más todas las intersecciones
        encontradas. Esta partición define los subintervalos en los
        que una de las dos funciones domina consistentemente sobre la
        otra (necesarios para aplicar correctamente el valor absoluto
        del área, ver Parte 4).
        """
        intersecciones = self.encontrar(a, b)
        cortes = [a] + [p.x for p in intersecciones] + [b]
        cortes = sorted(set(round(c, 12) for c in cortes))
        # Eliminar cortes casi-idénticos a los extremos
        cortes = _deduplicar(cortes, tol=1e-8)
        return cortes


def _deduplicar_puntos(puntos, tol=1e-6):
    if not puntos:
        return []
    resultado = [puntos[0]]
    for p in puntos[1:]:
        if abs(p.x - resultado[-1].x) > tol:
            resultado.append(p)
    return resultado


def _deduplicar(valores, tol=1e-8):
    if not valores:
        return []
    resultado = [valores[0]]
    for v in valores[1:]:
        if abs(v - resultado[-1]) > tol:
            resultado.append(v)
    return resultado
