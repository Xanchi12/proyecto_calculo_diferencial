# -*- coding: utf-8 -*-
"""
Módulo: funciones.py
=====================================================================
Define la clase `FuncionMatematica`, encargada de representar f(x) o
g(x) como una expresión simbólica de SymPy y de exponer utilidades
para:

    - Evaluarla numéricamente (de forma vectorizada, vía lambdify).
    - Determinar su dominio real de definición.
    - Detectar puntos donde no está definida dentro de un intervalo.
    - Calcular derivadas (útil para análisis de crecimiento/concavidad
      opcional) y límites laterales en un punto.

Esta clase es la base de todo el proyecto: el análisis de
integrabilidad, la búsqueda de intersecciones, la integración numérica
y la visualización dependen de ella para trabajar con f(x) y g(x) de
manera uniforme y segura.

Autor: Proyecto de recuperación - Cálculo Integral
=====================================================================
"""

from __future__ import annotations

import numpy as np
import sympy as sp
from sympy.calculus.util import continuous_domain
from sympy.sets.sets import Interval, Union, FiniteSet, EmptySet


class ErrorFuncionInvalida(Exception):
    """Se lanza cuando la expresión proporcionada no puede interpretarse
    como una función matemática válida de una sola variable real x."""


class FuncionMatematica:
    """
    Representa una función matemática f(x) de una variable real.

    Parameters
    ----------
    expresion_str : str
        Expresión en sintaxis tipo Python/SymPy, por ejemplo:
        "x**2 - 3*x + 2", "sin(x)", "1/(x-2)", "sqrt(4 - x**2)".
    nombre : str, opcional
        Nombre simbólico de la función (para mostrar en reportes y
        gráficas), por defecto "f".

    Attributes
    ----------
    x : sympy.Symbol
        Variable simbólica utilizada en la expresión.
    expr : sympy.Expr
        Expresión simbólica ya parseada y simplificada.
    nombre : str
        Nombre de la función.
    """

    # Diccionario de transformaciones seguras para sympify: evita que se
    # evalúen construcciones peligrosas y añade funciones matemáticas
    # comunes que el estudiante puede escribir en notación natural.
    _LOCALES_PERMITIDOS = {
        "sin": sp.sin, "cos": sp.cos, "tan": sp.tan,
        "asin": sp.asin, "acos": sp.acos, "atan": sp.atan,
        "sinh": sp.sinh, "cosh": sp.cosh, "tanh": sp.tanh,
        "exp": sp.exp, "log": sp.log, "ln": sp.log,
        "sqrt": sp.sqrt, "Abs": sp.Abs, "abs": sp.Abs,
        "pi": sp.pi, "E": sp.E,
    }

    def __init__(self, expresion_str: str, nombre: str = "f"):
        self.x = sp.symbols("x", real=True)
        self.nombre = nombre
        self.expr_original_str = expresion_str
        self.expr = self._parsear(expresion_str)

    # ------------------------------------------------------------------
    # Parsing y utilidades básicas
    # ------------------------------------------------------------------
    def _parsear(self, expresion_str: str) -> sp.Expr:
        """Convierte el string de entrada en una expresión SymPy segura."""
        if not expresion_str or not expresion_str.strip():
            raise ErrorFuncionInvalida("La expresión no puede estar vacía.")
        try:
            expr = sp.sympify(
                expresion_str,
                locals={**self._LOCALES_PERMITIDOS, "x": self.x},
                convert_xor=True,
            )
        except (sp.SympifyError, TypeError, SyntaxError) as err:
            raise ErrorFuncionInvalida(
                f"No se pudo interpretar la expresión '{expresion_str}': {err}"
            ) from err

        # Verificar que la única variable libre sea x (o ninguna, función
        # constante).
        variables_libres = expr.free_symbols - {self.x}
        if variables_libres:
            raise ErrorFuncionInvalida(
                f"La expresión contiene variables no permitidas: "
                f"{variables_libres}. Solo se admite la variable 'x'."
            )
        # IMPORTANTE: no se aplica sp.simplify() aquí a propósito.
        # Simplificar cancelaría factores comunes en expresiones como
        # (x**2 - 4)/(x - 2) -> x + 2, borrando justamente la
        # discontinuidad removible en x=2 que la Parte 2 de la
        # actividad pide detectar y clasificar. Se conserva la
        # expresión tal como fue escrita por el usuario.
        return expr

    def __repr__(self) -> str:
        return f"{self.nombre}(x) = {sp.pretty(self.expr, use_unicode=False)}"

    def __str__(self) -> str:
        return f"{self.nombre}(x) = {self.expr}"

    def latex(self) -> str:
        """Devuelve la representación LaTeX de la función (para reportes)."""
        return f"{self.nombre}(x) = {sp.latex(self.expr)}"

    # ------------------------------------------------------------------
    # Evaluación numérica
    # ------------------------------------------------------------------
    def funcion_numerica(self):
        """
        Devuelve una función Python/NumPy vectorizada equivalente a la
        expresión simbólica, lista para evaluar arreglos de puntos.

        Returns
        -------
        callable
            Función f(x_array) -> valores numéricos (o NaN donde no esté
            definida / no sea real).
        """
        f_lambdificada = sp.lambdify(self.x, self.expr, modules=["numpy"])

        def f_segura(valores):
            valores = np.atleast_1d(np.asarray(valores, dtype=float))
            with np.errstate(divide="ignore", invalid="ignore"):
                try:
                    resultado = f_lambdificada(valores)
                except Exception:
                    # Si falla la evaluación vectorizada (p. ej. por
                    # ramas condicionales), se evalúa punto a punto.
                    resultado = np.array(
                        [self._evaluar_punto(v) for v in valores]
                    )
            resultado = np.asarray(resultado, dtype=complex)
            # Funciones constantes (p. ej. g(x) = 0) producen un array
            # 0-dimensional al ser lambdificadas: se expande al tamaño
            # de la entrada para mantener siempre un array 1-D.
            if resultado.ndim == 0:
                resultado = np.full(valores.shape, resultado, dtype=complex)
            # Descartar partes imaginarias residuales por error numérico.
            parte_real = np.where(
                np.abs(resultado.imag) < 1e-9, resultado.real, np.nan
            )
            return parte_real

        return f_segura

    def _evaluar_punto(self, valor: float) -> float:
        """Evalúa la expresión simbólica en un único punto, devolviendo
        NaN si no está definida o el resultado no es real."""
        try:
            resultado = self.expr.subs(self.x, valor)
            resultado = complex(resultado.evalf())
            if abs(resultado.imag) > 1e-9:
                return float("nan")
            return resultado.real
        except (TypeError, ValueError, ZeroDivisionError):
            return float("nan")

    def evaluar(self, valor: float) -> float:
        """Evalúa f(valor) devolviendo un escalar (float o nan)."""
        return float(self.funcion_numerica()(np.array([valor]))[0])

    # ------------------------------------------------------------------
    # Análisis simbólico de dominio
    # ------------------------------------------------------------------
    def dominio_real(self):
        """
        Calcula el dominio real de la función mediante SymPy.

        Returns
        -------
        sympy.Set
            Conjunto (posiblemente unión de intervalos) donde la función
            está definida y toma valores reales.
        """
        try:
            return continuous_domain(self.expr, self.x, sp.S.Reals)
        except (NotImplementedError, TypeError):
            # Si SymPy no logra determinarlo simbólicamente, se asume
            # que el dominio se explorará numéricamente en su lugar.
            return sp.S.Reals

    def esta_definida_en(self, punto: float) -> bool:
        """Indica si el dominio simbólico incluye un punto dado."""
        dominio = self.dominio_real()
        try:
            return bool(dominio.contains(punto))
        except TypeError:
            valor = self._evaluar_punto(punto)
            return not np.isnan(valor)

    def derivada(self, orden: int = 1) -> sp.Expr:
        """Devuelve la derivada simbólica de orden dado."""
        return sp.diff(self.expr, self.x, orden)

    def limite(self, punto: float, direccion: str = "+"):
        """
        Calcula el límite lateral de la función en un punto.

        Parameters
        ----------
        punto : float
        direccion : str
            '+' para límite por la derecha, '-' para límite por la
            izquierda.
        """
        return sp.limit(self.expr, self.x, punto, dir=direccion)

    def diferencia_con(self, otra: "FuncionMatematica") -> sp.Expr:
        """Devuelve la expresión simbólica f(x) - g(x)."""
        return sp.simplify(self.expr - otra.expr)
