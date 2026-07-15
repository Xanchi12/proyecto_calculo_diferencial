# -*- coding: utf-8 -*-
"""
Módulo: visualizacion.py
=====================================================================
Implementa la "Parte 6: Visualización" de la actividad: generar una
representación gráfica donde se muestren ambas funciones, se
identifique (sombree) el área entre ellas y se distingan claramente
los puntos de intersección y cualquier discontinuidad relevante.
"""

from __future__ import annotations
from area_entre_curvas import ResultadoAreaEntreCurvas
from funciones import FuncionMatematica
import matplotlib.pyplot as plt

import base64
import io

import numpy as np
import matplotlib
matplotlib.use("Agg")  # backend sin ventana, apto para entornos sin GUI


def _construir_figura_area(resultado: ResultadoAreaEntreCurvas,
                           puntos_malla: int = 2000,
                           mostrar_subareas: bool = True):
    """Construye (sin guardar) la figura de matplotlib del área entre
    curvas. Función compartida por `graficar_area_entre_curvas` (que la
    guarda en disco) y `area_entre_curvas_a_base64` (que la codifica
    para incrustarla directamente en una página web)."""
    f, g = resultado.f, resultado.g
    a, b = resultado.a, resultado.b
    margen = 0.05 * (b - a) if b > a else 1.0

    xs = np.linspace(a, b, puntos_malla)
    f_num = f.funcion_numerica()
    g_num = g.funcion_numerica()
    ys_f = f_num(xs)
    ys_g = g_num(xs)

    fig, ax = plt.subplots(figsize=(10, 6.5), dpi=150)

    ax.plot(
        xs, ys_f, label=f"${f.nombre}(x) = {_a_latex(f)}$", color="#1f77b4", linewidth=2)
    ax.plot(
        xs, ys_g, label=f"${g.nombre}(x) = {_a_latex(g)}$", color="#d62728", linewidth=2)

    colores_subareas = plt.cm.viridis(np.linspace(
        0.15, 0.85, max(len(resultado.subintervalos), 1)))

    for idx, sub in enumerate(resultado.subintervalos):
        xs_sub = np.linspace(sub.x_izq, sub.x_der, max(
            int(puntos_malla / max(len(resultado.subintervalos), 1)), 50))
        yf_sub = f_num(xs_sub)
        yg_sub = g_num(xs_sub)
        color = colores_subareas[idx] if mostrar_subareas else "#7f7f7f"
        ax.fill_between(
            xs_sub, yf_sub, yg_sub, color=color, alpha=0.35,
            label=(f"Área [{sub.x_izq:.3g}, {sub.x_der:.3g}] "
                   f"(domina {sub.funcion_dominante}) = {sub.area_parcial:.4f}")
            if mostrar_subareas else None,
        )

    # Puntos de intersección
    if resultado.intersecciones:
        xs_int = [p.x for p in resultado.intersecciones]
        ys_int = [p.y for p in resultado.intersecciones]
        ax.scatter(xs_int, ys_int, color="black", zorder=5, s=55,
                   marker="o", label="Puntos de intersección")
        for p in resultado.intersecciones:
            ax.annotate(
                f"({p.x:.3g}, {p.y:.3g})",
                (p.x, p.y), textcoords="offset points", xytext=(8, 10),
                fontsize=8.5,
            )

    # Discontinuidades detectadas (si las hay) marcadas con línea punteada
    puntos_discontinuidad = set()
    for sub in resultado.subintervalos:
        for d in (sub.reporte_integrabilidad_f.discontinuidades +
                  sub.reporte_integrabilidad_g.discontinuidades):
            puntos_discontinuidad.add(round(d.punto, 8))
    for xp in puntos_discontinuidad:
        ax.axvline(xp, color="gray", linestyle=":", linewidth=1.2, alpha=0.8)

    ax.set_xlim(a - margen, b + margen)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(
        f"Área entre {f.nombre}(x) y {g.nombre}(x) en [{a}, {b}]\n"
        f"Área total ≈ {resultado.area_total:.6f}   "
        f"(error estimado ≈ {resultado.error_estimado:.2e})",
        fontsize=12,
    )
    ax.legend(loc="best", fontsize=8, framealpha=0.9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def graficar_area_entre_curvas(resultado: ResultadoAreaEntreCurvas,
                               ruta_salida: str = "area_entre_curvas.png",
                               puntos_malla: int = 2000,
                               mostrar_subareas: bool = True):
    """
    Genera y guarda en disco una figura con las curvas, el área
    sombreada por subintervalo, los puntos de intersección y las
    discontinuidades relevantes (Parte 6 de la actividad).
    """
    fig = _construir_figura_area(resultado, puntos_malla, mostrar_subareas)
    fig.savefig(ruta_salida)
    plt.close(fig)
    return ruta_salida


def area_entre_curvas_a_base64(resultado: ResultadoAreaEntreCurvas,
                               puntos_malla: int = 2000,
                               mostrar_subareas: bool = True) -> str:
    """
    Igual que `graficar_area_entre_curvas`, pero en vez de guardar un
    archivo en disco devuelve la imagen codificada en base64, lista
    para incrustarse directamente en una página HTML
    (``<img src="data:image/png;base64,...">``). Se usa en la interfaz
    web para mostrar la gráfica sin depender del sistema de archivos.
    """
    fig = _construir_figura_area(resultado, puntos_malla, mostrar_subareas)
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png")
    plt.close(fig)
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("ascii")


def _construir_figura_convergencia(resultados_convergencia,
                                   metodo_nombre="Simpson 1/3"):
    ns = np.array([r[0] for r in resultados_convergencia], dtype=float)
    errores = np.array([max(r[2], 1e-16) for r in resultados_convergencia])

    fig, ax = plt.subplots(figsize=(7.5, 5.5), dpi=150)
    ax.loglog(ns, errores, "o-", color="#2ca02c",
              label=f"Error {metodo_nombre}")

    orden_teorico = 4 if "simpson" in metodo_nombre.lower() else 2
    referencia = errores[0] * (ns[0] / ns) ** orden_teorico
    ax.loglog(ns, referencia, "--", color="gray",
              label=f"Pendiente teórica O(n^-{orden_teorico})")

    ax.set_xlabel("Número de subintervalos (n)")
    ax.set_ylabel("Error absoluto")
    ax.set_title(f"Convergencia del método: {metodo_nombre}")
    ax.legend()
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    return fig


def graficar_convergencia(resultados_convergencia, ruta_salida="convergencia.png",
                          metodo_nombre="Simpson 1/3"):
    """
    Grafica en escala log-log el error absoluto de un método numérico
    en función del número de subintervalos n, para visualizar
    empíricamente su orden de convergencia (parte del análisis de
    error pedido en la Parte 5).

    Parameters
    ----------
    resultados_convergencia : list[tuple[int, float, float]]
        Salida de `IntegradorNumerico.analisis_convergencia`: lista de
        (n, valor_aproximado, error_absoluto).
    """
    fig = _construir_figura_convergencia(
        resultados_convergencia, metodo_nombre)
    fig.savefig(ruta_salida)
    plt.close(fig)
    return ruta_salida


def convergencia_a_base64(resultados_convergencia, metodo_nombre="Simpson 1/3") -> str:
    """Igual que `graficar_convergencia`, pero devuelve la imagen
    codificada en base64 en vez de guardarla en disco."""
    fig = _construir_figura_convergencia(
        resultados_convergencia, metodo_nombre)
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png")
    plt.close(fig)
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("ascii")


def _a_latex(func: FuncionMatematica) -> str:
    """Devuelve el LaTeX de la expresión, con manejo de errores para
    que la figura nunca falle por un problema de formato."""
    try:
        import sympy as sp
        return sp.latex(func.expr)
    except Exception:
        return str(func.expr)
