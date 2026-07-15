# -*- coding: utf-8 -*-
"""
Módulo: main.py
=====================================================================
Programa principal del proyecto "Área entre Curvas, Integrabilidad y
Programación".

Uso
---
Modo interactivo (pide f, g, a, b por teclado):

    python main.py

Modo con argumentos (útil para pruebas rápidas o para el reporte):

    python main.py --f "x**2 - 1" --g "1 - x**2/2" --a -2 --b 2

Modo con ejemplos predefinidos que ilustran distintos casos
(intersección simple, discontinuidad removible, asíntota vertical con
integral impropia convergente y divergente):

    python main.py --ejemplo 1
    python main.py --ejemplo 2
    python main.py --ejemplo 3
    python main.py --ejemplo 4
"""

from __future__ import annotations

import argparse
import os
import sys
import webbrowser

from funciones import FuncionMatematica, ErrorFuncionInvalida
from integrabilidad import AnalizadorIntegrabilidad
from area_entre_curvas import CalculadoraAreaEntreCurvas
from visualizacion import graficar_area_entre_curvas, graficar_convergencia
from integracion_numerica import IntegradorNumerico


def abrir_en_navegador(ruta: str):
    """Abre un archivo de imagen en el navegador predeterminado, como si
    fuera una página web (útil para ver la gráfica sin tener que
    buscarla manualmente en el explorador de archivos)."""
    ruta_absoluta = os.path.abspath(ruta)
    try:
        webbrowser.open(f"file://{ruta_absoluta}")
    except Exception as err:
        print(f"[Aviso] No se pudo abrir automáticamente el navegador: {err}")
        print(f"Puedes abrir la imagen manualmente en: {ruta_absoluta}")


EJEMPLOS = {
    "1": {
        "descripcion": "Dos parábolas que se cruzan en dos puntos (caso clásico).",
        "f": "x**2", "g": "-x**2 + 4", "a": -3, "b": 3,
    },
    "2": {
        "descripcion": "f tiene una discontinuidad removible dentro del intervalo.",
        "f": "(x**2 - 4)/(x - 2)", "g": "x", "a": -1, "b": 5,
    },
    "3": {
        "descripcion": "f tiene una asíntota vertical con integral impropia CONVERGENTE.",
        "f": "1/sqrt(Abs(x))", "g": "0", "a": -1, "b": 1,
    },
    "4": {
        "descripcion": "f tiene una asíntota vertical con integral impropia DIVERGENTE.",
        "f": "1/x**2", "g": "0", "a": -1, "b": 2,
    },
}


def construir_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Calcula el área entre dos curvas, analizando "
                    "integrabilidad, intersecciones y aproximación numérica."
    )
    parser.add_argument(
        "--f", type=str, help="Expresión de f(x), p. ej. 'x**2 - 1'")
    parser.add_argument(
        "--g", type=str, help="Expresión de g(x), p. ej. 'sin(x)'")
    parser.add_argument("--a", type=float,
                        help="Extremo izquierdo del intervalo")
    parser.add_argument("--b", type=float,
                        help="Extremo derecho del intervalo")
    parser.add_argument(
        "--metodo", type=str, default="simpson_1_3",
        choices=["trapecio", "simpson_1_3",
                 "simpson_3_8", "gauss_legendre", "romberg"],
        help="Método numérico a usar para el área (por defecto: simpson_1_3)",
    )
    parser.add_argument(
        "--ejemplo", type=str, choices=list(EJEMPLOS.keys()),
        help="Ejecuta uno de los ejemplos predefinidos (1-4) en vez de pedir datos.",
    )
    parser.add_argument(
        "--salida", type=str, default="area_entre_curvas.png",
        help="Ruta del archivo PNG de salida para la gráfica.",
    )
    parser.add_argument(
        "--sin-graficar", action="store_true",
        help="No generar la gráfica (solo texto en consola).",
    )
    parser.add_argument(
        "--no-abrir", action="store_true",
        help="No abrir automáticamente las imágenes generadas en el navegador.",
    )
    return parser


def pedir_datos_interactivo():
    print("=" * 70)
    print(" ÁREA ENTRE CURVAS, INTEGRABILIDAD Y PROGRAMACIÓN ")
    print("=" * 70)
    print("Ingrese las funciones usando sintaxis tipo Python/SymPy.")
    print("Ejemplos válidos: x**2 - 3*x + 2 | sin(x) | 1/(x-2) | sqrt(4-x**2)\n")
    f_str = input("f(x) = ") or "x**2"
    g_str = input("g(x) = ") or "-x**2 + 4"
    a = float(input("a (extremo izquierdo) = ") or -2)
    b = float(input("b (extremo derecho)   = ") or 2)
    return f_str, g_str, a, b


def ejecutar_analisis(f_str: str, g_str: str, a: float, b: float,
                      metodo: str = "simpson_1_3",
                      ruta_salida: str = "area_entre_curvas.png",
                      graficar: bool = True,
                      abrir_navegador: bool = True):
    """Corre el flujo completo de la actividad y muestra los resultados
    en consola (y opcionalmente genera la gráfica)."""

    try:
        f = FuncionMatematica(f_str, nombre="f")
        g = FuncionMatematica(g_str, nombre="g")
    except ErrorFuncionInvalida as err:
        print(f"\n[ERROR] {err}")
        sys.exit(1)

    print("\n" + "#" * 70)
    print("# PARTE 1: DEFINICIÓN DEL PROBLEMA")
    print("#" * 70)
    print(f)
    print(g)
    print(f"Intervalo: [{a}, {b}]")

    print("\n" + "#" * 70)
    print("# PARTE 2: ANÁLISIS DE INTEGRABILIDAD (individual, de f y de g)")
    print("#" * 70)
    analizador_f = AnalizadorIntegrabilidad(f)
    analizador_g = AnalizadorIntegrabilidad(g)
    reporte_f = analizador_f.analizar(a, b)
    reporte_g = analizador_g.analizar(a, b)
    print(reporte_f.resumen())
    print()
    print(reporte_g.resumen())

    calculadora = CalculadoraAreaEntreCurvas(f, g)
    resultado = calculadora.calcular(a, b, metodo=metodo)

    print("\n" + "#" * 70)
    print("# PARTE 3 y 4: INTERSECCIONES, PARTICIÓN Y PLANTEAMIENTO DEL ÁREA")
    print("#" * 70)
    print("# PARTE 5: CÁLCULO DEL ÁREA (método numérico elegido: "
          f"{metodo})")
    print("#" * 70)
    print(resultado.resumen())

    print("\n" + "#" * 70)
    print("# COMPARACIÓN DE MÉTODOS NUMÉRICOS Y ANÁLISIS DE ERROR")
    print("#" * 70)
    print(f"Referencia de alta precisión (Gauss-Kronrod adaptativa) = "
          f"{resultado.referencia_alta_precision:.10f}")
    for m in resultado.metodos_comparados:
        print("   " + str(m))

    if graficar:
        print("\n" + "#" * 70)
        print("# PARTE 6: VISUALIZACIÓN")
        print("#" * 70)
        ruta = graficar_area_entre_curvas(resultado, ruta_salida=ruta_salida)
        print(f"Gráfica guardada en: {ruta}")
        if abrir_navegador:
            abrir_en_navegador(ruta)

        hay_asintotas = any(
            "infinita" in d.tipo.value
            for d in (reporte_f.discontinuidades + reporte_g.discontinuidades)
        )
        if not hay_asintotas:
            def h_abs(xs):
                fx = f.funcion_numerica()(xs)
                gx = g.funcion_numerica()(xs)
                import numpy as np
                return np.abs(fx - gx)

            integrador = IntegradorNumerico(h_abs)
            convergencia = integrador.analisis_convergencia(
                a, b, metodo="simpson")
            ruta_conv = graficar_convergencia(
                convergencia, ruta_salida=ruta_salida.replace(
                    ".png", "_convergencia.png"),
                metodo_nombre="Simpson 1/3",
            )
            print(f"Gráfica de convergencia guardada en: {ruta_conv}")
            if abrir_navegador:
                abrir_en_navegador(ruta_conv)
        else:
            print(
                "Nota: se omite la gráfica de convergencia de malla fija porque "
                "el intervalo contiene una asíntota vertical (no aplica a "
                "integrales impropias)."
            )

    return resultado


def main():
    parser = construir_parser()
    args = parser.parse_args()

    if args.ejemplo:
        ejemplo = EJEMPLOS[args.ejemplo]
        print(f"Ejecutando ejemplo {args.ejemplo}: {ejemplo['descripcion']}\n")
        f_str, g_str, a, b = ejemplo["f"], ejemplo["g"], ejemplo["a"], ejemplo["b"]
    elif args.f and args.g and args.a is not None and args.b is not None:
        f_str, g_str, a, b = args.f, args.g, args.a, args.b
    else:
        f_str, g_str, a, b = pedir_datos_interactivo()

    ejecutar_analisis(
        f_str, g_str, a, b,
        metodo=args.metodo, ruta_salida=args.salida,
        graficar=not args.sin_graficar,
        abrir_navegador=not args.no_abrir,
    )


if __name__ == "__main__":
    main()
