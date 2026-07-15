# -*- coding: utf-8 -*-
"""
Módulo: app.py
=====================================================================
Interfaz web local del proyecto "Área entre Curvas, Integrabilidad y
Programación". Sirve una única página (index.html) con un formulario
para ingresar f(x), g(x), a, b y el método numérico, y expone cada
parte del análisis como un endpoint independiente
(/api/integrabilidad, /api/intersecciones, /api/area,
/api/convergencia). El frontend (ver templates/index.html) los llama
en secuencia, así que cada sección de la página se va llenando a
medida que el cálculo correspondiente termina — en vez de esperar a
que TODO el análisis esté listo para mostrar cualquier cosa.

Uso
---
    python app.py

Luego abre http://127.0.0.1:5000 en el navegador.
"""

from __future__ import annotations

import numpy as np
from flask import Flask, jsonify, render_template, request

from funciones import FuncionMatematica, ErrorFuncionInvalida
from integrabilidad import AnalizadorIntegrabilidad
from intersecciones import BuscadorIntersecciones
from area_entre_curvas import CalculadoraAreaEntreCurvas
from integracion_numerica import IntegradorNumerico
from visualizacion import area_entre_curvas_a_base64, convergencia_a_base64

app = Flask(__name__)


def _leer_funciones(datos: dict):
    """Construye f y g a partir del JSON recibido, o lanza un error
    controlado (400) si la expresión no es válida."""
    f_str = datos.get("f", "").strip()
    g_str = datos.get("g", "").strip()
    a = float(datos.get("a"))
    b = float(datos.get("b"))
    if a >= b:
        raise ValueError(
            "El extremo izquierdo (a) debe ser menor que el derecho (b).")
    f = FuncionMatematica(f_str, nombre="f")
    g = FuncionMatematica(g_str, nombre="g")
    return f, g, a, b


@app.errorhandler(ErrorFuncionInvalida)
@app.errorhandler(ValueError)
def _manejar_error_datos(err):
    return jsonify({"error": str(err)}), 400


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/integrabilidad", methods=["POST"])
def api_integrabilidad():
    """Parte 2: análisis de integrabilidad de f y de g por separado."""
    f, g, a, b = _leer_funciones(request.get_json())

    reporte_f = AnalizadorIntegrabilidad(f).analizar(a, b)
    reporte_g = AnalizadorIntegrabilidad(g).analizar(a, b)

    def _serializar(reporte):
        return {
            "definida_en_todo_el_intervalo": reporte.definida_en_todo_el_intervalo,
            "es_integrable": reporte.es_integrable,
            "requiere_integral_impropia": reporte.requiere_integral_impropia,
            "justificacion": reporte.justificacion,
            "discontinuidades": [
                {"punto": d.punto, "tipo": d.tipo.value, "detalle": d.detalle}
                for d in reporte.discontinuidades
            ],
        }

    return jsonify({
        "f_latex": f.latex(), "g_latex": g.latex(),
        "reporte_f": _serializar(reporte_f),
        "reporte_g": _serializar(reporte_g),
    })


@app.route("/api/intersecciones", methods=["POST"])
def api_intersecciones():
    """Parte 3: puntos donde f(x) = g(x) dentro de [a, b]."""
    f, g, a, b = _leer_funciones(request.get_json())
    buscador = BuscadorIntersecciones(f, g)
    puntos = buscador.encontrar(a, b)
    return jsonify({
        "intersecciones": [
            {"x": p.x, "y": p.y, "metodo": p.metodo} for p in puntos
        ]
    })


@app.route("/api/area", methods=["POST"])
def api_area():
    """Partes 4, 5 y 6: partición, cálculo numérico del área y
    gráfica, en un solo paso (es el cómputo más pesado)."""
    datos = request.get_json()
    f, g, a, b = _leer_funciones(datos)
    metodo = datos.get("metodo", "simpson_1_3")

    calculadora = CalculadoraAreaEntreCurvas(f, g)
    resultado = calculadora.calcular(a, b, metodo=metodo)

    subintervalos = [
        {
            "x_izq": s.x_izq, "x_der": s.x_der,
            "dominante": s.funcion_dominante,
            "area_parcial": s.area_parcial,
            "metodo": s.metodo_usado,
        }
        for s in resultado.subintervalos
    ]

    metodos_comparados = [
        {
            "metodo": m.metodo, "valor": m.valor,
            "n": m.n_evaluaciones, "error": m.error_estimado,
        }
        for m in resultado.metodos_comparados
    ]

    imagen_base64 = area_entre_curvas_a_base64(resultado)

    hay_asintotas = any(
        "infinita" in d.tipo.value
        for s in resultado.subintervalos
        for d in (s.reporte_integrabilidad_f.discontinuidades +
                  s.reporte_integrabilidad_g.discontinuidades)
    )

    imagen_convergencia = None
    if not hay_asintotas and np.isfinite(resultado.area_total):
        def h_abs(xs):
            fx = f.funcion_numerica()(xs)
            gx = g.funcion_numerica()(xs)
            return np.abs(fx - gx)
        integrador = IntegradorNumerico(h_abs)
        convergencia = integrador.analisis_convergencia(a, b, metodo="simpson")
        imagen_convergencia = convergencia_a_base64(
            convergencia, metodo_nombre="Simpson 1/3")

    return jsonify({
        "area_total": resultado.area_total if np.isfinite(resultado.area_total) else None,
        "es_infinita": not np.isfinite(resultado.area_total),
        "referencia": resultado.referencia_alta_precision if np.isfinite(resultado.referencia_alta_precision) else None,
        "error_estimado": resultado.error_estimado,
        "subintervalos": subintervalos,
        "metodos_comparados": metodos_comparados,
        "imagen": imagen_base64,
        "imagen_convergencia": imagen_convergencia,
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)
