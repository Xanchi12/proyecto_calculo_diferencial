# -*- coding: utf-8 -*-
"""
Módulo: test_proyecto.py
=====================================================================
Pruebas unitarias básicas del proyecto, usando el módulo estándar
`unittest`. Verifican los casos más importantes de cada módulo:

    - Parsing y evaluación de funciones.
    - Clasificación correcta de discontinuidades conocidas.
    - Detección de intersecciones en un caso con solución analítica
      conocida.
    - Precisión de los métodos de integración numérica sobre una
      función cuya integral exacta se conoce (x**2 en [0, 3] = 9).
    - Cálculo de área entre curvas en un caso con solución exacta
      conocida (dos parábolas).

Ejecutar con:
    python -m unittest test_proyecto.py -v
"""

import unittest
import numpy as np

from funciones import FuncionMatematica, ErrorFuncionInvalida
from integrabilidad import AnalizadorIntegrabilidad, TipoDiscontinuidad
from intersecciones import BuscadorIntersecciones
from integracion_numerica import IntegradorNumerico
from area_entre_curvas import CalculadoraAreaEntreCurvas


class TestFuncionMatematica(unittest.TestCase):
    def test_evaluacion_basica(self):
        f = FuncionMatematica("x**2 + 1")
        self.assertAlmostEqual(f.evaluar(2), 5.0)

    def test_expresion_invalida(self):
        with self.assertRaises(ErrorFuncionInvalida):
            FuncionMatematica("")

    def test_variable_no_permitida(self):
        with self.assertRaises(ErrorFuncionInvalida):
            FuncionMatematica("x + y")

    def test_funcion_numerica_vectorizada(self):
        f = FuncionMatematica("sin(x)")
        xs = np.array([0, np.pi / 2, np.pi])
        ys = f.funcion_numerica()(xs)
        np.testing.assert_allclose(ys, [0, 1, 0], atol=1e-8)


class TestIntegrabilidad(unittest.TestCase):
    def test_funcion_continua_sin_discontinuidades(self):
        f = FuncionMatematica("x**2 - 3*x + 2")
        analizador = AnalizadorIntegrabilidad(f)
        reporte = analizador.analizar(-5, 5)
        self.assertTrue(reporte.es_integrable)
        self.assertEqual(len(reporte.discontinuidades), 0)

    def test_discontinuidad_removible(self):
        # (x^2 - 4)/(x - 2) = x + 2 para todo x != 2 (removible en x=2)
        f = FuncionMatematica("(x**2 - 4)/(x - 2)")
        analizador = AnalizadorIntegrabilidad(f)
        reporte = analizador.analizar(0, 5)
        tipos = [d.tipo for d in reporte.discontinuidades]
        self.assertIn(TipoDiscontinuidad.REMOVIBLE, tipos)
        self.assertTrue(reporte.es_integrable)

    def test_asintota_vertical(self):
        f = FuncionMatematica("1/x")
        analizador = AnalizadorIntegrabilidad(f)
        reporte = analizador.analizar(-1, 1)
        tipos = [d.tipo for d in reporte.discontinuidades]
        self.assertIn(TipoDiscontinuidad.INFINITA, tipos)
        # 1/x en [-1,1] es una integral impropia DIVERGENTE (simétrica,
        # las partes no se cancelan en el sentido de Riemann impropio).
        self.assertFalse(reporte.es_integrable)

    def test_asintota_convergente(self):
        # 1/sqrt(x) en (0, 1]: integral impropia CONVERGENTE (= 2).
        f = FuncionMatematica("1/sqrt(x)")
        analizador = AnalizadorIntegrabilidad(f)
        reporte = analizador.analizar(0, 1)
        self.assertTrue(reporte.es_integrable)
        self.assertTrue(reporte.requiere_integral_impropia)


class TestIntersecciones(unittest.TestCase):
    def test_dos_parabolas(self):
        # x^2 = -x^2 + 4  ->  2x^2 = 4  ->  x = ±sqrt(2)
        f = FuncionMatematica("x**2", nombre="f")
        g = FuncionMatematica("-x**2 + 4", nombre="g")
        buscador = BuscadorIntersecciones(f, g)
        puntos = buscador.encontrar(-3, 3)
        xs = sorted(p.x for p in puntos)
        self.assertEqual(len(xs), 2)
        self.assertAlmostEqual(xs[0], -np.sqrt(2), places=4)
        self.assertAlmostEqual(xs[1], np.sqrt(2), places=4)

    def test_sin_interseccion(self):
        f = FuncionMatematica("x**2 + 10", nombre="f")
        g = FuncionMatematica("x", nombre="g")
        buscador = BuscadorIntersecciones(f, g)
        puntos = buscador.encontrar(-5, 5)
        self.assertEqual(len(puntos), 0)


class TestIntegracionNumerica(unittest.TestCase):
    def setUp(self):
        f = FuncionMatematica("x**2")
        self.integrador = IntegradorNumerico(f.funcion_numerica())

    def test_trapecio_converge_al_valor_exacto(self):
        # integral de x^2 en [0,3] = 9
        resultado = self.integrador.trapecio(0, 3, n=5000)
        self.assertAlmostEqual(resultado.valor, 9.0, places=2)

    def test_simpson_es_mas_preciso_que_trapecio_con_mismo_n(self):
        n = 20
        r_trap = self.integrador.trapecio(0, 3, n=n)
        r_simp = self.integrador.simpson_1_3(0, 3, n=n)
        error_trap = abs(r_trap.valor - 9.0)
        error_simp = abs(r_simp.valor - 9.0)
        self.assertLess(error_simp, error_trap)

    def test_gauss_legendre_exacto_para_polinomio_bajo_grado(self):
        # x^2 tiene grado 2; Gauss-Legendre con n=2 nodos es exacto
        # para polinomios de grado <= 2n-1 = 3.
        resultado = self.integrador.gauss_legendre(0, 3, n=2)
        self.assertAlmostEqual(resultado.valor, 9.0, places=8)


class TestAreaEntreCurvas(unittest.TestCase):
    def test_area_dos_parabolas_valor_conocido(self):
        # Área ENTRE LAS INTERSECCIONES de x^2 y -x^2+4:
        # se cortan en x = ±sqrt(2). Ahí g domina:
        # A = ∫_{-√2}^{√2} [(4 - x^2) - x^2] dx = ∫(4 - 2x^2)dx
        #   = [4x - 2x^3/3] evaluado en ±√2 = 16*sqrt(2)/3 ≈ 7.5425
        f = FuncionMatematica("x**2", nombre="f")
        g = FuncionMatematica("-x**2 + 4", nombre="g")
        calculadora = CalculadoraAreaEntreCurvas(f, g)
        raiz2 = np.sqrt(2)
        resultado = calculadora.calcular(-raiz2, raiz2)
        valor_esperado = (16 * raiz2) / 3
        self.assertAlmostEqual(resultado.area_total, valor_esperado, places=2)

    def test_area_incluye_lobulos_fuera_de_las_intersecciones(self):
        # En [-2, 2] el área total debe ser la suma de los tres
        # subintervalos (fuera y dentro de las intersecciones), y por
        # lo tanto debe ser MAYOR que el área del lóbulo central solo.
        f = FuncionMatematica("x**2", nombre="f")
        g = FuncionMatematica("-x**2 + 4", nombre="g")
        calculadora = CalculadoraAreaEntreCurvas(f, g)
        resultado = calculadora.calcular(-2, 2)
        lobulo_central = (16 * np.sqrt(2)) / 3
        self.assertGreater(resultado.area_total, lobulo_central)
        self.assertEqual(len(resultado.subintervalos), 3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
