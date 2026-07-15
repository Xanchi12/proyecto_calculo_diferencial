# -*- coding: utf-8 -*-
"""
Módulo: area_entre_curvas.py
=====================================================================
Orquesta las Partes 4 y 5 de la actividad: dado que

    A = ∫[a,b] |f(x) - g(x)| dx

el valor absoluto obliga a "romper" el intervalo en los puntos donde
f(x) - g(x) cambia de signo (las intersecciones), pues en cada
subintervalo entre dos cortes consecutivos una sola función domina de
forma consistente sobre la otra, y allí:

    |f(x) - g(x)| = f(x) - g(x)   si f domina, o
    |f(x) - g(x)| = g(x) - f(x)   si g domina.

Este módulo:
    1. Usa `BuscadorIntersecciones` para partir [a, b] en subintervalos.
    2. Para cada subintervalo, determina qué función domina evaluando
       el punto medio.
    3. Usa `AnalizadorIntegrabilidad` sobre CADA subintervalo (de f, de
       g y de la diferencia) para confirmar que el área parcial puede
       calcularse.
    4. Integra numéricamente |f-g| en cada subintervalo (Simpson 1/3
       por defecto) y suma las áreas parciales.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

from funciones import FuncionMatematica
from integrabilidad import AnalizadorIntegrabilidad, ReporteIntegrabilidad
from intersecciones import BuscadorIntersecciones, PuntoInterseccion
from integracion_numerica import IntegradorNumerico, ResultadoIntegracion


@dataclass
class SubintervaloArea:
    x_izq: float
    x_der: float
    funcion_dominante: str  # nombre de f o g
    reporte_integrabilidad_f: ReporteIntegrabilidad
    reporte_integrabilidad_g: ReporteIntegrabilidad
    area_parcial: float
    metodo_usado: str

    def resumen(self) -> str:
        return (
            f"[{self.x_izq:.6g}, {self.x_der:.6g}]  "
            f"domina {self.funcion_dominante}  ->  "
            f"área parcial = {self.area_parcial:.8f}  "
            f"({self.metodo_usado})"
        )


@dataclass
class ResultadoAreaEntreCurvas:
    f: FuncionMatematica
    g: FuncionMatematica
    a: float
    b: float
    intersecciones: list
    subintervalos: list
    area_total: float
    referencia_alta_precision: float
    error_estimado: float
    metodos_comparados: list = field(default_factory=list)

    def resumen(self) -> str:
        lineas = [
            "=" * 70,
            f"ÁREA ENTRE {self.f.nombre}(x) Y {self.g.nombre}(x) EN "
            f"[{self.a}, {self.b}]",
            "=" * 70,
            f"{self.f.nombre}(x) = {self.f.expr}",
            f"{self.g.nombre}(x) = {self.g.expr}",
            "",
        ]
        if self.intersecciones:
            lineas.append(
                f"Intersecciones encontradas ({len(self.intersecciones)}):")
            for p in self.intersecciones:
                lineas.append(f"   - {p}")
        else:
            lineas.append(
                "No se encontraron intersecciones dentro del intervalo: "
                "una función domina en todo [a, b]."
            )
        lineas.append("")
        lineas.append("Partición del intervalo y área por subintervalo:")
        for s in self.subintervalos:
            lineas.append("   " + s.resumen())
        lineas.append("")
        lineas.append(
            f"ÁREA TOTAL (suma de áreas parciales) = {self.area_total:.8f}")
        lineas.append(
            f"Valor de referencia de alta precisión   = "
            f"{self.referencia_alta_precision:.8f}"
        )
        lineas.append(
            f"Error absoluto estimado                 = {self.error_estimado:.3e}")
        return "\n".join(lineas)


class CalculadoraAreaEntreCurvas:
    """Clase principal que integra todo el flujo pedido por la actividad."""

    def __init__(self, f: FuncionMatematica, g: FuncionMatematica):
        self.f = f
        self.g = g
        self.buscador = BuscadorIntersecciones(f, g)

    def _funcion_dominante_en(self, x_izq: float, x_der: float) -> str:
        """
        Determina qué función domina (es mayor) dentro de un
        subintervalo, evaluando en el punto medio. Si el punto medio
        cae justo sobre una discontinuidad removible (valor NaN), se
        prueba con puntos ligeramente desplazados hacia ambos lados
        hasta obtener un valor numérico válido.
        """
        centro = (x_izq + x_der) / 2
        ancho = x_der - x_izq
        offsets = [0.0, 1e-6 * ancho, -1e-6 *
                   ancho, 1e-3 * ancho, -1e-3 * ancho]
        for off in offsets:
            x_prueba = centro + off
            if not (x_izq < x_prueba < x_der) and off != 0.0:
                continue
            yf = self.f.evaluar(x_prueba)
            yg = self.g.evaluar(x_prueba)
            if not (np.isnan(yf) or np.isnan(yg)):
                return self.f.nombre if yf >= yg else self.g.nombre
        return "indeterminado"

    def calcular(self, a: float, b: float,
                 metodo: str = "simpson_1_3", n_por_subintervalo: int = 2000
                 ) -> ResultadoAreaEntreCurvas:
        """
        Calcula el área entre f y g en [a, b], siguiendo el flujo
        completo: intersecciones -> partición -> integrabilidad ->
        integración numérica por subintervalo -> suma total.

        Parameters
        ----------
        metodo : str
            Uno de: 'trapecio', 'simpson_1_3', 'simpson_3_8',
            'gauss_legendre', 'romberg'.
        """
        intersecciones = self.buscador.encontrar(a, b)
        cortes_interseccion = self.buscador.particion_por_intersecciones(a, b)

        analizador_f = AnalizadorIntegrabilidad(self.f)
        analizador_g = AnalizadorIntegrabilidad(self.g)

        # El intervalo también debe partirse en cualquier punto donde
        # f o g tengan una discontinuidad (removible, salto o
        # asíntota). Esto es indispensable para las asíntotas
        # verticales: si no se excluye el punto exacto de la malla de
        # integración, los métodos numéricos de paso fijo (Trapecio,
        # Simpson, etc.) evaluarían justo sobre la singularidad.
        reporte_global_f = analizador_f.analizar(a, b)
        reporte_global_g = analizador_g.analizar(a, b)
        puntos_discontinuidad = sorted(set(
            round(d.punto, 10) for d in
            (reporte_global_f.discontinuidades + reporte_global_g.discontinuidades)
        ))

        cortes = sorted(set(cortes_interseccion) | set(puntos_discontinuidad))
        cortes = [c for c in cortes if a - 1e-9 <= c <= b + 1e-9]

        def h_abs(xs):
            fx = self.f.funcion_numerica()(xs)
            gx = self.g.funcion_numerica()(xs)
            return np.abs(fx - gx)

        def h_abs_escalar(t):
            return float(h_abs(np.array([t]))[0])

        integrador_global = IntegradorNumerico(h_abs)

        subintervalos = []
        area_total = 0.0
        margen_singularidad = 1e-8

        for i in range(len(cortes) - 1):
            x_izq, x_der = cortes[i], cortes[i + 1]
            if x_der - x_izq < 1e-10:
                continue

            rep_f = analizador_f.analizar(x_izq, x_der)
            rep_g = analizador_g.analizar(x_izq, x_der)

            dominante = self._funcion_dominante_en(x_izq, x_der)

            # ¿Alguno de los extremos de este subintervalo es una
            # asíntota vertical? En ese caso el subintervalo representa
            # una integral impropia y debe tratarse de forma especial.
            tocando_asintota = any(
                d.tipo.value == "infinita (asíntota vertical)" and
                (abs(d.punto - x_izq) < 1e-6 or abs(d.punto - x_der) < 1e-6)
                for d in (rep_f.discontinuidades + rep_g.discontinuidades)
            )

            # ¿Este subintervalo completo cae fuera del dominio real de
            # f o de g (p. ej. sqrt(4-x**2) para x fuera de [-2, 2], o
            # log(x) para x <= 0)? Se detecta porque AMBOS extremos del
            # subintervalo quedan marcados como NO_DEFINIDA_EN_DOMINIO
            # en el reporte de integrabilidad de ese mismo subintervalo
            # (si solo uno de los extremos lo estuviera, el otro lado sí
            # tendría valores reales y no sería este caso). No se
            # integra numéricamente: dejar que Trapecio/Simpson/etc.
            # evalúen ahí y confiar en que `nan_to_num(..., nan=0.0)`
            # "arregle" el resultado sería implícito y frágil, y además
            # rompe la referencia de alta precisión (scipy.integrate.quad
            # no tolera evaluar la función en la zona compleja).
            fuera_de_dominio = (not tocando_asintota) and any(
                d.tipo.value == "fuera del dominio real" and
                (abs(d.punto - x_izq) < 1e-6 or abs(d.punto - x_der) < 1e-6)
                for d in (rep_f.discontinuidades + rep_g.discontinuidades)
            )

            if fuera_de_dominio:
                subintervalos.append(SubintervaloArea(
                    x_izq=x_izq, x_der=x_der, funcion_dominante=dominante,
                    reporte_integrabilidad_f=rep_f, reporte_integrabilidad_g=rep_g,
                    area_parcial=0.0,
                    metodo_usado="Subintervalo fuera del dominio real de f o g: "
                                 "no se integra (no hay valores reales que sumar "
                                 "al área en este tramo)",
                ))
                continue

            if tocando_asintota:
                convergente = rep_f.es_integrable and rep_g.es_integrable
                if not convergente:
                    subintervalos.append(SubintervaloArea(
                        x_izq=x_izq, x_der=x_der, funcion_dominante=dominante,
                        reporte_integrabilidad_f=rep_f, reporte_integrabilidad_g=rep_g,
                        area_parcial=float("inf"),
                        metodo_usado="DIVERGENTE: la integral impropia no converge "
                                     "(área infinita en este subintervalo)",
                    ))
                    area_total = float("inf")
                    continue
                else:
                    # Integral impropia convergente: se integra con
                    # cuadratura adaptativa (Gauss-Kronrod) evitando el
                    # extremo exacto de la singularidad mediante un
                    # margen infinitesimal, en vez de una malla fija que
                    # produciría NaN/Inf justo en ese punto.
                    izq_eval = x_izq + margen_singularidad * \
                        max(1.0, abs(x_der - x_izq))
                    der_eval = x_der - margen_singularidad * \
                        max(1.0, abs(x_der - x_izq))
                    izq_eval = min(izq_eval, der_eval)
                    from scipy import integrate as sci_integrate
                    valor, _ = sci_integrate.quad(
                        h_abs_escalar, izq_eval, der_eval, limit=400
                    )
                    subintervalos.append(SubintervaloArea(
                        x_izq=x_izq, x_der=x_der, funcion_dominante=dominante,
                        reporte_integrabilidad_f=rep_f, reporte_integrabilidad_g=rep_g,
                        area_parcial=valor,
                        metodo_usado="Cuadratura adaptativa (integral impropia "
                                     "convergente, evaluada como límite)",
                    ))
                    if area_total != float("inf"):
                        area_total += valor
                    continue

            integrador_local = IntegradorNumerico(h_abs)
            metodo_funcs = {
                "trapecio": integrador_local.trapecio,
                "simpson_1_3": integrador_local.simpson_1_3,
                "simpson_3_8": integrador_local.simpson_3_8,
                "gauss_legendre": integrador_local.gauss_legendre,
                "romberg": lambda a_, b_: integrador_local.romberg(a_, b_, niveles=6),
            }
            if metodo not in metodo_funcs:
                raise ValueError(f"Método desconocido: {metodo}")

            if metodo in ("trapecio", "simpson_1_3", "simpson_3_8"):
                resultado_parcial = metodo_funcs[metodo](
                    x_izq, x_der, n_por_subintervalo)
            else:
                resultado_parcial = metodo_funcs[metodo](x_izq, x_der)

            area_parcial = resultado_parcial.valor
            if area_total != float("inf"):
                area_total += area_parcial

            subintervalos.append(SubintervaloArea(
                x_izq=x_izq, x_der=x_der, funcion_dominante=dominante,
                reporte_integrabilidad_f=rep_f, reporte_integrabilidad_g=rep_g,
                area_parcial=area_parcial, metodo_usado=resultado_parcial.metodo,
            ))

        hay_asintotas = len(puntos_discontinuidad) > 0 and any(
            "infinita" in d.tipo.value
            for d in (reporte_global_f.discontinuidades + reporte_global_g.discontinuidades)
        )
        # Igual que con las asíntotas: si f o g no toman valores reales
        # en parte de [a, b], una cuadratura global (adaptativa como
        # Gauss-Kronrod, o de malla fija) sobre TODO [a, b] evaluaría la
        # función en la zona compleja y devolvería NaN, invalidando la
        # referencia y la comparación de métodos. El área total ya se
        # calculó correctamente subintervalo a subintervalo arriba.
        hay_fuera_de_dominio = len(puntos_discontinuidad) > 0 and any(
            d.tipo.value == "fuera del dominio real"
            for d in (reporte_global_f.discontinuidades + reporte_global_g.discontinuidades)
        )

        if area_total == float("inf") or hay_asintotas or hay_fuera_de_dominio:
            # La comparación "ingenua" de métodos de malla fija
            # (Trapecio, Simpson, etc.) sobre TODO [a, b] no es
            # confiable cuando existe una asíntota vertical dentro del
            # intervalo (evaluarían justo sobre la singularidad). El
            # área total ya fue calculada correctamente subintervalo a
            # subintervalo arriba, así que aquí solo se reporta el
            # área total como referencia (sin comparación de métodos
            # adicionales, que no aplican a integrales impropias).
            referencia_valor = area_total
            error_estimado = 0.0
            metodos_comparados = []
        else:
            puntos_criticos = sorted(set(
                [p.x for p in intersecciones] + list(puntos_discontinuidad)
            ))
            referencia = integrador_global.referencia_alta_precision(
                a, b, puntos_criticos=puntos_criticos
            )
            referencia_valor = referencia.valor
            error_estimado = abs(area_total - referencia_valor)

            _, metodos_comparados = integrador_global.comparar_metodos(
                a, b, puntos_criticos=puntos_criticos
            )

        return ResultadoAreaEntreCurvas(
            f=self.f, g=self.g, a=a, b=b,
            intersecciones=intersecciones,
            subintervalos=subintervalos,
            area_total=area_total,
            referencia_alta_precision=referencia_valor,
            error_estimado=error_estimado,
            metodos_comparados=metodos_comparados,
        )
