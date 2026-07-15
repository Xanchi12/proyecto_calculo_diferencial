# -*- coding: utf-8 -*-
"""
Módulo: integrabilidad.py
=====================================================================
Implementa el análisis pedido en la "Parte 2: Análisis de
integrabilidad" de la actividad:

    1. ¿La función está definida en todo el intervalo [a, b]?
    2. ¿Existen discontinuidades?
    3. ¿Son removibles o infinitas (de salto/asíntota)?
    4. ¿Es posible calcular la integral (propia o impropia)?

Fundamento teórico
-------------------
Una función f es Riemann-integrable en [a, b] si es acotada en [a, b]
y su conjunto de discontinuidades tiene medida de Lebesgue cero
(criterio de Lebesgue). En la práctica, para las funciones elementales
que se manejan en un curso de Cálculo Integral, basta con verificar:

    - Continuidad en todo punto salvo, a lo sumo, un número finito de
      ellos (funciones continuas a trozos), lo que garantiza medida
      cero automáticamente.
    - Que las discontinuidades sean evitables o de salto finito
      (integral propia), o bien
    - Que, siendo de tipo asíntota vertical, la integral impropia
      asociada converja (en cuyo caso el área sigue siendo calculable
      como límite, aunque la función no sea Riemann-integrable en el
      sentido clásico).

Este módulo intenta detectar automáticamente los puntos "sospechosos"
(usando singularidades simbólicas de SymPy y un barrido numérico) y
clasificarlos.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import sympy as sp

from funciones import FuncionMatematica


class TipoDiscontinuidad(Enum):
    REMOVIBLE = "removible"
    SALTO_FINITO = "salto finito"
    INFINITA = "infinita (asíntota vertical)"
    NO_DEFINIDA_EN_DOMINIO = "fuera del dominio real"
    INDETERMINABLE = "no se pudo clasificar automáticamente"


@dataclass
class Discontinuidad:
    """Representa una discontinuidad detectada en un punto x0."""
    punto: float
    tipo: TipoDiscontinuidad
    limite_izq: object = None
    limite_der: object = None
    detalle: str = ""

    def __str__(self):
        return (
            f"x = {self.punto:.6g}  ->  {self.tipo.value}"
            + (f"  ({self.detalle})" if self.detalle else "")
        )


@dataclass
class ReporteIntegrabilidad:
    """Resultado completo del análisis de integrabilidad de una función
    en un intervalo [a, b]."""
    nombre_funcion: str
    a: float
    b: float
    definida_en_todo_el_intervalo: bool
    discontinuidades: list = field(default_factory=list)
    es_integrable_propia: bool = True
    requiere_integral_impropia: bool = False
    es_integrable: bool = True
    justificacion: str = ""

    def resumen(self) -> str:
        lineas = [
            f"--- Análisis de integrabilidad de {self.nombre_funcion} "
            f"en [{self.a}, {self.b}] ---",
            f"¿Definida en todo el intervalo?: "
            f"{'Sí' if self.definida_en_todo_el_intervalo else 'No'}",
        ]
        if self.discontinuidades:
            lineas.append(
                f"Discontinuidades encontradas ({len(self.discontinuidades)}):"
            )
            for d in self.discontinuidades:
                lineas.append(f"   - {d}")
        else:
            lineas.append("No se detectaron discontinuidades en el intervalo.")

        lineas.append(
            f"¿Integral propia?: {'Sí' if self.es_integrable_propia else 'No'}"
        )
        lineas.append(
            f"¿Requiere tratarse como integral impropia?: "
            f"{'Sí' if self.requiere_integral_impropia else 'No'}"
        )
        lineas.append(
            f"Conclusión: {'ES integrable' if self.es_integrable else 'NO es integrable'} "
            f"en el sentido explicado abajo."
        )
        lineas.append(f"Justificación: {self.justificacion}")
        return "\n".join(lineas)


class AnalizadorIntegrabilidad:
    """
    Analiza si una `FuncionMatematica` es integrable en un intervalo
    cerrado [a, b], clasificando cualquier discontinuidad encontrada.

    El análisis combina:
        (a) Búsqueda simbólica de singularidades vía SymPy
            (sympy.calculus.singularities más análisis del dominio).
        (b) Un barrido numérico fino del intervalo para detectar
            valores NaN/Inf que el análisis simbólico pudiera no
            capturar (por ejemplo, funciones definidas por partes o
            expresiones muy complejas).
    """

    def __init__(self, funcion: FuncionMatematica, tolerancia: float = 1e-9,
                 puntos_barrido: int = 4000):
        self.funcion = funcion
        self.tolerancia = tolerancia
        self.puntos_barrido = puntos_barrido

    # ------------------------------------------------------------------
    def _candidatos_singularidad_simbolicos(self, a: float, b: float):
        """Obtiene candidatos a puntos de discontinuidad resolviendo,
        simbólicamente, dónde el denominador se anula o el dominio
        excluye puntos (raíces pares, logaritmos, etc.)."""
        x = self.funcion.x
        candidatos = set()
        try:
            singularidades = sp.singularities(self.funcion.expr, x)
            for s in singularidades:
                if s.is_real:
                    val = float(s)
                    if a - 1e-9 <= val <= b + 1e-9:
                        candidatos.add(val)
        except (NotImplementedError, TypeError):
            pass

        # Puntos donde el dominio simbólico tiene "huecos" dentro de [a, b]
        try:
            dominio = self.funcion.dominio_real()
            complemento = sp.Complement(sp.Interval(a, b), dominio)
            if complemento is not sp.EmptySet:
                puntos = _extraer_puntos_frontera(complemento)
                candidatos.update(p for p in puntos if a - 1e-9 <= p <= b + 1e-9)
        except (TypeError, NotImplementedError):
            pass

        return sorted(candidatos)

    def _candidatos_numericos(self, a: float, b: float):
        """Barrido numérico para detectar NaN/Inf que el análisis
        simbólico no haya podido resolver (red de seguridad)."""
        f = self.funcion.funcion_numerica()
        xs = np.linspace(a, b, self.puntos_barrido)
        ys = f(xs)
        malos = xs[~np.isfinite(ys)]
        candidatos = []
        # Agrupar puntos consecutivos "malos" en un único candidato
        # (representan la misma discontinuidad muestreada varias veces).
        if malos.size:
            grupo = [malos[0]]
            paso = (b - a) / self.puntos_barrido
            for p in malos[1:]:
                if p - grupo[-1] <= 3 * paso:
                    grupo.append(p)
                else:
                    candidatos.append(float(np.mean(grupo)))
                    grupo = [p]
            candidatos.append(float(np.mean(grupo)))
        return candidatos

    def _clasificar_punto(self, punto: float, a: float, b: float) -> Discontinuidad:
        """Clasifica un punto candidato como removible, salto finito,
        infinita o fuera de dominio, usando límites simbólicos cuando
        es posible."""
        x = self.funcion.x
        expr = self.funcion.expr

        # Si el punto es un extremo del intervalo, solo se puede evaluar
        # el límite lateral "hacia adentro".
        direcciones = []
        if punto > a + 1e-12:
            direcciones.append("-")
        if punto < b - 1e-12:
            direcciones.append("+")
        if not direcciones:
            direcciones = ["-", "+"]

        limites = {}
        for d in direcciones:
            try:
                lim = sp.limit(expr, x, punto, dir=d)
                limites[d] = lim
            except (NotImplementedError, TypeError):
                limites[d] = None

        valores_finitos = [
            l for l in limites.values()
            if l is not None and l.is_real and l.is_finite
        ]
        hay_infinito = any(
            l is not None and (l is sp.oo or l is -sp.oo)
            for l in limites.values()
        )

        limite_izq = limites.get("-")
        limite_der = limites.get("+")

        if hay_infinito:
            return Discontinuidad(
                punto, TipoDiscontinuidad.INFINITA, limite_izq, limite_der,
                detalle="al menos un límite lateral diverge a ±infinito "
                        "(asíntota vertical)"
            )

        if len(valores_finitos) == 2 and abs(valores_finitos[0] - valores_finitos[1]) < self.tolerancia:
            # Límites laterales finitos e iguales: la función solo
            # "falla" por no estar definida (o estar mal definida) en
            # el punto exacto -> discontinuidad removible.
            return Discontinuidad(
                punto, TipoDiscontinuidad.REMOVIBLE, limite_izq, limite_der,
                detalle=f"ambos límites laterales existen y valen "
                        f"{valores_finitos[0]} pero f no coincide (o no "
                        f"está definida) en x0"
            )

        if len(valores_finitos) == 2:
            return Discontinuidad(
                punto, TipoDiscontinuidad.SALTO_FINITO, limite_izq, limite_der,
                detalle=f"límite izq = {limite_izq}, límite der = {limite_der}"
            )

        if len(valores_finitos) == 1:
            return Discontinuidad(
                punto, TipoDiscontinuidad.SALTO_FINITO, limite_izq, limite_der,
                detalle="solo un límite lateral existe (punto extremo del "
                        "intervalo o dominio restringido)"
            )

        return Discontinuidad(
            punto, TipoDiscontinuidad.INDETERMINABLE, limite_izq, limite_der,
            detalle="SymPy no pudo evaluar los límites laterales de forma "
                    "concluyente"
        )

    # ------------------------------------------------------------------
    def analizar(self, a: float, b: float) -> ReporteIntegrabilidad:
        """
        Ejecuta el análisis completo de integrabilidad en [a, b].

        Returns
        -------
        ReporteIntegrabilidad
        """
        if a >= b:
            raise ValueError("Se requiere a < b para definir el intervalo [a, b].")

        candidatos = set(self._candidatos_singularidad_simbolicos(a, b))
        candidatos.update(self._candidatos_numericos(a, b))
        # Descartar candidatos que estén prácticamente en un mismo punto.
        candidatos = _deduplicar(sorted(candidatos), tol=1e-6)

        discontinuidades = [
            self._clasificar_punto(p, a, b) for p in candidatos
        ]

        definida_en_todo = len(discontinuidades) == 0

        hay_infinitas = any(
            d.tipo == TipoDiscontinuidad.INFINITA for d in discontinuidades
        )
        hay_indeterminables = any(
            d.tipo == TipoDiscontinuidad.INDETERMINABLE for d in discontinuidades
        )

        es_integrable_propia = not hay_infinitas and not hay_indeterminables
        requiere_impropia = hay_infinitas

        # Si hay asíntotas verticales, se intenta verificar convergencia
        # de la integral impropia evaluando el límite simbólico de la
        # antiderivada cerca del punto conflictivo.
        es_integrable = es_integrable_propia
        justificacion = ""

        if not discontinuidades:
            justificacion = (
                "La función es continua en todo [a, b] (no se detectaron "
                "discontinuidades ni puntos fuera de dominio); toda función "
                "continua en un intervalo cerrado y acotado es Riemann-"
                "integrable en él."
            )
            es_integrable = True
        elif es_integrable_propia:
            justificacion = (
                "Las discontinuidades detectadas son removibles o de salto "
                "finito. Una función acotada con un número finito de "
                "discontinuidades de este tipo tiene un conjunto de "
                "discontinuidades de medida cero, por lo que sigue siendo "
                "Riemann-integrable (criterio de Lebesgue)."
            )
            es_integrable = True
        elif hay_infinitas:
            convergentes = []
            for d in discontinuidades:
                if d.tipo == TipoDiscontinuidad.INFINITA:
                    convergente = _verificar_convergencia_impropia(
                        self.funcion, a, b, d.punto
                    )
                    convergentes.append(convergente)
            es_integrable = all(convergentes) if convergentes else False
            if es_integrable:
                justificacion = (
                    "Existen asíntotas verticales dentro de [a, b], por lo "
                    "que la integral NO es propia; sin embargo, evaluada "
                    "como límite (integral impropia de segunda especie) el "
                    "área converge a un valor finito, por lo que el área "
                    "sí puede calcularse."
                )
            else:
                justificacion = (
                    "Existen asíntotas verticales dentro de [a, b] y la "
                    "integral impropia correspondiente DIVERGE, por lo que "
                    "el área entre las curvas no está definida (es infinita) "
                    "en ese subintervalo."
                )
        else:
            justificacion = (
                "No fue posible clasificar automáticamente todas las "
                "discontinuidades; se recomienda revisión manual."
            )
            es_integrable = False

        return ReporteIntegrabilidad(
            nombre_funcion=self.funcion.nombre,
            a=a, b=b,
            definida_en_todo_el_intervalo=definida_en_todo,
            discontinuidades=discontinuidades,
            es_integrable_propia=es_integrable_propia,
            requiere_integral_impropia=requiere_impropia,
            es_integrable=es_integrable,
            justificacion=justificacion,
        )


# ----------------------------------------------------------------------
# Funciones auxiliares de módulo
# ----------------------------------------------------------------------
def _extraer_puntos_frontera(conjunto):
    """Extrae los extremos numéricos de un conjunto SymPy (Interval,
    Union, FiniteSet) como una lista de floats."""
    puntos = []
    if isinstance(conjunto, sp.Interval):
        if conjunto.start.is_finite:
            puntos.append(float(conjunto.start))
        if conjunto.end.is_finite:
            puntos.append(float(conjunto.end))
    elif isinstance(conjunto, sp.Union):
        for arg in conjunto.args:
            puntos.extend(_extraer_puntos_frontera(arg))
    elif isinstance(conjunto, sp.FiniteSet):
        for elemento in conjunto.args:
            try:
                puntos.append(float(elemento))
            except (TypeError, ValueError):
                pass
    return puntos


def _deduplicar(valores, tol=1e-6):
    """Elimina valores numéricos casi-duplicados de una lista ordenada."""
    if not valores:
        return []
    resultado = [valores[0]]
    for v in valores[1:]:
        if abs(v - resultado[-1]) > tol:
            resultado.append(v)
    return resultado


def _verificar_convergencia_impropia(funcion: FuncionMatematica, a: float,
                                      b: float, punto_singular: float,
                                      epsilon: float = 1e-6) -> bool:
    """
    Verifica si la integral impropia de `funcion` en [a, b], con una
    asíntota vertical en `punto_singular`, converge, evaluando el
    límite simbólico de la antiderivada cuando el límite de
    integración se acerca a la singularidad.
    """
    x = funcion.x
    try:
        antiderivada = sp.integrate(funcion.expr, x)
    except (NotImplementedError, TypeError):
        return _verificar_convergencia_numerica(funcion, a, b, punto_singular)

    eps = sp.symbols("epsilon", positive=True)
    try:
        if abs(punto_singular - a) < 1e-9:
            valor = sp.limit(antiderivada.subs(x, a + eps), eps, 0, dir="+")
            resultado = antiderivada.subs(x, b) - valor
        elif abs(punto_singular - b) < 1e-9:
            valor = sp.limit(antiderivada.subs(x, b - eps), eps, 0, dir="+")
            resultado = valor - antiderivada.subs(x, a)
        else:
            lim_izq = sp.limit(antiderivada.subs(x, punto_singular - eps), eps, 0, dir="+")
            lim_der = sp.limit(antiderivada.subs(x, punto_singular + eps), eps, 0, dir="+")
            resultado = (lim_izq - antiderivada.subs(x, a)) + \
                        (antiderivada.subs(x, b) - lim_der)
        return bool(sp.im(resultado) == 0) and resultado.is_finite is not False
    except (NotImplementedError, TypeError, ValueError):
        return _verificar_convergencia_numerica(funcion, a, b, punto_singular)


def _verificar_convergencia_numerica(funcion: FuncionMatematica, a: float,
                                      b: float, punto_singular: float) -> bool:
    """Respaldo numérico: evalúa si la integral truncada cerca de la
    singularidad se estabiliza (converge) al reducir progresivamente el
    margen epsilon, usando scipy.integrate.quad."""
    from scipy import integrate as sci_integrate
    f = funcion.funcion_numerica()

    def f_escalar(t):
        return f(np.array([t]))[0]

    valores = []
    for eps in (1e-2, 1e-4, 1e-6):
        izq = max(a, punto_singular - (punto_singular - a))
        try:
            if abs(punto_singular - a) < 1e-9:
                val, _ = sci_integrate.quad(f_escalar, a + eps, b, limit=200)
            elif abs(punto_singular - b) < 1e-9:
                val, _ = sci_integrate.quad(f_escalar, a, b - eps, limit=200)
            else:
                v1, _ = sci_integrate.quad(f_escalar, a, punto_singular - eps, limit=200)
                v2, _ = sci_integrate.quad(f_escalar, punto_singular + eps, b, limit=200)
                val = v1 + v2
            valores.append(val)
        except Exception:
            return False

    if len(valores) < 2:
        return False
    # Si los valores se estabilizan (la diferencia entre las últimas
    # aproximaciones tiende a cero), se considera que converge.
    return abs(valores[-1] - valores[-2]) < 1e-2 * (1 + abs(valores[-1]))
