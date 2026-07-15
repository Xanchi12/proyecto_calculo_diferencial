# Área entre Curvas, Integrabilidad y Programación

Proyecto para la actividad de recuperación de **Cálculo Integral**. Calcula
el área entre dos funciones f(x) y g(x) en un intervalo [a, b], realizando
primero un análisis riguroso de integrabilidad e intersecciones, y luego un
cálculo numérico del área con comparación de varios métodos y su error.

## Estructura del proyecto

```
proyecto_calculo/
├── funciones.py              # Parte 1: representación y evaluación de f(x), g(x)
├── integrabilidad.py         # Parte 2: análisis de continuidad / discontinuidades
├── intersecciones.py         # Parte 3: búsqueda de puntos donde f(x) = g(x)
├── area_entre_curvas.py      # Parte 4 y 5: partición del intervalo y cálculo del área
├── integracion_numerica.py   # Parte 5: métodos numéricos y análisis de error
├── visualizacion.py          # Parte 6: gráficas del área y de convergencia
├── app.py                    # Interfaz web (Flask) — recomendada
├── templates/index.html      # Página de la interfaz web
├── main.py                   # Programa de consola (CLI)
├── test_proyecto.py          # Pruebas unitarias de cada módulo
├── preguntas_reflexion.md    # Respuestas al análisis y reflexión + conclusión
├── requirements.txt
└── README.md
```

## Instalación

Requiere Python 3.10+ (se usa la sintaxis `float | None`).

```bash
pip install -r requirements.txt
```

## Uso

### 0. Interfaz web (recomendado — todo en una sola página)

```bash
python app.py
```

Luego abre **http://127.0.0.1:5000** en tu navegador. Vas a ver un
formulario para ingresar f(x), g(x), a y b. Al presionar "Calcular y
graficar", cada parte del análisis (integrabilidad, intersecciones,
área, comparación de métodos y la gráfica) va apareciendo en la misma
página a medida que se calcula — no hace falta buscar archivos `.png`
generados, todo se muestra ahí mismo. También incluye 4 botones de
ejemplos rápidos para probar sin escribir nada.

### 1. Modo interactivo (por consola)

```bash
python main.py
```
Pide f(x), g(x), a y b por teclado.

### 2. Modo con argumentos (por consola)

```bash
python main.py --f "x**2 - 1" --g "1 - x**2/2" --a -2 --b 2 --metodo simpson_1_3
```

Métodos disponibles en `--metodo`: `trapecio`, `simpson_1_3`, `simpson_3_8`,
`gauss_legendre`, `romberg`.

### 3. Ejemplos predefinidos (por consola)

El proyecto incluye 4 ejemplos que ilustran distintos casos pedidos por la
actividad:

```bash
python main.py --ejemplo 1   # Dos parábolas que se cruzan en dos puntos
python main.py --ejemplo 2   # f con una discontinuidad removible
python main.py --ejemplo 3   # f con asíntota vertical, integral impropia CONVERGENTE
python main.py --ejemplo 4   # f con asíntota vertical, integral impropia DIVERGENTE
```

Cada ejecución imprime en consola el análisis completo (Partes 1 a 6) y
genera dos imágenes: la gráfica del área entre curvas y (cuando aplica) la
gráfica de convergencia del método numérico.

## Ejecutar las pruebas

```bash
python -m unittest test_proyecto.py -v
```

## Notas de diseño relevantes para el reporte

- Las funciones se parsean con SymPy **sin simplificar automáticamente**,
  para no cancelar accidentalmente discontinuidades removibles (por ejemplo,
  simplificar `(x**2-4)/(x-2)` a `x+2` borraría la discontinuidad en x=2 que
  la actividad pide detectar).
- El intervalo [a, b] se particiona tanto en las **intersecciones** de f y g
  como en cualquier **discontinuidad** detectada de f o de g, porque ambos
  tipos de puntos requieren tratamiento especial: los primeros por el valor
  absoluto de la fórmula del área, los segundos porque un método numérico de
  malla fija no puede evaluar de forma confiable justo sobre una
  singularidad.
- Cuando un subintervalo tiene una asíntota vertical en un extremo, se
  distingue automáticamente si la integral impropia converge o diverge
  (evaluando límites simbólicos con SymPy, con respaldo numérico si SymPy no
  puede resolverlo), y se reporta el área como un valor finito o como
  infinita, en vez de dejar que un método numérico ciego produzca un número
  sin sentido.
