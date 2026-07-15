# Análisis y reflexión

## 1. ¿Qué significa que una función sea integrable?

Que una función f sea integrable (en el sentido de Riemann) en un intervalo
[a, b] significa que existe un único número real al que convergen tanto las
sumas superiores como las sumas inferiores de Riemann cuando la partición del
intervalo se hace arbitrariamente fina. Intuitivamente, significa que el
"área con signo" bajo la curva está bien definida: no importa cómo se
particione el intervalo ni qué puntos se elijan dentro de cada subintervalo
para evaluar f, el proceso de sumar rectángulos siempre converge al mismo
valor.

El criterio de Lebesgue formaliza esto de manera más general: una función
acotada en [a, b] es Riemann-integrable si y solo si el conjunto de sus
discontinuidades tiene medida cero. En la práctica esto cubre todos los casos
relevantes de un curso de Cálculo Integral: toda función continua es
integrable, y también lo es cualquier función continua a trozos (con un
número finito de discontinuidades removibles o de salto), porque un conjunto
finito de puntos siempre tiene medida cero.

Cuando la función no está acotada (tiene una asíntota vertical) dentro del
intervalo, ya no es Riemann-integrable en sentido estricto, pero muchas veces
el área sigue pudiéndose calcular tratando la integral como *impropia*: se
integra hasta un punto cercano a la asíntota y se toma el límite cuando ese
punto se acerca a la singularidad. Si ese límite existe y es finito, se dice
que la integral impropia *converge*; si no, *diverge* y el área no está
definida (es infinita).

## 2. ¿Cuál es la diferencia entre área y valor de la integral?

El valor de la integral definida ∫[a,b] f(x) dx es un número con signo: mide
la diferencia entre el área que queda por encima del eje x y la que queda por
debajo. Si f es negativa en una parte del intervalo, esa parte "resta" al
resultado.

El área, en cambio, es siempre una cantidad no negativa: es la suma de todas
las regiones geométricas encerradas, sin importar si están por encima o por
debajo del eje (o, en el caso del área entre curvas, sin importar cuál de las
dos funciones esté por arriba). Por eso, para calcular un área hay que
integrar el valor absoluto de la diferencia, |f(x) - g(x)|, en vez de
integrar simplemente f(x) - g(x): esto último daría un resultado con signo
que podría cancelar regiones donde f y g intercambian cuál está por encima,
subestimando el área real.

## 3. ¿Qué problemas surgen si la función no es continua?

Depende del tipo de discontinuidad:

- **Discontinuidad removible** (el límite existe pero no coincide con el
  valor de la función, o la función no está definida en ese punto puntual):
  no afecta la integrabilidad ni el valor de la integral, porque cambiar el
  valor de una función en un solo punto no cambia el área bajo la curva. Sin
  embargo, sí puede causar errores *numéricos*: si un método de integración
  con malla fija evalúa exactamente en ese punto, puede obtener `NaN` o un
  valor erróneo si no se maneja con cuidado (en este proyecto se resuelve
  ignorando esos puntos o partiendo el intervalo justo ahí).

- **Discontinuidad de salto finito**: la función sigue siendo acotada, así
  que la integral en el sentido de Riemann sigue existiendo (siempre que haya
  solo un número finito de saltos). Geométricamente, el área se calcula igual
  de bien, solo que la curva "salta" de un valor a otro sin pasar por los
  valores intermedios.

- **Discontinuidad infinita (asíntota vertical)**: aquí el problema es más
  serio, porque la función deja de estar acotada. La integral de Riemann
  propiamente no existe; hay que recurrir a una integral impropia. El área
  puede ser finita (si la "velocidad" con la que la función diverge es lo
  bastante lenta, como en 1/√|x|) o infinita (si diverge muy rápido, como en
  1/x²). Determinar cuál de los dos casos ocurre requiere evaluar un límite,
  no basta con "mirar la gráfica".

## 4. ¿Por qué es necesario dividir el intervalo en algunos casos?

Por dos razones distintas, ambas presentes en este proyecto:

1. **Por el valor absoluto en la fórmula del área.** Como
   A = ∫[a,b] |f(x) - g(x)| dx, y el valor absoluto no es una función
   derivable/integrable "de una sola pieza" cuando su argumento cambia de
   signo, hay que partir [a, b] en los puntos donde f(x) - g(x) = 0 (las
   intersecciones). En cada subintervalo resultante, f - g mantiene un signo
   constante, así que ahí sí se puede quitar el valor absoluto reemplazándolo
   por (f - g) o (g - f) según cuál domine, y aplicar la integral normal.

2. **Por las discontinuidades.** Si f o g tienen una asíntota vertical
   dentro de [a, b], no se puede integrar de corrido sobre todo el
   intervalo: hay que partir justo en el punto singular y tratar cada lado
   como una integral impropia independiente (tomando el límite cuando el
   extremo de integración se acerca a la singularidad). Esto es necesario
   incluso cuando ambos lados convergen, porque la integral "de corrido" no
   está definida hasta que no se demuestra que ambos límites existen.

## 5. ¿Cómo afecta el método numérico al resultado?

El método numérico elegido determina tanto la exactitud del resultado como
su costo computacional (número de evaluaciones de la función):

- **Trapecio** tiene error de orden O(h²): si se duplica el número de
  subintervalos, el error se reduce aproximadamente a la cuarta parte. Es
  simple pero relativamente lento en converger.
- **Simpson 1/3** tiene error de orden O(h⁴): con el mismo número de
  subintervalos que Trapecio, típicamente da un resultado mucho más preciso,
  porque aproxima la curva con parábolas en vez de segmentos rectos.
- **Gauss-Legendre** puede lograr una precisión muy alta con muy pocos
  nodos cuando la función es suave (polinomial o cercana a serlo), porque
  está diseñado para ser exacto en polinomios de grado alto.
- **Romberg** mejora Trapecio mediante extrapolación de Richardson, sin
  necesidad de evaluar derivadas, logrando convergencia rápida en funciones
  suaves.
- **Monte Carlo** converge mucho más lentamente (error de orden O(1/√N)),
  por lo que necesita muchísimas más evaluaciones para alcanzar la misma
  precisión que los métodos anteriores; su ventaja aparece en problemas de
  dimensión alta, no en integrales de una sola variable como esta.

Además, cuando el integrando tiene un punto "difícil" (una intersección con
un pico agudo en |f - g|, o una singularidad), los métodos de malla fija
que no saben de antemano dónde está ese punto pueden perder precisión si el
punto cae justo entre dos nodos consecutivos, o incluso fallar (dividir por
cero) si el punto coincide exactamente con un nodo. Por eso este proyecto
parte el intervalo exactamente en las intersecciones y discontinuidades
antes de aplicar cualquier método numérico, y usa cuadratura adaptativa
(Gauss-Kronrod) como valor de referencia de alta precisión para poder medir
el error real de cada método.

---

# Conclusión

Este proyecto muestra que calcular el área entre dos curvas no es solamente
"aplicar una fórmula de integración numérica"; requiere primero un análisis
matemático cuidadoso que la implementación computacional debe respetar en
cada paso:

1. El **análisis de integrabilidad** determina si la integral tiene sentido
   antes de intentar calcularla. Ignorar este paso puede llevar a un
   programa que reporte un número "razonable" para una integral que en
   realidad diverge (como ocurriría con 1/x² si solo se aplicara Trapecio
   sin verificar primero si la integral impropia converge).

2. El **análisis de intersecciones** no es un simple detalle estético para
   la gráfica: define la partición del intervalo que hace válido reemplazar
   |f(x) - g(x)| por una expresión sin valor absoluto en cada subintervalo,
   que es justamente lo que permite integrar numéricamente de forma
   correcta.

3. La **elección e implementación del método numérico** debe ir de la mano
   del análisis anterior: un método de malla fija aplicado ciegamente sobre
   todo el intervalo, sin tener en cuenta ni las intersecciones ni las
   discontinuidades, puede producir errores grandes o incluso resultados sin
   sentido (`NaN`, `inf`) exactamente en los puntos donde el análisis
   matemático predice que hay que tener cuidado.

En resumen: el código no reemplaza el análisis matemático, lo *automatiza*
una vez que ese análisis ya identificó dónde están los puntos delicados
(intersecciones, discontinuidades, asíntotas). Un programa que calcule áreas
"a ciegas", sin ese análisis previo, puede parecer que funciona en los casos
sencillos pero fallar silenciosamente —o ruidosamente— en los casos donde
más importa ser riguroso.
