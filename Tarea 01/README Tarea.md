# Tarea Integradora: Regresión Lineal y Agrupamiento 

## 🎯 Objetivo general

Suponga que tiene que asesorar a un inversionista que busca identificar oportunidades en el mercado inmobiliario o en la distribución de combustibles, utilizando:
	•	Aprendizaje supervisado: modelos de regresión lineal (simple y múltiple)

	•	Aprendizaje no supervisado: modelos de agrupamiento (clustering)

Esta actividad es en equipos de 3 o 4 personas. Los equipos elegirán un dataset y una técnica (Regresión o Clustering).

## 📊 Conjuntos de datos disponibles
	1.	Propiedades residenciales (Zillow)
Variables típicas: precio, ubicación, recámaras, baños, superficie, año de construcción, etc.

	2.	Precios diarios de gasolina (México)
Variables: estación, ubicación, tipo de combustible, precio diario, etc.

## 📁 Datos: 
https://drive.google.com/drive/folders/1tRHUCpZlN7jL2QUYOQm1xLnvw6TFLFw4?usp=sharing

## 👥 Modalidad y entrega
	•	Equipos con las personas que han venido trabajando (recomendado).

	•	Entrega: Informe + código reproducible.

	•	Fecha de entrega: 1 de marzo de 2026.

	•	Un miemboro del equipo, enviar a: benjov@ciencias.unam.mx; indicando el nombre de todas las personas integrantes.

# Dos opciones 

## Opción A: 📈 Regresión lineal (simple/múltiple)

🎯 Objetivo específico

Construir un modelo de regresión lineal para explicar y predecir el precio:
	•	Inmobiliario: precio de propiedad (o precio por m², si tiene más sentido)

	•	Gasolina: precio por litro (para un tipo de combustible y/o región)

y convertirlo en recomendaciones de inversión basadas en evidencia.

💼 Contexto de decisión
	•	Comprar propiedades donde el modelo sugiera mayor “valor esperado” dadas características y ubicación.

	•	Identificar zonas/estaciones donde el precio (o nivel esperado) sea consistentemente alto/bajo, o donde ciertas variables estén asociadas a incrementos relevantes.

🧠 Actividades
	1.	Definir variable objetivo (Y)

	•	Zillow: precio o log(precio) si hay mucha asimetría.

	•	Gasolina: precio (filtrando por combustible) y controlando por región/estación/tiempo.

	2.	Seleccionar variables explicativas (X)

	•	Continuas: superficie, baños, recámaras, año, etc.

	•	Categóricas: ciudad/estado/colonia, tipo de combustible, marca, etc. (usar dummies/one-hot).

	3.	Limpieza y preprocesamiento mínimo

	•	Tratamiento de faltantes (decisión explícita: eliminar/imputar simple).

	•	Codificación de categóricas.

	•	Transformaciones justificadas (ej. log(precio)).

	4.	Partición entrenamiento/prueba

	•	Separar en train/test (y reportar proporción).

	5.	Ajuste de modelos

	•	Modelo base (simple o con pocas X).

	•	Modelo múltiple (mejorado).

	•	Comparación entre modelos (qué variables aportan y por qué).

	6.	Evaluación y diagnóstico (lo esencial)

	•	Métricas en test: RMSE y MAE (y R² si lo trabajaron en clase).

	•	Revisión básica de supuestos / diagnóstico gráfico (según lo visto):

	•	residuales vs ajustados

	•	normalidad aproximada de residuales

	•	outliers/influencia (al menos discusión si aparecen)

	7.	Interpretación económica y recomendación

	•	¿Qué variables elevan el precio y cuánto (signos/magnitudes)?

	•	Conclusión: ¿dónde conviene invertir y por qué?

	•	Incluir al menos 2 escenarios tipo “si una propiedad tiene X características…” o “si una estación está en X zona…”

## Opción B: 🔍 Clustering (agrupamiento)

🎯 Objetivo específico

Encontrar grupos de observaciones similares para identificar “perfiles” de zonas/estaciones/propiedades y generar una recomendación de inversión basada en esos perfiles.

Ejemplos:
	•	Zillow: agrupar propiedades por características (precio, m², recámaras, baños, antigüedad) y ver qué clusters representan “valor” vs “sobreprecio”.

	•	Gasolina: agrupar estaciones por comportamiento (promedio/variabilidad del precio, nivel por tipo de combustible, etc.) o por patrones agregados por región.

💼 Contexto de decisión
	•	Detectar clusters “premium”, “económicos”, “alta variabilidad”, “estables”, etc.

	•	Recomendar invertir en clusters con mejor relación costo–potencial o menor riesgo.

🧠 Actividades
	1.	Definir unidad de análisis

	•	Zillow: una fila = propiedad.

	•	Gasolina: decidir si una fila = estación (con variables agregadas) o estación-fecha (y luego resumir).

Sugerencia: para clustering suele funcionar mejor construir features agregadas (promedio, desviación, min/max, tendencia simple).
	2.	Seleccionar variables para agrupar

	•	Preferir variables numéricas y comparables (precio, m², etc.).

	•	Si hay categóricas, justificar cómo se usan (o excluirlas del clustering principal).

	3.	Estandarización / normalización

	•	Indispensable para K-means y distancia euclidiana (según clase).

	4.	Aplicar 1–2 técnicas de clustering vistas en clase

	•	K-means (obligatorio o principal)

	•	y/o jerárquico / DBSCAN (si se vio y aplica)

	5.	Elección del número de clusters (si aplica)

	•	Método del codo, silhouette, o el criterio que hayan visto.

	6.	Visualización e interpretación

	•	Gráficas: dispersión (si reduces dimensión), boxplots por cluster, centroides/perfiles.

	•	Si hay ubicación: mapa simple o visual por región (opcional recomendado).

	7.	Recomendación de inversión

	•	Identificar clusters con características deseables (ej. “precio moderado + alta superficie” o “estaciones estables con precios competitivos”).

	•	Justificar con evidencia: perfiles y métricas descriptivas por cluster.

# 📝 Entregables

1) Informe técnico (máx. 6 páginas, PDF)

Debe incluir:
	1.	Planteamiento del problema (qué decisión tomará el inversionista)

	2.	Datos (fuente, limpieza, variables seleccionadas, unidad de análisis)

	3.	Metodología (regresión o clustering, justificación y pasos)

	4.	Resultados

	•	Regresión: métricas, comparación de modelos, diagnóstico básico, interpretación de coeficientes

	•	Clustering: elección de k/criterio, perfiles por cluster, visualizaciones

	5.	Recomendación final (con argumentos cuantitativos)

2) Código reproducible
	•	Jupyter Notebook o script en Python/R

	•	Debe correr de inicio a fin (con rutas claras o instrucciones para cargar datos).

3) Visualización geográfica (opcional, recomendado)
	•	Mapa por estado/ciudad o scatter con coordenadas si existen.

# ✅ Criterios de evaluación (100 pts)
	•	(25) Correcta aplicación de lo visto (pipeline y método)

	•	(20) Limpieza/preprocesamiento y decisiones justificadas

	•	(20) Evaluación y evidencia (métricas/criterios + visualizaciones)

	•	(20) Interpretación orientada a decisión (no solo “hacer modelo”)

	•	(15) Claridad del informe, narrativa y presentación

# Errores comunes (para evitar)

## Regresión lineal
	1.	Data leakage: evaluar con los mismos datos con los que entrenaste (sin train/test).

	2.	Interpretar mal dummies: “ubicación = 1” no significa más ubicación; significa pertenecer a una categoría vs base.

	3.	No revisar escala/transformación: precios muy sesgados → log(precio) puede mejorar.

	4.	Confundir correlación con causalidad: el modelo explica/predice, no prueba causalidad.

	5.	Overfitting con demasiadas variables: muchos predictores sin justificación y sin validación.

## Clustering
	1.	No estandarizar antes de K-means/jerárquico con distancia euclidiana.

	2.	Mezclar variables con escalas muy distintas (precio vs año) sin normalizar.

	3.	Elegir k “a ojo” sin método del codo/silhouette o sin argumentarlo.

	4.	Clusters sin interpretación: si no puedes describirlos con perfiles, no sirven para decisión.

	5.	Usar variables irrelevantes o redundantes que dominan la distancia.

## Checklist rápido para equipos (lo mínimo indispensable)

Si eliges Regresión
	•	Definí claramente Y y mis X.

	•	Separé train/test (y lo reporté).

	•	Entrené al menos 2 modelos (base vs mejorado).

	•	Reporté MAE + RMSE en test (y R² si aplica).

	•	Incluí al menos 2 gráficos (pred vs real; residuales o similares).

	•	Cerré con una recomendación concreta (zona/segmento/escenario).

Si eliges Clustering
	•	Elegí variables numéricas relevantes para agrupar.

	•	Estandaricé variables.

	•	Elegí k con codo/silhouette (o justifiqué DBSCAN/jerárquico).

	•	Mostré perfiles por cluster (tabla + boxplots/medias).
	
	•	Recomendé qué cluster conviene y por qué (riesgo/beneficio).

