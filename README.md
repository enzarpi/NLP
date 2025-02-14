# Análisis del Corpus: REVIEWS HOME AND KITCHEN

# Contenido del Notebook 1: Descarga y Exploración del Corpus

## Instalación y Carga de Librerías
- Instalación de librerías necesarias: **spaCy, nltk, gensim, pandas, wordcloud, matplotlib, seaborn**, entre otras.
- Importación de librerías para el análisis de datos, procesamiento de texto y visualización.

## Carga y Exploración Inicial del Dataset
- Descarga del dataset **REVIEWS HOME AND KITCHEN** desde un archivo comprimido JSON.
- Visualización de las primeras filas del dataset y comprobación de valores nulos.

## Selección y Preparación del Dataset
- Filtrado de columnas relevantes (**reviewText** y **overall**).
- Comprobación de valores nulos y descripción de la distribución de datos.
- **Justificación de la selección de columnas**:
  - **reviewText**: Contiene la opinión del usuario y será el texto preprocesado.
  - **overall**: Representa la calificación asignada por el usuario, útil para el análisis de sentimiento.

## Análisis Exploratorio de Texto
- Generación de **n-gramas** para identificar patrones comunes en las reseñas.
- Cálculo de **frecuencias de palabras** y visualización con **WordCloud**.
- Identificación de **palabras más comunes** en las reseñas.

## Visualización de Datos
- Distribución de la **longitud de las reseñas** en función de la calificación.
- Representación gráfica de la **frecuencia de palabras** más comunes en el corpus.


# Contenido del Notebook 2: Preprocesado de Texto

## Instalación y Carga de Librerías
- Instalación de librerías necesarias para procesamiento de lenguaje natural y análisis de texto (**spaCy, nltk, gensim, stop-words, scikit-learn**, etc.).
- Importación de las librerías utilizadas en el preprocesamiento.

## Carga y Exploración Inicial del Dataset
- Carga del dataset comprimido JSON (**REVIEWS HOME AND KITCHEN**).
- Exploración de datos:
  - Visualización de la estructura del dataset.
  - Eliminación de valores nulos.
  - Selección de columnas relevantes (**reviewText** y **overall**).

## Preprocesado de Texto
- Creación de una **función en Python** para aplicar preprocesamiento de texto.
- **Procesos aplicados**:
  - Eliminación de caracteres especiales y **tokenización**.
  - Conversión de texto a **minúsculas** y eliminación de **stopwords**.
  - **Lematización** para reducir palabras a su forma base.
- Identificación de **errores en tokenización** y ajuste del preprocesamiento según los **n-gramas** analizados.

## Vectorización y Representación del Texto
- Uso de **CountVectorizer** para representar el texto de forma numérica.
- Generación de **embeddings con Word2Vec** para representar palabras en un espacio vectorial.
- Aplicación de **reducción de dimensionalidad con PCA** para visualizar la relación entre palabras.

## Visualización y Análisis
- Creación de **n-gramas** y análisis de **frecuencias**.
- Generación de **nubes de palabras (WordCloud)** para visualizar términos más utilizados.
- Distribución de **palabras clave en función de la calificación** de las reseñas.


# Contenido del Notebook 3: Entrenamiento y Testeo de un Modelo de Análisis de Sentimiento

## Instalación y Carga de Librerías
- Instalación y carga de librerías necesarias para el entrenamiento del modelo (**scikit-learn, spaCy, pandas, numpy, matplotlib, seaborn**, etc.).

## Preparación de Datos
- Carga del dataset preprocesado (**preprocessed_reviews2.csv**).
- División de los datos en conjuntos de **entrenamiento y prueba**, manteniendo la **proporción de clases** (estratificación).

## Codificación de Textos con TF-IDF (Bag-of-Words)
- Transformación del texto en una representación numérica mediante **TF-IDF**.
- **Configuración del vectorizador**:
  - `max_features=5000`: Limita el vocabulario a las **5000 palabras más frecuentes**.
  - `ngram_range=(1,2)`: Incluye **unigramas y bigramas**.
  - `stop_words='english'`: Elimina palabras irrelevantes.

## Entrenamiento de Modelos
- Comparación de **dos modelos de Machine Learning**:
  - **Regresión Logística**
  - **Random Forest**
- Ajuste de **hiperparámetros y optimización** de cada modelo.

## Evaluación y Comparación de Modelos
- Cálculo de **métricas de rendimiento**:
  - **Precisión (Accuracy)**
  - **Matriz de confusión**
  - **Reporte de clasificación** (**Precision, Recall, F1-Score**)
- Comparación de los modelos para elegir el **mejor enfoque**.

## Predicción sobre Nuevas Reseñas
- Aplicación del **modelo entrenado** sobre **nuevas reseñas** para predecir su **sentimiento**.
- Análisis de ejemplos de **predicciones correctas e incorrectas**.


# Contenido del Notebook 3.1: Entrenamiento y Testeo de un Modelo de Análisis de Sentimiento con Undersampling

## Instalación y Carga de Librerías
- Instalación y carga de librerías necesarias (**scikit-learn, seaborn, pandas, numpy, matplotlib**, etc.).

## Preparación de Datos con Undersampling
- Carga del dataset preprocesado (**preprocessed_reviews2.csv**).
- Conversión de la variable objetivo (**overall**) en una clasificación **binaria** (positivo/negativo).
- Aplicación de **undersampling** para balancear las clases eliminando datos de la clase mayoritaria.

## Codificación de Textos con TF-IDF (Bag-of-Words)
- Transformación del texto en una representación numérica mediante **TF-IDF**.
- **Configuración del vectorizador**:
  - `max_features=5000`: Reduce la **dimensionalidad** y mejora la **eficiencia**.
  - `ngram_range=(1,2)`: Usa **unigramas y bigramas**.
  - `stop_words='english'`: Filtra **palabras irrelevantes**.

## Entrenamiento de Modelos
- Entrenamiento de **dos modelos**:
  - **Regresión Logística**
  - **Random Forest**
- Comparación del **rendimiento** de ambos modelos.

## Evaluación de Modelos
- Análisis de **métricas de rendimiento**:
  - **Precisión (Accuracy)**
  - **Matriz de confusión**
  - **F1-score**
  - **Reporte de clasificación**
- Comparación de los modelos **con y sin balanceo de clases**.

## Predicción y Análisis
- Uso del modelo para **predecir el sentimiento** de nuevas reseñas.
- Evaluación del **impacto del undersampling** en los resultados.


# Reporte de métricas y conclusiones

En los notebooks **3.0** y **3.1** se incluye la **cuarta etapa de la práctica**, así como los **comentarios** y las **conclusiones finales**.
