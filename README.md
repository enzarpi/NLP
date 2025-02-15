# Análisis del Corpus: REVIEWS HOME AND KITCHEN

# Contenido del Notebook 1: Descarga y Exploración del Corpus

## Instalación y Carga de Librerías
- Instalación de librerías necesarias (**spaCy, scikit-learn, seaborn, pandas**, etc.).
- Configuración inicial del entorno de trabajo.

## Carga y Exploración del Dataset
- Descarga y carga del dataset comprimido JSON (**reviews_Home_and_Kitchen_5.json.gz**).
- Exploración inicial del dataset:
  - Visualización de las primeras filas.
  - Comprobación de valores nulos.
  - Información estadística de los datos.

## Selección y Preparación del Dataset
- Filtrado de las columnas relevantes para el **análisis de sentimiento**:
  - **reviewText**: Texto de la reseña que será analizado.
  - **overall**: Calificación del usuario (representa el sentimiento).
- Justificación de la eliminación de **columnas irrelevantes** (**reviewerID, asin**, etc.).
- Verificación de **datos finales** antes del preprocesamiento.

## Análisis Exploratorio
### **Cardinalidad del Vocabulario**
- Análisis del **número total de palabras únicas** en el corpus.
- Distribución de la **frecuencia de palabras** en el dataset.

### **Distribución de Clases en las Reviews**
- Visualización de la **cantidad de reseñas** (positivas, neutras y negativas).
- Análisis de la **proporción** de cada categoría en el dataset.
- Estadísticas básicas sobre la **longitud de las reseñas**.
- Distribución de la **calificación (`overall`)** en el dataset.


# Contenido del Notebook 2: Preprocesado de Texto

## Instalación y Carga de Librerías
- Instalación de librerías necesarias (**spaCy, scikit-learn, nltk, gensim, wordcloud**, etc.).
- Configuración inicial del entorno de trabajo.

## Carga del Dataset Preprocesado de la Etapa 1
- Importación del dataset previamente **explorado y filtrado**.
- Revisión de los datos cargados y verificación de **valores nulos** antes del preprocesamiento.

## Preprocesado de Texto
### **Limpieza y Normalización**
- Eliminación de **caracteres especiales** y **espacios en blanco**.
- Conversión de texto a **minúsculas**.
- Eliminación de **stopwords**.

### **Tokenización y Lematización**
- Uso de **spaCy** para dividir el texto en palabras y convertirlas a su **forma base**.

### **Corrección de Errores y Refinamiento**
- Análisis de **palabras poco informativas**.
- Ajustes adicionales para mejorar la calidad del **corpus**.

## Análisis del Texto Preprocesado (Visualización)
### **Frecuencia de Palabras Más Comunes**
- Identificación de **palabras más usadas** en el corpus.

### **N-grams Más Frecuentes**
- Análisis de combinaciones de palabras (**bigrams y trigrams**) más repetidas.

### **Generación de Nube de Palabras (WordCloud)**
- Representación visual de los términos más frecuentes en el corpus preprocesado.

### **Distribución de la Longitud de las Reseñas**
- Visualización del **número de palabras por reseña** tras el preprocesamiento.

## Representación de Texto con Word Embeddings
### **Entrenamiento de Word Embeddings con Word2Vec**
- Generación de **representaciones vectoriales** de las palabras en base a su contexto.

### **Visualización en 2D de Word Embeddings**
- Uso de **PCA** para reducir la dimensionalidad de los embeddings y representar las relaciones entre palabras en un gráfico **bidimensional**.



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
