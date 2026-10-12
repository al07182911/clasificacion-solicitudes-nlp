
# Dataset – Bitext Customer Support

## Origen de los datos

El proyecto utiliza el conjunto de datos público *Bitext Customer Support – Training Dataset for Chatbots/Virtual Assistants*, disponible en Kaggle.

**Fuente:** https://www.kaggle.com/datasets/bitext/training-dataset-for-chatbotsvirtual-assistants

## Características

- Registros originales: 21,534.
- Registros después de la limpieza: 21,513.
- Categorías generales: 11.
- Intenciones específicas: 27.
- Idioma: inglés.
- Formato: CSV.

## Estructura del dataset

| Campo | Descripción |
|---|---|
| `flags` | Indicadores lingüísticos adicionales |
| `utterance` | Texto de la solicitud del cliente |
| `category` | Categoría general de la solicitud |
| `intent` | Intención específica del cliente |

## Preparación de los datos

El procedimiento de preparación incluye:

1. Obtención del dataset original desde Kaggle.
2. Carga del archivo CSV en Google Colab.
3. Validación de valores nulos y registros duplicados.
4. Eliminación de registros duplicados y textos repetidos.
5. División estratificada del conjunto limpio en 70 % para desarrollo y 30 % para prueba.
6. Utilización de una semilla aleatoria fija (`random_state=42`) para garantizar reproducibilidad.

## Resultados de preparación

| Conjunto | Registros |
|---|---:|
| Dataset original | 21,534 |
| Dataset limpio | 21,513 |
| Desarrollo (70 %) | 15,059 |
| Prueba (30 %) | 6,454 |

## Reproducibilidad

El notebook `notebooks/comparacion_modelos_nlp.ipynb` contiene el procedimiento utilizado para obtener y preparar los datos.

El dataset se descarga en el directorio temporal `/content/` de Google Colab, por lo que no es necesario almacenar una copia adicional en este repositorio.

La reproducción del procedimiento requiere acceso a la fuente pública de Kaggle y la ejecución de las celdas correspondientes del notebook.

## Seguridad y privacidad

Se utilizan exclusivamente datos públicos para evitar el tratamiento de información real de clientes de Seguros BBVA.

El dataset se emplea con fines académicos y de experimentación en Procesamiento de Lenguaje Natural.
