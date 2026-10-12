# Clasificación de solicitudes de clientes mediante NLP

## 1. Descripción del proyecto

Este proyecto forma parte de una actividad académica del Máster en Inteligencia Artificial y tiene como objetivo comparar tres modelos preentrenados de Hugging Face para la clasificación automatizada de solicitudes de clientes mediante técnicas de Procesamiento de Lenguaje Natural (NLP).

El caso de uso está inspirado en los procesos de atención y cancelación de pólizas del sector asegurador, tomando como referencia el contexto de Seguros BBVA.

El prototipo utiliza clasificación *zero-shot* para identificar la intención expresada por un cliente sin necesidad de realizar entrenamiento adicional de los modelos.

## 2. Objetivo

Evaluar comparativamente tres modelos preentrenados de Hugging Face, utilizando métricas de calidad predictiva y eficiencia computacional, para identificar la alternativa más adecuada para clasificar solicitudes de clientes.

## 3. Dataset

Se utiliza el dataset público **Bitext Customer Support – Training Dataset for Chatbots/Virtual Assistants**, disponible en Kaggle.

**Fuente:** https://www.kaggle.com/datasets/bitext/training-dataset-for-chatbotsvirtual-assistants

**Características:**

- Registros originales: 21,534.
- Registros después de la limpieza: 21,513.
- Categorías generales: 11.
- Intenciones específicas: 27.
- Variables: `flags`, `utterance`, `category` e `intent`.
- División estratificada: 70 % desarrollo y 30 % prueba.
- Semilla aleatoria: 42.

Por razones de seguridad de la información, confidencialidad y protección de datos personales, se utilizaron exclusivamente datos públicos, evitando el tratamiento de información real de clientes de Seguros BBVA.

El dataset se obtiene mediante el procedimiento documentado en el notebook y se almacena temporalmente en el directorio `/content/` de Google Colab para facilitar su procesamiento.

## 4. Modelos evaluados

Se implementaron los siguientes modelos preentrenados:

1. **DeBERTa-v3-small:** `cross-encoder/nli-deberta-v3-small`
2. **DeBERTa-v3-base:** `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`
3. **BART-large-MNLI:** `facebook/bart-large-mnli`

Los tres modelos se evaluaron mediante pipelines de clasificación *zero-shot*, utilizando las mismas 27 intenciones candidatas y la siguiente plantilla de hipótesis:

`This customer request is about {}.`

La implementación permite comparar el desempeño de los modelos sin realizar procesos adicionales de entrenamiento supervisado (*fine-tuning*).

## 5. Entorno de ejecución

El prototipo fue desarrollado utilizando el siguiente entorno tecnológico:

- Plataforma: Google Colab.
- Lenguaje: Python 3.13.15.
- GPU: NVIDIA Tesla T4.
- Framework: PyTorch con soporte CUDA.
- Librerías principales: Transformers, Accelerate, Hugging Face Hub, pandas, NumPy, scikit-learn y Matplotlib.
- Método de inferencia: *zero-shot classification*.

Las dependencias y versiones de referencia se encuentran documentadas en el archivo `requirements.txt`.

Para reproducir las mediciones de latencia en condiciones comparables, se recomienda utilizar una GPU NVIDIA Tesla T4 o equivalente.

La disponibilidad de GPU en Google Colab puede estar sujeta a límites de uso y disponibilidad de recursos.

## 6. Configuración de autenticación en Hugging Face Hub

Para ejecutar el notebook siguiendo el procedimiento implementado, es necesario disponer de una cuenta de Hugging Face y configurar un token personal de acceso con permisos de lectura (*Read*).

Aunque los modelos seleccionados son públicos y normalmente pueden descargarse sin autenticación, el notebook incluye un procedimiento de autenticación mediante Hugging Face Hub.
Configuración del token de Hugging Face en Google Colab

El token debe almacenarse en la sección Secretos de Google Colab, utilizando exactamente el siguiente nombre:

HF_TOKEN

Este nombre debe respetarse, ya que el código Python recupera la credencial mediante la instrucción userdata.get("HF_TOKEN").

Una vez creado el secreto, deberá habilitarse el acceso del notebook a este. El valor del token es personal y confidencial, por lo que no debe incluirse directamente en el código ni publicarse en GitHub.

### Procedimiento de configuración

1. Crear una cuenta en https://huggingface.co/
2. Acceder a https://huggingface.co/settings/tokens
3. Generar un token personal con permisos de lectura (*Read*).
4. Abrir el notebook del proyecto en Google Colab.
5. Seleccionar la sección **Secretos**, identificada mediante el icono de llave en el panel lateral.
6. Crear un nuevo secreto con el nombre `HF_TOKEN`.
7. Introducir el token generado como valor del secreto.
8. Habilitar el acceso del notebook al secreto.
9. Ejecutar la celda de autenticación antes de cargar los modelos.

### Consideraciones de seguridad

El token de Hugging Face constituye una credencial personal y confidencial.

Por razones de seguridad:

- No debe escribirse directamente en las celdas del notebook.
- No debe incluirse en `requirements.txt`.
- No debe publicarse en GitHub.
- Debe almacenarse utilizando el mecanismo de secretos de Google Colab.
- Se recomienda utilizar únicamente los permisos necesarios para ejecutar el proyecto.

Este procedimiento permite autenticar el acceso al Hub sin exponer credenciales en el código fuente.

## 7. Procedimiento de ejecución

Para reproducir el prototipo, se deberán realizar los siguientes pasos:

1. Acceder al repositorio de GitHub y abrir el notebook del proyecto en Google Colab.
2. Activar la GPU desde **Entorno de ejecución → Cambiar tipo de entorno de ejecución**.
3. Verificar que PyTorch detecte correctamente CUDA.
4. Instalar las dependencias necesarias conforme al archivo `requirements.txt`.
5. Configurar el token `HF_TOKEN` mediante los secretos de Google Colab.
6. Ejecutar la celda de autenticación en Hugging Face Hub.
7. Ejecutar las celdas correspondientes a la obtención y carga del dataset Bitext.
8. Realizar las validaciones de calidad, limpieza y eliminación de textos duplicados.
9. Ejecutar la división estratificada de los datos, utilizando el 70 % para desarrollo y el 30 % para prueba.
10. Cargar los tres modelos preentrenados de Hugging Face.
11. Ejecutar los pipelines de clasificación *zero-shot*.
12. Calcular las métricas de accuracy, precision, recall y F1-score.
13. Medir la latencia de inferencia de cada modelo.
14. Consultar las tablas, gráficos y conclusiones del análisis comparativo.

## 8. Resultados experimentales

La evaluación piloto se realizó con 100 solicitudes del conjunto de desarrollo, utilizando las mismas 27 intenciones candidatas para los tres modelos.

| Métrica | DeBERTa-small | DeBERTa-base | BART-large |
|---|---:|---:|---:|
| Accuracy | 67.00 % | 71.00 % | 80.00 % |
| Precision macro | 56.76 % | 61.04 % | 76.29 % |
| Recall macro | 58.56 % | 60.93 % | 73.77 % |
| F1-score macro | 51.99 % | 57.43 % | 71.44 % |
| Latencia media (s) | 0.4062 | 0.7379 | 0.6104 |
| Latencia mediana (s) | 0.3743 | 0.7034 | 0.5690 |

Los resultados corresponden a una evaluación piloto y no representan una validación definitiva sobre el conjunto de prueba reservado.

## 9. Conclusiones

BART-large-MNLI obtuvo el mejor desempeño predictivo de los tres modelos evaluados, alcanzando un accuracy del 80 % y un F1-score macro del 71.44 %.

DeBERTa-v3-small presentó la menor latencia media, con 0.4062 segundos por solicitud, lo que representa una ventaja para escenarios donde la velocidad de procesamiento sea prioritaria.

Por su parte, DeBERTa-v3-base obtuvo resultados intermedios en las métricas de clasificación, aunque registró la mayor latencia de los tres modelos.

Los resultados evidencian que existe un compromiso entre calidad predictiva y eficiencia computacional, por lo que la selección del modelo depende de las necesidades del caso de uso.

Con base en la evaluación piloto, se recomienda **BART-large-MNLI** como principal candidato para continuar la validación del prototipo, debido a su mayor desempeño en las métricas de clasificación y a una latencia intermedia.

## 10. Limitaciones y riesgos

El dataset utilizado contiene solicitudes generales de atención al cliente y no representa directamente los motivos específicos de cancelación de pólizas de seguros.

La evaluación piloto comprende únicamente 100 solicitudes y algunas intenciones presentan pocos ejemplos, lo que limita la solidez estadística de los resultados individuales por clase.

Asimismo, el dataset contiene expresiones en inglés, por lo que una posible aplicación en Seguros BBVA requeriría evaluar el desempeño con solicitudes en español y con vocabulario propio del sector asegurador.

Antes de considerar una implementación empresarial, será necesario:

- Validar el desempeño sobre un conjunto de prueba representativo.
- Establecer umbrales mínimos de calidad.
- Evaluar el comportamiento de las clases minoritarias.
- Incorporar mecanismos de revisión humana.
- Garantizar los controles de seguridad, privacidad y gobierno de datos.
- Evaluar la estabilidad de la latencia y el consumo de recursos.

## 11. Reproducibilidad

El proyecto utiliza una semilla aleatoria fija (`random_state=42`) para la división estratificada de los datos.

Los tres modelos se evaluaron utilizando el mismo conjunto piloto, las mismas etiquetas candidatas y la misma plantilla de hipótesis, con el propósito de mantener condiciones experimentales comparables.

El procedimiento de obtención y preparación del dataset se encuentra documentado en Google Colab, permitiendo recuperar el archivo desde su fuente y procesarlo en el directorio temporal `/content/`.

El repositorio está organizado para contener:

- `README.md`: descripción, configuración y conclusiones.
- `requirements.txt`: dependencias del entorno.
- `notebooks/`: cuaderno de Google Colab.
- `data/`: documentación del dataset y procedimiento de obtención.
- `results/`: métricas, tablas y visualizaciones.
- `src/`: scripts de soporte para reproducir el procedimiento.

La reproducibilidad de las mediciones de rendimiento requiere considerar las versiones de las librerías, la disponibilidad de GPU y las condiciones de ejecución.

---

**Uso:** Proyecto académico de experimentación con modelos preentrenados de inteligencia artificial. No constituye una solución productiva de Seguros BBVA ni implica el tratamiento de información real de sus clientes.
