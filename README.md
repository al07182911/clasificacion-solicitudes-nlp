# Clasificación de solicitudes de clientes mediante NLP

## 1. Descripción del proyecto

Este proyecto forma parte de una actividad académica del Máster en Inteligencia Artificial y tiene como objetivo comparar tres modelos preentrenados de Hugging Face para la clasificación automatizada de solicitudes de clientes mediante técnicas de Procesamiento de Lenguaje Natural (NLP).

El caso de uso está inspirado en los procesos de atención y cancelación de pólizas del sector asegurador, tomando como referencia el contexto de Seguros BBVA.

El prototipo utiliza clasificación *zero-shot* para identificar la intención expresada por un cliente sin necesidad de realizar entrenamiento adicional de los modelos.

## 2. Objetivo

Evaluar comparativamente tres modelos preentrenados de Hugging Face, utilizando métricas de calidad predictiva y eficiencia computacional, para identificar la alternativa más adecuada para clasificar solicitudes de clientes.

## 3. Dataset

Se utiliza el dataset público **Bitext Customer Support – Training Dataset for Chatbots/Virtual Assistants**, disponible en Kaggle.

Fuente: https://www.kaggle.com/datasets/bitext/training-dataset-for-chatbotsvirtual-assistants

Características:

- Registros originales: 21,534.
- Registros después de la limpieza: 21,513.
- Categorías generales: 11.
- Intenciones específicas: 27.
- Variables: `flags`, `utterance`, `category` e `intent`.
- División estratificada: 70 % desarrollo y 30 % prueba.
- Semilla aleatoria: 42.

Por razones de seguridad de la información y protección de datos personales, no se utilizaron datos reales de clientes de Seguros BBVA.

## 4. Modelos evaluados

Se implementaron los siguientes modelos:

1. **DeBERTa-v3-small:** `cross-encoder/nli-deberta-v3-small`
2. **DeBERTa-v3-base:** `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`
3. **BART-large-MNLI:** `facebook/bart-large-mnli`

Los tres modelos se evaluaron mediante pipelines de clasificación *zero-shot*, utilizando las mismas 27 intenciones candidatas y una plantilla de hipótesis común.

## 5. Entorno de ejecución

- Plataforma: Google Colab.
- Lenguaje: Python.
- GPU: NVIDIA Tesla T4.
- Librerías principales: PyTorch, Transformers, Hugging Face Hub, pandas, NumPy, scikit-learn y Matplotlib.
- Método de inferencia: zero-shot classification.

Las versiones de las dependencias utilizadas deberán documentarse en `requirements.txt`.

## 6. Procedimiento de ejecución

1. Abrir el notebook del proyecto en Google Colab.
2. Activar la GPU desde la configuración del entorno de ejecución.
3. Instalar las dependencias necesarias.
4. Configurar el token `HF_TOKEN` mediante los secretos de Colab.
5. Ejecutar las celdas para obtener y preparar el dataset.
6. Realizar la limpieza y división estratificada de los datos.
7. Cargar los tres modelos preentrenados de Hugging Face.
8. Ejecutar los pipelines de inferencia.
9. Calcular las métricas de clasificación y latencia.
10. Consultar las tablas, gráficos y conclusiones del experimento.

## 7. Resultados experimentales

La evaluación piloto se realizó con 100 solicitudes del conjunto de desarrollo.

| Métrica | DeBERTa-small | DeBERTa-base | BART-large |
|---|---:|---:|---:|
| Accuracy | 67.00 % | 71.00 % | 80.00 % |
| Precision macro | 56.76 % | 61.04 % | 76.29 % |
| Recall macro | 58.56 % | 60.93 % | 73.77 % |
| F1-score macro | 51.99 % | 57.43 % | 71.44 % |
| Latencia media (s) | 0.4062 | 0.7379 | 0.6104 |

Estos resultados corresponden a una evaluación piloto y no representan una validación definitiva sobre el conjunto de prueba reservado.

## 8. Conclusiones

BART-large-MNLI obtuvo el mejor desempeño predictivo, alcanzando un accuracy del 80 % y un F1-score macro del 71.44 %.

DeBERTa-v3-small presentó la menor latencia media, con 0.4062 segundos por solicitud.

Los resultados muestran que existe un compromiso entre calidad predictiva y eficiencia computacional, por lo que la selección del modelo depende de las necesidades específicas del caso de uso.

Con base en la evaluación piloto, se recomienda BART-large-MNLI como principal candidato para continuar la validación del prototipo.

## 9. Limitaciones y riesgos

El dataset contiene solicitudes generales de atención al cliente y no representa directamente los motivos de cancelación de seguros.

La evaluación piloto comprende únicamente 100 solicitudes y algunas intenciones presentan pocos ejemplos.

Antes de considerar una implementación empresarial, será necesario validar el desempeño con datos representativos del sector asegurador, establecer umbrales de aceptación y aplicar controles de seguridad, privacidad y gobierno de datos.

## 10. Reproducibilidad

El proyecto utiliza una semilla fija para la división de datos y mantiene las mismas etiquetas candidatas y condiciones de inferencia para los tres modelos.

El repositorio incorporará el notebook, los scripts de soporte, las dependencias y los resultados experimentales necesarios para reproducir el procedimiento.

**Uso:** proyecto académico de experimentación con modelos preentrenados. No constituye una solución productiva de Seguros BBVA.
