# Resultados experimentales

Esta carpeta contiene los resultados de la evaluación comparativa de tres modelos preentrenados de Hugging Face para clasificación zero-shot de solicitudes de clientes.

## Condiciones experimentales

- Dataset: Bitext Customer Support.
- Evaluación piloto: 100 solicitudes del conjunto de desarrollo.
- Número de intenciones: 27.
- Entorno: Google Colab.
- GPU: NVIDIA Tesla T4.
- Método: clasificación zero-shot sin fine-tuning.

## Resultados

| Métrica | DeBERTa-v3-small | DeBERTa-v3-base | BART-large-MNLI |
|---|---:|---:|---:|
| Accuracy | 67.00 % | 71.00 % | 80.00 % |
| Precision macro | 56.76 % | 61.04 % | 76.29 % |
| Recall macro | 58.56 % | 60.93 % | 73.77 % |
| F1-score macro | 51.99 % | 57.43 % | 71.44 % |
| Latencia media (s) | 0.4062 | 0.7379 | 0.6104 |
| Latencia mediana (s) | 0.3743 | 0.7034 | 0.5690 |

## Interpretación

BART-large-MNLI obtuvo el mayor desempeño predictivo, destacando en las cuatro métricas de clasificación.

DeBERTa-v3-small presentó la menor latencia de inferencia.

Estos resultados muestran un compromiso entre la calidad de clasificación y la eficiencia computacional.

## Limitaciones

Los resultados corresponden a una evaluación piloto de 100 solicitudes y no representan una validación definitiva sobre los 6,454 registros reservados para prueba.

Las métricas individuales de las intenciones menos frecuentes deben interpretarse con cautela debido al tamaño reducido de la muestra.

Los tiempos de inferencia pueden variar dependiendo del hardware y de las condiciones de ejecución.

## Reproducibilidad

El procedimiento de evaluación y generación de métricas se encuentra documentado en `notebooks/comparacion_modelos_nlp.ipynb`.

Los resultados se obtuvieron utilizando las mismas solicitudes y etiquetas candidatas para los tres modelos.
