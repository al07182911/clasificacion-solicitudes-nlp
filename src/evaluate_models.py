
# ==========================================================
# PROYECTO: CLASIFICACION DE SOLICITUDES MEDIANTE NLP
# SCRIPT: evaluate_models.py
# Evaluacion de metricas de clasificacion y latencia
# ==========================================================

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)


def evaluar_modelo(ruta_csv, salida="results", nombre_modelo="modelo"):
    """
    Evalua predicciones generadas por run_inference.py.

    Columnas requeridas:
        intent: etiqueta real
        prediccion: etiqueta predicha
        latencia_s: tiempo de inferencia en segundos
    """

    # 1. Cargar resultados
    df = pd.read_csv(ruta_csv)

    columnas = {"intent", "prediccion", "latencia_s"}

    if not columnas.issubset(df.columns):
        raise ValueError(
            "El archivo debe contener: "
            "intent, prediccion y latencia_s."
        )

    if df.empty:
        raise ValueError("El archivo de resultados esta vacio.")

    if df[list(columnas)].isnull().any().any():
        raise ValueError(
            "Existen valores nulos en los resultados."
        )

    # 2. Preparar etiquetas
    y_real = df["intent"]
    y_pred = df["prediccion"]

    etiquetas = sorted(
        set(y_real).union(set(y_pred))
    )

    # 3. Calcular metricas generales
    metricas = {
        "Modelo": nombre_modelo,
        "Registros": len(df),
        "Accuracy": accuracy_score(y_real, y_pred),
        "Precision_macro": precision_score(
            y_real, y_pred,
            labels=etiquetas,
            average="macro",
            zero_division=0
        ),
        "Recall_macro": recall_score(
            y_real, y_pred,
            labels=etiquetas,
            average="macro",
            zero_division=0
        ),
        "F1_macro": f1_score(
            y_real, y_pred,
            labels=etiquetas,
            average="macro",
            zero_division=0
        ),
        "Latencia_media_s": df["latencia_s"].mean(),
        "Latencia_mediana_s": df["latencia_s"].median()
    }

    # 4. Reporte detallado por intencion
    reporte = classification_report(
        y_real,
        y_pred,
        labels=etiquetas,
        output_dict=True,
        zero_division=0
    )

    df_reporte = pd.DataFrame(reporte).T

    # 5. Crear directorio de salida
    carpeta = Path(salida)
    carpeta.mkdir(parents=True, exist_ok=True)

    # 6. Guardar metricas generales
    df_metricas = pd.DataFrame([metricas])

    archivo_metricas = (
        carpeta / f"metricas_{nombre_modelo}.csv"
    )

    df_metricas.to_csv(
        archivo_metricas,
        index=False
    )

    # 7. Guardar metricas por intencion
    archivo_reporte = (
        carpeta / f"reporte_{nombre_modelo}.csv"
    )

    df_reporte.to_csv(archivo_reporte)

    # 8. Mostrar resultados
    print("=" * 55)
    print("EVALUACION DEL MODELO:", nombre_modelo)
    print("=" * 55)

    for clave, valor in metricas.items():
        if isinstance(valor, (int, float, np.number)):
            print(f"{clave}: {valor:.4f}")
        else:
            print(f"{clave}: {valor}")

    print("\nArchivos generados:")
    print(archivo_metricas)
    print(archivo_reporte)

    return df_metricas, df_reporte


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Evaluacion de modelos NLP"
    )

    parser.add_argument(
        "--input",
        required=True,
        help="CSV con predicciones"
    )

    parser.add_argument(
        "--model",
        default="modelo",
        help="Nombre del modelo evaluado"
    )

    parser.add_argument(
        "--output",
        default="results",
        help="Directorio de resultados"
    )

    args = parser.parse_args()

    evaluar_modelo(
        ruta_csv=args.input,
        salida=args.output,
        nombre_modelo=args.model
    )
