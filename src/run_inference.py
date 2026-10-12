
# ==========================================================
# PROYECTO: CLASIFICACION DE SOLICITUDES MEDIANTE NLP
# SCRIPT: run_inference.py
# Clasificacion zero-shot con Hugging Face
# ==========================================================

import argparse
import time

import pandas as pd
import torch
from transformers import pipeline


MODELOS = {
    "small": "cross-encoder/nli-deberta-v3-small",
    "base": "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
    "bart": "facebook/bart-large-mnli"
}

HIPOTESIS = "This customer request is about {}."


def ejecutar_inferencia(
    ruta_csv,
    modelo="bart",
    salida="results/predicciones.csv"
):
    """
    Clasifica solicitudes utilizando un modelo zero-shot.

    El CSV debe contener las columnas utterance e intent.
    """

    if modelo not in MODELOS:
        raise ValueError(
            f"Modelo no valido: {modelo}"
        )

    # Cargar datos
    df = pd.read_csv(ruta_csv)

    columnas = {"utterance", "intent"}

    if not columnas.issubset(df.columns):
        raise ValueError(
            "El CSV debe contener utterance e intent."
        )

    # Preparar etiquetas
    etiquetas = sorted(df["intent"].unique().tolist())

    mapa = {
        etiqueta.replace("_", " "): etiqueta
        for etiqueta in etiquetas
    }

    etiquetas_legibles = list(mapa.keys())

    # Cargar modelo
    dispositivo = 0 if torch.cuda.is_available() else -1

    print("Modelo:", MODELOS[modelo])
    print("Dispositivo:", "GPU" if dispositivo == 0 else "CPU")

    clasificador = pipeline(
        task="zero-shot-classification",
        model=MODELOS[modelo],
        device=dispositivo
    )

    predicciones = []
    latencias = []

    # Inferencia secuencial
    for i, texto in enumerate(df["utterance"]):

        if dispositivo == 0:
            torch.cuda.synchronize()

        inicio = time.perf_counter()

        resultado = clasificador(
            texto,
            candidate_labels=etiquetas_legibles,
            hypothesis_template=HIPOTESIS,
            multi_label=False
        )

        if dispositivo == 0:
            torch.cuda.synchronize()

        latencias.append(
            time.perf_counter() - inicio
        )

        predicciones.append(
            mapa[resultado["labels"][0]]
        )

        if (i + 1) % 20 == 0:
            print(f"Procesados: {i + 1}/{len(df)}")

    # Guardar resultados
    df["prediccion"] = predicciones
    df["latencia_s"] = latencias
    df["acierto"] = df["intent"] == df["prediccion"]

    from pathlib import Path

    ruta_salida = Path(salida)
    ruta_salida.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(ruta_salida, index=False)

    print("\nInferencia completada.")
    print("Archivo:", ruta_salida)

    return df


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Inferencia zero-shot NLP"
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Ruta del CSV a clasificar"
    )

    parser.add_argument(
        "--model",
        choices=["small", "base", "bart"],
        default="bart"
    )

    parser.add_argument(
        "--output",
        default="results/predicciones.csv"
    )

    args = parser.parse_args()

    ejecutar_inferencia(
        ruta_csv=args.input,
        modelo=args.model,
        salida=args.output
    )
