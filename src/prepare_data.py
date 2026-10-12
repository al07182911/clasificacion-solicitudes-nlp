
# ==========================================================
# PROYECTO: CLASIFICACION DE SOLICITUDES MEDIANTE NLP
# SCRIPT: prepare_data.py
#
# Descarga automatica del dataset Bitext desde Kaggle
# Limpieza, validacion y division estratificada 70/30
# ==========================================================

import argparse
import shutil
from pathlib import Path

import kagglehub
import pandas as pd
from sklearn.model_selection import train_test_split


DATASET_KAGGLE = (
    "bitext/training-dataset-for-chatbotsvirtual-assistants"
)

COLUMNAS = ["flags", "utterance", "category", "intent"]


def descargar_dataset(destino):
    """
    Descarga el dataset Bitext desde Kaggle y copia
    sus archivos al directorio indicado.
    """

    destino = Path(destino)
    destino.mkdir(parents=True, exist_ok=True)

    print("=" * 55)
    print("DESCARGA AUTOMATICA DEL DATASET")
    print("=" * 55)

    ruta_descarga = Path(
        kagglehub.dataset_download(DATASET_KAGGLE)
    )

    print("Origen:", ruta_descarga)

    # Copiar archivos desde la cache de Kaggle
    shutil.copytree(
        ruta_descarga,
        destino,
        dirs_exist_ok=True
    )

    print("Destino:", destino)

    return destino


def localizar_dataset(carpeta):
    """
    Localiza el CSV correspondiente al dataset
    de 20,000+ expresiones de chatbots.
    """

    carpeta = Path(carpeta)

    candidatos = []

    for archivo in carpeta.rglob("*.csv"):

        try:
            df_temp = pd.read_csv(archivo)

            if (
                set(COLUMNAS).issubset(df_temp.columns)
                and len(df_temp) == 21534
            ):
                candidatos.append(archivo)

        except Exception:
            continue

    if len(candidatos) != 1:
        raise ValueError(
            "No se pudo identificar un unico dataset "
            "Bitext de 21,534 registros. "
            "Verifica la version descargada."
        )

    print("\nDataset identificado:")
    print(candidatos[0])

    return candidatos[0]


def preparar_datos(ruta_csv, test_size=0.30, semilla=42):
    """
    Carga, limpia y divide el dataset Bitext.
    """

    # Cargar datos
    df = pd.read_csv(ruta_csv)

    print("\nREGISTROS ORIGINALES:", len(df))

    # Validar columnas
    faltantes = set(COLUMNAS) - set(df.columns)

    if faltantes:
        raise ValueError(
            f"Columnas faltantes: {faltantes}"
        )

    # Validar nulos
    if df[COLUMNAS].isnull().any().any():
        raise ValueError(
            "El dataset contiene valores nulos."
        )

    # Eliminar duplicados completos
    df_limpio = df.drop_duplicates().copy()

    # Verificar textos con intenciones contradictorias
    conflictos = (
        df_limpio.groupby("utterance")["intent"]
        .nunique()
    )

    if (conflictos > 1).any():
        raise ValueError(
            "Existen textos con intenciones diferentes."
        )

    # Eliminar textos repetidos
    df_modelo = (
        df_limpio
        .drop_duplicates(subset=["utterance"])
        .reset_index(drop=True)
    )

    # Division estratificada 70/30
    df_train, df_test = train_test_split(
        df_modelo,
        test_size=test_size,
        random_state=semilla,
        stratify=df_modelo["intent"]
    )

    df_train = df_train.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)

    # Validar que no existan textos compartidos
    comunes = (
        set(df_train["utterance"])
        & set(df_test["utterance"])
    )

    assert len(comunes) == 0

    print("\nRESULTADOS DE PREPARACION")
    print("-" * 55)

    print("Registros limpios:", len(df_modelo))
    print("Entrenamiento:", len(df_train))
    print("Prueba:", len(df_test))
    print("Intenciones:", df_modelo["intent"].nunique())
    print("Textos compartidos:", len(comunes))

    return df_train, df_test


def main():

    parser = argparse.ArgumentParser(
        description="Preparacion automatica del dataset Bitext"
    )

    parser.add_argument(
        "--output",
        default="/content/Actividad4JCP",
        help="Directorio de salida"
    )

    args = parser.parse_args()

    destino = Path(args.output)

    # 1. Descargar dataset
    carpeta = descargar_dataset(destino / "original")

    # 2. Localizar CSV
    ruta_csv = localizar_dataset(carpeta)

    # 3. Limpiar y dividir
    df_train, df_test = preparar_datos(ruta_csv)

    # 4. Guardar conjuntos preparados
    carpeta_procesados = destino / "processed"

    carpeta_procesados.mkdir(
        parents=True,
        exist_ok=True
    )

    df_train.to_csv(
        carpeta_procesados / "train.csv",
        index=False
    )

    df_test.to_csv(
        carpeta_procesados / "test.csv",
        index=False
    )

    print("\nARCHIVOS GENERADOS")
    print(carpeta_procesados / "train.csv")
    print(carpeta_procesados / "test.csv")

    print("\nProceso completado correctamente.")


if __name__ == "__main__":
    main()
