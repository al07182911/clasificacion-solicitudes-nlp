
# ==========================================================
# PROYECTO: CLASIFICACION DE SOLICITUDES MEDIANTE NLP
# SCRIPT: prepare_data.py
# DESCRIPCION: Limpieza y division del dataset Bitext
# ==========================================================

import pandas as pd
from sklearn.model_selection import train_test_split


def preparar_datos(ruta_csv, test_size=0.30, semilla=42):
    """
    Carga y prepara el dataset Bitext.

    Parametros:
        ruta_csv: Ruta del archivo CSV original.
        test_size: Proporcion del conjunto de prueba.
        semilla: Semilla para reproducibilidad.

    Retorna:
        df_train: Conjunto de desarrollo.
        df_test: Conjunto de prueba.
    """

    # 1. Cargar dataset
    df = pd.read_csv(ruta_csv)

    print("Registros originales:", len(df))

    # 2. Validar columnas
    columnas = [
        "flags",
        "utterance",
        "category",
        "intent"
    ]

    for columna in columnas:
        if columna not in df.columns:
            raise ValueError(
                f"Falta la columna: {columna}"
            )

    # 3. Validar valores nulos
    if df[columnas].isnull().any().any():
        raise ValueError(
            "El dataset contiene valores nulos."
        )

    # 4. Eliminar duplicados completos
    df_limpio = df.drop_duplicates().copy()

    # 5. Comprobar etiquetas contradictorias
    conflictos = (
        df_limpio.groupby("utterance")["intent"]
        .nunique()
    )

    if (conflictos > 1).any():
        raise ValueError(
            "Existen textos con intenciones diferentes."
        )

    # 6. Eliminar textos repetidos
    df_modelo = (
        df_limpio
        .drop_duplicates(subset=["utterance"])
        .reset_index(drop=True)
    )

    # 7. Division estratificada
    df_train, df_test = train_test_split(
        df_modelo,
        test_size=test_size,
        random_state=semilla,
        stratify=df_modelo["intent"]
    )

    df_train = df_train.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)

    # 8. Validar ausencia de textos compartidos
    comunes = (
        set(df_train["utterance"])
        & set(df_test["utterance"])
    )

    assert len(comunes) == 0

    # 9. Mostrar resumen
    print("\nPREPARACION COMPLETADA")
    print("-" * 45)
    print("Registros limpios:", len(df_modelo))
    print("Desarrollo:", len(df_train))
    print("Prueba:", len(df_test))
    print("Intenciones:", df_modelo["intent"].nunique())
    print("Textos compartidos:", len(comunes))

    return df_train, df_test


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Preparacion del dataset Bitext"
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Ruta del archivo CSV original"
    )

    parser.add_argument(
        "--output",
        default="data",
        help="Directorio para guardar los resultados"
    )

    args = parser.parse_args()

    from pathlib import Path

    salida = Path(args.output)
    salida.mkdir(parents=True, exist_ok=True)

    train, test = preparar_datos(args.input)

    train.to_csv(salida / "train.csv", index=False)
    test.to_csv(salida / "test.csv", index=False)

    print("\nArchivos generados correctamente.")

