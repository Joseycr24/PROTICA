import pandas as pd
from src.preprocesamiento import DataPreproc
from src.preprocesamiento import ExploraAnalysis
import os


def analizar_departamento(nombre_departamento, archivo_csv, cultivos_extra):
    print(f"\n=== Análisis para {nombre_departamento.upper()} ===")
    
    # 1. Cargar CSV
    path = os.path.join("reports", archivo_csv)
    df = pd.read_csv(path)

    # 2. Instanciar clases
    preproce = DataPreproc(df)
    explora = ExploraAnalysis(df)

    # 3. Ejecutar métodos
    preproce.run_all_preprocessing()
    explora.general_information()
    explora.null_data()
    explora.descript_statis()
    explora.show_duplicate_rows()
    explora.run_full_detection()

    # 4. Análisis de cultivos principales
    top = df.groupby('cultivo')['produccion_toneladas'].sum().sort_values(ascending=False).head(5)

    # Agregar cultivos extra si existen
    produccion_extra = df[df['cultivo'].isin(cultivos_extra)] \
        .groupby('cultivo')['produccion_toneladas'].sum()

    top_expandido = pd.concat([top, produccion_extra])
    top_expandido = top_expandido[~top_expandido.index.duplicated(keep='first')]
    top_expandido.sort_values(ascending=False, inplace=True)

    print("\nTop cultivos (expandido):")
    print(top_expandido)

    # 5. Crear tabla pivote por año
    cultivos_top = top_expandido.index.tolist()
    df_top = df[df['cultivo'].isin(cultivos_top)]

    produccion_por_anio = (
        df_top.groupby(['cultivo', 'anio'])['produccion_toneladas']
        .sum()
        .reset_index()
        .sort_values(by=['cultivo', 'anio'])
    )

    pivot = produccion_por_anio.pivot(index='anio', columns='cultivo', values='produccion_toneladas')
    pivot_ordenado = pivot[cultivos_top].fillna(0)

    return pivot_ordenado

