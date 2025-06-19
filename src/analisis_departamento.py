import os
import pandas as pd
from src.utils_plot import guardar_pivot, graficar_pivot  
from src.preprocesamiento import DataPreproc, ExploraAnalysis

def analizar_departamento(nombre_departamento, archivo_csv, cultivos_extra):
    print(f"\n=== Análisis para {nombre_departamento.upper()} ===")
    
    #Cargar CSV
    path = os.path.join("reports", archivo_csv)
    df = pd.read_csv(path)

    #Instanciar clases y ejecutar preprocesamiento
    preproce = DataPreproc(df)
    df = preproce.run_all_preprocessing()  
    explora = ExploraAnalysis(df)

    #Análisis exploratorio
    explora.general_information()
    explora.null_data()
    explora.descript_statis()
    explora.show_duplicate_rows()
    explora.run_full_detection()

    #Análisis de cultivos principales
    top = df.groupby('cultivo')['produccion_toneladas'].sum().sort_values(ascending=False).head(5)

    #Agregar cultivos extra si existen
    produccion_extra = df[df['cultivo'].isin(cultivos_extra)] \
        .groupby('cultivo')['produccion_toneladas'].sum()

    top_expandido = pd.concat([top, produccion_extra])
    top_expandido = top_expandido[~top_expandido.index.duplicated(keep='first')]
    top_expandido.sort_values(ascending=False, inplace=True)

    print("\nTop cultivos (expandido):")
    print(top_expandido)

    #tabla pivote por año
    cultivos_top = top_expandido.index.tolist()
    df_top = df[df['cultivo'].isin(cultivos_top)]

    produccion_por_anio = (
        df_top.groupby(['cultivo', 'anio'])['produccion_toneladas']
        .sum()
        .reset_index()
        .sort_values(by=['cultivo', 'anio'])
    )

    pivot = produccion_por_anio.pivot(index='anio', columns='cultivo', values='produccion_toneladas')

    #Convertir el índice a entero antes de continuar
    pivot.index = pivot.index.astype(int)

    pivot_ordenado = pivot[cultivos_top].fillna(0)

    #Guardar y graficar el pivot
    nombre_pivot = f"pivot_{nombre_departamento.lower()}.csv"
    guardar_pivot(pivot_ordenado, nombre_pivot)

    #Graficar y guardar imagen también
    graficar_pivot(
        pivot_ordenado,
        f"Producción de cultivos en {nombre_departamento}",
        guardar_imagen=True,
        nombre_archivo=f"grafico_{nombre_departamento.lower()}.png"
    )
    
    return pivot_ordenado
