import pandas as pd

# Función que ejecuta el pipeline completo por departamento
def ejecutar_pipeline(path_archivo, codigo_depto, nombre_depto, cultivos_extra=[], ruta_salida="reports"):
    # Carga el archivo limpio
    df = pd.read_csv(path_archivo)

    # Filtra por departamento
    df_depto = df[df['codigo_dane_departamento'] == codigo_depto]

    # Aquí podrías llamar a otras funciones de análisis, gráficas, etc.
    # Por ahora, simplemente guarda el archivo del departamento
    nombre_archivo = f"{ruta_salida}/{nombre_depto.replace(' ', '_')}_{codigo_depto}.csv"
    df_depto.to_csv(nombre_archivo, index=False)
    print(f"📁 Archivo generado: {nombre_archivo}")

# Función para transformar archivo 2006–2018
def procesar_eam_2006_2018(df):
    df['Rendimiento\n(t/ha)'] = df['Producción\n(t)'] / df['Área Cosechada\n(ha)']
    df['Rendimiento\n(t/ha)'] = df['Rendimiento\n(t/ha)'].fillna(0)

    mapeo = {
        'CÓD. \nDEP.': 'Codigo Dane departamento',
        'DEPARTAMENTO': 'Nombre Departamento',
        'CÓD. MUN.': 'Codigo Dane municipio',
        'MUNICIPIO': 'Nombre Municipio',
        'GRUPO \nDE CULTIVO': 'Grupo cultivo',
        'SUBGRUPO \nDE CULTIVO': 'Subgrupo Cultivo',
        'CULTIVO': 'Cultivo',
        'DESAGREGACIÓN REGIONAL Y/O SISTEMA PRODUCTIVO': 'Sistema productivo o region',
        'AÑO': 'Anio',
        'PERIODO': 'Periodo',
        'Área Sembrada\n(ha)': 'Area Sembrada ha',
        'Área Cosechada\n(ha)': 'Area Cosechada ha',
        'Producción\n(t)': 'Produccion Toneladas',
        'Rendimiento\n(t/ha)': 'Rendimiento ton ha',
        'ESTADO FISICO PRODUCCION': 'Estado fisico',
        'NOMBRE \nCIENTIFICO': 'Nombre cientifico',
        'CICLO DE CULTIVO': 'Ciclo cultivo'
    }

    df.rename(columns=mapeo, inplace=True)
    return df

# Función para transformar archivo 2019–2023
def procesar_eam_2019_2023(df):
    if 'Código del cultivo' in df.columns:
        df.drop(['Código del cultivo'], axis=1, inplace=True)

    mapeo = {
        'Código Dane departamento': 'Codigo Dane departamento',
        'Departamento': 'Nombre Departamento',
        'Código Dane municipio': 'Codigo Dane municipio',
        'Municipio': 'Nombre Municipio',
        'Grupo cultivo': 'Grupo cultivo',
        'Subgrupo': 'Subgrupo Cultivo',
        'Cultivo': 'Cultivo',
        'Desagregación cultivo': 'Sistema productivo o region',
        'Año': 'Anio',
        'Periodo': 'Periodo',
        'Área sembrada': 'Area Sembrada ha',
        'Área cosechada': 'Area Cosechada ha',
        'Producción': 'Produccion Toneladas',
        'Rendimiento': 'Rendimiento ton ha',
        'Ciclo del cultivo': 'Ciclo cultivo',
        'Estado físico del cultivo': 'Estado fisico',
        'Nombre científico del cultivo': 'Nombre cientifico'
    }

    df.rename(columns=mapeo, inplace=True)
    return df
