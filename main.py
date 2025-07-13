import pandas as pd
from pathlib import Path
from src.preprocesamiento import DataPreproc
from src.analisis_departamento import analizar_departamento 
from src.pipeline import ejecutar_pipeline, procesar_eam_2006_2018, procesar_eam_2019_2023



# ===Archivos históricos que se deben unir ===
archivos_consecutivos = [
    Path("DATOS\Evaluaciones_Agropecuarias_Municipales_EVA_20250615.csv"),# type: ignore
    Path("DATOS\Evaluaciones_Agropecuarias_Municipales___EVA._2019_-_2023._Base_Agr_cola_20250615.csv") # type: ignore
]

# ===Cargar y procesar individualmente ===
archivo_2006_2018 = archivos_consecutivos[0]
archivo_2019_2023 = archivos_consecutivos[1]

df_2006_2018 = pd.read_csv(archivo_2006_2018)
df_2006_2018 = procesar_eam_2006_2018(df_2006_2018)
print(f"📄 Cargado y procesado: {archivo_2006_2018.name}")

df_2019_2023 = pd.read_csv(archivo_2019_2023)
df_2019_2023 = procesar_eam_2019_2023(df_2019_2023)
print(f"📄 Cargado y procesado: {archivo_2019_2023.name}")

# ===Unir los archivos procesados ===
df_combinado = pd.concat([df_2006_2018, df_2019_2023], ignore_index=True)
print(f"✅ Archivos combinados (Bolívar 2006–2023): {len(df_combinado)} filas\n")

# ===Preprocesamiento general ===
preproc = DataPreproc(df_combinado)
df_limpio = preproc.run_all_preprocessing()

ruta_limpio = Path("data/procesado/archivo_limpio.csv")
df_limpio.to_csv(ruta_limpio, index=False)
print(f"✅ Archivo limpio guardado en: {ruta_limpio.resolve()}\n")

# ===Ejecutar pipeline por departamento ===
df = pd.read_csv(ruta_limpio, low_memory=False)
print("🧪 Columnas reales del archivo limpio:", df.columns.tolist())

#Asegúrate de que los nombres de columnas coincidan con los del archivo limpio
departamentos = df[['codigo_dane_departamento', 'nombre_departamento']].drop_duplicates()

#Cultivos adicionales
cultivos_extra = ['coco', 'patilla', 'platano', 'batata']

for _, row in departamentos.iterrows():
    cod = int(row['codigo_dane_departamento'])
    nombre = row['nombre_departamento']
    print(f"🔄 Procesando departamento: {nombre} ({cod})...")

    try:
        ejecutar_pipeline(
            path_archivo=ruta_limpio,
            codigo_depto=cod,
            nombre_depto=nombre,
            cultivos_extra=cultivos_extra,
            ruta_salida="reports"
        )
        print(f"✅ Terminado: {nombre}\n")
    except Exception as e:
        print(f"❌ Error procesando {nombre}: {e}\n")

# ===Procesar otro archivo individual ===
archivo_extra = Path("data/raw/Cesar/EVA_Cesar_20250615.csv")
if archivo_extra.exists():
    df_extra = pd.read_csv(archivo_extra)
    preproc_extra = DataPreproc(df_extra)
    df_extra_limpio = preproc_extra.run_all_preprocessing()
    ruta_extra = Path("data/processed/limpio_Cesar.csv")
    df_extra_limpio.to_csv(ruta_extra, index=False)

    ejecutar_pipeline(
        path_archivo=ruta_extra,
        codigo_depto=200,  
        nombre_depto="Cesar",
        cultivos_extra=cultivos_extra,
        ruta_salida="reports"
    )
    print("✅ Cesar procesado.")
else:
    print("⚠️ Archivo de Cesar no encontrado.")

# ===Análisis exploratorio por departamento ===
# Diccionario con cultivos extra por departamento
cultivos_por_departamento = {
    "san andres y providencia": ['name', 'maiz', 'arroz', 'palma de aceite'],
    "bolivar": ['coco', 'patilla', 'platano', 'batata'],
    "cordoba": ['coco', 'batata', 'palma de aceite', 'patilla'],
    "sucre": ['coco', 'platano', 'batata', 'palma de aceite']
}

print("\n🔍 INICIANDO ANÁLISIS EXPLORATORIO DEPARTAMENTAL\n")

for nombre_depto, cultivos_extra in cultivos_por_departamento.items():
    #Convertir nombre para archivo
    nombre_archivo = nombre_depto.lower().replace(" ", "_")
    
    #Obtener código DANE desde el DataFrame original
    try:
        codigo = df[df['nombre_departamento'].str.lower() == nombre_depto.lower()]['codigo_dane_departamento'].iloc[0]
    except IndexError:
        print(f"❌ No se encontró código DANE para: {nombre_depto}")
        continue

    archivo_csv = f"{nombre_archivo}_{codigo}.csv"

    try:
        pivot_resultado = analizar_departamento(
            nombre_departamento=nombre_depto,
            archivo_csv=archivo_csv,
            cultivos_extra=cultivos_extra
        )
        
        print(f"\n📊 Pivot de producción por año para {nombre_depto.title()}:\n")
        print(pivot_resultado)
        print("\n" + "-"*60 + "\n")

    except Exception as e:
        print(f"❌ Error procesando análisis para {nombre_depto.title()}: {e}")
