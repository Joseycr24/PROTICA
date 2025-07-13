# main.py
import os
from unidecode import unidecode
from pathlib import Path
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from src.preprocesamiento import DataPreproc
from src.analisis_departamento import analizar_departamento
from src.pipeline import ejecutar_pipeline, procesar_eam_2006_2018, procesar_eam_2019_2023

# === Archivos a procesar ===
archivos_consecutivos = [
    Path("data/raw/Bolivar/Evaluaciones_Agropecuarias_Municipales_EVA_20250615.csv"),
    Path("data/raw/Bolivar/Evaluaciones_Agropecuarias_Municipales___EVA._2019_-_2023._Base_Agr_cola_20250615.csv")
]

# === Cargar y procesar archivos históricos ===
df_2006_2018 = procesar_eam_2006_2018(pd.read_csv(archivos_consecutivos[0]))
df_2019_2023 = procesar_eam_2019_2023(pd.read_csv(archivos_consecutivos[1]))
df_combinado = pd.concat([df_2006_2018, df_2019_2023], ignore_index=True)
print(f"✅ Combinado Bolívar 2006–2023: {len(df_combinado)} filas\n")

# === Preprocesamiento general ===
preproc = DataPreproc(df_combinado)
df_limpio = preproc.run_all_preprocessing()
ruta_limpio = Path("data/processed/archivo_limpio.csv")
df_limpio.to_csv(ruta_limpio, index=False)
print(f"✅ Guardado archivo limpio: {ruta_limpio.resolve()}\n")

# === Ejecutar pipeline para cada departamento ===
df = pd.read_csv(ruta_limpio, low_memory=False)
departamentos = df[['codigo_dane_departamento', 'nombre_departamento']].drop_duplicates()

for _, row in departamentos.iterrows():
    cod = int(row['codigo_dane_departamento'])
    nombre = row['nombre_departamento']
    print(f"🔄 Procesando {nombre} ({cod})...")

    try:
        ejecutar_pipeline(
            path_archivo=ruta_limpio,
            codigo_depto=cod,
            nombre_depto=nombre,
            cultivos_extra=[],
            ruta_salida="reports"
        )
        print(f"✅ Terminado: {nombre}\n")
    except Exception as e:
        print(f"❌ Error en {nombre}: {e}\n")

# === Consolidar departamentos del Caribe ===
departamentos_interes = ['bolivar', 'cordoba', 'sucre', 'san_andres_y_providencia']
base_data_path = 'reports'
columnas_relevantes = [
    'codigo_dane_departamento', 'nombre_departamento',
    'codigo_dane_municipio', 'nombre_municipio',
    'cultivo', 'anio', 'area_sembrada_ha',
    'area_cosechada_ha', 'produccion_toneladas',
    'rendimiento_ton_ha', 'ciclo_cultivo'
]

def normalizar(texto):
    return unidecode(texto).lower().replace(" ", "_").replace("-", "_")

df_total = pd.DataFrame()
for archivo in os.listdir(base_data_path):
    if archivo.endswith('.csv') and not archivo.startswith('pivot_'):
        nombre_archivo = normalizar(archivo.replace(".csv", ""))
        for depto in departamentos_interes:
            if nombre_archivo.startswith(normalizar(depto)):
                df_temp = pd.read_csv(os.path.join(base_data_path, archivo))
                df_temp = df_temp[[col for col in columnas_relevantes if col in df_temp.columns]]
                df_total = pd.concat([df_total, df_temp], ignore_index=True)
                print(f"✅ Añadido: {archivo}")
                break

ruta_total = "reports/produccion_departamentos_caribe.csv"
df_total.to_csv(ruta_total, index=False)
print(f"\n📁 Consolidado guardado: {ruta_total}")

# === Generar top 5 por departamento ===
df = df_total.copy()

# ❌ Eliminar inconsistencias
df = df[~((df['produccion_toneladas'] > 0) & (df['area_cosechada_ha'] == 0))]
df = df[~((df['produccion_toneladas'] == 0) & (df['area_cosechada_ha'] == 0))]

# 🔄 Calcular rendimiento evitando división por cero
df['rendimiento_ton_ha'] = df.apply(
    lambda row: row['produccion_toneladas'] / row['area_cosechada_ha'] if row['area_cosechada_ha'] > 0 else 0,
    axis=1
)

agrupado = df.groupby(['nombre_departamento', 'cultivo']).agg(
    produccion_total=('produccion_toneladas', 'sum'),
    area_total=('area_cosechada_ha', 'sum'),
    frecuencia=('anio', 'nunique')
).reset_index()

agrupado['rendimiento_promedio'] = agrupado['produccion_total'] / agrupado['area_total']

scaler = MinMaxScaler()
metricas = agrupado[['produccion_total', 'area_total', 'rendimiento_promedio', 'frecuencia']]
agrupado[['produccion_norm', 'area_norm', 'rendimiento_norm', 'frecuencia_norm']] = scaler.fit_transform(metricas)

agrupado['importancia_score'] = (
    agrupado['produccion_norm'] * 0.4 +
    agrupado['area_norm'] * 0.2 +
    agrupado['rendimiento_norm'] * 0.2 +
    agrupado['frecuencia_norm'] * 0.2
)

top_productos_por_depto = (
    agrupado.sort_values(['nombre_departamento', 'importancia_score'], ascending=[True, False])
    .groupby('nombre_departamento').head(5)
)

# 💾 Guardar detalle completo de cultivos top 5
cultivos_top5 = top_productos_por_depto[['nombre_departamento', 'cultivo']].drop_duplicates()
df_top5_completo = df_total.merge(cultivos_top5, on=['nombre_departamento', 'cultivo'], how='inner')
ruta_top5 = "reports/top_cultivos_detalle_caribe.csv"
df_top5_completo.to_csv(ruta_top5, index=False)
print(f"✅ Guardado detalle de top 5 en: {ruta_top5}")

# 📊 Base comparativa con todos los cultivos top en algún departamento
cultivos_comparativos = top_productos_por_depto[['cultivo']].drop_duplicates()
df_comparativo = df_total[df_total['cultivo'].isin(cultivos_comparativos['cultivo'])]
ruta_comparativo = "reports/cultivos_comparativos_caribe.csv"
df_comparativo.to_csv(ruta_comparativo, index=False)
print(f"✅ Guardado comparativo en: {ruta_comparativo}")

# === Visualizar análisis comparativo de cultivos entre departamentos ===
print("\n🔍 Análisis exploratorio de cultivos comparativos:\n")
for nombre_depto in df_comparativo['nombre_departamento'].unique():
    nombre_archivo = normalizar(nombre_depto)
    codigo = df_comparativo[df_comparativo['nombre_departamento'] == nombre_depto]['codigo_dane_departamento'].iloc[0]
    archivo_csv = f"{nombre_archivo}_{codigo}.csv"

    try:
        pivot_resultado = analizar_departamento(
            nombre_departamento=nombre_depto,
            archivo_csv=archivo_csv,
            cultivos_extra=cultivos_comparativos['cultivo'].tolist()
        )
        print(f"\n📊 {nombre_depto.title()}:\n{pivot_resultado}\n" + "-"*60 + "\n")
    except Exception as e:
        print(f"❌ Error en análisis de {nombre_depto.title()}: {e}")

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# === Leer archivo de cultivos comparativos ===
df_comparativo = pd.read_csv("reports/cultivos_comparativos_caribe.csv")

print(df_comparativo.info())


# === 1. Gráfica: Evolución temporal por cultivo ===
def graficar_evolucion(df, cultivo_objetivo="arroz"):
    df_filtrado = df[df['cultivo'].str.lower() == cultivo_objetivo.lower()]
    if df_filtrado.empty:
        print(f"⚠️ No hay datos de {cultivo_objetivo} para graficar.")
        return
    
    pivot = df_filtrado.groupby(['anio', 'nombre_departamento'])['produccion_toneladas'].sum().reset_index()
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=pivot, x='anio', y='produccion_toneladas', hue='nombre_departamento', marker='o')
    plt.title(f"Evolución de la producción de {cultivo_objetivo.title()} por departamento")
    plt.ylabel("Producción (toneladas)")
    plt.xlabel("Año")
    plt.legend(title="Departamento")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"reports/evolucion_{cultivo_objetivo.lower()}.png")
    plt.show()

# === 2. Gráfica: Barras apiladas de cultivos top ===
def graficar_cultivos_top(df):
    pivot = df.groupby(['anio', 'cultivo'])['produccion_toneladas'].sum().reset_index()
    pivot = pivot.pivot(index='anio', columns='cultivo', values='produccion_toneladas').fillna(0)
    pivot.plot(kind='bar', stacked=True, figsize=(14, 7), colormap='tab20')
    plt.title("Comparación de producción de cultivos top - Caribe")
    plt.ylabel("Producción total (toneladas)")
    plt.xlabel("Año")
    plt.tight_layout()
    plt.savefig("reports/cultivos_top_comparacion.png")
    plt.show()

# === 3. Gráfica: Heatmap de eficiencia ===
def graficar_heatmap_rendimiento(df):
    pivot = df.groupby(['nombre_departamento', 'cultivo'])['rendimiento_ton_ha'].mean().reset_index()
    pivot = pivot.pivot(index='nombre_departamento', columns='cultivo', values='rendimiento_ton_ha')
    plt.figure(figsize=(12, 7))
    sns.heatmap(pivot, annot=True, fmt=".1f", cmap="YlGnBu")
    plt.title("Rendimiento promedio (ton/ha) por cultivo y departamento")
    plt.xlabel("Cultivo")
    plt.ylabel("Departamento")
    plt.tight_layout()
    plt.savefig("reports/heatmap_rendimiento.png")
    plt.show()

# === Ejecutar las gráficas ===
graficar_evolucion(df_comparativo, cultivo_objetivo="arroz")
graficar_cultivos_top(df_comparativo)
graficar_heatmap_rendimiento(df_comparativo)
