# %% 
# 1 Importar librerías
import os
import numpy as np
import pandas as pd
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
# %% 
# 2. Configuración de ruta
base_data_path = '../data/raw/presencia internacional/Expo'
start_year, end_year = 2005, 2025
meses = ['Enero','Febrero','Marzo','Abril','Mayo','Junio',
         'Julio','Agosto','Septiembre','Octubre','Noviembre','Diciembre']

# %% 
# 3. Renombres históricos por año
YEAR_SPECIFIC_RENAMES = {
    '2005': {'aduana':'adua','pais4':'pais','codpai4':'cod_pai4',
             'codsal3':'cod_sal1','odsal3':'cod_sal','deptoproced':'dpto2',
             'via4':'via','bandera4':'bandera','regimen4':'regim','modad4':'modad',
             'forpa4':'finalid','cerori3':'cer_ori1','sisesp4':'sisesp','posara':'posar',
             'dptor4':'dpto1','unidad4':'unid','coduni4':'coduni2','cantida4':'canti',
             'pbk4':'pbk','pnk4':'pnk','fobdol4':'fobdol','fobpes4':'fobpes','agrena4':'agrena',
             'fletes4':'fletes','seguro4':'seguro','otrosg4':'otrosg'},
    '2006': {'aduana':'adua','pais4':'pais','codpai4':'cod_pai4',
             'codsal':'cod_sal1','codsal3':'cod_sal','depto_proced':'dpto2',
             'via4':'via','bandera4':'bandera','regimen4':'regim','modad4':'modad',
             'forpa4':'finalid','cerori3':'cer_ori1','sisesp4':'sisesp','posara4':'posar',
             'dtor4':'dpto1','unidad4':'unid','coduni4':'coduni2','cantida4':'canti',
             'pbk4':'pbk','pnk4':'pnk','fobdol4':'fobdol','fobpes4':'fobpes','agrena4':'agrena',
             'fletes4':'fletes','seguro4':'seguro','otrosg4':'otrosg'},
    '2007': {'aduana':'adua','pais4':'pais','codpai4':'cod_pai4',
             'codsal':'cod_sal1','codsal3':'cod_sal','depto_proced':'dpto2',
             'via4':'via','bandera4':'bandera','regimen4':'regim','modad4':'modad',
             'forpa4':'finalid','cerori3':'cer_ori1','sisesp4':'sisesp','posara4':'posar',
             'dtor4':'dpto1','unidad4':'unid','coduni4':'coduni2','cantida4':'canti',
             'pbk4':'pbk','pnk4':'pnk','fobdol4':'fobdol','fobpes4':'fobpes','agrena4':'agrena',
             'fletes4':'fletes','seguro4':'seguro','otrosg4':'otrosg'},
}

# %%  
# 4. Códigos oficiales a conservar
CODIGOS_OFICIALES = [
    '1006400000','1213000000','0801111000','1005100000',
    '0714300000','1511100000','0807200000','0807110000',
    '0804300000','0803901100','0714100000'
]

# %%  
# 5. Leer, renombrar y unir CSVs
all_dfs = []
for anio in range(start_year, end_year+1):
    path_anio = os.path.join(base_data_path, f'Expo_{anio}')
    yearly = []
    for mes in meses:
        fp = os.path.join(path_anio, mes, f'{mes}.csv')
        if not os.path.exists(fp): continue
        for sep, enc in [(',', 'utf-8-sig'),(',', 'latin1'),(';','utf-8-sig'),(';','latin1')]:
            try:
                df = pd.read_csv(fp, sep=sep, encoding=enc, dtype=str, low_memory=False)
                if df.shape[1] > 5:
                    df.columns = [c.strip().lower().replace(' ','_').replace('-','_') for c in df.columns]
                    if str(anio) in YEAR_SPECIFIC_RENAMES:
                        df = df.rename(columns=YEAR_SPECIFIC_RENAMES[str(anio)])
                    yearly.append(df)
                    break
            except: continue
    if yearly:
        all_dfs.append(pd.concat(yearly, ignore_index=True))

# Concatenar todo
result = pd.concat(all_dfs, ignore_index=True)
print(f"✅ Unión completa: {result.shape}")

#%%
pre_nan = result[['agrena','fletes','seguro','otrosg']].isna().sum()
pre_nan

# %% 
# 6 Limpieza y normalización
# 6.1 Diagnóstico y reemplazo de NaN

print(result.isna().sum())
result.dropna(subset=['fech'], inplace=True)
result[['cod_sal', 'raz_sial', 'nit']] = result[['cod_sal', 'raz_sial', 'nit']].fillna(' ')

# Reglas para cod_pai4:
paises_xcf = ['948', '930', '907', '134', '02'] 
result.loc[result['cod_pai4'].isna() & result['pais'].isin(paises_xcf), 'cod_pai4'] = 'XCF'
paises_desconocidos  = ['964', '08']
result.loc[result['cod_pai4'].isna() & result['pais'].isin(paises_desconocidos), 'cod_pai4'] = 'DESCONOCIDO'
result['cod_pai4'] = result['cod_pai4'].replace({None: np.nan, '': np.nan})
result['pais'] = result['pais'].astype(str).str.strip()
result.loc[(result['pais'] == '688') & (result['cod_pai4'].isna()), 'cod_pai4'] = 'SCG_SRB'
result.loc[(result['pais'] == '47') & (result['cod_pai4'].isna()), 'cod_pai4'] = 'ANT'


# %% 
# 6.2 Diagnóstico de formatos problemáticos
print("🔎 Formatos problemáticos antes de limpiar:\n")
for col in ['agrena','fletes','seguro','otrosg']:
    raw = result[col].astype(str).str.strip().fillna('')
    # aplicamos nuestra limpieza “ligera” para testear
    tmp = ( raw
          .str.replace('^,', '0,',   regex=True)
          .str.replace('.',  '',     regex=False)
          .str.replace(',',  '.',    regex=False)
    )
    mask = pd.to_numeric(tmp, errors='coerce').isna() & raw.ne('')
    bad = raw[mask].unique()
    print(f"⚠️ `{col}` → {len(bad)} formatos extraños:", bad[:10])

# %% 
# 6.3 Limpieza robusta de montos

def clean_amount(series: pd.Series, fill_zero: bool = True) -> pd.Series:
    """
    - Extrae únicamente dígitos, puntos y comas.
    - Elimina puntos de miles (los que preceden a 3 dígitos).
    - Reemplaza coma decimal por punto.
    - Si fill_zero=True, cadenas vacías pasan a '0'.
    - Convierte finalmente a float.
    """
    s = series.fillna('').astype(str).str.strip()
    s = s.str.extract(r'([0-9\.,]+)', expand=False).fillna('')
    s = s.str.replace(r'\.(?=\d{3}(?:[.,]|$))', '', regex=True)
    s = s.str.replace(',', '.', regex=False)
    if fill_zero:
        s = s.replace('', '0')
    return pd.to_numeric(s, errors='coerce')

pre_nan  = result[['agrena','fletes','seguro','otrosg']].isna().sum()
for col in ['agrena','fletes','seguro','otrosg']:
    result[col] = clean_amount(result[col], fill_zero=True)
post_nan = result[['agrena','fletes','seguro','otrosg']].isna().sum()

print("\n📊 NaNs en montos:")
print("  Antes:  ", pre_nan.to_dict())
print("  Después:", post_nan.to_dict())

# %% 
# 6.4 Conversión de columnas numéricas

columnas_float = ['adua', 'cod_sal1', 'dpto2', 'modad']
columnas_int   = ['via', 'regim', 'finalid', 'cer_ori1', 'sisesp', 'dpto1', 'unid']

for col in columnas_float:
    result[col] = pd.to_numeric(result[col], errors='coerce')

for col in columnas_int:
    result[col] = pd.to_numeric(result[col], errors='coerce').astype('Int64')

# %% 
# 6.5 Normalización de códigos
result['pais']     = result['pais'].astype(str).str.zfill(3)
result['cod_pai4'] = result['cod_pai4'].astype(str).str.zfill(3)
result['cod_sal']  = result['cod_sal'].astype(str).str.zfill(3)
result['coduni2']  = result['coduni2'].astype(str).str.zfill(3)

# %% 
# 6.6 Limpieza de montos tipo texto

def clean_number(col):
    return (result[col].astype(str)
            .str.replace('.', '', regex=False)   
            .str.replace(',', '.', regex=False)  
            .pipe(pd.to_numeric, errors='coerce'))

for col in ['canti', 'pbk', 'pnk', 'fobdol', 'fobpes']:
    result[col] = clean_number(col)

# %% 
# 6.7 Diagnóstico final: ver cuántos NaN quedaron y qué valores causaron problemas

print("\n📊 NaNs después de conversión:\n")

columnas_float = ['adua', 'cod_sal1', 'dpto2', 'modad']
columnas_int   = ['via', 'regim', 'finalid', 'cer_ori1', 'sisesp', 'dpto1', 'unid']

columnas_texto_convertidas = ['canti', 'pbk', 'pnk', 'fobdol', 'fobpes']

for col in columnas_float + columnas_int + columnas_texto_convertidas:
    print(f"{col}: {result[col].isna().sum()} NaNs")

print("\n🔎 Valores problemáticos que causaron NaNs:\n")
for col in columnas_float + columnas_int:
    originales = result[col].astype(str).str.strip()
    limpios = originales.str.replace('.', '', regex=False).str.replace(',', '.', regex=False)
    convertidos = pd.to_numeric(limpios, errors='coerce')
    problem_values = originales[convertidos.isna() & originales.notna()].unique()
    if len(problem_values) > 0:
        print(f"{col}: {len(problem_values)} problemáticos → {problem_values[:5]}")

# %% 
# 7. Crear columna de fecha 'fech'
result['fech_str'] = result['fech'].astype(str).str.strip()
valid_fech_mask = result['fech_str'].str.fullmatch(r'\d{3,4}')
result['year_num'] = pd.NA
result['month_num'] = pd.NA

# Extraer año
result.loc[valid_fech_mask & (result['fech_str'].str.len() == 3), 'year_num'] = (
    result.loc[valid_fech_mask & (result['fech_str'].str.len() == 3), 'fech_str'].str[0].astype(int) + 2000
)
result.loc[valid_fech_mask & (result['fech_str'].str.len() == 4), 'year_num'] = (
    result.loc[valid_fech_mask & (result['fech_str'].str.len() == 4), 'fech_str'].str[:2].astype(int) + 2000
)
# Mes
result.loc[valid_fech_mask, 'month_num'] = (
    result.loc[valid_fech_mask, 'fech_str'].str[-2:].astype(int)
)
# Construir fecha nueva
fecha_nueva = pd.to_datetime(
    dict(year=result['year_num'], month=result['month_num'], day=1),
    errors='coerce'
)
# Reemplazar en misma posición
fech_idx = result.columns.get_loc('fech')
result.drop(columns='fech', inplace=True)
result.insert(fech_idx, 'fech', fecha_nueva)
# Limpiar columnas auxiliares
result.drop(columns=['fech_str', 'year_num', 'month_num'], inplace=True)

# %% 
# 8. Diagnóstico rápido
result.info()
result.sample(30)
result.isna().sum()

# %% 
# 9. Normalizar y filtrar por códigos oficiales
result['posar'] = result['posar'].astype(str).str.zfill(10)
final = result[result['posar'].isin(CODIGOS_OFICIALES)].copy()
print(f"✅ Filas finales tras filtrar por códigos oficiales: {final.shape[0]:,}")

# %% 
# 10. Añadir descripción
desc = pd.DataFrame([
    {'posar':'1006400000','descripcion':'arroz'},
    {'posar':'1213000000','descripcion':'Cana de miel'},
    {'posar':'0801111000','descripcion':'coco'},
    {'posar':'1005100000','descripcion':'maiz'},
    {'posar':'0714300000','descripcion':'name'},
    {'posar':'1511100000','descripcion':'palma de aceite'},
    {'posar':'0807200000','descripcion':'papaya'},
    {'posar':'0807110000','descripcion':'patilla'},
    {'posar':'0804300000','descripcion':'pina'},
    {'posar':'0803901100','descripcion':'platano'},
    {'posar':'0714100000','descripcion':'yuca'}
])
final = final.merge(desc, on='posar', how='left')

# %% 
# 11. Filtrar por dpto1 Caribe
departamentos_caribe = [13, 23, 70, 88]
df_caribe = final[final['dpto1'].isin(departamentos_caribe)].copy()
# Diccionario con nombres de departamentos del Caribe filtrados
divipola_caribe = {
    13: 'Bolívar',
    23: 'Córdoba',
    70: 'Sucre',
    88: 'San Andrés'
}

df_caribe['dpto1'] = df_caribe['dpto1'].astype(int)
df_caribe['dpto_nombre'] = df_caribe['dpto1'].map(divipola_caribe)

# %%
# 12. Análisis y gráficos
# 12.1. Evolución temporal de exportaciones
#  Línea temporal (lineplot) por producto.
df_caribe['anio'] = df_caribe['fech'].dt.year

evolucion = df_caribe.groupby(['anio', 'descripcion'])['fobdol'].sum().reset_index()

top10 = (
    evolucion.groupby('descripcion')['fobdol']
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

productos_10_a_6 = top10.iloc[5:][::-1].index   
productos_5_a_1 = top10.iloc[:5][::-1].index    

evolucion_10_a_6 = evolucion[evolucion['descripcion'].isin(productos_10_a_6)]
evolucion_5_a_1 = evolucion[evolucion['descripcion'].isin(productos_5_a_1)]

def graficar_evolucion_interactiva(data, titulo):
    fig = px.line(
        data,
        x='anio',
        y='fobdol',
        color='descripcion',
        markers=True,
        title=titulo,
        labels={'anio': 'Año', 'fobdol': 'Valor FOB (USD)', 'descripcion': 'Producto'},
        hover_data={'anio': True, 'fobdol': ':.2s'}  # valores legibles (ej. 1.2M)
    )
    fig.update_layout(
        xaxis=dict(dtick=1),
        yaxis_title='Valor FOB (USD)',
        legend_title='Producto',
        hovermode='x unified',
        title_font_size=18
    )
    fig.show()

graficar_evolucion_interactiva(evolucion_10_a_6, 'Exportaciones: Productos del puesto 10 al 6')

graficar_evolucion_interactiva(evolucion_5_a_1, 'Exportaciones: Top 5 productos')

#%%
# Suma anual del valor exportado por producto.
top10

#%%
# 12.2. Ranking de productos por cantidad exportada
ranking_canti = df_caribe.groupby('descripcion')['canti'].sum().sort_values(ascending=False)
top10_canti = ranking_canti.head(10)

# Separar en dos segmentos
top_10_a_6_canti = top10_canti.iloc[5:]
top_5_a_1_canti = top10_canti.iloc[:5]

# Convertir a DataFrame para Plotly
df_top_10_a_6 = top_10_a_6_canti.reset_index().rename(columns={'descripcion': 'Producto', 'canti': 'Cantidad'})
df_top_5_a_1 = top_5_a_1_canti.reset_index().rename(columns={'descripcion': 'Producto', 'canti': 'Cantidad'})

# Gráfico 1 – Interactivo para puestos 10 a 6
fig1 = px.bar(
    df_top_10_a_6.sort_values(by='Cantidad', ascending=True),
    x='Cantidad',
    y='Producto',
    orientation='h',
    title='Productos exportados - Posiciones 10 a 6 (por cantidad)',
    text='Cantidad',
    color='Producto',
    color_discrete_sequence=px.colors.sequential.Oranges_r
)
fig1.update_layout(xaxis_title='Cantidad exportada', yaxis_title='Producto')
fig1.update_traces(texttemplate='%{text:.2s}', textposition='outside')
fig1.show()

# Gráfico 2 – Interactivo para top 5
fig2 = px.bar(
    df_top_5_a_1.sort_values(by='Cantidad', ascending=True),
    x='Cantidad',
    y='Producto',
    orientation='h',
    title='Top 5 productos exportados (por cantidad)',
    text='Cantidad',
    color='Producto',
    color_discrete_sequence=px.colors.sequential.YlOrBr
)
fig2.update_layout(xaxis_title='Cantidad exportada', yaxis_title='Producto')
fig2.update_traces(texttemplate='%{text:.2s}', textposition='outside')
fig2.show()

#%%
# Mostrar valores del top 10 por cantidad
top10_canti

#%%
# 12.2.1 Ranking de productos más exportados
df_inter = df_caribe.groupby(['descripcion', 'dpto_nombre'])['fobdol'].sum().reset_index()

top_10 = df_inter.groupby('descripcion')['fobdol'].sum().nlargest(10).index
df_inter = df_inter[df_inter['descripcion'].isin(top_10)]

df_inter['fob_log'] = df_inter['fobdol'].replace(0, np.nan)
df_inter['fob_log'] = np.log10(df_inter['fob_log'])

fig = px.density_heatmap(
    df_inter,
    x='dpto_nombre',
    y='descripcion',
    z='fob_log',
    text_auto='.2f',
    color_continuous_scale='Viridis',
    labels={'fob_log': 'Log₁₀(FOB USD)'},
    title='🌎 Heatmap Interactivo - Exportaciones por Departamento y Producto (Top 10)'
)

fig.update_layout(height=600)
fig.show()

#%% 12.3. Comparación por departamento de origen barras apiladas

df_grouped = df_caribe.groupby(['descripcion', 'dpto_nombre'])['fobdol'].sum().reset_index()

top_10 = df_grouped.groupby('descripcion')['fobdol'].sum().nlargest(10).index
df_top = df_grouped[df_grouped['descripcion'].isin(top_10)]

fig = px.bar(
    df_top,
    x='descripcion',
    y='fobdol',
    color='dpto_nombre',
    title='Valor FOB exportado por producto y departamento de origen (Caribe)',
    labels={'descripcion': 'Producto', 'fobdol': 'Valor FOB (USD)', 'dpto_nombre': 'Departamento'},
    hover_data={'fobdol': ':.2f'}
)

fig.update_layout(barmode='stack', xaxis_tickangle=-45)
fig.show()

#%%  12.3.1 Comparación por departamento de origen Treemap

df_grouped = df_caribe.groupby(['descripcion', 'dpto_nombre'])['fobdol'].sum().reset_index()
top_10 = df_grouped.groupby('descripcion')['fobdol'].sum().nlargest(10).index
df_top = df_grouped[df_grouped['descripcion'].isin(top_10)]

fig = px.treemap(
    df_top,
    path=['descripcion', 'dpto_nombre'],
    values='fobdol',
    title='Exportaciones por producto y departamento (Caribe)',
    color='fobdol',
    color_continuous_scale='Blues'
)
fig.show()

#%% 12.4. Exportaciones por país de destino

df_pais = (
    df_caribe.groupby(['pais', 'descripcion'])['fobdol']
    .sum()
    .reset_index()
    .sort_values(by='fobdol', ascending=False)
)

fig = px.treemap(
    df_pais,
    path=['pais', 'descripcion'],
    values='fobdol',
    color='fobdol',
    color_continuous_scale='Viridis',
    title='Exportaciones por País de Destino y Producto (Valor FOB)'
)

fig.update_layout(margin=dict(t=50, l=10, r=10, b=10))
fig.show()


#%% 12.5. Valor FOB vs. Peso bruto neto - rendimiento por tonelada

df_rendimiento = df_caribe.copy()
df_rendimiento = df_rendimiento[df_rendimiento['pnk'] > 0]

# Crear columna: FOB por tonelada (USD / Ton)
df_rendimiento['fob_por_tonelada'] = df_rendimiento['fobdol'] / (df_rendimiento['pnk'] / 1000)

fig = px.scatter(
    df_rendimiento,
    x='pnk',
    y='fob_por_tonelada',
    color='descripcion',
    size='fobdol',
    hover_data=['pais', 'dpto_nombre', 'anio'],
    title='Valor FOB vs. Peso Neto — Rendimiento por Tonelada',
    labels={
        'pnk': 'Peso Neto (kg)',
        'fob_por_tonelada': 'Valor FOB por Tonelada (USD)'
    },
    log_x=True,
    log_y=True,
    height=600
)

fig.update_layout(margin=dict(t=50, l=10, r=10, b=10))
fig.show()

#%% 12.6. Eficiencia logística
df_log = df_caribe.copy()
df_log = df_log[df_log['fobdol'] > 0]

df_log['costo_logistico_relativo'] = (
    (df_log['fletes'] + df_log['seguro'] + df_log['otrosg']) / df_log['fobdol']
) * 100

fig = px.box(
    df_log,
    x='descripcion',
    y='costo_logistico_relativo',
    points='all',
    title='Eficiencia Logística — % Costo Logístico vs. Valor FOB por Producto',
    labels={'costo_logistico_relativo': 'Costo logístico relativo (%)'},
    hover_data=['anio', 'pais', 'dpto_nombre']
)

fig.update_layout(xaxis_tickangle=-45, height=600)
fig.show()

#%% 12.7. Evolución por tipo de producto


evolucion = (
    df_caribe.groupby(['anio', 'descripcion'], as_index=False)['fobdol']
    .sum()
)

top_productos = (
    evolucion.groupby('descripcion')['fobdol'].sum()
    .sort_values(ascending=False)
    .head(10)
    .index
)

evolucion_top = evolucion[evolucion['descripcion'].isin(top_productos)]

fig = px.line(
    evolucion_top,
    x='anio',
    y='fobdol',
    color='descripcion',
    markers=True,
    title='Evolución Anual del Valor FOB por Producto (Top 10)',
    labels={'fobdol': 'Valor FOB (USD)', 'anio': 'Año'},
)

fig.update_layout(height=600)
fig.show()

# %% Guardar resultado final
output = 'expo_union_limpio_con_descripcion.csv'
df_caribe.to_csv(output, index=False, encoding='utf-8-sig')
print(f"✅ Guardado: {output}")

# %%
df_cultivos = pd.read_csv("../reports/cultivos_comparativos_caribe.csv")
# %%
df_cultivos
# %%
df_cultivos.info()
# %%
df_caribe['anio'] = df_caribe['anio'].astype(int)
df_caribe['dpto1'] = df_caribe['dpto1'].astype(int)
df_cultivos['codigo_dane_departamento'] = df_cultivos['codigo_dane_departamento'].astype(int)
df_cultivos['anio'] = df_cultivos['anio'].astype(int)

# %%
df_merged = df_cultivos.merge(
    df_caribe,
    left_on=['codigo_dane_departamento', 'anio', 'cultivo'],
    right_on=['dpto1', 'anio', 'descripcion'],
    how='left'  # o 'outer' si quieres conservar todo
)

# %%
df_merged
# %%
# Separar las filas que sí tienen exportaciones
df_con_exportacion = df_merged[df_merged['fobdol'].notna()]

#%%
# Separar las filas que no tienen (solo producción)
df_solo_cultivo = df_merged[df_merged['fobdol'].isna()]

# %%
df_con_exportacion
# %%
df_solo_cultivo
# %%
# %% GUARDAR LOS DATAFRAMES

# 1. DataFrame completo: exportaciones + cultivos sin exportar
df_merged.to_csv('../data/processed/union_cultivos_exportaciones.csv', index=False)

# 2. Solo cultivos con exportación
df_con_exportacion.to_csv('../data/processed/cultivos_con_exportacion.csv', index=False)

# 3. Solo cultivos sin exportación (producción local no exportada)
df_solo_cultivo.to_csv('../data/processed/cultivos_no_exportados.csv', index=False)

# %%
