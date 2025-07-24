# %% 
# 1 Importar librerías
import os
import re
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
#%%
# CONVERSIÓN A TONEADAS DE CANTIDADES DE EXPORTACIÓN
def convertir_a_toneladas(row):
    # Extraer sólo las letras, ignorar dígitos
    raw = str(row["coduni2"])
    unidad = re.sub(r"\d", "", raw).strip().lower()  # e.g. "0KG" → "kg"
    
    if unidad == "kg":
        return row["canti"] / 1000
    elif unidad in ("gr", "g"):
        return row["canti"] / 1_000_000
    elif unidad in ("ton", "tonelada"):
        return row["canti"]
    else:
        return np.nan

# Aplica la conversión
result["canti_ton"] = result.apply(convertir_a_toneladas, axis=1)
# %% 
# 6.7 Diagnóstico final: ver cuántos NaN quedaron y qué valores causaron problemas

print("\n📊 NaNs después de conversión:\n")
for col in columnas_float + columnas_int + ['canti', 'canti_ton', 'pbk', 'pnk', 'fobdol', 'fobpes']:
    print(f"{col}: {result[col].isna().sum()} NaNs")
'''print("\n📊 NaNs después de conversión:\n")

columnas_float = ['adua', 'cod_sal1', 'dpto2', 'modad']
columnas_int   = ['via', 'regim', 'finalid', 'cer_ori1', 'sisesp', 'dpto1', 'unid']

columnas_texto_convertidas = ['canti', 'pbk', 'pnk', 'fobdol', 'fobpes']

for col in columnas_float + columnas_int + columnas_texto_convertidas:
    print(f"{col}: {result[col].isna().sum()} NaNs")
'''
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
    dict(year=result['year_num'], month=result['month_num'], day=1), # type: ignore
    errors='coerce'
) # type: ignore
# Reemplazar en misma posición
fech_idx = result.columns.get_loc('fech')
result.drop(columns='fech', inplace=True)
result.insert(fech_idx, 'fech', fecha_nueva) # type: ignore
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

#%%

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
    .drop('arroz', errors='ignore')
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
ranking_canti = df_caribe.groupby('descripcion')['canti_ton'].sum().sort_values(ascending=False)
top10_canti = ranking_canti.head(10)

# Separar en dos segmentos
top_10_a_6_canti = top10_canti.iloc[5:]
top_5_a_1_canti = top10_canti.iloc[:5]

# Convertir a DataFrame para Plotly
df_top_10_a_6 = top_10_a_6_canti.reset_index().rename(columns={'descripcion': 'Producto', 'canti_ton': 'Cantidad'})
df_top_5_a_1 = top_5_a_1_canti.reset_index().rename(columns={'descripcion': 'Producto', 'canti_ton': 'Cantidad'})

df_top_10_a_6['Cantidad'] = df_top_10_a_6['Cantidad'].round(2)
df_top_5_a_1['Cantidad'] = df_top_5_a_1['Cantidad'].round(2)

# Gráfico 1 – Interactivo para puestos 10 a 6
fig1 = px.bar(
    df_top_10_a_6.sort_values(by='Cantidad', ascending=False),
    x='Cantidad',
    y='Producto',
    orientation='h',
    title='Productos exportados - Posiciones 6 a 10 (por cantidad)',
    text='Cantidad',
    color='Producto',
    color_discrete_sequence=px.colors.sequential.Oranges_r
)
fig1.update_layout(xaxis_title='Cantidad exportada', yaxis_title='Producto')
fig1.update_traces(texttemplate='%{text:.2s}', textposition='outside')
fig1.show()

# Gráfico 2 – Interactivo para top 5
fig2 = px.bar(
    df_top_5_a_1.sort_values(by='Cantidad', ascending=False),
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
# 12.2.1 Ranking de productos más exportados (valores en millones)
df_inter = (
    df_caribe
    .groupby(['descripcion', 'dpto_nombre'])['fobdol']
    .sum()
    .reset_index()
)

# Filtrar solo Top 10 productos por FOB total
top_10 = (
    df_inter
    .groupby('descripcion')['fobdol']
    .sum()
    .nlargest(10)
    .index
)
df_inter = df_inter[df_inter['descripcion'].isin(top_10)].copy()

# Crear columna de FOB en millones
df_inter['fob_k'] = df_inter['fobdol'] / 1_000

# Graficar heatmap con valores en millones
# ... previo: agrupación y cálculo de df_inter con 'fob_mill' = fobdol / 1e6 ...

# Supongamos que ya tienes df_inter con 'fob_k' = fobdol / 1000


fig = px.density_heatmap(
    df_inter,
    x='dpto_nombre',
    y='descripcion',
    z='fob_k',
    text_auto='.0f',                 # muestra valores enteros en K USD # type: ignore
    color_continuous_scale='Viridis',
    range_color=[0, 20000],            # escala de 0 a 100 (miles USD)
    labels={
        'fob_k': 'FOB (miles USD)',
        'dpto_nombre': 'Departamento',
        'descripcion': 'Producto'
    },
    title='Exportaciones por Departamento y Producto (Top 10) en Miles USD'
)

fig.update_layout(
    height=600,
    xaxis_title='Departamento',
    yaxis_title='Producto'
)

# Opcional: ajustar ticks de la barra a saltos de 25K
fig.update_coloraxes(
    colorbar_tickvals=[0, 25, 50, 75, 100, 150, 1000, 15000, 20000],
    colorbar_title='Miles USD'
)

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
#%%
#Diccionario de pais a nombres de paises
codigo_nombre_pais = {
    "ABW": "Aruba",
    "BEL": "Bélgica",
    "CAN": "Canadá",
    "CUW": "Curazao",
    "DEU": "Alemania",
    "DOM": "República Dominicana",
    "ESP": "España",
    "FRA": "Francia",
    "GBR": "Reino Unido",
    "GLP": "Guadalupe",
    "GUY": "Guyana",
    "HND": "Honduras",
    "MEX": "México",
    "MTQ": "Martinica",
    "NLD": "Países Bajos",
    "PAN": "Panamá",
    "PRI": "Puerto Rico",
    "PRT": "Portugal",
    "TUR": "Turquía",
    "USA": "Estados Unidos",
    "VEN": "Venezuela"
}
#%% # REVISION codpais EN CASO DE ERROR
'''# REVISION EN CASO DE ERROR
df_caribe.info()
df_caribe[['cod_pai4', 'descripcion']].isna().sum()
df_caribe[['cod_pai4', 'descripcion', 'fobdol']].dropna(how='all').head(10)'''


#%%
df_caribe['cod_pai4'] = df_caribe['cod_pai4'].map(codigo_nombre_pais)
#%% 12.4. Exportaciones por país de destino

df_pais = (
    df_caribe.groupby(['cod_pai4', 'descripcion'])['fobdol']
    .sum()
    .reset_index()
    .sort_values(by='fobdol', ascending=False)
)

fig = px.treemap(
    df_pais,
    path=['cod_pai4', 'descripcion'],
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


import plotly.express as px

# 1. Cálculo de evolución
evolucion = (
    df_caribe
    .groupby(['anio', 'descripcion'], as_index=False)['fobdol']
    .sum()
)

# 2. Top 10 por valor total
top_productos = (
    evolucion
    .groupby('descripcion')['fobdol']
    .sum()
    .sort_values(ascending=False)
    .head(10)
    .index
)

# 3. Filtrar evolución a esos productos y quitar 'arroz'
productos_filtrados = [p for p in top_productos if p != 'arroz']

evolucion_top = evolucion[evolucion['descripcion'].isin(productos_filtrados)]

# 4. Gráfico
fig = px.line(
    evolucion_top,
    x='anio',
    y='fobdol',
    color='descripcion',
    markers=True,
    title='Evolución Anual del Valor FOB por Producto (Top 10 sin Arroz)',
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
#%%
df_con_exportacion.info()
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
df_con_exportacion.info()


######################################################

# %% 
# CAGR (Tasa de crecimiento anual compuesta) de FOB y volumen
# 1.1 Lee el CSV
# Cargar el dataset combinado
df = pd.read_csv('../data/processed/cultivos_con_exportacion.csv')

# Filtrar solo filas con FOB > 0
df_export = df[df['fobdol'] > 0].copy()

# 1. Encontrar año de inicio y fin para cada cultivo
years = (
    df_export.groupby('cultivo')['anio']
            .agg(start_year='min', end_year='max')
            .reset_index()
)

# 2. Calcular CAGR dinámico para cada cultivo
results = []
for _, row in years.iterrows():
    cultivo = row['cultivo']
    start, end = row['start_year'], row['end_year']
    n = end - start
    if n <= 0:
        continue
    fob_start = df_export[(df_export['cultivo'] == cultivo) & (df_export['anio'] == start)]['fobdol'].sum()
    fob_end   = df_export[(df_export['cultivo'] == cultivo) & (df_export['anio'] == end)]['fobdol'].sum()
    if fob_start > 0 and fob_end > 0:
        cagr = (fob_end / fob_start)**(1/n) - 1
    else:
        cagr = np.nan
    results.append({'cultivo': cultivo, 'start_year': start, 'end_year': end, 'periods': n, 'CAGR_FOB': cagr})

cagr_df = pd.DataFrame(results)

# Ordenar por crecimiento descendente y mostrar top
cagr_df = cagr_df.sort_values('CAGR_FOB', ascending=False)

print("Top 5 CAGR FOB válidos:")
print(cagr_df.dropna(subset=['CAGR_FOB']).head(10))

#%% # Ratio exportación / producción (Tasa de inserción)
# Ratio exportación / producción (Tasa de inserción)
# 2.1 Asumiendo que df ya está cargado:
# Para cada fila: tasa = exportaciones (ton) / producción (ton)
df['produccion_toneladas'] = pd.to_numeric(df['produccion_toneladas'], errors='coerce')
df['canti_ton']          = pd.to_numeric(df['canti_ton'], errors='coerce')

mask = (df['produccion_toneladas'] > 0) & (df['canti_ton'].notna())
df_valid = df.loc[mask].copy()

# 2.2 Agrega por cultivo y departamento (promedio histórico)
df_valid['ratio_insercion'] = df_valid['canti_ton'] / df_valid['produccion_toneladas']

ratio_agg = (
    df_valid
    .groupby(['dpto_nombre','cultivo'])['ratio_insercion']
    .mean()
    .reset_index()
    .sort_values('ratio_insercion', ascending=False)
)

# 2.3 Muestra los top 10 combinaciones más altas
print("Top 10 tasas de inserción (exportación/producción):")
print(ratio_agg.head(10).to_string(index=False))



#%%
# Índice de diversificación de mercados (Herfindahl–Hirschman)
# 3.1 Calcula participación de cada país destino dentro del FOB total del cultivo
from sklearn.preprocessing import normalize  # si no tienes, usa numpy puro

# FOB total por cultivo y país
pais_agg = (
    df.loc[df['fobdol']>0]
    .groupby(['cultivo','cod_pai4'])['fobdol']
    .sum()
    .reset_index(name='fob_cult_pais')
)


# FOB total por cultivo
pais_agg['total_fob'] = pais_agg.groupby('cultivo')['fob_cult_pais'].transform('sum')
pais_agg['share']     = pais_agg['fob_cult_pais'] / pais_agg['total_fob']

hh = (
    pais_agg
    .groupby('cultivo')['share']
    .agg(HHI=lambda s: (s**2).sum())
    .reset_index()
    .sort_values('HHI')
)

# Cultivos con menor HHI → más diversificados en mercados
print("Top 10 cultivos más diversificados (HHI más bajo):")
print(hh.head(10).to_string(index=False))
# %%
#GRAFICAS DE TASA DE CRECIMIENTO ANUAL COMPUESTA, 
# TASA DE INSERCIÓN E INDICE DE DIVERSIFICACIÓN DE MERCADOS

# 1) Top 5 CAGR FOB
# 1) Define los cultivos a excluir
excluir = ['arroz', 'coco']

# 2) Filtra el DataFrame y toma los Top 5
top5_cagr = (
    cagr_df
    .dropna(subset=['CAGR_FOB'])
    .loc[~cagr_df['cultivo'].isin(excluir)]   # quita arroz y coco
    .nlargest(5, 'CAGR_FOB')                   # los 5 mayores CAGR
)

fig1 = px.bar(
    top5_cagr,
    x='cultivo',
    y='CAGR_FOB',
    title='Top 5 Tasa de crecimiento anual compuesta FOB por Cultivo',
    labels={'cultivo':'Cultivo','CAGR_FOB':'CAGR FOB'},
    text=top5_cagr['CAGR_FOB'].round(2)
)
fig1.update_traces(textposition='outside')
fig1.update_layout(yaxis_tickformat='.0%')
fig1.show()

#%%
# 2) Top 10 Ratio Inserción

# 1. Carga el dataset
df = pd.read_csv('../data/processed/cultivos_con_exportacion.csv')

# 2. Convierte a numérico por si quedan strings
df['canti_ton']            = pd.to_numeric(df['canti_ton'], errors='coerce')
df['produccion_toneladas'] = pd.to_numeric(df['produccion_toneladas'], errors='coerce')

# 3. Filtra para quedarte sólo con casos coherentes: export <= producción
df_clean = df.loc[df['canti_ton'] <= df['produccion_toneladas']].copy()

# 4. Calcula ratio de inserción
df_clean['ratio_insercion'] = df_clean['canti_ton'] / df_clean['produccion_toneladas']

# 5. Agrega promedio histórico por departamento y cultivo
ratio_agg_clean = (
    df_clean
    .groupby(['dpto_nombre','cultivo'])['ratio_insercion']
    .mean()
    .reset_index()
    .sort_values('ratio_insercion', ascending=False)
)

# 6. Selecciona Top 10 para graficar
top10_clean = ratio_agg_clean.head(20)

# 7. Gráfico de barras horizontales
fig = px.bar(
    top10_clean,
    x='ratio_insercion',
    y='cultivo',
    color='dpto_nombre',
    orientation='h',
    title='Top 9 Ratio Exportación/Producción',
    labels={
        'ratio_insercion':'Ratio Exp/Prod',
        'cultivo':'Cultivo',
        'dpto_nombre':'Departamento'
    },
    text=top10_clean['ratio_insercion'].round(2)
)
fig.update_traces(textposition='outside')
fig.update_layout(yaxis={'categoryorder':'total ascending'})
fig.show()


#%%
# 3) Top 10 Diversificación (HHI más bajo)
top10_hhi = hh.head(10)
fig3 = px.bar(
    top10_hhi,
    x='HHI',
    y='cultivo',
    orientation='h',
    title='Top 10 Cultivos Más Diversificados (HHI bajo)',
    labels={'HHI':'Herfindahl–Hirschman Index','cultivo':'Cultivo'},
    text=top10_hhi['HHI'].round(3)
)
fig3.update_traces(textposition='outside')
fig3.update_layout(xaxis_tickformat='.2f')
fig3.show()
#%%

# %%

# 1. Agrupa y suma FOB
df_pais = (
    df_caribe
    .groupby(['cod_pai4', 'descripcion'])['fobdol']
    .sum()
    .reset_index()
)

# 2. Filtrar solo los Top N productos si prefieres
top_products = (
    df_pais
    .groupby('descripcion')['fobdol']
    .sum()
    .nlargest(10)
    .index
)
df_top = df_pais[df_pais['descripcion'].isin(top_products)].copy()

# 3. Pivotar: filas = país, columnas = producto, valores = FOB
heatmap_df = df_top.pivot(index='cod_pai4', columns='descripcion', values='fobdol').fillna(0)

# 4. Crear el heatmap
fig = px.imshow(
    heatmap_df,
    labels=dict(x="Producto", y="País de Destino", color="FOB (USD)"),
    x=heatmap_df.columns,
    y=heatmap_df.index,
    text_auto='.2s',             # formato legible, ej. "1.2M" # type: ignore
    aspect="auto",
    color_continuous_scale='Viridis',
    title='Heatmap de Exportaciones por País de Destino y Producto (Top 10)'
)

# 5. Ajustes de layout
fig.update_xaxes(tickangle=45)
fig.update_layout(
    height=600,
    margin=dict(t=50, l=100, r=10, b=100)
)

fig.show()

# %%

# 1. Carga tu DataFrame combinado (producción + exportación)
df = pd.read_csv('../data/processed/cultivos_con_exportacion.csv')

# 2. Asegúrate de que 'produccion_toneladas' sea numérico
df['produccion_toneladas'] = pd.to_numeric(df['produccion_toneladas'], errors='coerce')

# 3. Agrupa por cultivo y suma la producción, luego toma el Top 10
top10_prod = (
    df
    .groupby('cultivo', as_index=False)['produccion_toneladas']
    .sum()
    .nlargest(10, 'produccion_toneladas')
)

# 4. Gráfico de barras con Plotly Express
fig = px.bar(
    top10_prod,
    x='cultivo',
    y='produccion_toneladas',
    text=top10_prod['produccion_toneladas'].round(0),
    labels={
        'cultivo': 'Cultivo',
        'produccion_toneladas': 'Producción Total (toneladas)'
    },
    title='Top 10 Cultivos por Producción Total Acumulada',
    log_y=True
)

# 5. Ajustes de estilo
fig.update_traces(textposition='outside', marker_color='seagreen')
fig.update_layout(
    xaxis_tickangle=45,
    yaxis=dict(title='Producción Total (toneladas)', showgrid=True),
    margin=dict(t=60, b=120)
)

fig.show()

# %%
##########################################################################
##############################################################################
# Carga los archivos locales
union = pd.read_csv("../data/processed/union_cultivos_exportaciones.csv")
print("\nDepartamentos únicos en UNION:")
print(union['nombre_departamento'].unique())
con_exp = pd.read_csv("../data/processed/cultivos_con_exportacion.csv")
sin_exp = pd.read_csv("../data/processed/cultivos_no_exportados.csv")

# Verificación rápida de columnas
print("Columnas en UNION:", union.columns)
print("Columnas en CON_EXP:", con_exp.columns)
print("Columnas en SIN_EXP:", sin_exp.columns)

# Filtra por región Caribe y producción válida
region_caribe = ['Bolívar', 'Córdoba', 'San Andrés', 'Sucre', 'Atlántico', 'Magdalena']

# Normalizamos nombres por si hay problemas de mayúsculas/minúsculas
union['nombre_departamento'] = union['nombre_departamento'].str.lower().str.strip()

# Paso 1: Filas solo por región Caribe
region_caribe = ['bolivar', 'cordoba', 'san andres y providencia', 'sucre']
solo_region = union[union['nombre_departamento'].isin(region_caribe)]
print("\n📌 Filas solo por región Caribe:", solo_region.shape)

# Paso 2: Filas solo con producción > 0
solo_produccion = union[union['produccion_toneladas'] > 0]
print("📌 Filas solo con producción > 0:", solo_produccion.shape)

# Paso 3: Ambas condiciones (el filtro real)
union_filtrado = union[
    (union['nombre_departamento'].isin(region_caribe)) &
    (union['produccion_toneladas'] > 0)
]
print("📌 Filas con ambas condiciones:", union_filtrado.shape)


union_filtrado = union[
    (union['nombre_departamento'].isin(region_caribe)) &
    (union['produccion_toneladas'] > 0)
]

print("Filas tras filtrar por Caribe y producción > 0:", union_filtrado.shape)

# Si existe la columna 'canti', eliminar registros con valores nulos en exportación
if 'canti' in union_filtrado.columns:
    union_filtrado = union_filtrado[~union_filtrado['canti'].isnull()]
    print("Filas con 'canti' no nulo:", union_filtrado.shape)

# También filtra con_exp y sin_exp por la región Caribe
con_exp = con_exp[con_exp['nombre_departamento'].isin(region_caribe)]
sin_exp = sin_exp[sin_exp['nombre_departamento'].isin(region_caribe)]

# Tasa de exportación / producción
prod_exp = union_filtrado.groupby(['cultivo', 'anio']).agg({
    'produccion_toneladas': 'sum',
    'canti': 'sum'
}).reset_index()
prod_exp['tasa_exportacion'] = (prod_exp['canti'] / prod_exp['produccion_toneladas']) * 100

# Rendimiento: producción / área cosechada
yield_df = union_filtrado.groupby(['cultivo', 'anio']).agg({
    'produccion_toneladas': 'sum',
    'area_cosechada_ha': 'sum'
}).reset_index()
yield_df['rendimiento_t_ha'] = yield_df['produccion_toneladas'] / yield_df['area_cosechada_ha']

# Precio promedio por tonelada exportada
con_exp_year = con_exp.groupby(['cultivo', 'anio']).agg({
    'fobdol': 'sum',
    'canti': 'sum'
}).reset_index()
con_exp_year['precio_usd_ton'] = con_exp_year['fobdol'] / con_exp_year['canti']

# Diversificación de exportaciones
divers = con_exp.groupby('anio').apply(
    lambda df: pd.Series({
        'anio': df['anio'].iloc[0],
        'herfindahl_export': ((df.groupby('cultivo')['canti'].sum() / df['canti'].sum())**2).sum(),
        'productos_exportados': df['cultivo'].nunique()
    })
).reset_index(drop=True)

# Tendencias de producción y exportación por año
trend = union_filtrado.groupby('anio').agg({
    'produccion_toneladas': 'sum',
    'canti': 'sum'
}).reset_index()

# Cálculo de producción media y tasa de exportación media por cultivo
avg_prod = union_filtrado.groupby('cultivo')['produccion_toneladas'].mean()

if prod_exp.empty:
    print("\n⚠️ El DataFrame 'prod_exp' está vacío. No se puede calcular tasa_exportacion promedio.")
    avg_rate = pd.Series(dtype='float64')
else:
    avg_rate = prod_exp.groupby('cultivo')['tasa_exportacion'].mean()

# Unir ambos en un nuevo DataFrame
potential = pd.DataFrame({
    'produccion_media': avg_prod,
    'tasa_export_media': avg_rate
}).reset_index()

# Diagnóstico
print("\n--- Producción media (describe) ---")
print(potential['produccion_media'].describe())

print("\n--- Tasa de exportación media (describe) ---")
print(potential['tasa_export_media'].describe())

print("\n📊 Cultivos con sus valores promedio:")
print(potential.sort_values(by='produccion_media', ascending=False))

# Comprobar si el DataFrame no está vacío antes de aplicar filtros
if not potential.empty:
    potential = potential[
        (potential['produccion_media'] > potential['produccion_media'].quantile(0.30)) &
        (potential['tasa_export_media'] < potential['tasa_export_media'].quantile(0.60))
    ]


    print("\n🔍 Posibles cultivos con alta producción y baja exportación:\n")
    print(potential.sort_values(by='produccion_media', ascending=False).head(10))
else:
    print("\n⚠️ No hay datos en 'potential' para analizar cultivos con potencial de exportación.")


#----------------------------------------------------------------GRAFICA----------------------------------------------------------------



fig = px.bar(
    potential,
    x='cultivo',
    y='produccion_media',
    text='tasa_export_media',
    labels={
        'cultivo': 'Cultivo',
        'produccion_media': 'Producción media (toneladas)',
        'tasa_export_media': 'Tasa exportación (%)'
    },
    title='Cultivos con Alta Producción y Baja Exportación - Región Caribe'
)

# Formatear el texto para que no tenga decimales
fig.update_traces(
    texttemplate='%{text:.0f}',
    textposition='outside',
    marker_color='skyblue'
)

# Ajustes de layout
fig.update_layout(
    xaxis_tickangle=45,
    yaxis=dict(
        title='Producción media (toneladas)',
        showgrid=True,
        gridcolor='LightGray',
        gridwidth=0.5
    ),
    uniformtext_minsize=8,
    uniformtext_mode='hide',
    margin=dict(t=60, b=100),
    showlegend=False
)

fig.show()
# %%
