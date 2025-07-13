
'''Este código es para limpiar y filtrar los datos para obtener información
sobre las exportaciones de los productos que ya previamente hemos tabulado
en el codigo EAM2019... Logramos obtener las exportaciones de cada uno de
los productos para los paises que se exportaron y la cantidad en kg,
ajustado por los años'''
#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#%%
df = pd.read_csv("../DATOS/colombia_exports_hs6_2013_2023.csv", encoding='latin1')
dfcopy = df.copy()
#%%
print(df.columns)

# %%
#limpiar columnas
df = df.drop(['cmdDesc', 'partnerISO', 'reporterDesc','typeCode',
            'freqCode','refPeriodId','mosCode','motCode', 'motDesc',
            'customsDesc', 'customsCode', 'partner2Desc', 'period', 
            'refMonth', 'reporterCode', 'reporterISO', 'partner2Code', 
            'partner2ISO', 'classificationCode', 'isOriginalClassification',
            'classificationSearchCode','aggrLevel','isLeaf','customsCode',
            'customsDesc','altQtyUnitCode','altQtyUnitAbbr','altQty',
            'isAltQtyEstimated','netWgt','isNetWgtEstimated','grossWgt',
            'isGrossWgtEstimated','cifvalue','primaryValue',
            'legacyEstimationFlag','isReported', 'isQtyEstimated', 'isAggregate','flowDesc','flowCode', 'partnerCode'], axis=1)

# %%
df.info()
#%%
#quitar filas no deseadas
df = df[~df['cmdCode'].isin([121190, 80711])]
# %%
#quitar filas de World
df = df[~df['partnerDesc'].isin(["World"])]

#%%
#lo mismo pero con pivot porque jose quizo
top_pro = (
    pd.pivot_table(
        df,
        index = 'cmdCode',
        values = 'qty',
        aggfunc = "sum",
    )
    .sort_values(by='qty', ascending=False)
    .round(2)
)
# %%
# Lista de los 10 productos más exportados
top_10_codigos = top_pro.head(10).index.tolist()
#%%
# Filtrar para quedarnos con los top productos
df_top10 = df[df['cmdCode'].isin(top_10_codigos)]

#%%
# Agrupar por producto, país destino y año
exportaciones_por_pais = (
    df_top10.groupby(['refYear', 'cmdCode', 'partnerDesc'])['qty']
    .sum()
    .reset_index()
    .sort_values(by='qty', ascending=False)
)

exportaciones_por_pais # type: ignore

# %%
#AGRUPACIÓN PARA YUCA:
producto_71410 = df[df['cmdCode'] == 71410]

evolucion_anio_yuca = (
    producto_71410.groupby(['refYear', 'partnerDesc'])['qty']
    .sum()
    .reset_index()
    .sort_values(by='refYear')
)

evolucion_anio_yuca# type: ignore
# %%
#AGRUPACIÓN PARA PAPA:
producto_71420 = df[df['cmdCode'] == 71420]

evolucion_anio_papa = (
    producto_71420.groupby(['refYear', 'partnerDesc'])['qty']
    .sum()
    .reset_index()
    .sort_values(by='refYear')
)

evolucion_anio_papa# type: ignore
# %%
#AGRUPACIÓN PARA ÑAME:
producto_71430 = df[df['cmdCode'] == 71430]

evolucion_anio_niame = (
    producto_71430.groupby(['refYear', 'partnerDesc'])['qty']
    .sum()
    .reset_index()
    .sort_values(by='refYear')
)

evolucion_anio_niame# type: ignore
# %%
#AGRUPACIÓN PARA COCO:
producto_80119 = df[df['cmdCode'] == 80119]

evolucion_anio_coco = (
    producto_80119.groupby(['refYear', 'partnerDesc'])['qty']
    .sum()
    .reset_index()
    .sort_values(by='refYear')
)

evolucion_anio_coco# type: ignore
# %%
#AGRUPACIÓN PARA PLATANO:
producto_80390 = df[df['cmdCode'] == 80390]

evolucion_anio_platano = (
    producto_80390.groupby(['refYear', 'partnerDesc'])['qty']
    .sum()
    .reset_index()
    .sort_values(by='refYear')
)

evolucion_anio_platano# type: ignore
# %%
#AGRUPACIÓN PARA MAIZ:
producto_100590 = df[df['cmdCode'] == 100590]

evolucion_anio_maiz = (
    producto_100590.groupby(['refYear', 'partnerDesc'])['qty']
    .sum()
    .reset_index()
    .sort_values(by='refYear')
)

evolucion_anio_maiz# type: ignore
# %%
#AGRUPACIÓN PARA ARROZ:
producto_100610 = df[df['cmdCode'] == 100610]

evolucion_anio_arroz = (
    producto_100610.groupby(['refYear', 'partnerDesc'])['qty']
    .sum()
    .reset_index()
    .sort_values(by='refYear')
)

evolucion_anio_arroz# type: ignore
# %%
#GRÁFICOS SEPARADOS DE CADA PRODUCTO 
#EN FUNCIÓN DE AÑOS Y CANTIDAD EXPORTADA
productos = {
    71410: "Yuca",
    71420: "Papa",
    71430: "Ñame",
    80119: "Coco",
    80390: "Plátano",
    100590: "Maíz",
    100610: "Arroz"
}

# Lista de códigos
codigos = list(productos.keys())

# Filtrar solo los productos de interés
df_filtrado = df[df['cmdCode'].isin(codigos)]

# Crear figura y ejes para subplots (ajusta filas y columnas si necesitas)
fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(16, 10))
axes = axes.flatten()

# Graficar cada producto
for i, codigo in enumerate(codigos):
    # Filtrar por producto
    df_producto = df_filtrado[df_filtrado['cmdCode'] == codigo]
    
    # Agrupar por año (sin país) y sumar cantidad
    evolucion = df_producto.groupby('refYear')['qty'].sum()

    # Graficar
    axes[i].plot(evolucion.index, evolucion.values, marker='o')
    axes[i].set_title(f"{productos[codigo]} ({codigo})")
    axes[i].set_xlabel("Año")
    axes[i].set_ylabel("Cantidad")
    axes[i].grid(True)

# Eliminar los subgráficos vacíos si hay menos productos que espacios
for j in range(len(codigos), len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.suptitle("Evolución anual de exportaciones por producto", fontsize=16, y=1.03)
plt.show()


# %%
#GRÁFICA DE LINEAS PARA TODOS LOS PRODUCTOS 
# EN CUANTO A AÑOS Y CANTIDAD EXPORTADA
# Extraer lista de códigos
productos_interes = list(productos.keys())

# Filtrar datos
df_filtrado = df[df['cmdCode'].isin(productos_interes)]


df_agrupado = df_filtrado.groupby(['refYear', 'cmdCode'])['qty'].sum().reset_index()
df_pivot = df_agrupado.pivot(index='refYear', columns='cmdCode', values='qty').fillna(0)


plt.figure(figsize=(12, 6))

# Dibujar cada línea con nombre legible desde el diccionario
for codigo in df_pivot.columns:
    nombre = productos.get(codigo, f"Producto {codigo}")  # type: ignore
    plt.plot(df_pivot.index, df_pivot[codigo], marker='o', label=nombre)


plt.yscale("log")
plt.title("Evolución de exportaciones por producto")
plt.xlabel("Año")
plt.ylabel("Cantidad exportada (log)")
plt.legend(title="Producto", loc="upper left")
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.tight_layout()
plt.show()

# %%

