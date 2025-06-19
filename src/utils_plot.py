import os
import pandas as pd
import matplotlib.pyplot as plt

def graficar_pivot(pivot_df, titulo, guardar_imagen=False, nombre_archivo=None, ruta='reports'):
    plt.figure(figsize=(12, 6))

    #Asegurar que el índice (año) es entero
    pivot_df.index = pivot_df.index.astype(int)

    #Dibujar cada línea
    for col in pivot_df.columns:
        plt.plot(pivot_df.index, pivot_df[col], label=col, linewidth=2)

    #Forzar ticks enteros en el eje X
    plt.xticks(ticks=sorted(pivot_df.index.unique()))

    plt.title(titulo)
    plt.xlabel("Año")
    plt.ylabel("Producción (Toneladas)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    #Guardar imagen si se solicita
    if guardar_imagen:
        os.makedirs(ruta, exist_ok=True)
        if not nombre_archivo:
            nombre_archivo = titulo.replace(" ", "_").lower() + ".png"
        ruta_completa = os.path.join(ruta, nombre_archivo)
        plt.savefig(ruta_completa, dpi=300)
        print(f"🖼️ Imagen guardada en: {ruta_completa}")

    #Mostrar gráfica
    try:
        plt.show(block=True)
    except TypeError:
        plt.show()

def guardar_pivot(pivot_df, nombre_archivo, ruta='reports'):
    os.makedirs(ruta, exist_ok=True)
    ruta_completa = os.path.join(ruta, nombre_archivo)
    pivot_df.to_csv(ruta_completa, index=True)
    print(f"✅ Pivot guardado en: {ruta_completa}")

