import os
import pandas as pd
import matplotlib.pyplot as plt

def graficar_pivot(pivot_df, titulo):
    plt.figure(figsize=(12, 6))
    for col in pivot_df.columns:
        plt.plot(pivot_df.index, pivot_df[col], label=col, linewidth=2)

    plt.title(titulo)
    plt.xlabel("Año")
    plt.ylabel("Producción (Toneladas)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def guardar_pivot(pivot_df, nombre_archivo, ruta='reports'):
    os.makedirs(ruta, exist_ok=True)
    ruta_completa = os.path.join(ruta, nombre_archivo)
    pivot_df.to_csv(ruta_completa, index=True)
    print(f"✅ Pivot guardado en: {ruta_completa}")
