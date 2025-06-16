
#Series: Estructura de una dimensión. Inmutables en tamaño pero no en contenido
#DataFrame: Estructura de dos dimensiones (tablas).
#Panel: Estructura de tres dimensiones (cubos).

#Creación de una serie a partir de una lista
#%%
import pandas as pd
Asignaturas = pd.Series(["matematicas", "sociales", "castellano", "ingles"], dtype = str)
print(Asignaturas)
Asignaturas.size

#%%
#Creación de una serie a partir de un diccionario
import pandas as pd
Notas = pd.Series({"matematicas": 3.0, "economia": 4.0, "Sociales": 4.2})
print(Notas)
Notas.size
