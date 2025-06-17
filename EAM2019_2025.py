#%%
#Funciones y clases
#Clase para preprocesamiento de los datos
class DataPreproc():

    """
    Clase para preprocesamiento de datos.

    Permite convertir nombres de columnas a minúsculas, eliminar espacios en los nombres de columnas, 
    reemplazar espacios por guiones bajos en los nombres de columnas, escalar datos numéricos y revertir el escalado.
    """
    def __init__(self, data):
        
        """
        Inicializa la clase DataPreproc con un dataframe de datos y crea una instancia de StandardScaler.

        Args:
        - data (DataFrame): El dataframe de datos.
        """        
        self.df = data
        
    def convert_to_lowercase(self):
        
        """
        Convierte los nombres de las columnas a minúsculas.
        """             
        new_col_names = [old_name.lower() for old_name in self.df.columns]
        self.df.columns = new_col_names
        return self.df.columns

    def remove_spaces(self):
        
        """
        Elimina los espacios al principio y al final de los nombres de las columnas.
        """        
         
        new_col_names = [old_names.strip() for old_names in self.df.columns]
        self.df.columns = new_col_names
        return self.df.columns
    
    def add_underscore_between_words(self):
        """
        Añade guiones bajos entre palabras en los nombres de las columnas.
        """
        new_col_names = [re.sub(r'([a-z])([A-Z])', r'\1_\2', col).lower() for col in self.df.columns]
        self.df.columns = new_col_names
        return self.df.columns    
    

    def replace_spaces_with_underscore(self):
        
        """
        Reemplaza los espacios en los nombres de las columnas por guiones bajos.
        """        
        
        new_col_names = [old_names.replace(" ", "_") for old_names in self.df.columns]
        self.df.columns = new_col_names
        return self.df.columns
    
    def convert_text_to_lowercase(self):
        for col in self.df.select_dtypes(include='object').columns:
            self.df[col] = self.df[col].str.lower()
    
    def run_all_preprocessing(self):
        self.convert_to_lowercase()
        self.remove_spaces()
        self.replace_spaces_with_underscore()
        self.add_underscore_between_words()
        self.convert_text_to_lowercase()
        return self.df
#%%
class ExploraAnalysis():
    
    """
    Clase para análisis exploratorio de datos.
    
    Permite obtener información general, datos nulos, muestra aleatoria, estadísticas descriptivas, 
    matriz de correlación, eliminar filas duplicadas, detectar outliers utilizando el rango intercuartílico (IQR)
    e imputar valores faltantes utilizando CatBoost.
    
    """
    
    def __init__(self, data):
        
        """
        Inicializa la clase ExploraAnalysis con un dataframe de datos.

        Args:
        - data (DataFrame): El dataframe de datos.
        
        """
        
        self.data_preproc = DataPreproc(data)
        self.df = self.data_preproc.df
    
    def general_information(self):
        
        """
        Muestra información general sobre el dataframe, incluyendo tipos de datos y cantidad de valores no nulos.
        
        """        
        
        print("Información General:\n")
        self.df.info()
        
    def null_data(self):
        
        """
        Muestra la cantidad de datos nulos en cada columna del dataframe.
        """        
        
        print("Cantidad de datos nulos:\n")
        print(self.df.isnull().sum())
    
    def random_sample(self):
        
        """
        Muestra una muestra aleatoria de 30 filas del dataframe.
        """        
        
        print("Muestra Aleatoria:")
        print(self.df.sample(30))
    
    def descript_statis(self):
        """
        Muestra estadísticas descriptivas para columnas:
        - Numéricas (media, desviación, etc.)
        - Categóricas (frecuencia de valores)
        - Booleanas (incluyendo 1/0, True/False, F/V, etc.)
        """

        print("\nEstadísticas Descriptivas de Variables Numéricas:")
        print(self.df.describe(include=[float, int]))

        print("\nEstadísticas de Variables Categóricas (objetos):")
        cat_cols = self.df.select_dtypes(include='object')
        if not cat_cols.empty:
            print(cat_cols.describe())
        else:
            print("No hay columnas categóricas (tipo objeto).")

        print("\nEstadísticas de Variables Booleanas y Similares:")
        bool_like_df = self.df.copy()

        for col in bool_like_df.columns:
            if bool_like_df[col].dtype == 'object':
                bool_like_df[col] = bool_like_df[col].str.strip().str.upper()
                if bool_like_df[col].isin(['TRUE', 'FALSE', 'T', 'F', 'V']).all():
                    bool_like_df[col] = bool_like_df[col].replace({'TRUE': True, 'FALSE': False, 'T': True, 'F': False, 'V': True})

            if bool_like_df[col].dropna().nunique() == 2 and bool_like_df[col].dropna().isin([0, 1, True, False]).all():
                print(f"\nColumna booleana detectada: '{col}'")
                print(bool_like_df[col].value_counts())

        print("\nFin del resumen.")

    def corre_matri(self, plot=False, method='pearson', round_digits=2):
        """
        Calcula y muestra la matriz de correlación para columnas numéricas del DataFrame.

        Parámetros:
        - plot: bool -> Si es True, dibuja un heatmap de la matriz de correlación.
        - method: str -> Método de correlación: 'pearson', 'spearman' o 'kendall'.
        - round_digits: int -> Número de decimales para redondear la matriz.

        Requiere seaborn y matplotlib para visualizar si plot=True.
        """

        numeric_columns = self.df.select_dtypes(include='number')

        if numeric_columns.shape[1] < 2:
            print("No hay suficientes columnas numéricas para calcular correlación.")
            return

        print(f"\nMatriz de Correlación ({method.title()}):")

        correlation_matrix = numeric_columns.corr(method=method).round(round_digits)
        print(correlation_matrix)

        if plot:
            plt.figure(figsize=(10, 8))
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=f'.{round_digits}f')
            plt.title(f'Matriz de Correlación ({method.title()})')
            plt.tight_layout()
            plt.show()
    
    def show_duplicate_rows(self, subset=None):
        """
        Muestra todas las filas duplicadas del DataFrame, incluyendo todas las ocurrencias.

        Parámetros:
        - subset (list or None): Columnas que se usarán para determinar duplicados. Si None, se usan todas las columnas.
        """
        import sys
        
        subset = subset or self.df.columns.tolist()
        duplicates = self.df[self.df.duplicated(subset=subset, keep=False)]
        
        if duplicates.empty:
            print("No se encontraron filas duplicadas.")
        else:
            print(f"Se encontraron {duplicates.shape[0]} filas duplicadas según las columnas: {subset}")

            # Detectar si está en Jupyter Notebook
            try:
                shell = get_ipython().__class__.__name__
                if shell == 'ZMQInteractiveShell':
                    from IPython.display import display
                    display(duplicates)
                else:
                    print(duplicates)
            except NameError:
                # No está en Jupyter
                print(duplicates)
        
    def remove_duplicate_rows(self, subset=None, inplace=False, keep='first', show=True):
        """
        Detecta y elimina filas duplicadas del DataFrame.

        Parámetros:
        - subset (list or None): Columnas para considerar duplicados.
        - inplace (bool): Si True, elimina duplicados directamente en self.df.
        - keep (str): 'first', 'last' o False para conservar duplicado.
        - show (bool): Si True, muestra duplicados antes de eliminar.

        Retorna:
        - DataFrame sin duplicados si inplace=False.
        - None si inplace=True.
        """
        subset = subset or self.df.columns.tolist()
        duplicated_rows_all = self.df[self.df.duplicated(subset=subset, keep=False)]
        num_duplicates = duplicated_rows_all.shape[0]

        if num_duplicates > 0:
            #print(f"Se encontraron {num_duplicates} filas duplicadas según las columnas: {subset}")
            
            #if show:
            #    print("\nFilas duplicadas detectadas (todas las ocurrencias):")
            #    print(duplicated_rows_all)

            if inplace:
                self.df.drop_duplicates(subset=subset, keep=keep, inplace=True)
                print("Filas duplicadas eliminadas del DataFrame original.")
                return None
            else:
                cleaned_df = self.df.drop_duplicates(subset=subset, keep=keep)
                print("Se generó una copia del DataFrame sin duplicados.")
                return cleaned_df
        else:
            print("No se encontraron filas duplicadas.")
            return self.df if not inplace else None
        
    def is_boolean_like(self, series):
        """
        Detecta si una columna es booleana o parecida a booleano.
        Considera True/False, T/F, V/F, 1/0, 'True', 'False' (en mayúsculas o minúsculas).
        """
        s = series.dropna()
        
        if s.empty:
            return False
        
        # Valores posibles que consideramos booleanos
        bool_vals = {True, False, 1, 0, 'TRUE', 'FALSE', 'T', 'F', 'V'}
        
        # Convertir strings a mayúsculas si es object
        if s.dtype == object:
            s_upper = s.astype(str).str.strip().str.upper()
            unique_vals = set(s_upper.unique())
        else:
            unique_vals = set(s.unique())
        
        return unique_vals.issubset(bool_vals) and len(unique_vals) == 2

    def detect_outliers_iqr(self, column):
        if column not in self.df.columns:
            print(f"La columna '{column}' no existe.")
            return []
        if not pd.api.types.is_numeric_dtype(self.df[column]):
            print(f"La columna '{column}' no es numérica.")
            return []

        Q1 = self.df[column].quantile(0.25)
        Q3 = self.df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers = self.df[(self.df[column] < lower_bound) | (self.df[column] > upper_bound)]
        print(f"Se encontraron {outliers.shape[0]} outliers en la columna '{column}'.")
        return outliers
    
    def detect_rare_categories(self, column, threshold=0.01):
        if column not in self.df.columns:
            print(f"La columna '{column}' no existe.")
            return []
        if self.df[column].dtype == object or pd.api.types.is_categorical_dtype(self.df[column]):
            freq = self.df[column].value_counts(normalize=True)
            rares = freq[freq < threshold].index.tolist()
            print(f"[Rare Categories] Columna '{column}': {len(rares)} categorías raras detectadas.")
            return rares
        else:
            print(f"La columna '{column}' no es categórica.")
            return []
    
    def detect_boolean_imbalance(self, column, threshold=0.95):
        if column not in self.df.columns:
            print(f"La columna '{column}' no existe.")
            return None
        
        if not self.is_boolean_like(self.df[column]):
            print(f"La columna '{column}' no es booleana o similar.")
            return None
        
        s = self.df[column].dropna()
        
        if s.dtype == object:
            s = s.astype(str).str.strip().str.upper().replace({'TRUE': True, 'FALSE': False, 'T': True, 'F': False, 'V': True})
        
        counts = s.value_counts(normalize=True)
        max_prop = counts.max()
        
        if max_prop >= threshold:
            print(f"[Boolean Imbalance] Columna '{column}': desequilibrio detectado ({max_prop:.2f} dominante).")
            return counts.to_dict()
        else:
            print(f"[Boolean Balance] Columna '{column}': balanceada.")
            return None

    def detect_outliers_all_numeric(self):
        numeric_cols = self.df.select_dtypes(include='number').columns
        results = {}
        for col in numeric_cols:
            outliers = self.detect_outliers_iqr(col)
            # Corregido para evitar error de ambigüedad al evaluar DataFrame en if
            if not outliers.empty:
                results[col] = outliers
        if not results:
            print("No se detectaron outliers en columnas numéricas.")
        return results

    
    def detect_rare_categories_all(self, threshold=0.01):
        print("Detectando categorías raras en columnas categóricas...")
        cat_cols = self.df.select_dtypes(include='object').columns
        results = {}
        for col in cat_cols:
            rares = self.detect_rare_categories(col, threshold)
            if rares:
                results[col] = rares
        if not results:
            print("No se detectaron categorías raras en columnas categóricas.")
        return results
    
    def detect_boolean_imbalance_all(self, threshold=0.95):
        print("Detectando desequilibrio en columnas booleanas o similares...")
        bool_cols = [col for col in self.df.columns if self.is_boolean_like(self.df[col])]
        results = {}
        for col in bool_cols:
            imbalance = self.detect_boolean_imbalance(col, threshold)
            if imbalance:
                results[col] = imbalance
        if not results:
            print("No se detectaron desequilibrios en columnas booleanas.")
        return results
    
    def show_unique_values(self, show_counts=True):
        """
        Muestra los valores únicos por columna, sin importar el tipo de dato.

        Parámetros:
        - show_counts (bool): Si es True, también muestra la cantidad de ocurrencias por valor único.
        """
        print("\nValores únicos por columna:\n")
        for col in self.df.columns:
            print(f"Columna: {col}")
            unique_vals = self.df[col].dropna().unique()
            print(f"  Total únicos (sin contar NaN): {len(unique_vals)}")
            #if show_counts:
            #    print(self.df[col].value_counts(dropna=False))
            #else:
            #    print(unique_vals)
            #print("-" * 40)
    
    def run_full_detection(self):
        print("==== INICIANDO ANÁLISIS AUTOMÁTICO ====\n")
        uniques = self.show_unique_values()
        print("\n")
        outliers = self.detect_outliers_all_numeric()
        print("\n")
        rare_cats = self.detect_rare_categories_all()
        print("\n")
        bool_imbalance = self.detect_boolean_imbalance_all()
        #print("\n==== ANÁLISIS COMPLETADO ====")
        #return {
        #    "outliers": outliers,
        #    "rare_categories": rare_cats,
        #    "boolean_imbalance": bool_imbalance
        #}
    
    def impute_missing_values(self, features_columns, target_columns):
        
        """
        Imputa valores faltantes utilizando CatBoost.

        Args:
        - features_columns (list): Lista de nombres de columnas de características.
        - target_columns (list): Lista de nombres de columnas objetivo.

        Returns:
        - DataFrame: DataFrame con los valores faltantes imputados.
        """        
        
        imputed = self.df.copy()
        
        numeric_columns = imputed.select_dtypes(include=['int64', 'float64']).columns
        categorical_columns = imputed.select_dtypes(include=['object']).columns

        for target_column in target_columns:
            known_data = imputed.dropna(subset=[target_column])
            unknown_data = imputed[imputed[target_column].isnull()]

            features = known_data[features_columns]
            target = known_data[target_column]
            
            cat_feature_positions = [i for i, col in enumerate(features.columns) if col in categorical_columns]

            param_grid = {'model_size_reg': [0.1, 1]}
            grid_search = GridSearchCV(
                estimator=CatBoostClassifier(iterations=10, depth=5, learning_rate=0.1, loss_function='MultiClass', verbose=10),
                param_grid=param_grid,
                cv=5,
                verbose=10
            )
            grid_search.fit(features, target, cat_features=cat_feature_positions)

            best_model_size_reg = grid_search.best_params_['model_size_reg']

            model = CatBoostClassifier(iterations=100, depth=5, learning_rate=0.1, loss_function='MultiClass', model_size_reg=best_model_size_reg, verbose=10, random_seed=10101101)
            model.fit(features, target, cat_features=cat_feature_positions)
            
            self.model = model

            predictions_df = pd.DataFrame(model.predict(unknown_data[features_columns]), columns=[target_column], index=unknown_data.index)

            imputed.loc[unknown_data.index, target_column] = predictions_df[target_column]

        print("Valores faltantes imputados.")
        self.df = imputed
        return imputed

#%%
#Importacion librerias
import re
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

# %%
# EAM = EVALUACION AGROPEACUARIA MUNICIPALES
archivo = Path("eva_agro.csv")

if archivo.exists():
    eam = pd.read_csv(archivo)
    print("Archivo cargado correctamente.")
else:
    print("Archivo NO encontrado en:", archivo.resolve())

#%%
#Se filtra solamente Bolivar
eam_bolivar = eam[eam['Departamento']=='Bolívar']
eam_bolivar.drop(['Código Dane departamento','Departamento',\
        'Código Dane municipio','Código del cultivo',\
        'Nombre científico del cultivo','Grupo cultivo',\
        'Subgrupo','Estado físico del cultivo'],\
        axis=1, inplace=True)

# %%
#Se instancia las clase DataPreproc y ExploraAnalysis
preproce = DataPreproc(eam_bolivar)
explora = ExploraAnalysis(eam_bolivar)

#%%
#Se usa o llama la clase DataPreproc
preproce.run_all_preprocessing()

# %%
#Se usa o llama la clase ExploraAnalysis
explora.general_information()
#%%
explora.null_data()

# %%
explora.random_sample()
# %%
explora.descript_statis()
# %%
explora.corre_matri(plot=True)
# %%
explora.show_duplicate_rows()
# %%
explora.run_full_detection()
# %%
top_prductos_bolivar = eam_bolivar.groupby(by=['cultivo'])['producción'].sum().sort_values(ascending=False)
produccion_top_bolivar = eam_bolivar.groupby(by=['cultivo', 'año'])['producción'].sum().sort_values(ascending=False)
# %%
top_prductos_bolivar
# %%
produccion_top_bolivar
# %%
# analisis Cordoba
eam_cord =eam[eam['Departamento']=='Córdoba']
eam_cord.drop(['Código Dane departamento','Departamento',\
        'Código Dane municipio','Código del cultivo',\
        'Nombre científico del cultivo','Grupo cultivo',\
        'Subgrupo','Estado físico del cultivo'],\
        axis=1, inplace=True)
# %%
#Se instancia las clase DataPreproc y ExploraAnalysis
preproce = DataPreproc(eam_cord)
explora = ExploraAnalysis(eam_cord)

#%%
#Se usa o llama la clase DataPreproc
preproce.run_all_preprocessing()
# %%
#Se usa o llama la clase ExploraAnalysis
explora.general_information()
#%%
explora.null_data()

# %%
explora.descript_statis()

# %%
explora.show_duplicate_rows()
# %%
explora.run_full_detection()

# %%
top_prductos_cordoba = eam_cord.groupby(by=['cultivo'])['producción'].sum().sort_values(ascending=False)
produccion_top_cordoba = eam_cord.groupby(by=['cultivo', 'año'])['producción'].sum().sort_values(ascending=False)
# %%
top_prductos_cordoba
# %%
produccion_top_cordoba

# %%
# analisis Cordoba
eam_sucre =eam[eam['Departamento']=='Sucre']
eam_sucre.drop(['Código Dane departamento','Departamento',\
        'Código Dane municipio','Código del cultivo',\
        'Nombre científico del cultivo','Grupo cultivo',\
        'Subgrupo','Estado físico del cultivo'],\
        axis=1, inplace=True)
# %%
#Se instancia las clase DataPreproc y ExploraAnalysis
preproce = DataPreproc(eam_sucre)
explora = ExploraAnalysis(eam_sucre)

#%%
#Se usa o llama la clase DataPreproc
preproce.run_all_preprocessing()
# %%
#Se usa o llama la clase ExploraAnalysis
explora.general_information()
#%%
explora.null_data()

# %%
explora.descript_statis()

# %%
explora.show_duplicate_rows()
# %%
explora.run_full_detection()

# %%
top_prductos_sucre = eam_sucre.groupby(by=['cultivo'])['producción'].sum().sort_values(ascending=False)
produccion_top_sucre = eam_sucre.groupby(by=['cultivo', 'año'])['producción'].sum().sort_values(ascending=False)
# %%
top_prductos_sucre
# %%
produccion_top_sucre
# %%
