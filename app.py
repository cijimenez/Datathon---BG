# Author: Carlos Jimenez 
# Date: 2024-12-07

import streamlit as st
import shap
import pandas as pd
import matplotlib.pyplot as plt

# Carga de datos y el modelo (aquí agregas tu propio dataset y modelo)
df = pd.read_parquet(r'Datathon---BG\df_final_estudiantes.parquet') # Cargar el conjunto de datos

# Filtrar los datos de los NO COLOCADOS (COLOCADO == 0)
df_no_colocados = df[df['COLOCADO'] == 0]

# Mostrar las columnas de interés para los NO COLOCADOS
df_no_colocados_subset = df_no_colocados[['FECHA_CARGA', 'TIPO_APROBACION', 'CANAL', 'IVC', 'CUPO', 'SEGMENTO']]
## print(df_no_colocados_subset.head())


# VARIABLES CATEGÓRICAS
# Convertir variables categóricas en variables dummy (One-Hot Encoding)
df_no_colocados_encoded = pd.get_dummies(df_no_colocados[['TIPO_APROBACION', 'CANAL', 'SEGMENTO']], drop_first=True)

# Unir el DataFrame codificado con las demás variables numéricas
df_no_colocados_final = pd.concat([df_no_colocados[['IVC', 'CUPO', 'FECHA_CARGA']], df_no_colocados_encoded], axis=1)

# Convertir FECHA_CARGA a tipo datetime y extraer componentes relevantes (como mes y año)
df_no_colocados_final['FECHA_CARGA'] = pd.to_datetime(df_no_colocados_final['FECHA_CARGA'])
df_no_colocados_final['MES'] = df_no_colocados_final['FECHA_CARGA'].dt.month
df_no_colocados_final['AÑO'] = df_no_colocados_final['FECHA_CARGA'].dt.year

# Eliminar la columna original FECHA_CARGA si ya no es necesaria
df_no_colocados_final = df_no_colocados_final.drop(columns=['FECHA_CARGA'])

## print(df_no_colocados_final)


from sklearn.preprocessing import StandardScaler
# Estandarizar las variables numéricas
scaler = StandardScaler()
df_no_colocados_final[['CUPO', 'IVC']] = scaler.fit_transform(df_no_colocados_final[['CUPO', 'IVC']])

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Definir las variables dependientes e independientes
X = df_no_colocados_final
y = df_no_colocados['COLOCADO']  # Aquí usas la columna COLOCADO para predecir

# Dividir en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar el modelo Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


import shap

# Crear un explicador SHAP para el modelo
explainer = shap.TreeExplainer(model)

# Obtener los valores SHAP para el conjunto de prueba
shap_values = explainer.shap_values(X_test)

# Visualizar el resumen de la importancia de las características
# Visualizar el resumen de la importancia de las características para la clase 0 (NO COLOCADOS)
# Visualizar el resumen de la importancia de las características para ambas clases
# shap.summary_plot(shap_values, X_test)


# model = RandomForestClassifier()  # Tu modelo entrenado
# model.fit(X_train, y_train)  # Ajustar el modelo (ejemplo)

# Título de la app
st.title("Análisis de SHAP - Predicción de NO COLOCADOS")

# Explicación sobre lo que hace la app
st.write("Este panel muestra los gráficos de importancia de características para los individuos NO COLOCADOS.")

# Generar gráficos de dependencia
for feature in X_test.columns:
    st.subheader(f"Gráfico de dependencia para {feature}")
    shap.dependence_plot(feature, shap_values, X_test)
    st.pyplot()  # Muestra el gráfico generado por SHAP en la aplicación
