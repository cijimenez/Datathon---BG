import streamlit as st
import shap
import pandas as pd
import matplotlib.pyplot as plt

# Asegúrate de haber cargado previamente X_test y shap_values

# Título de la app
st.title("Análisis de SHAP - Predicción de NO COLOCADOS")

# Explicación sobre lo que hace la app
st.write("Este panel muestra los gráficos de importancia de características para los individuos NO COLOCADOS.")

# Generar gráficos de dependencia
for feature in X_test.columns:
    st.subheader(f"Gráfico de dependencia para {feature}")
    shap.dependence_plot(feature, shap_values, X_test)
    st.pyplot()  # Muestra el gráfico generado por SHAP en la aplicación
