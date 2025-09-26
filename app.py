import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Configuración de la página
st.set_page_config(
    page_title="Titanic Gaussian Analysis",
    page_icon="🚢",
    layout="wide"
)

# Título principal
st.title("🚢 Análisis Gaussiano del Titanic")
st.markdown("---")

# Sidebar
st.sidebar.header("Configuración")
st.sidebar.markdown("Proyecto de análisis del dataset Titanic usando distribuciones gaussianas")

# Generar datos de ejemplo (simulando datos del Titanic)
@st.cache_data
def generate_sample_data():
    np.random.seed(42)
    n_samples = 1000
    
    # Simular datos del Titanic
    data = {
        'Age': np.random.normal(30, 12, n_samples),
        'Fare': np.random.lognormal(3, 1, n_samples),
        'Survived': np.random.choice([0, 1], n_samples, p=[0.62, 0.38]),
        'Pclass': np.random.choice([1, 2, 3], n_samples, p=[0.24, 0.21, 0.55])
    }
    
    # Ajustar supervivencia por clase
    for i in range(n_samples):
        if data['Pclass'][i] == 1:
            data['Survived'][i] = np.random.choice([0, 1], p=[0.37, 0.63])
        elif data['Pclass'][i] == 2:
            data['Survived'][i] = np.random.choice([0, 1], p=[0.53, 0.47])
        else:
            data['Survived'][i] = np.random.choice([0, 1], p=[0.76, 0.24])
    
    return pd.DataFrame(data)

# Cargar datos
df = generate_sample_data()

# Layout principal
col1, col2 = st.columns(2)

with col1:
    st.header("📊 Datos Generales")
    st.dataframe(df.head(10))
    
    st.subheader("Estadísticas Descriptivas")
    st.dataframe(df.describe())

with col2:
    st.header("📈 Análisis Gaussiano - Edad")
    
    # Filtrar datos válidos
    age_data = df['Age'].dropna()
    
    # Ajustar distribución normal
    mu, sigma = stats.norm.fit(age_data)
    
    # Crear gráfico
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Histograma
    ax.hist(age_data, bins=30, density=True, alpha=0.7, color='skyblue', edgecolor='black')
    
    # Distribución gaussiana ajustada
    x = np.linspace(age_data.min(), age_data.max(), 100)
    y = stats.norm.pdf(x, mu, sigma)
    ax.plot(x, y, 'r-', linewidth=2, label=f'Gaussiana (μ={mu:.2f}, σ={sigma:.2f})')
    
    ax.set_xlabel('Edad')
    ax.set_ylabel('Densidad')
    ax.set_title('Distribución de Edad vs Ajuste Gaussiano')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    st.pyplot(fig)

# Análisis por supervivencia
st.header("🔍 Análisis por Supervivencia")

col3, col4 = st.columns(2)

with col3:
    st.subheader("Distribución de Edad por Supervivencia")
    
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    
    # Separar por supervivencia
    survived = df[df['Survived'] == 1]['Age'].dropna()
    not_survived = df[df['Survived'] == 0]['Age'].dropna()
    
    # Histogramas superpuestos
    ax2.hist(survived, bins=20, alpha=0.6, label='Supervivientes', color='green', density=True)
    ax2.hist(not_survived, bins=20, alpha=0.6, label='No supervivientes', color='red', density=True)
    
    # Ajustar gaussianas
    mu_s, sigma_s = stats.norm.fit(survived)
    mu_ns, sigma_ns = stats.norm.fit(not_survived)
    
    x = np.linspace(df['Age'].min(), df['Age'].max(), 100)
    y_s = stats.norm.pdf(x, mu_s, sigma_s)
    y_ns = stats.norm.pdf(x, mu_ns, sigma_ns)
    
    ax2.plot(x, y_s, 'g-', linewidth=2, label=f'Gaussiana Supervivientes (μ={mu_s:.2f})')
    ax2.plot(x, y_ns, 'r-', linewidth=2, label=f'Gaussiana No supervivientes (μ={mu_ns:.2f})')
    
    ax2.set_xlabel('Edad')
    ax2.set_ylabel('Densidad')
    ax2.set_title('Comparación de Distribuciones de Edad')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    st.pyplot(fig2)

with col4:
    st.subheader("Test de Normalidad")
    
    # Test de Shapiro-Wilk
    shapiro_survived = stats.shapiro(survived[:5000] if len(survived) > 5000 else survived)
    shapiro_not_survived = stats.shapiro(not_survived[:5000] if len(not_survived) > 5000 else not_survived)
    
    # Test de Kolmogorov-Smirnov
    ks_statistic, p_value = stats.ks_2samp(survived, not_survived)
    
    st.write("**Test de Shapiro-Wilk:**")
    st.write(f"- Supervivientes: p-value = {shapiro_survived.pvalue:.4f}")
    st.write(f"- No supervivientes: p-value = {shapiro_not_survived.pvalue:.4f}")
    
    st.write("**Test de Kolmogorov-Smirnov:**")
    st.write(f"- Estadístico KS: {ks_statistic:.4f}")
    st.write(f"- p-value: {p_value:.4f}")
    
    if p_value < 0.05:
        st.write("🔴 Las distribuciones son significativamente diferentes")
    else:
        st.write("🟢 No hay diferencia significativa entre las distribuciones")
    
    st.subheader("Parámetros Gaussianos")
    
    params_df = pd.DataFrame({
        'Grupo': ['Supervivientes', 'No supervivientes'],
        'Media (μ)': [mu_s, mu_ns],
        'Desviación (σ)': [sigma_s, sigma_ns],
        'Varianza (σ²)': [sigma_s**2, sigma_ns**2]
    })
    
    st.dataframe(params_df)

# Footer
st.markdown("---")
st.markdown("**Entorno:** `titanic_streamlit` | **Python:** 3.11.13 | **Streamlit:** 1.50.0")

# Información del entorno en sidebar
st.sidebar.markdown("---")
st.sidebar.subheader("ℹ️ Info del Entorno")
st.sidebar.text(f"Python: 3.11.13")
st.sidebar.text(f"Streamlit: 1.50.0")
st.sidebar.text(f"Pandas: {pd.__version__}")
st.sidebar.text(f"NumPy: {np.__version__}")
st.sidebar.success("✅ Entorno configurado correctamente")

#& "C:\Users\franp\anaconda3\envs\titanic_streamlit\python.exe" -m streamlit run "C:\Users\franp\Desktop\M2\data-science-tutorials\Titanic_Gauss_Streamlit\app.py"