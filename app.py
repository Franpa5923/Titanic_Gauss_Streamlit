import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
# Comentamos plotly por si no está instalado
# import plotly.express as px
# import plotly.graph_objects as go
# from plotly.subplots import make_subplots

# ==========================================
# CONFIGURACIÓN Y FUNCIONES DE UTILIDAD
# ==========================================

def configure_page():
    """Configuración inicial de la página de Streamlit"""
    st.set_page_config(
        page_title="Titanic Gaussian Analysis",
        page_icon="🚢",
        layout="wide",
        initial_sidebar_state="expanded"
    )

def load_titanic_data():
    """
    Carga y prepara el dataset real del Titanic desde seaborn
    """
    # Cargar dataset del Titanic
    titanic = sns.load_dataset('titanic')
    
    # Preparar datos para análisis
    df = titanic.copy()
    
    # Convertir columnas categóricas a numéricas si es necesario
    df['survived'] = df['survived'].astype(int)
    df['pclass'] = df['pclass'].astype(int)
    
    # Renombrar para consistencia
    df = df.rename(columns={
        'survived': 'Survived',
        'pclass': 'Pclass',
        'age': 'Age',
        'fare': 'Fare',
        'sex': 'Sex',
        'embarked': 'Embarked',
        'sibsp': 'SibSp',
        'parch': 'Parch'
    })
    
    return df

# ==========================================
# FUNCIONES DE ANÁLISIS ESTADÍSTICO
# ==========================================

def perform_normality_tests(data, name="Data"):
    """
    Realiza múltiples tests de normalidad en los datos
    """
    results = {}
    
    # Shapiro-Wilk test (máximo 5000 muestras)
    sample_size = min(len(data), 5000)
    sample_data = data.sample(sample_size) if len(data) > sample_size else data
    
    shapiro_stat, shapiro_p = stats.shapiro(sample_data)
    results['shapiro'] = {'statistic': shapiro_stat, 'p_value': shapiro_p}
    
    # Kolmogorov-Smirnov test con distribución normal teórica
    mu, sigma = stats.norm.fit(data)
    ks_stat, ks_p = stats.kstest(data, lambda x: stats.norm.cdf(x, mu, sigma))
    results['ks'] = {'statistic': ks_stat, 'p_value': ks_p}
    
    # Anderson-Darling test
    ad_stat, ad_critical, ad_significance = stats.anderson(data, dist='norm')
    results['anderson'] = {
        'statistic': ad_stat, 
        'critical_values': ad_critical,
        'significance_levels': ad_significance
    }
    
    return results

def fit_gaussian_parameters(data):
    """
    Ajusta parámetros gaussianos y calcula métricas adicionales
    """
    mu, sigma = stats.norm.fit(data)
    
    return {
        'mean': mu,
        'std': sigma,
        'variance': sigma**2,
        'skewness': stats.skew(data),
        'kurtosis': stats.kurtosis(data),
        'median': np.median(data),
        'mode': stats.mode(data, keepdims=True)[0][0] if len(data) > 0 else np.nan
    }

# ==========================================
# FUNCIONES DE VISUALIZACIÓN
# ==========================================

def create_gaussian_comparison_plot(data, title="Distribución vs Ajuste Gaussiano"):
    """
    Crea un gráfico comparando histograma con ajuste gaussiano
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Filtrar datos válidos
    clean_data = data.dropna()
    
    # Histograma
    n, bins, patches = ax.hist(clean_data, bins=30, density=True, alpha=0.7, 
                              color='skyblue', edgecolor='black', label='Datos observados')
    
    # Ajustar distribución normal
    mu, sigma = stats.norm.fit(clean_data)
    
    # Distribución gaussiana teórica
    x = np.linspace(clean_data.min(), clean_data.max(), 100)
    y_norm = stats.norm.pdf(x, mu, sigma)
    ax.plot(x, y_norm, 'r-', linewidth=3, label=f'Gaussiana (μ={mu:.2f}, σ={sigma:.2f})')
    
    # Añadir líneas de referencia
    ax.axvline(mu, color='red', linestyle='--', alpha=0.8, label=f'Media: {mu:.2f}')
    ax.axvline(mu + sigma, color='orange', linestyle='--', alpha=0.6, label=f'μ+σ: {mu+sigma:.2f}')
    ax.axvline(mu - sigma, color='orange', linestyle='--', alpha=0.6, label=f'μ-σ: {mu-sigma:.2f}')
    
    ax.set_xlabel('Valores', fontsize=12)
    ax.set_ylabel('Densidad', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    return fig

def create_qq_plot(data, title="Q-Q Plot"):
    """
    Crea un Q-Q plot para evaluar normalidad
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    clean_data = data.dropna()
    stats.probplot(clean_data, dist="norm", plot=ax)
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    return fig

def create_interactive_violin_plot(df):
    """
    Crea un violin plot con matplotlib (Plotly comentado)
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Datos para violin plot
    survived_ages = df[df['Survived'] == 1]['Age'].dropna()
    not_survived_ages = df[df['Survived'] == 0]['Age'].dropna()
    
    # Violin plot con matplotlib
    data_to_plot = [survived_ages, not_survived_ages]
    parts = ax.violinplot(data_to_plot, positions=[1, 2], showmeans=True, showmedians=True)
    
    # Colorear los violines
    parts['bodies'][0].set_facecolor('lightgreen')
    parts['bodies'][1].set_facecolor('lightcoral')
    
    ax.set_xticks([1, 2])
    ax.set_xticklabels(['Supervivientes', 'No Supervivientes'])
    ax.set_ylabel('Edad')
    ax.set_title('Distribución de Edad por Supervivencia (Violin Plot)')
    ax.grid(True, alpha=0.3)
    
    return fig

def create_correlation_heatmap(df):
    """
    Crea un heatmap de correlaciones
    """
    # Seleccionar solo columnas numéricas
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    corr_matrix = df[numeric_cols].corr()
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='RdBu_r', center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": .8}, ax=ax)
    
    ax.set_title('Matriz de Correlación - Dataset Titanic', fontsize=14, fontweight='bold')
    
    return fig

def create_scatter_plot(df):
    """
    Crea un scatter plot 2D con matplotlib
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    df_clean = df.dropna(subset=['Age', 'Fare'])
    
    # Scatter plot por supervivencia
    survived = df_clean[df_clean['Survived'] == 1]
    not_survived = df_clean[df_clean['Survived'] == 0]
    
    ax.scatter(survived['Age'], survived['Fare'], c='green', alpha=0.6, 
              label='Supervivientes', s=50)
    ax.scatter(not_survived['Age'], not_survived['Fare'], c='red', alpha=0.6, 
              label='No Supervivientes', s=50)
    
    ax.set_xlabel('Edad')
    ax.set_ylabel('Tarifa')
    ax.set_title('Relación Edad vs Tarifa por Supervivencia')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig

# ==========================================
# FUNCIÓN PRINCIPAL DE LA APLICACIÓN
# ==========================================

def main():
    """Función principal que ejecuta toda la aplicación"""
    configure_page()
    
    # Título principal
    st.title("🚢 Análisis Gaussiano Avanzado del Titanic")
    st.markdown("---")
    
    # Sidebar
    st.sidebar.header("🎛️ Configuración")
    st.sidebar.markdown("Análisis del **dataset real del Titanic** usando distribuciones gaussianas y visualizaciones avanzadas")

    # Cargar datos
    df = load_titanic_data()
    
    # Actualizar sidebar con información del dataset
    st.sidebar.info(f"📊 **{len(df)} pasajeros** en el dataset original")
    st.sidebar.success("✅ Datos cargados desde seaborn")
    
    # Controles interactivos en sidebar
    st.sidebar.markdown("### 🎯 Filtros de Análisis")
    analysis_type = st.sidebar.selectbox(
        "Selecciona el tipo de análisis:",
        ["Edad", "Tarifa", "Ambos"]
    )
    
    show_advanced = st.sidebar.checkbox("Mostrar análisis avanzados", value=True)
    show_interactive = st.sidebar.checkbox("Gráficos interactivos", value=True)
    
    # ==========================================
    # SECCIÓN 1: INFORMACIÓN BÁSICA
    # ==========================================
    
    st.subheader("📋 Información del Dataset")
    col_info1, col_info2, col_info3, col_info4, col_info5 = st.columns(5)
    
    with col_info1:
        st.metric("Total Pasajeros", len(df))
    with col_info2:
        st.metric("Supervivientes", df['Survived'].sum(), 
                 delta=f"{df['Survived'].mean():.1%}")
    with col_info3:
        st.metric("Edad Promedio", f"{df['Age'].mean():.1f} años", 
                 delta=f"±{df['Age'].std():.1f}")
    with col_info4:
        st.metric("Tarifa Promedio", f"${df['Fare'].mean():.1f}", 
                 delta=f"±${df['Fare'].std():.1f}")
    with col_info5:
        st.metric("Datos Faltantes", 
                 f"{df.isnull().sum().sum()}", 
                 delta=f"{df.isnull().sum().sum()/len(df):.1%}")
    
    st.markdown("---")
    
    # ==========================================
    # SECCIÓN 2: VISTA GENERAL DE DATOS
    # ==========================================
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📊 Vista General de Datos")
        
        # Tabla interactiva con filtros
        if st.checkbox("Mostrar solo supervivientes"):
            display_df = df[df['Survived'] == 1]
        else:
            display_df = df
            
        st.dataframe(display_df.head(15), use_container_width=True)
    
    with col2:
        st.header("📈 Estadísticas Rápidas")
        
        # Estadísticas por clase
        st.subheader("Por Clase")
        class_stats = df.groupby('Pclass')['Survived'].agg(['count', 'sum', 'mean'])
        class_stats.columns = ['Total', 'Supervivientes', 'Tasa']
        class_stats['Tasa'] = class_stats['Tasa'].apply(lambda x: f"{x:.1%}")
        st.dataframe(class_stats)
        
        # Estadísticas por sexo
        st.subheader("Por Sexo")
        sex_stats = df.groupby('Sex')['Survived'].agg(['count', 'sum', 'mean'])
        sex_stats.columns = ['Total', 'Supervivientes', 'Tasa']
        sex_stats['Tasa'] = sex_stats['Tasa'].apply(lambda x: f"{x:.1%}")
        st.dataframe(sex_stats)
    
    st.markdown("---")
    
    # ==========================================
    # SECCIÓN 3: ANÁLISIS GAUSSIANO PRINCIPAL
    # ==========================================
    
    if analysis_type in ["Edad", "Ambos"]:
        st.header("📈 Análisis Gaussiano - Edad")
        
        col3, col4 = st.columns(2)
        
        with col3:
            # Gráfico gaussiano para edad
            age_data = df['Age'].dropna()
            fig_age = create_gaussian_comparison_plot(age_data, "Distribución de Edad vs Ajuste Gaussiano")
            st.pyplot(fig_age)
            
        with col4:
            # Q-Q Plot para edad
            if show_advanced:
                fig_qq_age = create_qq_plot(age_data, "Q-Q Plot - Edad")
                st.pyplot(fig_qq_age)
            
            # Parámetros gaussianos
            age_params = fit_gaussian_parameters(age_data)
            st.subheader("📊 Parámetros Estadísticos")
            
            param_df = pd.DataFrame({
                'Métrica': ['Media', 'Desviación Estándar', 'Varianza', 'Asimetría', 'Curtosis', 'Mediana'],
                'Valor': [
                    f"{age_params['mean']:.2f}",
                    f"{age_params['std']:.2f}",
                    f"{age_params['variance']:.2f}",
                    f"{age_params['skewness']:.3f}",
                    f"{age_params['kurtosis']:.3f}",
                    f"{age_params['median']:.2f}"
                ]
            })
            st.dataframe(param_df, use_container_width=True)
    
    if analysis_type in ["Tarifa", "Ambos"]:
        st.header("💰 Análisis Gaussiano - Tarifa")
        
        col5, col6 = st.columns(2)
        
        with col5:
            # Gráfico gaussiano para tarifa (log-transform)
            fare_data = df['Fare'].dropna()
            fare_data_log = np.log1p(fare_data)  # log(1+x) para evitar log(0)
            
            fig_fare = create_gaussian_comparison_plot(fare_data_log, "Distribución de Log(Tarifa) vs Ajuste Gaussiano")
            st.pyplot(fig_fare)
            
        with col6:
            if show_advanced:
                fig_qq_fare = create_qq_plot(fare_data_log, "Q-Q Plot - Log(Tarifa)")
                st.pyplot(fig_qq_fare)
            
            # Parámetros gaussianos para tarifa
            fare_params = fit_gaussian_parameters(fare_data_log)
            st.subheader("📊 Parámetros Log(Tarifa)")
            
            fare_param_df = pd.DataFrame({
                'Métrica': ['Media', 'Desviación Estándar', 'Varianza', 'Asimetría', 'Curtosis'],
                'Valor': [
                    f"{fare_params['mean']:.2f}",
                    f"{fare_params['std']:.2f}",
                    f"{fare_params['variance']:.2f}",
                    f"{fare_params['skewness']:.3f}",
                    f"{fare_params['kurtosis']:.3f}"
                ]
            })
            st.dataframe(fare_param_df, use_container_width=True)

    st.markdown("---")
    
    # ==========================================
    # SECCIÓN 4: ANÁLISIS POR SUPERVIVENCIA
    # ==========================================
    
    st.header("🔍 Análisis Comparativo por Supervivencia")
    
    col7, col8 = st.columns(2)
    
    with col7:
        st.subheader("Distribución de Edad por Supervivencia")
        
        # Separar por supervivencia
        survived = df[df['Survived'] == 1]['Age'].dropna()
        not_survived = df[df['Survived'] == 0]['Age'].dropna()
        
        fig2, ax2 = plt.subplots(figsize=(12, 8))
        
        # Histogramas superpuestos con mejor visualización
        ax2.hist(survived, bins=25, alpha=0.6, label='Supervivientes', 
                color='green', density=True, edgecolor='darkgreen')
        ax2.hist(not_survived, bins=25, alpha=0.6, label='No supervivientes', 
                color='red', density=True, edgecolor='darkred')
        
        # Ajustar gaussianas
        mu_s, sigma_s = stats.norm.fit(survived)
        mu_ns, sigma_ns = stats.norm.fit(not_survived)
        
        x = np.linspace(df['Age'].min(), df['Age'].max(), 100)
        y_s = stats.norm.pdf(x, mu_s, sigma_s)
        y_ns = stats.norm.pdf(x, mu_ns, sigma_ns)
        
        ax2.plot(x, y_s, 'g-', linewidth=3, label=f'Gaussiana Supervivientes (μ={mu_s:.1f})')
        ax2.plot(x, y_ns, 'r-', linewidth=3, label=f'Gaussiana No supervivientes (μ={mu_ns:.1f})')
        
        # Líneas de referencia
        ax2.axvline(mu_s, color='green', linestyle='--', alpha=0.8)
        ax2.axvline(mu_ns, color='red', linestyle='--', alpha=0.8)
        
        ax2.set_xlabel('Edad', fontsize=12)
        ax2.set_ylabel('Densidad', fontsize=12)
        ax2.set_title('Comparación de Distribuciones de Edad', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        st.pyplot(fig2)
    
    with col8:
        st.subheader("Tests Estadísticos")
        
        # Realizar tests de normalidad
        survived_tests = perform_normality_tests(survived, "Supervivientes")
        not_survived_tests = perform_normality_tests(not_survived, "No Supervivientes")
        
        # Test de Kolmogorov-Smirnov para comparar distribuciones
        ks_statistic, ks_p_value = stats.ks_2samp(survived, not_survived)
        
        # Mostrar resultados
        st.write("**🧪 Test de Shapiro-Wilk (Normalidad):**")
        st.write(f"- Supervivientes: p = {survived_tests['shapiro']['p_value']:.4f}")
        st.write(f"- No supervivientes: p = {not_survived_tests['shapiro']['p_value']:.4f}")
        
        st.write("**🔬 Test de Kolmogorov-Smirnov (Comparación):**")
        st.write(f"- Estadístico KS: {ks_statistic:.4f}")
        st.write(f"- p-value: {ks_p_value:.4f}")
        
        if ks_p_value < 0.05:
            st.error("🔴 Las distribuciones son significativamente diferentes")
        else:
            st.success("🟢 No hay diferencia significativa entre las distribuciones")
        
        # T-test para comparar medias
        t_stat, t_p_value = stats.ttest_ind(survived, not_survived)
        st.write(f"**📊 T-test (Comparación de medias):**")
        st.write(f"- Estadístico t: {t_stat:.4f}")
        st.write(f"- p-value: {t_p_value:.4f}")
        
        st.subheader("📋 Parámetros Gaussianos Comparados")
        
        params_comparison = pd.DataFrame({
            'Métrica': ['Media (μ)', 'Desviación (σ)', 'Varianza (σ²)', 'Asimetría', 'Curtosis'],
            'Supervivientes': [
                f"{mu_s:.2f}",
                f"{sigma_s:.2f}",
                f"{sigma_s**2:.2f}",
                f"{stats.skew(survived):.3f}",
                f"{stats.kurtosis(survived):.3f}"
            ],
            'No Supervivientes': [
                f"{mu_ns:.2f}",
                f"{sigma_ns:.2f}",
                f"{sigma_ns**2:.2f}",
                f"{stats.skew(not_survived):.3f}",
                f"{stats.kurtosis(not_survived):.3f}"
            ],
            'Diferencia': [
                f"{abs(mu_s - mu_ns):.2f}",
                f"{abs(sigma_s - sigma_ns):.2f}",
                f"{abs(sigma_s**2 - sigma_ns**2):.2f}",
                f"{abs(stats.skew(survived) - stats.skew(not_survived)):.3f}",
                f"{abs(stats.kurtosis(survived) - stats.kurtosis(not_survived)):.3f}"
            ]
        })
        
        st.dataframe(params_comparison, use_container_width=True)
    
    # ==========================================
    # SECCIÓN 5: VISUALIZACIONES AVANZADAS
    # ==========================================
    
    if show_advanced:
        st.markdown("---")
        st.header("🎨 Visualizaciones Avanzadas")
        
        col9, col10 = st.columns(2)
        
        with col9:
            st.subheader("🎻 Distribución por Supervivencia (Violin Plot)")
            fig_violin = create_interactive_violin_plot(df)
            st.pyplot(fig_violin)
        
        with col10:
            st.subheader("🌡️ Matriz de Correlación")
            fig_corr = create_correlation_heatmap(df)
            st.pyplot(fig_corr)
        
        # Gráfico 2D 
        if show_interactive:
            st.subheader("📊 Análisis Edad vs Tarifa")
            fig_scatter = create_scatter_plot(df)
            st.pyplot(fig_scatter)
    
    # ==========================================
    # SECCIÓN 6: ANÁLISIS POR CATEGORÍAS
    # ==========================================
    
    st.markdown("---")
    st.header("📂 Análisis por Categorías")
    
    # Selector de categoría
    category = st.selectbox(
        "Selecciona una categoría para análisis detallado:",
        ["Clase (Pclass)", "Sexo", "Puerto de Embarque"]
    )
    
    if category == "Clase (Pclass)":
        st.subheader("🎭 Análisis por Clase Social")
        
        col11, col12 = st.columns(2)
        
        with col11:
            # Gráfico por clase
            fig_class, axes = plt.subplots(1, 3, figsize=(18, 6))
            
            for i, pclass in enumerate([1, 2, 3]):
                class_ages = df[df['Pclass'] == pclass]['Age'].dropna()
                if len(class_ages) > 0:
                    axes[i].hist(class_ages, bins=15, alpha=0.7, color=f'C{i}', density=True)
                    
                    mu_class, sigma_class = stats.norm.fit(class_ages)
                    x_class = np.linspace(class_ages.min(), class_ages.max(), 100)
                    y_class = stats.norm.pdf(x_class, mu_class, sigma_class)
                    axes[i].plot(x_class, y_class, 'r-', linewidth=2)
                    
                    axes[i].set_title(f'Clase {pclass}\n(μ={mu_class:.1f}, σ={sigma_class:.1f})')
                    axes[i].set_xlabel('Edad')
                    if i == 0:
                        axes[i].set_ylabel('Densidad')
                    axes[i].grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig_class)
        
        with col12:
            # Estadísticas por clase
            class_stats_detailed = df.groupby('Pclass').agg({
                'Age': ['count', 'mean', 'std', 'min', 'max'],
                'Survived': ['sum', 'mean'],
                'Fare': ['mean', 'std']
            }).round(2)
            
            st.subheader("📊 Estadísticas Detalladas por Clase")
            st.dataframe(class_stats_detailed)
    
    elif category == "Sexo":
        st.subheader("👥 Análisis por Sexo")
        
        col13, col14 = st.columns(2)
        
        with col13:
            # Análisis por sexo
            fig_sex, axes = plt.subplots(1, 2, figsize=(15, 6))
            
            for i, sex in enumerate(['male', 'female']):
                sex_ages = df[df['Sex'] == sex]['Age'].dropna()
                if len(sex_ages) > 0:
                    axes[i].hist(sex_ages, bins=20, alpha=0.7, 
                               color='blue' if sex == 'male' else 'pink', density=True)
                    
                    mu_sex, sigma_sex = stats.norm.fit(sex_ages)
                    x_sex = np.linspace(sex_ages.min(), sex_ages.max(), 100)
                    y_sex = stats.norm.pdf(x_sex, mu_sex, sigma_sex)
                    axes[i].plot(x_sex, y_sex, 'r-', linewidth=2)
                    
                    axes[i].set_title(f'{sex.capitalize()}\n(μ={mu_sex:.1f}, σ={sigma_sex:.1f})')
                    axes[i].set_xlabel('Edad')
                    if i == 0:
                        axes[i].set_ylabel('Densidad')
                    axes[i].grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig_sex)
        
        with col14:
            # Estadísticas por sexo
            sex_stats_detailed = df.groupby('Sex').agg({
                'Age': ['count', 'mean', 'std'],
                'Survived': ['sum', 'mean'],
                'Fare': ['mean', 'std']
            }).round(2)
            
            st.subheader("📊 Estadísticas por Sexo")
            st.dataframe(sex_stats_detailed)

    # ==========================================
    # SECCIÓN 7: INSIGHTS Y CONCLUSIONES
    # ==========================================
    
    st.markdown("---")
    st.header("🎯 Insights y Conclusiones")
    
    col15, col16 = st.columns(2)
    
    with col15:
        st.subheader("🔍 Hallazgos Principales")
        
        # Calcular insights automáticamente
        age_diff = abs(mu_s - mu_ns)
        survival_rate = df['Survived'].mean()
        
        insights = [
            f"📊 **Tasa de supervivencia general:** {survival_rate:.1%}",
            f"🎂 **Diferencia de edad promedio:** {age_diff:.1f} años entre grupos",
            f"🚢 **Clase más segura:** {df.groupby('Pclass')['Survived'].mean().idxmax()}ª clase ({df.groupby('Pclass')['Survived'].mean().max():.1%} supervivencia)",
            f"👥 **Sexo con mayor supervivencia:** {df.groupby('Sex')['Survived'].mean().idxmax()} ({df.groupby('Sex')['Survived'].mean().max():.1%})",
            f"📈 **Asimetría de edad:** {'Positiva' if stats.skew(df['Age'].dropna()) > 0 else 'Negativa'} ({stats.skew(df['Age'].dropna()):.2f})"
        ]
        
        for insight in insights:
            st.write(insight)
    
    with col16:
        st.subheader("📋 Recomendaciones")
        
        recommendations = [
            "🔬 **Análisis estadístico:** Los datos de edad no siguen perfectamente una distribución normal",
            "📊 **Modelado:** Considerar transformaciones logarítmicas para variables asimétricas",
            "🎯 **Factores clave:** Sexo y clase social son predictores fuertes de supervivencia",
            "⚠️ **Datos faltantes:** Manejar cuidadosamente los 177 valores faltantes de edad",
            "🔍 **Análisis futuro:** Explorar interacciones entre edad, clase y sexo"
        ]
        
        for rec in recommendations:
            st.write(rec)
    
    # ==========================================
    # FOOTER E INFORMACIÓN TÉCNICA
    # ==========================================
    
    st.markdown("---")
    
    # Información técnica en expander
    with st.expander("🔧 Información Técnica"):
        col17, col18 = st.columns(2)
        
        with col17:
            st.write("**📦 Librerías utilizadas:**")
            st.write(f"- Streamlit: {st.__version__}")
            st.write(f"- Pandas: {pd.__version__}")
            st.write(f"- NumPy: {np.__version__}")
            st.write("- Matplotlib: instalado")
            st.write(f"- Seaborn: {sns.__version__}")
            st.write("- SciPy: instalado")
        
        with col18:
            st.write("**🎛️ Funcionalidades:**")
            st.write("- Análisis gaussiano avanzado")
            st.write("- Tests de normalidad múltiples")
            st.write("- Visualizaciones interactivas")
            st.write("- Análisis comparativo por grupos")
            st.write("- Métricas estadísticas completas")
    
    # Footer principal
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center'>
            <p><strong>🚢 Análisis Gaussiano del Titanic</strong></p>
            <p>Dataset: Titanic real (seaborn) | Entorno: titanic_streamlit | Python 3.11.13</p>
            <p><em>Desarrollado con ❤️ usando Streamlit</em></p>
        </div>
        """, 
        unsafe_allow_html=True
    )
    
    # Información del entorno en sidebar
    st.sidebar.markdown("---")
    st.sidebar.subheader("ℹ️ Info del Entorno")
    st.sidebar.text(f"Python: 3.11.13")
    st.sidebar.text(f"Streamlit: {st.__version__}")
    st.sidebar.text(f"Pandas: {pd.__version__}")
    st.sidebar.text(f"NumPy: {np.__version__}")
    st.sidebar.success("✅ Entorno configurado correctamente")
    
    # Botón de ayuda
    st.sidebar.markdown("---")
    if st.sidebar.button("❓ Ayuda"):
        st.sidebar.info(
            """
            **Cómo usar esta aplicación:**
            
            1. 📊 Explora las estadísticas generales
            2. 📈 Revisa los análisis gaussianos
            3. 🔍 Compara supervivientes vs no supervivientes  
            4. 🎨 Interactúa con las visualizaciones avanzadas
            5. 📂 Analiza por categorías específicas
            6. 🎯 Lee los insights y conclusiones
            """
        )

# ==========================================
# EJECUCIÓN PRINCIPAL
# ==========================================

if __name__ == "__main__":
    main()

# Comando para ejecutar:
# streamlit run app.py