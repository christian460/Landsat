import streamlit as st
from Core.gee_init import inicializar_gee, obtener_zona_estudio, obtener_rio_chili

# SOLO el archivo principal tiene st.set_page_config
st.set_page_config(
    page_title="Monitoreo de la Calidad del Agua – Río Chili",
    layout="wide",
)

# Inicializar GEE primero
try:
    inicializar_gee()
except Exception as e:
    st.error(f"Error al inicializar Google Earth Engine: {str(e)}")
    st.info("Por favor, verifica tus credenciales de GEE en las variables de entorno.")
    st.stop()

# Luego obtener la zona de estudio
if "zona_estudio" not in st.session_state:
    try:
        st.session_state["zona_estudio"] = obtener_zona_estudio()
    except Exception as e:
        st.error(f"Error al cargar la zona de estudio: {str(e)}")
        st.info("Verifica que el asset 'projects/landsat-aguas/assets/uchumayo' exista y sea accesible.")
        st.stop()

if "rio_chili" not in st.session_state:
    try:
        st.session_state["rio_chili"] = obtener_rio_chili()
    except Exception as e:
        st.error(f"Error al cargar el río Chili: {str(e)}")
        st.info("Verifica que el asset 'projects/landsat-aguas/assets/rio_chili' exista y sea accesible.")
        st.stop()

# ===============================
# PÁGINA DE INICIO
# ===============================
st.title("Sistema de Monitoreo de la Calidad del Agua – Río Chili")

st.markdown("""
## Bienvenido al Sistema de Monitoreo

Este sistema permite analizar la evolución espacial y
temporal de las características espectrales del agua
del **Río Chili en el distrito de Uchumayo, Arequipa**,
utilizando imágenes satelitales Landsat y Google Earth Engine.

El análisis busca identificar cambios en la respuesta
espectral del río que puedan estar asociados con procesos
de variación o degradación ambiental.

## Funcionalidades disponibles:

**Exploración Espacial**
Permite visualizar espacialmente el Río Chili mediante
imágenes Landsat y analizar diferentes índices espectrales
relacionados con la presencia y comportamiento del agua.

**Análisis Multitemporal**
- Compara 3 años diferentes simultáneamente
- Visualiza series temporales (2000-2025)
- Analiza anomalías y tendencias
- Estadísticas por periodo

## Índices disponibles:
- **NDVI** - Índice de Vegetación Normalizado
- **SAVI** - Índice de Vegetación Ajustado al Suelo
- **EVI** - Índice de Vegetación Mejorado
- **GNDVI** - Índice Verde Normalizado
- **LSWI** - Índice de Agua en Onda Corta
- **NDWI** - Índice de Agua Normalizado
- **MNDWI** - Índice de Agua Modificado

También permite visualizar:

- Río Chili
- Zona de estudio
- Puntos de monitoreo
- Máscara de agua
---
""")

# ============================================================
# INFORMACIÓN DEL PROYECTO
# ============================================================

st.subheader("Área y datos de estudio")


col1, col2, col3 = st.columns(3)


with col1:

    st.metric(
        "Área de estudio",
        "Uchumayo"
    )


with col2:

    st.metric(
        "Elemento hidrográfico",
        "Río Chili"
    )


with col3:

    st.metric(
        "Fuente principal",
        "Landsat"
    )


# ============================================================
# ESTADO DEL SISTEMA
# ============================================================

st.subheader("Estado del sistema")


col1, col2 = st.columns(2)


with col1:

    st.success(
        "Zona de estudio de Uchumayo cargada correctamente"
    )


with col2:

    st.success(
        "Río Chili cargado correctamente"
    )
