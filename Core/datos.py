import ee
import streamlit as st
import pandas as pd

from Core.gee_init import asegurar_zona_estudio, asegurar_rio_chili
from Core.indices import INDICES, calcular_todos_indices

# ============================================================
# CONFIGURACIÓN
# ============================================================

CLOUD_COVER_MAX = 20

BUFFER_RIO_METROS = 150

ESCALA_LANDSAT = 30

# ── Selector de colección según año ─────────────────────────────────────────

def _coleccion_y_bandas(anio: int):
    if anio <= 2012:
        col = ee.ImageCollection("LANDSAT/LT05/C02/T1_L2").merge(
            ee.ImageCollection("LANDSAT/LE07/C02/T1_L2")
        )
        return col, ["SR_B1", "SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B7"]
    if anio >= 2022:
        col = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2").merge(
            ee.ImageCollection("LANDSAT/LC09/C02/T1_L2")
        )
        return col, ["SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7"]

    return (
        ee.ImageCollection("LANDSAT/LC08/C02/T1_L2"),
        ["SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7"],
    )

# ============================================================
# GEOMETRÍA DEL RÍO
# ============================================================

@st.cache_resource
def _geometria_rio():
    rio_chili = asegurar_rio_chili()

    return rio_chili.geometry().buffer(BUFFER_RIO_METROS)

# ============================================================
# IMAGEN BASE LANDSAT
# ============================================================

@st.cache_data(show_spinner=False)
def _imagen_base(anio: int):
    geometria_rio = _geometria_rio()
    coleccion, bandas_origen = _coleccion_y_bandas(anio)

    imagen = (
        coleccion
        .filterDate(
            f"{anio}-01-01",
            f"{anio}-12-31"
        )
        .filterBounds(
            geometria_rio
        )
        .filter(
            ee.Filter.lt(
                "CLOUD_COVER",
                CLOUD_COVER_MAX
            )
        )
        .median()
        .select(
            bandas_origen
        )
        .rename(
            [
                "BLUE",
                "GREEN",
                "RED",
                "NIR",
                "SWIR1",
                "SWIR2",
            ]
        )
        .multiply(0.0000275)
        .add(-0.2)
        .clip(
            geometria_rio
        )
    )

    return imagen


# ============================================================
# IMAGEN MULTIBANDA DE ÍNDICES
# ============================================================

@st.cache_data(show_spinner=False)
def _imagen_indices(anio: int):
    """Construye y almacena en caché la imagen multibanda con los 7 índices espectrales."""
    imagen = _imagen_base(anio)
    return calcular_todos_indices(imagen)


# ============================================================
# OBTENER UN ÍNDICE
# ============================================================

@st.cache_data(show_spinner=False)
def obtener_indice(anio: int, indice: str):
    """Obtiene la imagen de un índice espectral reutilizando la imagen multibanda anual."""
    return _imagen_indices(anio).select(indice)

# ============================================================
# MÁSCARA DE AGUA
# ============================================================

@st.cache_data(show_spinner=False)
def obtener_mascara_agua(
    anio: int,
    indice: str = "MNDWI",
    umbral: float = 0.0
):

    imagen = obtener_indice(
        anio,
        indice
    )

    mascara = (
        imagen
        .gt(umbral)
        .rename("water_mask")
    )

    return mascara

# ============================================================
# ESTADÍSTICAS DE TODOS LOS ÍNDICES
# ============================================================

@st.cache_data(show_spinner=False)
def estadisticas_todos_indices(anio: int):
    geometria_rio = _geometria_rio()
    imagen = _imagen_indices(anio)

    estadisticas = imagen.reduceRegion(
        reducer=(
            ee.Reducer.mean()
            .combine(
                ee.Reducer.min(),
                "",
                True
            )
            .combine(
                ee.Reducer.max(),
                "",
                True
            )
        ),
        geometry=geometria_rio,
        scale=ESCALA_LANDSAT,
        maxPixels=1e9,
    )

    return estadisticas.getInfo()
    
# ============================================================
# ESTADÍSTICAS DE UN ÍNDICE
# ============================================================

@st.cache_data(show_spinner=False)
def estadisticas_indice(
    anio: int,
    indice: str
):
    """
    Devuelve las estadísticas del índice solicitado.

    Se reutiliza el cálculo conjunto de todos los índices.
    """

    datos = estadisticas_todos_indices(
        anio
    )

    return {
        "mean": datos.get(
            f"{indice}_mean"
        ),
        "min": datos.get(
            f"{indice}_min"
        ),
        "max": datos.get(
            f"{indice}_max"
        ),
    }

# ============================================================
# ÁREA DE AGUA
# ============================================================

@st.cache_data(show_spinner=False)
def area_agua(anio: int, indice: str = "MNDWI", umbral: float = 0.0):
    geometria_rio = _geometria_rio()

    mascara = obtener_mascara_agua(
        anio,
        indice,
        umbral
    )

    area = (
        mascara
        .selfMask()
        .multiply(
            ee.Image.pixelArea()
        )
        .reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=geometria_rio,
            scale=ESCALA_LANDSAT,
            maxPixels=1e9,
        )
    )

    resultado = area.getInfo()

    if not resultado:
        return 0.0

    area_m2 = resultado.get("water_mask", 0)

    if area_m2 is None:
        return 0.0

    return area_m2 / 1_000_000

# ============================================================
# SERIE TEMPORAL DE TODOS LOS ÍNDICES
# ============================================================

@st.cache_data(show_spinner=False)
def _serie_temporal_todos(inicio: int = 2000, fin: int = 2025):
    geometria_rio = _geometria_rio()

    def reducir_anio(anio: int):
        imagen = _imagen_indices(
            anio
        )

        estadisticas = imagen.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=geometria_rio,
            scale=ESCALA_LANDSAT,
            maxPixels=1e9,
        )

        return ee.Feature(None,estadisticas.set("Año",anio))

    features = [reducir_anio(anio) for anio in range(inicio,fin + 1)]

    coleccion = ee.FeatureCollection(features)

    datos = coleccion.getInfo()

    resultado = {}

    for feature in datos["features"]:
        propiedades = feature["properties"]
        anio = int(propiedades["Año"])

        resultado[anio] = propiedades

    return resultado


@st.cache_data(show_spinner=False)
def serie_temporal(indice: str, inicio: int = 2000, fin: int = 2025):
    """Calcula la serie temporal de un índice reutilizando los datos multitemporales de todos los índices."""
    datos = _serie_temporal_todos(inicio, fin)
    return [
        {
            "Año": anio,
            "Valor": datos.get(anio, {}).get(indice),
        }
        for anio in range(inicio, fin + 1)
    ]

# ============================================================
# SERIE TEMPORAL DEL ÁREA DE AGUA
# ============================================================

@st.cache_data(show_spinner=False)
def serie_area_agua(inicio: int = 2000,fin: int = 2025,indice: str = "MNDWI",umbral: float = 0.0):
    return [
        {
            "Año": anio,
            "Área_km2": area_agua(anio,indice,umbral),
        }
        for anio in range(inicio,fin + 1)
    ]


# ============================================================
# DATOS DE MUESTREO
# ============================================================

_URL_SHEETS = (
    "https://docs.google.com/spreadsheets/d/"
    "1yQ3TJRpGAGqSnSfGgQP4c9UwwDt-RZwS/export?format=xlsx"
)

_COLUMNAS = ["Punto", "Profundidad", "Fertilidad", "pH", "Humedad", "Temperatura"]


@st.cache_data(ttl=300, show_spinner=False)
def cargar_tabla_muestreo(url: str = _URL_SHEETS) -> pd.DataFrame:
    df = pd.read_excel(url)
    # Renombrar por posición (los encabezados del Sheet pueden variar)
    nuevas = list(_COLUMNAS) + list(df.columns[len(_COLUMNAS):])
    df.columns = nuevas
    df["Punto"] = df["Punto"].ffill()
    df = df.dropna(subset=["Profundidad"])
    df = (
        df.groupby("Punto", as_index=False)
        .head(2)
        .reset_index(drop=True)
    )
    return df

