import ee
import streamlit as st
import pandas as pd

from Core.gee_init import asegurar_zona_estudio
from Core.indices import INDICES, calcular_todos_indices


# ── Selector de colección según año ─────────────────────────────────────────

def _coleccion_y_bandas(anio: int):
    if anio <= 2011:
        return (
            ee.ImageCollection("LANDSAT/LE07/C02/T1_L2"),
            ["SR_B1", "SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B7"],
        )
    if anio == 2012:
        col = ee.ImageCollection("LANDSAT/LT05/C02/T1_L2").merge(
            ee.ImageCollection("LANDSAT/LE07/C02/T1_L2")
        )
        return col, ["SR_B1", "SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B7"]

    return (
        ee.ImageCollection("LANDSAT/LC08/C02/T1_L2"),
        ["SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7"],
    )


# ── Imagen base anual ────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def _imagen_base(anio: int):
    """Construye y almacena en caché la imagen base Landsat procesada para un año."""
    zona_estudio = asegurar_zona_estudio()
    coleccion, bandas_origen = _coleccion_y_bandas(anio)

    return (
        coleccion
        .filterDate(f"{anio}-01-01", f"{anio}-12-31")
        .filterBounds(zona_estudio)
        .filter(ee.Filter.lt("CLOUD_COVER", 20))
        .median()
        .select(bandas_origen)
        .rename(["BLUE", "GREEN", "RED", "NIR", "SWIR1", "SWIR2"])
        .clip(zona_estudio)
    )


# ── Imagen multibanda de los 7 índices ──────────────────────────────────────

@st.cache_data(show_spinner=False)
def _imagen_indices(anio: int):
    """Construye y almacena en caché la imagen multibanda con los 7 índices espectrales."""
    imagen = _imagen_base(anio)
    return calcular_todos_indices(imagen)


# ── Imagen de un índice para un año ─────────────────────────────────────────

@st.cache_data(show_spinner=False)
def obtener_indice(anio: int, indice: str):
    """Obtiene la imagen de un índice espectral reutilizando la imagen multibanda anual."""
    return _imagen_indices(anio).select(indice)


# ── Estadísticas agrupadas de los 7 índices para un año ──────────────────────

@st.cache_data(show_spinner=False)
def estadisticas_todos_indices(anio: int):
    """Calcula las estadísticas para los 7 índices en una sola operación reduceRegion."""
    zona_estudio = asegurar_zona_estudio()
    img_indices = _imagen_indices(anio)

    stats = img_indices.reduceRegion(
        reducer=(
            ee.Reducer.mean()
            .combine(ee.Reducer.min(), "", True)
            .combine(ee.Reducer.max(), "", True)
        ),
        geometry=zona_estudio,
        scale=30,
        maxPixels=1e9,
    )
    return stats.getInfo()


@st.cache_data(show_spinner=False)
def estadisticas_indice(anio: int, indice: str):
    """Calcula o recupera del caché las estadísticas para un índice en un año dado."""
    return estadisticas_todos_indices(anio)


# ── Serie temporal unificada ──────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def _serie_temporal_todos(inicio: int = 2000, fin: int = 2025):
    """Calcula la serie temporal de los 7 índices en una sola llamada GEE."""
    zona_estudio = asegurar_zona_estudio()

    def reducir_anio(anio: int):
        img_indices = _imagen_indices(anio)
        red = img_indices.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=zona_estudio,
            scale=30,
            maxPixels=1e9,
        )
        return ee.Feature(None, red.set("Año", anio))

    fc = ee.FeatureCollection([reducir_anio(a) for a in range(inicio, fin + 1)])
    features = fc.getInfo()["features"]

    res = {}
    for f in features:
        props = f["properties"]
        anio = int(props["Año"])
        res[anio] = props
    return res


@st.cache_data(show_spinner=False)
def serie_temporal(indice: str, inicio: int = 2000, fin: int = 2025):
    """Calcula la serie temporal de un índice reutilizando los datos multitemporales de todos los índices."""
    datos_completos = _serie_temporal_todos(inicio, fin)
    return [
        {
            "Año": anio,
            "Valor": datos_completos.get(anio, {}).get(indice),
        }
        for anio in range(inicio, fin + 1)
    ]


# ── Tabla de puntos de muestreo desde Google Sheets ─────────────────────────

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

