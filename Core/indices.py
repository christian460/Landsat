import ee

INDICES = {
    "NDVI":  lambda img: img.normalizedDifference(["NIR", "RED"]),

    "SAVI":  lambda img: img.expression(
        "(NIR - RED) / (NIR + RED + 0.5) * 1.5",
        {"NIR": img.select("NIR"), "RED": img.select("RED")},
    ),

    "EVI":   lambda img: img.expression(
        "2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1))",
        {
            "NIR":  img.select("NIR"),
            "RED":  img.select("RED"),
            "BLUE": img.select("BLUE"),
        },
    ),

    "GNDVI": lambda img: img.normalizedDifference(["NIR", "GREEN"]),
    "LSWI":  lambda img: img.normalizedDifference(["NIR", "SWIR1"]),
    "NDWI":  lambda img: img.normalizedDifference(["GREEN", "NIR"]),
    "MNDWI": lambda img: img.normalizedDifference(["GREEN", "SWIR1"]),
    "AWEI":  lambda img: img.expression(
        "4 * (GREEN - SWIR1) - (0.25 * NIR + 2.75 * SWIR2)",
        {
            "GREEN": img.select("GREEN"),
            "NIR":   img.select("NIR"),
            "SWIR1": img.select("SWIR1"),
            "SWIR2": img.select("SWIR2"),
        },
    ),
}

# ── Parámetros de visualización para folium / GEE ────────────────────────────
VIS_PARAMS = {
    "NDVI":  {"min": -0.2, "max": 0.9, "palette": ["brown", "yellow", "green"]},
    "SAVI":  {"min": -0.2, "max": 0.9, "palette": ["brown", "yellow", "green"]},
    "EVI":   {"min": -0.2, "max": 0.9, "palette": ["brown", "yellow", "green"]},
    "GNDVI": {"min": -0.2, "max": 0.9, "palette": ["brown", "yellow", "green"]},
    "LSWI":  {"min": -0.5, "max": 0.8, "palette": ["brown", "white", "blue"]},
    "NDWI":  {"min": -0.5, "max": 0.8, "palette": ["white", "cyan", "blue"]},
    "MNDWI": {"min": -0.5, "max": 0.8, "palette": ["white", "lightblue", "darkblue"]},
    "AWEI":  {"min": -1.5, "max": 0.5, "palette": ["white", "cyan", "darkblue"]},
}


def calcular_indice(img, nombre):
    """Calcula un índice espectral dado una imagen con bandas renombradas."""
    if nombre not in INDICES:
        raise ValueError(f"Índice no soportado: {nombre}")
    return INDICES[nombre](img)


def calcular_todos_indices(img):
    """Calcula los 7 índices espectrales y los combina como bandas de una sola imagen."""
    bandas = [INDICES[nombre](img).rename(nombre) for nombre in INDICES]
    return ee.Image.cat(bandas)

