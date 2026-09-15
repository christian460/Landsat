import json
import os

import ee
import streamlit as st


# ============================================================
# CONFIGURACIÓN DE GEE
# ============================================================

PROJECT_ID = "landsat-aguas"

ASSET_ZONA_ESTUDIO = (
    "projects/landsat-aguas/assets/uchumayo"
)

ASSET_RIO_CHILI = (
    "projects/landsat-aguas/assets/rio_chili_uchumayo"
)


# ============================================================
# INICIALIZACIÓN DE GOOGLE EARTH ENGINE
# ============================================================

def inicializar_gee():
    """Inicializa Google Earth Engine usando credenciales locales o OAuth2."""

    try:
        project = "landsat-aguas"

        # 1. Intentar primero las credenciales locales existentes
        try:
            ee.Initialize(project=project)
            return
        except Exception:
            pass

        # 2. Si no funcionan, intentar OAuth2 mediante variables de entorno
        client_id = (
            os.getenv("EE_CLIENT_ID")
            or os.getenv("CLIENT_ID")
        )

        client_secret = (
            os.getenv("EE_CLIENT_SECRET")
            or os.getenv("CLIENT_SECRET")
        )

        refresh_token = (
            os.getenv("EE_REFRESH_TOKEN")
            or os.getenv("REFRESH_TOKEN")
        )

        # 3. Si no existen variables, intentar Streamlit Secrets
        if not all([client_id, client_secret, refresh_token]):
            client_id = st.secrets.get("EE_CLIENT_ID")
            client_secret = st.secrets.get("EE_CLIENT_SECRET")
            refresh_token = st.secrets.get("EE_REFRESH_TOKEN")

        if client_id and client_secret and refresh_token:
            credentials = {
                "client_id": client_id,
                "client_secret": client_secret,
                "refresh_token": refresh_token,
                "type": "authorized_user"
            }

            cred_dir = os.path.join(
                os.path.expanduser("~"),
                ".config",
                "earthengine"
            )

            os.makedirs(cred_dir, exist_ok=True)

            with open(os.path.join(cred_dir, "credentials"), "w") as f:
                json.dump(credentials, f)

            ee.Initialize(project=project)
            return

        raise RuntimeError("No se encontraron credenciales de Google Earth Engine.")

    except Exception as e:
        raise RuntimeError(f"Error inicializando GEE: {e}")



# ============================================================
# ZONA DE ESTUDIO
# ============================================================

@st.cache_resource
def obtener_zona_estudio():
    """Carga el asset de la zona de estudio de Uchumayo."""

    return ee.FeatureCollection(
        ASSET_ZONA_ESTUDIO
    )


def asegurar_zona_estudio():
    """Obtiene la zona de estudio desde session_state o GEE."""

    if "zona_estudio" not in st.session_state:
        st.session_state["zona_estudio"] = (
            obtener_zona_estudio()
        )

    return st.session_state["zona_estudio"]


# ============================================================
# RÍO CHILI
# ============================================================

@st.cache_resource
def obtener_rio_chili():
    """Carga el asset correspondiente al Río Chili."""

    return ee.FeatureCollection(
        ASSET_RIO_CHILI
    )


def asegurar_rio_chili():
    """Obtiene el Río Chili desde session_state o GEE."""

    if "rio_chili" not in st.session_state:
        st.session_state["rio_chili"] = (
            obtener_rio_chili()
        )

    return st.session_state["rio_chili"]