import os
import streamlit as st
import ee
from google.oauth2.credentials import Credentials


def inicializar_gee():
    """Inicializa Google Earth Engine con credenciales OAuth2."""
    try:
        # Obtener credenciales desde Streamlit Secrets
        client_id = (
            os.getenv("EE_CLIENT_ID")
            or os.getenv("CLIENT_ID")
            or st.secrets["EE_CLIENT_ID"]
        )

        client_secret = (
            os.getenv("EE_CLIENT_SECRET")
            or os.getenv("CLIENT_SECRET")
            or st.secrets["EE_CLIENT_SECRET"]
        )

        refresh_token = (
            os.getenv("EE_REFRESH_TOKEN")
            or os.getenv("REFRESH_TOKEN")
            or st.secrets["EE_REFRESH_TOKEN"]
        )

        # Proyecto de Google Cloud / Earth Engine
        project = st.secrets.get("EE_PROJECT", "landsat-aguas")

        if not client_id or not client_secret or not refresh_token:
            raise RuntimeError(
                "Faltan credenciales OAuth2 de Earth Engine."
            )

        # Crear credenciales OAuth2 directamente
        credentials = Credentials(
            token=None,
            refresh_token=refresh_token,
            client_id=client_id,
            client_secret=client_secret,
            token_uri="https://oauth2.googleapis.com/token",
            scopes=[
                "https://www.googleapis.com/auth/earthengine"
            ]
        )

        # Inicializar Earth Engine usando explícitamente
        # las credenciales OAuth2
        ee.Initialize(
            credentials=credentials,
            project=project,
            opt_url="https://earthengine.googleapis.com"
        )

    except Exception as e:
        raise RuntimeError(
            f"Error inicializando GEE: {e}"
        )


def obtener_zona_estudio():
    """Obtiene la geometría de la zona de estudio desde GEE"""
    try:
        return ee.FeatureCollection(
            "projects/landsat-aguas/assets/uchumayo"
        ).geometry()
    except Exception as e:
        raise RuntimeError(f"Error al cargar zona de estudio: {e}")


def asegurar_zona_estudio():
    """
    Asegura que la zona de estudio esté cargada en session_state.
    Llama a esta función al inicio de cada página de Streamlit.
    """
    if "zona_estudio" not in st.session_state:
        try:
            inicializar_gee()
            st.session_state["zona_estudio"] = obtener_zona_estudio()
        except Exception as e:
            st.error(f"Error al inicializar el sistema: {str(e)}")
            st.stop()

    return st.session_state["zona_estudio"]