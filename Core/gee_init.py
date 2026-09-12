import os
import streamlit as st
import ee
from google.oauth2.credentials import Credentials


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