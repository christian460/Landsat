import os
import streamlit as st
import ee
import json


def inicializar_gee():
    """Inicializa Google Earth Engine usando credenciales locales, variables de entorno o Streamlit Secrets."""
    project = "landsat-aguas"

    # 1. Intentar primero las credenciales locales existentes de la máquina/servidor
    try:
        ee.Initialize(project=project)
        return
    except Exception:
        pass  # Si falla, procedemos a buscar credenciales explícitas

    try:
        # Función auxiliar para buscar en entorno o en Streamlit Secrets de forma segura
        def buscar_credencial(key_env, key_alt):
            # 1. Buscar en variables de entorno de la máquina
            valor = os.getenv(key_env) or os.getenv(key_alt)
            if valor:
                return valor
            
            # 2. Si no está en el entorno, buscar en Streamlit Secrets de forma segura
            try:
                return st.secrets.get(key_env) or st.secrets.get(key_alt)
            except Exception:
                return None

        # Obtener las credenciales con prioridad: Variables Entorno > Streamlit Secrets
        client_id = buscar_credencial("EE_CLIENT_ID", "CLIENT_ID")
        client_secret = buscar_credencial("EE_CLIENT_SECRET", "CLIENT_SECRET")
        refresh_token = buscar_credencial("EE_REFRESH_TOKEN", "REFRESH_TOKEN")

        # 2. Si tenemos las tres piezas, creamos el archivo de configuración de GEE
        if client_id and client_secret and refresh_token:
            credentials = {
                "client_id": client_id,
                "client_secret": client_secret,
                "refresh_token": refresh_token,
                "type": "authorized_user"
            }

            cred_dir = os.path.join(os.path.expanduser("~"), ".config", "earthengine")
            os.makedirs(cred_dir, exist_ok=True)

            with open(os.path.join(cred_dir, "credentials"), "w") as f:
                json.dump(credentials, f)

            ee.Initialize(project=project)
            return

        raise RuntimeError("No se encontraron credenciales válidas en el entorno ni en Secrets.")

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