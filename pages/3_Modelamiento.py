import plotly.graph_objects as go
import streamlit as st
import pandas as pd
from Core.datos import serie_temporal, serie_area_agua
from Core.indices import INDICES
from Core.prediccion import predecir_serie

# ── Configuración del horizonte de proyección ───────────────────────────────
ANIOS_FUTUROS = (2026, 2027, 2028, 2029, 2030)

@st.cache_data(show_spinner=False)
def _obtener_prediccion_agua(anios_futuros: tuple = ANIOS_FUTUROS):
    serie_agua = serie_area_agua()
    return predecir_serie(
        serie_agua,
        campo_valor="Área_km2",
        campo_anio="Año",
        anios_futuros=anios_futuros,
    )

@st.cache_data(show_spinner=False)
def _obtener_prediccion_indice(indice_sel: str, anios_futuros: tuple = ANIOS_FUTUROS):
    serie = serie_temporal(indice_sel)
    return predecir_serie(
        serie,
        campo_valor="Valor",
        campo_anio="Año",
        anios_futuros=anios_futuros,
    )

# ── Interfaz – sidebar ───────────────────────────────────────────────────────
st.title("Modelamiento y Predicción – Calidad del Agua")

with st.sidebar:
    indice = st.selectbox("Índice espectral", list(INDICES.keys()))

tab_agua, tab_comparacion = st.tabs([
    "Superficie de Agua",
    "Comparación Histórica vs. Predicha",
])

# ── Tab 1 – Evolución y predicción de la superficie de agua ──────────────────
with tab_agua:
    st.subheader("Evolución y predicción de la superficie de agua (km²)")
    try:
        pred_agua = _obtener_prediccion_agua(ANIOS_FUTUROS)

        if pred_agua["historico"]:
            anios_a_hist = [d["Año"] for d in pred_agua["historico"]]
            vals_a_hist = [d["Valor"] for d in pred_agua["historico"]]
            preds_a = pred_agua["predicciones"]
            metricas_a = pred_agua["metricas"]

            fig_agua = go.Figure()

            # Histórico
            fig_agua.add_trace(
                go.Scatter(
                    x=anios_a_hist,
                    y=vals_a_hist,
                    mode="lines+markers",
                    name="Área observada (2000–2025)",
                    line=dict(color="#0288d1", width=2.5),
                    marker=dict(size=6),
                )
            )

            # Proyección
            if pred_agua["valido"] and preds_a:
                anios_ap = [anios_a_hist[-1]] + [p["Año"] for p in preds_a]
                vals_ap = [vals_a_hist[-1]] + [p["Valor"] for p in preds_a]

                anios_a_banda = [p["Año"] for p in preds_a]
                max_a_banda = [p["Max"] for p in preds_a]
                min_a_banda = [max(0.0, p["Min"]) if p.get("Min") is not None else None for p in preds_a]

                if all(m is not None for m in max_a_banda) and all(m is not None for m in min_a_banda):
                    fig_agua.add_trace(
                        go.Scatter(
                            x=anios_a_banda + anios_a_banda[::-1],
                            y=max_a_banda + min_a_banda[::-1],
                            fill="toself",
                            fillcolor="rgba(255, 127, 14, 0.2)",
                            line=dict(color="rgba(255,255,255,0)"),
                            name="Intervalo incertidumbre (95%)",
                        )
                    )

                fig_agua.add_trace(
                    go.Scatter(
                        x=anios_ap,
                        y=vals_ap,
                        mode="lines+markers",
                        name=f"Área proyectada ({preds_a[0]['Año']}–{preds_a[-1]['Año']})",
                        line=dict(color="#e65100", width=2.5, dash="dash"),
                        marker=dict(size=7, symbol="diamond"),
                    )
                )

                # Línea divisoria entre histórico y predicción
                fig_agua.add_vline(
                    x=anios_a_hist[-1] + 0.5,
                    line_width=2,
                    line_dash="dot",
                    line_color="#7f8c8d",
                    annotation_text="Límite Histórico / Futuro",
                    annotation_position="top left",
                )

                # Áreas sombreadas de fondo para diferenciar periodos
                fig_agua.add_vrect(
                    x0=anios_a_hist[0] - 0.5,
                    x1=anios_a_hist[-1] + 0.5,
                    fillcolor="rgba(31, 119, 180, 0.04)",
                    layer="below",
                    line_width=0,
                )

                fig_agua.add_vrect(
                    x0=anios_a_hist[-1] + 0.5,
                    x1=preds_a[-1]["Año"] + 0.5,
                    fillcolor="rgba(255, 127, 14, 0.12)",
                    layer="below",
                    line_width=0,
                )

            fig_agua.update_layout(
                xaxis=dict(title="Año", dtick=2),
                yaxis=dict(title="Superficie de agua (km²)"),
                hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(l=20, r=20, t=30, b=20),
            )
            st.plotly_chart(fig_agua, use_container_width=True)

            with st.expander("📊 Ver datos históricos y proyección de superficie de agua"):
                datos_agua = [
                    {
                        "Año": d["Año"],
                        "Tipo": "Observado",
                        "Superficie (km²)": d["Valor"],
                        "Límite inferior 95%": None,
                        "Límite superior 95%": None,
                    }
                    for d in pred_agua["historico"]
                ]

                if pred_agua["valido"] and preds_a:
                    datos_agua.extend([
                        {
                            "Año": p["Año"],
                            "Tipo": "Predicción",
                            "Superficie (km²)": p["Valor"],
                            "Límite inferior 95%": p.get("Min"),
                            "Límite superior 95%": p.get("Max"),
                        }
                        for p in preds_a
                    ])

                df_agua = pd.DataFrame(datos_agua)
                st.dataframe(
                    df_agua.round(4),
                    use_container_width=True,
                    hide_index=True,
                )

                st.caption(
                    "Los intervalos de incertidumbre corresponden únicamente "
                    "a las predicciones y se muestran cuando están disponibles."
                )

            if pred_agua["valido"] and metricas_a.get("MAE") is not None:
                st.caption(
                    f"**Validación modelo de superficie:** MAE = `{metricas_a.get('MAE'):.4f} km²` | "
                    f"RMSE = `{metricas_a.get('RMSE'):.4f} km²` | "
                    f"R² = `{metricas_a.get('R2'):.4f}`. "
                    "Estimación calculada a partir de la máscara de agua MNDWI (umbral 0.0) en el buffer del Río Chili."
                )
                st.caption(
                    f"Este gráfico muestra la evolución de la superficie de agua del Río Chili y estima su comportamiento "
                    f"para los próximos años a partir de los cambios detectados mediante el índice MNDWI. Permite identificar "
                    f"aumentos o disminuciones en la extensión de la superficie de agua, así como posibles cambios respecto al "
                    f"comportamiento histórico observado."
                )

            # Comparación numérica de superficie de agua
            if pred_agua.get("valido") and preds_a:
                st.markdown("---")
                st.markdown("#### Comparación de superficie de agua: Histórica vs. Proyectada")

                ultimo_a_hist = pred_agua["historico"][-1]
                area_ref = ultimo_a_hist["Valor"]
                anio_a_ref = ultimo_a_hist["Año"]

                cols_a = st.columns(len(preds_a) + 1)
                with cols_a[0]:
                    st.metric(
                        label=f"Área observada ({anio_a_ref})",
                        value=f"{area_ref:.3f} km²",
                    )
                for idx, pa in enumerate(preds_a):
                    with cols_a[idx + 1]:
                        dif_a = pa["Valor"] - area_ref
                        pct_a = (dif_a / area_ref * 100.0) if area_ref > 0 else 0.0
                        st.metric(
                            label=f"Área predicha ({pa['Año']})",
                            value=f"{pa['Valor']:.3f} km²",
                            delta=f"{dif_a:+.3f} km² ({pct_a:+.1f}%)" if area_ref > 0 else f"{dif_a:+.3f} km²",
                        )
        else:
            st.warning("No se pudieron cargar los datos históricos de superficie de agua.")
    except Exception as e:
        st.warning(f"No fue posible generar la proyección de superficie de agua: {e}")

# ── Tab 2 – Comparación histórica vs. predicha ───────────────────────────────
with tab_comparacion:
    st.subheader(f"Comparación histórica vs. predicha – {indice}")

    pred_res = _obtener_prediccion_indice(indice, ANIOS_FUTUROS)
    completos = pred_res["historico"]
    preds = pred_res["predicciones"]
    metricas = pred_res["metricas"]
    anios_hist = [d["Año"] for d in completos]
    valores_hist = [d["Valor"] for d in completos]

    if completos and pred_res["valido"] and preds:
        ultimo_hist = completos[-1]
        val_ref = ultimo_hist["Valor"]
        anio_ref = ultimo_hist["Año"]
        m_anual = metricas.get("Pendiente_anual", 0.0)

        # 1. Gráfico comparativo con separación visual explícita
        fig_comp = go.Figure()

        # Zona histórica
        fig_comp.add_trace(
            go.Scatter(
                x=anios_hist,
                y=valores_hist,
                mode="lines+markers",
                name=f"Histórico ({anios_hist[0]}–{anio_ref})",
                line=dict(color="#1f77b4", width=2.5),
                marker=dict(size=6, symbol="circle"),
            )
        )

        # Banda de incertidumbre (únicamente si el modelo seleccionado provee intervalos analíticos)
        anios_banda = [p["Año"] for p in preds]
        max_banda = [p["Max"] for p in preds]
        min_banda = [p["Min"] for p in preds]

        if all(m is not None for m in max_banda) and all(m is not None for m in min_banda):
            fig_comp.add_trace(
                go.Scatter(
                    x=anios_banda + anios_banda[::-1],
                    y=max_banda + min_banda[::-1],
                    fill="toself",
                    fillcolor="rgba(255, 127, 14, 0.2)",
                    line=dict(color="rgba(255,255,255,0)"),
                    name="Incertidumbre proyectada (95%)",
                )
            )

        # Zona proyectada unida a la última observación
        fig_comp.add_trace(
            go.Scatter(
                x=[anio_ref] + [p["Año"] for p in preds],
                y=[val_ref] + [p["Valor"] for p in preds],
                mode="lines+markers",
                name=f"Predicción ({preds[0]['Año']}–{preds[-1]['Año']})",
                line=dict(color="#ff7f0e", width=2.5, dash="dash"),
                marker=dict(size=8, symbol="diamond"),
            )
        )

        # Línea divisoria entre histórico y predicción
        fig_comp.add_vline(
            x=anio_ref + 0.5,
            line_width=2,
            line_dash="dot",
            line_color="#7f8c8d",
            annotation_text="Límite Histórico / Futuro",
            annotation_position="top left",
        )

        # Áreas sombreadas de fondo para diferenciar periodos
        fig_comp.add_vrect(
            x0=anios_hist[0] - 0.5,
            x1=anio_ref + 0.5,
            fillcolor="rgba(31, 119, 180, 0.04)",
            layer="below",
            line_width=0,
        )
        fig_comp.add_vrect(
            x0=anio_ref + 0.5,
            x1=preds[-1]["Año"] + 0.5,
            fillcolor="rgba(255, 127, 14, 0.06)",
            layer="below",
            line_width=0,
        )

        fig_comp.update_layout(
            xaxis=dict(title="Año", dtick=2),
            yaxis=dict(title=f"Valor de {indice}"),
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=20, r=20, t=30, b=20),
        )
        st.plotly_chart(fig_comp, use_container_width=True)

        # 2. Comparación numérica (Tabla y Tarjetas)
        st.markdown(f"#### Tabla de variación proyectada respecto a {anio_ref} (Valor base: `{val_ref:.4f}`)")

        filas_comparacion = []
        for p in preds:
            dif_abs = p["Valor"] - val_ref
            if abs(val_ref) > 1e-4:
                var_pct = (dif_abs / abs(val_ref)) * 100.0
                var_pct_str = f"{var_pct:+.2f}%"
            else:
                var_pct_str = "N/D (base ~0)"

            if p.get("Min") is not None and p.get("Max") is not None:
                rango_str = f"[{p['Min']:.4f} a {p['Max']:.4f}]"
            else:
                rango_str = "No disponible"

            filas_comparacion.append({
                "Año": p["Año"],
                "Tipo": "Predicción",
                "Valor Proyectado": f"{p['Valor']:.4f}",
                "Rango 95%": rango_str,
                "Diferencia Absoluta": f"{dif_abs:+.4f}",
                "Variación (%)": var_pct_str,
            })

        st.dataframe(filas_comparacion, use_container_width=True, hide_index=True)

        # 3. Tendencia e Interpretación automática
        if m_anual > 0.0005:
            tendencia_txt = "tendencia creciente sostenida"
            dinamica_txt = "un incremento gradual en los valores esperados"
        elif m_anual < -0.0005:
            tendencia_txt = "tendencia decreciente sostenida"
            dinamica_txt = "una disminución progresiva en los valores esperados"
        else:
            tendencia_txt = "comportamiento aproximadamente estable"
            dinamica_txt = "variaciones mínimas cercanas al promedio histórico"

        anio_fin = preds[-1]["Año"]
        val_fin = preds[-1]["Valor"]
        min_fin = preds[-1]["Min"]
        max_fin = preds[-1]["Max"]

        col_met1, col_met2, col_met3 = st.columns(3)
        with col_met1:
            st.metric(
                label=f"Último observado ({anio_ref})",
                value=f"{val_ref:.4f}",
            )
        with col_met2:
            dif_fin = val_fin - val_ref
            pct_fin = (dif_fin / abs(val_ref) * 100) if abs(val_ref) > 1e-4 else 0.0
            st.metric(
                label=f"Proyección final ({anio_fin})",
                value=f"{val_fin:.4f}",
                delta=f"{dif_fin:+.4f} ({pct_fin:+.1f}%)" if abs(val_ref) > 1e-4 else f"{dif_fin:+.4f}",
            )
        with col_met3:
            st.metric(
                label="Tasa de tendencia anual",
                value=f"{m_anual:+.5f} / año",
                help="Pendiente anual estimada por el modelo de mínimos cuadrados.",
            )

        r2_str = f"{metricas.get('R2'):.4f}" if metricas.get('R2') is not None else "N/D"
        mae_str = f"{metricas.get('MAE'):.5f}" if metricas.get('MAE') is not None else "N/D"
        rmse_str = f"{metricas.get('RMSE'):.5f}" if metricas.get('RMSE') is not None else "N/D"
        modelo_sel = metricas.get('Modelo', 'N/D')

        if min_fin is not None and max_fin is not None:
            incertidumbre_txt = f"El intervalo al 95% para {anio_fin} se sitúa entre `{min_fin:.4f}` y `{max_fin:.4f}`."
        else:
            incertidumbre_txt = "Intervalo analítico no aplicable para el modelo seleccionado."

        st.markdown(
            f"""
            **Interpretación de la proyección:**
            - **Comportamiento histórico:** La serie observada (2000–{anio_ref}) presenta una {tendencia_txt}, con una tasa media de `{m_anual:+.5f}` por año.
            - **Comportamiento proyectado:** Para el periodo {preds[0]['Año']}–{anio_fin}, el modelo **{modelo_sel}** proyecta la continuidad de este comportamiento, estimando {dinamica_txt} hasta alcanzar `{val_fin:.4f}` en {anio_fin}.
            - **Nivel de incertidumbre:** {incertidumbre_txt}
            - **Evaluación del ajuste:** MAE = `{mae_str}`, RMSE = `{rmse_str}`, R² = `{r2_str}`.

            > *Nota:* Estas proyecciones representan una extrapolación estadística del comportamiento histórico del índice espectral en el cauce y su entorno ripario. No constituyen una inferencia directa de calidad química o bacteriológica del agua.
            """
        )

        # Tabla de evaluación de modelos
        modelos_ev = pred_res.get("modelos_evaluados", [])
        if modelos_ev:
            with st.expander("Comparación de modelos evaluados", expanded=False):
                st.caption(
                    pred_res.get(
                        "mensaje_seleccion",
                        f"Modelo seleccionado: **{modelo_sel}** (menor MAE en validación rolling-origin)."
                    )
                )
                filas_mod = []
                for m in modelos_ev:
                    r2_m = f"{m['R2']:.4f}" if m.get('R2') is not None else "N/D"
                    mae_m = f"{m['MAE']:.5f}" if m.get('MAE') is not None else "N/D"
                    rmse_m = f"{m['RMSE']:.5f}" if m.get('RMSE') is not None else "N/D"
                    splits_m = str(m.get('Splits', '-'))
                    filas_mod.append({
                        "Modelo": m["Modelo"],
                        "MAE": mae_m,
                        "RMSE": rmse_m,
                        "R²": r2_m,
                        "Splits": splits_m,
                        "Seleccionado": "✅" if m["Modelo"] == modelo_sel else "",
                    })
                st.dataframe(filas_mod, use_container_width=True, hide_index=True)
    else:
        st.info("No hay información suficiente para generar la comparación histórica vs. predicha.")
