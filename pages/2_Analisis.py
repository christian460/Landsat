import folium
import plotly.graph_objects as go
import streamlit as st
import pandas as pd
from streamlit_folium import st_folium

from Core.datos import estadisticas_indice, obtener_indice, serie_temporal
from Core.gee_init import asegurar_zona_estudio, asegurar_rio_chili
from Core.indices import INDICES, VIS_PARAMS
from Core.prediccion import predecir_serie, calcular_anomalias_proyectadas

# ── Contexto ─────────────────────────────────────────────────────────────────
zona_estudio = asegurar_zona_estudio()
rio_chili = asegurar_rio_chili()

# ── Interfaz – sidebar ───────────────────────────────────────────────────────
st.title("Análisis Multitemporal – Calidad del Agua")

with st.sidebar:
    indice = st.selectbox("Índice espectral", list(INDICES.keys()))
    anios_sel = [
        st.selectbox("Año 1", range(2000, 2026), index=23),
        st.selectbox("Año 2", range(2000, 2026), index=20),
        st.selectbox("Año 3", range(2000, 2026), index=17),
    ]
    opacity = st.slider("Opacidad", 0.0, 1.0, 0.6, 0.1)

# Serie temporal (caché compartido con 1_Exploracion si ya se calculó)
serie = serie_temporal(indice)

tab_mapas, tab_graficos = st.tabs(["Mapas y estadísticas", "Gráficos Analíticos"])

# ── Tab 1 – Mapas ─────────────────────────────────────────────────────────────
with tab_mapas:
    cols = st.columns(3)
    for i, (col, anio) in enumerate(zip(cols, anios_sel)):
        with col:
            st.subheader(f"{indice} – {anio}")
            img = obtener_indice(anio, indice)
            tiles = img.getMapId(VIS_PARAMS[indice])

            mapa = folium.Map(
                location=[-16.435, -71.60],
                zoom_start=12,
                tiles="OpenStreetMap",
            )

            folium.TileLayer(
                tiles=tiles["tile_fetcher"].url_format,
                attr="Google Earth Engine",
                opacity=opacity,
            ).add_to(mapa)

            st_folium(
                mapa,
                width=450,
                height=380,
                key=f"mapa_{indice}_{anio}_{i}",
            )
            stats = estadisticas_indice(anio, indice)

            st.markdown(
                f"**Promedio:** {stats['mean']:.3f}  \n"
                f"**Mínimo:** {stats['min']:.3f}  \n"
                f"**Máximo:** {stats['max']:.3f}"
            )

    st.divider()
    st.subheader("Evolución temporal (rango seleccionado)")
    rango = [d for d in serie if d["Valor"] is not None and min(anios_sel) <= d["Año"] <= max(anios_sel)]
    if rango:
        st.line_chart({str(d["Año"]): d["Valor"] for d in rango})
    else:
        st.warning("No hay datos suficientes para el rango seleccionado.")

    MENSAJES_EVOLUCION = {
        "NDVI": "Según la tendencia del valor de este índice vemos que se ha incrementado relativamente con el correr "
                "de los años, pero sin embargo es muy bajo estando en centésimos, que indica que hay perdida de cobertura vegetal, aun baja.",
        "SAVI": "Según la tendencia del valor de este índice vemos que se ha incrementado relativamente con el correr de los años, es relativamente "
                "un poco mayor el NAVI pero sin embargo es muy bajo estando en rango de centésimos, lo cual sigue indicando que hay perdida de cobertura "
                "vegetal, aunque relativamente baja.",
        "EVI": "Según la tendencia del valor de este índice, es más sensible para mostrar el área con alta vegetación y vemos que se ha mantenido idéntico"
                "con el correr de los años, pero sin embargo es muy bajo estando en rango de décimos, lo cual sigue indicando que hay perdida de cobertura vegetal, "
                "aunque relativamente baja en algunas zonas.",
        "GNDVI": "Según la tendencia del valor de este índice vemos que se ha incrementado relativamente con el correr de los años, pero sin embargo es muy bajo"
                 "estando en rango de centésimos, los valores bajos indican zonas relacionadas con bajo contenido de clorofila, indicador indirecto de fertilidad "
                 "y disponibilidad de nutrientes.",
        "LSWI": "Según la tendencia del valor de este índice vemos que se mantiene constante y con un valor negativo con el correr de los años, sus valores bajos "
                "indican sequía, riesgo de compactación y pérdida de estructura.",
        "NDWI": "Según la tendencia del valor de este índice vemos que se ha disminuido relativamente con el correr de los años, pero sin embargo es muy bajo estando"
                "en rango de centésimos negativos, ayuda a discriminar humedad, que influye en la descomposición y aporte de materia orgánica. Enfocado en agua superficial "
                "y humedad del suelo. Útil para monitorear sequía agrícola y degradación por falta de agua.",
        "MNDWI": "Según la tendencia del valor de este índice vemos que a disminuido relativamente con el correr de los años, pero sin embargo es muy bajo estando en rango "
                 "de decimos negativos, ayuda a identificar zonas con pérdida de agua, indicador de degradación.",
        "AWEI": "Según la tendencia del valor de este índice vemos que a disminuido relativamente con el correr de los años, pero sin embargo es muy bajo estando en rango "
                 "de decimos negativos, ayuda a identificar zonas con pérdida de agua, indicador de degradación."
    }
    st.markdown(f"{MENSAJES_EVOLUCION[indice]}")

# ── Tab 2 – Gráficos analíticos ───────────────────────────────────────────────
with tab_graficos:
    # 1. Cálculo de predicción para el índice seleccionado
    pred_res = predecir_serie(serie, campo_valor="Valor", campo_anio="Año", anios_futuros=(2026, 2027, 2028, 2029, 2030))
    
    completos = pred_res["historico"]
    anios_hist = [d["Año"] for d in completos]
    valores_hist = [d["Valor"] for d in completos]
    preds = pred_res["predicciones"]
    metricas = pred_res["metricas"]

    # — Serie completa y Proyección 2026-2030 —
    st.subheader(f"Evolución temporal y proyección del {indice}")
    
    if completos:
        fig_evo = go.Figure()
        
        # Serie histórica observada
        fig_evo.add_trace(
            go.Scatter(
                x=anios_hist,
                y=valores_hist,
                mode="lines+markers",
                name="Observado (2000–2025)",
                line=dict(color="#1f77b4", width=2.5),
                marker=dict(size=6),
            )
        )
        
        # Proyección futura con unión al último punto histórico
        if pred_res["valido"] and preds:
            anios_p = [anios_hist[-1]] + [p["Año"] for p in preds]
            vals_p = [valores_hist[-1]] + [p["Valor"] for p in preds]
            
            # Banda de incertidumbre (95%)
            anios_banda = [p["Año"] for p in preds]
            max_banda = [p["Max"] for p in preds]
            min_banda = [p["Min"] for p in preds]
            
            if all(m is not None for m in max_banda) and all(m is not None for m in min_banda):
                fig_evo.add_trace(
                    go.Scatter(
                        x=anios_banda + anios_banda[::-1],
                        y=max_banda + min_banda[::-1],
                        fill="toself",
                        fillcolor="rgba(255, 127, 14, 0.2)",
                        line=dict(color="rgba(255,255,255,0)"),
                        name="Intervalo de incertidumbre (95%)",
                        showlegend=True,
                    )
                )
            
            fig_evo.add_trace(
                go.Scatter(
                    x=anios_p,
                    y=vals_p,
                    mode="lines+markers",
                    name="Proyección (2026–2028)",
                    line=dict(color="#ff7f0e", width=2.5, dash="dash"),
                    marker=dict(size=7, symbol="diamond"),
                )
            )

            # Línea divisoria entre histórico y predicción
            fig_evo.add_vline(
                x=anios_hist[-1] + 0.5,
                line_width=2,
                line_dash="dot",
                line_color="#7f8c8d",
                annotation_text="Límite Histórico / Futuro",
                annotation_position="top left",
            )

            # Áreas sombreadas de fondo para diferenciar periodos
            fig_evo.add_vrect(
                x0=anios_hist[0] - 0.5,
                x1=anios_hist[-1] + 0.5,
                fillcolor="rgba(31, 119, 180, 0.04)",
                layer="below",
                line_width=0,
            )
            fig_evo.add_vrect(
                x0=anios_hist[-1] + 0.5,
                x1=preds[-1]["Año"] + 0.5,
                fillcolor="rgba(255, 127, 14, 0.06)",
                layer="below",
                line_width=0,
            )

        fig_evo.update_layout(
            xaxis=dict(title="Año", dtick=2),
            yaxis=dict(title=f"Valor medio de {indice}"),
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=20, r=20, t=30, b=20),
        )
        st.plotly_chart(fig_evo, use_container_width=True)

        with st.expander("📊 Ver datos de evolución y proyección"):
            datos_grafico = [
                {
                    "Año": d["Año"],
                    "Tipo": "Observado",
                    "Valor": d["Valor"],
                    "Mínimo (95%)": None,
                    "Máximo (95%)": None,
                }
                for d in completos
            ]

            if pred_res["valido"] and preds:
                datos_grafico.extend([
                    {
                        "Año": p["Año"],
                        "Tipo": "Predicción",
                        "Valor": p["Valor"],
                        "Mínimo (95%)": p.get("Min"),
                        "Máximo (95%)": p.get("Max"),
                    }
                    for p in preds
                ])

            df_grafico = pd.DataFrame(datos_grafico)
            st.dataframe(
                df_grafico.round(4),
                use_container_width=True,
                hide_index=True,
            )

            st.caption(
                "Los límites del 95 % se muestran cuando el modelo "
                "puede calcularlos válidamente."
            )
        
        if pred_res["valido"] and metricas.get("MAE") is not None:
            st.info(
                f"**Métricas del modelo ({metricas.get('Modelo')}):** "
                f"MAE: `{metricas.get('MAE'):.4f}` | "
                f"RMSE: `{metricas.get('RMSE'):.4f}` | "
                f"R²: `{metricas.get('R2'):.4f}` | "
                f"Tendencia anual: `{metricas.get('Pendiente_anual', 0):+.5f} / año`"
            )
    else:
        st.warning("No hay datos históricos disponibles para generar la evolución temporal.")

    MENSAJES_SERIES = {
        "NDVI": f"Este gráfico representa la evolución temporal del índice espectral NDVI, donde observamos"
                "tendencias de incremento o disminución asociadas a procesos de degradación o recuperación del suelo"
                "en el área de estudio, entre valores de 0.022 a 0.043, valores bajos, que de preferencia indicarían "
                "proceso de degradación o perdida de cubierta vegetal.",
        "SAVI": "Este gráfico representa la evolución temporal del índice espectral SAVI, donde observamos tendencias "
                "de incremento o disminución asociadas a procesos de degradación o recuperación del suelo en el área de "
                "estudio, entre valores de 0.03 a 0.09, valores bajos, que de preferencia indicarían proceso de degradación "
                "pérdida de cobertura vegetal.",
        "EVI": "Este gráfico representa la evolución temporal del índice espectral EVI, donde observamos tendencias de incremento "
               "o disminución asociadas a procesos de degradación o recuperación del suelo en el área de estudio, entre valores de "
               "-0.034 a 0.30, valores bajos, que de preferencia indicarían proceso de degradación y pérdida de cobertura vegetal.",
        "GNDVI": "Este gráfico representa la evolución temporal del índice espectral GNDVI, donde observamos tendencias de incremento"
                 "o disminución asociadas a perdida de agua y procesos de degradación o recuperación del suelo en el área de estudio, "
                 "entre valores de 0.07 a 0.15, valores bajos, que de preferencia indicarían bajo contenido de clorofila, indicador indirecto "
                 "de fertilidad y disponibilidad de nutrientes.",
        "LSWI": "Este gráfico representa la evolución temporal del índice espectral LSWI, donde observamos tendencias constantes, sin mayor "
                "variación asociadas a riesgo de compactación y pérdida de estructura, entre valores de -0.05 a -0.03.",
        "NDWI": "Este gráfico representa la evolución temporal del índice espectral NDWI, donde observamos tendencias de disminución asociadas "
                "a disminución de humedad de agua y procesos de sequía agrícola y degradación por falta de agua, esta entre valores de -0.06 a -0.11.",
        "MNDWI": "Este gráfico representa la evolución temporal del índice espectral MNDWI, donde observamos tendencias de mínima disminución asociadas "
                 "a zonas con pérdida de agua, indicador de degradación, esta entre valores de -0.011 y -0.015.",
        "AWEI": "Este gráfico representa la evolución temporal del índice espectral AWEI, donde observamos tendencias de mínima disminución asociadas "
                "a zonas con pérdida de agua, indicador de degradación, esta entre valores de -0.011 y -0.015.",    
    }
    st.caption(
        "**Evolución temporal y proyección del índice espectral seleccionado. Permite identificar "
        f"tendenci  as históricas y proyecciones futuras en la respuesta espectral del Río Chili.**\n\n{MENSAJES_SERIES[indice]}"
    )

    st.divider()

    # — Distribución por periodos con posición de predicciones —
    st.subheader(f"Distribución del {indice} por periodos y contraste proyectado")
    if completos:
        fig_dist = go.Figure([
            go.Box(y=[v for a, v in zip(anios_hist, valores_hist) if a <= 2006], name="2000–2006", marker_color="#e74c3c"),
            go.Box(y=[v for a, v in zip(anios_hist, valores_hist) if 2007 <= a <= 2012], name="2007–2012", marker_color="#f39c12"),
            go.Box(y=[v for a, v in zip(anios_hist, valores_hist) if a >= 2013], name="2013–2025", marker_color="#2ecc71"),
        ])
        
        # Superponer puntos proyectados para comparar con las distribuciones históricas
        if pred_res["valido"] and preds:
            fig_dist.add_trace(
                go.Scatter(
                    x=["2013–2025"] * len(preds),
                    y=[p["Valor"] for p in preds],
                    mode="markers+text",
                    text=[f"{p['Año']}: {p['Valor']:.3f}" for p in preds],
                    textposition="top right",
                    marker=dict(color="#8e44ad", size=10, symbol="diamond-dot"),
                    name="Proyección (2026–2030)",
                )
            )
            
        fig_dist.update_layout(
            yaxis_title=f"Valor de {indice}",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=20, r=20, t=30, b=20),
        )
        st.plotly_chart(fig_dist, use_container_width=True)

        with st.expander("📊 Ver datos de distribución por periodos"):
            datos_dist = []

            for anio, valor in zip(anios_hist, valores_hist):
                if anio <= 2006:
                    periodo = "2000–2006"
                elif anio <= 2012:
                    periodo = "2007–2012"
                else:
                    periodo = "2013–2025"

                datos_dist.append({
                    "Año": anio,
                    "Periodo": periodo,
                    "Tipo": "Observado",
                    "Valor": valor,
                })

            if pred_res["valido"] and preds:
                datos_dist.extend([
                    {
                        "Año": p["Año"],
                        "Periodo": "2026–2030",
                        "Tipo": "Predicción",
                        "Valor": p["Valor"],
                    }
                    for p in preds
                ])

            st.dataframe(
                pd.DataFrame(datos_dist).round(4),
                use_container_width=True,
                hide_index=True,
            )

        st.caption(
            "**Variabilidad del índice espectral en distintos intervalos temporales y ubicación relativa "
            "de los valores proyectados 2026–2028 frente a las cajas de dispersión histórica.**"
        )

    st.divider()

    # — Anomalías Observadas y Proyectadas —
    st.subheader(f"Análisis de anomalías del {indice} (Históricas y Proyectadas)")
    if completos:
        anom_data = calcular_anomalias_proyectadas(completos, preds)
        media = anom_data["media"]
        std = anom_data["std"]
        
        def _color(a):
            if abs(a) >= std:       return "#1b5e20" if a > 0 else "#b71c1c"
            if abs(a) >= 0.5 * std: return "#4caf50" if a > 0 else "#f44336"
            return                         "#a5d6a7" if a > 0 else "#ef9a9a"

        # Barras históricas observadas
        hist_anom = anom_data["historico"]
        fig_anom = go.Figure()
        
        fig_anom.add_trace(
            go.Bar(
                x=[d["Año"] for d in hist_anom],
                y=[d["Anomalia"] for d in hist_anom],
                marker_color=[_color(d["Anomalia"]) for d in hist_anom],
                name="Anomalía observada",
            )
        )
        
        # Barras proyectadas 2026-2028 con patrón rayado
        if pred_res["valido"] and anom_data["proyectado"]:
            proj_anom = anom_data["proyectado"]
            fig_anom.add_trace(
                go.Bar(
                    x=[d["Año"] for d in proj_anom],
                    y=[d["Anomalia"] for d in proj_anom],
                    marker=dict(
                        color=[_color(d["Anomalia"]) for d in proj_anom],
                        pattern_shape="/",
                        line=dict(color="#333333", width=1.5),
                    ),
                    name="Anomalía proyectada (2026–2030)",
                )
            )

            # Línea divisoria entre histórico y predicción
            fig_anom.add_vline(
                x=hist_anom[-1]["Año"] + 0.5,
                line_width=2,
                line_dash="dot",
                line_color="#7f8c8d",
                annotation_text="Límite Histórico / Futuro",
                annotation_position="top left",
            )

            # Áreas sombreadas de fondo para diferenciar periodos
            fig_anom.add_vrect(
                x0=hist_anom[0]["Año"] - 0.5,
                x1=hist_anom[-1]["Año"] + 0.5,
                fillcolor="rgba(31, 119, 180, 0.04)",
                layer="below",
                line_width=0,
            )
            fig_anom.add_vrect(
                x0=hist_anom[-1]["Año"] + 0.5,
                x1=proj_anom[-1]["Año"] + 0.5,
                fillcolor="rgba(44, 160, 44, 0.06)",
                layer="below",
                line_width=0,
            )

        fig_anom.add_hline(y=0, line_width=1.5, line_color="black")
        fig_anom.add_hline(y=std, line_dash="dot", line_color="gray", annotation_text="+1σ", annotation_position="top right")
        fig_anom.add_hline(y=-std, line_dash="dot", line_color="gray", annotation_text="-1σ", annotation_position="bottom right")

        fig_anom.update_layout(
            xaxis=dict(title="Año", dtick=2),
            yaxis_title="Anomalía",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=20, r=20, t=40, b=20),
        )
        
        MENSAJES_ANOMALIAS = {
            "NDVI": "Este gráfico identifica valores atípicos que se desvían del comportamiento promedio del índice espectral, "
                    "los cuales pueden estar asociados a eventos ambientales extremos o cambios abruptos en las condiciones del suelo.",
            "SAVI": "Este gráfico identifica valores atípicos que se desvían del comportamiento promedio del índice espectral, "
                    "los cuales pueden estar asociados a eventos ambientales extremos o cambios en la cobertura vegetal.",
            "EVI": "Este gráfico identifica valores atípicos de vigor vegetal respecto a la media histórica observada.",
            "GNDVI": "Este gráfico identifica valores atípicos asociados a variaciones en contenido de clorofila y disponibilidad de nutrientes.",
            "LSWI": "Este gráfico identifica anomalías de humedad y riesgo de sequía/compactación de ribera.",
            "NDWI": "Este gráfico identifica anomalías en la presencia de agua superficial y humedad de suelo.",
            "MNDWI": "Este gráfico identifica desviaciones en la lámina de agua identificada en el cauce.",
            "AWEI": "Este gráfico identifica desviaciones en la extracción automatizada del espejo de agua.",
        }
        st.plotly_chart(fig_anom, use_container_width=True)

        with st.expander("📊 Ver datos de anomalías"):
            datos_anomalias = [
                {
                    "Año": d["Año"],
                    "Tipo": "Observada",
                    "Anomalía": d["Anomalia"],
                }
                for d in anom_data["historico"]
            ]

            datos_anomalias.extend([
                {
                    "Año": d["Año"],
                    "Tipo": "Proyectada",
                    "Anomalía": d["Anomalia"],
                }
                for d in anom_data["proyectado"]
            ])

            st.dataframe(
                pd.DataFrame(datos_anomalias).round(4),
                use_container_width=True,
                hide_index=True,
            )
            
        st.caption(f"**Valores atípicos que se desvían del comportamiento promedio histórico.**\n\n{MENSAJES_ANOMALIAS.get(indice, '')}")
        st.markdown(
            f"**Promedio histórico:** `{media:.4f}`  \n"
            f"**Desviación estándar (1σ):** `{std:.4f}`"
        )
        st.info(
            "ℹ️ **Nota metodológica:** Las anomalías proyectadas representan posibles desviaciones respecto "
            "al comportamiento histórico esperado del índice y no constituyen una medición directa de contaminación "
            "química o microbiológica."
        )

