# -*- coding: utf-8 -*-
"""
DengueSense: interfaz de pronóstico y riesgo de dengue por departamento (Perú).

Ejecutar desde la raíz del proyecto, con el entorno activado:
    streamlit run app/tablero.py

Usa resultados ya calculados (no entrena nada):
  - data/processed/tablero.parquet          (notebook 13)
  - data/processed/panel_depto_semana.parquet (script 07)
  - outputs/*.csv                            (notebooks 09, 11, 12, 13 y 14; script 10)
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

RAIZ = Path(__file__).resolve().parents[1]
PROC, OUT = RAIZ / "data" / "processed", RAIZ / "outputs"
GEOJSON = Path(__file__).resolve().parent / "peru_departamental.geojson"

AZUL, NARANJA, AQUA, AMARILLO, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
TINTA, TINTA_2, TENUE, REJ, FONDO = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"
MODELO_DEFECTO = "XGBoost sin clima"
REGIONES = ["PIURA", "LORETO", "SAN MARTIN"]
NOMBRE_MODELO = {"Regresion con rezagos": "Regresión con rezagos"}
COLOR_MODELO = {"Persistencia": AQUA, "Regresion con rezagos": AMARILLO, "XGBoost sin clima": MAGENTA,
                "XGBoost con clima": AZUL, "LSTM sin clima": TENUE, "LSTM con clima": NARANJA}
ORDEN_MODELOS = list(COLOR_MODELO)

st.set_page_config(page_title="DengueSense · Riesgo de dengue", page_icon="🦟", layout="wide")

st.markdown(f"""
<style>
.block-container {{padding-top: 1.6rem; padding-bottom: 2rem;}}
.hero {{background: linear-gradient(120deg, #0d366b 0%, {AZUL} 70%, #4f95e6 100%); color: white;
        padding: 1.4rem 1.8rem; border-radius: 14px; margin-bottom: 1.2rem;}}
.hero h1 {{color: white; font-size: 2.0rem; margin: 0 0 .3rem 0; padding: 0;}}
.hero p {{color: #e6efff; margin: 0; font-size: 1.02rem;}}
.tarjeta {{background: white; border: 1px solid {REJ}; border-radius: 12px; padding: .9rem 1.1rem;
           box-shadow: 0 1px 3px rgba(0,0,0,.05); height: 100%;}}
.tarjeta .titulo {{color: {TINTA_2}; font-size: .85rem; margin-bottom: .25rem;}}
.tarjeta .valor {{color: {TINTA}; font-size: 1.75rem; font-weight: 700; line-height: 1.15;}}
.tarjeta .nota {{color: {TENUE}; font-size: .8rem; margin-top: .25rem;}}
.etiqueta {{display: inline-block; padding: .15rem .55rem; border-radius: 999px; font-size: .8rem;
            font-weight: 600; margin-right: .3rem;}}
.alerta {{background: #fde1d5; color: #a3330c;}}
.ok {{background: #d8f3e8; color: #0f6e4c;}}
.caja {{background: #f1f4f8; border-left: 4px solid {AZUL}; padding: .8rem 1rem; border-radius: 6px;
        margin: .4rem 0 1rem 0; color: {TINTA};}}
.caja.naranja {{border-left-color: {NARANJA}; background: #fdf1eb;}}
div[data-testid="stTabs"] button p {{font-size: 1.0rem; font-weight: 600;}}
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------- utilidades ---
def nombre(m: str) -> str:
    return NOMBRE_MODELO.get(m, m)


def tarjeta(columna, titulo: str, valor: str, nota: str = "", color: str = AZUL):
    columna.markdown(f"<div class='tarjeta' style='border-top: 4px solid {color}'><div class='titulo'>{titulo}</div>"
                     f"<div class='valor'>{valor}</div><div class='nota'>{nota}</div></div>", unsafe_allow_html=True)


def caja(texto: str, naranja: bool = False):
    st.markdown(f"<div class='caja{' naranja' if naranja else ''}'>{texto}</div>", unsafe_allow_html=True)


def miles(x: float) -> str:
    return f"{x:,.0f}".replace(",", " ")


def estilo(fig, alto: int = 380, leyenda_abajo: bool = True):
    fig.update_layout(height=alto, plot_bgcolor="white", paper_bgcolor="white", font=dict(color=TINTA),
                      margin=dict(l=10, r=10, t=40, b=10), hoverlabel=dict(bgcolor="white"))
    if leyenda_abajo:
        fig.update_layout(legend=dict(orientation="h", y=-0.18, x=0, title=None))
    fig.update_xaxes(gridcolor=REJ, zeroline=False)
    fig.update_yaxes(gridcolor=REJ, zeroline=False)
    return fig


# ------------------------------------------------------------------ datos ---
@st.cache_data
def cargar_tablero() -> pd.DataFrame:
    d = pd.read_parquet(PROC / "tablero.parquet")
    d["fecha"] = pd.to_datetime(d["fecha"])
    d["razon_umbral"] = d["prediccion"] / d["umbral"].clip(lower=1)
    return d


@st.cache_data
def cargar_csv(nombre_archivo: str) -> pd.DataFrame | None:
    ruta = OUT / nombre_archivo
    return pd.read_csv(ruta) if ruta.exists() else None


@st.cache_data
def cargar_panel() -> pd.DataFrame | None:
    ruta = PROC / "panel_depto_semana.parquet"
    if not ruta.exists():
        return None
    p = pd.read_parquet(ruta)
    p["fecha"] = pd.to_datetime(p["fecha"])
    return p


@st.cache_data
def cargar_geo() -> dict:
    return json.loads(GEOJSON.read_text(encoding="utf-8"))


if not (PROC / "tablero.parquet").exists():
    st.error("No se encontró `data/processed/tablero.parquet`. Ejecuta primero "
             "`notebooks/13_modelos_finales_y_pronostico.ipynb`.")
    st.stop()

datos = cargar_tablero()
geo = cargar_geo()
todos_deptos = sorted(f["properties"]["NOMBDEP"] for f in geo["features"])
deptos_modelo = sorted(datos["departamento"].unique())

# ------------------------------------------------------------- encabezado ---
pron = datos[(datos["tipo"] == "Pronostico") & (datos["modelo"] == MODELO_DEFECTO)]
st.markdown("""
<div class='hero'>
  <h1>🦟 DengueSense</h1>
  <p>Pronóstico semanal de casos de dengue y alerta de brotes por departamento en el Perú, con 1 a 4 semanas
  de anticipación. Datos abiertos del CDC-MINSA (2010–2025), clima e índice de El Niño costero.</p>
</div>
""", unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
en_alerta = sorted(pron.loc[pron["alerta"], "departamento"].unique())
tarjeta(c1, "Departamentos con alerta (pronóstico 2026, semanas 1–4)", f"{len(en_alerta)} de {pron['departamento'].nunique()}",
        ", ".join(d.title() for d in en_alerta) or "ninguno", NARANJA)
oos = cargar_csv("fuera_de_muestra_2025_metricas.csv")
if oos is not None:
    m = oos.pivot_table(index=["departamento", "horizonte"], columns="modelo", values="mae")
    gana = int((m[MODELO_DEFECTO] < m["Persistencia"]).sum())
    tarjeta(c2, "Prueba real 2025 (año nunca visto)", f"{gana / len(m):.0%}",
            "de las combinaciones departamento-horizonte en que el modelo supera a la persistencia", AQUA)
tarjeta(c3, "Departamentos modelados", f"{len(deptos_modelo)}", "con ≥ 1 000 casos en 2010–2021 (sin Lima)", AZUL)
alertas = cargar_csv("fase5_alertas.csv")
if alertas is not None:
    a = alertas[(alertas["horizonte"] == 2) & (alertas["modelo"] == MODELO_DEFECTO)].iloc[0]
    tarjeta(c4, "Semanas de brote con alerta (2 semanas antes)", f"{a['recall']:.0%}",
            f"precisión {a['precision']:.0%} · brotes nuevos anticipados {a['inicios_anticipados']:.0%}", MAGENTA)
st.write("")

pestanas = st.tabs(["🗺️ Mapa de riesgo", "📈 Departamento", "🎯 Desempeño de los modelos",
                    "🌦️ Factores climáticos", "🏙️ Lima", "ℹ️ Acerca del proyecto"])

# ============================================================ 1. MAPA ======
with pestanas[0]:
    tipos = {"Pronostico": "Pronóstico 2026 (semanas 1 a 4)", "Fuera de muestra 2025": "Prueba fuera de muestra 2025",
             "Validacion 2016-2024": "Validación 2016–2024"}
    f1, f2, f3 = st.columns([1.3, 1.2, 1])
    tipo = f1.radio("Periodo", list(tipos), format_func=tipos.get, horizontal=False, key="tipo_mapa")
    sub = datos[datos["tipo"] == tipo]
    modelos = sorted(sub["modelo"].unique(), key=lambda m: (m != MODELO_DEFECTO, m))
    modelo = f2.selectbox("Modelo", modelos, format_func=nombre, key="modelo_mapa")
    horizonte = f3.select_slider("Anticipación (semanas)", options=[1, 2, 3, 4], value=2, key="h_mapa")
    sub = sub[(sub["modelo"] == modelo) & (sub["horizonte"] == horizonte)]
    semanas = sub[["ano", "semana"]].drop_duplicates().sort_values(["ano", "semana"])
    opciones = {f"{a} · semana {s}": (int(a), int(s)) for a, s in zip(semanas["ano"], semanas["semana"])}
    if tipo == "Pronostico":
        elegida = next(iter(opciones.values()))
    else:
        defecto = "2024 · semana 15" if "2024 · semana 15" in opciones else list(opciones)[-1]
        elegida = opciones[st.select_slider("Semana epidemiológica", options=list(opciones), value=defecto, key="sem_mapa")]
    semana = sub[(sub["ano"] == elegida[0]) & (sub["semana"] == elegida[1])]
    fecha = semana["fecha"].iloc[0]

    k1, k2, k3, k4 = st.columns(4)
    tarjeta(k1, "Semana", f"{elegida[1]} / {elegida[0]}", f"miércoles {fecha:%d/%m/%Y}", AZUL)
    tarjeta(k2, "Departamentos con alerta", f"{int(semana['alerta'].sum())} de {len(semana)}",
            f"pronóstico hecho {horizonte} semana{'s' if horizonte > 1 else ''} antes", NARANJA)
    tarjeta(k3, "Casos pronosticados", miles(semana["prediccion"].sum()), nombre(modelo), MAGENTA)
    if semana["casos"].notna().any():
        tarjeta(k4, "Casos observados", miles(semana["casos"].sum()),
                f"{(semana['alerta'] == semana['brote']).mean():.0%} de departamentos bien clasificados", AQUA)
    else:
        tarjeta(k4, "Casos observados", "—", "todavía no disponibles", AQUA)
    st.write("")

    izq, der = st.columns([1.15, 1])
    with izq:
        mapa = pd.DataFrame({"departamento": todos_deptos}).merge(semana, on="departamento", how="left")
        mapa["riesgo"] = mapa["razon_umbral"].clip(upper=3)
        mapa["Estado"] = mapa["alerta"].map({True: "⚠️ ALERTA", False: "Sin alerta"}).fillna("No modelado")
        mapa["Departamento"] = mapa["departamento"].str.title()
        fig = px.choropleth(
            mapa, geojson=geo, locations="departamento", featureidkey="properties.NOMBDEP", color="riesgo",
            color_continuous_scale=[(0, "#cde2fb"), (0.33, "#f4f6f8"), (0.45, "#fbd3bf"), (1, "#a3330c")],
            range_color=(0, 3), hover_name="Departamento",
            hover_data={"departamento": False, "riesgo": False, "Estado": True, "prediccion": ":.0f",
                        "umbral": ":.0f", "casos": ":.0f"},
            labels={"prediccion": "Pronóstico", "umbral": "Umbral de brote", "casos": "Observados"})
        fig.update_traces(marker_line_color="white", marker_line_width=0.8)
        fig.update_geos(fitbounds="locations", visible=False, bgcolor="rgba(0,0,0,0)")
        fig.update_layout(height=640, margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor="rgba(0,0,0,0)",
                          coloraxis_colorbar=dict(title="Pronóstico<br>/ umbral", tickvals=[0, 1, 2, 3],
                                                  ticktext=["0", "1 = umbral", "2", "≥ 3"], len=0.6))
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
        st.caption("Rojo: el pronóstico supera el canal endémico (alerta). Gris: departamento fuera del modelo.")
    with der:
        st.markdown("##### Departamentos ordenados por riesgo")
        orden = semana.sort_values("razon_umbral", ascending=True)
        fig = go.Figure(go.Bar(
            x=orden["razon_umbral"], y=orden["departamento"].str.title(), orientation="h",
            marker_color=[NARANJA if a else "#9ec5f4" for a in orden["alerta"]],
            text=[f"{p:,.0f} casos".replace(",", " ") for p in orden["prediccion"]], textposition="outside",
            hovertemplate="%{y}<br>pronóstico / umbral: %{x:.2f}<extra></extra>"))
        fig.add_vline(x=1, line_dash="dash", line_color=TINTA_2, annotation_text="umbral", annotation_position="top")
        fig = estilo(fig, alto=600, leyenda_abajo=False)
        fig.update_layout(xaxis_title="Pronóstico / umbral del canal endémico", margin=dict(t=30))
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

    with st.expander("Ver la tabla de la semana"):
        tabla = (semana.sort_values("razon_umbral", ascending=False)
                 [["departamento", "prediccion", "umbral", "casos", "alerta", "brote"]]
                 .rename(columns={"departamento": "Departamento", "prediccion": "Pronóstico", "umbral": "Umbral",
                                  "casos": "Observados", "alerta": "Alerta", "brote": "Brote real"}))
        st.dataframe(tabla.round(0), hide_index=True, width="stretch")

# ====================================================== 2. DEPARTAMENTO ====
with pestanas[1]:
    f1, f2, f3 = st.columns([1.2, 1, 1.3])
    depto = f1.selectbox("Departamento", deptos_modelo, index=deptos_modelo.index("PIURA"),
                         format_func=str.title, key="depto")
    h_d = f2.select_slider("Anticipación (semanas)", options=[1, 2, 3, 4], value=2, key="h_depto")
    modelo_d = f3.selectbox("Modelo", [MODELO_DEFECTO, "XGBoost con clima", "Persistencia", "Regresion con rezagos"],
                            format_func=nombre, key="modelo_depto")
    serie = datos[(datos["departamento"] == depto) & (datos["horizonte"] == h_d)]
    serie_m = serie[serie["modelo"] == modelo_d].sort_values("fecha")
    if serie_m.empty:
        st.info("Ese modelo no tiene predicciones para todo el periodo.")
    else:
        obs = serie_m.dropna(subset=["casos"])
        brotes = obs[obs["brote"]]
        k1, k2, k3, k4 = st.columns(4)
        tarjeta(k1, "Casos 2016–2025", miles(obs["casos"].sum()), f"pico: {miles(obs['casos'].max())} casos en una semana", AZUL)
        tarjeta(k2, "Semanas de brote", f"{len(brotes)}", f"{len(brotes) / max(len(obs), 1):.0%} de las semanas", NARANJA)
        tarjeta(k3, "Brotes con alerta a tiempo", f"{(brotes['alerta']).mean():.0%}" if len(brotes) else "—",
                f"alerta {h_d} semana{'s' if h_d > 1 else ''} antes", MAGENTA)
        error = (obs["casos"] - obs["prediccion"]).abs().mean()
        tarjeta(k4, "Error medio del pronóstico", f"{error:,.1f}".replace(",", " "), "casos por semana (MAE)", AQUA)
        st.write("")

        fig = go.Figure()
        tope = max(obs["casos"].max(), serie_m["prediccion"].max()) * 1.08
        fig.add_trace(go.Bar(x=brotes["fecha"], y=[tope] * len(brotes), marker_color="#fde7dd", marker_line_width=0,
                             width=7 * 24 * 3600 * 1000, name="Semana de brote", hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=serie_m["fecha"], y=serie_m["umbral"], name="Umbral (canal endémico)",
                                 line=dict(color=NARANJA, dash="dash", width=1.3)))
        fig.add_trace(go.Scatter(x=obs["fecha"], y=obs["casos"], name="Casos observados", line=dict(color=TINTA, width=2.2)))
        hist = serie_m[serie_m["tipo"] != "Pronostico"]
        fig.add_trace(go.Scatter(x=hist["fecha"], y=hist["prediccion"], name=f"Pronóstico ({nombre(modelo_d)})",
                                 line=dict(color=AZUL, width=1.6)))
        futuro = serie[(serie["tipo"] == "Pronostico") & (serie["modelo"] == MODELO_DEFECTO)]
        fig.add_trace(go.Scatter(x=futuro["fecha"], y=futuro["prediccion"], mode="markers+lines", name="Pronóstico 2026",
                                 marker=dict(color=NARANJA, size=10, line=dict(color="white", width=1.5)),
                                 line=dict(color=NARANJA, width=2)))
        fig.add_vrect(x0="2025-01-01", x1=str(obs["fecha"].max().date()), fillcolor=AQUA, opacity=0.06, line_width=0,
                      annotation_text="2025: fuera de muestra", annotation_position="top left")
        fig = estilo(fig, alto=500)
        fig.update_layout(barmode="overlay", hovermode="x unified", yaxis_title="Casos por semana",
                          xaxis=dict(rangeslider=dict(visible=True, thickness=0.06),
                                     rangeselector=dict(buttons=[dict(count=1, label="1 año", step="year", stepmode="backward"),
                                                                 dict(count=3, label="3 años", step="year", stepmode="backward"),
                                                                 dict(step="all", label="Todo")])))
        st.plotly_chart(fig, width="stretch")
        caja("Las barras naranjas claras marcan las <b>semanas de brote</b> (casos sobre el canal endémico). "
             "Para cada semana, la línea azul es lo que el modelo pronosticó <b>"
             f"{h_d} semana{'s' if h_d > 1 else ''} antes</b>, usando solo datos de años anteriores.")

# ========================================================= 3. DESEMPEÑO ====
with pestanas[2]:
    rolling = cargar_csv("fase5_validacion_temporal_metricas.csv")
    if rolling is not None:
        st.markdown("#### ¿Qué tan bien pronostica? Validación con 9 años de prueba (2016–2024)")
        caja("Cada año se pronosticó con modelos entrenados <b>solo con los años anteriores</b>. "
             "Menor error = mejor.")
        izq, der = st.columns(2)
        mae = rolling.groupby(["modelo", "horizonte"])["mae"].median().reset_index()
        mae["Modelo"] = mae["modelo"].map(nombre)
        fig = px.bar(mae, x="horizonte", y="mae", color="modelo", barmode="group",
                     color_discrete_map=COLOR_MODELO, labels={"horizonte": "Semanas de anticipación",
                                                               "mae": "Error mediano (casos por semana)"},
                     category_orders={"modelo": [m for m in ORDEN_MODELOS if m in mae["modelo"].unique()]})
        fig.for_each_trace(lambda t: t.update(name=nombre(t.name)))
        fig.update_layout(title="Error mediano por departamento y año")
        izq.plotly_chart(estilo(fig), width="stretch")

        p = rolling.pivot_table(index=["anio_evaluado", "departamento", "horizonte"], columns="modelo", values="mae")
        gana = (p["XGBoost con clima"] < p["XGBoost sin clima"]).groupby(level=["anio_evaluado", "horizonte"]).sum().reset_index(name="deptos")
        n = rolling["departamento"].nunique()
        fig = px.line(gana, x="anio_evaluado", y="deptos", color="horizonte", markers=True,
                      color_discrete_sequence=["#cde2fb", "#86b6ef", "#256abf", "#0d366b"],
                      labels={"anio_evaluado": "Año evaluado", "deptos": f"Departamentos (de {n})", "horizonte": "Semanas"})
        fig.add_hline(y=n / 2, line_dash="dash", line_color=TINTA_2, annotation_text="mitad: el clima no aporta")
        fig.update_layout(title="¿El clima reduce el error? (hipótesis H2)", yaxis_range=[0, n])
        der.plotly_chart(estilo(fig), width="stretch")
        caja("<b>Hallazgo principal:</b> el clima reduce el error en alrededor de la mitad de los departamentos cada año, "
             "lo que se esperaría si no aportara nada. El XGBoost sí mejora a la persistencia, pero <b>la ventaja "
             "viene del modelo, no del clima</b>.", naranja=True)

    if alertas is not None:
        st.markdown("#### Alertas de brote")
        h_a = st.select_slider("Anticipación (semanas)", options=[1, 2, 3, 4], value=2, key="h_alertas")
        a = alertas[alertas["horizonte"] == h_a].copy()
        a["Modelo"] = a["modelo"].map(nombre)
        larga = a.melt(id_vars=["modelo", "Modelo"], value_vars=["recall", "precision", "inicios_anticipados"],
                       var_name="indicador", value_name="valor")
        larga["indicador"] = larga["indicador"].map({"recall": "Brotes con alerta (recall)",
                                                     "precision": "Alertas correctas (precisión)",
                                                     "inicios_anticipados": "Brotes NUEVOS anticipados"})
        larga["valor"] *= 100
        fig = px.bar(larga, x="indicador", y="valor", color="modelo", barmode="group", color_discrete_map=COLOR_MODELO,
                     text=larga["valor"].round(0).astype(int).astype(str) + "%",
                     labels={"valor": "%", "indicador": ""})
        fig.for_each_trace(lambda t: t.update(name=nombre(t.name)))
        fig.update_traces(textposition="outside")
        st.plotly_chart(estilo(fig, alto=400), width="stretch")
        caja("Avisar que un brote <b>sigue</b> es fácil (≈ 90 %). Lo difícil es anticipar un brote <b>nuevo</b> "
             "(sin brote en las 4 semanas previas): ningún modelo pasa del 28 %.")

    c1, c2 = st.columns(2)
    if oos is not None:
        with c1:
            st.markdown("#### Prueba real: 2025")
            m = oos.pivot_table(index=["departamento", "horizonte"], columns="modelo", values="mae")
            g = (m[MODELO_DEFECTO] < m["Persistencia"]).groupby(level="horizonte").sum().reset_index(name="deptos")
            fig = px.bar(g, x="horizonte", y="deptos", text="deptos", color_discrete_sequence=[AZUL],
                         labels={"horizonte": "Semanas de anticipación", "deptos": "Departamentos (de 17)"})
            fig.update_layout(title="Departamentos donde XGBoost supera a la persistencia", yaxis_range=[0, 17])
            fig.update_traces(textposition="outside")
            st.plotly_chart(estilo(fig, alto=360), width="stretch")
            st.caption("Modelos entrenados hasta 2024 y aplicados a 2025, un año que nunca vieron.")
    lstm = cargar_csv("fase4_lstm_metricas.csv")
    if lstm is not None:
        with c2:
            st.markdown("#### Extensión: red neuronal LSTM (prueba 2024)")
            lp = lstm[lstm["conjunto"] == "Prueba"].groupby(["modelo", "horizonte"])["mae"].median().reset_index()
            fig = px.line(lp, x="horizonte", y="mae", color="modelo", markers=True, color_discrete_map=COLOR_MODELO,
                          labels={"horizonte": "Semanas de anticipación", "mae": "Error mediano"},
                          category_orders={"modelo": ORDEN_MODELOS})
            fig.for_each_trace(lambda t: t.update(name=nombre(t.name), line_dash="dash" if "LSTM" in t.name else "solid"))
            fig.update_layout(title="Error mediano entre departamentos")
            st.plotly_chart(estilo(fig, alto=360), width="stretch")
            st.caption("El LSTM con clima destaca en Piura, pero es un solo año de prueba: es una línea prometedora.")

# ================================================ 4. FACTORES CLIMÁTICOS ===
with pestanas[3]:
    shap_df = cargar_csv("fase3_shap_importancia.csv")
    if shap_df is None:
        st.info("Falta outputs/fase3_shap_importancia.csv (notebook 09).")
    else:
        caja("SHAP mide cuánto <b>usa</b> el modelo cada variable para pronosticar. Muestra asociaciones, "
             "<b>no causas</b>.")
        h_s = st.select_slider("Anticipación (semanas)", options=[1, 2, 3, 4], value=2, key="h_shap")
        s = shap_df[shap_df["horizonte"] == h_s]
        cats = ["temperatura", "lluvia", "humedad", "viento", "icen"]
        por_cat = s.groupby(["departamento", "categoria"])["pct"].sum().unstack(fill_value=0)
        clima = por_cat[cats].div(por_cat[cats].sum(axis=1), axis=0) * 100
        orden_d = s.groupby("departamento")["importancia"].sum().sort_values(ascending=False).index
        clima = clima.loc[[d for d in orden_d if d in clima.index]]
        izq, der = st.columns([1.1, 1])
        with izq:
            fig = px.imshow(clima.round(0), text_auto=True, aspect="auto",
                            color_continuous_scale=["#f4f8fd", "#86b6ef", "#256abf", "#0d366b"], zmin=0, zmax=100,
                            labels=dict(color="% del clima"),
                            x=["Temperatura", "Lluvia", "Humedad", "Viento", "ICEN"], y=[d.title() for d in clima.index])
            fig.update_layout(title="Peso de cada factor climático, por departamento")
            st.plotly_chart(estilo(fig, alto=620, leyenda_abajo=False), width="stretch")
        with der:
            depto_s = st.selectbox("Departamento", list(clima.index), format_func=str.title,
                                   index=list(clima.index).index("PIURA") if "PIURA" in clima.index else 0, key="depto_shap")
            top = s[s["departamento"] == depto_s].nlargest(12, "pct").iloc[::-1]
            color_cat = {"casos pasados": "#c3c2b7", "estacionalidad": TENUE, "temperatura": NARANJA, "lluvia": AZUL,
                         "humedad": AQUA, "viento": MAGENTA, "icen": AMARILLO}
            fig = px.bar(top, x="pct", y="variable", color="categoria", orientation="h", color_discrete_map=color_cat,
                         labels={"pct": "% de la importancia", "variable": "", "categoria": "Categoría"})
            fig.update_layout(title=f"Variables que más usa el modelo · {depto_s.title()}", yaxis=dict(categoryorder="total ascending"))
            st.plotly_chart(estilo(fig, alto=470), width="stretch")
            casos_pct = por_cat.loc[depto_s, ["casos pasados", "estacionalidad"]].sum()
            caja(f"En {depto_s.title()}, los <b>casos pasados y la estacionalidad</b> explican el "
                 f"{casos_pct:.0f} % de lo que usa el modelo; el clima, el {100 - casos_pct:.0f} %.")
        caja("<b>Costa norte</b> (Piura, Lambayeque, Ica): domina la temperatura. <b>Selva</b> (Ucayali, Madre de Dios, "
             "Loreto): pesan más la humedad, la lluvia y el viento. Así se confirma la hipótesis H1: los factores "
             "climáticos cambian por región.")

# ============================================================== 5. LIMA ====
with pestanas[4]:
    panel = cargar_panel()
    lima_pred = cargar_csv("lima_predicciones_h2.csv")
    if panel is None:
        st.info("Falta data/processed/panel_depto_semana.parquet (script 07).")
    else:
        pl = panel[panel["departamento"] == "LIMA"].sort_values("fecha")
        anual = pl.groupby("ano")["casos"].sum()
        k1, k2, k3, k4 = st.columns(4)
        tarjeta(k1, "Casos en 2022", miles(anual.get(2022, 0)), "antes de la emergencia", AZUL)
        tarjeta(k2, "Casos en 2023", miles(anual.get(2023, 0)), "año de El Niño costero", NARANJA)
        tarjeta(k3, "Casos en 2024", miles(anual.get(2024, 0)), "la mayor epidemia de Lima", MAGENTA)
        tarjeta(k4, "Distritos con casos por primera vez", "36 de 79", "en 2023 o 2024", AQUA)
        st.write("")
        izq, der = st.columns(2)
        with izq:
            fig = px.bar(anual.reset_index(), x="ano", y="casos", color_discrete_sequence=[AZUL],
                         labels={"ano": "Año", "casos": "Casos"})
            fig.update_traces(marker_color=[NARANJA if a >= 2023 else AZUL for a in anual.index])
            fig.update_layout(title="Casos de dengue por año, departamento de Lima")
            st.plotly_chart(estilo(fig, alto=380), width="stretch")
        with der:
            ref = pl[pl["ano"] <= 2022].groupby(pl["semana"].clip(upper=52))["temp_media"].mean()
            pl = pl.assign(anomalia=pl["temp_media"] - pl["semana"].clip(upper=52).map(ref))
            v = pl[pl["ano"] >= 2022]
            fig = go.Figure()
            fig.add_trace(go.Bar(x=v["fecha"], y=v["casos"], name="Casos por semana", marker_color="#fbd3bf", yaxis="y2"))
            fig.add_trace(go.Scatter(x=v["fecha"], y=v["anomalia"], name="Anomalía de temperatura (°C)", line=dict(color=AZUL)))
            fig.add_trace(go.Scatter(x=v["fecha"], y=v["icen"], name="ICEN (El Niño costero)", line=dict(color=AMARILLO, width=2.5)))
            fig.update_layout(title="Clima, El Niño costero y casos, 2022–2024",
                              yaxis=dict(title="°C / ICEN"), yaxis2=dict(title="Casos", overlaying="y", side="right", showgrid=False))
            st.plotly_chart(estilo(fig, alto=380), width="stretch")
        if lima_pred is not None:
            lima_pred["fecha"] = pd.to_datetime(lima_pred["fecha"])
            colores = {"Persistencia": AQUA, "XGBoost solo Lima": MAGENTA, "XGBoost costa, sin clima": AMARILLO,
                       "XGBoost costa, con clima": AZUL}
            fig = go.Figure()
            base = lima_pred[lima_pred["modelo"] == "Persistencia"]
            fig.add_trace(go.Scatter(x=base["fecha"], y=base["casos"], name="Casos observados", line=dict(color=TINTA, width=2.5)))
            for mdl, col in colores.items():
                d = lima_pred[lima_pred["modelo"] == mdl]
                fig.add_trace(go.Scatter(x=d["fecha"], y=d["prediccion"], name=mdl, line=dict(color=col, width=1.5)))
            fig.update_layout(title="¿Se podía pronosticar? Pronóstico a 2 semanas en 2023–2024", hovermode="x unified",
                              yaxis_title="Casos por semana")
            st.plotly_chart(estilo(fig, alto=420), width="stretch")
            err = (lima_pred.assign(e=(lima_pred["casos"] - lima_pred["prediccion"]).abs())
                   .pivot_table(index="modelo", columns="anio", values="e", aggfunc="mean").round(0))
            c1, c2 = st.columns([1, 1.4])
            c1.dataframe(err.rename_axis("Error medio (casos/semana)"), width="stretch")
            with c2:
                caja("Un modelo entrenado <b>solo con la historia de Lima</b> es el peor en 2023: no había visto dengue. "
                     "Uno entrenado con los <b>departamentos de la costa</b> reduce el error un 30 %. "
                     "Se confirma la hipótesis H3: el desempeño cae en periodos atípicos y sin historia.", naranja=True)

# ============================================================= 6. ACERCA ===
with pestanas[5]:
    izq, der = st.columns([1.2, 1])
    with izq:
        st.markdown("#### Cómo funciona")
        pasos = [
            ("1. Datos", "Casos notificados del CDC-MINSA (2000–2025), clima semanal por celda, ICEN (ENFEN) y población del INEI."),
            ("2. Limpieza y panel", "Casos por departamento y semana epidemiológica; clima ponderado por dónde ocurren los casos."),
            ("3. Líneas base", "Persistencia, promedio estacional y regresión con casos rezagados."),
            ("4. Modelo principal", "XGBoost por departamento y horizonte que aprende el <b>cambio</b> de casos; SHAP para explicarlo."),
            ("5. Evaluación honesta", "Validación con 9 años (origen móvil), alertas de brote y prueba real en 2025."),
            ("6. Pronóstico", "Modelos finales con 2010–2024 y pronóstico de las semanas siguientes al último dato."),
        ]
        for titulo, texto in pasos:
            st.markdown(f"<div class='tarjeta' style='margin-bottom:.5rem; border-left: 4px solid {AZUL}'>"
                        f"<b>{titulo}</b><br><span style='color:{TINTA_2}'>{texto}</span></div>", unsafe_allow_html=True)
    with der:
        st.markdown("#### Limitaciones")
        st.markdown("""
- Los datos abiertos tienen algo menos de casos que el total oficial (−3 % a −6 % en años epidémicos).
- Los modelos suponen que los casos recientes ya están completos; en la práctica llegan con retraso.
- El clima es el promedio de celdas de unos 55 km; faltan punto de rocío y presión.
- La población de 2010–2017 es estimada.
- Anticipar el **inicio** de un brote sigue siendo difícil.
""")
        ext = cargar_csv("validacion_externa_nacional.csv")
        if ext is not None:
            e = ext.melt(id_vars="ano", value_vars=["panel", "opendengue_admin0"], var_name="fuente", value_name="casos")
            e["fuente"] = e["fuente"].map({"panel": "Datos del proyecto", "opendengue_admin0": "Total nacional (OpenDengue)"})
            fig = px.bar(e, x="ano", y="casos", color="fuente", barmode="group",
                         color_discrete_sequence=[AZUL, "#c3c2b7"], labels={"ano": "Año", "casos": "Casos"})
            fig.update_layout(title="Validación externa de los casos")
            st.plotly_chart(estilo(fig, alto=330), width="stretch")
        st.markdown("#### Fuentes")
        st.caption("CDC-MINSA, datos abiertos (ODC-By) · ENFEN (ICEN) · INEI (población) · OpenDengue (validación) · "
                   "mapa: peru-geojson (MPL-2.0). Proyecto DengueSense.")
