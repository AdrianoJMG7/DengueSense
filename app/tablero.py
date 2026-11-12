# -*- coding: utf-8 -*-
"""
Tablero de riesgo de dengue por departamento (Perú).

Ejecutar desde la raíz del proyecto:
    streamlit run app/tablero.py

Lee data/processed/tablero.parquet, generado por
notebooks/13_modelos_finales_y_pronostico.ipynb.
"""
import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

RAIZ = Path(__file__).resolve().parents[1]
ARCHIVO = RAIZ / "data" / "processed" / "tablero.parquet"
GEOJSON = Path(__file__).resolve().parent / "peru_departamental.geojson"

AZUL, NARANJA, AQUA, TINTA, TINTA_2, REJ = "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#52514e", "#e1e0d9"
MODELO_DEFECTO = "XGBoost sin clima"

st.set_page_config(page_title="Riesgo de dengue por departamento", layout="wide")


@st.cache_data
def cargar() -> pd.DataFrame:
    datos = pd.read_parquet(ARCHIVO)
    datos["fecha"] = pd.to_datetime(datos["fecha"])
    datos["razon_umbral"] = datos["prediccion"] / datos["umbral"].clip(lower=1)
    return datos


@st.cache_data
def mapa_base() -> dict:
    return json.loads(GEOJSON.read_text(encoding="utf-8"))


if not ARCHIVO.exists():
    st.error("No se encontró data/processed/tablero.parquet. Ejecuta primero "
             "notebooks/13_modelos_finales_y_pronostico.ipynb.")
    st.stop()

datos = cargar()
geo = mapa_base()
todos_deptos = sorted(f["properties"]["NOMBDEP"] for f in geo["features"])

# ------------------------------------------------------------ controles ----
st.sidebar.title("Opciones")
tipos = ["Pronostico", "Fuera de muestra 2025", "Validacion 2016-2024"]
etiquetas = {"Pronostico": "Pronóstico (2026, semanas 1 a 4)", "Fuera de muestra 2025": "Prueba fuera de muestra 2025",
             "Validacion 2016-2024": "Validación 2016–2024"}
tipo = st.sidebar.radio("Periodo", tipos, format_func=etiquetas.get)
sub = datos[datos["tipo"] == tipo]

modelos = sorted(sub["modelo"].unique(), key=lambda m: (m != MODELO_DEFECTO, m))
modelo = st.sidebar.selectbox("Modelo", modelos, index=0)
horizonte = st.sidebar.select_slider("Anticipación (semanas)", options=[1, 2, 3, 4], value=2)
sub = sub[(sub["modelo"] == modelo) & (sub["horizonte"] == horizonte)]

semanas = sub[["ano", "semana", "fecha"]].drop_duplicates().sort_values("fecha")
# Etiquetas de texto: select_slider interpreta una tupla (ano, semana) como un rango
opciones = {f"{a}, semana {s}": (int(a), int(s)) for a, s in zip(semanas["ano"], semanas["semana"])}
if tipo == "Pronostico":
    elegida = next(iter(opciones.values()))   # en el pronostico cada horizonte tiene su propia semana
else:
    etiqueta = st.sidebar.select_slider("Semana epidemiológica", options=list(opciones), value=list(opciones)[-1])
    elegida = opciones[etiqueta]
semana = sub[(sub["ano"] == elegida[0]) & (sub["semana"] == elegida[1])]

st.sidebar.markdown("---")
st.sidebar.caption(
    "Una semana es de **brote** si los casos superan el canal endémico (Q3 de la misma semana en los 7 años "
    "previos no epidémicos) y hay al menos 5 casos. El modelo da **alerta** si su pronóstico supera ese umbral."
)

# --------------------------------------------------------------- titulo ----
st.title("Riesgo de dengue por departamento")
fecha_txt = semana["fecha"].iloc[0].strftime("%d/%m/%Y") if len(semana) else ""
st.markdown(f"**{etiquetas[tipo]}** · semana {elegida[1]} de {elegida[0]} (miércoles {fecha_txt}) · "
            f"pronóstico hecho con **{horizonte} semana{'s' if horizonte > 1 else ''}** de anticipación · "
            f"modelo: **{modelo}**")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Departamentos con alerta", f"{int(semana['alerta'].sum())} de {len(semana)}")
c2.metric("Casos pronosticados", f"{semana['prediccion'].sum():,.0f}".replace(",", " "))
if semana["casos"].notna().any():
    c3.metric("Casos observados", f"{semana['casos'].sum():,.0f}".replace(",", " "))
    acertadas = (semana["alerta"] == semana["brote"]).mean()
    c4.metric("Departamentos bien clasificados", f"{acertadas:.0%}")
else:
    c3.metric("Casos observados", "aún no disponibles")

# ------------------------------------------------------------------ mapa ----
izq, der = st.columns([1.1, 1])
with izq:
    st.subheader("Mapa de riesgo")
    mapa = pd.DataFrame({"departamento": todos_deptos}).merge(semana, on="departamento", how="left")
    mapa["riesgo"] = mapa["razon_umbral"].clip(upper=3)
    mapa["estado"] = mapa["alerta"].map({True: "ALERTA", False: "Sin alerta"}).fillna("No modelado")
    figura = px.choropleth(
        mapa, geojson=geo, locations="departamento", featureidkey="properties.NOMBDEP", color="riesgo",
        color_continuous_scale=[(0, "#cde2fb"), (0.33, "#f7f7f5"), (0.5, "#fbd3bf"), (1, "#b8380f")],
        range_color=(0, 3), hover_name="departamento",
        hover_data={"departamento": False, "riesgo": False, "estado": True,
                    "prediccion": ":.0f", "umbral": ":.0f", "casos": ":.0f"},
        labels={"prediccion": "Pronóstico", "umbral": "Umbral de brote", "casos": "Casos observados",
                "estado": "Estado", "riesgo": "Pronóstico / umbral"},
    )
    figura.update_geos(fitbounds="locations", visible=False)
    figura.update_layout(height=620, margin=dict(l=0, r=0, t=0, b=0),
                         coloraxis_colorbar=dict(title="Pronóstico /<br>umbral", tickvals=[0, 1, 2, 3],
                                                 ticktext=["0", "1 (umbral)", "2", "3 o más"]))
    st.plotly_chart(figura, width="stretch")
    st.caption("Gris: departamentos fuera del modelo (pocos casos, o Lima, que se analiza aparte en el notebook 12).")

with der:
    st.subheader("Ranking de la semana")
    ranking = (semana.sort_values("razon_umbral", ascending=False)
               [["departamento", "prediccion", "umbral", "casos", "alerta", "brote"]]
               .rename(columns={"departamento": "Departamento", "prediccion": "Pronóstico", "umbral": "Umbral",
                                "casos": "Observados", "alerta": "Alerta", "brote": "Brote real"}))
    if tipo == "Pronostico":
        ranking = ranking.drop(columns=["Observados", "Brote real"])
    st.dataframe(ranking.round(0), hide_index=True, width="stretch", height=620)

# ------------------------------------------------------- serie de tiempo ----
st.subheader("Serie de tiempo por departamento")
depto = st.selectbox("Departamento", sorted(datos["departamento"].unique()),
                     index=sorted(datos["departamento"].unique()).index("PIURA"))
serie = datos[(datos["departamento"] == depto) & (datos["horizonte"] == horizonte)]
serie = serie[serie["modelo"] == MODELO_DEFECTO].sort_values("fecha")

fig = go.Figure()
brotes = serie[serie["brote"]]
fig.add_trace(go.Bar(x=brotes["fecha"], y=[serie["casos"].max() * 1.05] * len(brotes), marker_color="#fde7dd",
                     marker_line_width=0, width=7 * 24 * 3600 * 1000, name="Semana de brote", hoverinfo="skip"))
fig.add_trace(go.Scatter(x=serie["fecha"], y=serie["umbral"], name="Umbral (canal endémico)",
                         line=dict(color=NARANJA, dash="dash", width=1.4)))
fig.add_trace(go.Scatter(x=serie["fecha"], y=serie["casos"], name="Casos observados", line=dict(color=TINTA, width=2)))
fig.add_trace(go.Scatter(x=serie["fecha"], y=serie["prediccion"], name=f"Pronóstico a {horizonte} semanas ({MODELO_DEFECTO})",
                         line=dict(color=AZUL, width=1.5)))
futuro = serie[serie["tipo"] == "Pronostico"]
fig.add_trace(go.Scatter(x=futuro["fecha"], y=futuro["prediccion"], mode="markers", name="Pronóstico 2026",
                         marker=dict(color=NARANJA, size=9)))
fig.update_layout(height=430, barmode="overlay", plot_bgcolor="white", hovermode="x unified",
                  legend=dict(orientation="h", y=-0.15), margin=dict(l=0, r=0, t=10, b=0),
                  yaxis=dict(title="Casos por semana", gridcolor=REJ), xaxis=dict(gridcolor=REJ))
st.plotly_chart(fig, width="stretch")

# --------------------------------------------------------------- notas -----
with st.expander("Cómo leer este tablero y sus limitaciones"):
    st.markdown("""
- **Validación 2016–2024:** cada año se pronosticó con modelos entrenados solo con los años anteriores (notebook 11).
- **Prueba fuera de muestra 2025:** modelos entrenados hasta 2024 y aplicados a 2025, que nunca vieron (notebook 13).
- **Pronóstico:** semanas 1 a 4 de 2026, a partir del último dato disponible (semana 53 de 2025).
- **Modelo recomendado: XGBoost sin clima.** Con 9 años de validación, el clima no mejoró el pronóstico de forma consistente.
- **Limitaciones:**
  - Los casos de 2025 son provisionales, y los datos abiertos subestiman un poco el total oficial (−3 % a −6 % en años epidémicos).
  - Los modelos suponen que los casos de las últimas semanas ya están completos; en la práctica llegan con retraso.
  - Anticipar el **inicio** de un brote es difícil: los modelos lo logran en un 12–28 % de los casos.
- Fuentes: CDC-MINSA (datos abiertos, ODC-By), clima de la tabla del equipo, ICEN (ENFEN), mapa *peru-geojson* (MPL-2.0).
""")
