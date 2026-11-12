# -*- coding: utf-8 -*-
"""
Genera las figuras y tablas del articulo (docs/articulo/figuras y docs/articulo/tablas)
a partir de los resultados del proyecto, para que ningun numero se copie a mano.

    python docs/articulo/generar_material.py
"""
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import PatchCollection
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.patches import Polygon

ROOT = Path(__file__).resolve().parents[2]
AQUI = Path(__file__).resolve().parent
FIG, TAB = AQUI / "figuras", AQUI / "tablas"
FIG.mkdir(exist_ok=True)
TAB.mkdir(exist_ok=True)
PROC, INTERIM, OUT, RAW = ROOT / "data/processed", ROOT / "data/interim", ROOT / "outputs", ROOT / "data/raw"
REGIONES = ["PIURA", "LORETO", "SAN MARTIN"]
AZUL, NARANJA, TINTA, TINTA_2, REJ = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e1e0d9"
plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})


def num(x, dec=2):
    """Numero con coma decimal y espacio fino de miles, para LaTeX en espanol."""
    if pd.isna(x):
        return "--"
    s = f"{x:,.{dec}f}".replace(",", "X").replace(".", ",").replace("X", "\\,")
    return s.replace("-", "$-$")


def titulo(d):
    return {"SAN MARTIN": "San Martín", "MADRE DE DIOS": "Madre de Dios", "HUANUCO": "Huánuco",
            "JUNIN": "Junín", "ANCASH": "Áncash", "LA LIBERTAD": "La Libertad"}.get(d, d.title())


# --------------------------------------------- 1. figuras ya existentes ----
for nombre in ["fase3_comparacion_regiones.png", "fase3_shap_categorias.png", "fase5_h2_por_anio.png",
               "fase5_alertas.png", "lima_clima_2022_2024.png", "fuera_de_muestra_2025.png",
               "fase2_lineas_base_2024.png", "fase4_lstm_comparacion.png"]:
    shutil.copy(OUT / nombre, FIG / nombre)

# ----------------------------------------- 2. casos anuales del Peru -------
casos_depto = pd.read_parquet(INTERIM / "casos_depto_semana.parquet")
anual = casos_depto.groupby("ano")["casos"].sum()
crudo25 = pd.read_csv(RAW / "datos_abiertos_vigilancia_dengue_2000_2025.csv", dtype=str, keep_default_na=False,
                      usecols=["ano"])
anual.loc[2025] = int((crudo25["ano"] == "2025").sum())
fig, ax = plt.subplots(figsize=(7.2, 2.4))
colores = [NARANJA if a in (2017, 2023, 2024) else AZUL for a in anual.index]
ax.bar(anual.index, anual.values / 1000, color=colores, width=0.75)
for a in (2017, 2023, 2024):
    ax.annotate(f"{anual[a] / 1000:.0f}".replace(".", ","), (a, anual[a] / 1000), ha="center", va="bottom", fontsize=7.5,
                color=TINTA_2, xytext=(0, 2), textcoords="offset points")
ax.set_ylabel("Casos (miles)")
ax.set_xticks(range(2000, 2026, 2))
ax.grid(axis="y", color=REJ, lw=0.6)
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig(FIG / "casos_anuales.pdf")
fig.savefig(FIG / "casos_anuales.png", dpi=200)
plt.close(fig)

# --------------------------------------- 3. mapa de incidencia media -------
panel = pd.read_parquet(PROC / "panel_depto_semana.parquet")
inc = (panel.groupby("departamento").agg(casos=("casos", "sum"), pob=("poblacion", "mean"))
       .assign(incidencia=lambda d: d["casos"] / panel["ano"].nunique() / d["pob"] * 1e5))
geo = json.loads((ROOT / "app/peru_departamental.geojson").read_text(encoding="utf-8"))
modelados = set(pd.read_parquet(PROC / "tablero.parquet")["departamento"].unique())
cmap = LinearSegmentedColormap.from_list("az", ["#eef4fc", "#86b6ef", "#256abf", "#0d366b"])
norma = LogNorm(vmin=1, vmax=1000)
fig, ax = plt.subplots(figsize=(3.5, 4.4))
parches, valores, bordes = [], [], []
for f in geo["features"]:
    d = f["properties"]["NOMBDEP"]
    geom = f["geometry"]
    poligonos = geom["coordinates"] if geom["type"] == "MultiPolygon" else [geom["coordinates"]]
    for pol in poligonos:
        parches.append(Polygon(np.array(pol[0]), closed=True))
        valores.append(inc["incidencia"].get(d, np.nan))
        bordes.append(TINTA if d in modelados else "white")
col = PatchCollection(parches, cmap=cmap, norm=norma, edgecolor=bordes, linewidth=[0.8 if b == TINTA else 0.4 for b in bordes])
valores = np.array(valores, dtype=float)
col.set_array(np.where(np.isnan(valores), 1, np.clip(valores, 1, 1000)))
ax.add_collection(col)
ax.autoscale_view()
ax.set_aspect("equal")
ax.axis("off")
for d, (x, y) in {"PIURA": (-80.3, -5.1), "LORETO": (-74.5, -4.2), "SAN MARTIN": (-76.8, -7.0), "LIMA": (-76.6, -11.8),
                  "UCAYALI": (-73.6, -9.6), "TUMBES": (-80.4, -3.4)}.items():
    ax.text(x, y, titulo(d), fontsize=6.5, ha="center", color="white" if inc["incidencia"].get(d, 0) > 300 else TINTA)
barra = fig.colorbar(col, ax=ax, fraction=0.04, pad=0.02, shrink=0.7)
barra.set_label("Casos por 100 000 hab. al año", fontsize=7.5)
barra.set_ticks([1, 10, 100, 1000], labels=["1", "10", "100", "1000"])
barra.ax.tick_params(labelsize=7)
fig.tight_layout()
fig.savefig(FIG / "mapa_incidencia.pdf")
fig.savefig(FIG / "mapa_incidencia.png", dpi=200)
plt.close(fig)

# ================================================================ tablas =====
# Tabla: R2 en prueba 2024 por region y modelo
lstm = pd.read_csv(OUT / "fase4_lstm_metricas.csv")
prueba = lstm[lstm["conjunto"] == "Prueba"]
modelos = [("Persistencia", "Persistencia"), ("Regresion con rezagos", "Regresión"), ("XGBoost sin clima", "XGBoost sin clima"),
           ("XGBoost con clima", "XGBoost con clima"), ("LSTM con clima", "LSTM con clima")]
filas = []
for depto in REGIONES:
    for i, (m, etiqueta) in enumerate(modelos):
        r = prueba[(prueba["departamento"] == depto) & (prueba["modelo"] == m)].set_index("horizonte")["r2"]
        mostrados = prueba[(prueba["departamento"] == depto) & prueba["modelo"].isin([x for x, _ in modelos])]
        mejor = mostrados.groupby("horizonte")["r2"].max()
        celdas = [(f"\\textbf{{{num(r[h])}}}" if abs(r[h] - mejor[h]) < 1e-9 else num(r[h])) for h in [1, 2, 3, 4]]
        region = f"\\multirow{{5}}{{*}}{{{titulo(depto)}}}" if i == 0 else ""
        filas.append(f"{region} & {etiqueta} & " + " & ".join(celdas) + " \\\\")
    filas.append("\\hline")
(TAB / "r2_prueba_2024.tex").write_text("\n".join(filas[:-1]) + "\n", encoding="utf-8")

# Tabla: validacion con origen movil (H2)
rolling = pd.read_csv(OUT / "fase5_validacion_temporal_metricas.csv")
pred = pd.read_parquet(PROC / "predicciones_validacion_temporal.parquet")
total = (pred.assign(e=(pred["casos"] - pred["prediccion"]).abs())
         .groupby(["departamento", "horizonte", "modelo"])["e"].mean().unstack())
mejora = (total["XGBoost sin clima"] - total["XGBoost con clima"]) / total["XGBoost sin clima"] * 100
p = rolling.pivot_table(index=["anio_evaluado", "departamento", "horizonte"], columns="modelo", values="mae")
gana = (p["XGBoost con clima"] < p["XGBoost sin clima"]).groupby(level=["anio_evaluado", "horizonte"]).sum().unstack()
vs_pers = (total["XGBoost sin clima"] < total["Persistencia"]).groupby(level="horizonte").sum()
filas = []
for h in [1, 2, 3, 4]:
    filas.append(f"{h} & {int((mejora.xs(h, level='horizonte') > 0).sum())}/17 & {num(mejora.xs(h, level='horizonte').median(), 1)} "
                 f"& {int((gana[h] > 8.5).sum())}/9 & {int(vs_pers[h])}/17 \\\\")
(TAB / "h2_origen_movil.tex").write_text("\n".join(filas) + "\n", encoding="utf-8")

# Tabla: alertas
al = pd.read_csv(OUT / "fase5_alertas.csv")
filas = []
for h in [2, 4]:
    for i, (m, etiqueta) in enumerate(modelos[:4]):
        r = al[(al["horizonte"] == h) & (al["modelo"] == m)].iloc[0]
        hcol = f"\\multirow{{4}}{{*}}{{{h}}}" if i == 0 else ""
        filas.append(f"{hcol} & {etiqueta} & {num(r.precision * 100, 1)} & {num(r.recall * 100, 1)} & "
                     f"{num(r.falsas_alarmas_por_depto_anio, 2)} & {num(r.inicios_anticipados * 100, 1)} \\\\")
    filas.append("\\hline")
(TAB / "alertas.tex").write_text("\n".join(filas[:-1]) + "\n", encoding="utf-8")

# Tabla: Lima
lp = pd.read_csv(OUT / "lima_predicciones_h2.csv")
err = lp.assign(e=(lp["casos"] - lp["prediccion"]).abs()).pivot_table(index="modelo", columns="anio", values="e", aggfunc="mean")
orden = ["Persistencia", "XGBoost solo Lima", "XGBoost costa, sin clima", "XGBoost costa, con clima"]
(TAB / "lima.tex").write_text("\n".join(f"{m} & {num(err.loc[m, 2023], 0)} & {num(err.loc[m, 2024], 0)} \\\\" for m in orden) + "\n",
                              encoding="utf-8")

# Tabla: prueba fuera de muestra 2025 y LSTM (MAE mediano)
oos = pd.read_csv(OUT / "fuera_de_muestra_2025_metricas.csv")
mo = oos.pivot_table(index=["departamento", "horizonte"], columns="modelo", values="mae")
g25 = (mo["XGBoost sin clima"] < mo["Persistencia"]).groupby(level="horizonte").sum()
m25 = ((mo["Persistencia"] - mo["XGBoost sin clima"]) / mo["Persistencia"] * 100).groupby(level="horizonte").median()
med = prueba.groupby(["modelo", "horizonte"])["mae"].median().unstack()
filas = [f"{h} & {int(g25[h])}/17 & {num(m25[h], 1)} & {num(med.loc['XGBoost sin clima', h], 1)} & "
         f"{num(med.loc['LSTM sin clima', h], 1)} & {num(med.loc['LSTM con clima', h], 1)} \\\\" for h in [1, 2, 3, 4]]
(TAB / "oos_lstm.tex").write_text("\n".join(filas) + "\n", encoding="utf-8")

# Cifras sueltas para el texto
print("casos anuales:", {int(a): int(c) for a, c in anual.loc[[2017, 2022, 2023, 2024, 2025]].items()})
print("incidencia (top):", inc["incidencia"].sort_values(ascending=False).head(6).round(0).to_dict())
print("panel:", len(panel), "filas,", int(panel["casos"].sum()), "casos")
print("R2 Piura XGB con clima h2/h3:", prueba[(prueba.departamento == "PIURA") & (prueba.modelo == "XGBoost con clima")].set_index("horizonte")["r2"].round(3).to_dict())
print("archivos:", sorted(p.name for p in FIG.iterdir()), sorted(p.name for p in TAB.iterdir()))

# ------------------------------------- diagrama del metodo (para Word) -----
pasos = [("Datos", "casos CDC-MINSA, clima por celda, ICEN, población INEI"),
         ("Limpieza y panel", "departamento × semana; clima ponderado por casos; control de fuga"),
         ("Variables (hasta T − h)", "casos y clima rezagados, ICEN, estacionalidad"),
         ("Modelos", "persistencia, estacional, regresión, XGBoost con/sin clima, LSTM"),
         ("Evaluación", "prueba 2024, origen móvil 2016–2024, alertas, fuera de muestra 2025"),
         ("Salidas", "SHAP por región, pronóstico 2026, mapa de riesgo e interfaz")]
fig, ax = plt.subplots(figsize=(4.8, 3.9))
fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
ax.axis("off")
alto, sep = 0.125, 0.04
for i, (t, d) in enumerate(pasos):
    y = 1 - i * (alto + sep) - alto
    color = "#fdf1eb" if i == len(pasos) - 1 else "#eef4fc"
    ax.add_patch(matplotlib.patches.FancyBboxPatch((0.02, y), 0.96, alto, boxstyle="round,pad=0.005,rounding_size=0.015",
                                                   fc=color, ec="#52514e", lw=0.8, transform=ax.transAxes))
    ax.text(0.5, y + alto * 0.66, t, ha="center", va="center", fontsize=8.5, fontweight="bold", transform=ax.transAxes)
    ax.text(0.5, y + alto * 0.28, d, ha="center", va="center", fontsize=6.6, color="#52514e", transform=ax.transAxes)
    if i < len(pasos) - 1:
        ax.annotate("", xy=(0.5, y - sep + 0.004), xytext=(0.5, y - 0.004), xycoords="axes fraction",
                    arrowprops=dict(arrowstyle="-|>", color="#52514e", lw=0.9))
fig.savefig(FIG / "flujo.png", dpi=250, bbox_inches="tight")
plt.close(fig)
print("diagrama: figuras/flujo.png")
