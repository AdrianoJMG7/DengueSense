# -*- coding: utf-8 -*-
"""
12_exportar_interfaz_html.py - Genera DengueSense.html: la interfaz del proyecto
en un solo archivo HTML que cualquiera puede abrir con doble clic, sin Python,
sin instalar nada y sin internet.

Incrusta los datos (los mismos que usa app/tablero.py) y la libreria Plotly.js.
Volver a ejecutarlo cada vez que cambien los resultados:

    python scripts/12_exportar_interfaz_html.py

Entradas: data/processed/tablero.parquet, data/processed/panel_depto_semana.parquet,
outputs/*.csv y app/peru_departamental.geojson.
Salida: DengueSense.html (raiz del proyecto).
"""
import datetime as dt
import json
from pathlib import Path

import numpy as np
import pandas as pd
from plotly.offline import get_plotlyjs

ROOT = Path(__file__).resolve().parents[1]
PROC, OUT = ROOT / "data" / "processed", ROOT / "outputs"
SALIDA = ROOT / "DengueSense.html"

MODELO_DEFECTO = "XGBoost sin clima"
ORDEN_MODELOS = ["Persistencia", "Regresion con rezagos", "XGBoost sin clima", "XGBoost con clima",
                 "LSTM sin clima", "LSTM con clima"]
TIPOS = ["Pronostico", "Fuera de muestra 2025", "Validacion 2016-2024"]
CATS = ["temperatura", "lluvia", "humedad", "viento", "icen"]


def limpio(x):
    """Convierte NaN en None y numpy en tipos de Python, para JSON."""
    if isinstance(x, (list, tuple)):
        return [limpio(v) for v in x]
    if isinstance(x, dict):
        return {k: limpio(v) for k, v in x.items()}
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        return None if np.isnan(x) else round(float(x), 3)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return x


# ------------------------------------------------------------- tablero -----
tab = pd.read_parquet(PROC / "tablero.parquet")
tab["fecha"] = pd.to_datetime(tab["fecha"])
deptos = sorted(tab["departamento"].unique())
modelos = [m for m in ORDEN_MODELOS if m in tab["modelo"].unique()]
semanas = tab[["ano", "semana", "fecha"]].drop_duplicates(["ano", "semana"]).sort_values(["ano", "semana"])
idx_sem = {(a, s): i for i, (a, s) in enumerate(zip(semanas["ano"], semanas["semana"]))}
tab = tab[tab["modelo"].isin(modelos)]
filas = {
    "d": tab["departamento"].map({d: i for i, d in enumerate(deptos)}).tolist(),
    "w": [idx_sem[(a, s)] for a, s in zip(tab["ano"], tab["semana"])],
    "h": tab["horizonte"].tolist(),
    "m": tab["modelo"].map({m: i for i, m in enumerate(modelos)}).tolist(),
    "t": tab["tipo"].map({t: i for i, t in enumerate(TIPOS)}).tolist(),
    "c": [None if pd.isna(v) else int(v) for v in tab["casos"]],
    "p": tab["prediccion"].round(1).tolist(),
    "u": tab["umbral"].round(1).tolist(),
    "al": tab["alerta"].astype(int).tolist(),
    "b": tab["brote"].astype(int).tolist(),
}
datos = {
    "deptos": deptos, "modelos": modelos, "tipos": TIPOS, "modelo_defecto": modelos.index(MODELO_DEFECTO),
    "semanas": {"a": semanas["ano"].tolist(), "s": semanas["semana"].tolist(),
                "f": semanas["fecha"].dt.strftime("%Y-%m-%d").tolist()},
    "filas": filas,
}

# ---------------------------------------------------------- desempeno ------
rolling = pd.read_csv(OUT / "fase5_validacion_temporal_metricas.csv")
med = rolling.groupby(["modelo", "horizonte"])["mae"].median().unstack()
datos["mae_validacion"] = {m: med.loc[m].tolist() for m in ORDEN_MODELOS if m in med.index}
p = rolling.pivot_table(index=["anio_evaluado", "departamento", "horizonte"], columns="modelo", values="mae")
gana = (p["XGBoost con clima"] < p["XGBoost sin clima"]).groupby(level=["anio_evaluado", "horizonte"]).sum().unstack()
datos["h2"] = {"anios": gana.index.tolist(), "por_h": {int(h): gana[h].tolist() for h in gana.columns},
               "n": int(rolling["departamento"].nunique())}
alertas = pd.read_csv(OUT / "fase5_alertas.csv")
datos["alertas"] = {f"{int(r.horizonte)}|{r.modelo}": [r.recall, r.precision, r.inicios_anticipados]
                    for r in alertas.itertuples()}
oos = pd.read_csv(OUT / "fuera_de_muestra_2025_metricas.csv")
m = oos.pivot_table(index=["departamento", "horizonte"], columns="modelo", values="mae")
g = (m[MODELO_DEFECTO] < m["Persistencia"])
datos["oos"] = {"por_h": g.groupby(level="horizonte").sum().tolist(), "total": float(g.mean())}
lstm = pd.read_csv(OUT / "fase4_lstm_metricas.csv")
lm = lstm[lstm["conjunto"] == "Prueba"].groupby(["modelo", "horizonte"])["mae"].median().unstack()
datos["lstm"] = {m: lm.loc[m].tolist() for m in ORDEN_MODELOS if m in lm.index}

# ---------------------------------------------------------------- shap -----
shap = pd.read_csv(OUT / "fase3_shap_importancia.csv")
datos["shap"] = {}
for h, s in shap.groupby("horizonte"):
    por_cat = s.groupby(["departamento", "categoria"])["pct"].sum().unstack(fill_value=0)
    clima = por_cat[CATS].div(por_cat[CATS].sum(axis=1), axis=0) * 100
    orden = [d for d in s.groupby("departamento")["importancia"].sum().sort_values(ascending=False).index]
    top = {d: s[s["departamento"] == d].nlargest(12, "pct")[["variable", "categoria", "pct"]].values.tolist() for d in orden}
    datos["shap"][int(h)] = {"deptos": orden, "matriz": clima.loc[orden].round(1).values.tolist(), "top": top,
                             "casos_pct": {d: float(por_cat.loc[d, ["casos pasados", "estacionalidad"]].sum()) for d in orden}}

# ---------------------------------------------------------------- lima -----
panel = pd.read_parquet(PROC / "panel_depto_semana.parquet")
pl = panel[panel["departamento"] == "LIMA"].sort_values(["ano", "semana"])
anual = pl.groupby("ano")["casos"].sum()
ref = pl[pl["ano"] <= 2022].groupby(pl["semana"].clip(upper=52))["temp_media"].mean()
pl = pl.assign(anomalia=pl["temp_media"] - pl["semana"].clip(upper=52).map(ref))
v = pl[pl["ano"] >= 2022]
lp = pd.read_csv(OUT / "lima_predicciones_h2.csv")
err = (lp.assign(e=(lp["casos"] - lp["prediccion"]).abs()).pivot_table(index="modelo", columns="anio", values="e", aggfunc="mean"))
datos["lima"] = {
    "anual": {"a": anual.index.tolist(), "c": anual.tolist()},
    "clima": {"f": pd.to_datetime(v["fecha"]).dt.strftime("%Y-%m-%d").tolist(), "c": v["casos"].tolist(),
              "anom": v["anomalia"].round(2).tolist(), "icen": v["icen"].tolist()},
    "pred": {mod: {"f": d["fecha"].tolist(), "c": d["casos"].tolist(), "p": d["prediccion"].round(0).tolist()}
             for mod, d in lp.groupby("modelo")},
    "error": {mod: [round(err.loc[mod, 2023]), round(err.loc[mod, 2024])] for mod in err.index},
}

# ------------------------------------------------------ validacion externa -
ext = pd.read_csv(OUT / "validacion_externa_nacional.csv")
datos["externa"] = {"a": ext["ano"].tolist(), "panel": ext["panel"].tolist(), "od": ext["opendengue_admin0"].tolist()}
datos["generado"] = dt.date.today().strftime("%d/%m/%Y")

geo = json.loads((ROOT / "app" / "peru_departamental.geojson").read_text(encoding="utf-8"))
geo["features"] = [{"type": "Feature", "properties": {"NOMBDEP": f["properties"]["NOMBDEP"]}, "geometry": f["geometry"]}
                   for f in geo["features"]]
datos["geo"] = geo
datos["deptos_mapa"] = sorted(f["properties"]["NOMBDEP"] for f in geo["features"])

plantilla = (ROOT / "scripts" / "plantilla_interfaz.html").read_text(encoding="utf-8")
html = (plantilla.replace("/*__PLOTLY__*/", get_plotlyjs())
        .replace("/*__DATOS__*/null", json.dumps(limpio(datos), ensure_ascii=False, separators=(",", ":"))))
SALIDA.write_text(html, encoding="utf-8")
print(f"Generado {SALIDA.name}: {SALIDA.stat().st_size / 1e6:.1f} MB "
      f"({len(filas['d']):,} pronosticos semanales, {len(deptos)} departamentos)")
