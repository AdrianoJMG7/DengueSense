# -*- coding: utf-8 -*-
"""
07_panel_region.py - Fase 1 de la guia metodologica (y paso 3.1 de la guia de
limpieza): panel departamento x semana.

  - casos: suma de los casos de los distritos del departamento (ubigeo -> depto,
    exacto; no pasa por la celda, porque 24 celdas abarcan mas de un depto).
  - clima: promedio de las celdas que contienen al menos un distrito del
    departamento, PONDERADO por los casos de entrenamiento (2010-2021) de los
    distritos del departamento en cada celda. Asi el clima representa donde
    ocurre el dengue: con el promedio simple, Piura salia 2.2 grados mas fria y
    5 puntos mas humeda por incluir celdas de sierra sin casos. Solo se usan
    anos de entrenamiento para no filtrar informacion de validacion o prueba.
    Si un departamento no tiene casos en esos anos, se usa el promedio simple.
    La comparacion queda en outputs/clima_ponderado_vs_simple.csv.
  - ICEN y fecha: iguales para todas las celdas; se toman del panel de celdas.

Solo variables base del clima: los rezagos se crean en la fase de modelado
segun el horizonte. Las *_anomalia no se usan (posible fuga).

Poblacion e incidencia (casos por 100 000 habitantes):
  - Fuente: INEI, poblacion proyectada al 30 de junio por distrito, 2018-2026
    (data/external/poblacion_distrital_2018_2026.csv). Se suman TODOS los
    distritos del departamento (codigo = primeros 2 digitos del ubigeo).
  - 2010-2017 no estan en la fuente: se estiman hacia atras con la tasa de
    crecimiento anual compuesta de cada departamento entre 2018 y 2024
    (SUPUESTO; las proyecciones 2018+ incluyen la migracion venezolana, asi
    que la poblacion de 2010-2017 puede quedar subestimada).

Salida: data/processed/panel_depto_semana.parquet
"""
from pathlib import Path

import numpy as np
import pandas as pd

from comun import ENTRENAMIENTO

ROOT = Path(__file__).resolve().parents[1]
INTERIM = ROOT / "data" / "interim"
PROC = ROOT / "data" / "processed"
OUT = ROOT / "outputs"

CLIMA_BASE = ["temp_media", "temp_max", "temp_min", "rango_term", "precip_total",
              "precip_max_d", "dias_lluvia", "hum_rel", "viento"]
CLAVE = ["departamento", "ano", "semana"]

panel_celda = pd.read_parquet(PROC / "panel_celda_semana.parquet")
casos_depto = pd.read_parquet(INTERIM / "casos_depto_semana.parquet")
mapa = pd.read_parquet(INTERIM / "mapa_typed.parquet")

# --------------------------------------------- calendario y clima por depto -
celdas_depto = mapa[["departamento", "celda_grilla"]].drop_duplicates()
print(f"Departamentos: {celdas_depto['departamento'].nunique()}, "
      f"pares depto-celda: {len(celdas_depto)}")

clima = panel_celda[["celda_grilla", "ano", "semana"] + CLIMA_BASE]

# Peso de cada (depto, celda): casos de entrenamiento de sus distritos
casos_limpios = pd.read_parquet(INTERIM / "casos_limpios.parquet", columns=["ubigeo", "ano"])
en_train = casos_limpios["ano"].between(*ENTRENAMIENTO)
peso = (casos_limpios[en_train].groupby("ubigeo").size().rename("peso").reset_index()
        .merge(mapa[["ubigeo", "departamento", "celda_grilla"]], on="ubigeo")
        .groupby(["departamento", "celda_grilla"])["peso"].sum().reset_index())
celdas_depto = celdas_depto.merge(peso, on=["departamento", "celda_grilla"], how="left")
celdas_depto["peso"] = celdas_depto["peso"].fillna(0)
sin_casos = celdas_depto.groupby("departamento")["peso"].transform("sum") == 0
celdas_depto.loc[sin_casos, "peso"] = 1  # promedio simple si no hay casos de entrenamiento
print("Departamentos sin casos de entrenamiento (clima con promedio simple):",
      sorted(celdas_depto.loc[sin_casos, "departamento"].unique()))

unido = celdas_depto.merge(clima, on="celda_grilla", how="inner")
ponderado = unido[CLIMA_BASE].mul(unido["peso"], axis=0)
ponderado[CLAVE] = unido[CLAVE]
ponderado["peso"] = unido["peso"]
sumas = ponderado.groupby(CLAVE)[CLIMA_BASE + ["peso"]].sum()
clima_depto = sumas[CLIMA_BASE].div(sumas["peso"], axis=0).reset_index()
clima_simple = unido.groupby(CLAVE)[CLIMA_BASE].mean()

resumen_celdas = celdas_depto.groupby("departamento").agg(
    n_celdas=("celda_grilla", "size"),
    pct_casos_celda_principal=("peso", lambda s: round(100 * s.max() / s.sum(), 1)),
).reset_index()
clima_depto = clima_depto.merge(resumen_celdas[["departamento", "n_celdas"]], on="departamento")

# Comparacion ponderado vs. simple (media de la diferencia y correlacion semanal)
comp = clima_depto.set_index(CLAVE)[CLIMA_BASE].join(clima_simple.add_suffix("_simple"))
filas_comp = []
for depto, d in comp.groupby(level="departamento"):
    fila = {"departamento": depto}
    for v in ["temp_media", "precip_total", "hum_rel"]:
        fila[f"{v}_dif_media"] = (d[v] - d[f"{v}_simple"]).mean()
        fila[f"{v}_corr"] = d[v].corr(d[f"{v}_simple"])
    filas_comp.append(fila)
(resumen_celdas.merge(pd.DataFrame(filas_comp), on="departamento")
 .to_csv(OUT / "clima_ponderado_vs_simple.csv", index=False, float_format="%.3f"))

icen_cols = [c for c in panel_celda.columns if c.startswith("icen")]
calendario = panel_celda[["ano", "semana", "fecha", "mes"] + icen_cols].drop_duplicates(["ano", "semana"])

n_esperado = celdas_depto["departamento"].nunique() * len(calendario)
assert len(clima_depto) == n_esperado, f"clima_depto: {len(clima_depto)} vs {n_esperado}"

# ------------------------------------------------------------ unir casos ---
panel = clima_depto.merge(casos_depto, on=CLAVE, how="left")
assert len(panel) == n_esperado, "la union con casos creo filas"
panel["casos"] = panel["casos"].fillna(0).astype(int)
panel = panel.merge(calendario, on=["ano", "semana"], how="left")
assert len(panel) == n_esperado, "la union con el calendario creo filas"
panel = panel.sort_values(CLAVE).reset_index(drop=True)

# ------------------------------------------------- poblacion e incidencia --
EXTERNAL = ROOT / "data" / "external"
ANIO_BASE, ANIO_TASA = 2018, 2024
pob = pd.read_csv(EXTERNAL / "poblacion_distrital_2018_2026.csv", dtype={"ubigeo": str})
codigo_depto = (mapa.assign(cod=mapa["ubigeo"].str[:2])[["cod", "departamento"]]
                .drop_duplicates())
assert not codigo_depto["cod"].duplicated().any(), "un codigo con dos nombres de departamento"
pob_depto = (pob.assign(cod=pob["ubigeo"].str[:2])
             .groupby(["cod", "anio"], as_index=False)["poblacion"].sum()
             .merge(codigo_depto, on="cod", how="inner")
             .pivot(index="departamento", columns="anio", values="poblacion"))
tasa = (pob_depto[ANIO_TASA] / pob_depto[ANIO_BASE]) ** (1 / (ANIO_TASA - ANIO_BASE)) - 1
for anio in range(panel["ano"].min(), ANIO_BASE):
    pob_depto[anio] = pob_depto[ANIO_BASE] / (1 + tasa) ** (ANIO_BASE - anio)
pob_largo = (pob_depto.stack().rename("poblacion").reset_index()
             .rename(columns={"anio": "ano"}))
pob_largo["poblacion_estimada"] = pob_largo["ano"] < ANIO_BASE
panel = panel.merge(pob_largo, on=["departamento", "ano"], how="left")
assert len(panel) == n_esperado, "la union con la poblacion creo filas"
assert panel["poblacion"].notna().all(), "departamento o ano sin poblacion"
panel["poblacion"] = panel["poblacion"].round().astype(int)
panel["incidencia"] = panel["casos"] / panel["poblacion"] * 100_000
print(f"Poblacion: INEI {ANIO_BASE}-2026; {panel['ano'].min()}-{ANIO_BASE - 1} estimada "
      f"(crecimiento anual {tasa.min():.2%} a {tasa.max():.2%})")

# ---------------------------------------------------------------- chequeos -
assert not panel.duplicated(CLAVE).any()
assert (panel["casos"] >= 0).all()
casos_esperados = casos_depto.loc[casos_depto["ano"].between(panel["ano"].min(), panel["ano"].max()), "casos"].sum()
assert panel["casos"].sum() == casos_esperados, "se perdieron casos al unir"
total_celdas = panel_celda["casos"].sum()
print(f"Panel depto x semana: {len(panel):,} filas, {panel['departamento'].nunique()} deptos, "
      f"anos {panel['ano'].min()}-{panel['ano'].max()}")
print(f"Casos: {panel['casos'].sum():,} (panel de celdas: {total_celdas:,})")
print(f"Semanas depto sin casos: {(panel['casos'] == 0).mean():.1%}")

resumen = (panel.groupby("departamento")
           .agg(n_celdas=("n_celdas", "first"), casos=("casos", "sum"),
                poblacion_2024=("poblacion", "last"),
                incidencia_anual_media=("incidencia", lambda s: s.sum() / panel["ano"].nunique()),
                semanas_con_casos=("casos", lambda s: int((s > 0).sum())))
           .sort_values("casos", ascending=False))
resumen.to_csv(OUT / "panel_depto_resumen.csv")
panel.to_parquet(PROC / "panel_depto_semana.parquet", index=False)
print("Guardado data/processed/panel_depto_semana.parquet y outputs/panel_depto_resumen.csv")
