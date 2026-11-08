# -*- coding: utf-8 -*-
"""
04_icen.py — Paso 6 de la guia: preparar el ICEN.

El ICEN es mensual; se convierte a semanal asignando a cada semana el mes de
su fecha central (misma convencion de fecha que usa 05_panel.py).

Convencion de fecha: semana epidemiologica estandar (domingo a sabado, la
semana 1 contiene el 4 de enero); se usa el miercoles de cada semana. Ver
scripts/comun.py.
"""
from pathlib import Path

import pandas as pd

from comun import agregar_fecha_y_mes

ROOT = Path(__file__).resolve().parents[1]
INTERIM = ROOT / "data" / "interim"
OUT = ROOT / "outputs"

icen = pd.read_parquet(INTERIM / "icen_typed.parquet")

# 1) Unicidad y completitud mensual
n_dup = icen.duplicated(["ano", "mes"]).sum()
print(f"ICEN: {len(icen)} filas, {icen['ano'].min()}-{icen['ano'].max()}, "
      f"duplicados (ano,mes): {n_dup}")
full = pd.MultiIndex.from_product(
    [range(icen["ano"].min(), icen["ano"].max() + 1), range(1, 13)],
    names=["ano", "mes"],
)
missing = full.difference(pd.MultiIndex.from_frame(icen[["ano", "mes"]]))
if len(missing):
    print(f"Meses faltantes: {len(missing)} -> {list(missing)[:20]}")
    miss_df = pd.DataFrame(missing.tolist(), columns=["ano", "mes"])
    miss_df.to_csv(OUT / "icen_meses_faltantes.csv", index=False)

# 2) Rellenar huecos con NA (no interpolar)
icen = (icen.set_index(["ano", "mes"])
        .reindex(full)
        .reset_index())

# 3) Pasar a semanal: para cada (ano, semana) del clima, calcular la fecha
#    aproximada y tomar el mes correspondiente.
clima = pd.read_parquet(INTERIM / "clima_typed.parquet",
                        columns=["celda_grilla", "ano", "semana"])
cal = agregar_fecha_y_mes(clima[["ano", "semana"]].drop_duplicates())
# El mes de la semana debe ser del mismo ano (el ICEN se une por ano y mes)
assert (cal["fecha"].dt.year == cal["ano"]).all(), "una semana cae en otro ano"

# 4) Unir ICEN por (ano, mes); conservar una fila por (ano, semana)
icen_cols = [c for c in icen.columns if c not in ("ano", "mes")]
icen_sem = cal[["ano", "semana", "mes"]].merge(
    icen[["ano", "mes"] + icen_cols], on=["ano", "mes"], how="left"
).drop(columns="mes")
assert not icen_sem.duplicated(["ano", "semana"]).any()

icen_sem.to_parquet(INTERIM / "icen_semanal.parquet", index=False)
print(f"Guardado icen_semanal: {len(icen_sem)} filas (ano-semana), "
      f"meses sin ICEN: {icen_sem['icen'].isna().sum()}")
print("Rango:", icen_sem["ano"].min(), "-", icen_sem["ano"].max())
