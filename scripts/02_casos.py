# -*- coding: utf-8 -*-
"""
02_casos.py — Paso 3 y 4 de la guia: limpiar la tabla de casos y construir
la variable objetivo (conteo de casos por celda x semana).

Salidas:
  data/interim/casos_limpios.parquet   (nivel caso)
  data/interim/casos_celda_semana.parquet (ubigeo->celda, conteo)
  outputs/exploracion_casos_*.csv      (tablas de conteo previas al filtro)
  outputs/embudo_filtros.csv           (filas antes/despues de cada filtro)
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INTERIM = ROOT / "data" / "interim"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

casos = pd.read_parquet(INTERIM / "casos_typed.parquet")
mapa = pd.read_parquet(INTERIM / "mapa_typed.parquet")

# ------------------------------------------------- exploracion previa ------
print("=== Exploracion previa (antes de filtrar) ===")
for col in ["enfermedad", "diagnostic", "tipo_edad", "sexo", "diresa"]:
    vc = casos[col].value_counts().head(20)
    vc.to_csv(OUT / f"exploracion_{col}.csv")
    print(f"\n{col}: {casos[col].nunique()} valores unicos")
    print(vc.head(10).to_string())

casos.groupby("ano").size().to_csv(OUT / "exploracion_ano.csv", header=["n"])
casos.groupby("semana").size().to_csv(OUT / "exploracion_semana.csv", header=["n"])
print("\nRango anos:", casos["ano"].min(), "-", casos["ano"].max())
print("Rango semanas:", casos["semana"].min(), "-", casos["semana"].max())

# ------------------------------------------------------------- limpieza ----
funnel = []


def registrar(paso: str, df: pd.DataFrame):
    funnel.append({"paso": paso, "filas": len(df)})
    print(f"  {paso:45s} -> {len(df):>10,}")


registrar("0_raw", casos)

# 1) Solo dengue segun CIE-10 (A97.x). La columna tipo_dx no existe en este
#    archivo; la definicion de caso queda por diagnostico + enfermedad.
antes = len(casos)
casos = casos[casos["diagnostic"].str.startswith("A97", na=False)]
registrar("1_solo_dengue_A97", casos)
print(f"    descartados por diagnostic no-A97: {antes - len(casos):,}")

# 2) Fecha valida: ano 2000-2024 y semana epidemiologica 1-53
antes = len(casos)
casos = casos[
    casos["ano"].between(2000, 2024)
    & casos["semana"].between(1, 53)
]
registrar("2_ano_semana_validos", casos)
print(f"    descartados por fecha invalida: {antes - len(casos):,}")

# 3) ubigeo presente y mapeable a celda de grilla
mapa_ub = set(mapa["ubigeo"])
sin_celda = casos[~casos["ubigeo"].isin(mapa_ub)]
sin_celda.groupby("departamento").size().rename("n").to_csv(
    OUT / "casos_sin_celda_por_depto.csv"
)
antes = len(casos)
casos = casos[casos["ubigeo"].isin(mapa_ub)]
registrar("3_ubigeo_con_celda", casos)
print(f"    descartados por ubigeo sin celda: {antes - len(casos):,}")
print(f"    ({antes - len(casos)} filas = {((antes - len(casos)) / antes * 100):.2f}% de las filas del paso anterior)")

# 4) Edad en anos segun tipo_edad (A=anos, M=meses, D=dias) - supuesto a verificar
casos = casos.copy()
casos["edad_anos"] = np.nan
mask_a = casos["tipo_edad"] == "A"
mask_m = casos["tipo_edad"] == "M"
mask_d = casos["tipo_edad"] == "D"
casos.loc[mask_a, "edad_anos"] = casos.loc[mask_a, "edad"]
casos.loc[mask_m, "edad_anos"] = casos.loc[mask_m, "edad"] / 12
casos.loc[mask_d, "edad_anos"] = casos.loc[mask_d, "edad"] / 365
print("  tipo_edad no reconocido:", casos.loc[~casos["tipo_edad"].isin(["A", "M", "D"]), "tipo_edad"].unique())
print("  edad_anos NA:", casos["edad_anos"].isna().sum())

# 5) Duplicados exactos: se MARCAN, no se borran (no hay ID de caso)
casos["duplicado_exacto"] = casos.duplicated(keep=False)
n_dup = casos["duplicado_exacto"].sum()
print(f"  filas en grupos duplicados exactos: {n_dup:,} "
      f"({n_dup / len(casos) * 100:.2f}%) - se conservan")

# --------------------------------------------------- variable objetivo -----
casos_limpios = casos.drop(columns=["duplicado_exacto"])
casos_limpios.to_parquet(INTERIM / "casos_limpios.parquet", index=False)
print(f"\nGuardado casos_limpios: {len(casos_limpios):,} filas")

# Conteo por ubigeo x ano x semana, luego a celda de grilla
casos_sem = (
    casos_limpios.groupby(["ubigeo", "ano", "semana"], as_index=False)
    .size()
    .rename(columns={"size": "casos"})
)
casos_sem = casos_sem.merge(
    mapa[["ubigeo", "celda_grilla", "departamento"]], on="ubigeo", how="left"
)
casos_celda = (
    casos_sem.groupby(["celda_grilla", "ano", "semana"], as_index=False)["casos"]
    .sum()
)
casos_celda.to_parquet(INTERIM / "casos_celda_semana.parquet", index=False)
print(f"Guardado casos_celda_semana: {len(casos_celda):,} filas, "
      f"{casos_celda['celda_grilla'].nunique()} celdas")

# Tambien a nivel departamento (la unidad de analisis recomendada por la guia)
casos_depto = (
    casos_sem.groupby(["departamento", "ano", "semana"], as_index=False)["casos"]
    .sum()
)
casos_depto.to_parquet(INTERIM / "casos_depto_semana.parquet", index=False)
print(f"Guardado casos_depto_semana: {len(casos_depto):,} filas")

# --------------------------------------------------------------- embudo ----
pd.DataFrame(funnel).to_csv(OUT / "embudo_filtros.csv", index=False)
print("\nEmbudo guardado en outputs/embudo_filtros.csv")
