# -*- coding: utf-8 -*-
"""
05_panel.py — Paso 7 de la guia: unir todo en un panel celda x semana.

  clima (calendario completo)
    + casos_celda (rellenados con 0)
    + icen_semanal

Chequeos: el numero de filas del panel debe ser igual al del clima tras
cada union. Salida: data/processed/panel_celda_semana.parquet
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INTERIM = ROOT / "data" / "interim"
PROC = ROOT / "data" / "processed"
OUT = ROOT / "outputs"
PROC.mkdir(exist_ok=True, parents=True)

clima = pd.read_parquet(INTERIM / "clima_typed.parquet")
casos_celda = pd.read_parquet(INTERIM / "casos_celda_semana.parquet")
icen_sem = pd.read_parquet(INTERIM / "icen_semanal.parquet")

n0 = len(clima)

panel = clima.merge(casos_celda, on=["celda_grilla", "ano", "semana"], how="left")
assert len(panel) == n0, f"join casos creo filas: {len(panel)} vs {n0}"
panel["casos"] = panel["casos"].fillna(0).astype(int)

panel = panel.merge(icen_sem, on=["ano", "semana"], how="left")
assert len(panel) == n0, f"join icen creo filas: {len(panel)} vs {n0}"

# Fecha aproximada de la semana (lunes de la semana epidemiologica)
panel["fecha"] = (pd.to_datetime(panel["ano"].astype(str) + "-01-01")
                  + pd.to_timedelta(7 * (panel["semana"] - 1), unit="D"))
panel["mes"] = panel["fecha"].dt.month

# Chequeos basicos
assert (panel["casos"] >= 0).all() and panel["casos"].dtype.kind == "i"
assert not panel.duplicated(["celda_grilla", "ano", "semana"]).any()
sem_casos = panel["casos"].sum()
print(f"Panel celda x semana: {len(panel):,} filas x {panel.shape[1]} cols")
print(f"Celdas: {panel['celda_grilla'].nunique()}, "
      f"anos {panel['ano'].min()}-{panel['ano'].max()}")
print(f"Casos totales en panel: {sem_casos:,}")
print(f"Semanas celda sin casos (ceros): {(panel['casos'] == 0).mean() * 100:.1f}%")
print(f"ICEN NA: {panel['icen'].isna().sum()} filas "
      f"({panel.loc[panel['icen'].isna(), 'ano'].unique()})")

panel.to_parquet(PROC / "panel_celda_semana.parquet", index=False)

# ------------------------------------------------- mapa celda -> depto -----
# Para agregar a nivel departamento: cada celda se asigna al departamento
# con mas distritos (mapa_ubigeo_celda da ubigeo->departamento->celda).
mapa = pd.read_parquet(INTERIM / "mapa_typed.parquet")
celda_depto = (mapa.groupby(["celda_grilla", "departamento"])
               .size().reset_index(name="n")
               .sort_values("n", ascending=False)
               .drop_duplicates("celda_grilla")
               [["celda_grilla", "departamento"]])
celda_depto.to_parquet(INTERIM / "celda_depto.parquet", index=False)
print(f"\nMapa celda->depto: {len(celda_depto)} celdas asignadas")
print("Celdas sin distrito mapeado:",
      panel['celda_grilla'].nunique() - len(celda_depto))
