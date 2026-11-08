# -*- coding: utf-8 -*-
"""
06_modelado_prep.py — Pasos 8, 9 y 10 de la guia: atributos, division
temporal y dataset de modelado.

Decisiones (documentadas en docs/diccionario.csv):
  - Horizonte h = 2 semanas.
  - Solo se usan predictores con rezago >= h; se EXCLUYEN las variables base
    sin rezago y las *_anomalia (calculadas con todos los anos = fuga).
  - Split temporal: train <= 2023, test >= 2024 (clima cubre 2010-2024;
    los casos empiezan a ser densos ~2010).
  - ICEN: solo icen_lag3..6 (ver MIN_REZAGO_ICEN).
  - Objetivo: casos (conteo) y log1p_casos. incidencia queda pendiente de
    poblacion INEI.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
OUT = ROOT / "outputs"

H = 2
ANIO_CORTE = 2024

# Rezago minimo del ICEN, en meses. Al predecir la semana t se conoce hasta la
# semana t - H. Con H = 2:
#   - icen_lag1 (mes m-1): en el 38 % de las semanas, t-2 cae en el mes m-1,
#     que aun no termina -> fuga de informacion.
#   - icen_lag2 (mes m-2): el mes ya termino, pero el ICEN se calcula con
#     medias moviles de 3 meses y se publica con retraso (supuesto a
#     confirmar con ENFEN / quien armo la tabla).
#   - icen_lag3 en adelante: disponibles al predecir.
MIN_REZAGO_ICEN = 3

panel = pd.read_parquet(PROC / "panel_celda_semana.parquet")
print(f"Panel: {len(panel):,} filas")

# ------------------------------------------------- atributos ---------------
panel = panel.sort_values(["celda_grilla", "ano", "semana"]).reset_index(drop=True)
g = panel.groupby("celda_grilla", sort=False)

# Casos pasados: solo rezagos >= H
for lag in (2, 3, 4):
    panel[f"casos_lag{lag}"] = g["casos"].shift(lag)

# Estacionalidad
panel["sem_sin"] = np.sin(2 * np.pi * panel["semana"] / 52)
panel["sem_cos"] = np.cos(2 * np.pi * panel["semana"] / 52)

# Contexto
panel["log1p_casos"] = np.log1p(panel["casos"])

# ------------------------------------------------- seleccion de predictores -
base_cols = ["celda_grilla", "ano", "semana", "fecha", "mes",
             "casos", "log1p_casos"]
target_cols = base_cols + [f"casos_lag{l}" for l in (2, 3, 4)]

# Excluir: variables base sin rezago, *_anomalia (fuga), precip_max_d/viento
# (sin rezagos disponibles), ICEN del mismo mes (rezago de publicacion).
predictors = []
for c in panel.columns:
    if c in target_cols:
        continue
    if c.endswith("_anomalia"):
        continue  # fuga: calculadas con todos los anos
    if c.startswith("icen"):
        # Va antes de la regla de clima: icen_lag2/4/6 tambien terminan en
        # "_lag2/4/6", pero sus rezagos son en meses, no en semanas.
        # Se excluyen icen, icen_trad, icen_cap, icen_mm3, icen_tendencia y
        # variantes de verano: corresponden al mes en curso.
        if c.startswith("icen_lag") and int(c.removeprefix("icen_lag")) >= MIN_REZAGO_ICEN:
            predictors.append(c)
    elif c.endswith(("_lag2", "_lag4", "_lag6", "_lag8", "_mm4_lag2")):
        predictors.append(c)

print(f"Predictores seleccionados: {len(predictors)}")
print("  (se excluyeron variables sin rezago y *_anomalia por fuga de informacion)")

# ------------------------------------------------- division temporal -------
train = panel[panel["ano"] < ANIO_CORTE].copy()
test = panel[panel["ano"] >= ANIO_CORTE].copy()
print(f"Train: {len(train):,} filas ({train['ano'].min()}-{train['ano'].max()})")
print(f"Test:  {len(test):,} filas ({test['ano'].min()}-{test['ano'].max()})")

# ------------------------------------------------- guardado -----------------
cols_final = ["celda_grilla", "ano", "semana", "fecha", "mes",
              "casos", "log1p_casos"] + \
             [f"casos_lag{l}" for l in (2, 3, 4)] + \
             ["sem_sin", "sem_cos"] + predictors

panel_out = panel[cols_final]
panel_out.to_parquet(PROC / "panel_modelado.parquet", index=False)
panel_out[panel_out["ano"] < ANIO_CORTE].to_parquet(
    PROC / "panel_train.parquet", index=False)
panel_out[panel_out["ano"] >= ANIO_CORTE].to_parquet(
    PROC / "panel_test.parquet", index=False)
panel_out[cols_final].groupby("celda_grilla").size().rename("n").reset_index() \
    .to_csv(OUT / "panel_cobertura_celdas.csv", index=False)

# chequeo de fuga: ningun predictor sin rezago en cols_final
sospechosos = [c for c in cols_final
               if not any(k in c for k in ("lag", "mm4", "sin", "cos",
                                           "casos", "celda", "ano", "semana",
                                           "fecha", "mes"))]
print("Posibles predictores sin rezago (revisar):", sospechosos)
print(f"\nGuardado panel_modelado/train/test en data/processed/")
print(f"Columnas finales: {len(cols_final)}")
