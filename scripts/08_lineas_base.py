# -*- coding: utf-8 -*-
"""
08_lineas_base.py - Fase 2 de la guia metodologica: lineas base.

Para cada departamento y horizonte h = 1..4 semanas se pronostican los casos de
la semana T usando solo informacion disponible en T - h:

  - Persistencia (referencia ingenua): casos de T - h.
  - Modelo 1, promedio estacional: media de los casos de la misma semana
    epidemiologica en los anos anteriores (la semana 53 usa la historia de la
    52). No depende de h: los anos anteriores siempre se conocen.
  - Modelo 2, regresion con casos rezagados: regresion lineal de log1p(casos)
    sobre log1p de los casos en T-h, T-h-1, T-h-2 y T-h-3. Un modelo por
    departamento y horizonte.

Division temporal (scripts/comun.py), igual a la que usara XGBoost en la fase 3:
  - "Validacion": se ajusta con entrenamiento (2010-2021) y se evalua en 2022-2023.
  - "Prueba": se ajusta con entrenamiento + validacion (2010-2023) y se evalua en 2024.
Metricas en la escala de casos: RMSE, MAE y R2.

Si el XGBoost con clima de la fase 3 no supera estas lineas base, el clima no
esta aportando.

Salidas:
  outputs/fase2_lineas_base_metricas.csv   (todos los departamentos)
  outputs/fase2_lineas_base_2024.png       (regiones principales, h = 2)
  data/processed/predicciones_lineas_base.parquet
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from comun import ENTRENAMIENTO, PRUEBA, VALIDACION

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
OUT = ROOT / "outputs"

HORIZONTES = [1, 2, 3, 4]
N_REZAGOS = 4               # rezagos de casos del modelo 2: h .. h + 3
# Regiones principales (guia, fase 0): costa norte, selva baja y selva alta.
# Alternativas para la tercera: UCAYALI o MADRE DE DIOS.
REGIONES = ["PIURA", "LORETO", "SAN MARTIN"]
# Departamentos con datos suficientes para evaluar
MIN_CASOS_TRAIN = 1000
# (nombre, ultimo ano de ajuste, rango evaluado)
EVALUACIONES = [("Validacion", ENTRENAMIENTO[1], VALIDACION), ("Prueba", VALIDACION[1], PRUEBA)]

PERSISTENCIA, ESTACIONAL, REGRESION = "Persistencia", "Promedio estacional", "Regresion con rezagos"

panel = pd.read_parquet(PROC / "panel_depto_semana.parquet")
panel = panel.sort_values(["departamento", "ano", "semana"]).reset_index(drop=True)
g = panel.groupby("departamento", sort=False)


# ------------------------------------------------- promedio estacional -----
def promedio_estacional(df: pd.DataFrame) -> np.ndarray:
    """Media de la misma semana en los anos anteriores, por departamento."""
    sem = df["semana"].clip(upper=52)  # la semana 53 usa la historia de la 52
    tabla = (df.assign(sem=sem).pivot_table(index=["departamento", "sem"], columns="ano",
                                           values="casos", aggfunc="sum"))
    # Media expansiva de los anos ESTRICTAMENTE anteriores
    previa = tabla.T.expanding().mean().shift(1).T
    largo = previa.stack().rename("estacional").reset_index()
    return df.assign(sem=sem).merge(largo, on=["departamento", "sem", "ano"], how="left")["estacional"].to_numpy()


panel["pred_estacional"] = promedio_estacional(panel)
panel["log_casos"] = np.log1p(panel["casos"])

filas_metricas, predicciones = [], []
casos_train = panel[panel["ano"] <= ENTRENAMIENTO[1]].groupby("departamento")["casos"].sum()
deptos = casos_train[casos_train >= MIN_CASOS_TRAIN].index.tolist()
print(f"Departamentos evaluados ({len(deptos)}): {', '.join(deptos)}")

for h in HORIZONTES:
    rezagos = [f"log_casos_l{h + k}" for k in range(N_REZAGOS)]
    for k in range(N_REZAGOS):
        panel[f"log_casos_l{h + k}"] = g["log_casos"].shift(h + k)
    panel[f"pred_persistencia_h{h}"] = g["casos"].shift(h)

    for depto in deptos:
        d = panel[panel["departamento"] == depto]
        for conjunto, ultimo_ajuste, (inicio, fin) in EVALUACIONES:
            ajuste = d[d["ano"] <= ultimo_ajuste].dropna(subset=rezagos)
            evaluado = d[d["ano"].between(inicio, fin)]
            assert evaluado[rezagos + ["pred_estacional"]].notna().all().all()

            modelo = LinearRegression().fit(ajuste[rezagos], ajuste["log_casos"])
            pred_reg = np.clip(np.expm1(modelo.predict(evaluado[rezagos])), 0, None)
            preds = {
                PERSISTENCIA: evaluado[f"pred_persistencia_h{h}"].to_numpy(),
                ESTACIONAL: evaluado["pred_estacional"].to_numpy(),
                REGRESION: pred_reg,
            }
            y = evaluado["casos"].to_numpy()
            for nombre, p in preds.items():
                filas_metricas.append({
                    "conjunto": conjunto, "departamento": depto, "horizonte": h, "modelo": nombre,
                    "rmse": float(np.sqrt(mean_squared_error(y, p))),
                    "mae": float(mean_absolute_error(y, p)),
                    "r2": float(r2_score(y, p)),
                    "casos_evaluados": int(y.sum()), "n_semanas_evaluadas": len(y),
                    "n_semanas_ajuste": len(ajuste),
                })
            predicciones.append(pd.DataFrame({
                "conjunto": conjunto, "departamento": depto, "ano": evaluado["ano"].to_numpy(),
                "semana": evaluado["semana"].to_numpy(), "fecha": evaluado["fecha"].to_numpy(),
                "horizonte": h, "casos": y, "pred_persistencia": preds[PERSISTENCIA],
                "pred_estacional": preds[ESTACIONAL], "pred_regresion": preds[REGRESION],
            }))

metricas = pd.DataFrame(filas_metricas)
metricas.to_csv(OUT / "fase2_lineas_base_metricas.csv", index=False, float_format="%.4f")
pred = pd.concat(predicciones, ignore_index=True)
pred.to_parquet(PROC / "predicciones_lineas_base.parquet", index=False)

# ------------------------------------------------------------- resumen -----
pd.set_option("display.width", 160)
foco = metricas[metricas["departamento"].isin(REGIONES)]
for conjunto, _, (inicio, fin) in EVALUACIONES:
    sub = foco[foco["conjunto"] == conjunto]
    print(f"\n{conjunto} {inicio}-{fin}, regiones principales, R2:")
    print(sub.pivot_table(index=["departamento", "modelo"], columns="horizonte", values="r2").round(3).to_string())
prueba = metricas[metricas["conjunto"] == "Prueba"]
mejor = (prueba.loc[prueba.groupby(["departamento", "horizonte"])["mae"].idxmin()]
         .groupby(["horizonte", "modelo"]).size().unstack(fill_value=0))
print("\nPrueba: modelo con menor MAE, numero de departamentos por horizonte:")
print(mejor.to_string())

# -------------------------------------------------------------- grafico ----
AZUL, NARANJA, AQUA, TINTA, TINTA_2, REJILLA = "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#52514e", "#e1e0d9"
H_GRAFICO = 2
fig, ejes = plt.subplots(len(REGIONES), 1, figsize=(10, 2.8 * len(REGIONES)), sharex=True)
for ax, depto in zip(np.atleast_1d(ejes), REGIONES):
    p = pred[(pred["departamento"] == depto) & (pred["horizonte"] == H_GRAFICO) & (pred["conjunto"] == "Prueba")]
    ax.plot(p["fecha"], p["casos"], color=TINTA, lw=1.8, label="Casos observados")
    ax.plot(p["fecha"], p["pred_persistencia"], color=AQUA, lw=1.4, label="Persistencia")
    ax.plot(p["fecha"], p["pred_estacional"], color=NARANJA, lw=1.4, label="Promedio estacional")
    ax.plot(p["fecha"], p["pred_regresion"], color=AZUL, lw=1.4, label="Regresion con rezagos")
    ax.set_title(f"{depto.title()}", loc="left", fontsize=11, fontweight="bold", color=TINTA)
    ax.set_ylabel("Casos por semana", color=TINTA_2)
    ax.grid(color=REJILLA, lw=0.6)
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
np.atleast_1d(ejes)[0].legend(loc="upper right", fontsize=8.5, frameon=False, ncol=2)
fig.suptitle(f"Lineas base en la prueba 2024, pronostico a {H_GRAFICO} semanas", x=0.01, ha="left",
             fontsize=12, fontweight="bold", color=TINTA)
fig.tight_layout()
fig.savefig(OUT / "fase2_lineas_base_2024.png", dpi=120)
print("\nGuardado outputs/fase2_lineas_base_metricas.csv y outputs/fase2_lineas_base_2024.png")
