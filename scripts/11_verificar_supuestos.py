# -*- coding: utf-8 -*-
"""
11_verificar_supuestos.py - Comprueba con los datos los supuestos que la guia de
limpieza marco como "a verificar".

  1. mm4_lag2: ¿es la media movil de 4 semanas rezagada 2? Se prueban varias
     definiciones y se reporta la diferencia maxima con cada una.
  2. *_anomalia: ¿con que periodo de referencia se calcularon? Se compara con
     "valor - media de la misma celda y semana" usando distintos rangos de anos.
     Si coincide con TODOS los anos (incluidos los de prueba) hay fuga.
  3. Semanas del clima: ¿siguen el calendario epidemiologico (MMWR)? Los anos
     con semana 53 deben ser los mismos que en el calendario oficial.
  4. ICEN: relacion entre icen, icen_mm3 e icen_lag1 dentro de la tabla.

Salida: outputs/supuestos_verificacion.csv
"""
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
from epiweeks import Week

ROOT = Path(__file__).resolve().parents[1]
INTERIM = ROOT / "data" / "interim"
OUT = ROOT / "outputs"

clima = (pd.read_parquet(INTERIM / "clima_typed.parquet")
         .sort_values(["celda_grilla", "ano", "semana"]).reset_index(drop=True))
g = clima.groupby("celda_grilla", sort=False)
resultados = []


def registrar(supuesto, hipotesis, dif_max, detalle=""):
    resultados.append({"supuesto": supuesto, "hipotesis": hipotesis,
                       "diferencia_maxima": round(float(dif_max), 6),
                       "coincide": bool(dif_max < 1e-3), "detalle": detalle})
    print(f"  [{'SI' if dif_max < 1e-3 else 'no'}] {supuesto}: {hipotesis} (dif. max {dif_max:.4g}) {detalle}")


# ------------------------------------------------------------ 1. mm4_lag2 --
print("1. Definicion de *_mm4_lag2")
bases = ["temp_media", "precip_total", "hum_rel"]
for v in bases:
    real = clima[f"{v}_mm4_lag2"]
    opciones = {
        "media de t-2..t-5 (rolling 4 y shift 2)": g[v].transform(lambda s: s.rolling(4).mean().shift(2)),
        "media de t-1..t-4 (rolling 4 y shift 1)": g[v].transform(lambda s: s.rolling(4).mean().shift(1)),
        "media de t-2..t-5 aceptando ventanas incompletas": g[v].transform(lambda s: s.rolling(4, min_periods=1).mean().shift(2)),
        "media de t-3..t (rolling 4 sin shift)": g[v].transform(lambda s: s.rolling(4).mean()),
    }
    for nombre, calc in opciones.items():
        dif = (real - calc).abs()
        registrar(f"{v}_mm4_lag2", nombre, dif.max(skipna=True),
                  f"NA reales {int(real.isna().sum())}, NA calculados {int(calc.isna().sum())}")

# --------------------------------------------------------- 2. anomalias ----
print("\n2. Periodo de referencia de *_anomalia")
for v in ["temp_media", "precip_total", "hum_rel"]:
    real = clima[f"{v}_anomalia"]
    for inicio, fin in [(2010, 2024), (2010, 2023), (2010, 2020), (2010, 2019), (2011, 2020), (2014, 2023)]:
        ref = (clima[clima["ano"].between(inicio, fin)].groupby(["celda_grilla", "semana"])[v].mean()
               .rename("ref").reset_index())
        calc = clima[["celda_grilla", "semana", v]].merge(ref, on=["celda_grilla", "semana"], how="left")
        dif = (real.to_numpy() - (calc[v] - calc["ref"]).to_numpy())
        registrar(f"{v}_anomalia", f"valor - media de la celda y semana en {inicio}-{fin}", np.nanmax(np.abs(dif)))
    # Tambien por mes en vez de semana
    mes = (pd.to_datetime(clima["ano"].astype(str) + "-01-01") + pd.to_timedelta(7 * (clima["semana"] - 1), unit="D")).dt.month
    ref = clima.assign(mes=mes).groupby(["celda_grilla", "mes"])[v].transform("mean")
    registrar(f"{v}_anomalia", "valor - media de la celda y mes en 2010-2024", (real - (clima[v] - ref)).abs().max())

# ------------------------------------------------ 3. semanas epidemiologicas
print("\n3. Calendario de semanas del clima")
con_53 = sorted(clima.loc[clima["semana"] == 53, "ano"].unique().tolist())
mmwr_53 = [a for a in range(int(clima["ano"].min()), int(clima["ano"].max()) + 1)
           if Week.fromdate(dt.date(a, 12, 31), system="cdc").week == 53]
iso_53 = [a for a in range(int(clima["ano"].min()), int(clima["ano"].max()) + 1)
          if dt.date(a, 12, 28).isocalendar()[1] == 53]
registrar("semanas del clima", f"anos con semana 53 = calendario epidemiologico MMWR {mmwr_53}",
          0 if con_53 == mmwr_53 else 1, f"clima: {con_53}; ISO: {iso_53}")

# ------------------------------------------------------------- 4. ICEN ------
print("\n4. Columnas del ICEN")
icen = pd.read_parquet(INTERIM / "icen_typed.parquet").sort_values(["ano", "mes"]).reset_index(drop=True)
registrar("icen_lag1", "icen del mes anterior", (icen["icen_lag1"] - icen["icen"].shift(1)).abs().max())
registrar("icen_lag3", "icen de tres meses antes", (icen["icen_lag3"] - icen["icen"].shift(3)).abs().max())
registrar("icen_mm3", "media de icen en los meses m, m-1 y m-2", (icen["icen_mm3"] - icen["icen"].rolling(3).mean()).abs().max())
registrar("icen_mm3", "media centrada de icen en m-1, m y m+1",
          (icen["icen_mm3"] - icen["icen"].rolling(3, center=True).mean()).abs().max())
registrar("icen_tendencia", "icen - icen del mes anterior", (icen["icen_tendencia"] - icen["icen"].diff()).abs().max())
registrar("icen_tendencia", "icen - icen de tres meses antes", (icen["icen_tendencia"] - icen["icen"].diff(3)).abs().max())
registrar("icen_cap", "icen igual a icen_cap", (icen["icen_cap"] - icen["icen"]).abs().max(),
          f"filas distintas: {int((icen['icen_cap'] != icen['icen']).sum())}, max icen {icen['icen'].max()}, max icen_cap {icen['icen_cap'].max()}")
registrar("icen_trad", "icen igual a icen_trad", (icen["icen_trad"] - icen["icen"]).abs().max(),
          f"correlacion {icen['icen'].corr(icen['icen_trad']):.3f}")

pd.DataFrame(resultados).to_csv(OUT / "supuestos_verificacion.csv", index=False)
print("\nGuardado outputs/supuestos_verificacion.csv")
