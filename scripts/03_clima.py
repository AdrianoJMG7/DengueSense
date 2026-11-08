# -*- coding: utf-8 -*-
"""
03_clima.py — Paso 5 de la guia: control de calidad del clima semanal.

La tabla ya viene derivada (lags, medias moviles, anomalías). Aqui solo se
verifica coherencia; no se recalculan los rezagos. Resultados a outputs/.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INTERIM = ROOT / "data" / "interim"
OUT = ROOT / "outputs"

clima = pd.read_parquet(INTERIM / "clima_typed.parquet")
reporte = []


def check(nombre, ok, detalle=""):
    estado = "OK" if ok else "FALLA"
    reporte.append({"control": nombre, "estado": estado, "detalle": detalle})
    print(f"[{estado}] {nombre}: {detalle}")


print(f"Clima: {len(clima):,} filas, {clima['celda_grilla'].nunique()} celdas, "
      f"anos {clima['ano'].min()}-{clima['ano'].max()}")

# 1) Duplicados en la llave
n_dup = clima.duplicated(["celda_grilla", "ano", "semana"]).sum()
check("sin duplicados celda-ano-semana", n_dup == 0, f"{n_dup} duplicados")

# 2) Rangos fisicos (solo variables base)
base = ["temp_media", "temp_max", "temp_min", "rango_term", "precip_total",
        "precip_max_d", "dias_lluvia", "hum_rel", "viento"]
viol = {}
viol["temp_min<=temp_max"] = int((clima["temp_min"] > clima["temp_max"]).sum())
viol["precip_total>=0"] = int((clima["precip_total"] < 0).sum())
viol["dias_lluvia 0-7"] = int(((clima["dias_lluvia"] < 0) | (clima["dias_lluvia"] > 7)).sum())
viol["hum_rel 0-100"] = int(((clima["hum_rel"] < 0) | (clima["hum_rel"] > 100)).sum())
viol["temp_media entre min y max"] = int(
    ((clima["temp_media"] < clima["temp_min"]) | (clima["temp_media"] > clima["temp_max"])).sum()
)
for k, v in viol.items():
    check(f"rango fisico: {k}", v == 0, f"{v} violaciones")

# 3) Valores NA por variable (esperado: primeras 8 semanas por celda en rezagos)
na_counts = clima[base].isna().sum()
print("\nNA en variables base:\n", na_counts[na_counts > 0].to_string())
na_counts.to_csv(OUT / "clima_na_base.csv", header=["n_na"])

# NA en rezagos: deberian estar solo en las primeras ~8 semanas de cada celda
clima_s = clima.sort_values(["celda_grilla", "ano", "semana"]).reset_index(drop=True)
# semanas desde el inicio de la serie de cada celda
clima_s["sem_idx"] = clima_s.groupby("celda_grilla").cumcount()
lag_cols = [c for c in clima.columns if c.endswith(("_lag2", "_lag4", "_lag6", "_lag8",
           "_mm4_lag2", "_anomalia"))]
na_fuera_inicial = {}
for c in lag_cols[:20]:  # muestra representativa por familia
    na_late = clima_s.loc[clima_s[c].isna() & (clima_s["sem_idx"] >= 12)]
    na_fuera_inicial[c] = len(na_late)
check("NA en rezagos solo en primeras semanas",
      all(v == 0 for v in na_fuera_inicial.values()),
      f"cols con NA tardios: {sum(1 for v in na_fuera_inicial.values() if v)}")

# 4) Verificar rezagos: lag2 de temp_media debe coincidir con temp_media_lag2
#    (secuencia ano-semana no es exactamente lag temporal si hay semanas
#    faltantes; usamos desplazamiento dentro de la serie ordenada)
clima_s["tmp_lag2_calc"] = clima_s.groupby("celda_grilla")["temp_media"].shift(2)
diff = (clima_s["tmp_lag2_calc"] - clima_s["temp_media_lag2"]).abs()
check("temp_media_lag2 == lag(temp_media,2)",
      bool((diff.dropna() < 1e-6).all()),
      f"max diff = {diff.max():.3g}")
clima_s["tmp_lag4_calc"] = clima_s.groupby("celda_grilla")["temp_media"].shift(4)
diff4 = (clima_s["tmp_lag4_calc"] - clima_s["temp_media_lag4"]).abs()
check("temp_media_lag4 == lag(temp_media,4)",
      bool((diff4.dropna() < 1e-6).all()),
      f"max diff = {diff4.max():.3g}")

# 5) Cobertura temporal por celda
sem_celda = clima.groupby("celda_grilla")["semana"].count()
semanas_totales = len(clima) / clima["celda_grilla"].nunique()
check("series completas por celda",
      bool((sem_celda == sem_celda.max()).all()),
      f"min={sem_celda.min()}, max={sem_celda.max()} semanas por celda")

# 6) Valores extremos plausibles
extremos = pd.DataFrame({
    "variable": base,
    "min": [clima[c].min() for c in base],
    "max": [clima[c].max() for c in base],
})
extremos.to_csv(OUT / "clima_rangos.csv", index=False)
print("\nRangos observados:\n", extremos.to_string(index=False))

pd.DataFrame(reporte).to_csv(OUT / "clima_qc.csv", index=False)
print("\nReporte guardado en outputs/clima_qc.csv")
