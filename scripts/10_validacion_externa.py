# -*- coding: utf-8 -*-
"""
10_validacion_externa.py - Paso 11 de la guia de limpieza: validar los casos
contra fuentes externas.

1. OpenDengue V1.3.1 (LSHTM, https://github.com/OpenDengue/master-repo):
   - Admin2 (provincia x semana, "probables y confirmados", 2000-2023): se suma
     por departamento y ano. OJO: OpenDengue lo construyo con el mismo archivo
     abierto del CDC, asi que NO es una validacion independiente; confirma que
     la limpieza y las uniones no perdieron ni duplicaron casos.
   - Admin0 (total nacional anual): si es independiente del archivo abierto.
     En 2023 hay una fila anual y una semanal parcial; se usa solo la anual.
2. Minsa, alerta epidemiologica AE 006-2024: 273,684 casos notificados en 2023
   (232,099 confirmados y 41,585 probables).

El archivo de OpenDengue (55 MB comprimido, 500 MB sin comprimir) se descarga a
data/external/opendengue/ (ignorado por git) y se lee sin descomprimir.

Salidas: outputs/validacion_externa_depto.csv y outputs/validacion_externa_nacional.csv
"""
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
OUT = ROOT / "outputs"
CARPETA_OD = ROOT / "data" / "external" / "opendengue"
VERSION_OD = "V1.3.1"
URL_OD = ("https://raw.githubusercontent.com/OpenDengue/master-repo/main/data/releases/"
          f"{VERSION_OD}/Spatial_extract_{VERSION_OD.replace('.', '_')}.zip")
MINSA_2023 = {"fuente": "Minsa AE 006-2024", "ano": 2023, "casos": 273_684}
UMBRAL_DIF = 0.10   # diferencias mayores al 10 % se marcan para revisar

# ---------------------------------------------------------- descarga ------
CARPETA_OD.mkdir(parents=True, exist_ok=True)
zip_od = CARPETA_OD / Path(URL_OD).name
if not zip_od.exists():
    print(f"Descargando OpenDengue {VERSION_OD} (55 MB)...")
    urllib.request.urlretrieve(URL_OD, zip_od)

columnas = ["adm_0_name", "adm_1_name", "adm_2_name", "Year", "dengue_total", "S_res", "T_res",
            "case_definition_standardised"]
partes = []
for parte in pd.read_csv(zip_od, usecols=columnas, chunksize=500_000, low_memory=False):
    partes.append(parte[parte["adm_0_name"].str.upper().str.contains("PERU", na=False)])
od = pd.concat(partes, ignore_index=True)
print(f"OpenDengue Peru: {len(od):,} filas; resoluciones: {od['S_res'].value_counts().to_dict()}")

panel = pd.read_parquet(PROC / "panel_depto_semana.parquet")
nuestro = panel.groupby(["departamento", "ano"])["casos"].sum().rename("panel")

# ------------------------------------------------ por departamento --------
adm2 = od[od["S_res"] == "Admin2"]
od_depto = (adm2.groupby(["adm_1_name", "Year"])["dengue_total"].sum()
            .rename("opendengue_admin2"))
od_depto.index = od_depto.index.set_names(["departamento", "ano"])
depto = pd.concat([nuestro, od_depto], axis=1).reset_index()
depto = depto[depto["ano"].between(panel["ano"].min(), int(adm2["Year"].max()))].fillna(0)
depto["dif"] = depto["panel"] - depto["opendengue_admin2"]
depto["dif_rel"] = depto["dif"] / depto["opendengue_admin2"].where(depto["opendengue_admin2"] > 0)
depto.to_csv(OUT / "validacion_externa_depto.csv", index=False, float_format="%.4f")
iguales = (depto["dif"] == 0).mean()
cerca = (depto["dif_rel"].abs() <= 0.02).mean()
print(f"\nPor departamento ({len(depto)} depto-ano, {depto['ano'].min()}-{depto['ano'].max()}): "
      f"{iguales:.0%} identicos, {cerca:.0%} a menos del 2 %; diferencia maxima {depto['dif'].abs().max():.0f} casos")

# ------------------------------------------------------ nacional ----------
adm0 = od[(od["S_res"] == "Admin0") & (od["T_res"] == "Year")]
nacional = pd.concat([
    panel.groupby("ano")["casos"].sum().rename("panel"),
    adm0.groupby("Year")["dengue_total"].sum().rename("opendengue_admin0"),
], axis=1).loc[panel["ano"].min():panel["ano"].max()]
nacional["minsa"] = pd.Series({MINSA_2023["ano"]: MINSA_2023["casos"]})
nacional["dif_rel_opendengue"] = nacional["panel"] / nacional["opendengue_admin0"] - 1
nacional["dif_rel_minsa"] = nacional["panel"] / nacional["minsa"] - 1
nacional["revisar"] = nacional["dif_rel_opendengue"].abs() > UMBRAL_DIF
nacional.index.name = "ano"
nacional.to_csv(OUT / "validacion_externa_nacional.csv", float_format="%.4f")
print("\nNacional (panel vs. totales oficiales):")
print(nacional.round(3).to_string())
print(f"\nAnos con diferencia > {UMBRAL_DIF:.0%} frente a OpenDengue Admin0: "
      f"{nacional.index[nacional['revisar']].tolist()}")
print("Guardado outputs/validacion_externa_depto.csv y outputs/validacion_externa_nacional.csv")
